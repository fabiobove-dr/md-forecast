import asyncio
import hashlib
import json
import os
import shutil
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml
from pydantic import ValidationError

from md_forecast.cli import main
from md_forecast.core.config import DownloadSettings
from md_forecast.core.exceptions import AcquisitionError
from md_forecast.data import acquisition
from md_forecast.data.acquisition import (
    AcquisitionSource,
    SourceArtifact,
    acquire,
    load_source,
)

PAYLOAD = b"abcdefghij"


class BytesStream(httpx.AsyncByteStream):
    def __init__(self, data: bytes, fail: bool = False) -> None:
        self.data = data
        self.fail = fail

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.data
        if self.fail:
            raise httpx.ReadError("interrupted local stream")


def source_for(
    name: str = "test.txt", mode: str = "metadata", data: bytes = PAYLOAD
) -> AcquisitionSource:
    return AcquisitionSource(
        dataset_id="misato",
        dataset_version="1.0.0",
        source_record="test-record",
        artifacts={
            name: SourceArtifact(
                mode=mode,
                url=f"https://example.test/{name}",
                size_bytes=len(data),
                checksum="md5:" + hashlib.md5(data).hexdigest(),
            )
        },
    )


def settings_for(path: Path, **kwargs: object) -> DownloadSettings:
    return DownloadSettings.model_validate(
        {
            "destination": path,
            "disk_reserve_bytes": 0,
            "chunk_bytes": 2,
            "retries": 0,
            "retry_delay": 0,
            **kwargs,
        }
    )


def response(
    data: bytes = PAYLOAD,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return httpx.Response(status, headers=headers, stream=BytesStream(data))


def test_download_receipt_and_idempotence(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["Accept-Encoding"] == "identity"
        assert request.extensions["timeout"] == {
            "connect": 10.0,
            "read": 60.0,
            "write": 30.0,
            "pool": 10.0,
        }
        return response()

    source = source_for()
    settings = settings_for(tmp_path)
    transport = httpx.MockTransport(handler)
    result = asyncio.run(acquire(source, settings, transport=transport))
    final = tmp_path / "test.txt"
    assert result == [final]
    assert final.read_bytes() == PAYLOAD
    assert not final.with_name("test.txt.part").exists()
    receipt = json.loads((tmp_path / "test.txt.receipt.json").read_text())
    assert receipt["source_record"] == "test-record"
    assert receipt["filename"] == "test.txt"
    assert receipt["checksum"] == source.artifacts["test.txt"].checksum
    assert not any(str(tmp_path) in str(value) for value in receipt.values())
    asyncio.run(acquire(source, settings, transport=transport))
    assert calls == 1


@pytest.mark.parametrize("supports_range", [True, False])
def test_resume_or_restart(tmp_path: Path, supports_range: bool) -> None:
    (tmp_path / "test.txt.part").write_bytes(PAYLOAD[:4])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Range"] == "bytes=4-"
        if supports_range:
            return response(PAYLOAD[4:], 206, {"Content-Range": "bytes 4-9/10"})
        return response()

    asyncio.run(
        acquire(
            source_for(), settings_for(tmp_path), transport=httpx.MockTransport(handler)
        )
    )
    assert (tmp_path / "test.txt").read_bytes() == PAYLOAD


def test_interrupted_stream_retries_from_saved_bytes(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, stream=BytesStream(PAYLOAD[:4], fail=True))
        assert request.headers["Range"] == "bytes=4-"
        return response(PAYLOAD[4:], 206, {"Content-Range": "bytes 4-9/10"})

    asyncio.run(
        acquire(
            source_for(),
            settings_for(tmp_path, retries=1),
            transport=httpx.MockTransport(handler),
        )
    )
    assert calls == 2
    assert (tmp_path / "test.txt").read_bytes() == PAYLOAD


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"1234567890", "Checksum mismatch"),
        (b"abc", "Incomplete artifact"),
        (PAYLOAD + b"extra", "too many bytes"),
    ],
)
def test_bad_payload_never_promoted(tmp_path: Path, data: bytes, message: str) -> None:
    with pytest.raises(AcquisitionError, match=message):
        asyncio.run(
            acquire(
                source_for(),
                settings_for(tmp_path),
                transport=httpx.MockTransport(lambda _: response(data)),
            )
        )
    assert not (tmp_path / "test.txt").exists()
    assert not (tmp_path / "test.txt.receipt.json").exists()


@pytest.mark.parametrize(
    ("status", "headers", "message"),
    [
        (206, {"Content-Range": "bytes 0-9/10"}, "Content-Range"),
        (206, {}, "Content-Range"),
        (204, {}, "HTTP status"),
        (200, {"Content-Encoding": "gzip"}, "encoded bytes"),
        (503, {}, "Download failed"),
    ],
)
def test_response_contract_preserves_partial(
    tmp_path: Path, status: int, headers: dict[str, str], message: str
) -> None:
    partial = tmp_path / "test.txt.part"
    partial.write_bytes(PAYLOAD[:4])
    with pytest.raises(AcquisitionError, match=message):
        asyncio.run(
            acquire(
                source_for(),
                settings_for(tmp_path),
                transport=httpx.MockTransport(
                    lambda _: response(status=status, headers=headers)
                ),
            )
        )
    assert partial.read_bytes() == PAYLOAD[:4]
    assert not (tmp_path / "test.txt").exists()


def test_completed_partial_and_invalid_existing_final(tmp_path: Path) -> None:
    partial = tmp_path / "test.txt.part"
    partial.write_bytes(PAYLOAD)
    transport = httpx.MockTransport(
        lambda _: pytest.fail("No request for complete bytes")
    )
    asyncio.run(acquire(source_for(), settings_for(tmp_path), transport=transport))
    final = tmp_path / "test.txt"
    final.write_bytes(b"wrong-size")
    with pytest.raises(AcquisitionError, match="Checksum mismatch"):
        asyncio.run(acquire(source_for(), settings_for(tmp_path), transport=transport))
    assert final.read_bytes() == b"wrong-size"


def test_preflight_and_filesystem_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, settings = source_for(), settings_for(tmp_path)
    transport = httpx.MockTransport(
        lambda _: pytest.fail("No HTTP after preflight failure")
    )
    with monkeypatch.context() as patch:
        patch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
        with pytest.raises(AcquisitionError, match="Not enough disk"):
            asyncio.run(acquire(source, settings, transport=transport))
    (tmp_path / "test.txt.part").write_bytes(PAYLOAD + b"extra")
    with pytest.raises(AcquisitionError, match="Oversized partial"):
        asyncio.run(acquire(source, settings, transport=transport))
    (tmp_path / "test.txt.part").unlink()
    (tmp_path / "test.txt.part").symlink_to(tmp_path / "other")
    with pytest.raises(AcquisitionError, match="symlink"):
        asyncio.run(acquire(source, settings, transport=transport))
    (tmp_path / "test.txt.part").unlink()
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("preserve")
    with pytest.raises(AcquisitionError, match="Storage operation failed"):
        asyncio.run(acquire(source, settings_for(blocked), transport=transport))


def test_selection_and_bounded_concurrency(tmp_path: Path) -> None:
    source = source_for()
    source.artifacts.update(source_for("other.txt").artifacts)
    source.artifacts.update(source_for("MD.hdf5", "md").artifacts)
    source.artifacts.update(source_for("QM.hdf5", "qm").artifacts)
    active, peak = 0, 0
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        seen.append(request.url.path)
        await asyncio.sleep(0.01)
        active -= 1
        return response()

    asyncio.run(
        acquire(
            source,
            settings_for(tmp_path, concurrency=1),
            transport=httpx.MockTransport(handler),
        )
    )
    assert set(seen) == {"/test.txt", "/other.txt"}
    assert peak == 1
    asyncio.run(
        acquire(
            source, settings_for(tmp_path), "md", transport=httpx.MockTransport(handler)
        )
    )
    assert "/MD.hdf5" in seen and "/QM.hdf5" not in seen
    with pytest.raises(AcquisitionError, match="No acquisition artifacts"):
        asyncio.run(acquire(source_for(mode="qm"), settings_for(tmp_path)))


def test_cancellation_closes_stream_and_resumes(tmp_path: Path) -> None:
    async def scenario() -> None:
        waiting = asyncio.Event()
        closed = asyncio.Event()

        class PausedStream(httpx.AsyncByteStream):
            async def __aiter__(self) -> AsyncIterator[bytes]:
                yield PAYLOAD[:4]
                waiting.set()
                await asyncio.Event().wait()

            async def aclose(self) -> None:
                closed.set()

        transport = httpx.MockTransport(
            lambda _: httpx.Response(200, stream=PausedStream())
        )
        task = asyncio.create_task(
            acquire(source_for(), settings_for(tmp_path), transport=transport)
        )
        await waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()
        assert (tmp_path / "test.txt.part").read_bytes() == PAYLOAD[:4]
        assert not (tmp_path / "test.txt").exists()
        transport = httpx.MockTransport(
            lambda _: response(PAYLOAD[4:], 206, {"Content-Range": "bytes 4-9/10"})
        )
        await acquire(source_for(), settings_for(tmp_path), transport=transport)

    asyncio.run(scenario())
    assert (tmp_path / "test.txt").read_bytes() == PAYLOAD


def test_file_work_off_loop_and_cancelled_open_cleanup() -> None:
    async def scenario() -> None:
        loop_thread = threading.get_ident()
        assert await acquisition._file_job(threading.get_ident) != loop_thread
        started, release = threading.Event(), threading.Event()
        cleaned: list[int] = []

        def open_resource() -> int:
            started.set()
            release.wait(timeout=5)
            return 42

        task = asyncio.create_task(
            acquisition._file_job(open_resource, cleanup=cleaned.append)
        )
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleaned == [42]

    asyncio.run(scenario())


def test_destination_lock_rejects_overlapping_acquisition(tmp_path: Path) -> None:
    lock = acquisition._lock_destination(tmp_path)
    try:
        with pytest.raises(AcquisitionError, match="Storage operation failed"):
            asyncio.run(acquire(source_for(), settings_for(tmp_path)))
    finally:
        os.close(lock)


@pytest.mark.parametrize("checksum", ["crc32:12345678", "md5:no", "sha256:AAA"])
def test_checksum_validation(checksum: str) -> None:
    with pytest.raises(ValidationError, match="checksum"):
        SourceArtifact.model_validate(
            {
                "mode": "metadata",
                "url": "https://example.test/a",
                "size_bytes": 1,
                "checksum": checksum,
            }
        )


def test_source_boundary_validation() -> None:
    artifact = source_for().artifacts["test.txt"].model_dump(mode="json")
    artifact["url"] = "https://user:secret@example.test/a"
    with pytest.raises(ValidationError, match="credentials"):
        SourceArtifact.model_validate(artifact)
    source = source_for().model_dump(mode="json")
    with pytest.raises(ValidationError, match="schema version"):
        AcquisitionSource.model_validate({**source, "schema_version": 9})
    with pytest.raises(ValidationError, match="at least one artifact"):
        AcquisitionSource.model_validate({**source, "artifacts": {}})
    with pytest.raises(ValidationError):
        DownloadSettings(concurrency=0, read_timeout=0)
    with pytest.raises(ValidationError):
        DownloadSettings(read_timeout=float("inf"))


def test_default_destination_and_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = source_for()
    artifact = source.artifacts["test.txt"].model_dump(mode="json")
    artifact["checksum"] = "sha256:" + hashlib.sha256(PAYLOAD).hexdigest()
    source.artifacts["test.txt"] = SourceArtifact.model_validate(artifact)
    monkeypatch.delenv("MD_FORECAST_DESTINATION", raising=False)
    settings = DownloadSettings(data_dir=tmp_path, disk_reserve_bytes=0)
    paths = asyncio.run(
        acquire(source, settings, transport=httpx.MockTransport(lambda _: response()))
    )
    assert paths == [tmp_path / "external" / "misato" / "test.txt"]


def test_retry_limit_and_hashing_off_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("local offline source")

    with pytest.raises(AcquisitionError, match="Download failed"):
        asyncio.run(
            acquire(
                source_for(),
                settings_for(tmp_path, retries=1),
                transport=httpx.MockTransport(handler),
            )
        )
    assert calls == 2

    original = acquisition._validate_file

    async def scenario() -> None:
        loop_thread = threading.get_ident()

        def checked(path: Path, artifact: SourceArtifact) -> None:
            assert threading.get_ident() != loop_thread
            original(path, artifact)

        monkeypatch.setattr(acquisition, "_validate_file", checked)
        await acquire(
            source_for(),
            settings_for(tmp_path),
            transport=httpx.MockTransport(lambda _: response()),
        )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "name", ["../outside", "/absolute", "a/b", "a.part", "a.receipt.json"]
)
def test_filename_validation(name: str) -> None:
    with pytest.raises(ValidationError, match="filename|suffix"):
        source_for(name)


def test_source_config_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "source.yaml"
    config.write_text(yaml.safe_dump(source_for().model_dump(mode="json")))
    assert load_source(config).source_record == "test-record"
    for invalid in ["[broken", "schema_version: 9", "null"]:
        config.write_text(invalid)
        with pytest.raises(AcquisitionError, match="Cannot load source"):
            load_source(config)
    with pytest.raises(AcquisitionError, match="Cannot load source"):
        load_source(tmp_path / "missing")
    monkeypatch.setattr(
        "sys.argv", ["md-forecast", "download", "--source", str(config)]
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "Cannot load source" in capsys.readouterr().err

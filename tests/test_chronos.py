"""Chronos boundary regressions without optional dependencies, downloads or GPU."""

import importlib
import tomllib
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError
from test_baselines import pairs

from md_forecast.core.constants import ModelId
from md_forecast.core.exceptions import DataContractError, ForecastError
from md_forecast.data.artifacts import metadata_hash
from md_forecast.data.forecast import ForecastBatch
from md_forecast.models.base import ForecastModel, ModelConfig
from md_forecast.models.baselines import StatisticalBaseline
from md_forecast.models.chronos import Chronos2Adapter, ChronosConfig, QuantileForecast


def settings(**changes: object) -> ChronosConfig:
    data = tomllib.loads(Path("configs/models/chronos2.toml").read_text())
    return ChronosConfig.model_validate(data | changes)


def install_backend(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    torch, pipeline, backend, hub = (MagicMock() for _ in range(4))
    torch.float32 = "float32"
    torch.cuda.is_available.return_value = True
    torch.cuda.device_count.return_value = 1
    torch.cuda.max_memory_allocated.return_value = 100
    torch.cuda.max_memory_reserved.return_value = 200
    torch.random.fork_rng.side_effect = lambda **_: nullcontext()
    torch.inference_mode.side_effect = nullcontext
    torch.cuda.device.side_effect = lambda _: nullcontext()
    pipeline.model.device = "cuda:0"
    pipeline.model.dtype = "float32"
    pipeline.quantiles = [0.1, 0.5, 0.9]
    pipeline.model_context_length = 8192
    pipeline.model_prediction_length = 1024

    def predict(inputs: Any, **kwargs: Any) -> tuple[list[MagicMock], list[object]]:
        items = []
        for window in inputs:
            values = np.broadcast_to(
                np.array(kwargs["quantile_levels"]),
                (window.shape[0], kwargs["prediction_length"], 3),
            ).copy()
            tensor = MagicMock()
            tensor.detach.return_value.cpu.return_value.numpy.return_value = values
            items.append(tensor)
        return items, []

    pipeline.predict_quantiles.side_effect = predict
    backend.Chronos2Pipeline.from_pretrained.return_value = pipeline
    hub.snapshot_download.return_value = "/pinned/snapshot"
    original = importlib.import_module
    modules = {"torch": torch, "chronos": backend, "huggingface_hub": hub}
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, package=None: (
            modules[name] if name in modules else original(name, package)
        ),
    )
    return torch, pipeline


def test_quantiles_identity_axes_and_common_protocol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch, pipeline = install_backend(monkeypatch)
    adapter = Chronos2Adapter(settings(), cache_dir=tmp_path)
    hub = importlib.import_module("huggingface_hub")
    hub.snapshot_download.assert_called_once_with(
        repo_id=settings().model_name,
        revision=settings().revision,
        cache_dir=str(tmp_path),
        allow_patterns=["config.json", "model.safetensors"],
    )
    backend = importlib.import_module("chronos")
    backend.Chronos2Pipeline.from_pretrained.assert_called_once_with(
        "/pinned/snapshot",
        local_files_only=True,
        trust_remote_code=False,
        device_map="cuda:0",
        dtype="float32",
        use_safetensors=True,
    )
    model: ForecastModel = adapter
    batch, _ = pairs()[0]
    result = adapter.forecast(batch)
    assert result.indices == batch.indices and result.spec == batch.spec
    assert result.quantile_levels == (0.1, 0.5, 0.9)
    assert result.values.shape == (2, 2, 2, 3)
    np.testing.assert_array_equal(model.predict(batch), result.median)
    assert not result.values.flags.writeable
    assert result.runtime.peak_reserved_bytes == 200
    assert result.runtime.seconds >= 0
    call = pipeline.predict_quantiles.call_args
    np.testing.assert_array_equal(call.args[0], batch.context.transpose(0, 2, 1))
    assert call.kwargs["cross_learning"] is False
    assert call.kwargs["limit_prediction_length"] is True
    assert call.kwargs["context_length"] == batch.spec.context_frames
    torch.cuda.synchronize.assert_called_with("cuda:0")
    assert adapter.config.adapter_config_hash == metadata_hash(settings())
    assert (
        ModelConfig.model_validate_json(adapter.config.model_dump_json())
        == model.config
    )
    with pytest.raises(ForecastError, match="pretrained"):
        StatisticalBaseline(adapter.config)


@pytest.mark.parametrize(
    "changes",
    [
        {"revision": "main"},
        {"device": "cpu"},
        {"dtype": "float16"},
        {"batch_size": 0},
        {"max_vram_bytes": 25 * 1024**3},
        {"seed": -1},
        {"quantile_levels": [0.9, 0.5]},
        {"quantile_levels": [0.5, 0.5]},
        {"quantile_levels": [0, 0.5]},
        {"quantile_levels": [0.1, 0.9]},
        {"quantile_levels": [0.5, 1]},
        {"quantile_levels": [float("nan"), 0.5]},
    ],
)
def test_invalid_settings(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        settings(**changes)


@pytest.mark.parametrize(
    "failure", ["cuda", "placement", "dtype", "quantile", "budget", "import", "load"]
)
def test_load_failure_is_explicit(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch, pipeline = install_backend(monkeypatch)
    if failure == "cuda":
        torch.cuda.is_available.return_value = False
    elif failure == "placement":
        pipeline.model.device = "cpu"
    elif failure == "dtype":
        pipeline.model.dtype = "float16"
    elif failure == "quantile":
        pipeline.quantiles = [0.5]
    elif failure == "budget":
        torch.cuda.max_memory_reserved.return_value = 30 * 1024**3
    elif failure == "import":
        monkeypatch.setattr(
            importlib,
            "import_module",
            MagicMock(side_effect=ImportError("missing extra")),
        )
    else:
        pipeline.model.eval.side_effect = RuntimeError("load error")
    with pytest.raises(ForecastError):
        Chronos2Adapter(settings(), cache_dir=tmp_path)


def test_request_bounds_and_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch, pipeline = install_backend(monkeypatch)
    adapter = Chronos2Adapter(settings(), cache_dir=tmp_path)
    batch, _ = pairs()[0]
    for field, limit in [("model_context_length", 1), ("model_prediction_length", 1)]:
        old = getattr(pipeline, field)
        setattr(pipeline, field, limit)
        with pytest.raises(ForecastError, match="native"):
            adapter.forecast(batch)
        setattr(pipeline, field, old)
    narrow = Chronos2Adapter(settings(batch_size=1), cache_dir=tmp_path)
    with pytest.raises(ForecastError, match="multivariate"):
        narrow.forecast(batch)
    pipeline.predict_quantiles.side_effect = RuntimeError("out of memory")
    with pytest.raises(ForecastError, match="out of memory"):
        adapter.forecast(batch)
    pipeline.predict_quantiles.side_effect = None
    pipeline.predict_quantiles.return_value = ([MagicMock()], [])
    with pytest.raises((DataContractError, ForecastError)):
        adapter.forecast(batch)
    torch.cuda.max_memory_reserved.return_value = 30 * 1024**3
    with pytest.raises(ForecastError, match="VRAM"):
        adapter._runtime(0)


def test_variable_univariate_context_horizon(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_backend(monkeypatch)
    adapter = Chronos2Adapter(settings(), cache_dir=tmp_path)
    batch, _ = pairs()[0]
    spec = batch.spec.model_copy(
        update={"context_frames": 3, "horizon_frames": 4, "feature_ids": ("position",)}
    )
    indices = tuple(
        replace(index, context_frames=3, horizon_frames=4) for index in batch.indices
    )
    small = ForecastBatch(spec, indices, batch.context[:, :3, :1])
    result = adapter.forecast(small)
    assert result.values.shape == (2, 4, 1, 3)
    assert result.indices[0].target_slice == small.indices[0].target_slice
    with pytest.raises(DataContractError):
        QuantileForecast(
            spec, indices, (0.1, 0.5, 0.9), result.values[:, :1], result.runtime
        )


def test_adapter_hash_not_silently_ignored() -> None:
    assert (
        "adapter_config_hash"
        not in ModelConfig(model_id=ModelId.PERSISTENCE, seed=42).model_dump()
    )
    with pytest.raises(ValidationError):
        ModelConfig(model_id=ModelId.CHRONOS2, seed=42)
    with pytest.raises(ValidationError):
        ModelConfig(
            model_id=ModelId.PERSISTENCE,
            seed=42,
            adapter_config_hash=metadata_hash(settings()),
        )

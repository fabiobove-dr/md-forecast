"""Evidence linkage, immutable publication and invalid synthesis inputs."""

import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError
from test_benchmark import setup
from test_chronos import settings

from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash, write_metadata
from md_forecast.evaluation.benchmark import GridManifest, evaluate_grid
from md_forecast.evaluation.mvp import (
    BenchmarkInput,
    MetadataInput,
    MVPConfig,
    MVPSnapshot,
    generate_mvp,
    load_mvp_config,
    read_mvp,
)
from md_forecast.evaluation.report import write_benchmark


@pytest.fixture
def cohort(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> MVPConfig:
    models, cell, pairs = setup(monkeypatch, tmp_path)
    grid = GridManifest(cells=(cell,))
    report = write_benchmark(
        tmp_path / "source", grid, evaluate_grid((models,), (pairs,), grid)
    )
    chronos = settings()
    write_metadata(tmp_path / "chronos.json", chronos)
    (tmp_path / "uv.lock").write_text("synthetic lock\n")
    monkeypatch.chdir(tmp_path)
    return MVPConfig(
        report_version="synthetic-v1",
        benchmarks=(
            BenchmarkInput(
                key="integration",
                path=Path("source"),
                expected_hash=report.artifact_id,
                role="integration-train",
            ),
        ),
        metadata=(
            MetadataInput(
                key="chronos",
                path=Path("chronos.json"),
                expected_hash=metadata_hash(chronos),
                kind="chronos",
            ),
        ),
        limitations=("Synthetic evidence is not scientific validation.",),
    )


def generate(config: MVPConfig, output: str = "final") -> MVPSnapshot:
    return generate_mvp(
        config, Path(output), code_commit="a" * 40, lockfile=Path("uv.lock")
    )


def test_repeat_and_exact_evidence(cohort: MVPConfig) -> None:
    first = generate(cohort)
    assert read_mvp(Path("final")) == first
    assert generate(cohort, "repeat").artifact_id == first.artifact_id
    original = pq.read_table("source/cell-0/metrics.parquet").to_pylist()
    for evidence in first.evidence:
        matches = [
            r for r in original if all(r[k] == v for k, v in evidence.selector.items())
        ]
        assert len(matches) == 1
        assert matches[0]["value"] == evidence.value
        assert (
            evidence.table_hash
            == first.benchmarks["integration"].cells[0].table_hashes["metrics"]
        )
    assert all(
        e.unit == "dimensionless"
        for e in first.evidence
        if e.selector["metric"] == "coverage"
    )
    assert (
        Path("final/integration/cell-0/horizon-0.svg").read_bytes()
        == Path("source/cell-0/horizon-0.svg").read_bytes()
    )
    assert "Scientific success" in Path("final/report.md").read_text()
    assert "Corrected CI" in Path("final/integration/cell-0/details.md").read_text()
    with pytest.raises(DataContractError, match="exists"):
        generate(cohort)
    Path("final/report.md").write_text("edited")
    with pytest.raises(DataContractError, match="checksum"):
        read_mvp(Path("final"))


@pytest.mark.parametrize(
    "change,match",
    [
        ({"expected_hash": "sha256:" + "0" * 64}, "identity"),
        ({"role": "development-validation"}, "partition"),
        ({"headline_features": ("absent",)}, "features"),
        ({"model_labels": {"sha256:" + "0" * 64: "unknown"}}, "unknown"),
    ],
)
def test_wrong_source_role_or_labels(
    cohort: MVPConfig, change: dict[str, Any], match: str
) -> None:
    source = cohort.benchmarks[0].model_copy(update=change)
    config = cohort.model_copy(update={"benchmarks": (source,)})
    with pytest.raises(DataContractError, match=match):
        generate(config)
    assert not Path("final").exists()


@pytest.mark.parametrize(
    "change,match",
    [
        ({"metadata": ()}, "linked Chronos"),
        ({"max_metadata_bytes": 1}, "oversized"),
        ({"max_table_rows": 1}, "budget"),
    ],
)
def test_bounded_missing_provenance(
    cohort: MVPConfig, change: dict[str, Any], match: str
) -> None:
    with pytest.raises(DataContractError, match=match):
        generate(cohort.model_copy(update=change))
    assert not Path("final").exists()


def test_corrupt_scientific_table(cohort: MVPConfig) -> None:
    Path("source/cell-0/metrics.parquet").write_bytes(b"corrupt")
    with pytest.raises(DataContractError):
        generate(cohort)
    assert not Path("final").exists()


def test_configuration_boundaries(cohort: MVPConfig) -> None:
    Path("config.json").write_text(cohort.model_dump_json())
    assert load_mvp_config(Path("config.json")) == cohort
    with pytest.raises(DataContractError, match="cannot load"):
        load_mvp_config(Path("missing.json"))
    with pytest.raises(ValidationError, match="duplicate"):
        MVPConfig.model_validate(
            cohort.model_dump() | {"benchmarks": cohort.benchmarks * 2}
        )
    with pytest.raises(ValidationError, match="portable"):
        BenchmarkInput.model_validate(
            cohort.benchmarks[0].model_dump() | {"path": "../source"}
        )


def test_qc_effects_and_cli(
    cohort: MVPConfig,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_qc import analyze

    from md_forecast.cli import main
    from md_forecast.evaluation.mvp import ArtifactInput, _file_hash

    qc = analyze()
    write_metadata(Path("qc.json"), qc)
    effect = pq.read_table("source/cell-0/comparisons.parquet")
    pq.write_table(effect, "effect.parquet")
    config = cohort.model_copy(
        update={
            "metadata": (
                *cohort.metadata,
                MetadataInput(
                    key="qc",
                    path=Path("qc.json"),
                    expected_hash=metadata_hash(qc),
                    kind="qc",
                ),
            ),
            "effects": (
                ArtifactInput(
                    key="effects",
                    path=Path("effect.parquet"),
                    expected_hash=_file_hash(Path("effect.parquet")),
                ),
            ),
        }
    )
    Path("config.json").write_text(config.model_dump_json())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "md-forecast",
            "report-mvp",
            "--config",
            "config.json",
            "--output",
            "final",
            "--code-commit",
            "a" * 40,
        ],
    )
    main()
    text = Path("final/report.md").read_text()
    assert "TRAIN-only source QC" in text
    assert "unavailable" in text
    assert "Whole-horizon effect row" in text
    assert read_mvp(Path("final")).config == config
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "md-forecast",
            "report-mvp",
            "--config",
            "missing.json",
            "--output",
            "another",
            "--code-commit",
            "a" * 40,
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "cannot load" in capsys.readouterr().err

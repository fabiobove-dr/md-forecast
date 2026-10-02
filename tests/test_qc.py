"""Development-only selection, diagnostic identities, and publication regressions."""

import csv
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from pydantic import ValidationError
from test_splits_windows import dataset, grid, official, record, tables

from md_forecast.analysis import report as reporting
from md_forecast.analysis.qc import (
    DevelopmentConfig,
    DevelopmentManifest,
    QCConfig,
    QCReport,
    analyze_development,
    autocorrelation,
    select_development,
)
from md_forecast.core.constants import Split
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash, read_metadata
from md_forecast.data.public.misato import ExtractionReport
from md_forecast.data.registry import Registry, write_registry
from md_forecast.data.schemas import Provenance, TrajectoryManifest
from md_forecast.data.series import to_arrow, write_series
from md_forecast.data.splits import SplitConfig, build_split
from md_forecast.data.windows import WindowConfig

COMMIT = "a" * 40
LOCK_HASH = "sha256:" + "b" * 64


def config(**changes: object) -> QCConfig:
    return QCConfig.model_validate(
        {
            "development": {"seed": 42, "max_groups": 3},
            "near_constant_std": {"position": 0.1, "constant": 0.1},
            "max_lag_frames": 2,
            "event_sigma": 1.0,
            "windows": grid().model_dump(),
        }
        | changes
    )


def analyze(settings: QCConfig | None = None) -> QCReport:
    split = official()
    series = tables(split)
    loaded = []

    def loader(source: TrajectoryManifest) -> pa.Table:
        assert source.split == Split.TRAIN
        loaded.append(source.trajectory_id)
        return series[source.trajectory_id]

    result = analyze_development(
        split, settings or config(), loader, code_commit=COMMIT, lockfile_hash=LOCK_HASH
    )
    assert tuple(loaded) == result.development.trajectory_ids
    return result


def test_development_selection_and_round_trip() -> None:
    split = official()
    selection = DevelopmentConfig(seed=7, max_groups=1)
    subset = select_development(split, selection)
    assert len(subset.trajectory_ids) == 1
    assert subset == select_development(split, selection)
    changed = select_development(split, DevelopmentConfig(seed=8, max_groups=1))
    assert metadata_hash(subset) != metadata_hash(changed)
    assert DevelopmentManifest.model_validate_json(subset.model_dump_json()) == subset
    with pytest.raises(ValidationError, match="selection"):
        DevelopmentManifest.model_validate(
            subset.model_dump() | {"trajectory_ids": ("trajectory-5",)}
        )
    grouped = build_split(
        split.registry, SplitConfig(mode="grouped", seed=2, ratios=(0.5, 1 / 3, 1 / 6))
    )
    with pytest.raises(DataContractError, match="official"):
        select_development(grouped, selection)
    no_train = build_split(
        Registry(dataset=dataset(), trajectories=(record(5),)),
        SplitConfig(mode="official", seed=2),
    )
    with pytest.raises(DataContractError, match="no official training"):
        select_development(no_train, selection)
    dependent = record(0, trajectory_id="replica", replicate_id="replica")
    grouped_source = Registry(
        dataset=dataset(), trajectories=(record(0), dependent, record(5))
    )
    related = select_development(
        build_split(grouped_source, SplitConfig(mode="official", seed=2)), selection
    )
    assert related.trajectory_ids == ("replica", "trajectory-0")


def test_numerical_diagnostics_and_feasibility() -> None:
    result = analyze()
    assert QCReport.model_validate_json(result.model_dump_json()) == result
    varying, constant = result.trajectories[0].features
    np.testing.assert_allclose(varying.quantiles, [0, 1.25, 2.5, 3.75, 5])
    assert varying.acf is not None
    np.testing.assert_allclose(np.asarray(varying.acf), [1, 0.5, 1 / 17.5])
    assert varying.variance == pytest.approx(35 / 12)
    assert varying.linear_drift_per_frame == pytest.approx(1)
    assert varying.decay_frames == 2 and not varying.decay_censored
    assert varying.half_mean_shift_sd == pytest.approx(3 / np.std(np.arange(6)))
    assert varying.jump_count == 0 and varying.missing_count == 0
    assert constant.constant and constant.near_constant
    assert constant.acf is None and constant.half_mean_shift_sd is None
    assert result.trajectories[0].cross_correlation == ((1.0, None), (None, None))
    assert sum(row.window_count for row in result.feasibility) == 24
    short = analyze(config(max_lag_frames=1))
    assert short.trajectories[0].features[0].decay_censored
    impossible = analyze(
        config(
            windows=WindowConfig(
                contexts=(6.0,), horizons=(1.0,), stride_frames=1
            ).model_dump()
        )
    )
    assert all(row.window_count == 0 and row.reason for row in impossible.feasibility)
    physical = analyze(
        config(
            windows=WindowConfig(
                contexts=(1.0,), horizons=(1.0,), unit="ps", stride_frames=1
            ).model_dump()
        )
    )
    assert all("uniform" in (row.reason or "") for row in physical.feasibility)
    assert autocorrelation(np.ones(6), 5) is None
    curve = autocorrelation(np.arange(3, dtype=np.float64), 9)
    assert curve is not None
    np.testing.assert_allclose(curve, [1, 0, -0.5])


@pytest.mark.parametrize(
    "changes",
    [
        {"split_hash": LOCK_HASH},
        {"trajectories": []},
        {"feasibility": []},
        {"code_commit": "bad"},
        {"schema_version": 2},
    ],
)
def test_invalid_report(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        QCReport.model_validate(analyze().model_dump() | changes)


def test_qc_failures_before_loading() -> None:
    split = official()
    with pytest.raises(DataContractError, match="thresholds"):
        analyze_development(
            split,
            config(near_constant_std={}),
            lambda _: pytest.fail("must not load"),
            code_commit=COMMIT,
            lockfile_hash=LOCK_HASH,
        )
    wrong = tables(split)["trajectory-1"]
    with pytest.raises(DataContractError, match="source metadata"):
        analyze_development(
            split,
            config(),
            lambda _: wrong,
            code_commit=COMMIT,
            lockfile_hash=LOCK_HASH,
        )
    one = build_split(
        Registry(dataset=dataset(), trajectories=(record(0, frame_count=1),)),
        SplitConfig(mode="official", seed=42),
    )
    with pytest.raises(DataContractError, match="two samples"):
        analyze_development(
            one,
            config(),
            lambda _: pytest.fail("must not load"),
            code_commit=COMMIT,
            lockfile_hash=LOCK_HASH,
        )
    series = tables(split)
    source = split.registry.trajectories[0]
    huge = series[source.trajectory_id].set_column(
        1, "position", pa.array([1e308, -1e308] * 3)
    )
    with pytest.raises(DataContractError, match="overflow"):
        analyze_development(
            split, config(), lambda _: huge, code_commit=COMMIT, lockfile_hash=LOCK_HASH
        )


def test_jumps_and_near_constant_channels() -> None:
    split = official()
    series = tables(split)
    for source in split.registry.trajectories:
        values = np.column_stack(
            [
                np.array([0, 0, 10, 0, 0, 0], dtype=np.float64),
                np.arange(6, dtype=np.float64) * 1e-8,
            ]
        )
        series[source.trajectory_id] = to_arrow(
            source, dataset(), np.arange(6, dtype=np.float64), values
        )
    result = analyze_development(
        split,
        config(),
        lambda source: series[source.trajectory_id],
        code_commit=COMMIT,
        lockfile_hash=LOCK_HASH,
    )
    jump, nearly = result.trajectories[0].features
    assert jump.jump_count == 2
    assert nearly.near_constant and not nearly.constant and nearly.acf is not None


@pytest.mark.parametrize(
    "field,value",
    [
        ("feature_id", "wrong"),
        ("quantiles", []),
        ("missing_count", 1),
        ("acf", [1.0]),
    ],
)
def test_invalid_feature_rows(field: str, value: object) -> None:
    payload = analyze().model_dump(mode="json")
    payload["trajectories"][0]["features"][0][field] = value
    with pytest.raises(ValidationError):
        QCReport.model_validate(payload)


def test_invalid_report_linkage_and_matrix() -> None:
    payload = analyze().model_dump(mode="json")
    changed = config(development={"seed": 43, "max_groups": 3})
    payload["config"] = changed.model_dump(mode="json")
    payload["config_hash"] = metadata_hash(changed)
    with pytest.raises(ValidationError, match="selection"):
        QCReport.model_validate(payload)
    payload = analyze().model_dump(mode="json")
    payload["trajectories"][0]["cross_correlation"] = [[1.0]]
    with pytest.raises(ValidationError, match="dimensions"):
        QCReport.model_validate(payload)


def extraction_fixture(root: Path) -> None:
    root.mkdir()
    split = official()
    write_registry(root / "registry.json", split.registry)
    series = tables(split)
    paths = {}
    for source in split.registry.trajectories:
        filename = source.trajectory_id + ".parquet"
        paths[source.trajectory_id] = filename
        # No held-out numerical file is present: an accidental read must fail.
        if source.split == Split.TRAIN:
            write_series(root / filename, series[source.trajectory_id])
    provenance = Provenance.model_validate(
        split.registry.trajectories[0].model_dump(include=set(Provenance.model_fields))
    )
    extraction = ExtractionReport(
        source=provenance,
        split_checksums={split: "md5:" + "a" * 32 for split in Split},
        requested=tuple(paths),
        exported=paths,
        issues=(),
    )
    (root / "qc.json").write_text(extraction.model_dump_json())


def test_atomic_qc_bundle_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from md_forecast.cli import main

    root = tmp_path / "input"
    extraction_fixture(root)
    lock = tmp_path / "uv.lock"
    lock.write_text("fixture")
    settings = tmp_path / "config.json"
    settings.write_text(config().model_dump_json())
    output = tmp_path / "report"
    monkeypatch.setattr(
        "sys.argv",
        [
            "md-forecast",
            "qc-misato",
            "--input",
            str(root),
            "--output",
            str(output),
            "--config",
            str(settings),
            "--code-commit",
            COMMIT,
            "--lockfile",
            str(lock),
        ],
    )
    main()
    result = read_metadata(output / "report.json", QCReport)
    assert len(result.trajectories) == 3
    assert (
        read_metadata(output / "development.json", DevelopmentManifest)
        == result.development
    )
    with (output / "acf.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12 and not any("trajectory-5" in row.values() for row in rows)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    before = set(tmp_path.iterdir())

    def fail(*args: object) -> None:
        raise OSError("simulated curve write failure")

    monkeypatch.setattr(reporting, "_write_curves", fail)
    with pytest.raises(DataContractError, match="before publication"):
        reporting.run_misato_qc(
            root, tmp_path / "failed", config(), code_commit=COMMIT, lockfile=lock
        )
    assert set(tmp_path.iterdir()) == before
    with pytest.raises(DataContractError, match="cannot load QC"):
        reporting.load_qc_config(tmp_path / "absent")
    metadata = json.loads((root / "qc.json").read_text())
    metadata["exported"]["trajectory-0"] = "../outside.parquet"
    (root / "qc.json").write_text(json.dumps(metadata))
    with pytest.raises(DataContractError, match="direct local"):
        reporting.run_misato_qc(
            root, tmp_path / "unsafe", config(), code_commit=COMMIT, lockfile=lock
        )


def test_reject_mapped_heldout_metadata_before_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyarrow.parquet as pq

    root = tmp_path / "input"
    extraction_fixture(root)
    # Accidental mapping/substitution of a held-out file under a training filename.
    write_series(root / "trajectory-0.parquet", tables(official())["trajectory-5"])
    lock = tmp_path / "uv.lock"
    lock.write_text("fixture")

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("held-out numerical payload must not be read")

    monkeypatch.setattr(pq.ParquetFile, "read", forbidden)
    with pytest.raises(DataContractError, match="before payload read"):
        reporting.run_misato_qc(
            root, tmp_path / "report", config(), code_commit=COMMIT, lockfile=lock
        )

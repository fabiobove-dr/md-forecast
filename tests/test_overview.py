"""Offline saved-result integrity, matched windows and safe publication."""

import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError
from test_benchmark import setup

from md_forecast.core.exceptions import DataContractError
from md_forecast.evaluation.benchmark import GridManifest, evaluate_cell
from md_forecast.evaluation.overview import (
    OverviewConfig,
    OverviewSource,
    SavedPoint,
    _match,
    _point,
    confined,
    file_hash,
    generate_overview,
    load_overview,
)
from md_forecast.evaluation.report import write_benchmark
from md_forecast.evaluation.structure_view import (
    StructureConfig,
    _atoms,
    _static,
    render_structure,
)


def saved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> OverviewConfig:
    """Produce a tiny real-format bundle through existing evaluation helpers."""
    models, manifest, pairs = setup(monkeypatch, tmp_path)
    root = tmp_path / "benchmark"
    report = write_benchmark(
        root, GridManifest(cells=(manifest,)), (evaluate_cell(models, pairs, manifest),)
    )
    return OverviewConfig(
        conclusion='<script>alert("escaped")</script>',
        sources=(
            OverviewSource(
                title="Synthetic <untrusted>",
                role="Synthetic regression",
                note="Not scientific evidence",
                kind="benchmark",
                root=root,
                expected_hash=report.artifact_id,
            ),
        ),
    )


def test_saved_benchmark_to_offline_html(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = saved(monkeypatch, tmp_path)
    output = tmp_path / "overview.html"
    panels = generate_overview(config, output)
    assert len(panels) == 2
    assert len(panels[0].windows) == 15
    assert panels[0].models
    assert panels[0].metrics
    assert panels[0].windows[0].target == (6.0, 7.0)
    page = output.read_text()
    assert "&lt;script&gt;alert" in page
    assert "Synthetic \\u003cuntrusted>" in page
    assert "UNRATED" in page and "No structure supplied" in page
    assert "<script src=" not in page and "fetch(" not in page
    assert "Signed error" in page and "Download every window" in page
    assert generate_overview(config, output) == panels
    with pytest.raises(DataContractError, match="immutable"):
        generate_overview(config, config.sources[0].root / "index.html")
    with pytest.raises(DataContractError, match="max_rows"):
        load_overview(config.model_copy(update={"max_rows": 1}))
    with pytest.raises(DataContractError, match="max_input_bytes"):
        load_overview(config.model_copy(update={"max_input_bytes": 1}))
    wrong = config.sources[0].model_copy(update={"expected_hash": "sha256:" + "0" * 64})
    with pytest.raises(DataContractError, match="identity differs"):
        load_overview(config.model_copy(update={"sources": (wrong,)}))
    table = config.sources[0].root / "cell-0/predictions.parquet"
    table.write_bytes(table.read_bytes() + b"tamper")
    before = output.read_bytes()
    with pytest.raises(DataContractError, match="checksum"):
        generate_overview(config, output)
    assert output.read_bytes() == before


def test_matching_rejects_false_comparisons(tmp_path: Path) -> None:
    points = [
        SavedPoint("t", "g", 0, lead, float(lead), float(lead), origin=0.0)
        for lead in (1, 2)
    ]
    matched = _match({"zero-shot": points, "fine-tuned": list(reversed(points))}, 2)
    assert matched[0].points["zero-shot"] == (1.0, 2.0)
    with pytest.raises(DataContractError, match="unmatched"):
        _match(
            {"zero": points, "fine": [replace(points[0], target=10.0), points[1]]}, 2
        )
    with pytest.raises(DataContractError, match="duplicate"):
        _match({"zero": points + points}, 2)
    with pytest.raises(DataContractError, match="missing or extra"):
        _match({"zero": points[:1]}, 2)
    with pytest.raises(DataContractError, match="finite"):
        _match({"zero": [replace(p, point=float("nan")) for p in points]}, 2)
    with pytest.raises(DataContractError, match="invalid saved frame"):
        _match({"zero": [replace(points[0], start=-1), points[1]]}, 2)
    with pytest.raises(DataContractError, match="identity"):
        _match({"zero": [replace(points[0], group_id=""), points[1]]}, 2)
    with pytest.raises(DataContractError, match="origin"):
        _match({"zero": [replace(points[0], origin=2.0), points[1]]}, 2)
    with pytest.raises(DataContractError, match="no forecasts"):
        _match({"zero": []}, 2)
    with pytest.raises(DataContractError, match="escapes"):
        confined(tmp_path, "../outside.parquet")
    data = dict(
        trajectory_id="t",
        group_id="g",
        start=0,
        lead=1,
        point=2.0,
        target=3.0,
        levels=(0.1, 0.5, 0.9),
        quantiles=(1.0, 2.0, 3.0),
    )
    assert _point(data).point == 2.0
    for changes in (
        {"quantiles": (3.0, 2.0, 1.0)},
        {"point": 1.0},
        {"levels": (0.1, 0.1, 0.9)},
        {"levels": (-1.0, 0.5, 0.9)},
    ):
        with pytest.raises(DataContractError):
            _point(data | changes)


def test_followup_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = setup(monkeypatch, tmp_path)[1].spec.dataset
    root = tmp_path / "followup"
    (root / "c2-h2").mkdir(parents=True)
    models = {}
    hashes = {}
    for model in ("persistence", "zero-shot", "fine-tuned"):
        rows = [
            dict(
                trajectory_id="t",
                group_id="g",
                start=0,
                context_frames=2,
                horizon_frames=2,
                feature_id="position",
                lead=i,
                point=float(i),
                target=float(i) + 1.0,
                origin=0.0,
                **{
                    "quantile-0.1": float(i) - 1.0,
                    "quantile-0.5": float(i),
                    "quantile-0.9": float(i) + 1.0,
                },
            )
            for i in (1, 2)
        ]
        path = root / "c2-h2" / f"position-{model}-leads.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        hashes[str(path.relative_to(root))] = file_hash(path).removeprefix("sha256:")
        models[model] = dict(
            errors=[dict(lead=0, mae=1.0, rmse=1.0, bias=-1.0)],
            per_lead_error=[dict(lead=i, mae=1.0, rmse=1.0, bias=-1.0) for i in (1, 2)],
            probability=[
                dict(
                    lead=0,
                    metrics={"coverage-0.1-0.9": 1.0},
                    marginal_ci={"coverage-0.1-0.9": [1.0, 1.0]},
                    interval_status="synthetic",
                )
            ],
            per_lead_probability=[],
        )
    summary = dict(
        results={"c2-h2": {"position": models}},
        file_sha256=hashes,
        code_commit="0" * 40,
        split_hash="sha256:" + "1" * 64,
        config_hash="sha256:" + "2" * 64,
    )
    path = root / "summary.json"
    path.write_text(json.dumps(summary))
    config = OverviewConfig(
        conclusion="Synthetic only",
        sources=(
            OverviewSource(
                title="Followup",
                role="Synthetic",
                note="Not real",
                kind="probability",
                root=root,
                expected_hash=file_hash(path),
                dataset=dataset,
            ),
        ),
    )
    panels = generate_overview(config, tmp_path / "followup.html")
    assert panels[0].windows[0].bands["zero-shot"] == ((0.0, 2.0), (1.0, 3.0))
    assert panels[0].windows[0].target == (2.0, 3.0)
    path.write_text("{}")
    with pytest.raises(DataContractError, match="checksum"):
        load_overview(config)
    with pytest.raises(ValidationError, match="definitions"):
        OverviewSource(
            title="x",
            role="x",
            note="x",
            kind="probability",
            root=root,
            expected_hash="sha256:" + "0" * 64,
        )


def test_exact_structure_and_static_fallback(tmp_path: Path) -> None:
    pdb = (
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00"
        "           C  \n"
        "HETATM    2  C1  LIG A   2       2.000   3.000   4.000  1.00  0.00"
        "           C  \n"
    )
    path = tmp_path / "source.pdb"
    path.write_text(pdb)
    config = StructureConfig(
        pdb=path,
        pdb_hash=file_hash(path),
        library=tmp_path / "library.js",
        license=tmp_path / "LICENSE",
        source_uri="https://example.org/source",
        license_id="CC0",
        description="Observed synthetic reference",
        embedding_permitted=True,
        protein=(1,),
        ligand=(2,),
        regions={"A": (1,)},
    )
    atoms = _atoms(pdb)
    assert atoms[1] == (1.0, 2.0, 3.0)
    assert "Atom serial 1" in _static(atoms, config)
    assert "xy plane" in _static(atoms, config)
    with pytest.raises(DataContractError, match="unique-serial"):
        _atoms(pdb + pdb)
    with pytest.raises(DataContractError, match="finite"):
        _atoms("")
    with pytest.raises(ValidationError, match="embedding"):
        StructureConfig.model_validate(
            config.model_dump() | {"embedding_permitted": False}
        )
    with pytest.raises(ValidationError, match="overlap"):
        StructureConfig.model_validate(config.model_dump() | {"ligand": (1,)})
    with pytest.raises(ValidationError, match="protein selection"):
        StructureConfig.model_validate(config.model_dump() | {"regions": {"bad": (2,)}})
    with pytest.raises(ValidationError, match="unique"):
        StructureConfig.model_validate(config.model_dump() | {"protein": (1, 1)})
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json())
    with pytest.raises(DataContractError, match="cannot embed"):
        render_structure(config_path)


def test_local_structure_publication_and_integrity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tiny renderer fixture tests assembly; the real renderer is browser-verified."""
    import md_forecast.evaluation.structure_view as module

    pdb = tmp_path / "source.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00\n"
        "HETATM    2  C1  LIG A   2       2.000   3.000   4.000  1.00  0.00\n"
    )
    library = tmp_path / "library.js"
    library.write_text("// synthetic renderer fixture\n")
    license_path = tmp_path / "LICENSE"
    license_path.write_text("Synthetic license <escaped>")
    monkeypatch.setattr(module, "RENDERER_HASH", file_hash(library))
    monkeypatch.setattr(module, "LICENSE_HASH", file_hash(license_path))
    config = StructureConfig(
        pdb=pdb,
        pdb_hash=file_hash(pdb),
        library=library,
        license=license_path,
        source_uri="https://example.org/ref",
        license_id="CC0",
        description="__LIBRARY__ <script>unsafe</script>",
        embedding_permitted=True,
        protein=(1,),
        ligand=(2,),
        regions={"A": (1,)},
    )
    path = tmp_path / "structure.json"
    path.write_text(config.model_dump_json())
    page = render_structure(path)
    assert "iframe" in page and "srcdoc=" in page
    assert "__LIBRARY__" in page  # placeholders in user text are never substituted
    assert "unsafe" in page and "&amp;lt;script&amp;gt;" in page
    assert "xy plane" in page and "Export PNG" in page
    library.write_text("corrupted")
    with pytest.raises(DataContractError, match="checksum"):
        render_structure(path)
    library.write_text("// synthetic renderer fixture\n")
    invalid = config.model_dump() | {"protein": (3,)}
    path.write_text(
        StructureConfig.model_validate(invalid | {"regions": {}}).model_dump_json()
    )
    with pytest.raises(DataContractError, match="absent"):
        render_structure(path)


def test_overview_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from md_forecast.cli import main

    config = saved(monkeypatch, tmp_path)
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json())
    output = tmp_path / "result.html"
    argv = [
        "md-forecast",
        "report-forecast",
        "--config",
        str(path),
        "--output",
        str(output),
    ]
    monkeypatch.setattr("sys.argv", argv)
    main()
    assert output.exists()
    path.unlink()
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "cannot read or write overview" in capsys.readouterr().err
    assert output.exists()

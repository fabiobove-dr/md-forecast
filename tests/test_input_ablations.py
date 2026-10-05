"""Ablation pairing refuses changed future labels, groups and ambiguous inputs."""

import tomllib
from pathlib import Path

import pyarrow as pa
import pytest
from pydantic import ValidationError

from md_forecast.core.exceptions import DataContractError
from md_forecast.evaluation.ablations import AblationConfig, paired_effects, target_rows


def budget() -> AblationConfig:
    return AblationConfig.model_validate(
        tomllib.loads(Path("configs/benchmarks/input-ablations.toml").read_text())
    )


def test_input_sets_require_target_only_reference() -> None:
    data = budget().model_dump()
    for variants in (
        {"base": ("other",), "next": ("ligand_rmsd",)},
        {"base": ("ligand_rmsd",), "next": ("other",)},
        {"base": ("ligand_rmsd",), "next": ("ligand_rmsd",)},
        {"base": ("ligand_rmsd",), "next": ("ligand_rmsd", "ligand_rmsd")},
    ):
        with pytest.raises(ValidationError):
            AblationConfig.model_validate(data | {"variants": variants})


def metrics(offset: float = 0) -> pa.Table:
    return pa.Table.from_pylist(
        [
            dict(
                level="group",
                metric="mae",
                feature_id="ligand_rmsd",
                model_hash="model",
                step=step,
                identity=f"g{group}",
                value=group + offset,
            )
            for step in (0, 1, 2)
            for group in range(6)
        ]
    )


def test_paired_native_unit_effects_and_unresolved_corrected_tails() -> None:
    effects = paired_effects(
        metrics(), metrics(-0.5), "ligand_rmsd", "model", budget().evaluation
    )
    assert [r["step"] for r in effects] == [0, 1, 2]
    assert all(r["mae_difference"] == -0.5 for r in effects)
    assert all(r["groups"] == 6 for r in effects)
    assert all(r["ci_lower"] is None for r in effects)
    assert all(
        r["interval_status"] == "insufficient-bootstrap-resolution" for r in effects
    )
    assert all(r["marginal_ci_lower"] == -0.5 for r in effects)
    with pytest.raises(DataContractError, match="groups/lead"):
        paired_effects(
            metrics(), metrics().slice(1), "ligand_rmsd", "model", budget().evaluation
        )
    with pytest.raises(DataContractError, match="duplicate"):
        paired_effects(
            metrics(),
            pa.concat_tables([metrics(), metrics()]),
            "ligand_rmsd",
            "model",
            budget().evaluation,
        )


def test_exact_future_label_pairing_ignores_order_and_auxiliary_channels() -> None:
    rows = [
        dict(
            trajectory_id="t",
            system_id="s",
            group_id="g",
            start=0,
            step=step,
            target_frame=40 + step,
            target=float(step),
            feature_id=feature,
            model_hash="model",
        )
        for step in (1, 2)
        for feature in ("ligand_rmsd", "auxiliary")
    ]
    table = pa.Table.from_pylist(rows)
    expected = target_rows(table, "ligand_rmsd", "model")
    assert expected.num_rows == 2
    assert target_rows(pa.Table.from_pylist(rows[::-1]), "ligand_rmsd", "model").equals(
        expected
    )
    rows[0]["target"] = 99.0
    assert not target_rows(pa.Table.from_pylist(rows), "ligand_rmsd", "model").equals(
        expected
    )


def test_multiple_grid_cells_are_rejected() -> None:
    data = budget().model_dump()
    data["grid"]["contexts"] = (20.0, 40.0)
    with pytest.raises(ValidationError, match="one frozen"):
        AblationConfig.model_validate(data)


def test_observed_control_preserves_target_groups_and_uses_no_future() -> None:
    import numpy as np
    from test_baselines import pairs

    from md_forecast.data.forecast import ForecastBatch
    from md_forecast.evaluation.ablations import observed_input

    original, future = pairs()[0]
    values = original.context.copy()
    values[:, :, 1] = np.arange(values.shape[1])
    batch = ForecastBatch(original.spec, original.indices, values)
    shifted = observed_input(batch, "position", "shifted", shift_frames=3)
    np.testing.assert_array_equal(shifted.context[:, :, 0], values[:, :, 0])
    np.testing.assert_array_equal(
        shifted.context[:, :, 1], np.roll(values[:, :, 1], 3, axis=1)
    )
    assert shifted.indices == batch.indices
    np.testing.assert_array_equal(batch.context, values)
    assert not shifted.context.flags.writeable
    only = observed_input(batch, "position", "target-only", shift_frames=3)
    assert only.spec.feature_ids == ("position",)
    np.testing.assert_array_equal(only.context[:, :, 0], values[:, :, 0])
    assert observed_input(batch, "position", "joint", shift_frames=3) is batch
    assert future is not None
    future[:] = 1e9
    np.testing.assert_array_equal(
        observed_input(batch, "position", "shifted", shift_frames=3).context,
        shifted.context,
    )
    for shift in (0, 6):
        with pytest.raises(DataContractError, match="shift"):
            observed_input(batch, "position", "shifted", shift_frames=shift)
    with pytest.raises(DataContractError, match="absent"):
        observed_input(batch, "missing", "joint", shift_frames=3)
    with pytest.raises(DataContractError, match="auxiliary"):
        observed_input(only, "position", "shifted", shift_frames=3)

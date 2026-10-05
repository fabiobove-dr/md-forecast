"""Select regularized statistical and NLinear comparators on frozen native VAL."""

import argparse
import hashlib
import json
import subprocess
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Annotated

import numpy as np
from pydantic import Field
from run_followup_diagnostics import load_native

from md_forecast.core.constants import ModelId, Split
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash, read_metadata, write_metadata
from md_forecast.data.forecast import ForecastBatch, iter_forecasts
from md_forecast.data.preprocessing import SeriesLoader
from md_forecast.data.schemas import BoundaryModel
from md_forecast.data.series import FloatArray
from md_forecast.data.splits import SplitManifest
from md_forecast.data.windows import WindowConfig
from md_forecast.evaluation.baselines import evaluate_baselines, normalized_group_mae
from md_forecast.features.structural import BASE_FEATURE_IDS
from md_forecast.models.base import ForecastModel, ModelConfig
from md_forecast.models.baselines import StatisticalBaseline
from md_forecast.models.learned import (
    NLinearModel,
    NLinearState,
    TrainingConfig,
    fit_nlinear,
    select_nlinear,
)


class BaselineSelectionConfig(BoundaryModel):
    """Finite candidate/budget declaration; no automatic architecture search."""

    grid: WindowConfig
    training: TrainingConfig
    statistics: Annotated[tuple[ModelConfig, ...], Field(min_length=1)]
    compact: Annotated[tuple[ModelConfig, ...], Field(min_length=1)]


def conditioning(
    model: NLinearModel, split: SplitManifest, grid: WindowConfig, loader: SeriesLoader
) -> dict[str, object]:
    """Describe actual scaled training design, including its centered zero column."""
    batches = list(
        iter_forecasts(
            split,
            grid,
            Split.TRAIN,
            BASE_FEATURE_IDS,
            loader,
            batch_size=model.state.training.batch_size,
        )
    )
    contexts = np.concatenate([batch.context for batch, _ in batches])
    centered = (contexts - contexts[:, -1:, :]) / np.asarray(model.state.scaler.scale)
    design = np.concatenate(
        (centered, np.ones((len(centered), 1, centered.shape[2]))), axis=1
    )
    channels = {}
    for index, feature in enumerate(BASE_FEATURE_IDS):
        matrix = design[:, :, index]
        singular = np.linalg.svd(matrix, compute_uv=False)
        tolerance = singular[0] * max(matrix.shape) * np.finfo(np.float64).eps
        rank = int(np.count_nonzero(singular > tolerance))
        ridge_conditions = {}
        for ridge in (0.0, model.config.ridge):
            penalty = np.sqrt(ridge) * np.eye(matrix.shape[1])
            penalty[-1, -1] = 0
            augmented = np.vstack((matrix, penalty))
            values = np.linalg.svd(augmented, compute_uv=False)
            augmented_rank = int(
                np.count_nonzero(
                    values > values[0] * max(augmented.shape) * np.finfo(np.float64).eps
                )
            )
            ridge_conditions[str(ridge)] = {
                "rank": augmented_rank,
                "condition_number": float(values[0] / values[-1])
                if augmented_rank == matrix.shape[1]
                else None,
            }
        channels[feature] = {
            "rank": rank,
            "columns": matrix.shape[1],
            "unregularized_condition_number": float(singular[0] / singular[-1])
            if rank == matrix.shape[1]
            else None,
            "selected_augmented_design": ridge_conditions,
        }
    return {
        "rows": len(contexts),
        "columns_per_channel": design.shape[1],
        "channels": channels,
        "design_bytes": design.nbytes,
        "parameters": len(BASE_FEATURE_IDS)
        * design.shape[1]
        * model.state.spec.horizon_frames,
    }


def select_cell(
    root: Path,
    config: BaselineSelectionConfig,
    split: SplitManifest,
    loader: SeriesLoader,
    grid: WindowConfig,
) -> dict[str, object]:
    """Reuse TRAIN fitting and grouped VAL selection; preserve ridge-zero evidence."""
    start = perf_counter()
    compact, choice = select_nlinear(
        split, grid, BASE_FEATURE_IDS, loader, config.compact, config.training
    )
    write_metadata(root / "compact-state.json", compact.state)
    write_metadata(root / "compact-selection.json", choice)
    zero_config = next(c for c in config.compact if c.ridge == 0)
    zero = (
        compact
        if compact.config == zero_config
        else fit_nlinear(
            split, grid, BASE_FEATURE_IDS, loader, zero_config, config.training
        )
    )
    write_metadata(root / "unregularized-state.json", zero.state)
    candidates = tuple(StatisticalBaseline(c) for c in config.statistics)

    def pairs() -> Iterator[tuple[ForecastBatch, FloatArray | None]]:
        return iter_forecasts(
            split,
            grid,
            Split.VALIDATION,
            BASE_FEATURE_IDS,
            loader,
            batch_size=config.training.batch_size,
            with_targets=True,
        )

    stats = evaluate_baselines(candidates, pairs())
    scales = np.asarray(compact.state.scaler.scale, dtype=np.float64)
    scores = normalized_group_mae(stats, scales)
    selected = {
        model: int(
            min(
                (i for i, c in enumerate(config.statistics) if c.model_id == model),
                key=lambda i: (scores[i], metadata_hash(config.statistics[i])),
            )
        )
        for model in (ModelId.AR, ModelId.VAR)
    }
    selected_models: tuple[ForecastModel, ...] = tuple(
        candidates[i]
        for i in range(len(candidates))
        if candidates[i].config.model_id
        in (ModelId.PERSISTENCE, ModelId.CONTEXT_MEAN, ModelId.LINEAR)
        or i in selected.values()
    ) + (compact,)
    if zero.config != compact.config:
        selected_models += (zero,)
    matched = evaluate_baselines(selected_models, pairs())
    write_metadata(root / "statistical-validation.json", stats)
    write_metadata(root / "matched-validation.json", matched)
    # Every fitted-state reload must retain exact adapter identity and predictions.
    first = next(pairs())[0]
    reloaded = NLinearModel(read_metadata(root / "compact-state.json", NLinearState))
    assert reloaded.artifact_hash == compact.artifact_hash
    np.testing.assert_array_equal(reloaded.predict(first), compact.predict(first))
    per_feature = {
        feature: min(
            stats.scores,
            key=lambda s: (
                float(np.mean([group.mae[index] for group in s.groups])),
                s.config_hash,
            ),
        ).config_hash
        for index, feature in enumerate(BASE_FEATURE_IDS)
    }
    return {
        "selected_compact_hash": compact.artifact_hash,
        "compact_config": compact.config.model_dump(mode="json"),
        "training_trajectories": len(compact.state.training_trajectory_ids),
        "training_windows": compact.state.training_window_count,
        "conditioning": conditioning(compact, split, grid, loader),
        "selected_statistics": {
            key: config.statistics[i].model_dump(mode="json")
            for key, i in selected.items()
        },
        "statistics_dimensions": {
            key: {
                "observations": int(grid.contexts[0])
                - (config.statistics[i].lags or 0),
                "design_columns": (config.statistics[i].lags or 0)
                * (len(BASE_FEATURE_IDS) if key == ModelId.VAR else 1)
                + 1,
            }
            for key, i in selected.items()
        },
        "normalized_statistical_validation_mae": {
            s.config_hash: float(value)
            for s, value in zip(stats.scores, scores, strict=True)
        },
        "best_statistical_per_feature": per_feature,
        "reload_exact": True,
        "seconds": perf_counter() - start,
    }


def run(args: argparse.Namespace) -> None:
    """Publish bounded selection evidence without confirmation access."""
    if args.output.exists():
        raise DataContractError("output exists; choose a fresh directory")
    if subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only", "--", "src", "examples", "configs"],
        text=True,
    ).strip():
        raise DataContractError("commit source/config before selection provenance")
    config = BaselineSelectionConfig.model_validate_json(args.config.read_text())
    split, loader = load_native(args.native)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=args.output.parent, prefix=".baseline-selection-"
    ) as temporary:
        root = Path(temporary)
        write_metadata(root / "config.json", config)
        cells = {}
        for context in config.grid.contexts:
            for horizon in config.grid.horizons:
                grid = config.grid.model_copy(
                    update={"contexts": (context,), "horizons": (horizon,)}
                )
                name = f"c{int(context)}-h{int(horizon)}"
                directory = root / name
                directory.mkdir()
                cells[name] = select_cell(directory, config, split, loader, grid)
                print(name, cells[name]["compact_config"], flush=True)
        provenance = {
            "config_hash": metadata_hash(config),
            "split_hash": metadata_hash(split),
            "cohort_file_sha256": hashlib.sha256(
                (args.native / "cohort.json").read_bytes()
            ).hexdigest(),
            "code_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "lockfile_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
            "cells": cells,
            "file_sha256": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*")
                if p.is_file()
            },
        }
        (root / "provenance.json").write_text(
            json.dumps(provenance, indent=2, allow_nan=False) + "\n"
        )
        root.rename(args.output)
    print(f"published {args.output}")


def main() -> None:
    """Require an explicit fresh output; no confirmation purpose is exposed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/followup-baselines.json"),
    )
    parser.add_argument(
        "--native", type=Path, default=Path("data/processed/followup-development")
    )
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

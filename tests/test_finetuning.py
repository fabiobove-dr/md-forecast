"""Training, checkpoint integrity and leakage regressions without Torch/GPU."""

import hashlib
import importlib
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError
from test_baselines import grid, series, source
from test_chronos import install_backend, settings

from md_forecast.core.constants import Split
from md_forecast.core.exceptions import DataContractError, ForecastError
from md_forecast.data.artifacts import metadata_hash, read_metadata, write_metadata
from md_forecast.data.forecast import ForecastBatch, iter_forecasts
from md_forecast.data.predictions import RuntimeStats
from md_forecast.data.registry import Registry
from md_forecast.data.schemas import TrajectoryManifest
from md_forecast.data.splits import SplitConfig, build_split
from md_forecast.data.windows import WindowConfig
from md_forecast.models import finetuning as ft


def training(**changes: object) -> ft.FineTuneConfig:
    values = tomllib.loads(Path("configs/models/chronos2-finetuning.toml").read_text())
    return ft.FineTuneConfig.model_validate(values | changes)


def manifest(**changes: object) -> ft.FineTuneManifest:
    split = source()
    arrays = series(split)

    def loader(record: TrajectoryManifest) -> Any:
        assert record.split != Split.TEST
        return arrays[record.trajectory_id]

    result = ft.prepare_finetuning(
        split=split,
        grid=grid(),
        features=("position", "constant"),
        settings=settings(),
        training=training(),
        loader=loader,
        task="official-validation",
        code_commit="a" * 40,
        lockfile_hash="sha256:" + "b" * 64,
        hardware="synthetic",
    )
    return ft.FineTuneManifest.model_validate(result.model_dump() | changes)


def checkpoint(
    path: Path, spec: ft.FineTuneManifest, step: int = 4
) -> ft.TrainingCheckpoint:
    path.mkdir(parents=True, exist_ok=True)
    names = (
        "config.json",
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
    )
    for name in names:
        (path / name).write_text(json.dumps({"global_step": step}))
    files = {
        name: "sha256:" + hashlib.sha256((path / name).read_bytes()).hexdigest()
        for name in names
    }
    result = ft.TrainingCheckpoint(
        manifest=spec,
        step=step,
        files=files,
        dataset_hash=spec.dataset_hash,
        feature_set_hash=spec.feature_set_hash,
        config_hash=metadata_hash(spec.training),
    )
    write_metadata(path / "provenance.json", result)
    return result


def backend(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    torch, pipeline = install_backend(monkeypatch)
    torch.distributed.is_initialized.return_value = False
    pipeline.model_output_patch_size = 16
    modules: dict[str, Any] = {
        "chronos.chronos2.dataset": MagicMock(),
        "chronos.chronos2.trainer": MagicMock(),
        "transformers.trainer_callback": SimpleNamespace(TrainerCallback=object),
        "transformers": SimpleNamespace(
            TrainingArguments=lambda **kwargs: SimpleNamespace(
                device="cuda:0", **kwargs
            )
        ),
    }
    original = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, package=None: (
            modules[name] if name in modules else original(name, package)
        ),
    )
    return torch, modules["chronos.chronos2.trainer"].Chronos2Trainer


@pytest.mark.parametrize(
    "changes",
    [
        {"num_steps": 0},
        {"checkpoint_steps": 3},
        {"warmup_steps": 8},
        {"save_total_limit": 1},
        {"precision": "float16"},
        {"adam_beta1": 1},
        {"adam_epsilon": 0},
        {"learning_rate": float("nan")},
    ],
)
def test_invalid_config(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        training(**changes)


def test_inputs_are_exact_windows_and_holdout_is_explicit() -> None:
    frozen = manifest()
    arrays = series(frozen.split)
    inputs, validation, digest = ft._inputs(frozen, lambda r: arrays[r.trajectory_id])
    assert digest == frozen.input_hash
    assert len(inputs) == 15 and len(validation) == 10
    np.testing.assert_array_equal(inputs[0][0], np.arange(8))
    only_train = Registry(
        dataset=frozen.spec.dataset,
        trajectories=tuple(
            r for r in frozen.split.registry.trajectories if r.split == Split.TRAIN
        ),
    )
    split = build_split(
        only_train, SplitConfig(mode="grouped", seed=42, ratios=(2 / 3, 1 / 3, 0))
    )
    changed = manifest(
        split=split,
        spec=frozen.spec.model_copy(update={"split_hash": metadata_hash(split)}),
        task="development-train-holdout",
    )
    assert all(r.split == Split.TRAIN for r in changed.split.registry.trajectories)
    with pytest.raises(ForecastError, match="official TRAIN"):
        ft._check_original_train(frozen.split)
    with pytest.raises(ForecastError, match="grouped"):
        ft._check_holdout(frozen.split)
    with pytest.raises(ForecastError, match="official split"):
        ft._check_partitions(split, "official-validation")


@pytest.mark.parametrize("field,value", [("max_windows", 1), ("max_input_bytes", 1)])
def test_budgets_fail_before_reading(field: str, value: int) -> None:
    frozen = manifest(training=training(**{field: value}))
    loader = MagicMock(side_effect=AssertionError("payload must not be read"))
    with pytest.raises(ForecastError, match="budget"):
        ft._inputs(frozen, loader)
    loader.assert_not_called()


def test_checkpoint_integrity_resume_and_local_inference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, factory = backend(monkeypatch)
    frozen = manifest()
    run = tmp_path / "run"
    ft._resume(frozen, run, None)
    saved = checkpoint(run / "checkpoint-4", frozen)
    assert ft.read_checkpoint(run / "checkpoint-4") == saved
    ft._resume(frozen, run, run / "checkpoint-4")
    with pytest.raises(ForecastError, match="already exists"):
        ft._resume(frozen, run, None)
    with pytest.raises(ForecastError, match="same run"):
        ft._resume(frozen, tmp_path / "other", run / "checkpoint-4")
    with pytest.raises(ForecastError, match="provenance"):
        ft._resume(manifest(code_commit="c" * 40), run, run / "checkpoint-4")
    adapter = ft.FineTunedChronos2Adapter(
        settings(), checkpoint_dir=run / "checkpoint-4"
    )
    arrays = series(frozen.split)
    batch, _ = next(
        iter_forecasts(
            frozen.split,
            grid(),
            Split.VALIDATION,
            frozen.spec.feature_ids,
            lambda r: arrays[r.trajectory_id],
            batch_size=2,
        )
    )
    assert adapter.forecast(batch).values.shape == (2, 2, 2, 3)
    assert adapter.config.adapter_config_hash == adapter.artifact_hash
    args = ft._training_args(frozen, run)
    assert args["save_only_model"] is False and args["ignore_data_skip"] is True
    assert args["metric_for_best_model"] == "eval_loss"
    factory.assert_not_called()
    (run / "checkpoint-4" / "optimizer.pt").write_text("corrupt")
    with pytest.raises(ForecastError, match="checksum"):
        ft.read_checkpoint(run / "checkpoint-4")
    with pytest.raises(DataContractError):
        ft.read_checkpoint(tmp_path / "missing")


def test_upstream_train_pause_resume_and_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch, factory = backend(monkeypatch)
    frozen = manifest()
    arrays = series(frozen.split)
    run = tmp_path / "run"
    trainer = factory.return_value
    trainer.state.global_step = 4

    def train(**kwargs: Any) -> None:
        assert "resume_from_checkpoint" in kwargs
        cfg = factory.call_args.kwargs
        args = cfg["args"]
        if kwargs["resume_from_checkpoint"]:
            trainer.state.global_step = 8
        path = run / f"checkpoint-{trainer.state.global_step}"
        checkpoint(path, frozen, trainer.state.global_step)
        control = SimpleNamespace(should_training_stop=False)
        cfg["callbacks"][0].on_save(args, trainer.state, control)
        trainer.state.best_model_checkpoint = str(path)
        trainer.state.best_metric = 0.25

    trainer.train.side_effect = train
    assert (
        ft.train_chronos(
            frozen,
            lambda r: arrays[r.trajectory_id],
            output_dir=run,
            cache_dir=tmp_path,
            stop_after_checkpoint=4,
        )
        is None
    )
    assert not (run / "result.json").exists()
    result = ft.train_chronos(
        frozen,
        lambda r: arrays[r.trajectory_id],
        output_dir=run,
        cache_dir=tmp_path,
        resume_from_checkpoint=run / "checkpoint-4",
    )
    assert result and result.selected_step == 8 and result.completed_steps == 8
    assert result.runtime.peak_reserved_bytes == 200
    assert read_metadata(run / "result.json", ft.FineTuneResult) == result
    kwargs = factory.call_args.kwargs
    assert (
        kwargs["model"]
        is importlib.import_module(
            "chronos"
        ).Chronos2Pipeline.from_pretrained.return_value.model
    )
    ds = importlib.import_module("chronos.chronos2.dataset").Chronos2Dataset
    assert ds.call_args.kwargs["min_past"] == 6
    torch.cuda.reset_peak_memory_stats.assert_called_with("cuda:0")
    with pytest.raises(ForecastError, match="terminal"):
        ft._resume(frozen, run, run / "checkpoint-8")


@pytest.mark.parametrize(
    "failure",
    ["multi-gpu", "distributed", "bf16", "vram", "backend", "input", "stop", "native"],
)
def test_training_failures_are_explicit(
    failure: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch, factory = backend(monkeypatch)
    frozen = manifest()
    arrays = series(frozen.split)
    if failure == "multi-gpu":
        torch.cuda.device_count.return_value = 2
    elif failure == "distributed":
        torch.distributed.is_initialized.return_value = True
    elif failure == "bf16":
        torch.cuda.is_bf16_supported.return_value = False
    elif failure == "vram":
        torch.cuda.max_memory_reserved.return_value = 30 * 1024**3
    elif failure == "backend":
        factory.return_value.train.side_effect = RuntimeError("OOM")
    elif failure == "input":
        frozen = manifest(input_hash="sha256:" + "0" * 64)
    elif failure == "native":
        importlib.import_module(
            "chronos"
        ).Chronos2Pipeline.from_pretrained.return_value.model_context_length = 1
    with pytest.raises(ForecastError):
        ft.train_chronos(
            frozen,
            lambda r: arrays[r.trajectory_id],
            output_dir=tmp_path / "run",
            cache_dir=tmp_path,
            stop_after_checkpoint=3 if failure == "stop" else None,
        )


def test_manifest_and_checkpoint_reject_forged_protocol(tmp_path: Path) -> None:
    frozen = manifest()
    invalid_manifests: list[dict[str, object]] = [
        {"spec": frozen.spec.model_copy(update={"split_hash": "sha256:" + "f" * 64})},
        {"spec": frozen.spec.model_copy(update={"context_frames": 1})},
        {"training": training(batch_size=3)},
    ]
    for changes in invalid_manifests:
        with pytest.raises(ValidationError):
            ft.FineTuneManifest.model_validate(frozen.model_dump() | changes)
    with pytest.raises(ForecastError, match="frame-grid"):
        ft._check_grid(
            WindowConfig(contexts=(1.0, 2.0), horizons=(1.0,), stride_frames=1)
        )
    no_val = build_split(
        Registry(
            dataset=frozen.spec.dataset,
            trajectories=tuple(
                r for r in frozen.split.registry.trajectories if r.split == Split.TRAIN
            ),
        ),
        SplitConfig(mode="official", seed=42),
    )
    with pytest.raises(ForecastError, match="TRAIN and VAL"):
        ft._check_partitions(no_val, "official-validation")
    saved = checkpoint(tmp_path / "checkpoint", frozen)
    invalid_checkpoints: list[dict[str, object]] = [
        {"files": {}},
        {"files": saved.files | {"../escape": "sha256:" + "0" * 64}},
        {"step": 3},
        {"dataset_hash": "sha256:" + "0" * 64},
    ]
    for changes in invalid_checkpoints:
        with pytest.raises(ValidationError):
            ft.TrainingCheckpoint.model_validate(saved.model_dump() | changes)
    with pytest.raises(ForecastError, match="window"):
        ft._check_input_budget(frozen, [1, 0])


def test_local_model_rejects_wrong_revision_and_protocol(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend(monkeypatch)
    frozen = manifest()
    checkpoint(tmp_path / "checkpoint", frozen)
    with pytest.raises(ForecastError, match="revision"):
        ft.FineTunedChronos2Adapter(
            settings(revision="b" * 40), checkpoint_dir=tmp_path / "checkpoint"
        )
    adapter = ft.FineTunedChronos2Adapter(
        settings(), checkpoint_dir=tmp_path / "checkpoint"
    )
    arrays = series(frozen.split)
    batch, _ = next(
        iter_forecasts(
            frozen.split,
            grid(),
            Split.VALIDATION,
            frozen.spec.feature_ids,
            lambda r: arrays[r.trajectory_id],
            batch_size=1,
        )
    )
    wrong = ForecastBatch(
        batch.spec.model_copy(update={"feature_ids": ("position",)}),
        batch.indices,
        batch.context[:, :, :1],
    )
    with pytest.raises(ForecastError, match="source/features"):
        adapter.forecast(wrong)
    factory = importlib.import_module("chronos.chronos2.trainer").Chronos2Trainer
    factory.return_value.state.best_model_checkpoint = str(tmp_path / "checkpoint")
    factory.return_value.state.best_metric = float("nan")
    with pytest.raises(ForecastError, match="finite"):
        ft._result(
            frozen,
            factory.return_value,
            RuntimeStats(seconds=1, peak_allocated_bytes=0, peak_reserved_bytes=0),
            None,
        )

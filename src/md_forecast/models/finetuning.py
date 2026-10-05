"""Bounded full fine-tuning with the upstream Chronos dataset and HF Trainer."""

import hashlib
import importlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Literal, Self

import numpy as np
from pydantic import Field, model_validator

from md_forecast.core.constants import ModelId, Split, TimeUnit
from md_forecast.core.exceptions import ForecastError
from md_forecast.data.artifacts import metadata_hash, read_metadata, write_metadata
from md_forecast.data.forecast import ForecastBatch, ForecastSpec, iter_forecasts
from md_forecast.data.predictions import QuantileForecast, RuntimeStats
from md_forecast.data.preprocessing import SeriesLoader
from md_forecast.data.schemas import ArtifactHash, BoundaryModel
from md_forecast.data.series import FloatArray
from md_forecast.data.splits import SplitManifest
from md_forecast.data.windows import WindowConfig, iter_windows
from md_forecast.models.base import ModelConfig
from md_forecast.models.chronos import Chronos2Adapter, ChronosConfig


class FineTuneConfig(BoundaryModel):
    """All optimizer/schedule/resource choices are frozen before training."""

    num_steps: Annotated[int, Field(strict=True, gt=0)]
    batch_size: Annotated[int, Field(strict=True, gt=0)]
    learning_rate: Annotated[float, Field(gt=0)]
    weight_decay: Annotated[float, Field(ge=0)]
    adam_beta1: Annotated[float, Field(ge=0, lt=1)]
    adam_beta2: Annotated[float, Field(ge=0, lt=1)]
    adam_epsilon: Annotated[float, Field(gt=0)]
    max_grad_norm: Annotated[float, Field(gt=0)]
    warmup_steps: Annotated[int, Field(strict=True, ge=0)]
    gradient_accumulation_steps: Annotated[int, Field(strict=True, gt=0)]
    precision: Literal["float32", "bfloat16"]
    full_determinism: bool
    checkpoint_steps: Annotated[int, Field(strict=True, gt=0)]
    save_total_limit: Annotated[int, Field(strict=True, ge=2)]
    max_windows: Annotated[int, Field(strict=True, gt=0)]
    max_input_bytes: Annotated[int, Field(strict=True, gt=0)]

    @model_validator(mode="after")
    def check_schedule(self) -> Self:
        """Every terminal step must have a selectable/resumable checkpoint."""
        if (
            self.num_steps % self.checkpoint_steps
            or self.warmup_steps >= self.num_steps
        ):
            raise ValueError(
                "checkpoint interval must divide steps; warmup must be shorter"
            )
        return self


class FineTuneManifest(BoundaryModel):
    """Complete immutable training inputs; original source split labels survive."""

    task: Literal["development-train-holdout", "official-validation"]
    spec: ForecastSpec
    split: SplitManifest
    grid: WindowConfig
    settings: ChronosConfig
    training: FineTuneConfig
    input_hash: ArtifactHash
    code_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    lockfile_hash: ArtifactHash
    hardware: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def check_protocol(self) -> Self:
        """Bind the cell/source and reject missing VAL or unofficial TEST reuse."""
        expected = (
            self.split.registry.dataset,
            metadata_hash(self.split),
            metadata_hash(self.grid),
        )
        actual = (self.spec.dataset, self.spec.split_hash, self.spec.window_config_hash)
        if actual != expected:
            raise ValueError("fine-tuning source/split/grid hashes differ")
        _check_grid(self.grid)
        if (self.spec.context_frames, self.spec.horizon_frames) != (
            int(self.grid.contexts[0]),
            int(self.grid.horizons[0]),
        ):
            raise ValueError("fine-tuning dimensions differ from the declared grid")
        _check_partitions(self.split, self.task)
        if self.training.batch_size % len(self.spec.feature_ids):
            raise ValueError(
                "scalar training batch must fit whole multivariate windows"
            )
        return self

    @property
    def dataset_hash(self) -> str:
        """Include source records, checksums, features, units and exclusions."""
        return metadata_hash(self.split.registry)

    @property
    def feature_set_hash(self) -> str:
        """Bind feature order and definitions, not just a feature-set name."""
        features = {f.feature_id: f for f in self.spec.dataset.features}
        encoded = json.dumps(
            {
                "feature_set_version": self.spec.dataset.feature_set_version,
                "features": [
                    features[name].model_dump(mode="json")
                    for name in self.spec.feature_ids
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


class TrainingCheckpoint(BoundaryModel):
    """File integrity, optimizer/RNG state and complete scientific provenance."""

    manifest: FineTuneManifest
    step: Annotated[int, Field(strict=True, gt=0)]
    files: dict[str, ArtifactHash]
    dataset_hash: ArtifactHash
    feature_set_hash: ArtifactHash
    config_hash: ArtifactHash

    @model_validator(mode="after")
    def check_files(self) -> Self:
        """Reject unsafe paths, incomplete optimizer state and invalid step numbers."""
        required = {
            "config.json",
            "model.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "trainer_state.json",
        }
        if not required <= self.files.keys():
            raise ValueError(
                "checkpoint is missing model/optimizer/scheduler/RNG/trainer state"
            )
        _check_checkpoint_names(self.files)
        _check_checkpoint_step(self)
        expected = (
            self.manifest.dataset_hash,
            self.manifest.feature_set_hash,
            metadata_hash(self.manifest.training),
        )
        if (self.dataset_hash, self.feature_set_hash, self.config_hash) != expected:
            raise ValueError("checkpoint dataset/features/config hashes differ")
        return self


def _check_checkpoint_names(files: dict[str, str]) -> None:
    if any(Path(name).name != name or name in {"", ".", ".."} for name in files):
        raise ValueError("checkpoint filenames must be local basenames")


def _check_checkpoint_step(checkpoint: TrainingCheckpoint) -> None:
    cfg = checkpoint.manifest.training
    if checkpoint.step > cfg.num_steps or checkpoint.step % cfg.checkpoint_steps:
        raise ValueError("checkpoint step differs from the frozen schedule")


class FineTuneResult(BoundaryModel):
    """Operational telemetry is separate from checkpoint scientific identity."""

    manifest_hash: ArtifactHash
    selected_checkpoint_hash: ArtifactHash
    selected_step: Annotated[int, Field(strict=True, gt=0)]
    completed_steps: Annotated[int, Field(strict=True, gt=0)]
    validation_loss: float
    runtime: RuntimeStats
    optimizer_steps_per_second: Annotated[float, Field(gt=0)]


class TrainingExposure(BoundaryModel):
    """Actual forward-consumed training input counts, in frozen window order.

    Prepared/prefetched batches are excluded until their forward succeeds.
    Counts survive checkpoint resume; old checkpoints explicitly leave earlier
    exposure unknown. Optimizer updates remain separate trainer telemetry.
    """

    input_hash: ArtifactHash
    window_counts: tuple[Annotated[int, Field(strict=True, ge=0)], ...]
    forward_batches: Annotated[int, Field(strict=True, ge=0)]
    earlier_unknown_optimizer_steps: Annotated[int, Field(strict=True, ge=0)] = 0


class _ExposureCounter:
    def __init__(
        self, manifest: FineTuneManifest, windows: int, resume: Path | None
    ) -> None:
        self.input_hash = manifest.input_hash
        self.counts = [0] * windows
        self.calls = 0
        self.unknown_steps = 0 if resume is None else read_checkpoint(resume).step
        if resume is not None and (resume / "exposure.json").exists():
            self._restore(read_metadata(resume / "exposure.json", TrainingExposure))

    def _restore(self, saved: TrainingExposure) -> None:
        if saved.input_hash != self.input_hash or len(saved.window_counts) != len(
            self.counts
        ):
            raise ForecastError("training exposure differs from frozen inputs")
        self.counts = list(saved.window_counts)
        self.calls = saved.forward_batches
        self.unknown_steps = saved.earlier_unknown_optimizer_steps

    def record(self, indices: list[int]) -> None:
        for index in indices:
            self.counts[index] += 1
        self.calls += 1

    def snapshot(self) -> TrainingExposure:
        return TrainingExposure(
            input_hash=self.input_hash,
            window_counts=tuple(self.counts),
            forward_batches=self.calls,
            earlier_unknown_optimizer_steps=self.unknown_steps,
        )


def _track_exposure(dataset: Any, trainer: Any, counter: _ExposureCounter) -> None:
    """Observe pinned upstream sampling without altering its RNG or input tensors."""
    build, compute = dataset._build_batch, trainer.compute_loss

    def build_batch(indices: list[int]) -> Any:
        batch = build(indices)
        batch["md_forecast_exposure_indices"] = list(indices)
        return batch

    def compute_loss(model: Any, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        indices = inputs.pop("md_forecast_exposure_indices", None)
        result = compute(model, inputs, *args, **kwargs)
        if indices is not None:
            counter.record(indices)
        return result

    dataset._build_batch = build_batch
    trainer.compute_loss = compute_loss


def _check_grid(grid: WindowConfig) -> None:
    if (grid.unit, len(grid.contexts), len(grid.horizons)) != (TimeUnit.FRAME, 1, 1):
        raise ForecastError("fine-tune one explicit frame-grid cell at a time")


def _check_partitions(split: SplitManifest, task: str) -> None:
    partitions = {entry.split for entry in split.assignments}
    if not {Split.TRAIN, Split.VALIDATION} <= partitions:
        raise ForecastError(
            "fine-tuning requires separate nonempty TRAIN and VAL groups"
        )
    if task == "development-train-holdout":
        _check_holdout(split)
    elif split.config.mode != "official":
        raise ForecastError("official-validation requires the official split")


def _check_holdout(split: SplitManifest) -> None:
    partitions = {entry.split for entry in split.assignments}
    if split.config.mode != "grouped" or Split.TEST in partitions:
        raise ForecastError("development holdout requires grouped TRAIN/VAL only")
    _check_original_train(split)


def _check_original_train(split: SplitManifest) -> None:
    if any(record.split != Split.TRAIN for record in split.registry.trajectories):
        raise ForecastError(
            "development holdout accepts only originally official TRAIN records"
        )


def _inputs(
    manifest: FineTuneManifest, loader: SeriesLoader
) -> tuple[list[FloatArray], list[FloatArray], str]:
    counts = [
        sum(1 for _ in iter_windows(manifest.split, manifest.grid, part))
        for part in (Split.TRAIN, Split.VALIDATION)
    ]
    _check_input_budget(manifest, counts)
    digest = hashlib.sha256()
    inputs: list[list[FloatArray]] = []
    for partition in (Split.TRAIN, Split.VALIDATION):
        windows = _partition_inputs(manifest, loader, partition)
        for values in windows:
            digest.update(values.astype("<f8").tobytes())
        inputs.append(windows)
    return inputs[0], inputs[1], "sha256:" + digest.hexdigest()


def _partition_inputs(
    manifest: FineTuneManifest, loader: SeriesLoader, partition: Split
) -> list[FloatArray]:
    windows = []
    for batch, targets in iter_forecasts(
        manifest.split,
        manifest.grid,
        partition,
        manifest.spec.feature_ids,
        loader,
        batch_size=1,
        with_targets=True,
    ):
        assert targets is not None
        windows.append(np.concatenate((batch.context, targets), axis=1)[0].T.copy())
    return windows


def _check_input_budget(manifest: FineTuneManifest, counts: list[int]) -> None:
    if not all(counts) or sum(counts) > manifest.training.max_windows:
        raise ForecastError(
            "empty training/validation windows or window budget exceeded"
        )
    # ponytail: bounded eager inputs; use lazy prepared inputs for larger corpora.
    estimated = (
        sum(counts)
        * len(manifest.spec.feature_ids)
        * (manifest.spec.context_frames + manifest.spec.horizon_frames)
        * 16
    )
    if estimated > manifest.training.max_input_bytes:
        raise ForecastError("fine-tuning input array budget exceeded")


def prepare_finetuning(
    *,
    split: SplitManifest,
    grid: WindowConfig,
    features: tuple[str, ...],
    settings: ChronosConfig,
    training: FineTuneConfig,
    loader: SeriesLoader,
    task: Literal["development-train-holdout", "official-validation"],
    code_commit: str,
    lockfile_hash: str,
    hardware: str,
) -> FineTuneManifest:
    """Freeze and hash actual TRAIN/VAL values; never open TEST through this API."""
    _check_grid(grid)
    manifest = FineTuneManifest(
        task=task,
        spec=ForecastSpec(
            dataset=split.registry.dataset,
            feature_ids=features,
            context_frames=int(grid.contexts[0]),
            horizon_frames=int(grid.horizons[0]),
            split_hash=metadata_hash(split),
            window_config_hash=metadata_hash(grid),
        ),
        split=split,
        grid=grid,
        settings=settings,
        training=training,
        input_hash="sha256:" + "0" * 64,
        code_commit=code_commit,
        lockfile_hash=lockfile_hash,
        hardware=hardware,
    )
    _, _, digest = _inputs(manifest, loader)
    return FineTuneManifest.model_validate(
        manifest.model_dump() | {"input_hash": digest}
    )


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def read_checkpoint(path: Path) -> TrainingCheckpoint:
    """Verify all local state before inference or explicitly trusted resume."""
    checkpoint = read_metadata(path / "provenance.json", TrainingCheckpoint)
    try:
        for name, digest in checkpoint.files.items():
            _verify_file(path / name, digest)
        state = json.loads((path / "trainer_state.json").read_text())
        if state["global_step"] != checkpoint.step:
            raise ForecastError("checkpoint trainer step differs from provenance")
    except (OSError, ValueError, KeyError) as error:
        raise ForecastError(f"cannot verify checkpoint: {error}") from error
    return checkpoint


def _verify_file(path: Path, digest: str) -> None:
    if path.is_symlink() or _file_hash(path) != digest:
        raise ForecastError("checkpoint file checksum mismatch")


class FineTunedChronos2Adapter(Chronos2Adapter):
    """Load verified local safetensors using the unchanged forecast implementation."""

    def __init__(self, settings: ChronosConfig, *, checkpoint_dir: Path) -> None:
        """Verify provenance/state before loading local model weights."""
        self.checkpoint = read_checkpoint(checkpoint_dir)
        trained = self.checkpoint.manifest.settings
        if (settings.model_name, settings.revision) != (
            trained.model_name,
            trained.revision,
        ):
            raise ForecastError(
                "fine-tuned checkpoint differs from the base model/revision"
            )
        super().__init__(settings, cache_dir=checkpoint_dir, _snapshot=checkpoint_dir)
        self.config = ModelConfig(
            model_id=ModelId.CHRONOS2,
            seed=settings.seed,
            adapter_config_hash=self.artifact_hash,
        )

    @property
    def artifact_hash(self) -> str:
        """Checkpoint weights/state and inference settings both affect identity."""
        return checkpoint_artifact_hash(self.checkpoint, self.settings)

    def forecast(self, batch: ForecastBatch) -> QuantileForecast:
        """Reject a different scientific feature/split protocol before inference."""
        trained = self.checkpoint.manifest.spec
        expected = (trained.dataset, trained.feature_ids, trained.split_hash)
        if (
            batch.spec.dataset,
            batch.spec.feature_ids,
            batch.spec.split_hash,
        ) != expected:
            raise ForecastError(
                "fine-tuned forecast differs from trained source/features/split"
            )
        return super().forecast(batch)


def _training_args(manifest: FineTuneManifest, output_dir: Path) -> dict[str, object]:
    cfg, settings = manifest.training, manifest.settings
    return dict(
        output_dir=str(output_dir),
        max_steps=cfg.num_steps,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        adam_beta1=cfg.adam_beta1,
        adam_beta2=cfg.adam_beta2,
        adam_epsilon=cfg.adam_epsilon,
        max_grad_norm=cfg.max_grad_norm,
        warmup_steps=cfg.warmup_steps,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        bf16=cfg.precision == "bfloat16",
        tf32=False,
        seed=settings.seed,
        data_seed=settings.seed,
        full_determinism=cfg.full_determinism,
        dataloader_num_workers=0,
        report_to="none",
        disable_tqdm=True,
        save_only_model=False,
        save_total_limit=cfg.save_total_limit,
        save_strategy="steps",
        eval_strategy="steps",
        logging_strategy="steps",
        save_steps=cfg.checkpoint_steps,
        eval_steps=cfg.checkpoint_steps,
        logging_steps=cfg.checkpoint_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        label_names=["future_target"],
        prediction_loss_only=True,
        # Infinite sampling resumes saved RNG, without replaying consumed batches.
        ignore_data_skip=True,
    )


def _callback(
    manifest: FineTuneManifest,
    torch: Any,
    stop_after_checkpoint: int | None,
    exposure: _ExposureCounter | None = None,
) -> Any:
    backend = importlib.import_module("transformers.trainer_callback")
    callback_base: Any = backend.TrainerCallback

    class CheckpointProvenance(callback_base):  # type: ignore[misc]
        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if exposure is not None:
                write_metadata(path / "exposure.json", exposure.snapshot())
            files = {
                file.name: _file_hash(file)
                for file in path.iterdir()
                if file.is_file() and file.name != "provenance.json"
            }
            write_metadata(
                path / "provenance.json",
                TrainingCheckpoint(
                    manifest=manifest,
                    step=state.global_step,
                    files=files,
                    dataset_hash=manifest.dataset_hash,
                    feature_set_hash=manifest.feature_set_hash,
                    config_hash=metadata_hash(manifest.training),
                ),
            )
            _check_vram(torch, manifest.settings)
            if state.global_step == stop_after_checkpoint:
                control.should_training_stop = True
            return control

    return CheckpointProvenance()


def _check_vram(torch: Any, settings: ChronosConfig) -> None:
    if torch.cuda.max_memory_reserved(settings.device) > settings.max_vram_bytes:
        raise ForecastError("fine-tuning exceeded the configured VRAM budget")


def _resume(
    manifest: FineTuneManifest, output_dir: Path, checkpoint: Path | None
) -> None:
    if checkpoint is None:
        if output_dir.exists():
            raise ForecastError(
                "training output already exists; use explicit resume or a fresh path"
            )
        output_dir.mkdir(parents=True)
        write_metadata(output_dir / "manifest.json", manifest)
        return
    if checkpoint.resolve().parent != output_dir.resolve():
        raise ForecastError("resume checkpoint must belong to the same run directory")
    previous = read_checkpoint(checkpoint)
    _check_resume_manifest(manifest, output_dir, previous)
    _check_resume_status(manifest, output_dir, previous)


def _check_resume_status(
    manifest: FineTuneManifest, output_dir: Path, previous: TrainingCheckpoint
) -> None:
    if previous.step >= manifest.training.num_steps:
        raise ForecastError("cannot resume a terminal checkpoint")
    if (output_dir / "result.json").exists():
        raise ForecastError("run is already complete; refuse to rewind its checkpoints")


def _check_resume_manifest(
    manifest: FineTuneManifest, output_dir: Path, previous: TrainingCheckpoint
) -> None:
    saved = read_metadata(output_dir / "manifest.json", FineTuneManifest)
    if previous.manifest != manifest or saved != manifest:
        raise ForecastError("resume provenance differs from frozen inputs/config/code")


def train_chronos(
    manifest: FineTuneManifest,
    loader: SeriesLoader,
    *,
    output_dir: Path,
    cache_dir: Path,
    resume_from_checkpoint: Path | None = None,
    stop_after_checkpoint: int | None = None,
) -> FineTuneResult | None:
    """Use upstream training/selection/resume; None denotes an intentionally paused run.

    Resume is only for trusted locally generated optimizer/RNG files: their Torch
    deserialization is not an untrusted-checkpoint ingestion mechanism.
    """
    manifest = FineTuneManifest.model_validate_json(manifest.model_dump_json())
    _check_stop(manifest, stop_after_checkpoint)
    inputs, validation, digest = _inputs(manifest, loader)
    if digest != manifest.input_hash:
        raise ForecastError("training input values differ from the frozen hash")
    _resume(manifest, output_dir, resume_from_checkpoint)
    try:
        adapter = Chronos2Adapter(manifest.settings, cache_dir=cache_dir)
        _check_native(manifest, adapter)
        trainer, torch = _trainer(
            manifest,
            output_dir,
            adapter,
            inputs,
            validation,
            stop_after_checkpoint,
            resume_from_checkpoint,
        )
        torch.cuda.synchronize(manifest.settings.device)
        torch.cuda.reset_peak_memory_stats(manifest.settings.device)
        start = perf_counter()
        trainer.train(
            resume_from_checkpoint=str(resume_from_checkpoint)
            if resume_from_checkpoint
            else None
        )
        runtime = adapter._runtime(start)
        if trainer.state.global_step != manifest.training.num_steps:
            return None
        result = _result(manifest, trainer, runtime, resume_from_checkpoint)
        write_metadata(output_dir / "result.json", result)
        return result
    except (ImportError, OSError, ValueError, TypeError, RuntimeError) as error:
        raise ForecastError(f"Chronos-2 training failed: {error}") from error


def _check_stop(manifest: FineTuneManifest, step: int | None) -> None:
    if step is not None and (
        step <= 0
        or step >= manifest.training.num_steps
        or step % manifest.training.checkpoint_steps
    ):
        raise ForecastError("pause step must be an intermediate checkpoint")


def _check_native(manifest: FineTuneManifest, adapter: Chronos2Adapter) -> None:
    if (
        manifest.spec.context_frames > adapter._pipeline.model_context_length
        or manifest.spec.horizon_frames > adapter._pipeline.model_prediction_length
    ):
        raise ForecastError("training exceeds native Chronos-2 context/horizon limits")


def _trainer(
    manifest: FineTuneManifest,
    output_dir: Path,
    adapter: Chronos2Adapter,
    inputs: list[FloatArray],
    validation: list[FloatArray],
    stop: int | None,
    resume: Path | None = None,
) -> tuple[Any, Any]:
    torch = adapter._torch
    dataset = importlib.import_module("chronos.chronos2.dataset")
    backend = importlib.import_module("chronos.chronos2.trainer")
    transformers = importlib.import_module("transformers")
    _check_training_device(torch, manifest)
    args = transformers.TrainingArguments(**_training_args(manifest, output_dir))
    if str(args.device) != manifest.settings.device:
        raise ForecastError(
            "Trainer device differs from the explicitly selected CUDA device"
        )
    common = dict(
        context_length=manifest.spec.context_frames,
        prediction_length=manifest.spec.horizon_frames,
        batch_size=manifest.training.batch_size,
        output_patch_size=adapter._pipeline.model_output_patch_size,
        min_past=manifest.spec.context_frames,
    )
    exposure = _ExposureCounter(manifest, len(inputs), resume)
    training_dataset = dataset.Chronos2Dataset(
        inputs=inputs, mode=dataset.DatasetMode.TRAIN, **common
    )
    trainer = backend.Chronos2Trainer(
        model=adapter._pipeline.model,
        args=args,
        train_dataset=training_dataset,
        eval_dataset=dataset.Chronos2Dataset(
            inputs=validation, mode=dataset.DatasetMode.VALIDATION, **common
        ),
        callbacks=[_callback(manifest, torch, stop, exposure)],
    )
    _track_exposure(training_dataset, trainer, exposure)
    return trainer, torch


def _check_training_device(torch: Any, manifest: FineTuneManifest) -> None:
    if torch.cuda.device_count() != 1 or torch.distributed.is_initialized():
        raise ForecastError(
            "training requires exactly one visible GPU and no distributed process group"
        )
    if manifest.training.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ForecastError("bfloat16 is unavailable on the selected GPU")


def _result(
    manifest: FineTuneManifest, trainer: Any, runtime: RuntimeStats, resume: Path | None
) -> FineTuneResult:
    selected = read_checkpoint(Path(trainer.state.best_model_checkpoint))
    loss = float(trainer.state.best_metric)
    if not math.isfinite(loss):
        raise ForecastError("selected validation loss must be finite")
    initial = read_checkpoint(resume).step if resume else 0
    return FineTuneResult(
        manifest_hash=metadata_hash(manifest),
        selected_checkpoint_hash=metadata_hash(selected),
        selected_step=selected.step,
        completed_steps=trainer.state.global_step,
        validation_loss=loss,
        runtime=runtime,
        optimizer_steps_per_second=(trainer.state.global_step - initial)
        / runtime.seconds,
    )


def checkpoint_artifact_hash(
    checkpoint: TrainingCheckpoint, settings: ChronosConfig
) -> str:
    """Identify weights/config, training provenance, step and inference settings."""
    combined = hashlib.sha256(
        (
            metadata_hash(checkpoint.manifest)
            + checkpoint.files["model.safetensors"]
            + checkpoint.files["config.json"]
            + str(checkpoint.step)
            + metadata_hash(settings)
        ).encode()
    ).hexdigest()
    return "sha256:" + combined

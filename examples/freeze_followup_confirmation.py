"""Pin every selected model and protocol before any reserved trajectory is decoded."""

import argparse
import json
import subprocess
from pathlib import Path

from finetune_followup_geometry import AdaptationConfig
from run_followup_probability import ProbabilityConfig, check_file

from md_forecast.data.artifacts import metadata_hash, read_metadata, write_metadata
from md_forecast.data.cohort import CohortManifest, ExternalReserve
from md_forecast.data.registry import read_registry
from md_forecast.evaluation.confirmation import ConfirmationPlan
from md_forecast.evaluation.overview import file_hash
from md_forecast.features.structural import BASE_FEATURE_IDS
from md_forecast.models.finetuning import FineTuneResult, read_checkpoint
from md_forecast.models.learned import NLinearState
from md_forecast.models.residuals import ResidualState


def freeze(output: Path) -> None:
    """Revalidate saved development choices; inspect no reserved outcome values."""
    if output.exists():
        raise ValueError("confirmation plan exists; preserve frozen choices")
    probability = ProbabilityConfig.model_validate_json(
        Path("configs/experiments/followup-probability.json").read_text()
    )
    baselines = Path("data/processed/followup-baselines40-expanded")
    inputs = Path("data/processed/followup-inputs42")
    adaptation = Path("data/processed/followup-finetuning41-verified")
    choice_path = Path("data/reports/followup-inputs42-validation/selection.json")
    check_file(choice_path, probability.input_selection_sha256)
    check_file(baselines / "provenance.json", probability.baseline_provenance_sha256)
    choice = json.loads(choice_path.read_text())
    baseline = json.loads((baselines / "provenance.json").read_text())
    probability_root = Path("data/reports/followup-probability43-validation")
    saved = json.loads((probability_root / "summary.json").read_text())
    for relative, digest in saved["file_sha256"].items():
        check_file(probability_root / relative, "sha256:" + digest)
    settings = read_metadata(adaptation / "plan.json", AdaptationConfig).settings
    files = {
        str(p): file_hash(p)
        for p in [
            choice_path,
            baselines / "provenance.json",
            probability_root / "summary.json",
            Path("uv.lock"),
            Path("configs/experiments/followup-probability.json"),
            Path("configs/features/common-geometry.yaml"),
            Path("configs/datasets/followup-admission.json"),
            Path("configs/datasets/followup-external-reserve80.json"),
            Path("configs/datasets/followup-external-reserve.json"),
            Path("configs/datasets/followup-external80-metadata.json"),
            Path("data/manifests/followup-cohort.json"),
        ]
    }
    fine = {}
    conditions = {}
    for target in BASE_FEATURE_IDS:
        selection = choice["selected_inputs"][target]
        condition = selection["condition"]
        seed = selection["representative_seed"]
        assert seed == 42
        conditions[target] = condition
        trial = (
            f"target{BASE_FEATURE_IDS.index(target)}-s{seed}"
            if condition == "target-only"
            else f"lr1e-05-s{seed}"
        )
        root = inputs if condition == "target-only" else adaptation
        run = root / "runs" / trial
        result = read_metadata(run / "result.json", FineTuneResult)
        path = run / f"checkpoint-{result.selected_step}"
        checkpoint = read_checkpoint(path)
        assert (
            metadata_hash(checkpoint)
            == choice["results"][target][f"fine-{seed}"][condition][
                "training_identity"
            ]["checkpoint_hash"]
        )
        fine[target] = str(path)
        files[str(path / "provenance.json")] = file_hash(path / "provenance.json")
        files.update(
            {str(path / name): digest for name, digest in checkpoint.files.items()}
        )
        files[str(run / "result.json")] = file_hash(run / "result.json")
    compact = {}
    residuals = {}
    statistics = {}
    for cell, selected in baseline["cells"].items():
        path = baselines / cell / "compact-state.json"
        state = read_metadata(path, NLinearState)
        assert metadata_hash(state) == selected["selected_compact_hash"]
        compact[cell] = str(path)
        files[str(path)] = file_hash(path)
        statistics[cell] = selected["best_statistical_per_feature"]
        residuals[cell] = {}
        for model_hash, state_hash in saved["residual_state_hashes"][cell].items():
            path = (
                probability_root / cell / (model_hash.removeprefix("sha256:") + ".json")
            )
            state = read_metadata(path, ResidualState)
            assert metadata_hash(state) == state_hash
            residuals[cell][model_hash] = str(path)
            files[str(path)] = file_hash(path)
    native = read_metadata(Path("data/manifests/followup-cohort.json"), CohortManifest)
    external = ExternalReserve.model_validate_json(
        Path("configs/datasets/followup-external-reserve80.json").read_text()
    )
    seen = ExternalReserve.model_validate_json(
        Path("configs/datasets/followup-external-reserve.json").read_text()
    )
    plan = ConfirmationPlan(
        grid=probability.grid,
        primary=probability.primary,
        inference=probability.confirmation,
        descriptive=probability.development,
        settings=settings,
        training_dataset=read_registry(
            Path("data/processed/followup-development/registry.json")
        ).dataset,
        native_cohort_hash=metadata_hash(native),
        external_reserve_hash=metadata_hash(external),
        seen_reserve_hash=metadata_hash(seen),
        probability_config_hash=metadata_hash(probability),
        input_conditions=conditions,
        fine_checkpoints=fine,
        compact_states=compact,
        residual_states=residuals,
        statistical_choices=statistics,
        source_files=files,
        expected_groups=probability.expected_groups,
        worthwhile_reduction=probability.worthwhile_reduction,
        coverage_tolerance=probability.coverage_tolerance,
        calibration_scope="TRAIN-residuals-only; no confirmation refitting",
        confirmation_state="no reserved outcomes decoded",
        code_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    print(write_metadata(output, plan))


def main() -> None:
    """Freeze once; create a new experiment for any subsequent choice change."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/followup-confirmation45.json"),
    )
    freeze(parser.parse_args().output)


if __name__ == "__main__":
    main()

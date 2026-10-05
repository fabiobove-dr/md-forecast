"""Hash-verified confirmation decisions and native-unit SVG comparisons."""

import json
from html import escape
from pathlib import Path
from typing import Any

from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash
from md_forecast.evaluation.confirmation import ConfirmationPlan
from md_forecast.evaluation.overview import OverviewConfig, file_hash

METHODS = ("persistence", "selected-statistic", "compact", "zero-shot", "fine-tuned")
COLORS = ("#6f8196", "#8d61b6", "#ba8524", "#2676ce", "#117b69")


def render_confirmation(config: OverviewConfig) -> str:
    """Display decisions from pinned saved inference, with verified source bindings."""
    payload = _load_confirmation(config)
    if payload is None:
        return ""
    plan = ConfirmationPlan.model_validate(payload["frozen_plan"])
    tasks = payload["tasks"]
    rows, plots, notes = [], [], []
    for task, result in tasks.items():
        primary = result["cells"][result["primary_cell"]]
        for feature, data in primary.items():
            unit = _unit(plan, feature)
            plots.append(_bars(task, feature, unit, data["methods"]))
            rows.extend(_decision_rows(task, feature, data))
        notes.append(_task_note(task, result, plan))
    evidence = escape(json.dumps(payload, indent=2, allow_nan=False))
    return (
        '<section class="card" id="confirmation"><h2>Independent confirmation</h2>'
        "<p>The primary decision uses C40/H10, frozen before reserved outcomes. "
        "A useful result requires corrected MAE gains of at least 10% against "
        "persistence and the selected statistic, improved quantile loss, "
        "80% coverage whose entire corrected interval lies in 75–85%, and "
        "interval width no greater than the statistic. The stronger result "
        "also requires the 10% gain against the compact model. "
        "Each condition must hold; this is separate from per-window tolerance.</p>"
        + "".join(notes)
        + "<p>Paired primary inference resamples entire independent complexes: "
        "60,000 draws, seed 42, Bonferroni family 126. Six seen systems cannot "
        "support this inference. Other cells and lead-level intervals remain "
        "descriptive. Native physical cadence is unknown; external H10 spans "
        "2,000 ps at the verified 200 ps spacing.</p>"
        '<div class="tablewrap"><table><thead><tr><th>Task</th><th>Observable</th>'
        "<th>Foundation model</th><th>Measurable MAE gain</th><th>≥10% MAE gain</th>"
        "<th>Coverage and corrected CI</th><th>Calibrated</th><th>Pinball gain</th>"
        "<th>Width acceptable</th><th>Useful</th><th>Stronger</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        "<h3>Primary errors in native units</h3><p>Lower MAE is better. "
        "Whiskers are descriptive 95% group intervals; decisions above use "
        "the separately corrected paired intervals. The axes differ by "
        "observable; counts, fractions and Å cannot be compared numerically.</p>"
        '<div class="grid">' + "".join(plots) + "</div>"
        "<details><summary>Complete corrected effects, failed criteria and "
        "all-grid inference evidence</summary><pre>" + evidence + "</pre></details>"
        "</section>"
    )


def _load_confirmation(config: OverviewConfig) -> dict[str, Any] | None:
    source = config.confirmation_summary
    if source is None:
        return None
    if file_hash(source.path) != source.expected_hash:
        raise DataContractError("confirmation synthesis checksum differs")
    if source.path.stat().st_size > config.max_input_bytes:
        raise DataContractError("confirmation synthesis exceeds input budget")
    payload: dict[str, Any] = json.loads(source.path.read_text())
    plan = ConfirmationPlan.model_validate(payload["frozen_plan"])
    _validate_synthesis(payload, plan)
    _bind_sources(config, payload)
    return payload


def _validate_synthesis(payload: dict[str, Any], plan: ConfirmationPlan) -> None:
    if payload["config_hash"] != metadata_hash(plan):
        raise DataContractError("confirmation synthesis plan identity differs")
    if payload["primary_family_size"] != plan.inference.comparison_family_size:
        raise DataContractError("confirmation comparison family differs")
    if set(payload["tasks"]) != {"native", "external", "replica"}:
        raise DataContractError("confirmation requires three separate tasks")


def _bind_sources(config: OverviewConfig, payload: dict[str, Any]) -> None:
    pinned = {
        str(source.root / "summary.json"): source.expected_hash
        for source in config.sources
        if source.kind == "probability"
    }
    for path, expected in payload["source_summaries"].items():
        _check_synthesis_source(path, expected, pinned)


def _check_synthesis_source(path: str, expected: str, pinned: dict[str, str]) -> None:
    if pinned.get(path) != expected or file_hash(Path(path)) != expected:
        raise DataContractError("confirmation synthesis source differs")


def _unit(plan: ConfirmationPlan, feature: str) -> str:
    return next(
        definition.unit.value
        for definition in plan.training_dataset.features
        if definition.feature_id == feature
    )


def _task_note(task: str, result: dict[str, Any], plan: ConfirmationPlan) -> str:
    deviations = (
        "; ".join(
            f"{item['source_system_id']}: {item['reason']}"
            for item in result["deviations"]
        )
        or "none"
    )
    return (
        '<p class="callout"><b>'
        + escape(task)
        + "</b>: "
        + escape(
            f"{result['independent_groups']} independent groups "
            f"(planned {plan.expected_groups[task]}). {result['decision']}. "
            f"QC exclusions without replacement: {deviations}. "
            f"Corrected inference: {result['corrected_interval_status']}."
        )
        + "</p>"
    )


def _flag(value: bool) -> str:
    return (
        '<td class="pass">Yes</td>'
        if value
        else '<td class="fail">Not established</td>'
    )


def _decision_rows(task: str, feature: str, data: dict[str, Any]) -> list[str]:
    rows = []
    for method, effects in data["corrected_effects"].items():
        decision, coverage = effects["decision"], effects["coverage"]
        ci = coverage["ci"]
        interval = (
            "unresolved" if ci[0] is None else f"{100 * ci[0]:.1f}–{100 * ci[1]:.1f}%"
        )
        row = "<tr>" + "".join(
            "<td>" + escape(label) + "</td>" for label in (task, feature, method)
        )
        row += _flag(decision["measurable_point_gain"]) + _flag(
            decision["worthwhile_point_gain"]
        )
        row += "<td>" + escape(f"{100 * coverage['mean']:.1f}%; {interval}") + "</td>"
        row += "".join(
            _flag(decision[key])
            for key in (
                "calibrated",
                "pinball_gain",
                "width_not_worse",
                "useful_minimum",
                "useful_stronger",
            )
        )
        rows.append(row + "</tr>")
    return rows


def _bars(task: str, feature: str, unit: str, methods: dict[str, Any]) -> str:
    highs = [
        max(result["metrics"]["mae"], result["marginal_ci"]["mae"][1] or 0)
        for result in methods.values()
    ]
    maximum = max(highs) * 1.12 or 1.0
    elements = []
    for index, (method, color) in enumerate(zip(METHODS, COLORS, strict=True)):
        result = methods[method]
        value = result["metrics"]["mae"]
        y, width = 40 + index * 42, 300 * value / maximum
        elements.append(
            f'<text x="5" y="{y + 16}" font-size="13">{escape(method)}</text>'
            f'<rect x="145" y="{y}" width="{width}" height="24" fill="{color}"/>'
            f'<text x="{150 + width}" y="{y + 17}" font-size="12">{value:.5g}</text>'
        )
        elements.append(_whisker(result["marginal_ci"]["mae"], y, maximum))
    title = escape(f"{task} · {feature} · MAE ({unit})")
    return (
        "<div><h4>"
        + title
        + '</h4><svg viewBox="0 0 540 270" role="img" aria-label="'
        + title
        + '">'
        + "".join(elements)
        + "</svg></div>"
    )


def _whisker(ci: tuple[float | None, float | None], y: int, maximum: float) -> str:
    lo, hi = ci
    if lo is None or hi is None:
        return ""
    return (
        f'<path d="M {145 + 300 * lo / maximum} {y + 12} '
        f'H {145 + 300 * hi / maximum}" '
        'stroke="#182a43" stroke-width="2"/>'
    )

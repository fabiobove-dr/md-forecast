"""Verify a generated report offline with optional pinned Playwright tooling."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from playwright.sync_api import Page, sync_playwright


def check_values(page: Page) -> None:
    """Compare displayed values to embedded matched data; exercise rating rules."""
    data: dict[str, Any] = page.evaluate(
        "DATA[Number(document.getElementById('panel').value)]"
    )
    selected = data["windows"][0]
    model = page.locator("#model").input_value()
    row = page.locator("#point-errors tr").first
    for index, value in ((1, selected["target"][0]), (2, selected["points"][model][0])):
        shown = float(row.locator("td").nth(index).inner_text().replace(",", ""))
        assert abs(shown - value) <= max(abs(value) * 0.00002, 1e-8)
    assert page.locator("#window option").count() == len(data["windows"])
    assert page.locator("#point-errors tr").count() == data["horizon_frames"]
    assert page.locator("#status").inner_text().startswith("UNRATED")
    page.fill("#tolerance", "100000000")
    assert page.locator("#status").inner_text().startswith("PASS")
    errors = [
        abs(p - t)
        for p, t in zip(selected["points"][model], selected["target"], strict=True)
    ]
    maximum = max(errors)
    if maximum > sum(errors) / len(errors):
        page.fill("#tolerance", str((maximum + sum(errors) / len(errors)) / 2))
        assert page.locator("#status").inner_text().startswith("PASS")
        page.select_option("#rule", "all")
        assert page.locator("#status").inner_text().startswith("FAIL")
    page.fill("#tolerance", "")
    assert page.locator("#status").inner_text().startswith("UNRATED")


def downloads(page: Page, destination: Path) -> None:
    """CSV rows and SVG artifacts must actually download and preserve values."""
    for button, name in (
        ("download-errors", "errors.csv"),
        ("download-metrics", "metrics.csv"),
        ("download-curve", "curve.svg"),
        ("download-plot", "errors.svg"),
    ):
        with page.expect_download() as event:
            page.click("#" + button)
        path = destination / name
        event.value.save_as(path)
        assert path.stat().st_size > 100
    data: dict[str, Any] = page.evaluate(
        "DATA[Number(document.getElementById('panel').value)]"
    )
    with (destination / "errors.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(data["windows"]) * data["horizon_frames"]
    assert float(rows[0]["actual"]) == data["windows"][0]["target"][0]
    assert rows[0]["point_assessment"] == "UNRATED"


def check_failed_quantiles(page: Page, destination: Path) -> int:
    """Explicitly failed distributions remain visible, never plausible shaded bands."""
    failed = page.evaluate("""() => {
      const result=[];
      DATA.forEach((p,i)=>p.windows.forEach((w,j)=>
        Object.entries(w.quantile_failures??{}).forEach(([m,leads])=>
          result.push({panel:i,window:j,model:m,leads:Object.keys(leads)}))));
      return result;
    }""")
    if failed:
        first = failed[0]
        page.select_option("#panel", str(first["panel"]))
        page.select_option("#window", str(first["window"]))
        page.select_option("#model", first["model"])
        assert (
            "INVALID: crossing quantiles" in page.locator("#point-errors").inner_text()
        )
        assert "uncertainty band is suppressed" in page.locator("#overlay").inner_text()
        with page.expect_download() as event:
            page.click("#download-errors")
        path = destination / "failed-quantiles.csv"
        event.value.save_as(path)
        with path.open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        invalid = [row for row in rows if row["quantile_status"] == "INVALID_CROSSING"]
        expected = sum(
            len(row["leads"])
            for row in failed
            if row["panel"] == first["panel"] and row["model"] == first["model"]
        )
        assert len(invalid) == expected
        assert all(row["raw_failed_quantiles"] for row in invalid)
    return sum(len(row["leads"]) for row in failed)


def main() -> None:
    """Read an HTML artifact only; no model or data-acquisition dependency."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chrome", default="/usr/bin/google-chrome")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    errors = []
    requests = []
    with sync_playwright() as tool:
        browser = tool.chromium.launch(
            executable_path=args.chrome,
            headless=True,
            args=[
                "--no-sandbox",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
            ],
        )
        context = browser.new_context(accept_downloads=True)
        context.set_offline(True)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "request",
            lambda request: (
                requests.append(request.url)
                if request.url.startswith(("http://", "https://"))
                else None
            ),
        )
        page.goto(args.report.resolve().as_uri())
        page.wait_for_function("document.querySelector('#scores').children.length > 0")
        check_values(page)
        downloads(page, args.output)
        for i in range(page.locator("#panel option").count()):
            page.select_option("#panel", str(i))
            assert page.locator("#scores tr").count() > 0
            assert page.locator("#point-errors tr").count() > 0
        crossed = check_failed_quantiles(page, args.output)
        if len(page.frames) > 1:
            frame = page.frames[1]
            frame.wait_for_function(
                "window.moleculeViewer !== undefined", timeout=60000
            )
            with page.expect_download() as event:
                frame.click("#save")
            event.value.save_as(args.output / "molecule.png")
            frame.click("#reset")
            frame.uncheck("#regions")
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(args.output / "mobile.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert not errors, errors
        assert not requests, requests
        panels = page.locator("#panel option").count()
        browser.close()
        fallback = tool.chromium.launch(
            executable_path=args.chrome,
            headless=True,
            args=["--no-sandbox", "--disable-webgl"],
        )
        context = fallback.new_context()
        context.add_init_script("""
        const original = HTMLCanvasElement.prototype.getContext;
        HTMLCanvasElement.prototype.getContext = function(kind, ...args) {
          if (/webgl/i.test(kind)) return null;
          return original.call(this, kind, ...args);
        };""")
        context.set_offline(True)
        page = context.new_page()
        page.goto(args.report.resolve().as_uri())
        if len(page.frames) > 1:
            frame = page.frames[1]
            frame.wait_for_function(
                "() => document.getElementById('status').textContent"
                ".includes('unavailable')"
            )
            assert frame.locator("#fallback").get_attribute("open") is not None
            assert frame.locator("#fallback circle").count() > 0
        fallback.close()
    (args.output / "browser-audit.json").write_text(
        json.dumps(
            {
                "panels": panels,
                "page_errors": errors,
                "remote_requests": requests,
                "offline": True,
                "mobile_width": 390,
                "downloads": "CSV, SVG; PNG when a structure is supplied",
                "webgl_fallback": True,
                "explicit_crossed_quantile_points": crossed,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

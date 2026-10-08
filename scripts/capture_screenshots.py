"""Capture the real select/train/evaluate workflow; no mocked model or results.

python scripts/capture_screenshots.py --browser-channel msedge
See docs/SCREENSHOTS.md for prerequisites and dataset scope.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from playwright.sync_api import sync_playwright
import torch
from werkzeug.serving import make_server

import app as application


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-channel", default=None, help="e.g. msedge; default: Playwright Chromium")
    parser.add_argument("--port", type=int, default=5051)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit("Run this command from the repository root.")
    manifest_path = ROOT / "chest_xray" / "demo-manifest.json"
    if not manifest_path.exists():
        raise SystemExit("First run python scripts/prepare_demo_dataset.py in a fresh checkout.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = ROOT / "docs" / "images"
    output.mkdir(parents=True, exist_ok=True)
    local = ROOT / ".local"
    local.mkdir(exist_ok=True)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    report = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "transformers", "flask", "playwright")},
        "seed": 42,
        "torch_threads": 4,
        "dataset": manifest,
        "screenshots": [],
    }
    original_create_model = application.create_model

    def recorded_create_model():
        processor, model = original_create_model()
        report["model"] = {
            "name": model.config._name_or_path,
            "revision": getattr(model.config, "_commit_hash", None),
            "device": str(next(model.parameters()).device),
        }
        return processor, model

    application.create_model = recorded_create_model
    server = make_server("127.0.0.1", args.port, application.app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{args.port}"
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel=args.browser_channel)
            report["browser"] = browser.version
            context = browser.new_context(viewport={"width": 1280, "height": 1000}, reduced_motion="reduce")
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            response = page.goto(base, wait_until="networkidle")
            assert response.status == 200
            page.wait_for_function("Array.from(document.images).every(i => i.complete && i.naturalWidth > 0)")
            assert page.locator("#normal-images .img-preview").count() == 16
            assert page.locator("#pneumonia-images .img-preview").count() == 16
            page.screenshot(path=str(output / "01-overview.png"))
            report["screenshots"].append("01-overview.png")

            selected = []
            for label in ("NORMAL", "PNEUMONIA"):
                images = [image for image in manifest["images"] if image["split"] == "train" and image["label"] == label][:4]
                for image in images:
                    filename = Path(image["destination"]).name
                    page.locator(f'#upload-form .img-preview[data-filename="{filename}"]').click()
                    selected.append(filename)
            assert page.locator("#normal-count").inner_text() == "4 images selected"
            assert page.locator("#pneumonia-count").inner_text() == "4 images selected"
            assert page.locator("#submit-btn").is_enabled()
            page.locator("#upload-form").screenshot(path=str(output / "02-training-selection.png"))
            report["screenshots"].append("02-training-selection.png")
            report["selected_training_images"] = selected
            print("Running actual 8-epoch fine-tuning and evaluation on 24 images...", flush=True)
            started = time.monotonic()
            page.locator("#submit-btn").click()
            page.wait_for_url("**/results/*", timeout=900_000, wait_until="networkidle")
            assert page.locator(".card-header").first.inner_text() == "Pneumonia Detection Results"
            report["workflow_seconds"] = round(time.monotonic() - started, 2)
            page.wait_for_function("Array.from(document.images).every(i => i.complete && i.naturalWidth > 0)")
            assert len(application.RESULTS_CACHE) == 1
            result = next(iter(application.RESULTS_CACHE.values()))
            assert len(result["normal_results"]) == 12
            assert len(result["pneumonia_results"]) == 12
            assert result["training_info"] == {"total_training_images": 8, "normal_count": 4, "pneumonia_count": 4}
            # Independently cross-check summary counts against per-image outcomes.
            matrix = result["confusion_matrix"]
            assert sum(matrix.values()) == 24
            assert matrix["true_negative"] == sum(r["correct"] for r in result["normal_results"])
            assert matrix["true_positive"] == sum(r["correct"] for r in result["pneumonia_results"])
            assert result["accuracy"] == (matrix["true_negative"] + matrix["true_positive"]) / 24
            report["results"] = result
            for tab, pane, count in (
                ("all", "all", 24), ("normal", "normal", 12),
                ("pneumonia", "pneumonia", 12),
                ("correct", "correct", matrix["true_negative"] + matrix["true_positive"]),
                ("incorrect", "incorrect", matrix["false_negative"] + matrix["false_positive"]),
            ):
                page.locator(f"#{tab}-tab").click()
                assert page.locator(f"#{pane}-results").is_visible()
                assert page.locator(f"#{pane}-results .result-card").count() == count
            page.locator("#all-tab").click()

            page.evaluate("window.scrollTo(0, 0)")
            page.screenshot(path=str(output / "03-evaluation-metrics.png"))
            page.locator(".card").filter(has=page.locator(".confusion-matrix-visual")).last.screenshot(path=str(output / "04-confusion-matrix.png"))
            page.locator("#pneumonia-tab").click()
            page.locator("#pneumonia-results").screenshot(path=str(output / "05-predictions.png"))
            report["screenshots"].extend(("03-evaluation-metrics.png", "04-confusion-matrix.png", "05-predictions.png"))
            assert not errors, errors
            report["javascript_errors"] = errors
            browser.close()
            print(json.dumps({"model": report["model"], "seconds": report["workflow_seconds"], "confusion_matrix": matrix, "accuracy": result["accuracy"], "screenshots": report["screenshots"]}, indent=2))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
        application.create_model = original_create_model
        (local / "screenshot-run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

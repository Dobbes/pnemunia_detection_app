"""Check local Markdown links, screenshots, and the documented capture metrics."""

import json
from pathlib import Path
import re

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
documents = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
checked = 0
for document in documents:
    content = document.read_text(encoding="utf-8")
    for href in re.findall(r"\]\(([^)]+)\)", content):
        if href.startswith(("https://", "http://", "#")):
            continue
        path = document.parent / href.split("#")[0]
        assert path.is_file(), f"Missing documentation target: {document.name}: {href}"
        checked += 1

images = sorted((ROOT / "docs" / "images").glob("*.png"))
assert len(images) == 5
for path in images:
    with Image.open(path) as image:
        image.verify()

record = ROOT / ".local" / "screenshot-run.json"
if record.exists():
    report = json.loads(record.read_text(encoding="utf-8"))
    assert len(report["screenshots"]) == 5
    assert not report["javascript_errors"]
    result = report["results"]
    guide = (ROOT / "docs" / "SCREENSHOTS.md").read_text(encoding="utf-8")
    for name in ("accuracy", "sensitivity", "specificity", "precision", "f1_score"):
        assert f"{result[name] * 100:.1f}%" in guide, f"Stale documented {name}"
    matrix = result["confusion_matrix"]
    assert f'| Normal | {matrix["true_negative"]} | {matrix["false_positive"]} |' in guide
    assert f'| Pneumonia | {matrix["false_negative"]} | {matrix["true_positive"]} |' in guide
    manifest = report["dataset"]["images"]
    train = {item["patient_key"] for item in manifest if item["split"] == "train"}
    test = {item["patient_key"] for item in manifest if item["split"] == "test"}
    assert train.isdisjoint(test)
    print("Capture record agrees with documented metrics; filename-derived split IDs are disjoint.")

print(f"Verified {len(documents)} Markdown documents, {checked} local links, and {len(images)} PNGs.")

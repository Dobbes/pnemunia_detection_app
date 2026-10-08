"""Build a small, patient-disjoint UI demo from the repository's bundled images.

This is an illustrative subset, not the original dataset's official split.
Run from the repository root: python scripts/prepare_demo_dataset.py
"""

import json
from pathlib import Path
import re
import shutil

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "static" / "dataset"
DESTINATION = ROOT / "chest_xray"


def patient_key(path):
    name = path.name.lower()
    if name.startswith("person"):
        return re.match(r"person\d+", name).group()
    match = re.match(r"((?:normal2-)?im-\d+)", name)
    return match.group(1) if match else path.stem.lower()


def main():
    if DESTINATION.exists():
        raise SystemExit(
            "chest_xray/ already exists; leave it intact. Use a fresh checkout "
            "for this demo, or move your dataset before preparing the subset."
        )

    groups = {
        "NORMAL": sorted(
            path for path in SOURCE.glob("*.jpeg")
            if path.name.startswith(("IM-", "NORMAL2-"))
        ),
        "PNEUMONIA": sorted(SOURCE.glob("person*.jpeg")),
    }
    selected = {}
    for label, paths in groups.items():
        unique = {}
        for path in paths:
            unique.setdefault(patient_key(path), path)
        candidates = list(unique.values())
        if len(candidates) < 28:
            raise SystemExit(f"Need 28 distinct patient IDs for {label}.")
        selected[label] = {"train": candidates[:16], "test": candidates[16:28]}

    manifest = {
        "purpose": "Illustrative UI demo; not the official dataset split or a benchmark",
        "source": "Bundled static/dataset JPEGs; labels inferred from filename conventions",
        "split": "Sorted filenames, one image per filename-derived patient ID; first 16 train, next 12 test per class",
        "images": [],
    }
    for label, splits in selected.items():
        for split, paths in splits.items():
            destination = DESTINATION / split / label
            destination.mkdir(parents=True)
            for path in paths:
                # Explicit class names also support the app's filename-based labeling.
                name = f"demo_{label.lower()}_{path.name}"
                shutil.copyfile(path, destination / name)
                manifest["images"].append({
                    "source": path.relative_to(ROOT).as_posix(),
                    "destination": (destination / name).relative_to(ROOT).as_posix(),
                    "patient_key": patient_key(path),
                    "split": split,
                    "label": label,
                })

        # Supply the two explanatory images referenced by the homepage template.
        example = selected[label]["train"][0]
        Image.open(example).convert("RGB").save(
            ROOT / "static" / "images" / f"sample_{label.lower()}.jpg"
        )

    (DESTINATION / "demo-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("Prepared 16 training and 12 evaluation images per class (56 total).")
    print("Patient IDs are disjoint across splits. See chest_xray/demo-manifest.json.")
    print("This bundled-image subset is for illustrating the app, not model validation.")


if __name__ == "__main__":
    main()

# Screenshots and reproduction

[Back to the project overview](../README.md)

These images come from the running Flask application, including actual Hugging Face model loading, eight epochs of fine-tuning, and inference over 24 separate evaluation X-rays. No prediction or metric values were mocked or edited.

## Gallery

| Capture | What it shows |
| --- | --- |
| [01-overview.png](images/01-overview.png) | Normal/pneumonia examples and experiment instructions |
| [02-training-selection.png](images/02-training-selection.png) | Selection grids, four selected images per class, preview, enabled submission |
| [03-evaluation-metrics.png](images/03-evaluation-metrics.png) | Training summary and five computed metrics |
| [04-confusion-matrix.png](images/04-confusion-matrix.png) | True/false positive and negative counts |
| [05-predictions.png](images/05-predictions.png) | Individual pneumonia-class evaluation images, predictions, confidence, and mistakes |

## Capture environment

- Captured on **8 October 2026**, Windows, Python **3.11.9**.
- Flask **2.3.3**, PyTorch **2.0.1**, torchvision **0.15.2**, Transformers **4.32.1**.
- Playwright **1.58.0**, Microsoft Edge, 1280 × 1000 viewport; element captures retain their full height.
- CPU execution, four PyTorch threads; Python, NumPy, and PyTorch seeds set to `42`.
- Model: `google/vit-base-patch16-224`, checkpoint revision `3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3`.

## The dataset used

`scripts/prepare_demo_dataset.py` takes the committed JPEGs in `static/dataset/`, uses filename conventions to identify classes, and selects one image per filename-derived patient ID. For each class, the first 16 sorted candidates populate the training grid and the next 12 populate the evaluation folder.

The browser selects the first four prepared training images per class: **8 training images total**. The evaluation uses **12 normal + 12 pneumonia images**. Train/test filename-derived IDs do not overlap. The resulting manifest, `chest_xray/demo-manifest.json`, records every source, generated name, class, and split.

This is an illustrative bundled-image subset, **not the source dataset's official split**. Filename conventions are not independently reviewed labels or verified patient identities. The captures demonstrate the application's workflow and expose its mistakes.

## Captured outcome

| Actual class | Predicted normal | Predicted pneumonia |
| --- | ---: | ---: |
| Normal | 4 | 8 |
| Pneumonia | 3 | 9 |

| Metric | Displayed value |
| --- | ---: |
| Accuracy | 54.2% |
| Sensitivity | 75.0% |
| Specificity | 33.3% |
| Precision | 52.9% |
| F1 | 62.1% |

These values describe this particular small run. Loading, fine-tuning, and evaluation took approximately **95 seconds with the checkpoint already cached**. Seeds improve repeatability; filesystem ordering, asynchronous upload arrival, package versions, and hardware can still change training order, timing, and numerical results.

## Reproduce the screenshots

Use a fresh checkout with Python 3.11. Run commands from the repository root. First install the app dependencies and prepare the demo data as described in the [quick start](../README.md#quick-start).

### Windows with Microsoft Edge installed

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-docs.txt
.\.venv\Scripts\python.exe scripts/capture_screenshots.py --browser-channel msedge
```

### Playwright Chromium

```bash
python -m pip install -r requirements-docs.txt
python -m playwright install chromium
python scripts/capture_screenshots.py
```

The script starts its own temporary server at `127.0.0.1:5051`, performs actual image selection and submission, waits for computation, and writes screenshots to `docs/images/`. It checks all X-ray image loads, all five result tabs, the 8-image training summary, the 24-image evaluation count, and confusion-matrix/accuracy consistency. It shuts down its server when complete.

If port 5051 is occupied, pass `--port 5052`. A full machine-readable capture record—including the manifest, selected training filenames, model revision, environment, and per-image outputs—is written locally to `.local/screenshot-run.json` and is ignored by Git.

Regenerating the captures can produce different metrics. Update the documented outcome and captions from the new run together with the images.

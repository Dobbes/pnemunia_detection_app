# Technical guide

[Back to the project overview](../README.md)

## Request-to-result lifecycle

| Route | Method | Behavior |
| --- | --- | --- |
| `/` | GET | Randomly select up to 16 training candidates from each class and render the image grids |
| `/upload` | POST | Accept image files under `images` or dataset filenames under `selected_images[]`; create session state |
| `/process` | GET | Load a fresh model, fine-tune the selected images, evaluate test folders, cache results, redirect |
| `/results/<session_id>` | GET | Render a cached experiment; unknown IDs redirect to the homepage |
| `/example_mode` | GET | Select the bundled examples and launch the same training/evaluation pipeline |

The browser currently uses grid selection rather than a file-picker interface. `/upload` supports multipart uploads at the backend. The request limit is 16 MiB. The UI selection limit is four images per class.

The model-loading and computation steps are synchronous. There is no background job queue or progress endpoint. Session cookies retain selection/session identifiers; `RESULTS_CACHE` retains the computed result in memory. No fine-tuned model checkpoint is persisted.

## Model and training configuration

| Setting | Current implementation |
| --- | --- |
| Primary model | `google/vit-base-patch16-224` |
| Fallback model | `microsoft/resnet-50` |
| Output labels | `0: NORMAL`, `1: PNEUMONIA` |
| Classification head | Newly initialized two-class head via `ignore_mismatched_sizes=True` |
| Training input | 224 × 224 RGB tensors |
| Contrast | Equalize luminance in YCrCb color space |
| Augmentation | Horizontal flip, rotation, affine translation, color jitter, resized crop, optional Gaussian blur |
| Normalization | ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| Optimizer | AdamW, learning rate `5e-6`, weight decay `0.01` |
| Loss | Cross-entropy, with class weights when both filename-based classes are represented |
| Scheduler | ReduceLROnPlateau, factor `0.5`, patience `2` |
| Epochs | Up to 8; early-stopping patience `3` |
| Batch | One image per optimizer step |
| Device | CPU; the application does not explicitly move models or tensors to CUDA |

The Hugging Face processor handles evaluation resizing and normalization after contrast processing. Training uses the explicit torchvision augmentation pipeline. Training epoch loss and accuracy are printed to the terminal, not retained as a training-history chart.

The initial checkpoint is a general-image model. The app does not load an already validated pneumonia classifier. A Hugging Face warning about the replacement classification head is expected.

## Data conventions

Supply class folders under `chest_xray/train/` and `chest_xray/test/`, using uppercase `NORMAL` and `PNEUMONIA`. The app recognizes lowercase `.jpeg`, `.jpg`, and `.png` extensions in those folders.

Training labels are inferred from filenames/paths: `pneumonia` or a `person` filename containing `bacteria` or `virus` identifies the positive class; other names get the negative label. The class-weight and training-summary logic recognizes `normal` in negative filenames. **Use explicit `normal` and `pneumonia` filename markers to keep labels, weights, and displayed training counts consistent.** The demo-preparation script adds those markers and records original filenames in its manifest.

Evaluation labels come from the test folder, not from predictions. The app processes the first 30 eligible files per class in filesystem enumeration order. Images that fail processing are logged and excluded, so compare the displayed count to the intended test count when reviewing an experiment.

For a larger experiment, retain the source dataset's official test split, avoid patient overlap, and preserve a filename/label manifest. The supplied demo script deliberately uses a small, separate illustrative split from bundled files. Its filename-derived IDs do not establish independently verified patient identity or clinical ground truth.

Without a prepared dataset, the homepage falls back to placeholder filenames and the evaluation path can contain no images. The committed `static/dataset/` folder alone is not sufficient to populate the training and test folders; run the preparation script for the quick-start demo.

## Reading the results

Pneumonia is the positive class:

| Metric | Calculation |
| --- | --- |
| Accuracy | `(TP + TN) / (TP + TN + FP + FN)` |
| Sensitivity / recall | `TP / (TP + FN)` |
| Specificity | `TN / (TN + FP)` |
| Precision | `TP / (TP + FP)` |
| F1 | `2TP / (2TP + FP + FN)` |

A zero denominator produces zero in the implementation. Confidence is the softmax probability of the predicted class, not a calibrated clinical probability. The screenshots show the outcome of one small experiment; they are not a benchmark of the underlying architecture.

## Current experimental boundaries

- Tiny UI-selected training sets and filename-derived training labels.
- No cross-validation, external validation, or calibration study in this repository.
- No saved fine-tuned weights, database-backed experiment history, or asynchronous workers.
- `apply_gradcam()` is a placeholder returning a blank image; no genuine Grad-CAM or attribution view is wired into the UI.
- Example mode uses preselected pneumonia images, so it illustrates a different, one-class training selection rather than a balanced evaluation claim.
- Flask's built-in server and fixed development session secret reflect the local prototype setup. The quick-start binds it to localhost.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Dependency installation fails on a newer Python | Use Python 3.11 with the pinned 2023-era PyTorch/torchvision packages |
| OpenCV version cannot be found | The installable version is `4.8.0.76`; use the updated requirements file |
| Broken images on the homepage | Run `scripts/prepare_demo_dataset.py`, or supply the expected class folders and sample images |
| Preparation refuses to run | It protects an existing `chest_xray/` directory; use your current dataset or a fresh checkout |
| Hugging Face download fails | Check internet access to the checkpoint URLs; the loader attempts ResNet-50 if ViT fails |
| CPU training appears slow | Wait for the synchronous request and watch epoch logs in the server terminal |
| Results disappear after restarting | The cache is intentionally process-local; start a new experiment |
| Result counts are smaller than the test folders | Check the 30-per-class cap and terminal messages for failed images |
| Browser styling or result tabs fail | Bootstrap CSS/JS is loaded from jsDelivr; the browser needs access to that CDN |

## Documentation-run fixes

Running the app for documentation exposed a truncated results template, a duplicated result block with an unconditional incorrect badge, and Windows filesystem separators in browser image URLs. The accompanying changes complete the template, render each result once, use forward-slash static URLs, and keep the footer from covering result cards. The homepage model description now correctly identifies ImageNet pretraining.

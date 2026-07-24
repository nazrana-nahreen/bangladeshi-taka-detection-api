# Bangladeshi Taka Note Detection API

A YOLOv11-based REST API for detecting Bangladeshi banknote denominations (2, 5, 10, 20, 50, 100, 500, and 1000 Taka) from images, served via FastAPI, containerized with Docker, and deployed live with a Bengali-language web interface.

🔗 **Live App:** https://bangladeshi-taka-detection-api-j8gz.onrender.com
🔗 **API Docs (Swagger):** https://bangladeshi-taka-detection-api-j8gz.onrender.com/docs

*(Free-tier hosting — the first request after a period of inactivity may take 10–30 seconds while the service wakes up.)*

## Overview

This project detects and classifies Bangladeshi currency notes using a custom-trained YOLOv11 object detection model. Given an input image, the API returns the detected denomination, confidence score, and bounding box coordinates in JSON format. A lightweight Bengali-language web UI is served alongside the API for direct, non-technical use — upload a photo of a note and instantly see the result.

## Project Structure
bangladeshi-taka-detection-api/
├── app/
│ ├── init.py
│ ├── main.py # FastAPI application (routes, validation, error handling, CORS)
│ └── inference.py # Model loading and prediction logic
├── static/
│ └── index.html # Bengali-language frontend (upload UI, live results)
├── models/
│ └── best.pt # Trained YOLOv11 model weights
├── notebooks/
│ └── train_phase1.ipynb # Phase 1 training notebook (Kaggle)
├── test_images/ # Sample images used for API testing
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
## Phase 1: Model Training

- **Dataset:** [Bangladeshi Banknote Dataset](https://www.kaggle.com/datasets/rahnumatasnim1604103/bangladeshi-banknote-dataset) (Kaggle) — ~70K labeled images across 8 denominations.
- **Model:** YOLOv11n (Ultralytics), trained for 50 epochs on a Tesla T4 GPU (Kaggle).
- **Result:** mAP50-95 of 0.995 on the validation set.
- **Note:** Since the dataset consists of single, pre-cropped banknote images (no multi-object scenes), bounding boxes were generated covering the full image rather than tight object-level annotations. This produces near-perfect validation metrics but does not fully reflect real-world detection difficulty — see [Accuracy Discussion](#accuracy-discussion) below.

## Phase 2: Inference Pipeline & REST API

The trained model is served through a FastAPI application.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Bengali-language web UI (image upload + live results) |
| GET | `/health` | Health check (JSON) |
| POST | `/predict` | Accepts an image file, returns detections |

### Request

- **Content-Type:** `multipart/form-data`
- **Field:** `file` — JPEG or PNG image

### Response

```json
{
  "filename": "test.jpg",
  "num_detections": 1,
  "recognized": true,
  "detections": [
    {
      "denomination": "20",
      "confidence": 0.9949,
      "bbox": {
        "x1": 0.0,
        "y1": 0.1,
        "x2": 330.91,
        "y2": 152.0
      }
    }
  ]
}
```

`recognized` is `false` when the top detection's confidence falls below the configured threshold (`0.85`) — this is used by the frontend to show a friendly "note not recognized" message instead of a low-confidence guess.

### Error Handling

| Status Code | Condition |
|--------------|-----------|
| 200 | Successful request (detection may or may not be `recognized`) |
| 400 | Missing or empty file |
| 415 | Unsupported file type (non-JPEG/PNG) |
| 422 | Invalid/missing request body (FastAPI validation) |
| 500 | Internal inference error |

## Web Interface

A Bengali-language frontend (`static/index.html`) is served at the root path (`/`), styled with Bangladesh's national colors and a wave-motif inspired by the national emblem. Features:
- Drag-and-drop or click-to-upload image input
- Animated loading state during inference
- Large, clear result display in Bengali (e.g. "আপনার নোটটি ২০ টাকা!")
- Graceful handling of unrecognized notes, explicitly listing the denominations the model supports, since the model has no "background"/negative class and will otherwise force a prediction on any image

## Running Locally (without Docker)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/` for the web UI, or `http://localhost:8000/docs` for the interactive Swagger UI.

## Running with Docker

### Build the image

```bash
docker build -t taka-detector-api .
```

### Run the container

```bash
docker run -d -p 8000:8000 --name taka-detection-container taka-detector-api
```

### Verify

```bash
docker ps
```

Then visit `http://localhost:8000/`, or test the API directly:

```bash
curl -X POST http://localhost:8000/predict -F "file=@test_images/sample.jpg"
```

## Cloud Deployment

Deployed on **Render.com** (free tier) directly from this GitHub repository — Render auto-detects the `Dockerfile` and rebuilds automatically on every push to `main`.

**Live URL:** https://bangladeshi-taka-detection-api-j8gz.onrender.com

*(Railway.app and Hugging Face Spaces were evaluated first; Railway's free trial had expired and Hugging Face's Docker SDK required a paid plan on the account used, so Render.com was chosen as a genuinely free, card-free alternative with native Docker support.)*

## API Testing Summary

Tested against multiple images covering different denominations, plus edge cases:

| Denomination | Predicted | Confidence | Result |
|---|---|---|---|
| 20 Taka | 20 | 0.9949 | ✅ Correct |
| 2 Taka | 2 | 0.9969 | ✅ Correct |
| 2 Taka (out-of-dataset photo) | 1000 | 0.5877 | ❌ Misclassified (below 0.85 threshold → shown as "not recognized" in the UI) |
| ... | ... | ... | *(fill in remaining test results)* |

Invalid input handling was also verified: non-image files return `415`, requests missing the file field return `422`, and unrelated/non-note images are caught by the confidence threshold and surfaced as "note not recognized" rather than a confident wrong answer.

## Accuracy Discussion

The model achieves near-perfect metrics (mAP50-95: 0.995) on the validation split, but this reflects the dataset's structure — every image contains a single, centered, pre-cropped note, and bounding boxes were generated as full-image boxes. Consequently, the "detection" task in this dataset closely resembles classification rather than true multi-object detection.

When tested against out-of-distribution images (note photos sourced outside the training dataset, and entirely unrelated images), two limitations became apparent:
1. **Reduced confidence on real-world photos** — an out-of-dataset photo of a 2 Taka note was misclassified with a comparatively low confidence of 0.59.
2. **No "background" class** — since the model was trained exclusively on the 8 denomination classes, it has no concept of "not a banknote" and will force a prediction on any input image, occasionally with high confidence. A `CONFIDENCE_THRESHOLD` (0.85) was introduced in the API to mitigate this by treating low-confidence predictions as "unrecognized," though this does not fully eliminate confidently-wrong predictions on unrelated images.

This suggests that while the model generalizes very well within its training distribution, real-world deployment would benefit from a dedicated negative/background class and more diverse training images (varied lighting, angles, backgrounds, partial occlusion, multiple notes per frame) to maintain reliability outside the training distribution.

## Tech Stack

- **Model:** YOLOv11n (Ultralytics)
- **API Framework:** FastAPI + Uvicorn
- **Frontend:** HTML/CSS/JavaScript (Bengali language, Noto Sans Bengali & Baloo Da 2 fonts)
- **Containerization:** Docker
- **Training Environment:** Kaggle Notebooks (Tesla T4 GPU)
- **Deployment:** Render.com

## Author

Nazrana — IIUC, CSE 
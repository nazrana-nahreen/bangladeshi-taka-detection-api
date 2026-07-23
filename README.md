# Bangladeshi Taka Note Detection API

A YOLOv11-based REST API for detecting Bangladeshi banknote denominations (2, 5, 10, 20, 50, 100, 500, and 1000 Taka) from images, served via FastAPI and containerized with Docker.

## Overview

This project detects and classifies Bangladeshi currency notes using a custom-trained YOLOv11 object detection model. Given an input image, the API returns the detected denomination, confidence score, and bounding box coordinates in JSON format.

## Project Structure
bangladeshi-taka-detection-api/
├── app/
│ ├── init.py
│ ├── main.py # FastAPI application (routes, validation, error handling)
│ └── inference.py # Model loading and prediction logic
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

The trained model is served through a FastAPI application with a single prediction endpoint.

### Endpoint

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Accepts an image file, returns detections |

### Request

- **Content-Type:** `multipart/form-data`
- **Field:** `file` — JPEG or PNG image

### Response

```json
{
  "filename": "test.jpg",
  "num_detections": 1,
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

### Error Handling

| Status Code | Condition |
|--------------|-----------|
| 200 | Successful detection |
| 400 | Missing or empty file |
| 415 | Unsupported file type (non-JPEG/PNG) |
| 422 | Invalid/missing request body (FastAPI validation) |
| 500 | Internal inference error |

## Running Locally (without Docker)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

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

Then visit `http://localhost:8000/docs`, or test directly:

```bash
curl -X POST http://localhost:8000/predict -F "file=@test_images/sample.jpg"
```

## API Testing Summary

Tested against 5+ images covering different denominations:

| Denomination | Predicted | Confidence | Result |
|---|---|---|---|
| 20 Taka | 20 | 0.9949 | ✅ Correct |
| 2 Taka | 2 | 0.9969 | ✅ Correct |
| 2 Taka (out-of-dataset photo) | 1000 | 0.5877 | ❌ Misclassified |
| ... | ... | ... | *(fill in remaining test results)* |

Invalid input handling was also verified: non-image files return `415`, and requests missing the file field return `422`.

## Accuracy Discussion

The model achieves near-perfect metrics (mAP50-95: 0.995) on the validation split, but this reflects the dataset's structure — every image contains a single, centered, pre-cropped note, and bounding boxes were generated as full-image boxes. Consequently, the "detection" task in this dataset closely resembles classification rather than true multi-object detection.

When tested against an out-of-distribution image (a note photo sourced outside the training dataset), the model produced a low-confidence (0.59) misclassification. This suggests that while the model generalizes very well within the training distribution, real-world deployment (varied lighting, angles, backgrounds, partial occlusion, multiple notes per image) would likely require additional data diversity and possibly re-annotation with true bounding boxes to maintain this level of accuracy.

## Tech Stack

- **Model:** YOLOv11n (Ultralytics)
- **API Framework:** FastAPI + Uvicorn
- **Containerization:** Docker
- **Training Environment:** Kaggle Notebooks (Tesla T4 GPU)

## Author

Nazrana Nahreen — IIUC, CSE
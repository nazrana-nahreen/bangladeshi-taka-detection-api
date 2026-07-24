from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.inference import predict_image

app = FastAPI(
    title="Bangladeshi Taka Note Detection API",
    description="YOLOv11-based REST API for detecting Bangladeshi banknote denominations",
    version="1.0.0"
)

# Allow frontend (same origin, but keep open for flexibility)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}
CONFIDENCE_THRESHOLD = 0.85  # notice-kom-confidence hole "chinte parlam na" dekhabe

@app.get("/health")
def health():
    return {"message": "Taka Note Detection API is running. Use POST /predict with an image file."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Only JPEG/PNG allowed."
        )

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        detections = predict_image(image_bytes)

        # kono detection na paile, ba confidence kom hole "chinte parlam na" pathabo
        recognized = [d for d in detections if d["confidence"] >= CONFIDENCE_THRESHOLD]

        return JSONResponse(status_code=200, content={
            "filename": file.filename,
            "num_detections": len(detections),
            "recognized": len(recognized) > 0,
            "detections": detections
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# Static frontend serve koro (shob shesh e mount koro, jate /predict, /health override na hoy)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
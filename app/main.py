from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.inference import predict_image

app = FastAPI(
    title="Bangladeshi Taka Note Detection API",
    description="YOLOv11-based REST API for detecting Bangladeshi banknote denominations",
    version="1.0.0"
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}

@app.get("/")
def root():
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

        return JSONResponse(status_code=200, content={
            "filename": file.filename,
            "num_detections": len(detections),
            "detections": detections
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
FROM python:3.10-slim

WORKDIR /code

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=300 --retries 10 fastapi==0.115.0 uvicorn[standard]==0.30.6 python-multipart==0.0.9 pillow==10.4.0
RUN pip install --no-cache-dir --default-timeout=300 --retries 10 opencv-python-headless==4.10.0.84
RUN pip install --no-cache-dir --default-timeout=300 --retries 10 ultralytics==8.3.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY app/ ./app/
COPY models/ ./models/
COPY static/ ./static/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
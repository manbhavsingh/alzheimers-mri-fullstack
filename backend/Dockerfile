FROM python:3.11-slim

WORKDIR /code

# libgl1 + libglib2.0-0 are required by opencv-python-headless at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Place best_resnet.pth at app/weights/best_resnet.pth before building,
# or mount it / set MODEL_WEIGHTS_PATH to point elsewhere at runtime.

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

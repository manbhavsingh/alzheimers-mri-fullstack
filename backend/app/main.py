import os
from collections import deque
from threading import Lock
from time import monotonic
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from .gradcam_note import gradcam_note as summarize_gradcam
from .inference import predict
from .report import Report, ReportGenerationError, make_report

app = FastAPI(
    title="Explainable Alzheimer's Detection API",
    description="ResNet-18 + Grad-CAM inference for MRI-based Alzheimer's staging.",
    version="1.1.0",
)

# Restrict this to your deployed frontend origin in production via env var.
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
ActivationValue = Annotated[float, Field(ge=0.0, le=1.0)]
_REPORT_RATE_LIMIT = 10
_REPORT_RATE_WINDOW_SECONDS = 60.0
_REPORT_REQUESTS: dict[str, deque[float]] = {}
_REPORT_RATE_LOCK = Lock()


class ReportRequest(BaseModel):
    label: str = Field(min_length=1, max_length=120)
    probs: dict[str, Probability]
    gradcam_note: str | None = Field(default=None, min_length=1, max_length=2000)
    cam: list[list[ActivationValue]] | None = None

    @model_validator(mode="after")
    def validate_cam_or_note(self):
        if not 1 <= len(self.probs) <= 16:
            raise ValueError("probs must contain between 1 and 16 class scores.")
        if (self.cam is None) == (self.gradcam_note is None):
            raise ValueError("Provide exactly one of gradcam_note or cam.")
        if self.cam is not None:
            if not 1 <= len(self.cam) <= 224:
                raise ValueError("cam must have between 1 and 224 rows.")
            width = len(self.cam[0])
            if not 1 <= width <= 224:
                raise ValueError("cam must have between 1 and 224 columns.")
            if any(len(row) != width for row in self.cam):
                raise ValueError("cam rows must all have the same length.")
        return self


def _enforce_report_rate_limit(client_ip: str) -> None:
    now = monotonic()
    with _REPORT_RATE_LOCK:
        # Drop old buckets as well as expired entries to keep this small on Render.
        for ip in list(_REPORT_REQUESTS):
            bucket = _REPORT_REQUESTS[ip]
            while bucket and now - bucket[0] >= _REPORT_RATE_WINDOW_SECONDS:
                bucket.popleft()
            if not bucket:
                del _REPORT_REQUESTS[ip]

        bucket = _REPORT_REQUESTS.setdefault(client_ip, deque())
        if len(bucket) >= _REPORT_RATE_LIMIT:
            retry_after = max(1, int(_REPORT_RATE_WINDOW_SECONDS - (now - bucket[0])))
            raise HTTPException(
                status_code=429,
                detail="Report limit reached. Please wait a minute before trying again.",
                headers={"Retry-After": str(retry_after)},
            )
        bucket.append(now)


@app.get("/")
def root():
    """Provide a helpful landing response for the API base URL."""
    return {
        "service": "Neurio API",
        "status": "ok",
        "health": "/health",
        "docs": "/docs",
        "report_endpoint": "POST /report",
        "prediction_endpoint": "POST /predict",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Upload a PNG or JPEG MRI scan.",
        )

    image_bytes = await file.read()

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 10MB).")

    try:
        result = predict(image_bytes)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return result


@app.post("/report", response_model=Report)
def report_endpoint(payload: ReportRequest, request: Request) -> Report:
    client_ip = request.client.host if request.client else "unknown"
    _enforce_report_rate_limit(client_ip)

    note = payload.gradcam_note
    if payload.cam is not None:
        note = summarize_gradcam(payload.cam)

    try:
        return make_report(
            label=payload.label,
            probs=payload.probs,
            gradcam_note=note or "No Grad-CAM note was provided.",
        )
    except ReportGenerationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

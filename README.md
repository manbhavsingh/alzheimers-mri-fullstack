# Neurio — Explainable Alzheimer's MRI Staging (Full-Stack)

A minimal full-stack demo that wraps a ResNet-18 classifier + Grad-CAM visualization into a simple web product: upload an MRI slice, get a 4-class predicted stage and a Grad-CAM heatmap overlay shown with a before/after slider. Built as a FastAPI inference backend (PyTorch) and a React + Vite frontend.

Why this exists: researcher/portfolio demo to visualize which brain regions drive a CNN prediction. Not medical software — for research and demonstration only.

---

## Table of contents
- <a>Features</a>
- <a>Stack &amp; notable libraries</a>
- <a>Repository layout</a>
- <a>Get the trained model weights</a>
- <a>Run locally</a>
  - <a>Backend (FastAPI)</a>
  - <a>Frontend (React + Vite)</a>
- <a>API</a>
- <a>Deployment notes</a>
- <a>Troubleshooting</a>
- <a>Contributing</a>
- <a>License</a>
- <a>Acknowledgements</a>

---

## Features
- Single-file upload (PNG/JPEG) → 4-class Alzheimer’s staging prediction
- Grad-CAM heatmap generation and overlay (returned as base64 PNGs)
- Synchronized slider UI to compare original image and Grad-CAM overlay
- Simple health-check and ready-to-deploy project structure for Render (backend) + Vercel (frontend)

---

## Stack &amp; notable libraries
- Languages: Python (backend), JavaScript/React (frontend)
- Frameworks / runtime:
  - Backend: FastAPI + Uvicorn
  - Frontend: React + Vite
- Notable libraries:
  - PyTorch (model inference)
  - grad-cam (or custom Grad-CAM logic in repo)
  - Pillow / NumPy (image handling)
  - React (UI) + Vite (dev server &amp; build)

---

## Repository layout
```
README.md
backend/                 FastAPI app, model, inference utilities, Dockerfile, Render config
  ├─ app/
  │  ├─ main.py           FastAPI routes and startup
  │  ├─ inference.py      image preprocessing, model loading, Grad-CAM generation
  │  ├─ model.py          model architecture (must match training)
  │  └─ weights/          place trained checkpoint here (best_resnet.pth)
  ├─ requirements.txt
  └─ Dockerfile
frontend/                React + Vite UI
  ├─ index.html
  ├─ package.json
  ├─ src/
  │  ├─ App.jsx
  │  └─ components/      UploadPanel, ResultPanel, CompareSlider
  └─ .env.example
```

How it fits together: The frontend posts an uploaded PNG/JPEG to the backend `/predict` endpoint. The backend loads the trained checkpoint, runs a forward pass to compute class probabilities and a Grad-CAM heatmap, then returns the prediction + two base64 images (original + overlay). Frontend displays them with a slider.

---

## Get the trained model weights (required)
This repository intentionally does NOT include the trained checkpoint file `best_resnet.pth`. You must obtain it from the original training notebook (or a saved copy) and place it at:

```
backend/app/weights/best_resnet.pth
```

Options:
- Re-run the original Kaggle notebook `hack4health-alzheimer-mri-cnn.ipynb` and download `best_resnet.pth`.
- Use Git LFS to store the checkpoint (recommended if you want it in the repo).
- Upload the checkpoint to a storage service (S3, Hugging Face Hub) and modify `app/inference.py:load_model()` to download it on startup (example snippet below).

Example snippet to download model on first run (add to load_model if you host the file externally):
```python
import os
import requests

MODEL_PATH = "app/weights/best_resnet.pth"
MODEL_URL = os.getenv("MODEL_URL")  # set this to the public S3/HF URL

if not os.path.exists(MODEL_PATH) and MODEL_URL:
    r = requests.get(MODEL_URL, stream=True)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
```

---

## Run locally

Requirements:
- Python 3.8+ (backend)
- Node.js 16+ / npm (frontend)
- The checkpoint at backend/app/weights/best_resnet.pth

Backend (local development)
```bash
cd backend
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
# venv\Scripts\Activate.ps1
pip install -r requirements.txt

# ensure backend/app/weights/best_resnet.pth exists
uvicorn app.main:app --reload --port 8000
```
Health check: GET http://localhost:8000/health  → {"status": "ok"}

Frontend (local development)
```bash
cd frontend
npm install
cp .env.example .env        # or set VITE_API_URL=http://localhost:8000
npm run dev
```
Open the local Vite URL printed by the dev server (usually http://localhost:5173).

---

## API

POST /predict
- Content-Type: multipart/form-data
- Field: `file` — PNG/JPEG file (&lt;= 10MB)

curl example:
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/brain_slice.png" \
  -H "Accept: application/json"
```

Sample JSON response:
```json
{
  "predicted_class": "Mild Demented",
  "predicted_index": 2,
  "confidence": 0.87,
  "probabilities": {
    "Non-Demented": 0.02,
    "Very Mild Demented": 0.09,
    "Mild Demented": 0.87,
    "Moderate Demented": 0.02
  },
  "original_image": "data:image/png;base64,<...>",
  "gradcam_overlay": "data:image/png;base64,<...>"
}
```

Notes:
- The backend returns base64-encoded PNG data URIs for quick in-browser display.
- If the checkpoint is missing, the service returns HTTP 503 with an explanatory message.

---

## Deployment notes

Backend → Render
- Create a Render Web Service and set the root to `backend/`.
- This repo includes `backend/render.yaml` and a Dockerfile to help deployment.
- Options for providing the checkpoint:
  - Commit via Git LFS
  - Use Render persistent disk and upload the file into the container
  - Host externally and download on startup (see snippet above)
- Set environment variable `ALLOWED_ORIGINS` to your frontend URL to restrict CORS.

Frontend → Vercel
- Create a new Vercel Project, set root to `frontend/`, framework preset Vite.
- Add env var `VITE_API_URL` pointing to your backend (e.g., `https://<render-app>.onrender.com`).

CORS: After deploying, set the backend `ALLOWED_ORIGINS` to the frontend deployment URL.

---

## Troubleshooting
- 503 on /predict: most likely the model checkpoint is missing. Confirm `backend/app/weights/best_resnet.pth` exists.
- Model load errors (size mismatch / missing keys): ensure `backend/app/model.py` exactly matches the architecture used during training.
- Slow inference on CPU: consider using a GPU-enabled instance or optimize to TorchScript if needed.
- Check backend logs (Docker / Render logs / uvicorn console) for stack traces.

---

## Contributing
- Issues and PRs welcome — this is a research demo. If you add checkpoints, consider using Git LFS or external hosting.
- Suggestions:
  - Add model download support via MODEL_URL + secure storage.
  - Add unit tests for preprocessing and a small integration test for the `/predict` route.
  - Improve security (rate-limiting / authentication) before exposing a public endpoint.

---

## License
No license file included in this repository. Add a LICENSE to clarify reuse (MIT / Apache-2.0 / etc.) if you intend public distribution.

---

## Acknowledgements
- Built from the training notebook in Explainable-Alzheimers-Detection and adapted into a full-stack demo.
- If you use or adapt this project, please include a short note attributing the training/implementation sources.

# Neurio — Explainable Alzheimer's MRI Staging

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A full-stack diagnostic-viewer app around a ResNet-18 + Grad-CAM model that
classifies brain MRI scans into four Alzheimer's stages — and shows *why*,
via an interactive heatmap slider, instead of returning a black-box label.

Built on top of the original ML research in
[Explainable-Alzheimers-Detection](https://github.com/manbhavsingh/Explainable-Alzheimers-Detection)
(90% test accuracy ResNet-18 classifier with Grad-CAM interpretability),
wrapped in a FastAPI backend and React frontend so it's usable as a real
tool instead of a notebook.

## Demo

![Neurio prediction result — Mild Demented, 52% confidence, with Grad-CAM heatmap slider](docs/screenshot-result.png)

*Upload a scan → get a predicted stage, confidence breakdown, and a
drag-to-reveal comparison between the raw MRI and the model's attention
heatmap.*

## Architecture
├── backend/ FastAPI inference API — loads the trained ResNet-18,
│ runs classification + Grad-CAM, returns JSON + base64 images
└── frontend/ React + Vite UI — upload, results display, compare slider
`POST /predict` (multipart, PNG/JPEG ≤10MB) → predicted stage, per-class
confidence scores, and a Grad-CAM overlay image.

## How this was built

The ML model (ResNet-18 fine-tuning, Grad-CAM implementation, training
pipeline) is from the original research notebook. The full-stack layer —
FastAPI backend, React frontend, deployment config — was built with Claude
as a learning project to turn a research notebook into a usable app.

---

## 1. Get your trained weights

This repo does not include `best_resnet.pth` — the original repo trains it
inside a Kaggle notebook and never commits the checkpoint. You need to
export it once:

1. Open `hack4health-alzheimer-mri-cnn.ipynb` on Kaggle (or locally, with
   the MRI parquet dataset available) and run it top to bottom. It already
   saves the best checkpoint to `best_resnet.pth` during training (see the
   training-loop and Grad-CAM cells).
2. Download `best_resnet.pth` from the Kaggle output panel (or your local
   working directory).
3. Place it at `backend/app/weights/best_resnet.pth`.

The backend refuses to start inference (returns HTTP 503 with a clear
message) until this file exists, so you'll know immediately if it's
missing.

## 2. Run locally

**Backend:**
```bash
cd backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
# make sure app/weights/best_resnet.pth exists (see step 1)
uvicorn app.main:app --reload --port 8000
```
Visit `http://localhost:8000/health` — should return `{"status": "ok"}`.

**Frontend:**
```bash
cd frontend
npm install
cp .env.example .env      # VITE_API_URL=http://localhost:8000
npm run dev
```
Open the printed local URL (typically `http://localhost:5173`).

## 3. Deploy

**Backend → Render**
1. Push this repo to GitHub.
2. In Render: New → Web Service → connect the repo, set root directory to
   `backend`. Render will pick up `render.yaml` automatically (Docker
   runtime, health check at `/health`).
3. **Important:** `best_resnet.pth` isn't in git (it's large — add it to
   `.gitignore` or use Git LFS). Either:
   - commit it via Git LFS, or
   - add it as a [Render persistent disk](https://render.com/docs/disks) /
     upload it into the container at build time, or
   - host it externally (S3, HF Hub) and have `app/inference.py` download
     it on first boot — small change to `load_model()` if you go this route.
4. Set `ALLOWED_ORIGINS` env var to your deployed frontend URL once you
   have it (step below), to lock down CORS.

**Frontend → Vercel**
1. In Vercel: New Project → import the repo → set root directory to
   `frontend`.
2. Framework preset: Vite (auto-detected via `vercel.json`).
3. Add env var `VITE_API_URL` = your Render backend URL
   (e.g. `https://alzheimers-detection-api.onrender.com`).
4. Deploy. Then go back to Render and set `ALLOWED_ORIGINS` to this
   Vercel URL.

## API

`POST /predict` — multipart form, field `file` (PNG/JPEG, ≤10MB).

Response:
```json
{
  "predicted_class": "Mild Demented",
  "predicted_index": 2,
  "confidence": 0.87,
  "probabilities": { "Non-Demented": 0.02, "...": "..." },
  "original_image": "data:image/png;base64,...",
  "gradcam_overlay": "data:image/png;base64,..."
}
```

## Notes

- This is a research/portfolio tool, not a diagnostic device — the UI
  says so, and any real deployment should too.
- The model architecture in `backend/app/model.py` must exactly match the
  training notebook's (grayscale conv1, 4-class fc head) or the checkpoint
  won't load correctly — it's copied verbatim from the notebook.

## Deployment Notes

Neurio is deployed with a FastAPI backend on Render and a React + Vite frontend on Vercel.

**Live app:** https://alzheimers-mri-fullstack.vercel.app
**Backend API:** https://alzheimers-mri-fullstack.onrender.com

### Challenges solved during deployment

- **Memory limits on Render's free tier (512MB):** The default `torch` package pulls
  in a CUDA-enabled build (~700MB+), which exceeded Render's memory limit even though
  no GPU is available in production. Switched to the CPU-only PyTorch wheel via: --extra-index-url https://download.pytorch.org/whl/cpu
  in `requirements.txt`, cutting memory usage significantly and resolving repeated
  service restarts.

- **Single-worker inference:** Ensured Uvicorn runs with `--workers 1` so the ResNet-18
  model is loaded into memory only once, not duplicated per worker process.

- **CORS between Vercel and Render:** The deployed frontend (Vercel) and backend
  (Render) run on different origins, so the FastAPI backend explicitly allows the
  Vercel domain via `CORSMiddleware`, configured through an `ALLOWED_ORIGINS`
  environment variable rather than hardcoding it.

- **Cold starts:** Render's free tier spins down after inactivity, so the first
  request after idle time can take 30-60s while the container restarts.

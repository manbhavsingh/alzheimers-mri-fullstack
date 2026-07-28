# Neurio — Explainable Alzheimer's MRI Staging (Full-Stack)

A full-stack wrapper around the ResNet-18 + Grad-CAM model from
[Explainable-Alzheimers-Detection](https://github.com/manbhavsingh/Explainable-Alzheimers-Detection).
Upload a brain MRI scan, get a predicted stage, and see a Grad-CAM heatmap
overlay showing which regions drove the prediction — via a synced
before/after slider.

```
├── backend/          FastAPI inference API (PyTorch, Grad-CAM)
└── frontend/          React + Vite UI
```

## 1. Get your trained weights

This repo does **not** include `best_resnet.pth` — the original repo trains
it inside a Kaggle notebook and never commits the checkpoint. You need to
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

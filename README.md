# InsightDesk AI

AI-powered IT helpdesk platform that classifies support tickets, retrieves solutions via RAG, detects anomalies, and monitors model health — all through a modern React dashboard.

**Live demo → https://insightdesk-ai-sqck.onrender.com/** (health: `/health`)

> Previously hosted on Heroku; migrated to Render's free tier after Heroku removed free dynos.

---

## Features

| Module | What it does |
|---|---|
| **Ticket Categorization** | XGBoost classification with confidence scores (a TensorFlow variant exists in the code but isn't shipped to prod — see [Stack](#stack)) |
| **Solution Retrieval** | Semantic search over a knowledge base using FAISS + fastembed |
| **Anomaly Detection** | Flags unusual patterns in ticket volume, response times, and model outputs |
| **Model Monitoring** | Tracks accuracy, data drift, and latency across deployed models |
| **Feedback Loop** | Star ratings and agent corrections feed back into model evaluation |

---

## Stack

**Backend** — FastAPI · scikit-learn · XGBoost · FAISS · fastembed · Uvicorn  
**Frontend** — React 18 · Vite · Tailwind CSS · lucide-react  
**Infra** — Docker · Render (container stack)

There's also a TensorFlow classifier still in `src/models/`, wired up behind a try/except import so the app doesn't crash without it, but TensorFlow isn't in any `requirements*.txt` — it's dead weight from an earlier version, not something you can currently turn on. The RAG pipeline used to run on sentence-transformers + torch, which was swapped for fastembed to cut the Docker image down to something Render's free tier could actually build in time.

---

## Local development

```bash
git clone https://github.com/iamvisheshsrivastava/insightdesk-ai.git
cd insightdesk-ai

# Backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements-prod.txt
uvicorn src.api.main:app --reload

# Frontend (separate terminal)
cd frontend && npm install && npm run dev
```

Backend runs on `http://localhost:8000` · Frontend on `http://localhost:3000`  
API docs available at `http://localhost:8000/docs`

---

## Project structure

```
insightdesk-ai/
├── src/
│   ├── api/          # FastAPI app and routes
│   ├── models/       # Classifier logic
│   ├── retrieval/    # RAG pipeline (FAISS + embeddings)
│   ├── anomaly/      # Anomaly detection
│   ├── monitoring/   # Drift detection & performance metrics
│   └── feedback/     # Feedback collection and management
├── frontend/         # React + Vite + Tailwind
├── models/           # Trained model artifacts (.pkl / .joblib)
├── Dockerfile
├── render.yaml
└── heroku.yml         # legacy, no longer used for deployment
```

---

## Deployment

The app is containerised and deployed to Render (free tier) via `render.yaml`:

1. On [Render](https://render.com), choose **New > Blueprint** and point it at this repo.
2. Deploy. Render auto-redeploys on every push to `main` (`autoDeployTrigger: commit`).
3. Confirm it's running via `/health`.

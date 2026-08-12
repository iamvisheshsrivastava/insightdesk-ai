# InsightDesk AI

AI-powered IT helpdesk platform that classifies support tickets, retrieves solutions via RAG, detects anomalies, and monitors model health — all through a modern React dashboard.

**Live demo → https://insightdesk-ai-sqck.onrender.com/** (health: `/health`)

> Previously hosted on Heroku; migrated to Render's free tier after Heroku removed free dynos.

---

## Features

| Module | What it does |
|---|---|
| **Ticket Categorization** | Dual-model classification (XGBoost + deep model) with confidence scores |
| **Solution Retrieval** | Semantic search over a knowledge base using FAISS + fastembed |
| **Anomaly Detection** | Flags unusual patterns in ticket volume, response times, and model outputs |
| **Model Monitoring** | Tracks accuracy, data drift, and latency across deployed models |
| **Feedback Loop** | Star ratings and agent corrections feed back into model evaluation |

---

## Stack

**Backend** — FastAPI · scikit-learn · XGBoost · FAISS · fastembed · Uvicorn  
**Frontend** — React 18 · Vite · Tailwind CSS · lucide-react  
**Infra** — Docker · Render (container stack)

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

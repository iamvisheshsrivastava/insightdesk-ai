# InsightDesk AI

AI-powered IT helpdesk platform that classifies support tickets, retrieves solutions via RAG, detects anomalies, and monitors model health — all through a modern React dashboard.

**Live demo → https://insightdesk-ai-77c8cfe1286c.herokuapp.com/**

---

## Features

| Module | What it does |
|---|---|
| **Ticket Categorization** | Dual-model classification (XGBoost + deep model) with confidence scores |
| **Solution Retrieval** | Semantic search over a knowledge base using FAISS + sentence-transformers |
| **Anomaly Detection** | Flags unusual patterns in ticket volume, response times, and model outputs |
| **Model Monitoring** | Tracks accuracy, data drift, and latency across deployed models |
| **Feedback Loop** | Star ratings and agent corrections feed back into model evaluation |

---

## Stack

**Backend** — FastAPI · scikit-learn · XGBoost · FAISS · sentence-transformers · Uvicorn  
**Frontend** — React 18 · Vite · Tailwind CSS · lucide-react  
**Infra** — Docker · Heroku (container stack)

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
└── heroku.yml
```

---

## Deployment

The app is containerised and deployed to Heroku via Docker:

```bash
git push heroku main   # triggers a Docker build and deploy
```

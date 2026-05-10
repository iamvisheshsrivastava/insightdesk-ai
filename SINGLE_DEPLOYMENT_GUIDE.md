# Single-Container Deployment Guide

## Overview

Your InsightDesk AI application can now be deployed as a **single Docker container** that serves:
- ✅ React frontend (build output at `frontend/dist/`)
- ✅ FastAPI backend (all `/api/*` endpoints)
- ✅ Static assets (CSS, JS, images)
- ✅ File storage (FAISS vectors, ML models, data)

**No separate frontend service needed.** One container = Frontend + Backend.

---

## Architecture

```
┌─────────────────────────────────┐
│  Single Docker Container        │
├─────────────────────────────────┤
│  FastAPI (uvicorn on :8000)     │
│  ├─ /api/*            → Endpoints
│  ├─ /assets/*         → React CSS/JS
│  ├─ /{anything-else}  → index.html (SPA)
│  └─ Persistent Vol.   → data/, models/, vectors/
├─────────────────────────────────┤
│  Optional: Neo4j Service        │
└─────────────────────────────────┘
```

**Browser requests flow:**
1. User opens `https://insightdesk.example.com/`
2. FastAPI serves `index.html` (SPA entry point)
3. React app loads and proxies `/api/` calls to same origin
4. All requests stay within one container/process

---

## Local Development

### **Option A: Run locally (without Docker)**

```bash
# 1. Install backend dependencies
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements-prod.txt

# 2. Build frontend
cd frontend
npm install
npm run build
cd ..

# 3. Start FastAPI (serves both frontend + API)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Open http://localhost:8000
# Frontend loads from frontend/dist/
# API available at http://localhost:8000/api/*
```

### **Option B: Run with Docker (local)**

```bash
# Build and run
docker compose -f docker-compose.single.yml up --build

# Frontend: http://localhost:8000
# API: http://localhost:8000/api/*
# Neo4j Browser: http://localhost:7474 (optional)
```

---

## Cloud Deployment

### **Option 1: Fly.io (Recommended - Easiest)**

**Free tier:** 3x shared VMs + 3GB storage

```bash
# 1. Install Fly CLI
curl -L https://fly.io/install.sh | sh

# 2. Authenticate
fly auth login

# 3. Launch app (auto-detects Dockerfile)
fly launch --name insightdesk-ai

# 4. Deploy
fly deploy

# 5. Set environment variables (if needed)
fly secrets set NEO4J_URI=bolt://your-aura-instance.databases.neo4j.io
fly secrets set NEO4J_PASSWORD=your-password

# Your app is live at: https://insightdesk-ai.fly.dev
```

**Why Fly.io?**
- ✅ Auto-detects your Dockerfile
- ✅ Persistent storage for models/FAISS (3GB free)
- ✅ Simple `fly deploy` for updates
- ✅ PostgreSQL/Neo4j support
- ✅ Free tier is generous

---

### **Option 2: AWS EC2 (Free Tier)**

**Free tier:** t2.micro, 750 hours/month (always free)

```bash
# 1. Launch EC2 instance
# - AMI: Ubuntu 22.04 LTS
# - Instance type: t2.micro (free tier)
# - Storage: 30GB (free tier)

# 2. SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# 3. Install Docker & Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# 4. Clone your repo
git clone https://github.com/yourusername/insightdesk-ai.git
cd insightdesk-ai

# 5. Deploy
docker compose -f docker-compose.single.yml up -d

# 6. Set up reverse proxy (nginx)
sudo apt install nginx
# Configure nginx to forward :80 to :8000
sudo systemctl restart nginx

# Your app is live at: http://your-instance-ip
```

---

### **Option 3: Railway.app (Simple Alternative)**

**Free tier:** $5 credit/month

```bash
# 1. Connect GitHub repo to Railway
# - Push code to GitHub
# - Go to railway.app
# - Click "New Project" → Deploy from GitHub

# 2. Railway auto-detects Dockerfile
# 3. Generates public URL automatically
# Your app is live at: https://insightdesk-production.up.railway.app
```

---

### **Option 4: Render.com (Free with limitations)**

**Free tier:** Sleeps after 15 mins inactivity (cold start ~30s on wake)

```bash
# 1. Connect GitHub repo
# 2. Create "Web Service"
# 3. Set build command: (leave default)
# 4. Render auto-deploys on push
```

---

## Deployment Comparison

| Service | Free Tier | Cold Start | Persistence | Best For |
|---------|-----------|-----------|------------|----------|
| **Fly.io** | 3 VMs + 3GB | ~2-3s | ✅ 3GB storage | ⭐ **Recommended** |
| **AWS EC2** | t2.micro | ~1s | ✅ 30GB | Large models, control |
| **Railway** | $5/month | ~1s | ❌ Ephemeral | Quick deployment |
| **Render** | Free (limited) | ~30s (cold) | ❌ Ephemeral | Demo projects |
| **Vercel** | ✅ Unlimited | <100ms | ✅ For static | Frontend only (not API) |

---

## Environment Variables

Create `.env` file or set in cloud dashboard:

```bash
# Neo4j (optional - for Graph-RAG features)
NEO4J_URI=bolt://your-neo4j-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j

# Optional: LLM integrations (if you add them later)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

---

## File Structure in Container

```
/app
├── frontend/
│   ├── dist/              ← Built React app (served by FastAPI)
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── src/
│   ├── api/
│   │   └── main.py        ← Serves frontend + API
│   ├── agentic/
│   ├── anomaly/
│   ├── retrieval/
│   └── ...
├── data/                  ← Persistent storage
├── models/                ← ML models
├── vector_store/          ← FAISS indices
└── requirements-prod.txt
```

---

## Troubleshooting

### **"Frontend dist not built"**
```bash
cd frontend
npm run build
cd ..
docker build -t insightdesk-ai .
```

### **"Port 8000 already in use"**
```bash
# Use different port
docker run -p 9000:8000 insightdesk-ai:latest
# Access at: http://localhost:9000
```

### **"Neo4j connection fails"**
- Ensure `NEO4J_URI` is correct (use Aura or local instance)
- Check network connectivity from container to Neo4j
- Verify credentials match

### **"Models too large for container"**
- Use AWS S3 / Cloud Storage: download models at startup
- Or run on larger machine: `fly scale compute performance`

---

## Next Steps

### 1️⃣ **Local Testing**
```bash
docker compose -f docker-compose.single.yml up --build
# Visit http://localhost:8000
```

### 2️⃣ **Deploy to Fly.io (5 min)**
```bash
fly launch --name insightdesk-ai
fly deploy
# Your app is live!
```

### 3️⃣ **Connect Custom Domain** (optional)
```bash
fly certs create yourdomain.com
# Follow DNS setup instructions
```

### 4️⃣ **Monitor & Logs**
```bash
fly logs          # View live logs
fly ssh console   # SSH into container
fly monitoring    # View metrics
```

---

## Summary

✅ **Single Docker container** serves frontend + API  
✅ **No nginx reverse proxy needed** (FastAPI handles routing)  
✅ **Deployment to Fly.io in <5 minutes**  
✅ **Free tier: $0/month** (Fly.io + Neo4j Aura)  
✅ **Scalable:** Upgrade to paid tier as traffic grows  

**Recommended flow:**
1. Test locally: `docker compose -f docker-compose.single.yml up`
2. Push to GitHub
3. Deploy to Fly.io: `fly launch && fly deploy`
4. Done! 🚀

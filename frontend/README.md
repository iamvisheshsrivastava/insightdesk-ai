# Frontend (Vite + React)

Quick lightweight frontend for the InsightDesk demo. Uses Vite and React and proxies `/api` to `http://localhost:8000` during development.

Run locally:

```bash
cd frontend
npm install
npm run dev
```

Build for production:

```bash
npm run build
# serve `dist/` with nginx or mount into FastAPI StaticFiles
```

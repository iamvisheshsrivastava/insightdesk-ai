# Contributing to InsightDesk AI

Thanks for your interest in contributing! InsightDesk AI is a FastAPI + React
platform with a few ML models (XGBoost for classification, FAISS/fastembed
for retrieval) underneath it. This guide covers how to get set up, run the
tests, and submit a change.

## Project layout

```
insightdesk-ai/
├── src/
│   ├── api/          # FastAPI app and routes (src/api/main.py)
│   ├── models/       # Classifier logic
│   ├── retrieval/    # RAG pipeline (FAISS + embeddings)
│   ├── anomaly/       # Anomaly detection
│   ├── monitoring/    # Drift detection & performance metrics
│   └── feedback/      # Feedback collection and management
├── frontend/           # React + Vite + Tailwind dashboard
├── scripts/            # Training, demo, and maintenance scripts
├── tests/              # pytest suite for the backend
└── models/              # Trained model artifacts (.pkl / .joblib)
```

## Development setup

### Backend (FastAPI)

```bash
git clone https://github.com/iamvisheshsrivastava/insightdesk-ai.git
cd insightdesk-ai

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements-dev.txt   # installs requirements-prod.txt plus test/lint tooling
```

Copy `.env.example` to `.env` and adjust values if you need Neo4j/S3/MLflow
features — the API runs fine without any of these for day-to-day frontend or
API work, since the ML/graph integrations degrade gracefully when the
relevant service isn't configured.

Run the API locally:

```bash
uvicorn src.api.main:app --reload
```

The backend serves on `http://localhost:8000`; interactive API docs are at
`http://localhost:8000/docs`.

### Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

The dev server runs on `http://localhost:3000` and proxies `/api` requests to
the backend, so run both at once while working on UI changes that hit live
endpoints.

## Running tests

Backend tests use `pytest` and live in `tests/`:

```bash
# Full suite with coverage
pytest --cov=src --cov-report=term-missing tests/

# A single file or test
pytest tests/test_api.py
pytest tests/test_anomaly.py -k test_volume_spike_detection -v
```

You can also use the provided `Makefile` shortcuts:

```bash
make test          # full suite with coverage
make test-unit      # unit-marked tests only
make lint            # black --check, isort --check, flake8
make format          # auto-format with black + isort
make security        # bandit + safety
```

There is no automated frontend test suite yet — if you touch
`frontend/src`, please manually verify the affected page(s) in the browser
(`npm run dev`) and check the browser console for errors.

### Linting and formatting

The project uses `black`, `isort`, and `flake8` (configured in
`pyproject.toml` and `.flake8`). Before opening a PR:

```bash
black .
isort .
flake8 .
```

`make lint` runs all three in check-only mode, matching what CI enforces.

## Making a change

1. **Open an issue first** for anything beyond a trivial fix, so we can agree
   on the approach before you invest time. For small, well-scoped bugs
   (typos, clearly broken behavior), a PR without a prior issue is fine.
2. **Fork and branch** from `main` using a descriptive branch name, e.g.
   `fix/anomalies-critical-severity` or `feat/analytics-date-filter`.
3. **Keep PRs focused.** One logical change per PR — it's much easier to
   review and revert if needed. Avoid bundling unrelated refactors with a
   bug fix.
4. **Add or update tests** for any behavior change in `src/`. New backend
   endpoints or logic should have corresponding tests in `tests/`.
5. **Run `make lint` and `make test`** (or the equivalent `pytest`/`black`/
   `flake8` commands above) locally before pushing. CI runs the same checks
   and will block merges on failures.
6. **Write a clear PR description**: what changed, why, and how you verified
   it (commands run, screenshots for UI changes, etc.).
7. A maintainer will review, may ask for changes, and will merge once the
   PR is approved and CI is green.

## Finding something to work on

Issues labeled [`good first issue`](https://github.com/iamvisheshsrivastava/insightdesk-ai/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
are scoped to be approachable without deep familiarity with the codebase.
[`help wanted`](https://github.com/iamvisheshsrivastava/insightdesk-ai/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
issues are open to anyone and may be larger in scope.

## Code style notes

- Backend: standard `black`-formatted Python, type hints where practical,
  Pydantic models for request/response schemas (see `src/feedback/feedback_models.py`
  and `src/anomaly/anomaly_models.py` for examples of the existing pattern).
- Frontend: functional React components with hooks, Tailwind utility
  classes for styling, `lucide-react` for icons. Follow the fetch/error/
  loading-state pattern already used in `frontend/src/pages/Feedback.jsx`
  (`loading` / `error` / success state via `useState`) when adding new data
  fetching to a page.

## Questions

If anything here is unclear or you're unsure whether an idea fits the
project, open an issue describing what you'd like to do — that's the best
way to get quick feedback before writing code.

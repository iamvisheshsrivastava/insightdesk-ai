# InsightDesk AI

InsightDesk AI is an intelligent support-operations platform for classifying incoming tickets, retrieving relevant solutions with RAG, monitoring model behavior, and collecting feedback from support teams. The repository includes a FastAPI backend, a Streamlit dashboard, model training scripts, and evaluation utilities.

## Main components

- FastAPI API for prediction, retrieval, monitoring, and feedback workflows
- Streamlit dashboard for ticket triage and operational visibility
- XGBoost and TensorFlow ticket classifiers
- retrieval pipelines for solution search, including graph-oriented modules
- anomaly detection, drift monitoring, and feedback-loop utilities
- training, benchmarking, and launch scripts under `scripts/`

## Repository layout

```text
insightdesk-ai/
|-- src/
|   |-- api/
|   |-- models/
|   |-- retrieval/
|   |-- monitoring/
|   |-- feedback/
|   `-- ...
|-- scripts/
|-- tests/
|-- app.py
|-- demo_dashboard.py
|-- Makefile
|-- Dockerfile
|-- docker-compose.yml
`-- README.md
```

## Quickstart

### 1. Install dependencies

```bash
git clone https://github.com/iamvisheshsrivastava/insightdesk-ai.git
cd insightdesk-ai
python -m venv venv
```

PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

### 2. Prepare data and artifacts

```bash
python scripts/unzip_and_load.py
python scripts/build_features.py
python scripts/train_and_compare_models.py
python scripts/build_rag_index.py
```

### 3. Run the API

```bash
cd src
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Run the dashboard

From the repository root in a second terminal:

```bash
streamlit run app.py --server.port 8501
```

### 5. Optional shortcuts

```bash
make lint
make test
make dashboard
make full-stack
```

## Key endpoints

- `GET /health`
- `POST /predict/category`
- `POST /retrieve/solutions`
- `GET /anomalies/recent`
- `GET /monitoring/status`
- `GET /feedback/stats`

Interactive API docs are available at `http://localhost:8000/docs`.

## Status

This is a working prototype and experimentation repo. Some of the more advanced graph, monitoring, and feedback integrations depend on local services or optional infrastructure to be fully enabled.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

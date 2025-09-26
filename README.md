# insightdesk-ai

InsightDesk AI is an intelligent support platform in Python that classifies and prioritizes tickets, retrieves solutions with hybrid RAG + Graph-RAG, detects anomalies in real time, and learns from agent feedback. The system is fully containerized, supports model versioning and monitoring, and provides reproducible ML pipelines.

---

## 🚀 Features

* **Ticket Categorization & Prioritization**
  Multi-model ML (XGBoost / TensorFlow) for category and priority prediction.
* **Hybrid RAG + Graph-RAG Retrieval**
  Combines semantic search, keyword matching, and graph relationships to surface the best solutions.
* **Real-time Anomaly Detection**
  Detects emerging problems, unusual ticket volume patterns, and sentiment shifts.
* **Continuous Learning**
  Captures agent corrections and customer feedback to improve models over time.
* **Production-Ready Engineering**
  Containerized deployment, model versioning, monitoring, and drift detection.

---

## 🗂 Project Structure (suggested)

```
insightdesk-ai/
├─ data/                 # raw and processed datasets
├─ src/
│  ├─ ingestion/         # JSON ingestion & preprocessing
│  ├─ features/          # feature engineering & feature store
│  ├─ models/            # ML models (XGBoost, TensorFlow)
│  ├─ retrieval/         # RAG + Graph-RAG logic
│  ├─ anomaly/           # anomaly detection module
│  ├─ api/               # FastAPI endpoints
│  └─ utils/             # shared helpers, logging, configs
├─ tests/                # unit and integration tests
├─ notebooks/            # experiments & model evaluation
├─ docker/               # Dockerfiles, docker-compose
├─ requirements.txt
├─ LICENSE
└─ README.md
```

---

## ⚡ Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/insightdesk-ai.git
   cd insightdesk-ai
   ```

2. **Set up Python environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate     # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the FastAPI service**

   ```bash
   uvicorn src.api.main:app --reload
   ```

4. **API test example**

   ```bash
   curl -X POST http://localhost:8000/tickets \
        -H "Content-Type: application/json" \
        -d '{"ticket_id": "TK-2025-000001", "description": "Sync error ..."}'
   ```

---

## 🧰 Tech Stack

* **Python 3.12+**, FastAPI, Pydantic
* **Machine Learning**: scikit-learn, XGBoost, TensorFlow/Keras
* **Retrieval**: FAISS or Milvus, Neo4j/NetworkX (for Graph-RAG)
* **MLOps**: MLflow for experiment tracking & model registry
* **Containerization**: Docker & docker-compose
* **Monitoring**: Prometheus / Grafana (optional)

---

## 📊 Data

Use the provided JSON file with ~100k historical support tickets.
Recommended split: 70 % train / 15 % validation / 15 % test.
Handle class imbalance (e.g., stratified sampling or resampling) and document your strategy.

---

## 🥪 Development & Testing

* Format code with `black` and `isort`.
* Run unit tests:

  ```bash
  pytest
  ```

---

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions are welcome!
Please fork the repository and submit a pull request with a clear description of changes.

---

## 📧 Contact

**Author:** Vishesh Srivastava
[LinkedIn](https://linkedin.com/in/iamvisheshsrivastava) • [GitHub](https://github.com/iamvisheshsrivastava)

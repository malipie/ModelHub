# ML Model Serving Platform

> Production-ready platform for ML model deployment with A/B testing,
> automated drift detection, and CI/CD pipeline.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GitHub Actions CI/CD                         │
│  [push] → lint → test → build Docker → run Evidently → deploy      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Docker Compose Stack                           │
│                                                                     │
│  ┌──────────┐    ┌───────────────────────────────────────────┐      │
│  │  nginx   │───▶│  FastAPI Serving (x2 instances)           │      │
│  │ (gateway)│    │  ┌─────────────┐  ┌──────────────────┐    │      │
│  │          │    │  │ Champion    │  │ Challenger        │    │      │
│  │ 80/20    │    │  │ model v1    │  │ model v2          │    │      │
│  └──────────┘    │  port: 8001   │  port: 8002          │    │      │
│                  └───────────────────────────────────────┘    │      │
│                                                               │      │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐   │      │
│  │ MLflow       │  │ PostgreSQL   │  │ Training Pipeline │   │      │
│  │ Tracking     │  │ (metadata +  │  │ (retrain on       │   │      │
│  │ Server       │  │  predictions)│  │  schedule/trigger) │  │      │
│  └──────────────┘  └──────────────┘  └───────────────────┘   │      │
│                                                               │      │
│  ┌──────────────┐  ┌──────────────────────────────────────┐   │      │
│  │ Evidently    │  │ Grafana + Prometheus                 │   │      │
│  │ Drift Monitor│  │ (API metrics, latency, drift score)  │   │      │
│  └──────────────┘  └──────────────────────────────────────┘   │      │
└─────────────────────────────────────────────────────────────────────┘
```

## Features

- **Champion/Challenger A/B testing** with configurable traffic split (80/20)
- **Automated data drift detection** (Evidently AI)
- **Full CI/CD pipeline** (GitHub Actions)
- **Real-time monitoring** (Grafana + Prometheus)
- **Model versioning and registry** (MLflow)
- **API Gateway with rate limiting** (nginx)
- **E-Commerce churn prediction** — binary classification with 22 features

## Quick Start

```bash
# 1. Install dependencies
make setup

# 2. Train models (dataset must be present at Data/raw/ecommerce_churn.csv)
make train

# 3. Inspect results in MLflow UI
make mlflow-ui   # → http://localhost:5000

# 4. Run tests
make test
```

_(Full Docker Compose stack coming in Phase 3)_

## Dataset

**E-Commerce Customer Churn** — 8,500 customers from 6 European countries (PL, DE, FR, UK, ES, NL)

| Metric | Value |
|--------|-------|
| Records | 8,500 |
| Features | 22 |
| Churn rate | 16.9% |
| Strongest predictor | `satisfaction_score_1_5` (corr: -0.35) |

Key features: `tenure_months`, `monthly_spend_eur`, `satisfaction_score_1_5`,
`subscription_type`, `support_tickets_last_month`, `returns_count_12m`

> The dataset is stored locally at `Data/raw/ecommerce_churn.csv` and is **not committed to git**.

## Project Structure

```
ModelHub/
├── Data/                            # Raw dataset (gitignored)
│   └── raw/
│       ├── ecommerce_churn.csv
│       └── ecommerce_churn_metadata.json
├── training/                        # Training pipeline
│   ├── src/
│   │   ├── data_loader.py           # EcommerceChurnDataLoader + Pandera validation
│   │   ├── feature_engineering.py   # sklearn ColumnTransformer pipeline
│   │   ├── train.py                 # Orchestrator: 3 models + MLflow logging
│   │   ├── evaluate.py              # Metrics, confusion matrix, ROC/PR curves
│   │   ├── register_model.py        # MLflow Model Registry promotion
│   │   └── utils.py
│   ├── config/
│   │   └── base_config.yaml         # Hyperparameters, feature lists, thresholds
│   ├── notebooks/
│   │   └── 01_eda.ipynb             # Exploratory Data Analysis
│   ├── tests/                       # 41 unit tests
│   └── Dockerfile
├── serving/                         # FastAPI serving (Phase 2)
├── monitoring/                      # Evidently + Grafana (Phase 4)
├── gateway/                         # nginx (Phase 3)
├── .github/workflows/               # CI/CD (Phase 5)
├── docker-compose.yml               # Full stack (Phase 3)
├── Makefile
└── .env.example
```

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make setup` | Create venv and install dependencies |
| `make train` | Train 3 models, log to MLflow |
| `make test` | Run all tests with coverage |
| `make lint` | Check black + ruff |
| `make format` | Auto-format code |
| `make mlflow-ui` | Launch MLflow UI at localhost:5000 |
| `make data-check` | Verify dataset is present and valid |
| `make clean` | Remove venv and caches |

## Training Results (Baseline)

| Model | AUC-ROC | F1 | Accuracy | Precision | Recall |
|-------|---------|----|----------|-----------|--------|
| LogisticRegression | 0.853 | 0.531 | 0.768 | 0.404 | **0.778** |
| RandomForest | 0.850 | 0.541 | 0.794 | 0.434 | 0.719 |
| **XGBoost** ✅ | **0.853** | **0.537** | **0.792** | **0.432** | 0.712 |

_Best model (XGBoost) registered as `ecommerce-churn-predictor` v1 (Production) in MLflow._

### Model selection rationale

No single model dominates — differences across all metrics are minimal (0.001–0.02 range).

For churn prediction, **recall is the most business-critical metric**: it measures how many actual churners the model caught. The cost of missing a customer who will leave (false negative) is higher than the cost of sending a retention offer to a customer who would have stayed anyway.

From that perspective, **LogisticRegression has the strongest recall (0.778)** — it catches the most churners, at the cost of lower precision. XGBoost was selected as champion based on AUC-ROC, which is a sound, defensible technical choice. In a real e-commerce deployment, this trade-off would be worth revisiting in favour of LogisticRegression.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| ML | scikit-learn, XGBoost |
| MLOps | MLflow 3.x (SQLite backend) |
| Monitoring | Evidently AI, Grafana, Prometheus |
| API | FastAPI, Uvicorn |
| Gateway | nginx |
| Database | PostgreSQL |
| CI/CD | GitHub Actions |
| Containers | Docker, Docker Compose |
| Validation | Pydantic v2, Pandera |

## Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 — Foundation | ✅ Done | Training pipeline + MLflow |
| 2 — API Serving | 🔄 Next | FastAPI + A/B testing |
| 3 — Infrastructure | ⏳ Planned | Docker Compose + nginx |
| 4 — Monitoring | ⏳ Planned | Grafana + Evidently |
| 5 — CI/CD | ⏳ Planned | GitHub Actions |
| 6 — Polish | ⏳ Planned | Demo + Documentation |

# ModelHub — ML Model Serving Platform

> End-to-end MLOps platform for training, serving, and monitoring a binary classification model in an A/B testing setup. Built as a portfolio project demonstrating production-grade ML engineering practices.

---

## Business Context

An e-commerce company wants to identify customers at risk of churning so it can trigger targeted retention campaigns before they leave. The model ingests 22 behavioural and transactional features (spending patterns, support tickets, engagement metrics, subscription type) and outputs a churn probability for each customer.

The platform supports **Champion/Challenger A/B testing**: the current best model (champion) serves 80% of traffic while a newer candidate (challenger) runs in parallel on the remaining 20%. This enables safe, data-driven model upgrades without a big-bang deployment.

---

## Architecture

```
  HTTP Clients
       │
       ▼
┌─────────────┐        weighted upstream (4:1)
│    nginx    │──────────────────────────────────────────┐
│  (gateway)  │                                          │
│  port: 80   │──────────────────────────┐               │
└─────────────┘                          │               │
     rate limiting, CORS,                │               │
     JSON access logs,               ┌──▼──────────┐  ┌──▼────────────┐
     X-Request-Id passthrough        │  serving    │  │  serving      │
                                     │  champion   │  │  challenger   │
                                     │  port: 8000 │  │  port: 8000   │
                                     └──────┬──────┘  └──────┬────────┘
                                            │                 │
                          FastAPI + Uvicorn │                 │
                          Prometheus /metrics                 │
                          PostgreSQL audit log                │
                                            │                 │
                                     ┌──────▼─────────────────▼────────┐
                                     │         MLflow Server            │
                                     │  model registry  port: 5001     │
                                     └──────────────┬──────────────────┘
                                                    │
                                             ┌──────▼──────┐
                                             │  PostgreSQL  │
                                             │  port: 5432  │
                                             └─────────────┘

  ┌──────────────────┐    ┌──────────────────────────────────┐
  │   Prometheus     │    │           Grafana                │
  │   port: 9090     │───▶│           port: 3000             │
  │  scrapes /metrics│    │  throughput, latency, churn rate │
  └──────────────────┘    └──────────────────────────────────┘
```

**Services summary:**

| Service | Image / Dockerfile | Role |
|---|---|---|
| `gateway` | `gateway/Dockerfile` (nginx 1.27) | Reverse proxy, A/B routing, rate limiting |
| `serving-champion` | `serving/Dockerfile` (FastAPI) | Champion model inference |
| `serving-challenger` | `serving/Dockerfile` (FastAPI) | Challenger model inference |
| `mlflow-server` | `mlflow-server/Dockerfile` | Experiment tracking + model registry |
| `postgres` | `postgres:16-alpine` | MLflow backend + prediction audit log |
| `prometheus` | `prom/prometheus:v2.52.0` | Metrics collection |
| `grafana` | `grafana/grafana:10.4.3` | Dashboards |

---

## Dataset

**E-Commerce Customer Churn** — 8,500 customers from 6 European countries (PL, DE, FR, UK, ES, NL).

| Attribute | Value |
|---|---|
| Records | 8,500 |
| Features | 22 |
| Churn rate | 16.9% |
| Strongest predictor | `satisfaction_score_1_5` (Pearson r = −0.35) |
| Target imbalance | 83% no-churn / 17% churn → SMOTE applied |

Key feature groups:
- **Spend & activity**: `tenure_months`, `monthly_spend_eur`, `total_spent_eur`, `avg_order_value_eur`, `purchase_frequency_per_month`
- **Engagement**: `website_sessions_per_month`, `email_engagement_rate_percent`, `cart_abandonment_rate_percent`
- **Support & satisfaction**: `support_tickets_last_month`, `satisfaction_score_1_5`, `returns_count_12m`
- **Categorical**: `subscription_type`, `country`, `account_age_category`, `preferred_category`

> The dataset is stored locally at `Data/raw/ecommerce_churn.csv` and is **not committed to git**.

---

## Training Results

Three models were trained on an 80/20 stratified split with 5-fold cross-validation. Class imbalance was addressed with SMOTE. All results below are on the held-out test set.

| Model | AUC-ROC | F1 | Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| LogisticRegression | 0.853 | 0.531 | 0.768 | 0.404 | **0.778** |
| RandomForest | 0.850 | 0.541 | 0.794 | 0.434 | 0.719 |
| **XGBoost** ✅ champion | **0.853** | **0.537** | **0.792** | **0.432** | 0.712 |

All three models were logged to MLflow and registered in the Model Registry. XGBoost was promoted to `Production` (champion) and LogisticRegression to `Staging` (challenger).

---

## Results & Conclusions

### Model performance

The three models reach virtually identical AUC-ROC (0.850–0.853), which tells us the dataset difficulty is the binding constraint — not the choice of algorithm. No model dramatically outperforms the others, which is common for well-engineered tabular datasets at this scale.

**XGBoost** was selected as champion on the basis of balanced F1 + AUC. It has a slight edge in precision and accuracy. However, from a strict business perspective, **LogisticRegression's recall of 0.778 is the standout result**: it correctly identifies 78% of all actual churners, at the cost of flagging more false positives. The cost asymmetry (a missed churner is far more expensive than an unnecessary retention offer) means this trade-off is worth reconsidering before a production rollout.

### Feature importance insights (EDA)

- `satisfaction_score_1_5` is the single strongest predictor (negative correlation): dissatisfied customers churn at much higher rates regardless of spend.
- Customers with high `support_tickets_last_month` are significantly more likely to churn — unresolved issues drive abandonment.
- `tenure_months` is protective: long-term customers have much lower churn probability even when other signals are negative.
- `cart_abandonment_rate_percent` above ~60% is a strong early warning signal.
- Customers on the `Basic` subscription plan churn at roughly 2.5× the rate of `Premium` subscribers.

### Platform conclusions

- The **Champion/Challenger setup works end-to-end**: traffic splits correctly, both models serve real-time predictions, and the model name is returned in every response so downstream analysis can separate results.
- **Prometheus + Grafana** dashboards give live visibility into throughput, latency percentiles (p50/p95/p99), churn rate per model, and error rate — the minimum instrumentation needed to detect model degradation or infrastructure issues without waiting for batch reports.
- **MLflow** provides full experiment reproducibility: every run stores parameters, metrics, the fitted pipeline, and the input schema. Re-creating any model from scratch takes one `make train`.
- **PostgreSQL audit log** captures every prediction (features + output + model version) for future drift analysis and regulatory audit trails.
- The main architectural trade-off made here is **SQLite-backed MLflow for local dev, PostgreSQL for production**. The serving containers load models from a `.pkl` fallback rather than the MLflow HTTP API, which sidesteps MLflow 3.x's strict Host-header security but couples the Docker image to a pre-exported model file. In a full production setup this would be replaced by an S3/GCS artifact store with IAM-authenticated SDK calls.

---

## Quick Start (Docker)

```bash
# 1. Train models and export to serving/models/
make setup
make train
make export-models   # writes serving/models/champion.pkl + challenger.pkl

# 2. Build Docker images
make docker-build

# 3. Start the full production stack (nginx gateway on :80)
make docker-up

# 4. Or start the dev stack (services exposed directly, no nginx)
make docker-up-dev
```

**Endpoints after `docker-up`:**

| URL | Description |
|---|---|
| `POST http://localhost:80/predict` | A/B routed prediction (80% champion / 20% challenger) |
| `POST http://localhost:80/predict/champion` | Champion only |
| `POST http://localhost:80/predict/challenger` | Challenger only |
| `GET  http://localhost:80/health` | Health check |
| `GET  http://localhost:80/model` | Loaded model info |
| `GET  http://localhost:80/ab` | Current A/B split config |
| `GET  http://localhost:5001` | MLflow UI |
| `GET  http://localhost:9090` | Prometheus |
| `GET  http://localhost:3000` | Grafana (admin / modelhub) |

**Simulate traffic:**

```bash
make simulate-traffic   # 500 requests, concurrency=5, through nginx on :80
```

---

## API Usage

**Request:**

```bash
curl -s -X POST http://localhost:80/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 12,
    "monthly_spend_eur": 89.50,
    "total_spent_eur": 1050.00,
    "avg_order_value_eur": 65.0,
    "purchase_frequency_per_month": 2.5,
    "num_product_categories": 3,
    "support_tickets_last_month": 2,
    "website_sessions_per_month": 15,
    "cart_abandonment_rate_percent": 45.0,
    "email_engagement_rate_percent": 30.0,
    "reviews_left_count": 4,
    "returns_count_12m": 1,
    "last_purchase_days_ago": 14,
    "satisfaction_score_1_5": 3.2,
    "loyalty_program_member": 1,
    "payment_methods_used": 2,
    "country": "Poland",
    "account_age_category": "1-2 years",
    "subscription_type": "Standard",
    "preferred_category": "Electronics"
  }'
```

**Response:**

```json
{
  "request_id": "a3f1c2d4-...",
  "prediction": 0,
  "churn_probability": 0.187,
  "model_name": "champion",
  "model_version": "1",
  "latency_ms": 4.2
}
```

---

## Local Development (without Docker)

```bash
make setup            # create .venv, install training deps
make train            # train 3 models, log to MLflow
make mlflow-ui        # → http://localhost:5001
make test             # run 41 unit tests with coverage
make lint             # black --check + ruff
make format           # auto-format

make setup-serving    # install serving deps
make serve            # FastAPI on :8000 (dev mode, SQLite)
make test-serving     # serving unit tests
```

---

## Project Structure

```
ModelHub/
├── training/
│   ├── src/
│   │   ├── data_loader.py           # EcommerceChurnDataLoader + Pandera schema
│   │   ├── feature_engineering.py   # ColumnTransformer pipeline (numeric + categorical)
│   │   ├── train.py                 # Orchestrator: SMOTE, 3 models, MLflow logging
│   │   ├── evaluate.py              # Metrics, confusion matrix, ROC/PR curves
│   │   ├── register_model.py        # MLflow Model Registry promotion
│   │   └── utils.py
│   ├── config/base_config.yaml      # Hyperparameters, feature lists, thresholds
│   ├── notebooks/01_eda.ipynb       # EDA with embedded static charts
│   ├── tests/                       # 41 unit tests
│   └── requirements.txt
├── serving/
│   ├── src/
│   │   ├── main.py                  # FastAPI app, lifespan, /predict endpoints
│   │   ├── model_loader.py          # MLflow registry loader with .pkl fallback
│   │   ├── models.py                # Pydantic v2 request/response schemas
│   │   ├── metrics.py               # Prometheus counters and histograms
│   │   └── database.py              # Async PostgreSQL prediction audit logger
│   ├── models/                      # champion.pkl / challenger.pkl (gitignored)
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── gateway/
│   ├── nginx.conf                   # Weighted upstream, rate limiting, CORS, JSON logs
│   └── Dockerfile
├── mlflow-server/
│   └── Dockerfile                   # MLflow 3.x + psycopg2 on python:3.11-slim
├── monitoring/
│   ├── prometheus/prometheus.yml    # Scrape configs for both serving instances
│   └── grafana/provisioning/        # Auto-provisioned datasource + dashboard
│       ├── datasources/
│       └── dashboards/
│           └── api_metrics.json     # 7-panel dashboard (throughput, latency, churn rate)
├── scripts/
│   ├── simulate_traffic.py          # Load generator (configurable concurrency + n)
│   └── init_db.sql                  # PostgreSQL schema init
├── docker-compose.yml               # Production stack (nginx gateway)
├── docker-compose.dev.yml           # Dev stack (services exposed directly)
├── Makefile
└── .gitignore
```

---

## Makefile Reference

| Command | Description |
|---|---|
| `make setup` | Create `.venv`, install training deps |
| `make train` | Train 3 models, log all runs to MLflow |
| `make test` | 41 unit tests with coverage |
| `make lint` / `make format` | black + ruff check / fix |
| `make mlflow-ui` | Launch MLflow UI at localhost:5001 |
| `make data-check` | Validate dataset presence and schema |
| `make setup-serving` | Install serving deps |
| `make serve` | Run FastAPI locally (port 8000, SQLite) |
| `make test-serving` | Serving unit tests |
| `make export-models` | Export MLflow models to `serving/models/*.pkl` |
| `make docker-build` | Build all Docker images |
| `make docker-up` | Start production stack (nginx on :80) |
| `make docker-up-dev` | Start dev stack (services on direct ports) |
| `make docker-down` / `docker-down-dev` | Stop and remove containers |
| `make docker-logs` | Follow logs from production stack |
| `make simulate-traffic` | Send 500 test requests through the gateway |
| `make clean` | Remove venv and caches |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| MLOps | MLflow 3.x |
| API | FastAPI, Uvicorn, Pydantic v2 |
| Gateway | nginx 1.27 |
| Database | PostgreSQL 16, psycopg2 |
| Monitoring | Prometheus, Grafana |
| Validation | Pandera |
| Containers | Docker, Docker Compose |

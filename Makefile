.PHONY: help setup install train test lint format clean mlflow-ui \
        setup-serving test-serving lint-serving format-serving serve \
        docker-up docker-up-dev docker-down docker-down-dev docker-logs docker-ps \
        docker-build simulate-traffic export-models

PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
BLACK := $(VENV)/bin/black
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy
MLFLOW := $(VENV)/bin/mlflow
TRAINING_PYTHON := $(VENV)/bin/python

help: ## Show this help message
	@echo "ML Model Serving Platform — Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

$(VENV)/bin/activate: training/requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r training/requirements.txt

setup: $(VENV)/bin/activate ## Create virtual environment and install dependencies
	@echo "✅ Virtual environment ready at $(VENV)/"

install: setup ## Alias for setup

train: setup ## Run the full training pipeline (trains 3 models, logs to MLflow)
	@echo "🚀 Starting training pipeline..."
	$(TRAINING_PYTHON) -m training.src.train
	@echo "✅ Training complete. Run 'make mlflow-ui' to inspect results."

test: setup ## Run all tests with coverage
	@echo "🧪 Running tests..."
	$(PYTEST) training/tests/ -v --cov=training/src --cov-report=term-missing
	@echo "✅ All tests passed."

lint: setup ## Run linters (black --check + ruff)
	@echo "🔍 Running linters..."
	$(BLACK) --check --line-length=100 training/src/ training/tests/
	$(RUFF) check training/src/ training/tests/
	@echo "✅ Lint passed."

format: setup ## Auto-format code (black + ruff --fix)
	@echo "✏️  Formatting code..."
	$(BLACK) --line-length=100 training/src/ training/tests/
	$(RUFF) check --fix training/src/ training/tests/

mlflow-ui: ## Launch MLflow UI at http://localhost:5001
	@echo "🌐 Starting MLflow UI at http://localhost:5001 ..."
	$(MLFLOW) ui --backend-store-uri sqlite:///mlruns.db --host 0.0.0.0 --port 5001

clean: ## Remove virtual environment, caches, and generated files
	rm -rf $(VENV) __pycache__ .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "🧹 Cleaned."

data-check: setup ## Verify that the E-Commerce Churn dataset is present
	@echo "🔍 Checking dataset..."
	$(TRAINING_PYTHON) -c "from training.src.data_loader import EcommerceChurnDataLoader; EcommerceChurnDataLoader().load(validate=True); print('Dataset OK')"

# ── Serving (Phase 2) ────────────────────────────────────────────────────────

$(VENV)/bin/uvicorn: serving/requirements.txt $(VENV)/bin/activate
	$(PIP) install -r serving/requirements.txt

setup-serving: $(VENV)/bin/uvicorn ## Install serving dependencies into the shared venv
	@echo "✅ Serving dependencies installed."

test-serving: setup-serving ## Run serving tests with coverage
	@echo "🧪 Running serving tests..."
	$(PYTEST) serving/tests/ -v --cov=serving/src --cov-report=term-missing \
		--tb=short
	@echo "✅ All serving tests passed."

lint-serving: setup-serving ## Lint serving code (black --check + ruff)
	@echo "🔍 Linting serving code..."
	$(BLACK) --check --line-length=100 serving/src/ serving/tests/
	$(RUFF) check serving/src/ serving/tests/
	@echo "✅ Serving lint passed."

format-serving: setup-serving ## Auto-format serving code
	@echo "✏️  Formatting serving code..."
	$(BLACK) --line-length=100 serving/src/ serving/tests/
	$(RUFF) check --fix serving/src/ serving/tests/

serve: setup-serving ## Run the FastAPI serving app locally (port 8000)
	@echo "🚀 Starting serving API at http://localhost:8000 ..."
	@echo "   Swagger UI → http://localhost:8000/docs"
	MLFLOW_TRACKING_URI=sqlite:///mlruns.db \
	$(VENV)/bin/uvicorn serving.src.main:app --host 0.0.0.0 --port 8000 --reload

# ── Docker Compose (Phase 3) ─────────────────────────────────────────────────

docker-build: ## Build all Docker images
	@echo "🔨 Building Docker images..."
	docker compose build

docker-up: ## Start the full production stack (nginx + all services)
	@echo "🚀 Starting production stack..."
	docker compose up -d
	@echo "✅ Stack running."
	@echo "   Gateway    → http://localhost:80"
	@echo "   MLflow UI  → http://localhost:5001"
	@echo "   Prometheus → http://localhost:9090"
	@echo "   Grafana    → http://localhost:3000  (admin / modelhub)"

docker-up-dev: ## Start the dev stack (no nginx, services exposed directly)
	@echo "🚀 Starting dev stack..."
	docker compose -f docker-compose.dev.yml up -d
	@echo "✅ Dev stack running."
	@echo "   Champion   → http://localhost:8000"
	@echo "   Challenger → http://localhost:8001"
	@echo "   MLflow UI  → http://localhost:5001"
	@echo "   Prometheus → http://localhost:9090"
	@echo "   Grafana    → http://localhost:3000  (admin / modelhub)"

docker-down: ## Stop and remove production stack containers
	docker compose down

docker-down-dev: ## Stop and remove dev stack containers
	docker compose -f docker-compose.dev.yml down

docker-logs: ## Follow logs from all production stack containers
	docker compose logs -f

docker-ps: ## Show status of all production stack containers
	docker compose ps

simulate-traffic: ## Send 500 test requests to the gateway (requires running stack)
	@echo "🔫 Simulating traffic → http://localhost:80 ..."
	$(VENV)/bin/python scripts/simulate_traffic.py --url http://localhost:80 --n 500 --concurrency 5

export-models: setup-serving ## Export trained models from MLflow registry to serving/models/*.pkl (required before docker-build)
	@echo "📦 Exporting models from MLflow registry to serving/models/ ..."
	@mkdir -p serving/models
	$(VENV)/bin/python - <<'EOF'
import mlflow, pickle, os, sys
mlflow.set_tracking_uri("sqlite:///mlruns.db")
MODEL_NAME = os.getenv("MODEL_NAME", "ecommerce-churn-predictor")
for stage, role in [("Production", "champion"), ("Staging", "challenger")]:
    try:
        m = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{stage}")
        path = f"serving/models/{role}.pkl"
        with open(path, "wb") as f:
            pickle.dump(m, f)
        print(f"  ✅ {role} ({stage}) → {path}  ({os.path.getsize(path)//1024} KB)")
    except Exception as e:
        if role == "champion":
            print(f"  ❌ {role} FAILED: {e}", file=sys.stderr); sys.exit(1)
        else:
            print(f"  ⚠️  {role} not found (no Staging model) — skipping")
EOF
	@echo "✅ Models exported. Run 'make docker-build' next."

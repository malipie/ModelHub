"""Main training pipeline.

Usage:
    python -m training.src.train                          # uses base_config.yaml
    python -m training.src.train --config path/to/cfg.yaml

Trains LogisticRegression, RandomForest, and XGBoost on the Churn dataset,
logs everything to MLflow, and registers the best model.
"""

import argparse
import logging
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from training.src.data_loader import EcommerceChurnDataLoader
from training.src.evaluate import MetricsDict, compare_models, evaluate_model
from training.src.feature_engineering import build_preprocessing_pipeline, prepare_data
from training.src.register_model import register_best_model
from training.src.utils import get_project_root, load_config, setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Estimator factory
# ---------------------------------------------------------------------------


def _build_estimators(config: dict[str, Any]) -> dict[str, Any]:
    """Instantiate all enabled estimators from config.

    Args:
        config: Parsed base_config.yaml.

    Returns:
        Ordered dict mapping human-readable name → sklearn estimator.
    """
    models_cfg = config["models"]
    estimators: dict[str, Any] = {}

    if models_cfg["logistic_regression"]["enabled"]:
        estimators["LogisticRegression"] = LogisticRegression(
            **models_cfg["logistic_regression"]["params"]
        )

    if models_cfg["random_forest"]["enabled"]:
        estimators["RandomForest"] = RandomForestClassifier(**models_cfg["random_forest"]["params"])

    if models_cfg["xgboost"]["enabled"]:
        estimators["XGBoost"] = XGBClassifier(
            **models_cfg["xgboost"]["params"],
            verbosity=0,  # suppress XGBoost console output
        )

    return estimators


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------


def train_single_model(
    name: str,
    estimator: Any,
    preprocessor: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    experiment_name: str,
    config: dict[str, Any],
) -> tuple[str, MetricsDict, str]:
    """Train one model inside its own MLflow run.

    Args:
        name: Human-readable model name.
        estimator: Unfitted sklearn estimator.
        preprocessor: Unfitted ColumnTransformer.
        X_train / y_train: Training split.
        X_test / y_test: Test split.
        experiment_name: MLflow experiment name.
        config: Full parsed config.

    Returns:
        Tuple of (run_id, metrics_dict, model_artifact_uri).
    """
    with mlflow.start_run(run_name=name, nested=True) as run:
        run_id = run.info.run_id
        logger.info("--- Training %s (run_id=%s) ---", name, run_id)

        # Log config metadata
        mlflow.set_tag("model_type", name)
        mlflow.set_tag("dataset", "telco_churn")
        mlflow.log_params(config["models"][_cfg_key(name)]["params"])
        mlflow.log_param("test_size", config["data"]["test_size"])
        mlflow.log_param("random_seed", config["data"]["random_seed"])

        # Build and fit the full pipeline (preprocessor + estimator)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)

        # Evaluate
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            metrics = evaluate_model(pipeline, X_test, y_test, name, artifact_dir)

        # Log the model artifact
        model_uri = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=None,  # we register manually in register_best_model
        ).model_uri

        logger.info("%s → AUC-ROC: %.4f, F1: %.4f", name, metrics["auc_roc"], metrics["f1"])
        return run_id, metrics, model_uri


def _cfg_key(name: str) -> str:
    """Map display name → config YAML key."""
    mapping = {
        "LogisticRegression": "logistic_regression",
        "RandomForest": "random_forest",
        "XGBoost": "xgboost",
    }
    return mapping.get(name, name.lower())


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_training(config_path: str | Path) -> None:
    """Execute the full training pipeline.

    1. Load config and data.
    2. Split into train/test.
    3. Train each enabled model in a separate MLflow run.
    4. Compare models and register the best one.

    Args:
        config_path: Path to the YAML config file.
    """
    config = load_config(config_path)
    root = get_project_root()

    # ------------------------------------------------------------------
    # MLflow setup
    # ------------------------------------------------------------------
    tracking_uri = config["experiment"].get("tracking_uri", "sqlite:///mlruns.db")
    if "://" not in tracking_uri:
        # Plain relative path — convert to absolute so MLflow stores data in the project root
        tracking_uri = str(root / tracking_uri)
    elif tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        # Relative sqlite path (e.g. sqlite:///mlruns.db) → make absolute
        db_file = tracking_uri[len("sqlite:///") :]
        tracking_uri = f"sqlite:///{root / db_file}"
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = config["experiment"]["name"]
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow tracking URI: %s", tracking_uri)
    logger.info("Experiment: %s", experiment_name)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    loader = EcommerceChurnDataLoader(raw_path=config["data"]["raw_path"])
    df = loader.load(validate=True)

    X, y = prepare_data(df, config)

    test_size: float = config["data"]["test_size"]
    seed: int = config["data"]["random_seed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    logger.info(
        "Split: %d train / %d test (stratified, test_size=%.0f%%).",
        len(X_train),
        len(X_test),
        test_size * 100,
    )

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    cat_cols = config["features"]["categorical_columns"]
    num_cols = config["features"]["numeric_columns"]

    # Filter to columns actually present in X (safety guard)
    cat_cols = [c for c in cat_cols if c in X.columns]
    num_cols = [c for c in num_cols if c in X.columns]

    # ------------------------------------------------------------------
    # Train all models
    # ------------------------------------------------------------------
    estimators = _build_estimators(config)
    results: dict[str, MetricsDict] = {}
    run_ids: dict[str, str] = {}
    model_uris: dict[str, str] = {}

    with mlflow.start_run(run_name="training_session"):
        mlflow.set_tag("pipeline_version", "1.0")
        mlflow.log_artifact(str(config_path), artifact_path="config")

        for name, estimator in estimators.items():
            # Each model gets a fresh unfitted preprocessor clone
            preprocessor = build_preprocessing_pipeline(cat_cols, num_cols)

            run_id, metrics, model_uri = train_single_model(
                name=name,
                estimator=estimator,
                preprocessor=preprocessor,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                experiment_name=experiment_name,
                config=config,
            )
            results[name] = metrics
            run_ids[name] = run_id
            model_uris[name] = model_uri

    # ------------------------------------------------------------------
    # Select and register best model
    # ------------------------------------------------------------------
    primary_metric = config["registration"].get("primary_metric", "auc_roc")
    best_name = compare_models(results, primary_metric)
    best_run_id = run_ids[best_name]
    best_model_uri = model_uris[best_name]

    register_best_model(
        run_id=best_run_id,
        model_uri=best_model_uri,
        model_name=config["registration"]["model_name"],
        metrics=results[best_name],
        promotion_threshold_auc=config["registration"]["promotion_threshold_auc"],
        primary_metric=primary_metric,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("\n%s", "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    for name, m in results.items():
        logger.info(
            "  %-22s AUC-ROC=%.4f  F1=%.4f  Accuracy=%.4f",
            name,
            m["auc_roc"],
            m["f1"],
            m["accuracy"],
        )
    logger.info("  Best model: %s", best_name)
    logger.info("  Registered as: %s", config["registration"]["model_name"])
    logger.info("  MLflow UI: mlflow ui --port 5000")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Churn prediction training pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=str(get_project_root() / "training" / "config" / "base_config.yaml"),
        help="Path to YAML config file (default: training/config/base_config.yaml)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = _parse_args()
    run_training(args.config)

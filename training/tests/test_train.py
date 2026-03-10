"""Smoke tests for the training pipeline.

These tests verify that the pipeline runs end-to-end on a small synthetic
sample without requiring internet access, MLflow server, or GPU.
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from training.src.data_loader import EcommerceChurnDataLoader
from training.src.evaluate import compare_models, compute_metrics
from training.src.feature_engineering import (
    build_preprocessing_pipeline,
    prepare_data,
)
from training.src.utils import get_project_root, load_config


@pytest.fixture(scope="module")
def config() -> dict:
    root = get_project_root()
    return load_config(root / "training" / "config" / "base_config.yaml")


@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    """50-row synthetic dataset — fast for smoke tests."""
    loader = EcommerceChurnDataLoader()
    return loader.load_synthetic(n_samples=200, random_state=1)


@pytest.fixture(scope="module")
def split_data(small_df: pd.DataFrame, config: dict):
    X, y = prepare_data(small_df, config)
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


@pytest.fixture(scope="module")
def fitted_pipeline(split_data, config: dict) -> Pipeline:
    """Train a LogisticRegression pipeline on the small dataset."""
    X_train, X_test, y_train, y_test = split_data
    cat_cols = [c for c in config["features"]["categorical_columns"] if c in X_train.columns]
    num_cols = [c for c in config["features"]["numeric_columns"] if c in X_train.columns]

    preprocessor = build_preprocessing_pipeline(cat_cols, num_cols)
    estimator = LogisticRegression(max_iter=200, random_state=42, class_weight="balanced")

    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
    pipeline.fit(X_train, y_train)
    return pipeline


class TestPipelineFit:
    def test_pipeline_fits_without_error(self, fitted_pipeline: Pipeline) -> None:
        assert fitted_pipeline is not None

    def test_predict_returns_binary(self, fitted_pipeline: Pipeline, split_data) -> None:
        _, X_test, _, _ = split_data
        preds = fitted_pipeline.predict(X_test)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_shape(self, fitted_pipeline: Pipeline, split_data) -> None:
        _, X_test, _, y_test = split_data
        proba = fitted_pipeline.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)

    def test_predict_proba_sums_to_one(self, fitted_pipeline: Pipeline, split_data) -> None:
        _, X_test, _, _ = split_data
        proba = fitted_pipeline.predict_proba(X_test)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X_test)), atol=1e-6)

    def test_predict_length_matches_input(self, fitted_pipeline: Pipeline, split_data) -> None:
        _, X_test, _, _ = split_data
        preds = fitted_pipeline.predict(X_test)
        assert len(preds) == len(X_test)


class TestComputeMetrics:
    def test_all_metrics_present(self, fitted_pipeline: Pipeline, split_data) -> None:
        _, X_test, _, y_test = split_data
        y_pred = fitted_pipeline.predict(X_test)
        y_prob = fitted_pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test.values, y_pred, y_prob)

        expected_keys = {"accuracy", "precision", "recall", "f1", "auc_roc", "avg_precision"}
        assert expected_keys == set(metrics.keys())

    def test_metrics_in_valid_range(self, fitted_pipeline: Pipeline, split_data) -> None:
        _, X_test, _, y_test = split_data
        y_pred = fitted_pipeline.predict(X_test)
        y_prob = fitted_pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test.values, y_pred, y_prob)

        for name, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"Metric '{name}' out of range: {value}"

    def test_auc_above_random(self, fitted_pipeline: Pipeline, split_data) -> None:
        """Even on a small dataset, a trained LR should beat random (0.5 AUC)."""
        _, X_test, _, y_test = split_data
        y_pred = fitted_pipeline.predict(X_test)
        y_prob = fitted_pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test.values, y_pred, y_prob)
        assert metrics["auc_roc"] > 0.5, f"AUC-ROC too low: {metrics['auc_roc']}"


class TestCompareModels:
    def test_returns_valid_key(self) -> None:
        results = {
            "ModelA": {"auc_roc": 0.85, "f1": 0.70},
            "ModelB": {"auc_roc": 0.90, "f1": 0.75},
        }
        best = compare_models(results, primary_metric="auc_roc")
        assert best == "ModelB"

    def test_handles_single_model(self) -> None:
        results = {"OnlyModel": {"auc_roc": 0.78, "f1": 0.60}}
        best = compare_models(results, primary_metric="auc_roc")
        assert best == "OnlyModel"

    def test_uses_primary_metric(self) -> None:
        results = {
            "HighAUC": {"auc_roc": 0.90, "f1": 0.60},
            "HighF1": {"auc_roc": 0.75, "f1": 0.80},
        }
        assert compare_models(results, primary_metric="auc_roc") == "HighAUC"
        assert compare_models(results, primary_metric="f1") == "HighF1"


class TestMlflowLogging:
    """Verify that training writes runs to a local MLflow tracking store."""

    def test_mlflow_run_created(self, small_df: pd.DataFrame, config: dict, tmp_path: Path) -> None:
        """Train one model and confirm that an MLflow run artifact directory is created."""
        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("test_experiment")

        X, y = prepare_data(small_df, config)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0, stratify=y
        )

        cat_cols = [c for c in config["features"]["categorical_columns"] if c in X_train.columns]
        num_cols = [c for c in config["features"]["numeric_columns"] if c in X_train.columns]
        preprocessor = build_preprocessing_pipeline(cat_cols, num_cols)
        estimator = LogisticRegression(max_iter=100, random_state=0)
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
        pipeline.fit(X_train, y_train)

        with mlflow.start_run() as run:
            mlflow.log_param("test_run", True)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
            run_id = run.info.run_id

        # Verify the run exists
        client = mlflow.tracking.MlflowClient()
        fetched_run = client.get_run(run_id)
        assert fetched_run.info.run_id == run_id
        assert fetched_run.data.params.get("test_run") == "True"

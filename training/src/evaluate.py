"""Model evaluation: metrics, confusion matrix, ROC/PR curves."""

import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Type alias for computed metrics dict
MetricsDict = dict[str, float]


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> MetricsDict:
    """Compute a standard set of binary classification metrics.

    Args:
        y_true: Ground truth labels (0/1).
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Dictionary mapping metric names to float values.
    """
    metrics: MetricsDict = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
    }
    return metrics


def log_metrics_to_mlflow(metrics: MetricsDict) -> None:
    """Log all metrics to the active MLflow run.

    Args:
        metrics: Dictionary of metric name → float value.
    """
    mlflow.log_metrics(metrics)
    for name, value in metrics.items():
        logger.info("  %-20s %.4f", name, value)


def log_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    artifact_dir: Path,
) -> None:
    """Save confusion matrix plot as a PNG artifact.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        artifact_dir: Local directory to save the PNG before logging.
    """
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")

    path = artifact_dir / "confusion_matrix.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    mlflow.log_artifact(str(path), artifact_path="plots")
    logger.info("Confusion matrix saved → %s", path)


def log_roc_pr_curves(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    artifact_dir: Path,
) -> None:
    """Save ROC and Precision-Recall curve plots as PNG artifacts.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities for the positive class.
        artifact_dir: Local directory to save PNGs.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ROC curve
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title("ROC Curve")
    path = artifact_dir / "roc_curve.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    mlflow.log_artifact(str(path), artifact_path="plots")

    # PR curve
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title("Precision-Recall Curve")
    path = artifact_dir / "pr_curve.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    mlflow.log_artifact(str(path), artifact_path="plots")
    logger.info("ROC and PR curves saved.")


def print_classification_report(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> None:
    """Print a sklearn classification report to the logger.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        model_name: Human-readable model identifier for the log header.
    """
    report = classification_report(
        y_true, y_pred, target_names=["No Churn", "Churn"], zero_division=0
    )
    logger.info("\n=== %s — Classification Report ===\n%s", model_name, report)


def evaluate_model(
    model: Pipeline,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    model_name: str,
    artifact_dir: Path,
) -> MetricsDict:
    """Run full evaluation for a trained model pipeline.

    Computes metrics, prints report, and logs everything to MLflow.

    Args:
        model: Fitted sklearn Pipeline (preprocessor + estimator).
        X_test: Test features (raw, before preprocessing).
        y_test: Test labels.
        model_name: Human-readable name for logging.
        artifact_dir: Directory to store plot artifacts.

    Returns:
        Dictionary of computed metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)

    logger.info("=== %s Evaluation ===", model_name)
    log_metrics_to_mlflow(metrics)
    print_classification_report(y_test, y_pred, model_name)
    log_confusion_matrix(y_test, y_pred, artifact_dir)
    log_roc_pr_curves(y_test, y_prob, artifact_dir)

    return metrics


def compare_models(results: dict[str, MetricsDict], primary_metric: str = "auc_roc") -> str:
    """Select the best model based on a primary metric.

    Args:
        results: Dict mapping model_name → MetricsDict.
        primary_metric: Name of the metric to maximise.

    Returns:
        Name of the best model.
    """
    best_model = max(results, key=lambda name: results[name].get(primary_metric, 0.0))
    logger.info(
        "Best model by %s: %s (%.4f)",
        primary_metric,
        best_model,
        results[best_model][primary_metric],
    )
    return best_model

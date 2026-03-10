"""MLflow Model Registry: register and promote the best model."""

import logging

import mlflow
from mlflow.tracking import MlflowClient

from training.src.evaluate import MetricsDict

logger = logging.getLogger(__name__)


def register_best_model(
    run_id: str,
    model_uri: str,
    model_name: str,
    metrics: MetricsDict,
    promotion_threshold_auc: float,
    primary_metric: str = "auc_roc",
) -> mlflow.entities.model_registry.ModelVersion | None:
    """Register a trained model in the MLflow Model Registry.

    If the model's primary metric exceeds the promotion threshold,
    it is transitioned to the 'Production' stage. Otherwise it stays
    in 'Staging'.

    Args:
        run_id: MLflow run ID that produced the model.
        model_uri: Artifact URI to the logged sklearn model (e.g. 'runs:/RUN_ID/model').
        model_name: Registered model name in the MLflow Registry.
        metrics: Evaluation metrics dict for the run.
        promotion_threshold_auc: Minimum AUC-ROC to promote to Production.
        primary_metric: Metric key used for promotion decision.

    Returns:
        The registered ModelVersion object, or None if registration failed.
    """
    client = MlflowClient()

    # Ensure the registered model exists
    try:
        client.get_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        logger.info("Creating new registered model: '%s'.", model_name)
        client.create_registered_model(
            model_name,
            description="Binary churn classifier — Telco Customer Churn dataset",
        )

    logger.info("Registering model '%s' from run %s …", model_name, run_id)
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Set version description with key metrics
    metric_summary = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    client.update_model_version(
        name=model_name,
        version=model_version.version,
        description=f"Metrics: {metric_summary}",
    )

    # Promote based on threshold
    auc = metrics.get(primary_metric, 0.0)
    if auc >= promotion_threshold_auc:
        logger.info(
            "AUC-ROC %.4f ≥ threshold %.4f → transitioning to Production.",
            auc,
            promotion_threshold_auc,
        )
        target_stage = "Production"
    else:
        logger.info(
            "AUC-ROC %.4f < threshold %.4f → staying in Staging.",
            auc,
            promotion_threshold_auc,
        )
        target_stage = "Staging"

    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage=target_stage,
        archive_existing_versions=(target_stage == "Production"),
    )

    logger.info(
        "Model '%s' v%s successfully registered and moved to '%s'.",
        model_name,
        model_version.version,
        target_stage,
    )
    return model_version


def get_latest_production_model_uri(model_name: str) -> str | None:
    """Return the model URI for the latest Production version.

    Args:
        model_name: Registered model name.

    Returns:
        MLflow model URI string, or None if no Production version exists.
    """
    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        logger.warning("No Production model found for '%s'.", model_name)
        return None
    uri = f"models:/{model_name}/Production"
    logger.info("Found Production model: %s", uri)
    return uri

"""MLflow Model Registry loader with local .pkl fallback.

Usage:
    loader = ModelLoader(model_name="ecommerce-churn-predictor",
                         tracking_uri="sqlite:///mlruns.db")
    loader.load()                # called at app startup
    model = loader.get_champion()
    loader.reload()              # hot-reload without downtime
"""

import logging
import pickle
import threading
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)

# Directory where fallback .pkl files are stored (serving/models/)
_MODELS_DIR = Path(__file__).parent.parent / "models"


class ModelLoader:
    """Manages champion and challenger model instances.

    * Champion  → MLflow Registry stage "Production"
    * Challenger → MLflow Registry stage "Staging" (optional)
    * Fallback  → serving/models/{champion,challenger}.pkl

    Thread-safe: uses an internal RLock so reload() can be called from any
    thread without racing against active predictions.

    Args:
        model_name: Registered model name in the MLflow Registry.
        tracking_uri: MLflow tracking URI (e.g. ``sqlite:///mlruns.db``).
    """

    def __init__(self, model_name: str, tracking_uri: str) -> None:
        self.model_name = model_name
        self.tracking_uri = tracking_uri

        self._champion: Any | None = None
        self._challenger: Any | None = None
        self._champion_version: str = "unknown"
        self._challenger_version: str = "none"

        self._lock = threading.RLock()

        mlflow.set_tracking_uri(tracking_uri)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load champion and (optionally) challenger at startup."""
        start = time.time()

        champion, champ_ver = self._load_from_registry("Production")
        if champion is None:
            champion = self._load_from_pkl("champion")
            champ_ver = "pkl-fallback" if champion is not None else "none"

        challenger, chall_ver = self._load_from_registry("Staging")
        if challenger is None:
            challenger = self._load_from_pkl("challenger")
            chall_ver = "pkl-fallback" if challenger is not None else "none"

        with self._lock:
            self._champion = champion
            self._champion_version = champ_ver
            self._challenger = challenger
            self._challenger_version = chall_ver

        elapsed = time.time() - start
        logger.info(
            "Models loaded in %.2fs — champion: %s, challenger: %s",
            elapsed,
            champ_ver,
            chall_ver,
        )

    def reload(self) -> None:
        """Hot-reload models from the registry without stopping the server.

        Only swaps references that are successfully fetched; keeps old model
        alive if the reload fails.
        """
        logger.info("Reloading models from registry …")
        start = time.time()

        new_champ, new_champ_ver = self._load_from_registry("Production")
        new_chall, new_chall_ver = self._load_from_registry("Staging")

        with self._lock:
            if new_champ is not None:
                self._champion = new_champ
                self._champion_version = new_champ_ver
            if new_chall is not None:
                self._challenger = new_chall
                self._challenger_version = new_chall_ver

        elapsed = time.time() - start
        logger.info(
            "Reload done in %.2fs — champion: %s, challenger: %s",
            elapsed,
            self._champion_version,
            self._challenger_version,
        )

    def get_champion(self) -> Any:
        with self._lock:
            if self._champion is None:
                raise RuntimeError("Champion model is not loaded")
            return self._champion

    def get_challenger(self) -> Any | None:
        with self._lock:
            return self._challenger

    def get_info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "champion": {
                    "model_name": self.model_name,
                    "version": self._champion_version,
                    "stage": "Production",
                    "loaded": self._champion is not None,
                },
                "challenger": {
                    "model_name": self.model_name,
                    "version": self._challenger_version,
                    "stage": "Staging",
                    "loaded": self._challenger is not None,
                },
            }

    def load_time(self) -> float:
        """Return seconds taken by the last load() call (for Prometheus gauge)."""
        return getattr(self, "_last_load_elapsed", 0.0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_from_registry(self, stage: str) -> tuple[Any | None, str]:
        """Try to load a model from MLflow Registry by stage.

        Returns (model, version_string) or (None, "none") on any failure.
        """
        try:
            uri = f"models:/{self.model_name}/{stage}"
            model = mlflow.sklearn.load_model(uri)

            # Fetch version string for informational purposes
            client = mlflow.MlflowClient()
            versions = client.get_latest_versions(self.model_name, stages=[stage])
            version = f"v{versions[0].version}" if versions else "unknown"

            logger.info("Loaded %s model %s from MLflow Registry", stage, version)
            return model, version

        except Exception as exc:
            logger.warning("Could not load %s model from registry: %s", stage, exc)
            return None, "none"

    def _load_from_pkl(self, role: str) -> Any | None:
        """Load model from a local .pkl file (serving/models/{role}.pkl)."""
        pkl_path = _MODELS_DIR / f"{role}.pkl"
        if not pkl_path.exists():
            return None
        logger.info("Loading %s model from %s (pkl fallback)", role, pkl_path)
        with open(pkl_path, "rb") as fh:
            return pickle.load(fh)  # noqa: S301 — intentional, controlled path

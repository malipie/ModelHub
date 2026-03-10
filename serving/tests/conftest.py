"""Shared fixtures for serving tests."""

from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fake model — sklearn-compatible, no real ML
# ---------------------------------------------------------------------------


class FakeModel:
    """Minimal sklearn-compatible pipeline that returns predictable probabilities."""

    def predict_proba(self, X):  # noqa: N803
        n = len(X)
        return np.column_stack([np.full(n, 0.3), np.full(n, 0.7)])

    def predict(self, X):  # noqa: N803
        return np.ones(len(X), dtype=int)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def valid_payload() -> dict:
    """A complete, valid PredictionRequest payload."""
    return {
        "request_id": "test-req-001",
        "tenure_months": 24,
        "monthly_spend_eur": 89.99,
        "total_spent_eur": 2159.76,
        "avg_order_value_eur": 44.99,
        "purchase_frequency_per_month": 2.0,
        "num_product_categories": 4,
        "support_tickets_last_month": 2,
        "website_sessions_per_month": 12,
        "cart_abandonment_rate_percent": 45.0,
        "email_engagement_rate_percent": 20.0,
        "reviews_left_count": 3,
        "returns_count_12m": 2,
        "last_purchase_days_ago": 14,
        "satisfaction_score_1_5": 2.5,
        "loyalty_program_member": 0,
        "payment_methods_used": 2,
        "country": "Poland",
        "account_age_category": "1-2 years",
        "subscription_type": "Standard",
        "preferred_category": "Electronics",
    }


@pytest.fixture
def client(fake_model: FakeModel) -> TestClient:
    """TestClient with model loader mocked — no MLflow connection needed."""
    from serving.src import main as main_module

    # Patch load() so the lifespan does not attempt to connect to MLflow
    with patch.object(main_module.loader, "load", return_value=None):
        main_module.loader._champion = fake_model
        main_module.loader._champion_version = "v1-test"
        main_module.loader._challenger = fake_model
        main_module.loader._challenger_version = "v0-test"

        with TestClient(main_module.app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture
def client_no_challenger(fake_model: FakeModel) -> TestClient:
    """TestClient where no challenger model is loaded."""
    from serving.src import main as main_module

    with patch.object(main_module.loader, "load", return_value=None):
        main_module.loader._champion = fake_model
        main_module.loader._champion_version = "v1-test"
        main_module.loader._challenger = None
        main_module.loader._challenger_version = "none"

        with TestClient(main_module.app, raise_server_exceptions=True) as c:
            yield c

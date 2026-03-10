"""Tests for prediction output correctness and edge cases."""

import numpy as np
import pytest
from fastapi.testclient import TestClient


def test_prediction_is_binary(client: TestClient, valid_payload: dict) -> None:
    body = client.post("/predict/champion", json=valid_payload).json()
    assert body["prediction"] in (0, 1)


def test_probability_in_unit_interval(client: TestClient, valid_payload: dict) -> None:
    body = client.post("/predict/champion", json=valid_payload).json()
    assert 0.0 <= body["probability"] <= 1.0


def test_high_probability_predicts_churn(client: TestClient, valid_payload: dict) -> None:
    """FakeModel always returns 0.7 churn probability → label must be 1."""
    body = client.post("/predict/champion", json=valid_payload).json()
    assert body["prediction"] == 1
    assert body["probability"] == pytest.approx(0.7, abs=0.01)


def test_low_probability_predicts_no_churn(fake_model, valid_payload: dict) -> None:
    """A model that predicts 0.3 churn probability → label must be 0."""
    from unittest.mock import patch

    from serving.src import main as main_module

    class LowProbModel:
        def predict_proba(self, X):  # noqa: N803
            n = len(X)
            return np.column_stack([np.full(n, 0.7), np.full(n, 0.3)])

    with patch.object(main_module.loader, "load", return_value=None):
        main_module.loader._champion = LowProbModel()
        main_module.loader._champion_version = "low-v"
        main_module.loader._challenger = None
        main_module.loader._challenger_version = "none"

        from fastapi.testclient import TestClient

        with TestClient(main_module.app) as c:
            body = c.post("/predict/champion", json=valid_payload).json()

    assert body["prediction"] == 0
    assert body["probability"] == pytest.approx(0.3, abs=0.01)


def test_model_version_is_string(client: TestClient, valid_payload: dict) -> None:
    body = client.post("/predict/champion", json=valid_payload).json()
    assert isinstance(body["model_version"], str)
    assert len(body["model_version"]) > 0


def test_probability_rounded_to_4_decimals(client: TestClient, valid_payload: dict) -> None:
    body = client.post("/predict/champion", json=valid_payload).json()
    prob = body["probability"]
    # After round(x, 4) there should be at most 4 decimal places
    assert prob == round(prob, 4)


def test_repeated_calls_are_deterministic(client: TestClient, valid_payload: dict) -> None:
    """Same payload → same result every time."""
    results = [client.post("/predict/champion", json=valid_payload).json() for _ in range(5)]
    probs = [r["probability"] for r in results]
    assert all(p == probs[0] for p in probs)


def test_boundary_satisfaction_score_1(client: TestClient, valid_payload: dict) -> None:
    payload = {**valid_payload, "satisfaction_score_1_5": 1.0}
    resp = client.post("/predict/champion", json=payload)
    assert resp.status_code == 200


def test_boundary_satisfaction_score_5(client: TestClient, valid_payload: dict) -> None:
    payload = {**valid_payload, "satisfaction_score_1_5": 5.0}
    resp = client.post("/predict/champion", json=payload)
    assert resp.status_code == 200


def test_tenure_zero_is_valid(client: TestClient, valid_payload: dict) -> None:
    payload = {**valid_payload, "tenure_months": 0}
    resp = client.post("/predict/champion", json=payload)
    assert resp.status_code == 200


def test_all_subscription_types_accepted(client: TestClient, valid_payload: dict) -> None:
    for sub in ("Basic", "Standard", "Premium"):
        payload = {**valid_payload, "subscription_type": sub}
        resp = client.post("/predict/champion", json=payload)
        assert resp.status_code == 200, f"Failed for subscription_type={sub}"


def test_all_countries_accepted(client: TestClient, valid_payload: dict) -> None:
    for country in ("Poland", "Germany", "France", "UK", "Spain", "Netherlands"):
        payload = {**valid_payload, "country": country}
        resp = client.post("/predict/champion", json=payload)
        assert resp.status_code == 200, f"Failed for country={country}"

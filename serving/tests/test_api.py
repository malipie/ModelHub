"""Tests for FastAPI endpoint contract (shape, status codes, headers)."""

from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "models" in body
    assert "ab_config" in body


def test_health_champion_info_present(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["models"]["champion"]["loaded"] is True


def test_health_503_when_no_champion(fake_model) -> None:
    from unittest.mock import patch

    from serving.src import main as main_module

    with patch.object(main_module.loader, "load", return_value=None):
        main_module.loader._champion = None
        main_module.loader._champion_version = "none"
        main_module.loader._challenger = None
        main_module.loader._challenger_version = "none"

        with TestClient(main_module.app) as c:
            resp = c.get("/health")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /model/info
# ---------------------------------------------------------------------------


def test_model_info_shape(client: TestClient) -> None:
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("model_name", "version", "stage", "loaded"):
        assert key in body["champion"]
        assert key in body["challenger"]


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------


def test_metrics_returns_prometheus_text(client: TestClient) -> None:
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    # Prometheus exposition format starts with "# HELP" or has metric lines
    assert len(resp.text) > 0


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------


def test_predict_response_shape(client: TestClient, valid_payload: dict) -> None:
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {
        "prediction",
        "probability",
        "model_version",
        "model_name",
        "request_id",
    }


def test_predict_echoes_request_id(client: TestClient, valid_payload: dict) -> None:
    resp = client.post("/predict", json=valid_payload)
    assert resp.json()["request_id"] == valid_payload["request_id"]


def test_predict_auto_generates_request_id(client: TestClient, valid_payload: dict) -> None:
    payload = {k: v for k, v in valid_payload.items() if k != "request_id"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["request_id"]  # non-empty


def test_predict_model_name_is_valid(client: TestClient, valid_payload: dict) -> None:
    body = client.post("/predict", json=valid_payload).json()
    assert body["model_name"] in ("champion", "challenger")


def test_predict_422_on_invalid_country(client: TestClient, valid_payload: dict) -> None:
    payload = {**valid_payload, "country": "Mars"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_422_on_missing_field(client: TestClient, valid_payload: dict) -> None:
    payload = {k: v for k, v in valid_payload.items() if k != "tenure_months"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_422_on_out_of_range_satisfaction(client: TestClient, valid_payload: dict) -> None:
    payload = {**valid_payload, "satisfaction_score_1_5": 6.0}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /predict/champion
# ---------------------------------------------------------------------------


def test_predict_champion_always_champion(client: TestClient, valid_payload: dict) -> None:
    resp = client.post("/predict/champion", json=valid_payload)
    assert resp.status_code == 200
    assert resp.json()["model_name"] == "champion"


# ---------------------------------------------------------------------------
# POST /predict/challenger
# ---------------------------------------------------------------------------


def test_predict_challenger_ok(client: TestClient, valid_payload: dict) -> None:
    resp = client.post("/predict/challenger", json=valid_payload)
    assert resp.status_code == 200
    assert resp.json()["model_name"] == "challenger"


def test_predict_challenger_503_when_unavailable(
    client_no_challenger: TestClient, valid_payload: dict
) -> None:
    resp = client_no_challenger.post("/predict/challenger", json=valid_payload)
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# PUT /ab/config
# ---------------------------------------------------------------------------


def test_ab_config_update(client: TestClient) -> None:
    resp = client.put("/ab/config", json={"champion_pct": 70})
    assert resp.status_code == 200
    body = resp.json()
    assert body["champion_pct"] == 70
    assert body["challenger_pct"] == 30

    # Reset to 80 so other tests are not affected
    client.put("/ab/config", json={"champion_pct": 80})


def test_ab_config_422_out_of_range(client: TestClient) -> None:
    resp = client.put("/ab/config", json={"champion_pct": 110})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /model/reload
# ---------------------------------------------------------------------------


def test_model_reload_ok(client: TestClient) -> None:
    from unittest.mock import patch

    from serving.src import main as main_module

    with patch.object(main_module.loader, "reload", return_value=None):
        resp = client.post("/model/reload")
    assert resp.status_code == 200
    assert resp.json()["status"] == "reloaded"

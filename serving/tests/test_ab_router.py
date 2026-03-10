"""Tests for ABRouter — distribution, determinism, and reconfiguration."""

import uuid

import pytest

from serving.src.ab_router import ABRouter

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_request_id_always_same_variant() -> None:
    router = ABRouter(champion_pct=80)
    request_id = str(uuid.uuid4())
    variants = {router.route(request_id) for _ in range(20)}
    assert len(variants) == 1, "Same request_id must always map to the same variant"


def test_different_request_ids_can_split() -> None:
    router = ABRouter(champion_pct=50)
    results = {router.route(str(uuid.uuid4())) for _ in range(100)}
    # With 50/50 and 100 unique IDs, both variants must appear
    assert "champion" in results
    assert "challenger" in results


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------


def test_distribution_80_20() -> None:
    """With 10 000 requests, champion share should be within ±5pp of 80%."""
    router = ABRouter(champion_pct=80)
    n = 10_000
    champion_count = sum(1 for _ in range(n) if router.route(str(uuid.uuid4())) == "champion")
    ratio = champion_count / n
    assert 0.75 <= ratio <= 0.85, f"Expected ~80% champion, got {ratio:.1%}"


def test_distribution_100_0() -> None:
    """100% champion → all requests go to champion."""
    router = ABRouter(champion_pct=100)
    for _ in range(100):
        assert router.route(str(uuid.uuid4())) == "champion"


def test_distribution_0_100() -> None:
    """0% champion → all requests go to challenger."""
    router = ABRouter(champion_pct=0)
    for _ in range(100):
        assert router.route(str(uuid.uuid4())) == "challenger"


# ---------------------------------------------------------------------------
# Runtime reconfiguration
# ---------------------------------------------------------------------------


def test_set_split_changes_distribution() -> None:
    router = ABRouter(champion_pct=80)
    router.set_split(0)
    # After switching to 0%, every request must go to challenger
    for _ in range(50):
        assert router.route(str(uuid.uuid4())) == "challenger"


def test_get_config_reflects_current_split() -> None:
    router = ABRouter(champion_pct=80)
    cfg = router.get_config()
    assert cfg["champion_pct"] == 80
    assert cfg["challenger_pct"] == 20

    router.set_split(60)
    cfg = router.get_config()
    assert cfg["champion_pct"] == 60
    assert cfg["challenger_pct"] == 40


def test_set_split_rejects_out_of_range() -> None:
    router = ABRouter(champion_pct=80)
    with pytest.raises(ValueError):
        router.set_split(101)
    with pytest.raises(ValueError):
        router.set_split(-1)


def test_champion_pct_property() -> None:
    router = ABRouter(champion_pct=70)
    assert router.champion_pct == 70
    assert router.challenger_pct == 30


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_invalid_initial_pct_raises() -> None:
    with pytest.raises(ValueError):
        ABRouter(champion_pct=110)


def test_boundary_values_accepted() -> None:
    ABRouter(champion_pct=0)
    ABRouter(champion_pct=100)


# ---------------------------------------------------------------------------
# Routing via the API endpoint (integration)
# ---------------------------------------------------------------------------


def test_api_ab_route_is_deterministic(client, valid_payload: dict) -> None:
    """Two requests with the same request_id must land on the same model."""
    payload = {**valid_payload, "request_id": "determinism-check-xyz"}
    resp1 = client.post("/predict", json=payload).json()
    resp2 = client.post("/predict", json=payload).json()
    assert resp1["model_name"] == resp2["model_name"]


def test_api_ab_split_roughly_80_20(client, valid_payload: dict) -> None:
    """Over 1000 unique request IDs the split should be within ±10pp of 80%."""
    champion_count = 0
    n = 1000
    for i in range(n):
        payload = {**valid_payload, "request_id": f"load-test-{i:04d}"}
        resp = client.post("/predict", json=payload).json()
        if resp["model_name"] == "champion":
            champion_count += 1

    ratio = champion_count / n
    assert 0.70 <= ratio <= 0.90, f"Expected ~80% champion, got {ratio:.1%}"

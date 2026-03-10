#!/usr/bin/env python3
"""Traffic simulator for the ModelHub API gateway.

Sends a configurable stream of prediction requests and prints a live summary
of champion/challenger split and latencies.

Usage:
    python scripts/simulate_traffic.py [--url URL] [--n N] [--concurrency C]

Examples:
    # hit local dev stack directly (no nginx)
    python scripts/simulate_traffic.py --url http://localhost:8000 --n 200

    # hit production stack through nginx gateway
    python scripts/simulate_traffic.py --url http://localhost:80 --n 1000 --concurrency 10
"""

from __future__ import annotations

import argparse
import json
import random
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

try:
    import requests
except ImportError:
    raise SystemExit("requests not installed — run: pip install requests")

# ---------------------------------------------------------------------------
# Sample data pools
# ---------------------------------------------------------------------------

COUNTRIES = ["Poland", "Germany", "France", "UK", "Spain", "Netherlands"]
SUBSCRIPTION_TYPES = ["Basic", "Standard", "Premium"]
ACCOUNT_AGE_CATEGORIES = ["0-3 months", "3-12 months", "1-2 years", "2+ years"]
PREFERRED_CATEGORIES = ["Electronics", "Fashion", "Home & Garden", "Sports", "Books", "Beauty"]
YES_NO = [0, 1]


def _random_payload() -> dict[str, Any]:
    """Generate a realistic-looking random prediction request."""
    tenure = random.randint(1, 60)
    monthly = round(random.uniform(20.0, 200.0), 2)
    return {
        "request_id": str(uuid.uuid4()),
        "tenure_months": tenure,
        "monthly_spend_eur": monthly,
        "total_spent_eur": round(monthly * tenure * random.uniform(0.85, 1.0), 2),
        "avg_order_value_eur": round(random.uniform(15.0, 300.0), 2),
        "purchase_frequency_per_month": round(random.uniform(0.5, 10.0), 1),
        "num_product_categories": random.randint(1, 8),
        "support_tickets_last_month": random.randint(0, 5),
        "website_sessions_per_month": random.randint(1, 50),
        "cart_abandonment_rate_percent": round(random.uniform(0.0, 100.0), 1),
        "email_engagement_rate_percent": round(random.uniform(0.0, 100.0), 1),
        "reviews_left_count": random.randint(0, 20),
        "returns_count_12m": random.randint(0, 10),
        "last_purchase_days_ago": random.randint(1, 180),
        "satisfaction_score_1_5": round(random.uniform(1.0, 5.0), 1),
        "loyalty_program_member": random.choice(YES_NO),
        "payment_methods_used": random.randint(1, 5),
        "country": random.choice(COUNTRIES),
        "account_age_category": random.choice(ACCOUNT_AGE_CATEGORIES),
        "subscription_type": random.choice(SUBSCRIPTION_TYPES),
        "preferred_category": random.choice(PREFERRED_CATEGORIES),
    }


# ---------------------------------------------------------------------------
# HTTP worker
# ---------------------------------------------------------------------------


def _send_request(url: str, timeout: int = 10) -> dict[str, Any]:
    """Send one prediction request and return a result summary."""
    payload = _random_payload()
    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{url}/predict", json=payload, timeout=timeout)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            body = resp.json()
            return {
                "ok": True,
                "model_name": body.get("model_name", "unknown"),
                "prediction": body.get("prediction"),
                "latency_ms": elapsed_ms,
            }
        return {"ok": False, "status": resp.status_code, "latency_ms": elapsed_ms}
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {"ok": False, "error": str(exc), "latency_ms": elapsed_ms}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _print_summary(results: list[dict[str, Any]]) -> None:
    total = len(results)
    ok = [r for r in results if r.get("ok")]
    errors = [r for r in results if not r.get("ok")]
    latencies = sorted(r["latency_ms"] for r in ok)

    model_counts: Counter[str] = Counter(r["model_name"] for r in ok)
    churn_counts: Counter[int] = Counter(r["prediction"] for r in ok if r.get("prediction") is not None)

    print("\n" + "=" * 60)
    print("  ModelHub Traffic Simulation — Summary")
    print("=" * 60)
    print(f"  Total requests : {total}")
    print(f"  Successful     : {len(ok)} ({len(ok)/total*100:.1f}%)")
    print(f"  Errors         : {len(errors)}")

    if ok:
        print(f"\n  Champion       : {model_counts.get('champion', 0)} ({model_counts.get('champion', 0)/len(ok)*100:.1f}%)")
        print(f"  Challenger     : {model_counts.get('challenger', 0)} ({model_counts.get('challenger', 0)/len(ok)*100:.1f}%)")
        print(f"\n  Churn (label=1): {churn_counts.get(1, 0)} ({churn_counts.get(1, 0)/len(ok)*100:.1f}%)")
        print(f"  No churn (0)  : {churn_counts.get(0, 0)} ({churn_counts.get(0, 0)/len(ok)*100:.1f}%)")
        print(f"\n  Latency (ms):")
        print(f"    p50 : {latencies[len(latencies)//2]:.1f}")
        print(f"    p95 : {latencies[int(len(latencies)*0.95)]:.1f}")
        print(f"    p99 : {latencies[int(len(latencies)*0.99)]:.1f}")
        print(f"    max : {latencies[-1]:.1f}")

    if errors:
        print(f"\n  Error samples:")
        for err in errors[:3]:
            print(f"    {json.dumps(err)}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate traffic against the ModelHub API.")
    parser.add_argument("--url", default="http://localhost:80", help="Base URL of the API")
    parser.add_argument("--n", type=int, default=500, help="Number of requests to send")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of parallel workers")
    parser.add_argument("--timeout", type=int, default=10, help="Per-request timeout in seconds")
    args = parser.parse_args()

    print(f"Sending {args.n} requests to {args.url} (concurrency={args.concurrency})…")

    results: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(_send_request, args.url, args.timeout) for _ in range(args.n)]
        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if i % 50 == 0 or i == args.n:
                ok_count = sum(1 for r in results if r.get("ok"))
                print(f"  [{i:>{len(str(args.n))}}/{args.n}] {ok_count} OK", end="\r")

    total_elapsed = time.perf_counter() - t_start
    print(f"\n  Completed in {total_elapsed:.1f}s  ({args.n/total_elapsed:.0f} req/s)")

    _print_summary(results)


if __name__ == "__main__":
    main()

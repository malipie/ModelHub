"""Prometheus metrics for the serving API."""

from prometheus_client import Counter, Gauge, Histogram

# Latency histogram — one bucket set for both model roles
prediction_latency = Histogram(
    "prediction_latency_seconds",
    "End-to-end prediction latency in seconds",
    ["model_name"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Request counters
prediction_counter = Counter(
    "prediction_total",
    "Total number of predictions served",
    ["model_name", "prediction"],
)

prediction_errors_total = Counter(
    "prediction_errors_total",
    "Total number of prediction errors",
)

# Model load time (set once at startup, updated on reload)
model_load_time_seconds = Gauge(
    "model_load_time_seconds",
    "Time taken to load/reload the model from the registry",
    ["model_name"],
)

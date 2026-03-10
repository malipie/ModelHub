"""Async PostgreSQL writer for prediction audit log.

Uses a fire-and-forget queue so prediction latency is never impacted by DB I/O.
If the database is unavailable the record is silently dropped and an error is logged.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional psycopg2 import — DB logging is best-effort
# ---------------------------------------------------------------------------
try:
    import psycopg2  # type: ignore
    import psycopg2.extras  # type: ignore

    _PSYCOPG2_AVAILABLE = True
except ImportError:
    _PSYCOPG2_AVAILABLE = False
    logger.warning("psycopg2 not installed — prediction logging to PostgreSQL disabled")


_INSERT_PREDICTION = """
INSERT INTO predictions
    (request_id, timestamp, model_name, model_version,
     input_features, prediction, probability, latency_ms)
VALUES
    (%(request_id)s, %(timestamp)s, %(model_name)s, %(model_version)s,
     %(input_features)s, %(prediction)s, %(probability)s, %(latency_ms)s)
ON CONFLICT DO NOTHING;
"""


class PredictionLogger:
    """Non-blocking prediction logger backed by PostgreSQL.

    Internally maintains an asyncio queue; a background coroutine drains it
    and writes to Postgres in batches.  The main prediction path only calls
    `enqueue()` which never blocks.
    """

    def __init__(self) -> None:
        self._dsn: str | None = os.getenv("POSTGRES_DSN")
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=2000)
        self._task: asyncio.Task[None] | None = None
        self._enabled = _PSYCOPG2_AVAILABLE and bool(self._dsn)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background writer coroutine (call from lifespan startup)."""
        if not self._enabled:
            logger.info("PredictionLogger disabled (no POSTGRES_DSN or psycopg2).")
            return
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._writer_loop(), name="db-writer")
        logger.info("PredictionLogger started — DSN: %s", self._dsn)

    async def stop(self) -> None:
        """Flush remaining records and stop (call from lifespan shutdown)."""
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(
        self,
        *,
        request_id: str | UUID,
        model_name: str,
        model_version: str,
        input_features: dict[str, Any],
        prediction: int,
        probability: float,
        latency_ms: float,
    ) -> None:
        """Put a record onto the queue.  Never raises; drops silently if full."""
        if not self._enabled:
            return
        record = {
            "request_id": str(request_id),
            "timestamp": datetime.now(UTC).isoformat(),
            "model_name": model_name,
            "model_version": model_version,
            "input_features": psycopg2.extras.Json(input_features),  # type: ignore[name-defined]
            "prediction": prediction,
            "probability": probability,
            "latency_ms": latency_ms,
        }
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            logger.warning("PredictionLogger queue full — record dropped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _writer_loop(self) -> None:
        """Drain queue and write to Postgres; reconnect on failure."""
        conn = None
        while True:
            try:
                if conn is None or conn.closed:
                    conn = await asyncio.get_event_loop().run_in_executor(None, self._connect)
                record = await self._queue.get()
                await asyncio.get_event_loop().run_in_executor(None, self._write, conn, record)
                self._queue.task_done()
            except asyncio.CancelledError:
                # Flush remaining items synchronously before exit
                if conn and not conn.closed:
                    while not self._queue.empty():
                        try:
                            rec = self._queue.get_nowait()
                            self._write(conn, rec)
                        except Exception:
                            pass
                    conn.close()
                raise
            except Exception as exc:
                logger.error("DB writer error: %s — reconnecting", exc)
                if conn:
                    with contextlib.suppress(Exception):
                        conn.close()
                conn = None
                await asyncio.sleep(5)

    def _connect(self) -> Any:
        conn = psycopg2.connect(self._dsn)  # type: ignore[name-defined]
        conn.autocommit = True
        return conn

    def _write(self, conn: Any, record: dict[str, Any]) -> None:
        with conn.cursor() as cur:
            cur.execute(_INSERT_PREDICTION, record)


# Module-level singleton — imported by main.py
prediction_logger = PredictionLogger()

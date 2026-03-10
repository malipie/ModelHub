"""Deterministic A/B router for champion / challenger traffic splitting.

Routing is based on ``hashlib.md5(request_id)`` so the same customer (same
``request_id``) always lands on the same model variant — critical for fair
A/B experiment comparisons.

Usage:
    router = ABRouter(champion_pct=80)
    variant = router.route("some-uuid")     # → "champion" or "challenger"
    router.set_split(70)                    # live reconfiguration
"""

import hashlib
import logging
import threading

logger = logging.getLogger(__name__)

_MIN_PCT = 0
_MAX_PCT = 100


class ABRouter:
    """Thread-safe champion/challenger traffic router.

    Args:
        champion_pct: Percentage of traffic directed to champion (0–100).
                      The remainder goes to challenger.
    """

    def __init__(self, champion_pct: int = 80) -> None:
        self._validate(champion_pct)
        self._champion_pct = champion_pct
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def champion_pct(self) -> int:
        with self._lock:
            return self._champion_pct

    @property
    def challenger_pct(self) -> int:
        with self._lock:
            return _MAX_PCT - self._champion_pct

    def set_split(self, champion_pct: int) -> None:
        """Update the traffic split at runtime (thread-safe).

        Args:
            champion_pct: New percentage for champion traffic (0–100).

        Raises:
            ValueError: If ``champion_pct`` is outside [0, 100].
        """
        self._validate(champion_pct)
        with self._lock:
            old = self._champion_pct
            self._champion_pct = champion_pct
        logger.info(
            "A/B split updated: champion %d%% → %d%%, challenger %d%% → %d%%",
            old,
            champion_pct,
            _MAX_PCT - old,
            _MAX_PCT - champion_pct,
        )

    def route(self, request_id: str) -> str:
        """Return ``"champion"`` or ``"challenger"`` for a given request ID.

        The decision is deterministic: same ``request_id`` always maps to the
        same variant (given the same split configuration).

        Args:
            request_id: Unique request identifier (UUID recommended).

        Returns:
            ``"champion"`` or ``"challenger"``.
        """
        hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16) % _MAX_PCT  # noqa: S324
        with self._lock:
            return "champion" if hash_val < self._champion_pct else "challenger"

    def get_config(self) -> dict[str, int]:
        with self._lock:
            return {
                "champion_pct": self._champion_pct,
                "challenger_pct": _MAX_PCT - self._champion_pct,
            }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(champion_pct: int) -> None:
        if not (_MIN_PCT <= champion_pct <= _MAX_PCT):
            raise ValueError(
                f"champion_pct must be between {_MIN_PCT} and {_MAX_PCT}, got {champion_pct}"
            )

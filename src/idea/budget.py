"""Budget tracking — the outer constraint that replaces human patience."""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field


@dataclass
class Budget:
    """Hard limits on compute spend. The system runs until convergence or exhaustion."""

    max_api_calls: int = 200
    max_wall_seconds: float = 3600.0  # 1 hour default
    max_depth: int = 3
    max_branches: int = 3
    max_experiment_iters: int = 5
    max_cost_usd: float = 50.0

    # --- mutable state (thread-safe) ---
    _api_calls: int = field(default=0, repr=False)
    _cost_usd: float = field(default=0.0, repr=False)
    _start_time: float = field(default_factory=time.monotonic, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def tick(self, cost: float = 0.0) -> None:
        """Record one API call and its cost."""
        with self._lock:
            self._api_calls += 1
            self._cost_usd += cost

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def api_calls(self) -> int:
        with self._lock:
            return self._api_calls

    @property
    def cost_usd(self) -> float:
        with self._lock:
            return self._cost_usd

    def exhausted(self) -> bool:
        with self._lock:
            if self._api_calls >= self.max_api_calls:
                return True
            if self._cost_usd >= self.max_cost_usd:
                return True
        if self.elapsed >= self.max_wall_seconds:
            return True
        return False

    def split(self, n: int) -> Budget:
        """Divide remaining budget among n child branches."""
        if n <= 0:
            n = 1
        with self._lock:
            remaining_calls = max(1, (self.max_api_calls - self._api_calls) // n)
            remaining_cost = max(0.01, (self.max_cost_usd - self._cost_usd) / n)
            remaining_time = max(60.0, (self.max_wall_seconds - self.elapsed) / n)
        return Budget(
            max_api_calls=remaining_calls,
            max_wall_seconds=remaining_time,
            max_depth=self.max_depth,
            max_branches=self.max_branches,
            max_experiment_iters=self.max_experiment_iters,
            max_cost_usd=remaining_cost,
            # children share parent's counters for global tracking
            _api_calls=0,
            _cost_usd=0.0,
            _start_time=time.monotonic(),
        )

    def summary(self) -> dict:
        return {
            "api_calls": self.api_calls,
            "cost_usd": round(self.cost_usd, 4),
            "elapsed_seconds": round(self.elapsed, 1),
            "exhausted": self.exhausted(),
        }

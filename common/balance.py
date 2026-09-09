"""Scheduling and rebalancing across the registered slaves.

The balancer does two things:

  * decides *when* the current partition has gone stale (a device dropped, a
    device joined, or the stage times drifted apart), and
  * hands out request slots so no single slave is asked to run two forward
    passes at once.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from .partition import partition_layers
from .types import DeviceState, LayerProfile, LayerRange


@dataclass
class BalancePolicy:
    # Fraction by which the slowest stage may exceed the mean before we
    # re-partition. 0.35 tolerates normal jitter but catches a real imbalance.
    imbalance_tolerance: float = 0.35
    # A device is considered gone after this long without a heartbeat.
    heartbeat_timeout_s: float = 15.0
    # Do not re-partition more often than this; moving shards is expensive.
    min_repartition_interval_s: float = 30.0


class Balancer:
    def __init__(
        self,
        layers: list[LayerProfile],
        policy: BalancePolicy | None = None,
        model_id: str = "",
    ) -> None:
        self.layers = layers
        self.policy = policy or BalancePolicy()
        self.model_id = model_id
        self._lock = threading.Lock()
        self._plan: dict[str, LayerRange] = {}
        self._last_repartition = 0.0
        self._inflight: dict[str, int] = {}

    # -- planning ---------------------------------------------------------

    @property
    def plan(self) -> dict[str, LayerRange]:
        with self._lock:
            return dict(self._plan)

    def needs_repartition(self, devices: list[DeviceState], now: float) -> bool:
        alive = [d for d in devices if d.is_alive(self.policy.heartbeat_timeout_s)]
        if not alive:
            return False
        with self._lock:
            planned = set(self._plan)
            last = self._last_repartition
        current = {d.device_id for d in alive}
        if planned != current:
            return True  # membership changed: always re-plan
        if now - last < self.policy.min_repartition_interval_s:
            return False
        return self._is_imbalanced(alive)

    def _is_imbalanced(self, alive: list[DeviceState]) -> bool:
        times = [d.last_forward_ms for d in alive if d.last_forward_ms > 0.0]
        if len(times) < 2:
            return False
        mean = sum(times) / len(times)
        if mean <= 0.0:
            return False
        return (max(times) - mean) / mean > self.policy.imbalance_tolerance

    def repartition(self, devices: list[DeviceState], now: float) -> list[LayerRange]:
        """Recompute the plan over the currently alive devices."""
        alive = [d for d in devices if d.is_alive(self.policy.heartbeat_timeout_s)]
        ranges = partition_layers(self.layers, alive, self.model_id) if alive else []
        with self._lock:
            self._plan = {r.device_id: r for r in ranges}
            self._last_repartition = now
        return ranges

    def stage_order(self) -> list[LayerRange]:
        """The plan sorted the way the pipeline executes it."""
        return sorted(self.plan.values(), key=lambda r: r.start_layer)

    # -- admission --------------------------------------------------------

    def acquire(self, device_id: str, max_inflight: int = 1) -> bool:
        with self._lock:
            n = self._inflight.get(device_id, 0)
            if n >= max_inflight:
                return False
            self._inflight[device_id] = n + 1
            return True

    def release(self, device_id: str) -> None:
        with self._lock:
            n = self._inflight.get(device_id, 0)
            if n > 0:
                self._inflight[device_id] = n - 1

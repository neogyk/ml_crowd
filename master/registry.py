"""Thread-safe table of the slaves the master knows about."""
from __future__ import annotations

import threading
import time
import uuid

from ..common.types import DeviceCaps, DeviceState


class Registry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._devices: dict[str, DeviceState] = {}

    def register(self, hostname: str, caps: DeviceCaps) -> DeviceState:
        device_id = f"{hostname}-{uuid.uuid4().hex[:8]}"
        state = DeviceState(device_id=device_id, hostname=hostname, caps=caps,
                            free_ram_bytes=caps.ram_bytes)
        with self._lock:
            self._devices[device_id] = state
        return state

    def drop(self, device_id: str) -> None:
        with self._lock:
            self._devices.pop(device_id, None)

    def get(self, device_id: str) -> DeviceState | None:
        with self._lock:
            return self._devices.get(device_id)

    def all(self) -> list[DeviceState]:
        with self._lock:
            return list(self._devices.values())

    def alive(self, timeout_s: float) -> list[DeviceState]:
        return [d for d in self.all() if d.is_alive(timeout_s)]

    def touch(self, device_id: str, **fields) -> DeviceState | None:
        """Record a heartbeat and whatever telemetry came with it."""
        with self._lock:
            dev = self._devices.get(device_id)
            if dev is None:
                return None
            for key, value in fields.items():
                if value is not None and hasattr(dev, key):
                    setattr(dev, key, value)
            dev.last_seen = time.monotonic()
            return dev

    def reap(self, timeout_s: float) -> list[str]:
        """Remove devices that stopped heartbeating. Returns their ids."""
        with self._lock:
            dead = [d.device_id for d in self._devices.values()
                    if not d.is_alive(timeout_s)]
            for device_id in dead:
                del self._devices[device_id]
        return dead

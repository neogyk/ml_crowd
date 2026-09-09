"""Plain dataclasses shared by master and slave.

These mirror the protobuf messages but stay importable without the generated
stubs, so partitioning and balancing can be unit-tested on their own.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class DeviceCaps:
    platform: str = "linux"
    ram_bytes: int = 0
    cpu_cores: int = 1
    has_gpu: bool = False
    accelerator: str = "none"
    flops_estimate: float = 0.0
    uplink_mbps: float = 0.0
    downlink_mbps: float = 0.0


@dataclass
class DeviceState:
    """Everything the master knows about one slave."""

    device_id: str
    hostname: str
    caps: DeviceCaps
    battery_pct: float = 100.0
    load_avg: float = 0.0
    free_ram_bytes: int = 0
    last_forward_ms: float = 0.0
    last_seen: float = field(default_factory=time.monotonic)
    assignment: "LayerRange | None" = None

    def is_alive(self, timeout_s: float) -> bool:
        return (time.monotonic() - self.last_seen) < timeout_s


@dataclass(frozen=True)
class LayerRange:
    """Half-open range of transformer layers owned by one device."""

    device_id: str
    start_layer: int
    end_layer: int
    model_id: str = ""
    next_device_id: str = ""

    @property
    def n_layers(self) -> int:
        return self.end_layer - self.start_layer


@dataclass
class LayerProfile:
    """Per-layer cost, read out of the gguf metadata."""

    index: int
    param_bytes: int
    flops: float
    activation_bytes: int

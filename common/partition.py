"""Split a model's layers into contiguous blocks, one per device.

Blocks must stay contiguous: the pipeline runs layer 0..N in order, so a
device owning a non-contiguous set would force extra round trips.
"""
from __future__ import annotations

from .types import DeviceState, LayerProfile, LayerRange


def device_capacity(dev: DeviceState) -> float:
    """Relative throughput weight of a device. Higher means it takes more layers."""
    caps = dev.caps
    flops = caps.flops_estimate
    if flops <= 0.0:
        # Fall back to a crude core/GPU proxy when the device did not measure.
        flops = caps.cpu_cores * (4.0 if caps.has_gpu else 1.0)
    # A device that is loaded or low on battery gets a smaller share.
    load_factor = 1.0 / (1.0 + max(0.0, dev.load_avg))
    battery_factor = 0.5 if dev.battery_pct < 20.0 else 1.0
    return max(flops * load_factor * battery_factor, 1e-9)


def fits_in_ram(dev: DeviceState, layers: list[LayerProfile]) -> bool:
    budget = dev.free_ram_bytes or dev.caps.ram_bytes
    if budget <= 0:
        return True  # unknown; let the slave reject it at load time
    needed = sum(l.param_bytes for l in layers)
    return needed <= budget * 0.8  # leave headroom for activations and the OS


def partition_layers(
    layers: list[LayerProfile],
    devices: list[DeviceState],
    model_id: str = "",
) -> list[LayerRange]:
    """Assign every layer to exactly one device, proportional to capacity.

    Walks the layers in order, filling the current device until it has taken
    its share of the total cost, then moving to the next. Every device gets at
    least one layer as long as there are enough layers to go around.
    """
    if not devices:
        raise ValueError("no devices to partition across")
    if not layers:
        return []

    devices = sorted(devices, key=device_capacity, reverse=True)
    n_dev = min(len(devices), len(layers))
    devices = devices[:n_dev]

    weights = [device_capacity(d) for d in devices]
    total_weight = sum(weights)
    total_cost = sum(l.flops for l in layers) or float(len(layers))

    # Cost target for each device, in the same units as LayerProfile.flops.
    targets = [total_cost * w / total_weight for w in weights]

    ranges: list[LayerRange] = []
    idx = 0
    for d_i, dev in enumerate(devices):
        remaining_devices = n_dev - d_i - 1
        # Always leave one layer for each device still waiting.
        max_end = len(layers) - remaining_devices
        start = idx
        acc = 0.0
        end = start
        while end < max_end:
            cost = layers[end].flops or 1.0
            # Stop once we are closer to the target with this layer than without,
            # but never emit an empty block.
            if end > start and acc + cost / 2.0 > targets[d_i]:
                break
            acc += cost
            end += 1
        if d_i == n_dev - 1:
            end = len(layers)  # last device absorbs the remainder
        ranges.append(
            LayerRange(
                device_id=dev.device_id,
                start_layer=layers[start].index,
                end_layer=layers[end - 1].index + 1,
                model_id=model_id,
            )
        )
        idx = end

    return _chain(ranges)


def _chain(ranges: list[LayerRange]) -> list[LayerRange]:
    """Point each block at the device holding the next block."""
    out = []
    for i, r in enumerate(ranges):
        nxt = ranges[i + 1].device_id if i + 1 < len(ranges) else ""
        out.append(
            LayerRange(
                device_id=r.device_id,
                start_layer=r.start_layer,
                end_layer=r.end_layer,
                model_id=r.model_id,
                next_device_id=nxt,
            )
        )
    return out

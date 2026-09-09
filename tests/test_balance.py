import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml_crowd.common.balance import Balancer, BalancePolicy
from ml_crowd.common.types import DeviceCaps, DeviceState, LayerProfile


def layers(n=16):
    return [LayerProfile(index=i, param_bytes=1000, flops=1.0,
                         activation_bytes=100) for i in range(n)]


def device(name, forward_ms=0.0, flops=1.0):
    d = DeviceState(device_id=name, hostname=name,
                    caps=DeviceCaps(flops_estimate=flops, ram_bytes=1 << 30))
    d.last_forward_ms = forward_ms
    d.last_seen = time.monotonic()
    return d


def test_first_plan_is_needed():
    b = Balancer(layers())
    assert b.needs_repartition([device("a")], time.monotonic())


def test_new_device_triggers_repartition():
    b = Balancer(layers())
    now = time.monotonic()
    b.repartition([device("a"), device("b")], now)
    assert b.needs_repartition([device("a"), device("b"), device("c")], now)


def test_stable_membership_within_cooldown_is_quiet():
    b = Balancer(layers())
    now = time.monotonic()
    devs = [device("a", 100.0), device("b", 100.0)]
    b.repartition(devs, now)
    assert not b.needs_repartition(devs, now + 1.0)


def test_imbalance_triggers_repartition_after_cooldown():
    policy = BalancePolicy(min_repartition_interval_s=0.0)
    b = Balancer(layers(), policy)
    now = time.monotonic()
    devs = [device("a", 100.0), device("b", 100.0)]
    b.repartition(devs, now)
    devs[0].last_forward_ms = 400.0  # one stage is now far slower
    assert b.needs_repartition(devs, now + 1.0)


def test_balanced_stages_do_not_repartition():
    policy = BalancePolicy(min_repartition_interval_s=0.0)
    b = Balancer(layers(), policy)
    now = time.monotonic()
    devs = [device("a", 100.0), device("b", 110.0)]
    b.repartition(devs, now)
    assert not b.needs_repartition(devs, now + 1.0)


def test_dead_devices_are_excluded_from_the_plan():
    b = Balancer(layers())
    dead = device("dead")
    dead.last_seen = time.monotonic() - 999.0
    ranges = b.repartition([device("alive"), dead], time.monotonic())
    assert [r.device_id for r in ranges] == ["alive"]


def test_stage_order_follows_layer_order():
    b = Balancer(layers())
    b.repartition([device("a"), device("b"), device("c")], time.monotonic())
    stages = b.stage_order()
    starts = [s.start_layer for s in stages]
    assert starts == sorted(starts)


def test_admission_limits_one_inflight_per_device():
    b = Balancer(layers())
    assert b.acquire("a")
    assert not b.acquire("a")
    b.release("a")
    assert b.acquire("a")


def test_release_below_zero_is_a_noop():
    b = Balancer(layers())
    b.release("never-acquired")
    assert b.acquire("never-acquired")

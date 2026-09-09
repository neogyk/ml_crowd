import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml_crowd.common.types import DeviceCaps
from ml_crowd.master.registry import Registry


def test_register_assigns_unique_ids():
    r = Registry()
    a = r.register("phone", DeviceCaps(cpu_cores=8))
    b = r.register("phone", DeviceCaps(cpu_cores=8))
    assert a.device_id != b.device_id
    assert len(r.all()) == 2


def test_touch_updates_telemetry_and_last_seen():
    r = Registry()
    dev = r.register("phone", DeviceCaps(ram_bytes=1 << 30))
    before = dev.last_seen
    time.sleep(0.01)
    updated = r.touch(dev.device_id, battery_pct=42.0, load_avg=1.5)
    assert updated.battery_pct == 42.0
    assert updated.load_avg == 1.5
    assert updated.last_seen > before


def test_touch_unknown_device_returns_none():
    assert Registry().touch("ghost", battery_pct=10.0) is None


def test_reap_removes_only_stale_devices():
    r = Registry()
    fresh = r.register("fresh", DeviceCaps())
    stale = r.register("stale", DeviceCaps())
    r.get(stale.device_id).last_seen = time.monotonic() - 100.0
    dead = r.reap(timeout_s=15.0)
    assert dead == [stale.device_id]
    assert [d.device_id for d in r.all()] == [fresh.device_id]


def test_alive_filters_by_timeout():
    r = Registry()
    r.register("a", DeviceCaps())
    stale = r.register("b", DeviceCaps())
    r.get(stale.device_id).last_seen = time.monotonic() - 100.0
    assert len(r.alive(15.0)) == 1

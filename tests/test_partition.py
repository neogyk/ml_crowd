import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml_crowd.common.partition import partition_layers
from ml_crowd.common.types import DeviceCaps, DeviceState, LayerProfile


def make_layers(n, flops=1.0):
    return [LayerProfile(index=i, param_bytes=1000, flops=flops,
                         activation_bytes=100) for i in range(n)]


def make_device(name, flops, ram=1 << 30):
    return DeviceState(device_id=name, hostname=name,
                       caps=DeviceCaps(flops_estimate=flops, ram_bytes=ram),
                       free_ram_bytes=ram)


def test_covers_every_layer_exactly_once():
    ranges = partition_layers(make_layers(32), [make_device(f"d{i}", 1.0)
                                                for i in range(4)])
    covered = []
    for r in sorted(ranges, key=lambda r: r.start_layer):
        covered.extend(range(r.start_layer, r.end_layer))
    assert covered == list(range(32))


def test_equal_devices_split_evenly():
    ranges = partition_layers(make_layers(32), [make_device(f"d{i}", 1.0)
                                                for i in range(4)])
    assert sorted(r.n_layers for r in ranges) == [8, 8, 8, 8]


def test_faster_device_gets_more_layers():
    devices = [make_device("fast", 8.0), make_device("slow", 1.0)]
    ranges = {r.device_id: r.n_layers for r in partition_layers(make_layers(18), devices)}
    assert ranges["fast"] > ranges["slow"]


def test_every_device_gets_at_least_one_layer():
    devices = [make_device("huge", 1000.0)] + [make_device(f"tiny{i}", 0.01)
                                               for i in range(3)]
    ranges = partition_layers(make_layers(8), devices)
    assert len(ranges) == 4
    assert all(r.n_layers >= 1 for r in ranges)


def test_more_devices_than_layers_drops_the_slowest():
    devices = [make_device(f"d{i}", float(i + 1)) for i in range(6)]
    ranges = partition_layers(make_layers(3), devices)
    assert len(ranges) == 3
    # The three fastest devices are the ones kept.
    assert {r.device_id for r in ranges} == {"d5", "d4", "d3"}


def test_stages_are_chained_in_order():
    ranges = sorted(partition_layers(make_layers(12), [make_device(f"d{i}", 1.0)
                                                       for i in range(3)]),
                    key=lambda r: r.start_layer)
    assert ranges[0].next_device_id == ranges[1].device_id
    assert ranges[1].next_device_id == ranges[2].device_id
    assert ranges[2].next_device_id == ""


def test_no_devices_raises():
    try:
        partition_layers(make_layers(4), [])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_battery_and_load_shrink_a_device_share():
    tired = make_device("tired", 4.0)
    tired.battery_pct = 10.0
    tired.load_avg = 3.0
    fresh = make_device("fresh", 4.0)
    ranges = {r.device_id: r.n_layers for r in
              partition_layers(make_layers(16), [tired, fresh])}
    assert ranges["fresh"] > ranges["tired"]

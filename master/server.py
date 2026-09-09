"""Master node: gRPC server the slave devices connect to.

Run with:  python -m ml_crowd.master.server --model model.gguf --port 50051
"""
from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from concurrent import futures

import grpc

from ..common.balance import Balancer, BalancePolicy
from ..common.gguf import GGUFModel
from ..common.types import DeviceCaps
from ..proto import crowd_pb2 as pb
from ..proto import crowd_pb2_grpc as pb_grpc
from .registry import Registry

log = logging.getLogger("crowd.master")

SHARD_CHUNK_BYTES = 1 << 20  # 1 MiB per message keeps us under the gRPC frame cap


class CrowdMLServicer(pb_grpc.CrowdMLServicer):
    def __init__(self, model: GGUFModel, policy: BalancePolicy) -> None:
        self.model = model
        self.model_id = model.path.name
        self.registry = Registry()
        self.balancer = Balancer(model.layer_profiles(), policy, self.model_id)
        # One command queue per device, drained by that device's Heartbeat stream.
        self._commands: dict[str, queue.Queue] = {}
        self._lock = threading.Lock()

    # -- lifecycle --------------------------------------------------------

    def Register(self, request, context):
        caps = _caps_from_pb(request.caps)
        dev = self.registry.register(request.hostname, caps)
        with self._lock:
            self._commands[dev.device_id] = queue.Queue()
        log.info("registered %s (%s, %d cores)", dev.device_id, caps.platform,
                 caps.cpu_cores)
        return pb.RegisterReply(
            device_id=dev.device_id,
            heartbeat_interval_ms=int(self.balancer.policy.heartbeat_timeout_s * 1000 / 3),
        )

    def Heartbeat(self, request_iterator, context):
        """Bidirectional: consume heartbeats on a thread, yield commands here."""
        device_id_box: list[str] = []

        def consume():
            for hb in request_iterator:
                device_id_box.append(hb.device_id)
                self.registry.touch(
                    hb.device_id,
                    battery_pct=hb.battery_pct,
                    load_avg=hb.load_avg,
                    free_ram_bytes=hb.free_ram_bytes,
                    last_forward_ms=hb.last_forward_ms or None,
                )

        reader = threading.Thread(target=consume, daemon=True)
        reader.start()

        # Wait for the first heartbeat so we know which queue to drain.
        deadline = time.monotonic() + 10.0
        while not device_id_box and time.monotonic() < deadline:
            time.sleep(0.05)
        if not device_id_box:
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "no heartbeat received")
        device_id = device_id_box[0]

        with self._lock:
            q = self._commands.setdefault(device_id, queue.Queue())

        try:
            while context.is_active():
                try:
                    yield q.get(timeout=1.0)
                except queue.Empty:
                    continue
        finally:
            log.info("heartbeat stream closed for %s", device_id)

    def LoadShard(self, request_iterator, context):
        """Slaves do not call this; it exists so a slave can mirror to a peer."""
        total = 0
        for chunk in request_iterator:
            total += len(chunk.payload)
        return pb.LoadReply(ok=True, message="master does not host shards",
                            bytes_received=total)

    def Forward(self, request, context):
        context.abort(grpc.StatusCode.UNIMPLEMENTED,
                      "Forward is served by slaves, not the master")

    def Backward(self, request, context):
        context.abort(grpc.StatusCode.UNIMPLEMENTED,
                      "Backward is served by slaves, not the master")

    def PushGradients(self, request_iterator, context):
        n = 0
        for _ in request_iterator:
            n += 1
        return pb.PushReply(ok=True, round=n)

    # -- planning ---------------------------------------------------------

    def send(self, device_id: str, command) -> None:
        with self._lock:
            q = self._commands.get(device_id)
        if q is not None:
            q.put(command)

    def maybe_repartition(self) -> bool:
        """Recompute the plan if the balancer says it is stale. Returns True if it did."""
        now = time.monotonic()
        devices = self.registry.all()
        if not self.balancer.needs_repartition(devices, now):
            return False

        for device_id in self.registry.reap(self.balancer.policy.heartbeat_timeout_s):
            log.warning("device %s timed out", device_id)
            with self._lock:
                self._commands.pop(device_id, None)

        ranges = self.balancer.repartition(self.registry.all(), now)
        for r in ranges:
            dev = self.registry.get(r.device_id)
            if dev is not None:
                dev.assignment = r
            self.send(r.device_id, pb.Command(
                kind=pb.Command.ASSIGN, assignment=_assignment_to_pb(r)))
        log.info("re-partitioned across %d device(s): %s", len(ranges),
                 ", ".join(f"{r.device_id}[{r.start_layer}:{r.end_layer})" for r in ranges))
        return True

    def push_shards(self) -> None:
        """Stream each device the layers it now owns."""
        for r in self.balancer.stage_order():
            dev = self.registry.get(r.device_id)
            if dev is None:
                continue
            addr = getattr(dev, "address", None) or dev.hostname
            try:
                _stream_shard(addr, self.model, r)
            except grpc.RpcError as exc:
                log.error("shard push to %s failed: %s", r.device_id, exc)


def _stream_shard(address: str, model: GGUFModel, layer_range) -> None:
    with grpc.insecure_channel(address) as channel:
        stub = pb_grpc.CrowdMLStub(channel)
        stub.LoadShard(_shard_chunks(model, layer_range))


def _shard_chunks(model: GGUFModel, r):
    index = 0
    for layer in range(r.start_layer, r.end_layer):
        blob = model.read_layer_bytes(layer)
        for off in range(0, len(blob), SHARD_CHUNK_BYTES):
            piece = blob[off:off + SHARD_CHUNK_BYTES]
            last = (layer == r.end_layer - 1) and (off + len(piece) >= len(blob))
            yield pb.ShardChunk(
                device_id=r.device_id, model_id=r.model_id,
                start_layer=r.start_layer, end_layer=r.end_layer,
                chunk_index=index, last=last, payload=piece,
            )
            index += 1


def _caps_from_pb(c) -> DeviceCaps:
    return DeviceCaps(
        platform=c.platform, ram_bytes=c.ram_bytes, cpu_cores=c.cpu_cores,
        has_gpu=c.has_gpu, accelerator=c.accelerator,
        flops_estimate=c.flops_estimate, uplink_mbps=c.uplink_mbps,
        downlink_mbps=c.downlink_mbps,
    )


def _assignment_to_pb(r):
    return pb.Assignment(
        device_id=r.device_id, start_layer=r.start_layer, end_layer=r.end_layer,
        model_id=r.model_id, next_device_id=r.next_device_id,
    )


def serve(model_path: str, port: int, workers: int = 16) -> None:
    model = GGUFModel(model_path)
    log.info("loaded %s: %d layers, d_model=%d", model.path.name, model.n_layers,
             model.embedding_length)
    servicer = CrowdMLServicer(model, BalancePolicy())

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=workers),
        options=[("grpc.max_send_message_length", 64 << 20),
                 ("grpc.max_receive_message_length", 64 << 20)],
    )
    pb_grpc.add_CrowdMLServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info("master listening on :%d", port)

    stop = threading.Event()

    def planner():
        while not stop.wait(2.0):
            try:
                if servicer.maybe_repartition():
                    servicer.push_shards()
            except Exception:
                log.exception("planner iteration failed")

    threading.Thread(target=planner, daemon=True).start()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        stop.set()
        server.stop(grace=2.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Crowd ML master node")
    ap.add_argument("--model", required=True, help="path to the .gguf model")
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    serve(args.model, args.port, args.workers)


if __name__ == "__main__":
    main()

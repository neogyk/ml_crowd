"""Slave node: registers with the master, then serves its block of layers.

A slave is both a gRPC client (Register, Heartbeat) and a gRPC server
(LoadShard, Forward, Backward), because the master and the neighbouring stage
call into it once it owns a block.

Run with:
    python -m ml_crowd.slave.client --master host:50051 --port 50052
"""
from __future__ import annotations

import argparse
import logging
import os
import platform
import queue
import socket
import threading
import time
from concurrent import futures

import grpc

from ..proto import crowd_pb2 as pb
from ..proto import crowd_pb2_grpc as pb_grpc
from .executor import BlockExecutor, EchoExecutor

log = logging.getLogger("crowd.slave")


def probe_caps() -> pb.DeviceCaps:
    """Best-effort look at what this device can offer."""
    cores = os.cpu_count() or 1
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, AttributeError, OSError):
        ram = 0
    accelerator = "metal" if platform.system() == "Darwin" else "none"
    return pb.DeviceCaps(
        platform=platform.system().lower(),
        ram_bytes=ram,
        cpu_cores=cores,
        has_gpu=accelerator != "none",
        accelerator=accelerator,
        # Rough proxy; a real deployment should benchmark once at startup.
        flops_estimate=float(cores) * (4.0 if accelerator != "none" else 1.0),
    )


def free_ram_bytes() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, AttributeError, OSError):
        return 0


class SlaveServicer(pb_grpc.CrowdMLServicer):
    """The half of the slave that the master and peers call into."""

    def __init__(self, executor: BlockExecutor) -> None:
        self.executor = executor
        self.assignment: pb.Assignment | None = None
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self.last_forward_ms = 0.0

    def set_assignment(self, assignment: pb.Assignment) -> None:
        with self._lock:
            self.assignment = assignment
            self._buffer.clear()

    def release(self) -> None:
        with self._lock:
            self.assignment = None
            self._buffer.clear()
        self.executor.release()

    def LoadShard(self, request_iterator, context):
        total = 0
        start = end = 0
        model_id = ""
        buf = bytearray()
        for chunk in request_iterator:
            buf.extend(chunk.payload)
            total += len(chunk.payload)
            start, end, model_id = chunk.start_layer, chunk.end_layer, chunk.model_id
            if chunk.last:
                break
        try:
            self.executor.load(model_id, start, end, bytes(buf))
        except Exception as exc:  # a device may simply not have the RAM
            log.exception("shard load failed")
            return pb.LoadReply(ok=False, message=str(exc), bytes_received=total)
        log.info("loaded layers [%d:%d) of %s (%.1f MiB)", start, end, model_id,
                 total / (1 << 20))
        return pb.LoadReply(ok=True, message="", bytes_received=total)

    def Forward(self, request, context):
        t0 = time.monotonic()
        out = self.executor.forward(request.data, tuple(request.shape), request.dtype)
        self.last_forward_ms = (time.monotonic() - t0) * 1000.0
        return pb.ActivationBatch(
            request_id=request.request_id, model_id=request.model_id,
            shape=request.shape, dtype=request.dtype, data=out,
            from_layer=request.from_layer, to_layer=request.to_layer,
        )

    def Backward(self, request, context):
        out = self.executor.backward(request.data, tuple(request.shape), request.dtype)
        return pb.ActivationBatch(
            request_id=request.request_id, model_id=request.model_id,
            shape=request.shape, dtype=request.dtype, data=out,
            from_layer=request.from_layer, to_layer=request.to_layer,
        )


class Slave:
    def __init__(self, master_addr: str, port: int,
                 executor: BlockExecutor | None = None) -> None:
        self.master_addr = master_addr
        self.port = port
        self.servicer = SlaveServicer(executor or EchoExecutor())
        self.device_id = ""
        self.heartbeat_interval_s = 5.0
        self._stop = threading.Event()
        self._server: grpc.Server | None = None

    # -- server half ------------------------------------------------------

    def start_server(self) -> None:
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=8),
            options=[("grpc.max_send_message_length", 64 << 20),
                     ("grpc.max_receive_message_length", 64 << 20)],
        )
        pb_grpc.add_CrowdMLServicer_to_server(self.servicer, server)
        server.add_insecure_port(f"[::]:{self.port}")
        server.start()
        self._server = server
        log.info("slave serving on :%d", self.port)

    # -- client half ------------------------------------------------------

    def run(self) -> None:
        self.start_server()
        channel = grpc.insecure_channel(self.master_addr)
        stub = pb_grpc.CrowdMLStub(channel)

        hostname = f"{socket.gethostname()}:{self.port}"
        reply = stub.Register(pb.RegisterRequest(hostname=hostname, caps=probe_caps()))
        self.device_id = reply.device_id
        if reply.heartbeat_interval_ms:
            self.heartbeat_interval_s = reply.heartbeat_interval_ms / 1000.0
        log.info("registered with master as %s", self.device_id)

        outbox: queue.Queue = queue.Queue()

        def beats():
            while not self._stop.is_set():
                yield pb.HeartbeatRequest(
                    device_id=self.device_id,
                    battery_pct=100.0,
                    load_avg=os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0,
                    free_ram_bytes=free_ram_bytes(),
                    last_forward_ms=self.servicer.last_forward_ms,
                )
                if self._stop.wait(self.heartbeat_interval_s):
                    return

        try:
            for command in stub.Heartbeat(beats()):
                self._handle(command)
        except grpc.RpcError as exc:
            log.error("heartbeat stream ended: %s", exc.code())
        finally:
            self.stop()

    def _handle(self, command) -> None:
        if command.kind == pb.Command.ASSIGN:
            a = command.assignment
            log.info("assigned layers [%d:%d), next stage %s",
                     a.start_layer, a.end_layer, a.next_device_id or "(none)")
            self.servicer.set_assignment(a)
        elif command.kind == pb.Command.RELEASE:
            log.info("releasing block")
            self.servicer.release()
        elif command.kind == pb.Command.SHUTDOWN:
            log.info("master asked us to shut down")
            self._stop.set()

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            self._server.stop(grace=1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Crowd ML slave (edge device)")
    ap.add_argument("--master", required=True, help="master address, host:port")
    ap.add_argument("--port", type=int, default=50052,
                    help="port this slave serves Forward/Backward on")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    slave = Slave(args.master, args.port)
    try:
        slave.run()
    except KeyboardInterrupt:
        slave.stop()


if __name__ == "__main__":
    main()

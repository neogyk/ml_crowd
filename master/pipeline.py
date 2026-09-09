"""Drive one inference request through the chain of slave stages."""
from __future__ import annotations

import logging
import time
import uuid

import grpc

from ..proto import crowd_pb2 as pb
from ..proto import crowd_pb2_grpc as pb_grpc

log = logging.getLogger("crowd.pipeline")


class Pipeline:
    """Holds one channel per stage and walks activations along them."""

    def __init__(self, servicer, timeout_s: float = 30.0) -> None:
        self.servicer = servicer
        self.timeout_s = timeout_s
        self._channels: dict[str, grpc.Channel] = {}

    def _stub(self, device_id: str, address: str):
        channel = self._channels.get(device_id)
        if channel is None:
            channel = grpc.insecure_channel(
                address,
                options=[("grpc.max_send_message_length", 64 << 20),
                         ("grpc.max_receive_message_length", 64 << 20)],
            )
            self._channels[device_id] = channel
        return pb_grpc.CrowdMLStub(channel)

    def close(self) -> None:
        for channel in self._channels.values():
            channel.close()
        self._channels.clear()

    def forward(self, embeddings: bytes, shape: tuple[int, ...],
                dtype: str = "f32") -> pb.ActivationBatch:
        """Run the embedded input through every stage, in layer order."""
        stages = self.servicer.balancer.stage_order()
        if not stages:
            raise RuntimeError("no devices available")

        request_id = uuid.uuid4().hex
        batch = pb.ActivationBatch(
            request_id=request_id, model_id=self.servicer.model_id,
            shape=list(shape), dtype=dtype, data=embeddings,
        )

        for stage in stages:
            dev = self.servicer.registry.get(stage.device_id)
            if dev is None:
                raise RuntimeError(f"stage device {stage.device_id} disappeared")
            address = getattr(dev, "address", None) or dev.hostname

            if not self.servicer.balancer.acquire(stage.device_id):
                raise RuntimeError(f"{stage.device_id} is busy")
            try:
                batch.from_layer = stage.start_layer
                batch.to_layer = stage.end_layer
                t0 = time.monotonic()
                batch = self._stub(stage.device_id, address).Forward(
                    batch, timeout=self.timeout_s)
                dev.last_forward_ms = (time.monotonic() - t0) * 1000.0
            finally:
                self.servicer.balancer.release(stage.device_id)

        return batch

    def backward(self, grad: bytes, shape: tuple[int, ...],
                 dtype: str = "f32") -> pb.ActivationBatch:
        """Walk the gradient back through the stages in reverse."""
        stages = list(reversed(self.servicer.balancer.stage_order()))
        if not stages:
            raise RuntimeError("no devices available")

        batch = pb.ActivationBatch(
            request_id=uuid.uuid4().hex, model_id=self.servicer.model_id,
            shape=list(shape), dtype=dtype, data=grad,
        )
        for stage in stages:
            dev = self.servicer.registry.get(stage.device_id)
            if dev is None:
                raise RuntimeError(f"stage device {stage.device_id} disappeared")
            address = getattr(dev, "address", None) or dev.hostname
            batch.from_layer = stage.start_layer
            batch.to_layer = stage.end_layer
            batch = self._stub(stage.device_id, address).Backward(
                batch, timeout=self.timeout_s)
        return batch

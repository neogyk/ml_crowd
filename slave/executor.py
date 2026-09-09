"""The compute backend a slave runs over its block of layers.

`BlockExecutor` is the seam where a real runtime (llama.cpp, executorch, a
custom Metal/Vulkan kernel) plugs in. `EchoExecutor` is the default so the
transport can be exercised end to end without a runtime installed.
"""
from __future__ import annotations

from typing import Protocol


class BlockExecutor(Protocol):
    def load(self, model_id: str, start_layer: int, end_layer: int,
             weights: bytes) -> None:
        """Take ownership of the shard covering [start_layer, end_layer)."""

    def forward(self, data: bytes, shape: tuple[int, ...], dtype: str) -> bytes:
        """Run the block forward over the incoming activations."""

    def backward(self, grad: bytes, shape: tuple[int, ...], dtype: str) -> bytes:
        """Return the gradient w.r.t. this block's input."""

    def release(self) -> None:
        """Drop the shard and free its memory."""


class EchoExecutor:
    """Passes activations through unchanged. Useful for wiring tests."""

    def __init__(self) -> None:
        self.model_id = ""
        self.start_layer = 0
        self.end_layer = 0
        self.weights_bytes = 0

    def load(self, model_id: str, start_layer: int, end_layer: int,
             weights: bytes) -> None:
        self.model_id = model_id
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.weights_bytes = len(weights)

    def forward(self, data: bytes, shape, dtype: str) -> bytes:
        return data

    def backward(self, grad: bytes, shape, dtype: str) -> bytes:
        return grad

    def release(self) -> None:
        self.weights_bytes = 0
        self.model_id = ""

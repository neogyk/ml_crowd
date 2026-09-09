"""Minimal gguf reader: enough metadata to profile and shard a model.

Only the header, kv-metadata and tensor directory are parsed. Tensor payloads
are read lazily by byte range so the master never holds the whole model.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

from .types import LayerProfile

GGUF_MAGIC = b"GGUF"

# gguf metadata value types
(_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STR, _ARR, _U64, _I64, _F64) = range(13)

_SCALAR = {
    _U8: ("<B", 1), _I8: ("<b", 1), _U16: ("<H", 2), _I16: ("<h", 2),
    _U32: ("<I", 4), _I32: ("<i", 4), _F32: ("<f", 4), _BOOL: ("<?", 1),
    _U64: ("<Q", 8), _I64: ("<q", 8), _F64: ("<d", 8),
}

# Bytes per element for the quant types we care about, as (block_size, block_bytes).
_QUANT = {
    0: (1, 4),      # F32
    1: (1, 2),      # F16
    2: (32, 18),    # Q4_0
    3: (32, 20),    # Q4_1
    6: (32, 22),    # Q5_0
    7: (32, 24),    # Q5_1
    8: (32, 34),    # Q8_0
    12: (256, 110), # Q3_K
    13: (256, 144), # Q4_K
    14: (256, 176), # Q5_K
    15: (256, 210), # Q6_K
}

_BLOCK_RE = re.compile(r"blk\.(\d+)\.")


@dataclass
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: int
    offset: int          # relative to the start of the tensor data section
    n_bytes: int

    @property
    def block_index(self) -> int | None:
        m = _BLOCK_RE.match(self.name)
        return int(m.group(1)) if m else None


class GGUFModel:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.metadata: dict[str, object] = {}
        self.tensors: list[TensorInfo] = []
        self.data_offset = 0
        self._parse()

    # -- parsing ----------------------------------------------------------

    def _parse(self) -> None:
        with self.path.open("rb") as f:
            buf = f.read(4)
            if buf != GGUF_MAGIC:
                raise ValueError(f"{self.path} is not a gguf file")
            version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
            if version < 2:
                raise ValueError(f"unsupported gguf version {version}")

            for _ in range(n_kv):
                key = self._read_str(f)
                self.metadata[key] = self._read_value(f)

            for _ in range(n_tensors):
                name = self._read_str(f)
                n_dims = struct.unpack("<I", f.read(4))[0]
                shape = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
                dtype, offset = struct.unpack("<IQ", f.read(12))
                self.tensors.append(
                    TensorInfo(name, shape, dtype, offset, _tensor_bytes(shape, dtype))
                )

            alignment = int(self.metadata.get("general.alignment", 32) or 32)
            pos = f.tell()
            self.data_offset = pos + (-pos % alignment)

    def _read_str(self, f) -> str:
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", errors="replace")

    def _read_value(self, f):
        (vtype,) = struct.unpack("<I", f.read(4))
        return self._read_typed(f, vtype)

    def _read_typed(self, f, vtype):
        if vtype == _STR:
            return self._read_str(f)
        if vtype == _ARR:
            elem_type, n = struct.unpack("<IQ", f.read(12))
            return [self._read_typed(f, elem_type) for _ in range(n)]
        fmt, size = _SCALAR[vtype]
        return struct.unpack(fmt, f.read(size))[0]

    # -- derived views ----------------------------------------------------

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", "llama"))

    @property
    def n_layers(self) -> int:
        key = f"{self.architecture}.block_count"
        if key in self.metadata:
            return int(self.metadata[key])
        blocks = {t.block_index for t in self.tensors if t.block_index is not None}
        return len(blocks)

    @property
    def embedding_length(self) -> int:
        return int(self.metadata.get(f"{self.architecture}.embedding_length", 0))

    def tensors_for_layer(self, index: int) -> list[TensorInfo]:
        return [t for t in self.tensors if t.block_index == index]

    def layer_profiles(self, seq_len: int = 512) -> list[LayerProfile]:
        """Per-layer cost estimate used by the partitioner."""
        d_model = self.embedding_length or 4096
        out: list[LayerProfile] = []
        for i in range(self.n_layers):
            tensors = self.tensors_for_layer(i)
            param_bytes = sum(t.n_bytes for t in tensors)
            n_params = sum(_numel(t.shape) for t in tensors)
            # Two flops per parameter per token dominates; attention adds a term
            # quadratic in sequence length.
            flops = 2.0 * n_params * seq_len + 2.0 * seq_len * seq_len * d_model
            out.append(
                LayerProfile(
                    index=i,
                    param_bytes=param_bytes,
                    flops=flops,
                    activation_bytes=seq_len * d_model * 4,
                )
            )
        return out

    def read_layer_bytes(self, index: int) -> bytes:
        """Raw bytes of every tensor in one layer, concatenated in file order."""
        tensors = sorted(self.tensors_for_layer(index), key=lambda t: t.offset)
        chunks = []
        with self.path.open("rb") as f:
            for t in tensors:
                f.seek(self.data_offset + t.offset)
                chunks.append(f.read(t.n_bytes))
        return b"".join(chunks)


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _tensor_bytes(shape, dtype: int) -> int:
    n = _numel(shape)
    block_size, block_bytes = _QUANT.get(dtype, (1, 4))
    return (n // block_size) * block_bytes

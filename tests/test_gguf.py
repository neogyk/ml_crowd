import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml_crowd.common.gguf import GGUFModel

_U32, _STR = 4, 8


def _str(s: bytes) -> bytes:
    return struct.pack("<Q", len(s)) + s


def write_gguf(path: Path, n_layers=4, d_model=64):
    """Build a tiny but structurally valid gguf file."""
    kv = b""
    n_kv = 0
    for key, val in [(b"general.architecture", b"llama")]:
        kv += _str(key) + struct.pack("<I", _STR) + _str(val)
        n_kv += 1
    for key, val in [(b"llama.block_count", n_layers),
                     (b"llama.embedding_length", d_model)]:
        kv += _str(key) + struct.pack("<I", _U32) + struct.pack("<I", val)
        n_kv += 1

    tensors = b""
    n_tensors = 0
    offset = 0
    for i in range(n_layers):
        for suffix in (b"attn_q.weight", b"ffn_up.weight"):
            name = b"blk." + str(i).encode() + b"." + suffix
            shape = (d_model, d_model)
            tensors += (_str(name) + struct.pack("<I", 2)
                        + struct.pack("<QQ", *shape)
                        + struct.pack("<IQ", 0, offset))  # dtype 0 = F32
            offset += d_model * d_model * 4
            n_tensors += 1

    header = b"GGUF" + struct.pack("<IQQ", 3, n_tensors, n_kv)
    body = header + kv + tensors
    pad = -len(body) % 32
    path.write_bytes(body + b"\0" * pad + b"\0" * offset)
    return path


def test_parses_metadata_and_tensors(tmp_path):
    m = GGUFModel(write_gguf(tmp_path / "t.gguf"))
    assert m.architecture == "llama"
    assert m.n_layers == 4
    assert m.embedding_length == 64
    assert len(m.tensors) == 8


def test_tensors_are_grouped_by_layer(tmp_path):
    m = GGUFModel(write_gguf(tmp_path / "t.gguf"))
    for i in range(4):
        assert len(m.tensors_for_layer(i)) == 2


def test_layer_profiles_are_per_layer_and_positive(tmp_path):
    m = GGUFModel(write_gguf(tmp_path / "t.gguf"))
    profiles = m.layer_profiles(seq_len=128)
    assert len(profiles) == 4
    assert all(p.flops > 0 and p.param_bytes > 0 for p in profiles)
    # Identical layers should cost the same.
    assert len({p.param_bytes for p in profiles}) == 1


def test_read_layer_bytes_matches_declared_size(tmp_path):
    m = GGUFModel(write_gguf(tmp_path / "t.gguf"))
    expected = sum(t.n_bytes for t in m.tensors_for_layer(2))
    assert len(m.read_layer_bytes(2)) == expected


def test_rejects_non_gguf(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE" + b"\0" * 64)
    try:
        GGUFModel(bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError")

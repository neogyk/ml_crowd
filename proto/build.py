"""Generate the gRPC stubs from crowd.proto.

    python -m ml_crowd.proto.build
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def main() -> int:
    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"-I{HERE}",
        f"--python_out={HERE}",
        f"--grpc_python_out={HERE}",
        str(HERE / "crowd.proto"),
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        return result.returncode

    # protoc emits `import crowd_pb2`, which only resolves if the proto dir is
    # on sys.path. Rewrite it to a package-relative import.
    grpc_file = HERE / "crowd_pb2_grpc.py"
    text = grpc_file.read_text()
    grpc_file.write_text(text.replace(
        "\nimport crowd_pb2 as crowd__pb2",
        "\nfrom . import crowd_pb2 as crowd__pb2",
    ))
    print(f"generated stubs in {HERE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

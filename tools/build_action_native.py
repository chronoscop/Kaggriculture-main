#!/usr/bin/env python3
"""Build the owned Rust action simulator with a verifiable source/binary receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def source_hash(root: Path) -> str:
    paths = [root / "Cargo.toml", root / "Cargo.lock", root / "pyproject.toml", *sorted((root / "src").glob("*.rs"))]
    entries = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--offline", action="store_true", help="use locally cached Cargo dependencies")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    project = Path(__file__).resolve().parents[1]
    crate = project / "native/action_engine"
    output = args.out or project / "src/route_rl/ppo/_native"
    # Dependencies are committed in Cargo.lock. Never refresh their versions as
    # a side effect of building a deployable binary.
    digest = source_hash(crate)
    environment = os.environ.copy()
    environment["ROUTE_RL_NATIVE_SOURCE_SHA256"] = digest
    command = ["cargo", "build", "--manifest-path", str(crate / "Cargo.toml"), "--release", "--locked", "--features", "python", "--jobs", str(args.jobs)]
    if args.offline:
        command.append("--offline")
    subprocess.run(command, check=True, env=environment)
    binary = crate / "target/release/libroute_rl_action_engine.so"
    if not binary.exists():
        raise SystemExit("The build tool currently supports Linux shared libraries.")
    output.mkdir(parents=True, exist_ok=True)
    # Cargo enables abi3-py310; retain the portable stable-ABI suffix rather than
    # tying the artifact filename to the Python used to launch this build tool.
    destination = output / "route_rl_action_engine.abi3.so"
    shutil.copy2(binary, destination)
    receipt = {"contract": "owned-action-native-build-v1", "module": "route_rl_action_engine",
               "source_sha256": digest, "binary_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
               "binary": destination.name, "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
               "cargo_profile": "release", "features": ["python"]}
    (output / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"binary": str(destination), "receipt": str(output / "build.json"), **receipt}, indent=2))


if __name__ == "__main__":
    main()

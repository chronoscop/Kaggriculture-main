"""Build the project's owned C++17 season planner; runtime never compiles."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src/route_rl/season_search"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--output", type=Path, default=ASSETS / "terminal_search.so")
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = ["-O3", "-std=c++17", "-fPIC", "-shared", "-DNDEBUG", "-fno-math-errno",
             "-fno-trapping-math", "-ffp-contract=off"]
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        library = Path(temporary) / "terminal_search.so"
        subprocess.run([args.compiler, *flags, str(ASSETS / "terminal_search.cpp"), "-o", str(library)], check=True)
        library.replace(output)
    source = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(ASSETS.iterdir()) if path.suffix in (".cpp", ".inc")}
    receipt = {"contract": "owned-season-search-native-wdl-v1", "platform": platform.platform(),
               "machine": platform.machine(), "flags": flags,
               "compiler": subprocess.check_output([args.compiler, "--version"], text=True).splitlines()[0],
               "source_sha256": source, "binary_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix(".build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"event": "season_search_built", "path": str(output), **receipt}))


if __name__ == "__main__":
    main()

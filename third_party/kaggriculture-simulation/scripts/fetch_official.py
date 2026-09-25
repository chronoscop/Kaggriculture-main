#!/usr/bin/env python3
"""Download the pinned official engine wheel and verify it.

    python scripts/fetch_official.py [--dest .pinwork]

Downloads ``kaggle-environments==1.32.7`` (no dependencies) with pip,
checks the SHA-256 of ``envs/kaggriculture/kaggriculture.py`` and
``kaggriculture.json`` inside the wheel against the pins in
``kaggsim.official``, extracts it to ``<dest>/official`` and prints the
``KAGGSIM_OFFICIAL_PATH`` to use. Importing it additionally needs
``jsonschema`` and ``requests``.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import subprocess
import sys
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src-python"))

from kaggsim.official import (PINNED_ENGINE_SHA256, PINNED_SPEC_SHA256,  # noqa: E402,E501
                              PINNED_VERSION)

ENGINE = "kaggle_environments/envs/kaggriculture/kaggriculture.py"
SPEC = "kaggle_environments/envs/kaggriculture/kaggriculture.json"


def verify_wheel(path: str) -> None:
    """Raise ValueError unless the wheel's engine files match the pins."""
    with zipfile.ZipFile(path) as z:
        got = hashlib.sha256(z.read(ENGINE)).hexdigest()
        spec = hashlib.sha256(z.read(SPEC)).hexdigest()
    if got != PINNED_ENGINE_SHA256:
        raise ValueError(f"{path}: engine sha256 {got} != pin")
    if spec != PINNED_SPEC_SHA256:
        raise ValueError(f"{path}: spec sha256 {spec} != pin")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", default=os.path.join(ROOT, ".pinwork"))
    ap.add_argument("--wheel", default=None,
                    help="use this wheel instead of downloading")
    args = ap.parse_args(argv)
    os.makedirs(args.dest, exist_ok=True)
    wheel = args.wheel
    if wheel is None:
        pattern = os.path.join(
            args.dest, f"kaggle_environments-{PINNED_VERSION}-*.whl")
        found = glob.glob(pattern)
        if not found:
            subprocess.run([sys.executable, "-m", "pip", "download",
                            "--no-deps", "-q",
                            f"kaggle-environments=={PINNED_VERSION}",
                            "-d", args.dest], check=True)
            found = glob.glob(pattern)
        wheel = found[0]
    verify_wheel(wheel)
    target = os.path.join(args.dest, "official")
    with zipfile.ZipFile(wheel) as z:
        z.extractall(target)
    print(f"verified {os.path.basename(wheel)}")
    print(f"KAGGSIM_OFFICIAL_PATH={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

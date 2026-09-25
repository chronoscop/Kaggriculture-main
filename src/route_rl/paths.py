"""Repository-local assets used by the training checkout."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = PROJECT_ROOT / "third_party" / "kaggriculture-simulation"
BASELINE = PROJECT_ROOT / "agents" / "baseline.py"
RUNS = PROJECT_ROOT / "runs"


def add_kaggsim() -> None:
    package_root = SIM_ROOT / "src-python"
    if not (package_root / "kaggsim" / "serve.py").is_file():
        raise FileNotFoundError(f"Rust simulator Python client missing: {package_root}")
    path = str(package_root)
    if path not in sys.path:
        sys.path.insert(0, path)

"""Locate the ``kagg`` binary."""
from __future__ import annotations

import os
import shutil

_EXE = "kagg.exe" if os.name == "nt" else "kagg"
# Repository root when running from a source checkout (src-python/kaggsim).
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def find_kagg(explicit: str | None = None) -> str:
    """Resolve the kagg binary: explicit path, ``$KAGG_BIN``,
    ``$CARGO_TARGET_DIR/release``, the checkout's ``src-rust/target/release``,
    then ``PATH``."""
    cands = [explicit, os.environ.get("KAGG_BIN")]
    tdir = os.environ.get("CARGO_TARGET_DIR")
    if tdir:
        # cargo resolves a relative target dir against the directory it was
        # run from; the task runner runs it from the repository root
        if not os.path.isabs(tdir):
            tdir = os.path.join(_REPO, tdir)
        cands.append(os.path.join(tdir, "release", _EXE))
    cands.append(os.path.join(_REPO, "src-rust", "target", "release", _EXE))
    for c in cands:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    on_path = shutil.which("kagg")
    if on_path:
        return on_path
    raise FileNotFoundError(
        "kagg binary not found: build it with `make build` (cargo build "
        "--release --manifest-path src-rust/Cargo.toml) or set KAGG_BIN")

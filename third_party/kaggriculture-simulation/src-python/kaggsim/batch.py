"""Run ``kagg batch``: many tape pairs at engine speed."""
from __future__ import annotations

import os
import subprocess
import tempfile

from .binary import find_kagg
from .tape import batch_jobs


def run_batch(pairs, threads: int = 2, kagg: str | None = None):
    """Play ``(seed | None, tape_a, tape_b)`` pairs; returns a list (input
    order) of ``(seed, bank0, bank1)`` or raises on a job error."""
    exe = find_kagg(kagg)
    pairs = list(pairs)
    fd, jobs = tempfile.mkstemp(suffix=".tsv")
    os.close(fd)
    try:
        batch_jobs(pairs, jobs)
        out = subprocess.run([exe, "batch", jobs, str(int(threads))],
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", check=True)
    finally:
        os.unlink(jobs)
    results = []
    for line in out.stdout.splitlines():
        f = line.split("\t")
        if f[1] == "ERR":
            raise RuntimeError(f"job {f[0]}: {f[2]}")
        results.append((int(f[1]), float(f[2]), float(f[3])))
    return results

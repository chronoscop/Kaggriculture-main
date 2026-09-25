#!/usr/bin/env python3
"""Scrub / secret audit of the files git tracks.

    python scripts/scrub_audit.py [--extra-term WORD ...]

Fails (exit 1) on: absolute local paths, e-mail addresses (other than
noreply placeholders), common API-key / token shapes, private-key blocks,
and any extra terms passed with ``--extra-term`` or listed one per line in
``.scrub_terms.txt`` at the repository root (an optional, untracked file).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PATTERNS = {
    "windows absolute path": re.compile(r"\b[A-Za-z]:[\\/][A-Za-z0-9_.-]+[\\/]"),
    "home directory path": re.compile(r"/(?:home|Users)/[a-z0-9_.-]+/", re.I),
    "e-mail address": re.compile(
        r"[A-Za-z0-9._%+-]+@(?!example\.invalid|anthropic\.com)"
        r"[A-Za-z0-9-]+\.[A-Za-z]{2,}"),
    "aws access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "github token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    "slack token": re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    "generic api key": re.compile(
        r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*['\"][A-Za-z0-9/+_-]{16,}"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "kaggle key": re.compile(r"(?i)kaggle[_-]?key\s*[:=]"),
}
SKIP = {"LICENSE", "src-rust/Cargo.lock", "scripts/scrub_audit.py"}


def tracked_files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.splitlines()
    return [f for f in out if f not in SKIP]


def extra_terms(cli_terms):
    terms = list(cli_terms or [])
    local = os.path.join(ROOT, ".scrub_terms.txt")
    if os.path.exists(local):
        with open(local, encoding="utf-8") as fh:
            terms += [t.strip() for t in fh if t.strip()
                      and not t.startswith("#")]
    return terms


def audit(files=None, terms=()):
    findings = []
    pats = dict(PATTERNS)
    for t in terms:
        pats[f"forbidden term {t!r}"] = re.compile(re.escape(t), re.I)
    for f in files if files is not None else tracked_files():
        p = os.path.join(ROOT, f)
        try:
            with open(p, encoding="utf-8") as fh:
                text = fh.read()
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for name, rx in pats.items():
                if rx.search(line):
                    findings.append((f, n, name, line.strip()[:120]))
    return findings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--extra-term", action="append", default=[])
    args = ap.parse_args(argv)
    findings = audit(terms=extra_terms(args.extra_term))
    for f, n, name, line in findings:
        print(f"{f}:{n}: {name}: {line}")
    print(f"{len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())

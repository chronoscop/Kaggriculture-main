"""Thin Python front end for ``kagg tournament`` and ``kagg selfplay``.

All scheduling, parallelism, world selection, labelling, feature extraction
and statistics run in Rust (``kagg``); this module only writes the JSON
config, starts ``kagg`` and reads back the summary. See
``docs/tournaments.md`` for the config reference.

    from kaggsim.tournament import run_tournament
    summary = run_tournament({
        "name": "ab-v2",
        "candidate": {"name": "v2", "type": "python", "path": "v2/main.py"},
        "panel": [{"name": "v1", "type": "python", "path": "v1/main.py"},
                  {"name": "rnd", "type": "builtin", "kind": "random"}],
        "worlds": {"strategy": "stratified", "pool": [0, 3000],
                   "per_world": 4, "key_depth": 2},
        "workers": 8})
    print(summary["focus"]["overall"])

    python -m kaggsim.tournament template > t.json
    python -m kaggsim.tournament run t.json --set workers=8
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

from .binary import find_kagg


def _run(kind: str, cfg: dict, overrides=(), kagg=None, quiet=True,
         env=None) -> dict:
    exe = find_kagg(kagg)
    fd, path = tempfile.mkstemp(suffix=".json", prefix=f"kagg_{kind}_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh)
        cmd = [exe, kind, path]
        for o in overrides:
            cmd += ["--set", o]
        if quiet:
            cmd.append("--quiet")
        run_env = dict(os.environ)
        pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(
            __file__)))
        run_env["PYTHONPATH"] = os.pathsep.join(
            [pkg_parent] + [p for p in [run_env.get("PYTHONPATH")] if p])
        run_env.setdefault("KAGGSIM_PYTHON", sys.executable)
        run_env.update(env or {})
        out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                             errors="replace",
                             env=run_env)
    finally:
        os.unlink(path)
    if out.returncode != 0:
        raise RuntimeError(f"kagg {kind} failed ({out.returncode}): "
                           f"{out.stderr.strip()[-2000:]}")
    out_dir = None
    for line in out.stderr.splitlines():
        if line.startswith("results: "):
            out_dir = line[len("results: "):].strip()
    if out_dir is None:
        raise RuntimeError(f"kagg {kind} did not report its output dir")
    with open(os.path.join(out_dir, "summary.json"), encoding="utf-8") as fh:
        summary = json.load(fh)
    summary["out_dir"] = out_dir
    return summary


def run_tournament(cfg: dict, overrides=(), kagg=None, quiet=True) -> dict:
    """Run a tournament config; returns ``summary.json`` as a dict."""
    return _run("tournament", cfg, overrides, kagg, quiet)


def run_selfplay(cfg: dict, overrides=(), kagg=None, quiet=True) -> dict:
    """Run a self-play config; returns ``summary.json`` as a dict."""
    return _run("selfplay", cfg, overrides, kagg, quiet)


def template(kind: str = "tournament", kagg=None) -> dict:
    out = subprocess.run([find_kagg(kagg), "template", kind],
                         capture_output=True, text=True, encoding="utf-8",
                             errors="replace", check=True)
    return json.loads(out.stdout)


def compare(results_a: str, results_b: str, agent: str,
            agent_b: str | None = None, kagg=None) -> dict:
    """Paired McNemar of one agent across two results.jsonl files."""
    cmd = [find_kagg(kagg), "compare", results_a, results_b, agent]
    if agent_b:
        cmd.append(agent_b)
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                             errors="replace", check=True)
    return json.loads(out.stdout)


def load_results(path: str) -> list:
    """Rows of a results.jsonl (one dict per game)."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def main(argv=None, kind="tournament"):
    ap = argparse.ArgumentParser(prog=f"python -m kaggsim.{kind}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("template")
    r = sub.add_parser("run")
    r.add_argument("config")
    r.add_argument("--set", action="append", default=[])
    args = ap.parse_args(argv)
    if args.cmd == "template":
        print(json.dumps(template(kind), indent=1))
        return 0
    with open(args.config, encoding="utf-8") as fh:
        cfg = json.load(fh)
    fn = run_tournament if kind == "tournament" else run_selfplay
    summary = fn(cfg, args.set, quiet=False)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

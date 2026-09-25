"""Gauntlet: one candidate file against your opponent list, with a paired
comparison between builds.

Plays a candidate agent file against YOUR opponent files on a fixed seed set
in both seats, on the Rust engine in parallel (``kagg tournament``), and
records every game to a resumable ``results.jsonl`` keyed by a content hash
of both agents. Two candidates run on the same opponents and seeds pair up
game by game; :func:`compare` (``kagg compare``) runs McNemar's exact test.

    python -m kaggsim.gauntlet run cand.py --opp a=opp_a.py --opp b=opp_b.py \\
        --seeds 0-49 --label cand --workers 8
    python -m kaggsim.gauntlet compare gauntlets/new/results.jsonl \\
        gauntlets/old/results.jsonl --agent-a new --agent-b old
    python -m kaggsim.gauntlet verify-official cand.py opp_a.py --seeds 1,2,3

``verify-official`` replays games on the OFFICIAL runner and demands exact
banks against the Rust engine.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os

from ._util import parse_seeds
from .serve import Serve, load_agent, run_match
from .tournament import compare as _kagg_compare
from .tournament import run_tournament


def _spec(name, path):
    kind = "tape" if str(path).endswith(".tape") else "python"
    return {"name": name, "type": kind, "path": os.path.abspath(path)}


def run_gauntlet(candidate: str, opponents, seeds, label: str = "candidate",
                 out_dir: str = "gauntlets", workers: int = 2,
                 seats: str = "both", kagg=None, **extra) -> dict:
    """Play ``candidate`` against ``opponents`` = [(name, path), ...].

    Returns the tournament summary (``summary["focus"]`` holds the
    candidate's overall / per-opponent / per-world / per-seat scores and
    ``summary["out_dir"]`` the directory with ``results.jsonl``). Extra
    keyword arguments are merged into the tournament config (for example
    ``time_limits={"act_s": 1.0, "overage_s": 60.0}``).
    """
    names = [n for n, _ in opponents]
    if label in names or len(set(names)) != len(names):
        raise ValueError("candidate label and opponent names must be unique")
    cfg = {"name": label, "candidate": _spec(label, candidate),
           "panel": [_spec(n, p) for n, p in opponents],
           "schedule": "gauntlet", "seats": seats,
           "worlds": {"strategy": "list", "seeds": parse_seeds(seeds)},
           "workers": int(workers),
           "output": {"dir": out_dir, "resume": True}}
    cfg.update(extra)
    return run_tournament(cfg, kagg=kagg)


def compare(path_a: str, path_b: str, agent_a: str,
            agent_b: str | None = None, kagg=None) -> dict:
    """Paired McNemar of ``agent_a`` in file A vs ``agent_b`` in file B,
    over games matched by (opponent, seed, seat); the latest row per game
    wins. Delegates to ``kagg compare``."""
    return _kagg_compare(path_a, path_b, agent_a, agent_b, kagg=kagg)


def verify_official(candidate: str, opponent: str, seeds) -> list:
    """Rust serve vs the official runner: exact banks or bust."""
    from . import official
    out = []
    with Serve() as srv:
        for seed in parse_seeds(seeds):
            with contextlib.redirect_stdout(io.StringIO()):
                off, _ = official.run_agents(load_agent(candidate),
                                             load_agent(opponent), seed)
                rust = run_match(load_agent(candidate), load_agent(opponent),
                                 seed, srv)
            out.append({"seed": seed, "official": list(off),
                        "rust": list(rust),
                        "exact": tuple(off) == tuple(rust)})
    return out


def _opp(arg: str):
    if "=" not in arg:
        raise argparse.ArgumentTypeError(f"--opp needs name=path, got {arg!r}")
    name, path = arg.split("=", 1)
    return name, path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kaggsim.gauntlet",
                                 description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("candidate")
    r.add_argument("--opp", action="append", required=True, type=_opp,
                   help="name=path (repeatable)")
    r.add_argument("--seeds", default="0-19")
    r.add_argument("--seats", default="both")
    r.add_argument("--label", default="candidate")
    r.add_argument("--out", default="gauntlets")
    r.add_argument("--workers", type=int, default=2)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--agent-a", required=True)
    c.add_argument("--agent-b", default=None)
    v = sub.add_parser("verify-official")
    v.add_argument("candidate")
    v.add_argument("opponent")
    v.add_argument("--seeds", default="1,2,3")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        res = run_gauntlet(args.candidate, args.opp, args.seeds, args.label,
                           args.out, args.workers, args.seats)
    elif args.cmd == "compare":
        res = compare(args.a, args.b, args.agent_a, args.agent_b)
    else:
        res = verify_official(args.candidate, args.opponent, args.seeds)
        print(json.dumps(res, indent=1))
        return 0 if all(r["exact"] for r in res) else 1
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

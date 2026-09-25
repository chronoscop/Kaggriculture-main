"""Head-to-head benchmark: official engine vs the Rust engine.

Plays the SAME games (policy pair + seed) on:

* ``official``      -- the pinned official interpreter (``env.run``) in
                       Python worker processes;
* ``rust-tournament`` -- ``kagg tournament`` with the same Python policies
                       in ``kaggsim.host`` processes (what evaluating live
                       Python agents costs with this toolkit);
* ``rust-batch``    -- ``kagg batch`` replaying the recorded action streams
                       of those games (open loop, engine-bound);

sequentially (1 worker) and in parallel (``--workers``). Every game's final
banks must agree across all paths. An extra ``rust-builtin`` row plays
Rust fixture policies (no Python at all) for raw engine throughput.

    python -m kaggsim.benchmark --games 20 --workers 2
    python -m kaggsim.benchmark --games 200 --workers 8 --json bench.json

Timings include process start-up, so small ``--games`` understate the
parallel speed-up.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time

from .batch import run_batch
from .policies import make_policy
from .serve import Serve, run_match
from .tape import action_to_line, write_tape
from .tournament import load_results, run_tournament

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _init_worker(pkg_parent):
    if pkg_parent not in sys.path:
        sys.path.insert(0, pkg_parent)


def game_plan(n: int, kinds=("scripted", "random"), base_seed: int = 500):
    """``n`` games: (seed, kind0, policy_seed0, kind1, policy_seed1)."""
    return [(base_seed + i, kinds[0], 11, kinds[1], 12) for i in range(n)]


def play_official(game):
    from . import official
    seed, k0, s0, k1, s1 = game
    with contextlib.redirect_stdout(io.StringIO()):
        banks, _ = official.run_agents(make_policy(k0, s0),
                                       make_policy(k1, s1), seed)
    return tuple(banks)


def _official(plan, workers):
    t0 = time.perf_counter()
    if workers <= 1:
        res = [play_official(g) for g in plan]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker,
                      initargs=(_PKG_PARENT,)) as pool:
            res = pool.map(play_official, plan)
    return res, time.perf_counter() - t0


def _tournament(plan, workers, out_dir, kagg=None, builtin=False):
    seed0, k0, s0, k1, s1 = plan[0]
    if builtin:
        a = {"name": "a", "type": "builtin", "kind": "random", "seed": s0}
        b = {"name": "b", "type": "builtin", "kind": "chaos", "seed": s1}
    else:
        a = {"name": "a", "type": "pypolicy", "kind": k0, "seed": s0}
        b = {"name": "b", "type": "pypolicy", "kind": k1, "seed": s1}
    cfg = {"name": f"bench-{'builtin' if builtin else 'py'}-w{workers}",
           "candidate": a, "panel": [b], "seats": "seat0",
           "worlds": {"strategy": "list", "seeds": [g[0] for g in plan]},
           "workers": workers, "output": {"dir": out_dir, "resume": False}}
    t0 = time.perf_counter()
    summ = run_tournament(cfg, kagg=kagg)
    dt = time.perf_counter() - t0
    rows = {r["seed"]: tuple(r["banks"]) for r in load_results(
        os.path.join(summ["out_dir"], "results.jsonl"))}
    return [rows[g[0]] for g in plan], dt


def run_benchmark(games=20, workers=2, kinds=("scripted", "random"),
                  base_seed=500, include_official=True, kagg=None,
                  log=print) -> dict:
    plan = game_plan(games, kinds, base_seed)
    rows, banks = [], {}

    def add(name, w, secs, results):
        rows.append({"path": name, "workers": w, "games": len(results),
                     "seconds": round(secs, 3),
                     "games_per_sec": round(len(results) / secs, 2),
                     "sec_per_game": round(secs / len(results), 4)})
        banks[(name, w)] = results
        if log:
            log(f"  {name:<16} workers={w:<3} {secs:8.2f}s "
                f"{len(results) / secs:9.2f} games/s")

    with tempfile.TemporaryDirectory() as tmp:
        for w in sorted({1, workers}):
            res, dt = _tournament(plan, w, tmp, kagg)
            add("rust-tournament", w, dt, res)
        # record the streams once (serve), then replay them open loop
        streams = []
        with Serve(kagg) as srv:
            for seed, k0, s0, k1, s1 in plan:
                _, trace = run_match(make_policy(k0, s0), make_policy(k1, s1),
                                     seed, srv, record=True)
                streams.append(([action_to_line(t["actions"][0])
                                 for t in trace],
                                [action_to_line(t["actions"][1])
                                 for t in trace]))
        jobs = []
        for i, (g, (l0, l1)) in enumerate(zip(plan, streams)):
            a, b = (os.path.join(tmp, f"{i}_{s}.tape") for s in (0, 1))
            write_tape(a, g[0], l0)
            write_tape(b, g[0], l1)
            jobs.append((g[0], a, b))
        run_batch(jobs[:1], threads=1, kagg=kagg)   # warm-up (file cache)
        for w in sorted({1, workers}):
            t0 = time.perf_counter()
            res = run_batch(jobs, threads=w, kagg=kagg)
            add("rust-batch", w, time.perf_counter() - t0,
                [(b0, b1) for _, b0, b1 in res])
        builtin_plan = game_plan(max(games, 200), kinds, base_seed)
        res, dt = _tournament(builtin_plan, workers, tmp, kagg, builtin=True)
        rows.append({"path": "rust-builtin", "workers": workers,
                     "games": len(res), "seconds": round(dt, 3),
                     "games_per_sec": round(len(res) / dt, 2),
                     "sec_per_game": round(dt / len(res), 4),
                     "note": "Rust fixture policies, no Python"})
        if log:
            log(f"  {'rust-builtin':<16} workers={workers:<3} {dt:8.2f}s "
                f"{len(res) / dt:9.2f} games/s")
    if include_official:
        for w in sorted({1, workers}):
            res, dt = _official(plan, w)
            add("official", w, dt, res)
    reference = banks[("rust-tournament", 1)]
    agree = {f"{k[0]}@{k[1]}": sum(1 for x, y in zip(v, reference)
                                   if tuple(x) == tuple(y))
             for k, v in banks.items()}
    if include_official:
        base = next(r for r in rows if r["path"] == "official"
                    and r["workers"] == 1)["sec_per_game"]
        for r in rows:
            r["speedup_vs_official_1"] = round(base / r["sec_per_game"], 1)
    result = {"games": games, "workers": workers, "kinds": list(kinds),
              "rows": rows, "banks_agree_with_rust_tournament": agree,
              "all_banks_identical": all(v == games for v in agree.values()),
              "cpu_count": os.cpu_count()}
    if log:
        log(f"  banks identical across paths: {result['all_banks_identical']}"
            f" {agree}")
    return result


def markdown(result: dict) -> str:
    lines = ["| path | workers | games | seconds | games/s | s/game | "
             "speed-up vs official x1 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in result["rows"]:
        lines.append(f"| {r['path']} | {r['workers']} | {r['games']} | "
                     f"{r['seconds']} | {r['games_per_sec']} | "
                     f"{r['sec_per_game']} | "
                     f"{r.get('speedup_vs_official_1', '-')} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kaggsim.benchmark",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--kinds", default="scripted,random")
    ap.add_argument("--base-seed", type=int, default=500)
    ap.add_argument("--no-official", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    if args.games < 1 or args.workers < 1:
        ap.error("--games and --workers must be >= 1")
    res = run_benchmark(args.games, args.workers,
                        tuple(args.kinds.split(",")), args.base_seed,
                        not args.no_official)
    print(markdown(res))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
    return 0 if res["all_banks_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

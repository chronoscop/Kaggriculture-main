"""World generator: which town shops a game ACTUALLY unlocks.

The "world" of an episode -- above all its first two unlocked shops, which set
town demand for the rest of the season -- is not a function of the seed
alone. The engine draws shop unlocks from the same per-day RNG as weed spawns,
and weeds consume one draw per EMPTY tile. Anything that changes how many
tiles are empty at a day's end (planting, digging, building) shifts the draw.
So the realized world depends on the seed AND both players' action streams
AND seat order. A catalog built by driving idle (PASS) players labels worlds
that active play may never visit.

Split points (derived from the engine and checked by the test suite):

* the first shop is drawn at the end of day 2 (during step 71), so it is
  fixed by actions at steps < :data:`FIRST_SHOP_STEP` (72);
* the second at the end of day 5 (step 143), fixed by steps <
  :data:`SECOND_SHOP_STEP` (144).

A stream that differs from another only at or after a split keeps the other
stream's world key for that split.

    python -m kaggsim.worlds catalog --seeds 0-99 --a scripted:1 --b random:2
    python -m kaggsim.worlds catalog --seeds 0-99          # idle (PASS) drive
    python -m kaggsim.worlds screen games/*.json --prefix prefix.tape \\
        --opp o1.tape --opp o2.tape --seeds 0-199 --out screening
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import time

from . import stats
from ._util import parse_seeds
from .batch import run_batch
from .binary import find_kagg
from .constants import FINAL_STEP, FIRST_SHOP_STEP, SECOND_SHOP_STEP
from .policies import make_policy
from .serve import Serve, run_match
from .tape import (action_to_line, load_replay, read_tape, replay_to_tape,
                   splice_tapes)

__all__ = ["FIRST_SHOP_STEP", "SECOND_SHOP_STEP", "world_key",
           "realized_world", "build_map", "build_catalog", "idle_catalog",
           "screen_continuations", "screening_markdown"]


def world_key(shops, k: int = 2) -> str | None:
    """``"SHOP1|SHOP2"`` for the first ``k`` shops, or None if fewer."""
    shops = list(shops or [])
    return "|".join(shops[:k]) if len(shops) >= k else None


def _as_lines(stream):
    """Tape path, list of lines, or list of action dicts -> list of lines."""
    if isinstance(stream, str):
        return read_tape(stream)[1]
    return [a if isinstance(a, str) else action_to_line(a) for a in stream]


def realized_world(srv: Serve, seed: int, stream0, stream1, k: int = 2):
    """The first ``k`` shops unlocked when two open-loop streams play."""
    js = srv.gengame(seed, _as_lines(stream0)[:FINAL_STEP],
                     _as_lines(stream1)[:FINAL_STEP])
    return world_key(js["final"]["town"]["unlocked_shops"], k)


def build_map(stream, opponents, seeds, seats=(0, 1), k: int = 2,
              srv: Serve | None = None, progress=None) -> dict:
    """``{(opponent_index, seed, seat): world_key}`` for one open-loop
    ``stream`` against each opponent stream; ``seat`` is the stream's seat."""
    own = srv is None
    srv = srv or Serve()
    lines = _as_lines(stream)
    opps = [_as_lines(o) for o in opponents]
    out, done = {}, 0
    try:
        for oi, other in enumerate(opps):
            for s in seeds:
                for seat in seats:
                    a, b = (lines, other) if seat == 0 else (other, lines)
                    out[(oi, s, seat)] = realized_world(srv, s, a, b, k)
                    done += 1
                    if progress:
                        progress(done)
    finally:
        if own:
            srv.close()
    return out


def build_catalog(policy_a, policy_b, seeds, k: int = 2, mirror=False,
                  srv: Serve | None = None, label: str = "") -> dict:
    """Seed catalog for a CLOSED-LOOP policy pair.

    ``policy_a`` / ``policy_b`` are zero-argument factories returning fresh
    agent callables (fresh state per game). With ``mirror`` each seed is also
    played with seats swapped. The catalog only describes worlds as realized
    by THIS pair; regenerate it for the pair you evaluate.
    """
    own = srv is None
    srv = srv or Serve()
    worlds, banks = {}, {}
    t0 = time.time()
    try:
        for s in seeds:
            for swap in ((False, True) if mirror else (False,)):
                a, b = policy_a(), policy_b()
                if swap:
                    a, b = b, a
                (b0, b1), trace = run_match(a, b, s, srv, record=True)
                key = world_key(trace[-1]["state"]["town"]["unlocked_shops"],
                                k)
                tag = f"{s}{'m' if swap else ''}"
                worlds[tag] = key
                banks[tag] = [b0, b1]
    finally:
        if own:
            srv.close()
    by_world = {}
    for tag, key in worlds.items():
        by_world.setdefault(key or "?", []).append(tag)
    return {"generator": {"label": label, "k": k, "mirror": mirror,
                          "n_games": len(worlds),
                          "seconds": round(time.time() - t0, 1)},
            "worlds": worlds, "banks": banks,
            "by_world": dict(sorted(by_world.items()))}


def idle_catalog(seeds, k: int = 2, threads: int = 2, kagg=None) -> dict:
    """Catalog under an idle (PASS/PASS) drive, computed by ``kagg worlds``
    -- the cheapest baseline, and the one most likely to disagree with
    active play."""
    seeds = list(seeds)
    if not seeds:
        return {"generator": {"label": "idle", "k": k, "n_games": 0},
                "worlds": {}, "by_world": {}}
    spec = ",".join(str(s) for s in seeds)
    out = subprocess.run([find_kagg(kagg), "worlds", spec, str(int(k)),
                          str(int(threads))], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", check=True)
    js = json.loads(out.stdout)
    worlds = {str(s): js["worlds"].get(str(s)) for s in seeds}
    by_world = {}
    for tag, key in worlds.items():
        by_world.setdefault(key or "?", []).append(tag)
    return {"generator": {"label": "idle", "k": js["key_depth"],
                          "n_games": len(worlds)},
            "worlds": worlds, "by_world": dict(sorted(by_world.items()))}


# ------------------------------------------------ continuation screening --

def _key_depth(at_step: int) -> int:
    if at_step >= SECOND_SHOP_STEP:
        return 2
    if at_step >= FIRST_SHOP_STEP:
        return 1
    raise ValueError(
        f"at_step={at_step} is before the first shop split "
        f"({FIRST_SHOP_STEP}): the spliced stream would change the world it "
        "is evaluated in")


def _source_streams(sources, workdir):
    """Yield (label, seed, tape_seat0, tape_seat1, seat) per candidate.

    Sources: replay ``.json`` files (both seats become candidates, labelled
    by the recorded pair), ``(tape0, tape1)`` pairs (same), or single tapes
    (labelled against an idle opponent -- less faithful).
    """
    for i, src in enumerate(sources):
        if isinstance(src, (tuple, list)):
            a, b = src
            seed = read_tape(a)[0]
            for seat in (0, 1):
                yield (os.path.basename(src[seat]), seed, a, b, seat)
        elif str(src).lower().endswith(".json"):
            rep = load_replay(src)
            base = os.path.splitext(os.path.basename(src))[0]
            paths = []
            for seat in (0, 1):
                p = os.path.join(workdir, f"src{i}_s{seat}.tape")
                replay_to_tape(rep, seat, p)
                paths.append(p)
            seed = read_tape(paths[0])[0]
            for seat in (0, 1):
                yield (f"{base}:s{seat}", seed, paths[0], paths[1], seat)
        else:
            seed = read_tape(src)[0]
            yield (os.path.basename(src), seed, src, None, 0)


def _score_summary(scores):
    n = len(scores)
    mean = sum(scores) / n if n else float("nan")
    if n > 1:
        sd = math.sqrt(sum((x - mean) ** 2 for x in scores) / (n - 1))
        half = 1.959964 * sd / math.sqrt(n)
    else:
        half = float("nan")
    return {"n": n, "score": round(mean, 4),
            "wins": sum(1 for x in scores if x == 1.0),
            "draws": sum(1 for x in scores if x == 0.5),
            "losses": sum(1 for x in scores if x == 0.0),
            "ci95": [round(mean - half, 4), round(mean + half, 4)]}


def screen_continuations(sources, prefix, opponents, seeds, at_step=144,
                         threads=2, top_k=10, seats=(0, 1), out=None,
                         srv=None, workdir=None) -> dict:
    """Screen recorded continuations of a prefix tape, per realized world.

    1. Every source becomes per-seat tapes (replays via ``replay_to_tape``).
    2. Each tape is labelled with the world realized by ITS OWN recorded
       pair (a single tape: against an idle opponent).
    3. Evaluation cells are ``(opponent, seed, seat)``; a cell's world is the
       one realized by ``prefix`` vs that opponent. The key depends only on
       steps before ``at_step``, so every spliced candidate shares it.
    4. For each world, every candidate labelled with it is spliced as
       ``prefix[:at_step] + candidate[at_step:]`` and played against the
       opponent tapes on that world's cells with ``kagg batch``. Per-cell
       score: win 1, draw 0.5, loss 0 (the candidate seat's bank vs the
       opponent's).
    5. Candidates are ranked per world by mean score (top ``top_k`` kept);
       the leader is compared with the runner-up by ``stats.paired_test``
       on the shared cells.

    Tapes do not react to the board: this is OPEN-LOOP SCREENING, and a
    stream that ranks well here can still lose as a live agent. Confirm any
    choice with live agents (``kagg tournament``). Returns the result; with
    ``out`` (a path without extension) also writes ``<out>.json`` and
    ``<out>.md``.
    """
    at_step = int(at_step)
    k = _key_depth(at_step)
    seeds = list(seeds)
    own_srv = srv is None
    srv = srv or Serve()
    own_dir = workdir is None
    tmp = tempfile.mkdtemp(prefix="screen_") if own_dir else workdir
    t_start = time.time()
    try:
        cands = []
        for label, seed, ta, tb, seat in _source_streams(sources, tmp):
            world = realized_world(srv, seed, ta, tb if tb else [], k)
            cands.append({"name": label, "tape": ta if seat == 0 else tb,
                          "source_seed": seed, "world": world})
        cells = {}
        for oi, opp in enumerate(opponents):
            for s in seeds:
                for seat in seats:
                    a, b = (prefix, opp) if seat == 0 else (opp, prefix)
                    cells.setdefault(realized_world(srv, s, a, b, k),
                                     []).append((oi, s, seat))
        table = {}
        worlds = sorted({c["world"] for c in cands if c["world"]})
        for wi, world in enumerate(worlds):
            wc = cells.get(world, [])
            group = [c for c in cands if c["world"] == world]
            entry = {"cells": len(wc), "candidates": len(group),
                     "ranked": []}
            table[world] = entry
            if not wc:
                entry["note"] = "no evaluation cell realizes this world"
                continue
            scored = []
            for ci, c in enumerate(group):
                spliced = os.path.join(tmp, f"w{wi}_c{ci}.tape")
                splice_tapes(prefix, c["tape"], at_step, spliced,
                             validate=False)
                jobs = [(s, spliced, opponents[oi]) if seat == 0
                        else (s, opponents[oi], spliced)
                        for oi, s, seat in wc]
                res = run_batch(jobs, threads=threads)
                scores = [stats.score(b0, b1) if seat == 0
                          else stats.score(b1, b0)
                          for (_, b0, b1), (_, _, seat) in zip(res, wc)]
                scored.append((c, scores))
            scored.sort(key=lambda cs: -sum(cs[1]) / len(cs[1]))
            for c, scores in scored[:top_k]:
                row = {"name": c["name"], "source_seed": c["source_seed"]}
                row.update(_score_summary(scores))
                entry["ranked"].append(row)
            if len(scored) >= 2:
                pt = stats.paired_test(scored[0][1], scored[1][1])
                entry["leader_vs_runner_up"] = {
                    "better_leader": pt["better_a"],
                    "better_runner_up": pt["better_b"],
                    "p_value": round(pt["p_value"], 6),
                    "score_diff": round(pt["score_diff"], 4),
                    "ci95": [round(x, 4) for x in pt["ci95"]],
                    "significant": pt["significant"]}
        result = {"meta": {
            "mode": "open-loop screening", "at_step": at_step,
            "world_key_depth": k,
            "prefix": os.path.basename(str(prefix)),
            "opponents": [os.path.basename(str(o)) for o in opponents],
            "seeds": len(seeds), "seats": list(seats),
            "candidates": len(cands),
            "unlabelled": sum(1 for c in cands if not c["world"]),
            "cells_by_world": {str(w): len(v) for w, v in
                               sorted(cells.items(), key=lambda x: str(x[0]))},
            "seconds": round(time.time() - t_start, 1)},
            "worlds": table}
    finally:
        if own_srv:
            srv.close()
        if own_dir:
            shutil.rmtree(tmp, ignore_errors=True)
    if out:
        with open(out + ".json", "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1)
        with open(out + ".md", "w", encoding="utf-8") as fh:
            fh.write(screening_markdown(result))
    return result


def screening_markdown(result: dict) -> str:
    """Human-readable summary of a :func:`screen_continuations` result."""
    m = result["meta"]
    out = ["# Continuation screening (open loop)", "",
           f"Splice at step {m['at_step']} (world key = first "
           f"{m['world_key_depth']} shop(s)); {m['candidates']} candidate(s); "
           f"opponents: {', '.join(m['opponents'])}; "
           f"{m['seeds']} seed(s) x seats {m['seats']}.", "",
           "Tapes do not react to the board. Confirm any choice with "
           "live agents before relying on it.", ""]
    for world, e in result["worlds"].items():
        out += [f"## {world}", "",
                f"{e['candidates']} candidate(s), {e['cells']} evaluation "
                "cell(s).", ""]
        if not e["ranked"]:
            out += [f"_{e.get('note', 'no candidates')}_", ""]
            continue
        out += ["| rank | candidate | n | score | W/D/L | 95% CI |",
                "|---:|---|---:|---:|---|---|"]
        for i, r in enumerate(e["ranked"], 1):
            out.append(f"| {i} | {r['name']} | {r['n']} | {r['score']:.3f} "
                       f"| {r['wins']}/{r['draws']}/{r['losses']} | "
                       f"{r['ci95'][0]:.3f} .. {r['ci95'][1]:.3f} |")
        lv = e.get("leader_vs_runner_up")
        if lv:
            out += ["", f"Leader vs runner-up: better on "
                    f"{lv['better_leader']} cell(s), worse on "
                    f"{lv['better_runner_up']}; McNemar p = "
                    f"{lv['p_value']:.4g}"
                    f"{' (significant)' if lv['significant'] else ''}."]
        out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kaggsim.worlds",
                                 description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("catalog", help="build a seed -> world catalog")
    c.add_argument("--seeds", default="0-19")
    c.add_argument("--a", default=None,
                   help="policy kind:seed for seat 0 (e.g. scripted:1); "
                        "omit both for the idle drive")
    c.add_argument("--b", default=None)
    c.add_argument("--k", type=int, default=2)
    c.add_argument("--mirror", action="store_true")
    c.add_argument("--out", default=None)
    r = sub.add_parser("screen", help="screen recorded continuations of a "
                                      "prefix tape per realized world")
    r.add_argument("sources", nargs="+",
                   help="replay .json files or single-seat tapes")
    r.add_argument("--prefix", required=True, help="prefix tape")
    r.add_argument("--opp", action="append", required=True,
                   help="opponent tape (repeatable)")
    r.add_argument("--seeds", default="0-49")
    r.add_argument("--seats", default="0,1")
    r.add_argument("--at-step", type=int, default=SECOND_SHOP_STEP)
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--top-k", type=int, default=10)
    r.add_argument("--out", default="screening",
                   help="output path without extension (.json + .md)")
    args = ap.parse_args(argv)
    if args.cmd == "screen":
        res = screen_continuations(
            args.sources, args.prefix, args.opp, parse_seeds(args.seeds),
            args.at_step, args.threads, args.top_k,
            tuple(int(x) for x in args.seats.split(",")), args.out)
        print(screening_markdown(res))
        return 0
    seeds = parse_seeds(args.seeds)
    if args.a or args.b:
        ka, sa = (args.a or "scripted:1").split(":")
        kb, sb = (args.b or "scripted:2").split(":")
        cat = build_catalog(lambda: make_policy(ka, int(sa)),
                            lambda: make_policy(kb, int(sb)), seeds,
                            args.k, args.mirror,
                            label=f"{args.a} vs {args.b}")
    else:
        cat = idle_catalog(seeds, args.k)
    text = json.dumps(cat, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    print(json.dumps(cat["by_world"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

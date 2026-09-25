"""Fidelity certifier: the Rust engine vs the official interpreter.

Two instruments:

* ``certify`` -- differential run. Episodes are driven on the OFFICIAL
  interpreter step by step (closed-loop policies see official observations);
  the exact action streams are then replayed through ``kagg episode`` and a
  full-state digest is compared after EVERY step. The digest covers both
  moneys as IEEE-754 bit patterns, every field of every tile, farmer and hand
  positions, hires, quadrants, both private blocks (shed, seeds, carried
  inventories), market inventory and prices, and the unlocked shops.
* ``diverge`` -- first-divergence tracer for two agent FILES: plays them on
  the official runner and on ``kagg serve`` (same seed, same seats), and
  reports the first step where the state digests or the emitted actions
  differ, with the differing fields.

    python -m kaggsim.fidelity certify --episodes 20
    python -m kaggsim.fidelity diverge a.py b.py --seed 3

Both engines receive byte-equivalent input: every action is normalised
through the tape encoding (``action_to_line`` -> ``line_to_action``) before
the official engine sees it.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import time

from . import official
from .binary import find_kagg
from .constants import FINAL_STEP
from .policies import make_policy
from .serve import Serve, call_agent, load_agent, structify
from .tape import action_to_line, line_to_action, write_episode_tape


# ----------------------------------------------------------------- digest --

def bits(x) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def _fmt_sorted(d) -> str:
    return ",".join(f"{k}={int(v)}" for k, v in sorted(d.items()))


def digest(step, farms, privates, market, town) -> str:
    """EXACT mirror of ``State::digest()`` in src-rust/kagg-engine/src/state.rs."""
    parts = [f"t{step}"]
    for i, farm in enumerate(farms):
        hands = " ".join(f"{p[0]},{p[1]}" for p in farm["hands"])
        s = (f"f{i}:m{bits(farm['money'])};"
             f"p{farm['farmer'][0]},{farm['farmer'][1]};"
             f"h{hands};r{farm['hires_today']};"
             f"q{','.join(farm['unlocked_quadrants'])};")
        cells = []
        for row in farm["tiles"]:
            for t in row:
                if t is None:
                    cells.append(".")
                elif t == "LOCKED":
                    cells.append("L")
                elif t.get("kind") == "WEED":
                    cells.append("W")
                elif t.get("kind") == "PLANT":
                    cells.append(
                        f"[P:{t['crop']},{t['planted_day']},"
                        f"{int(t['watered_today'])},"
                        f"{t['consecutive_unwatered']},{t['yield_units']},"
                        f"{t['max_lifespan_step']},"
                        f"{t.get('fertilized_until_day', -1)}]")
                elif "animal" in t:
                    cells.append(
                        f"[A:{t['kind']},{t['animal']},{t['placed_day']},"
                        f"{t['yield_units']},{t['consecutive_unfed']},"
                        f"{int(t['fed_today'])},{int(t['cared_today'])},"
                        f"{int(t['fertilizer_available'])},"
                        f"{t.get('pending_care_bonus', 0)}]")
                else:
                    cells.append(f"[S:{t['kind']}]")
        parts.append(s + "".join(cells))
    for i, priv in enumerate(privates):
        invs = "/".join(_fmt_sorted(v) for v in priv["inventories"])
        parts.append(f"s{i}:{_fmt_sorted(priv['shed'])};"
                     f"{_fmt_sorted(priv['seeds'])};{invs}")
    parts.append(f"mk:{_fmt_sorted(market['inventory'])};"
                 f"{_fmt_sorted(market['prices'])}")
    parts.append(f"tw:{','.join(town['unlocked_shops'])}")
    return "|".join(parts)


def digest_official(state) -> str:
    o = state[0].observation
    return digest(o.step, o.farms, [state[0].observation.private,
                                    state[1].observation.private],
                  o.market, o.town)


def digest_json(js: dict) -> str:
    return digest(js["step"], js["farms"], js["private"], js["market"],
                  js["town"])


def first_field_diff(a: str, b: str):
    """(field index, a_field, b_field) of the first differing digest field."""
    pa, pb = a.split("|"), b.split("|")
    for i, (x, y) in enumerate(zip(pa, pb)):
        if x != y:
            return i, x, y
    return min(len(pa), len(pb)), "<len>", "<len>"


# ------------------------------------------------------------ official run --

def _official_obs(env, seat):
    """The observation the official runner would hand the agent in ``seat``:
    the runner's own shared-state merge (``step``, ``farms``, ``market``...
    come from seat 0's state, ``private`` from the seat's own), as an
    isolated, attribute-accessible copy."""
    shared = env._Environment__get_shared_state(seat).observation
    return structify(json.loads(json.dumps(shared)))


def run_official(seed: int, policy0, policy1, steps: int = FINAL_STEP):
    """Drive the official interpreter with env.step, recording normalised
    actions and the post-step digest. Returns (lines0, lines1, digests,
    banks)."""
    env = official.make(seed, actTimeout=60, runTimeout=10 ** 6)
    with contextlib.redirect_stdout(io.StringIO()):
        env.reset(2)
    lines0, lines1, digests = [], [], []
    while not env.done and len(digests) < steps:
        acts = []
        for seat, pol in ((0, policy0), (1, policy1)):
            a = call_agent(pol, _official_obs(env, seat))
            acts.append(action_to_line(a))
        lines0.append(acts[0])
        lines1.append(acts[1])
        with contextlib.redirect_stdout(io.StringIO()):
            env.step([line_to_action(acts[0]), line_to_action(acts[1])])
        digests.append(digest_official(env.state))
    o = env.state[0].observation
    banks = (float(o.farms[0]["money"]), float(o.farms[1]["money"]))
    return lines0, lines1, digests, banks


def run_official_lines(seed: int, lines0, lines1):
    """Official interpreter on fixed tape lines. Returns (digests, banks)."""
    it0, it1 = iter(lines0), iter(lines1)
    return run_official(seed, lambda o: line_to_action(next(it0)),
                        lambda o: line_to_action(next(it1)),
                        steps=min(len(lines0), len(lines1)))[2:]


def run_rust(seed: int, lines0, lines1, kagg=None, workdir=None):
    """``kagg episode`` on the same streams. Returns (digests, banks)."""
    exe = find_kagg(kagg)
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        path = os.path.join(tmp, "episode.tape")
        write_episode_tape(path, seed, lines0, lines1)
        out = subprocess.run([exe, "episode", path], capture_output=True,
                             text=True, timeout=300)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[:400])
    digests, banks = [], None
    for line in out.stdout.splitlines():
        if line.startswith("FINAL "):
            _, b0, b1 = line.split()
            banks = tuple(struct.unpack("<d", struct.pack("<Q", int(b)))[0]
                          for b in (b0, b1))
        else:
            digests.append(line.split(" ", 1)[1])
    return digests, banks


def compare_episode(name, seed, policy0, policy1, kagg=None, log=print):
    """Differential check of one episode. Returns a result dict."""
    t0 = time.time()
    l0, l1, py, py_banks = run_official(seed, policy0, policy1)
    t1 = time.time()
    rs, rs_banks = run_rust(seed, l0, l1, kagg)
    res = {"name": name, "seed": seed, "steps": len(py), "ok": True,
           "banks": py_banks, "official_s": round(t1 - t0, 2)}
    for k in range(min(len(py), len(rs))):
        if py[k] != rs[k]:
            i, a, b = first_field_diff(py[k], rs[k])
            res.update(ok=False, diverged_at=k + 1, field=i,
                       official=a[:240], rust=b[:240])
            break
    else:
        if len(py) != len(rs):
            res.update(ok=False, reason=f"lengths {len(py)} vs {len(rs)}")
        elif py_banks != rs_banks:
            res.update(ok=False, reason=f"banks {py_banks} vs {rs_banks}")
    if log:
        if res["ok"]:
            log(f"  {name}: {res['steps']} steps identical, banks "
                f"{py_banks[0]:,.0f} / {py_banks[1]:,.0f}")
        else:
            log(f"  {name}: DIVERGED {json.dumps(res)}")
    return res


def episode_plan(n: int, base_seed: int = 1000):
    """A deterministic mix of (name, seed, policy0, policy1) episodes."""
    kinds = [("chaos", "chaos"), ("scripted", "scripted"),
             ("random", "random"), ("scripted", "chaos"),
             ("random", "scripted"), ("last_turn", "random"),
             ("chaos", "scripted"), ("random", "last_turn")]
    plan = []
    for e in range(n):
        k0, k1 = kinds[e % len(kinds)]
        seed = base_seed + 7919 * e
        plan.append((f"{k0}-vs-{k1}_{e}", seed,
                     make_policy(k0, 2 * e + 1), make_policy(k1, 2 * e + 2)))
    return plan


def certify(n: int = 20, base_seed: int = 1000, kagg=None, log=print):
    official.engine_module()
    results = [compare_episode(name, seed, p0, p1, kagg, log)
               for name, seed, p0, p1 in episode_plan(n, base_seed)]
    ok = sum(r["ok"] for r in results)
    steps = sum(r["steps"] for r in results)
    if log:
        log(f"{ok}/{len(results)} episodes identical "
            f"({steps} step digests compared)")
    return results


# ---------------------------------------------------------------- diverge --

def diverge(path0: str, path1: str, seed: int, kagg=None, log=print):
    """First step where official and serve differ for two agent files."""
    rec = {0: [], 1: []}

    def wrap(fn, seat):
        def g(obs, configuration=None):
            a = call_agent(fn, obs) if configuration is None else \
                fn(obs, configuration) if fn.__code__.co_argcount > 1 \
                else fn(obs)
            rec[seat].append(json.loads(json.dumps(a)))
            return a
        return g

    banks_off, env = official.run_agents(wrap(load_agent(path0), 0),
                                         wrap(load_agent(path1), 1), seed)
    off = [digest(s[0]["observation"]["step"], s[0]["observation"]["farms"],
                  [s[0]["observation"]["private"],
                   s[1]["observation"]["private"]],
                  s[0]["observation"]["market"], s[0]["observation"]["town"])
           for s in env.steps[1:]]
    with Serve(kagg) as srv, contextlib.redirect_stdout(io.StringIO()):
        from .serve import run_match
        banks_rs, trace = run_match(load_agent(path0), load_agent(path1),
                                    seed, srv, record=True)
    for i, t in enumerate(trace):
        d_rs = digest_json(t["state"])
        d_off = off[i] if i < len(off) else None
        act_diff = [s for s in (0, 1) if i < len(rec[s]) and
                    json.dumps(rec[s][i], sort_keys=True)
                    != json.dumps(t["actions"][s], sort_keys=True)]
        if d_off != d_rs or act_diff:
            out = {"step": t["step"], "action_index": i,
                   "actions_differ_for_seats": act_diff,
                   "banks_official": banks_off, "banks_rust": banks_rs}
            if d_off is None:
                out["reason"] = "official episode is shorter"
            elif d_off != d_rs:
                f, a, b = first_field_diff(d_off, d_rs)
                out.update(field=f, official=a[:400], rust=b[:400])
            for s in act_diff:
                out[f"seat{s}_official_action"] = rec[s][i]
                out[f"seat{s}_rust_action"] = t["actions"][s]
            if log:
                log("FIRST DIVERGENCE " + json.dumps(out, indent=1))
            return out
    if log:
        log(f"no divergence across {len(trace)} steps; banks official "
            f"{banks_off} rust {banks_rs}")
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kaggsim.fidelity",
                                 description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("certify", help="differential run over synthetic "
                                       "episodes")
    c.add_argument("--episodes", type=int, default=20)
    c.add_argument("--base-seed", type=int, default=1000)
    c.add_argument("--kagg", default=None)
    c.add_argument("--json", default=None, help="write results here")
    d = sub.add_parser("diverge", help="first divergence of two agent files")
    d.add_argument("agent0")
    d.add_argument("agent1")
    d.add_argument("--seed", type=int, default=3)
    d.add_argument("--kagg", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "certify":
        res = certify(args.episodes, args.base_seed, args.kagg)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(res, fh, indent=1)
        return 0 if all(r["ok"] for r in res) else 1
    out = diverge(args.agent0, args.agent1, args.seed, args.kagg)
    return 1 if out else 0


if __name__ == "__main__":
    sys.exit(main())

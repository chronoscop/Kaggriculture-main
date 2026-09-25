"""Replay economics: where each seat's money came from and went.

For every step of a replay the recorded pre-step state and both recorded
actions are fed through the INSTALLED OFFICIAL interpreter (so the price
curve, lockstep market rules, shed capacity and hire costs are the engine's
own, not a re-implementation), with the market commit functions wrapped to
attribute every unit traded. Each step is then reconciled: the simulated
post-step money must equal the replay's recorded money for both seats, and
the report says how many steps reconciled.

    python -m kaggsim.economy replay.json [more.json ...] [--out rows.json]

Per seat the report holds revenue and units by product, the realised price as
a fraction of the product's base price, spend on seeds / animals / products /
hires / land, and the steps at which land was bought.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import statistics
import sys

from . import official
from .serve import structify
from .tape import load_replay, replay_seed


class _Env:
    def __init__(self, configuration, seed):
        cfg = dict(official.DEFAULT_CONFIGURATION)
        cfg.update({k: v for k, v in (configuration or {}).items()
                    if v is not None})
        self.configuration = structify(cfg)
        self.info = {"seed": seed}
        self.done = False


class _Seat:
    def __init__(self, observation, action):
        self.observation = observation
        self.action = action
        self.status = "ACTIVE"
        self.reward = 0


def _money(farms):
    return [float(f["money"]) for f in farms]


def analyze_replay(rep: dict, mod=None) -> list:
    """Per-seat economy rows for one replay dict (official replay schema)."""
    mod = mod or official.engine_module()
    steps = rep["steps"]
    seed = replay_seed(rep)
    env = _Env(rep.get("configuration"), seed)
    base = {k: v["base"] for k, v in mod.MARKET_PARAMS.items()}
    rows = [{"rev": collections.Counter(), "units": collections.Counter(),
             "spend": collections.Counter(), "land_steps": [], "hires": 0}
            for _ in (0, 1)]
    ctx = {"farms": None, "step": 0}
    real_commit, real_hire, real_land = (mod._commit_unit, mod._do_hire,
                                         mod._do_buy_land)

    def seat_of(farm):
        for i, f in enumerate(ctx["farms"]):
            if f is farm:
                return i
        return None

    def commit(op, item, price, farm, private, market, *a, **k):
        ok = real_commit(op, item, price, farm, private, market, *a, **k)
        p = seat_of(farm)
        if ok and p is not None:
            if op == "SELL":
                rows[p]["rev"][item] += price
                rows[p]["units"][item] += 1
            elif op == "BUY_SEED":
                rows[p]["spend"]["seeds"] += price
            elif op == "BUY_ANIMAL":
                rows[p]["spend"]["animals"] += price
            elif op == "BUY_PRODUCT":
                rows[p]["spend"]["products"] += price
        return ok

    def hire(farm, *a, **k):
        before = float(farm["money"])
        out = real_hire(farm, *a, **k)
        p = seat_of(farm)
        if p is not None and float(farm["money"]) < before:
            rows[p]["spend"]["hires"] += before - float(farm["money"])
            rows[p]["hires"] += 1
        return out

    def land(farm, *a, **k):
        before = float(farm["money"])
        out = real_land(farm, *a, **k)
        p = seat_of(farm)
        if p is not None and float(farm["money"]) < before:
            rows[p]["spend"]["land"] += before - float(farm["money"])
            rows[p]["land_steps"].append(ctx["step"])
        return out

    reconciled = 0
    mismatched = []
    mod._commit_unit, mod._do_hire, mod._do_buy_land = commit, hire, land
    try:
        for i in range(len(steps) - 1):
            pre, post = steps[i], steps[i + 1]
            obs = [structify(copy.deepcopy(pre[p]["observation"]))
                   for p in (0, 1)]
            acts = [post[p].get("action") if isinstance(
                post[p].get("action"), dict) else {} for p in (0, 1)]
            state = [_Seat(obs[p], acts[p]) for p in (0, 1)]
            ctx["farms"] = obs[0].farms
            ctx["step"] = obs[0].step
            mod.interpreter(state, env)
            want = _money(post[0]["observation"]["farms"])
            if _money(obs[0].farms) == want:
                reconciled += 1
            else:
                mismatched.append(i)
    finally:
        mod._commit_unit, mod._do_hire, mod._do_buy_land = (
            real_commit, real_hire, real_land)

    last = steps[-1]
    banks = _money(last[0]["observation"]["farms"])
    names = (rep.get("info") or {}).get("TeamNames") or ["seat0", "seat1"]
    town = last[0]["observation"].get("town") or {}
    out = []
    for p in (0, 1):
        r = rows[p]
        out.append({
            "seat": p, "name": names[p] if p < len(names) else f"seat{p}",
            "bank": banks[p], "won": banks[p] > banks[1 - p],
            "first_shops": "|".join((town.get("unlocked_shops") or [])[:2]),
            "revenue": dict(r["rev"]), "revenue_total": sum(r["rev"].values()),
            "units_sold": dict(r["units"]),
            "price_ratio": {k: round(r["rev"][k] / r["units"][k] / base[k], 4)
                            for k in r["units"] if r["units"][k]},
            "spend": dict(r["spend"]), "hires": r["hires"],
            "land_steps": r["land_steps"],
            "steps_reconciled": reconciled, "steps_total": len(steps) - 1,
            "mismatched_steps": mismatched[:20],
        })
    return out


def summarise(rows) -> dict:
    """Averages over winners (seats that finished ahead)."""
    win = [r for r in rows if r["won"]]
    prods = sorted({k for r in rows for k in r["revenue"]})
    if not win:
        return {"players": len(rows), "winners": 0}
    return {
        "players": len(rows), "winners": len(win),
        "winners_revenue_by_product": {
            k: round(statistics.mean(r["revenue"].get(k, 0) for r in win))
            for k in prods},
        "winners_price_ratio": {
            k: round(statistics.mean(r["price_ratio"][k] for r in win
                                     if k in r["price_ratio"]), 3)
            for k in prods if any(k in r["price_ratio"] for r in win)},
        "winners_bank": round(statistics.mean(r["bank"] for r in win)),
        "winners_first_land_step": (round(statistics.mean(
            r["land_steps"][0] for r in win if r["land_steps"]))
            if any(r["land_steps"] for r in win) else None),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kaggsim.economy",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("replays", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    mod = official.engine_module()
    rows = []
    for path in args.replays:
        try:
            rows += analyze_replay(load_replay(path), mod)
        except (KeyError, ValueError) as exc:
            print(f"skip {path}: {exc}", file=sys.stderr)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
    print(json.dumps(summarise(rows), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

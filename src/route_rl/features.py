"""Projected map, resources, time and dependencies; no profit-based pruning."""
from .routes import CROPS, ANIMALS, ITEMS, PRODUCTS, OPS, farm_of, positions, tile_at, cost

FEATURE_SIZE = 96
STATE_SIZE = 3200
SCHEMA = "dynamic-routes-v2"

def padded(v, size):
    if len(v) > size:
        raise ValueError(f"feature size {len(v)} exceeds {size}")
    return v + [0.0] * (size - len(v))

def tile_features(tile, day):
    kind = tile.get("kind") if isinstance(tile, dict) else (
        "EMPTY" if tile is None else tile)
    t = tile if isinstance(tile, dict) else {}
    v = [float(kind == k) for k in ("EMPTY", "LOCKED", "WEED", "PLANT", "COOP", "PASTURE")]
    v += [float(t.get("crop") == c) for c in CROPS]
    v += [float(t.get("animal") == a) for a in ANIMALS]
    v += [(day - t.get("planted_day", day)) / 30,
          (day - t.get("placed_day", day)) / 30,
          t.get("yield_units", 0) / 10, float(t.get("watered_today", False)),
          float(t.get("fed_today", False)), float(t.get("cared_today", False)),
          float(t.get("fertilizer_available", False)),
          (t.get("fertilized_until_day", -1) - day) / 30,
          t.get("consecutive_unwatered", 0) / 2,
          t.get("consecutive_unfed", 0) / 2,
          t.get("pending_care_bonus", 0) / 4,
          t.get("max_lifespan_step", -1) / 719]
    return v

def state_features(plan):
    obs, farm = plan.obs, farm_of(plan.obs)
    priv = obs["private"]
    v = [obs["step"] / 719, obs["hour"] / 24, plan.limit / 24,
         farm["money"] / 10000, obs["farms"][1 - obs["player"]]["money"] / 10000,
         plan.sale_credit / 10000, farm.get("hires_today", 0) / 20,
         len(farm["unlocked_quadrants"]) / 4, len(positions(obs)) / 20,
         len(obs["town"].get("unlocked_shops", [])) / 8]
    v += [obs["market"]["prices"].get(p, 0) / 200 for p in PRODUCTS]
    v += [obs["market"].get("inventory", {}).get(p, 0) / 1000 for p in PRODUCTS]
    v += [priv["seeds"].get(c, 0) / 20 for c in CROPS]
    v += [priv["shed"].get(p, 0) / 100 for p in ITEMS]
    v += [sum(b.get(p, 0) for b in priv["inventories"]) / 100 for p in ITEMS]
    workers = positions(obs)
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            v += tile_features(tile, obs["day"])
            here = [i for i, p in enumerate(workers) if p == (x, y)]
            owner = plan.owners.get(("tile", x, y))
            v += [len(here) / 20, sum(plan.elapsed[i] for i in here) / 480,
                  sum(i in plan.ended for i in here) / 20,
                  (plan.jobs[owner].finish / 24 if owner is not None else 0)]
    return padded(v, STATE_SIZE)

def encode(plan, job):
    obs, actor = plan.obs, job.actor
    start = positions(obs)[actor]
    op = job.op[0]
    item = job.op[1] if len(job.op) > 1 else None
    v = [float(op == k) for k in OPS] + [float(item == p) for p in ITEMS]
    v += tile_features(tile_at(obs, job.pos), obs["day"])
    v += [obs["private"]["inventories"][actor].get(p, 0) / 20 for p in ITEMS]
    v += [actor / 20, start[0] / 10, start[1] / 10, job.pos[0] / 10, job.pos[1] / 10,
          plan.elapsed[actor] / 24, job.finish / 24, (plan.limit - job.finish) / 24,
          len(job.deps) / 6, max((plan.jobs[d].finish for d in job.deps), default=0) / 24,
          cost(obs, job.op) / 10000, (job.op[2] / 100 if len(job.op) > 2 else 0),
          float(job.market), plan.sale_credit / 10000]
    return padded(v, FEATURE_SIZE)

def menu_features(plan, jobs):
    return [encode(plan, j) for j in jobs], [True] * len(jobs)

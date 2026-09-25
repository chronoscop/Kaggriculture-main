"""Autoregressive same-day planning with shared resource reservations."""
from __future__ import annotations
import copy
from dataclasses import dataclass, replace

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
SEED_COST = dict(zip(CROPS, (10, 20, 50, 100, 80)))
FIRST_YIELD = dict(zip(CROPS, (2, 2, 8, 10, 10)))
MAX_DAY = dict(zip(CROPS, (4, 3, 8, 10, 12)))
MAX_YIELD = dict(zip(CROPS, (6, 4, 4, 4, 6)))
ANIMALS = {"GOOSE": ("COOP", 300), "COW": ("PASTURE", 400), "SHEEP": ("PASTURE", 500)}
PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
PRODUCTS = CROPS + ("EGG", "MILK", "WOOL", "FERTILIZER")
ITEMS = PRODUCTS + tuple(ANIMALS)
SHEDS = ((4, 4), (5, 4), (4, 5), (5, 5))
MARKET_OPS = ("SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "HIRE", "BUY_LAND")
OPS = ("END", "WAIT", "HARVEST", "PLANT", "WATER", "FERTILIZE",
       "COLLECT_FERTILIZER", "FEED", "CARE", "DIG", "BUILD_COOP",
       "BUILD_PASTURE", "PICKUP", "PLACE", "DROP") + MARKET_OPS

def farm_of(obs):
    return obs["farms"][obs["player"]]

def tile_at(obs, pos):
    return farm_of(obs)["tiles"][pos[1]][pos[0]]

def animal(tile):
    return isinstance(tile, dict) and tile.get("animal") in ANIMALS

def positions(obs):
    farm = farm_of(obs)
    return [tuple(farm["farmer"])] + [tuple(p) for p in farm["hands"]]

def ripe(tile, day):
    return (isinstance(tile, dict) and tile.get("yield_units", 0) > 0 and
            (animal(tile) or (tile.get("kind") == "PLANT" and
             day - tile["planted_day"] >= FIRST_YIELD[tile["crop"]])))

def cost(obs, op):
    if op[0] == "BUY_SEED":
        return SEED_COST[op[1]] * op[2]
    if op[0] == "BUY_ANIMAL":
        return ANIMALS[op[1]][1] * op[2]
    if op[0] == "BUY_PRODUCT":
        return max(1, obs["market"]["prices"].get(op[1], 1)) * op[2]
    if op[0] == "HIRE":
        a, b = 1, 1
        for _ in range(farm_of(obs).get("hires_today", 0)):
            a, b = b, a + b
        return a
    if op[0] == "BUY_LAND":
        n = len(farm_of(obs)["unlocked_quadrants"]) - 1
        return (1000, 2000, 4000)[n] if n < 3 else float("inf")
    return 0

@dataclass(frozen=True)
class Job:
    actor: int
    pos: tuple[int, int]
    op: tuple
    id: int = -1
    deps: tuple[int, ...] = ()
    finish: int = 0

    @property
    def market(self):
        return self.op[0] in MARKET_OPS

def resources(job):
    """Read/write locks serialize conflicting tasks, not independent travel."""
    op = job.op[0]
    keys = [("actor", job.actor)]
    if op not in MARKET_OPS + ("END", "WAIT", "DROP", "PICKUP"):
        keys.append(("tile", *job.pos))
    if op in ("DROP", "PICKUP", "SELL", "BUY_PRODUCT", "BUY_ANIMAL"):
        keys.append(("shed",))
    if op in ("BUY_SEED", "PLANT"):
        keys.append(("seed", job.op[1]))
    if job.market:
        keys.append(("cash",))
    return keys

def legal(obs, job):
    farm, priv = farm_of(obs), obs["private"]
    op = job.op[0]
    bag, shed = priv["inventories"][job.actor], priv["shed"]
    tile = tile_at(obs, job.pos)
    if op in ("END", "WAIT"):
        return True
    if job.market:
        if op == "SELL":
            return job.op[1] in PRODUCTS and shed.get(job.op[1], 0) >= job.op[2]
        return (farm["money"] >= cost(obs, job.op) and
                (op not in ("BUY_PRODUCT", "BUY_ANIMAL") or sum(shed.values()) < 100))
    if op == "DROP":
        return job.pos in SHEDS and 0 < sum(bag.values()) <= 100 - sum(shed.values())
    if op == "PICKUP":
        return job.pos in SHEDS and shed.get(job.op[1], 0) >= job.op[2]
    if op == "PLANT":
        return tile is None and priv["seeds"].get(job.op[1], 0) > 0
    if op.startswith("BUILD_"):
        return tile is None
    if op == "DIG":
        return tile is not None and tile != "LOCKED" and not animal(tile)
    if op == "HARVEST":
        return ripe(tile, obs["day"])
    if op == "PLACE":
        return (isinstance(tile, dict) and tile.get("kind") == ANIMALS[job.op[1]][0]
                and not animal(tile) and bag.get(job.op[1], 0) > 0)
    if op in ("WATER", "FERTILIZE"):
        return (isinstance(tile, dict) and tile.get("kind") == "PLANT" and
                (not tile.get("watered_today") if op == "WATER" else bag.get("FERTILIZER", 0) > 0))
    if not animal(tile):
        return False
    if op == "FEED":
        return not tile.get("fed_today") and bag.get("WHEAT", 0) > 0
    if op == "CARE":
        return not tile.get("cared_today")
    return op == "COLLECT_FERTILIZER" and bool(tile.get("fertilizer_available"))

def project(obs, job):
    """Apply an immediate job to a private copy. Market cash is an estimate."""
    farm, priv = farm_of(obs), obs["private"]
    bag, shed = priv["inventories"][job.actor], priv["shed"]
    op, tile = job.op[0], tile_at(obs, job.pos)
    x, y = job.pos
    if not job.market:
        if job.actor == 0:
            farm["farmer"] = list(job.pos)
        else:
            farm["hands"][job.actor - 1] = list(job.pos)
    def add(inv, item, n):
        inv[item] = inv.get(item, 0) + n
    if op in ("END", "WAIT"):
        return
    if op == "SELL":
        add(shed, job.op[1], -job.op[2])
        farm["money"] += job.op[2] * obs["market"]["prices"].get(job.op[1], 1)
    elif job.market:
        farm["money"] -= cost(obs, job.op)
        if op == "BUY_SEED":
            add(priv["seeds"], job.op[1], job.op[2])
        elif op in ("BUY_ANIMAL", "BUY_PRODUCT"):
            add(shed, job.op[1], job.op[2])
        # New workers / land become available only after actual receipt.
    elif op == "DROP":
        for item, n in bag.items():
            add(shed, item, n)
        bag.clear()
    elif op == "PICKUP":
        add(shed, job.op[1], -job.op[2])
        add(bag, job.op[1], job.op[2])
    elif op == "PLANT":
        add(priv["seeds"], job.op[1], -1)
        farm["tiles"][y][x] = dict(kind="PLANT", crop=job.op[1], planted_day=obs["day"],
            yield_units=0 if job.op[1] in ("TOMATO", "STRAWBERRY") else 1,
            watered_today=False, fertilized_until_day=-1, consecutive_unwatered=1)
    elif op.startswith("BUILD_"):
        farm["tiles"][y][x] = {"kind": op[6:]}
    elif op == "DIG":
        farm["tiles"][y][x] = None
    elif op == "PLACE":
        add(bag, job.op[1], -1)
        tile.update(animal=job.op[1], placed_day=obs["day"], yield_units=0,
                    fed_today=False, cared_today=False, fertilizer_available=False,
                    consecutive_unfed=0, pending_care_bonus=0)
    elif op == "HARVEST":
        item = PRODUCT[tile["animal"]] if animal(tile) else tile["crop"]
        add(bag, item, tile["yield_units"])
        tile["yield_units"] = 0
        if tile.get("kind") == "PLANT" and item not in ("TOMATO", "STRAWBERRY"):
            farm["tiles"][y][x] = None
    elif op == "WATER":
        tile["watered_today"] = True
        crop, age = tile["crop"], obs["day"] - tile["planted_day"]
        if crop not in ("TOMATO", "STRAWBERRY") and (MAX_DAY[crop] + 1) // 2 <= age <= MAX_DAY[crop]:
            bonus = 2 if tile.get("fertilized_until_day", -1) >= obs["day"] else 1
            tile["yield_units"] = min(MAX_YIELD[crop], tile["yield_units"] + bonus)
    elif op == "FERTILIZE":
        add(bag, "FERTILIZER", -1)
        tile["fertilized_until_day"] = max(tile.get("fertilized_until_day", -1), obs["day"] + 2)
    elif op == "FEED":
        add(bag, "WHEAT", -1)
        tile["fed_today"] = True
    elif op == "CARE":
        tile["cared_today"] = True
    elif op == "COLLECT_FERTILIZER":
        tile["fertilizer_available"] = False
        add(bag, "FERTILIZER", 1)

class PlanningState:
    def __init__(self, obs, horizon=24):
        self.obs = copy.deepcopy(obs)
        self.limit = min(horizon, 24 - obs["hour"], 719 - obs["step"])
        self.elapsed = [0] * len(positions(obs))
        self.ended, self.jobs, self.owners, self.expansion = set(), [], {}, set()
        self.sale_credit = 0.0

    def schedule(self, job):
        deps = tuple(sorted({self.owners[k] for k in resources(job) if k in self.owners}))
        start = positions(self.obs)[job.actor]
        distance = 0 if job.market else abs(start[0] - job.pos[0]) + abs(start[1] - job.pos[1])
        finish = max(self.elapsed[job.actor] + distance,
                     max((self.jobs[d].finish for d in deps), default=0)) + 1
        return replace(job, deps=deps, finish=finish)

    def candidates(self, actor):
        obs, start = self.obs, positions(self.obs)[actor]
        candidates = [Job(actor, start, ("END",)), Job(actor, start, ("WAIT",))]
        for y, row in enumerate(farm_of(obs)["tiles"]):
            for x, tile in enumerate(row):
                if tile is None:
                    ops = [("PLANT", c) for c in CROPS] + [("BUILD_COOP",), ("BUILD_PASTURE",)]
                elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    ops = [("WATER",), ("HARVEST",), ("FERTILIZE",), ("DIG",)]
                elif animal(tile):
                    ops = [("FEED",), ("CARE",), ("HARVEST",), ("COLLECT_FERTILIZER",)]
                elif tile != "LOCKED":
                    ops = [("DIG",)] + [("PLACE", a) for a in ANIMALS]
                else:
                    ops = []
                candidates.extend(Job(actor, (x, y), op) for op in ops)
        for pos in SHEDS:
            candidates.append(Job(actor, pos, ("DROP",)))
            for item, n in obs["private"]["shed"].items():
                if n > 0:
                    candidates.append(Job(actor, pos, ("PICKUP", item, 1)))
                    if n > 1:
                        candidates.append(Job(actor, pos, ("PICKUP", item, n)))
        market = [("BUY_SEED", c, 1) for c in CROPS]
        market += [("BUY_ANIMAL", a, 1) for a in ANIMALS]
        market += [("BUY_PRODUCT", p, 1) for p in ("WHEAT", "FERTILIZER")]
        for item, n in obs["private"]["shed"].items():
            if item in PRODUCTS and n > 0:
                market.append(("SELL", item, 1))
                if n > 1:
                    market.append(("SELL", item, n))
        market += [(op,) for op in ("HIRE", "BUY_LAND") if op not in self.expansion]
        candidates.extend(Job(actor, start, op) for op in market)
        result = []
        for job in candidates:
            if legal(obs, job):
                job = self.schedule(job)
                if job.op[0] == "END" or job.finish <= self.limit:
                    result.append(job)
        return result

    def commit(self, job):
        if job.op[0] == "END":
            self.ended.add(job.actor)
            return
        job = replace(job, id=len(self.jobs))
        if job.op[0] == "SELL":
            self.sale_credit += job.op[2] * self.obs["market"]["prices"].get(job.op[1], 1)
        project(self.obs, job)
        self.elapsed[job.actor] = job.finish
        self.jobs.append(job)
        for key in resources(job):
            self.owners[key] = job.id
        if job.op[0] in ("HIRE", "BUY_LAND"):
            self.expansion.add(job.op[0])
        if job.finish >= self.limit:
            self.ended.add(job.actor)

def command_toward(current, target):
    x, y = current
    tx, ty = target
    if x != tx:
        return ["EAST" if x < tx else "WEST"]
    if y != ty:
        return ["SOUTH" if y < ty else "NORTH"]
    return None

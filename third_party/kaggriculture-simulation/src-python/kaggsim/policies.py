"""Synthetic policies for fixtures, tests and examples.

None of these is meant to play well. They exist to drive the engine through
as many code paths as possible so the Rust port can be compared with the
official interpreter:

* :func:`chaos_action` -- open-loop soup of every op, legal or not, plus
  malformed and empty (``[]``) market orders and hand slots.
* :class:`RandomPolicy` -- observation-aware random play: sensible op mix,
  counts drawn against what the seat actually holds.
* :class:`ScriptedFarmer` -- a closed-loop farmer that plants, waters,
  harvests, deposits and sells, buys seeds/animals, hires hands and buys land,
  so the market, shed and animal paths see realistic volumes.
* :class:`LastTurnSeller` -- hoards until the final turns, then sells: the
  policy whose bank depends on the exact episode length.

Every policy is deterministic given its constructor seed and the sequence
of observations it receives.
"""
from __future__ import annotations

import random

from .constants import ANIMALS, CROPS, FINAL_STEP, MOVES, PRODUCTS

_UNIT_SIMPLE = ["WATER", "HARVEST", "DROP", "FEED", "CARE",
                "COLLECT_FERTILIZER", "FERTILIZE", "DIG", "BUILD_COOP",
                "BUILD_PASTURE", "PASS"]


def _get(o, k, default=None):
    if isinstance(o, dict):
        return o.get(k, default)
    return getattr(o, k, default)


# ----------------------------------------------------------------- chaos --

def chaos_unit(rng: random.Random):
    r = rng.random()
    if r < 0.22:
        return [rng.choice(MOVES)]
    if r < 0.36:
        return ["WATER"]
    if r < 0.46:
        return ["HARVEST"]
    if r < 0.56:
        return ["PLANT", rng.choice(CROPS)]
    if r < 0.63:
        return ["DROP"]
    if r < 0.70:
        return ["PICKUP", rng.choice(PRODUCTS + ANIMALS), rng.randint(1, 5)]
    if r < 0.75:
        return (["PLACE", rng.choice(ANIMALS)] if rng.random() < 0.5
                else ["PLACE", rng.choice(PRODUCTS), rng.randint(1, 4)])
    if r < 0.97:
        return [rng.choice(_UNIT_SIMPLE)]
    return []


def chaos_order(rng: random.Random):
    r = rng.random()
    if r < 0.22:
        return ["BUY_SEED", rng.choice(CROPS), rng.randint(1, 6)]
    if r < 0.50:
        return ["SELL", rng.choice(PRODUCTS), rng.randint(1, 30)]
    if r < 0.64:
        return ["BUY_PRODUCT", rng.choice(["WHEAT", "FERTILIZER"]),
                rng.randint(1, 4)]
    if r < 0.74:
        return ["BUY_ANIMAL", rng.choice(ANIMALS), 1]
    if r < 0.84:
        return ["HIRE"]
    if r < 0.89:
        return ["BUY_LAND"]
    if r < 0.95:
        return []                       # positional placeholder
    # Malformed on purpose: both engines must ignore it identically.
    return rng.choice([["SELL", "GOOSE", 3], ["SELL"],
                       ["BUY_PRODUCT", "MILK", 2], ["NONSENSE", "X", 1],
                       ["SELL", "WHEAT", 0], ["BUY_SEED", "WHEAT", -2]])


def chaos_action(rng: random.Random) -> dict:
    """One seat's turn of seeded chaos (open loop)."""
    return {"farmer": chaos_unit(rng),
            "hands": [chaos_unit(rng) for _ in range(rng.randint(0, 3))],
            "market": [chaos_order(rng) for _ in range(rng.randint(0, 4))]}


class ChaosPolicy:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def __call__(self, obs, configuration=None):
        return chaos_action(self.rng)


# ---------------------------------------------------------------- random --

class RandomPolicy:
    """Observation-aware random play (valid shapes, random intent)."""

    def __init__(self, seed: int = 0, market_rate: float = 0.5):
        self.rng = random.Random(seed)
        self.market_rate = market_rate

    def __call__(self, obs, configuration=None):
        rng = self.rng
        me = _get(obs, "player", 0)
        farm = _get(obs, "farms")[me]
        priv = _get(obs, "private") or {}
        shed = _get(priv, "shed") or {}
        seeds = _get(priv, "seeds") or {}

        def unit():
            r = rng.random()
            if r < 0.35:
                return [rng.choice(MOVES)]
            if r < 0.55:
                return ["WATER"]
            if r < 0.65:
                return ["HARVEST"]
            if r < 0.75:
                held = [c for c in CROPS if seeds.get(c, 0) > 0]
                return ["PLANT", rng.choice(held or CROPS)]
            if r < 0.82:
                return ["DROP"]
            return [rng.choice(_UNIT_SIMPLE)]

        hands = [unit() if rng.random() < 0.9 else []
                 for _ in _get(farm, "hands") or []]
        market = []
        money = _get(farm, "money", 0)
        if rng.random() < self.market_rate:
            for _ in range(rng.randint(1, 3)):
                stocked = [p for p in PRODUCTS if shed.get(p, 0) > 0]
                r = rng.random()
                if stocked and r < 0.5:
                    p = rng.choice(stocked)
                    market.append(["SELL", p, rng.randint(1, shed[p])])
                elif r < 0.8 and money > 400:
                    market.append(["BUY_SEED", rng.choice(CROPS),
                                   rng.randint(1, 4)])
                elif r < 0.9:
                    market.append([])
                elif money > 200:
                    market.append(["HIRE"])
        return {"farmer": unit(), "hands": hands, "market": market}


# -------------------------------------------------------------- scripted --

def _step_toward(pos, target):
    (x, y), (tx, ty) = pos, target
    if x < tx:
        return "EAST"
    if x > tx:
        return "WEST"
    if y < ty:
        return "SOUTH"
    if y > ty:
        return "NORTH"
    return None


class ScriptedFarmer:
    """Closed-loop farmer: plant -> water -> harvest -> deposit -> sell.

    Works a fixed plot in the NW quadrant with a chosen crop, keeps sheep or
    geese when it can afford a structure, hires on some days, and buys land
    once rich. Parameters randomise the plot size, crop mix and timing so
    different instances exercise different code paths.
    """

    SHED_TILE = (4, 4)

    def __init__(self, seed: int = 0):
        rng = random.Random(seed)
        self.rng = rng
        self.crop = rng.choice(["WHEAT", "WHEAT", "CARROT", "TOMATO",
                                "STRAWBERRY", "MELON"])
        size = rng.randint(3, 8)
        tiles = [(x, y) for y in range(4) for x in range(4)]
        rng.shuffle(tiles)
        self.plot = sorted(tiles[:size], key=lambda t: (t[1], t[0]))
        self.animal = rng.choice([None, "GOOSE", "COW", "SHEEP"])
        self.hire_days = {d for d in range(30) if rng.random() < 0.3}
        self.land_money = rng.choice([4000, 8000, 10 ** 9])
        self.sell_every = rng.choice([1, 3, 24])

    STRUCT_TILE = (4, 0)
    FIRST_YIELD = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8, "STRAWBERRY": 10,
                   "MELON": 10}
    STRUCTURE = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}

    def _tile(self, farm, x, y):
        return _get(farm, "tiles")[y][x]

    def _go(self, pos, target, op):
        mv = _step_toward(pos, target)
        return [mv] if mv else op

    def _animal_task(self, farm, shed, inv, pos, day):
        if not self.animal:
            return None
        sx, sy = self.STRUCT_TILE
        t = self._tile(farm, sx, sy)
        if inv.get(self.animal, 0) > 0:
            return self._go(pos, (sx, sy), ["PLACE", self.animal])
        if t is None:
            return self._go(pos, (sx, sy), ["BUILD_" + self.STRUCTURE[
                self.animal]])
        if not isinstance(t, dict):
            return None
        if "animal" not in t:
            if shed.get(self.animal, 0) > 0:
                return self._go(pos, self.SHED_TILE,
                                ["PICKUP", self.animal, 1])
            return None
        if not _get(t, "fed_today"):
            if inv.get("WHEAT", 0) > 0:
                return self._go(pos, (sx, sy), ["FEED"])
            if shed.get("WHEAT", 0) > 0:
                return self._go(pos, self.SHED_TILE, ["PICKUP", "WHEAT", 1])
        if not _get(t, "cared_today"):
            return self._go(pos, (sx, sy), ["CARE"])
        if _get(t, "yield_units", 0) > 0:
            return self._go(pos, (sx, sy), ["HARVEST"])
        if _get(t, "fertilizer_available"):
            return self._go(pos, (sx, sy), ["COLLECT_FERTILIZER"])
        return None

    def _farmer(self, farm, shed, seeds, inv, day):
        pos = tuple(_get(farm, "farmer"))
        inv = dict(inv or {})
        task = self._animal_task(farm, shed, inv, pos, day)
        if task:
            return task
        keep = {"WHEAT"} if self.animal else set()
        if any(n > 0 and k not in keep for k, n in inv.items()):
            return self._go(pos, self.SHED_TILE, ["DROP"])
        for (x, y) in self.plot:
            t = self._tile(farm, x, y)
            if t == "LOCKED":
                continue
            want = None
            if t is None:
                if seeds.get(self.crop, 0) > 0:
                    want = ["PLANT", self.crop]
            elif _get(t, "kind") == "WEED":
                want = ["DIG"]
            elif _get(t, "kind") == "PLANT":
                age = day - _get(t, "planted_day", day)
                if not _get(t, "watered_today"):
                    want = ["WATER"]
                elif (_get(t, "yield_units", 0) > 0
                      and age >= self.FIRST_YIELD[_get(t, "crop")]):
                    want = ["HARVEST"]
                elif (inv.get("FERTILIZER", 0) > 0
                      and _get(t, "fertilized_until_day", -1) < day):
                    want = ["FERTILIZE"]
            if want:
                return self._go(pos, (x, y), want)
        if sum(inv.values()) > 0:
            return self._go(pos, self.SHED_TILE, ["DROP"])
        return ["PASS"]

    def __call__(self, obs, configuration=None):
        rng = self.rng
        me = _get(obs, "player", 0)
        step = _get(obs, "step", 0)
        day = step // 24
        farm = _get(obs, "farms")[me]
        priv = _get(obs, "private") or {}
        shed = dict(_get(priv, "shed") or {})
        seeds = dict(_get(priv, "seeds") or {})
        invs = _get(priv, "inventories") or [{}]
        money = _get(farm, "money", 0)

        farmer = self._farmer(farm, shed, seeds, invs[0] if invs else {},
                              day)
        hands = []
        for _ in _get(farm, "hands") or []:
            r = rng.random()
            hands.append([] if r < 0.1 else
                         ["WATER"] if r < 0.4 else
                         [rng.choice(MOVES)] if r < 0.8 else
                         ["HARVEST"])

        market = []
        if step % self.sell_every == 0:
            for p in PRODUCTS:
                keep = 2 if (p == "WHEAT" and self.animal) else 0
                if shed.get(p, 0) > keep:
                    market.append(["SELL", p, shed[p] - keep])
        if seeds.get(self.crop, 0) < len(self.plot) and money > 200:
            market.append(["BUY_SEED", self.crop, len(self.plot)])
        if self.animal and step % 24 == 5 and money > 900:
            market.append(["BUY_ANIMAL", self.animal, 1])
        if (self.animal and shed.get("WHEAT", 0) < 2 and money > 100
                and step % 24 == 3):
            market.append(["BUY_PRODUCT", "WHEAT", 2])
        if day in self.hire_days and step % 24 == 1:
            market.append(["HIRE"])
            if rng.random() < 0.5:
                market.append([])
                market.append(["BUY_PRODUCT", "FERTILIZER", 1])
        if money > self.land_money and step % 24 == 2:
            market.append(["BUY_LAND"])
        if rng.random() < 0.05:
            market.insert(0, [])
        return {"farmer": farmer, "hands": hands, "market": market[:10]}


class LastTurnSeller(ScriptedFarmer):
    """A farmer that never sells until the final turns, then dumps the shed.

    On the official runner its last action is at step 718. A driver that
    also applied a step-719 action would bank a different amount.
    """

    def __call__(self, obs, configuration=None):
        action = super().__call__(obs, configuration)
        step = _get(obs, "step", 0)
        action["market"] = [o for o in action["market"]
                            if not (o and o[0] == "SELL")]
        if step >= FINAL_STEP - 2:
            shed = _get(_get(obs, "private") or {}, "shed") or {}
            sells = [["SELL", p, 1] for p in PRODUCTS if shed.get(p, 0) > 0]
            action["market"] = sells + [["BUY_SEED", "WHEAT", 1]]
        return action


POLICIES = {"chaos": ChaosPolicy, "random": RandomPolicy,
            "scripted": ScriptedFarmer, "last_turn": LastTurnSeller}


def make_policy(kind: str, seed: int):
    return POLICIES[kind](seed)

# Adapted from msdsm/kaggriculture-solution 84057a0fda4238ccdebc46f9bf5496c6c4b2e00d; see docs/action_ppo_sources.md.
"""Local execution of Kaggriculture rules, transcribed from the official interpreter.

Source blob: 3c202c7ee921da239356789e266b694635103fc4 (2026-09-07).
This is a lightweight harness, NOT the installed kaggle-environments framework.
All scoring transitions follow the official source. Agent timeouts are measured
separately; game state is never exposed to agents except through observation().
"""

from __future__ import annotations
from copy import deepcopy
import math
import random

CROPS = {
    "WHEAT": dict(seed=10, first_yield_day=2, max_yield_day=4, interval=0, max_yield=6, ongoing=False),
    "CARROT": dict(seed=20, first_yield_day=2, max_yield_day=3, interval=0, max_yield=4, ongoing=False),
    "TOMATO": dict(seed=50, first_yield_day=8, max_yield_day=8, interval=1, max_yield=4, ongoing=True),
    "STRAWBERRY": dict(seed=100, first_yield_day=10, max_yield_day=10, interval=2, max_yield=4, ongoing=True),
    "MELON": dict(seed=80, first_yield_day=10, max_yield_day=12, interval=0, max_yield=6, ongoing=False),
}
ANIMALS = {
    "GOOSE": dict(cost=300, structure="COOP", first_yield_day=4, interval=1, max_held=4, product="EGG"),
    "COW": dict(cost=400, structure="PASTURE", first_yield_day=8, interval=2, max_held=6, product="MILK"),
    "SHEEP": dict(cost=500, structure="PASTURE", first_yield_day=6, interval=3, max_held=6, product="WOOL"),
}
PRODUCTS = list(CROPS) + ["EGG", "MILK", "WOOL", "FERTILIZER"]
ITEMS = PRODUCTS + list(ANIMALS)
_specs = [
    (25, 400, "sqrt", 0.8, "log", 0.2),
    (35, 450, "hinge", 1.0, "sqrt", 0.7),
    (60, 200, "hinge", 0.4, "sqrt", 0.6),
    (120, 100, "sqrt", 0.7, "linear", 1.6),
    (250, 300, "log", 0.2, "sq", 3.6),
    (50, 332, "hinge", 0.4, "log", 0.2),
    (160, 122, "sqrt", 0.6, "linear", 1.6),
    (200, 105, "log", 0.2, "sq", 3.2),
    (100, 200, "linear", 0.4, "linear", 0.4),
]
MARKET_PARAMS = {
    item: dict(zip(["base", "T", "below_func", "below_target", "above_func", "above_target"], row), I0=10000)
    for item, row in zip(PRODUCTS, _specs)
}
SHOPS = {
    "BAKERY": ["EGG", "WHEAT"],
    "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}
MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
LAND_ORDER = ["NE", "SW", "SE"]
LAND_PRICES = [1000, 2000, 4000]


def _shape(func, x, T=None):
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "log10":
        return math.log10(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return x


def market_price(item, inventory, params=None):
    p = (params or MARKET_PARAMS)[item]
    base = p["base"]
    I0 = p["I0"]
    T = p["T"]
    side = "below" if inventory < I0 else "above"
    f = p[side + "_func"]
    amp = p[side + "_target"] * base / _shape(f, T, T)
    price = base + (1 if inventory < I0 else -1) * amp * _shape(f, abs(inventory - I0), T)
    return max(1, int(round(price)))


def refresh_prices(market):
    for item in PRODUCTS:
        market["prices"][item] = market_price(item, market["inventory"][item], market.get("params"))


def quadrant(x, y, n=10):
    return ("N" if y < n // 2 else "S") + ("W" if x < n // 2 else "E")


def access(n=10):
    h = n // 2
    return [(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)]


def new_farm(n=10, money=3000):
    return dict(
        money=float(money),
        tiles=[[None if quadrant(x, y, n) == "NW" else "LOCKED" for x in range(n)] for y in range(n)],
        farmer=list(access(n)[0]),
        hands=[],
        unlocked_quadrants=["NW"],
        hires_today=0,
    )


def new_private():
    return dict(shed={i: 0 for i in ITEMS}, seeds={i: 0 for i in CROPS}, inventories=[{}])


def new_plant(crop, day, tpd=24):
    cd = CROPS[crop]
    return dict(
        kind="PLANT",
        crop=crop,
        planted_day=day,
        watered_today=False,
        consecutive_unwatered=1,
        yield_units=0 if cd["ongoing"] else 1,
        max_lifespan_step=-1 if cd["ongoing"] else (day + cd["max_yield_day"] + 1) * tpd,
        fertilized_until_day=-1,
    )


def new_animal(animal, day):
    return dict(
        kind=ANIMALS[animal]["structure"],
        animal=animal,
        placed_day=day,
        yield_units=0,
        consecutive_unfed=0,
        fed_today=False,
        cared_today=False,
        fertilizer_available=False,
        pending_care_bonus=0,
    )


def take(inv, item, n=1):
    if inv.get(item, 0) < n:
        return False
    inv[item] -= n
    if inv[item] == 0:
        del inv[item]
    return True


def unit_action(farm, private, idx, action, day, n=10, tpd=24, cap=100):
    if not isinstance(action, list) or not action:
        return
    op = action[0]
    pos = farm["farmer"] if idx == 0 else (farm["hands"][idx - 1] if idx <= len(farm["hands"]) else None)
    if pos is None:
        return
    fx, fy = pos
    while len(private["inventories"]) <= idx:
        private["inventories"].append({})
    inv = private["inventories"][idx]
    if op in MOVES:
        dx, dy = MOVES[op]
        nx, ny = fx + dx, fy + dy
        if 0 <= nx < n and 0 <= ny < n:
            if idx == 0:
                farm["farmer"] = [nx, ny]
            else:
                farm["hands"][idx - 1] = [nx, ny]
        return
    if op == "PASS":
        return
    tile = farm["tiles"][fy][fx]
    shed = private["shed"]
    adj = (fx, fy) in access(n)
    if op == "DROP":
        if not adj:
            return
        for item, k in list(inv.items()):
            if k <= 0:
                del inv[item]
                continue
            room = max(0, cap - sum(shed.values()))
            num = min(k, room)
            if num > 0:
                shed[item] = shed.get(item, 0) + num
            del inv[item]
        return
    if op == "PICKUP":
        if not adj or len(action) < 2:
            return
        item = action[1]
        k = int(action[2]) if len(action) >= 3 else 1
        if k <= 0:
            return
        k = min(k, shed.get(item, 0))
        if k <= 0:
            return
        shed[item] -= k
        inv[item] = inv.get(item, 0) + k
        return
    if op == "PLACE":
        if len(action) < 2:
            return
        item = action[1]
        if (
            item in ANIMALS
            and isinstance(tile, dict)
            and tile.get("kind") == ANIMALS[item]["structure"]
            and "animal" not in tile
        ):
            if take(inv, item):
                farm["tiles"][fy][fx] = new_animal(item, day)
            return
        if adj:
            k = int(action[2]) if len(action) >= 3 else 1
            if k <= 0:
                return
            k = min(k, inv.get(item, 0), max(0, cap - sum(shed.values())))
            if k <= 0:
                return
            take(inv, item, k)
            shed[item] = shed.get(item, 0) + k
        return
    if tile == "LOCKED":
        return
    if op == "PLANT":
        if len(action) < 2:
            return
        crop = action[1]
        if crop not in CROPS or tile is not None or private["seeds"].get(crop, 0) <= 0:
            return
        private["seeds"][crop] -= 1
        farm["tiles"][fy][fx] = new_plant(crop, day, tpd)
        return
    if op == "WATER":
        if not isinstance(tile, dict) or tile.get("kind") != "PLANT" or tile["watered_today"]:
            return
        tile["watered_today"] = True
        cd = CROPS[tile["crop"]]
        age = day - tile["planted_day"]
        if not cd["ongoing"] and (cd["max_yield_day"] + 1) // 2 <= age <= cd["max_yield_day"]:
            tile["yield_units"] = min(
                cd["max_yield"], tile["yield_units"] + (2 if tile["fertilized_until_day"] >= day else 1)
            )
        return
    if op == "HARVEST":
        if not isinstance(tile, dict) or tile.get("yield_units", 0) <= 0:
            return
        if tile.get("kind") == "PLANT":
            cd = CROPS[tile["crop"]]
            if day - tile["planted_day"] < cd["first_yield_day"]:
                return
            inv[tile["crop"]] = inv.get(tile["crop"], 0) + tile["yield_units"]
            tile["yield_units"] = 0
            if not cd["ongoing"]:
                farm["tiles"][fy][fx] = None
        elif "animal" in tile:
            item = ANIMALS[tile["animal"]]["product"]
            inv[item] = inv.get(item, 0) + tile["yield_units"]
            tile["yield_units"] = 0
        return
    if op == "FERTILIZE":
        if not isinstance(tile, dict) or tile.get("kind") != "PLANT" or not take(inv, "FERTILIZER"):
            return
        tile["fertilized_until_day"] = max(tile.get("fertilized_until_day", -1), day + 2)
        return
    if op == "DIG":
        if tile is None or isinstance(tile, dict) and "animal" in tile:
            return
        farm["tiles"][fy][fx] = None
        return
    if op in ("BUILD_COOP", "BUILD_PASTURE"):
        if tile is None:
            farm["tiles"][fy][fx] = {"kind": op[6:]}
        return
    if op == "FEED":
        if not isinstance(tile, dict) or "animal" not in tile or tile["fed_today"] or not take(inv, "WHEAT"):
            return
        tile["fed_today"] = True
        return
    if op == "COLLECT_FERTILIZER":
        if not isinstance(tile, dict) or "animal" not in tile or not tile["fertilizer_available"]:
            return
        tile["fertilizer_available"] = False
        inv["FERTILIZER"] = inv.get("FERTILIZER", 0) + 1
        return
    if op == "CARE":
        if isinstance(tile, dict) and "animal" in tile and not tile["cared_today"]:
            tile["cared_today"] = True


def daily_plants(farm, day, tpd=24):
    for y, row in enumerate(farm["tiles"]):
        for x, t in enumerate(row):
            if not isinstance(t, dict) or t.get("kind") != "PLANT":
                continue
            watered = t["watered_today"]
            t["consecutive_unwatered"] = 0 if watered else t["consecutive_unwatered"] + 1
            t["watered_today"] = False
            if t["consecutive_unwatered"] >= 2:
                row[x] = {"kind": "WEED"}
                continue
            cd = CROPS[t["crop"]]
            if not cd["ongoing"]:
                continue
            ds = day + 1 - t["planted_day"] - cd["first_yield_day"]
            if ds < 0 or ds % cd["interval"]:
                continue
            pc = ds // cd["interval"] + 1
            if pc > cd["max_yield"]:
                continue
            fert = watered and t.get("fertilized_until_day", -1) >= day
            t["yield_units"] = min(cd["max_yield"], t["yield_units"] + (2 if fert else 1))
            if pc == cd["max_yield"]:
                t["max_lifespan_step"] = (day + 2) * tpd


def daily_animals(farm, day):
    for row in farm["tiles"]:
        for x, t in enumerate(row):
            if not isinstance(t, dict) or "animal" not in t:
                continue
            t["consecutive_unfed"] = 0 if t["fed_today"] else t["consecutive_unfed"] + 1
            if t["consecutive_unfed"] >= 2:
                row[x] = {"kind": ANIMALS[t["animal"]]["structure"]}
                continue
            a = ANIMALS[t["animal"]]
            ds = day + 1 - t["placed_day"] - a["first_yield_day"]
            if ds >= 0 and ds % a["interval"] == 0:
                bonus = t.pop("pending_care_bonus", 0) if t["fed_today"] else 0
                t["yield_units"] = min(a["max_held"], t["yield_units"] + 1 + bonus)
                t["pending_care_bonus"] = 0
            if t["cared_today"] and t["fed_today"]:
                t["pending_care_bonus"] = t.get("pending_care_bonus", 0) + 1
            t["fertilizer_available"] = True
            t["fed_today"] = False
            t["cared_today"] = False


def fib(n):
    a = b = 1
    for _ in range(n):
        a, b = b, a + b
    return a


def do_hire(farm, private, n=10, mult=1):
    cost = mult * fib(farm["hires_today"])
    if farm["money"] < cost:
        return
    farm["money"] -= cost
    farm["hires_today"] += 1
    positions = [tuple(farm["farmer"])] + [tuple(p) for p in farm["hands"]]
    p = min(access(n), key=lambda p: positions.count(p))
    farm["hands"].append(list(p))
    private["inventories"].append({})


def do_land(farm, n=10):
    i = len(farm["unlocked_quadrants"]) - 1
    if i >= 3 or farm["money"] < LAND_PRICES[i]:
        return
    farm["money"] -= LAND_PRICES[i]
    q = LAND_ORDER[i]
    farm["unlocked_quadrants"].append(q)
    for y, row in enumerate(farm["tiles"]):
        for x, t in enumerate(row):
            if quadrant(x, y, n) == q and t == "LOCKED":
                row[x] = None


def parse_order(o):
    if not isinstance(o, list) or not o:
        return None
    op = o[0]
    if op in ("HIRE", "BUY_LAND"):
        return {"type": op}
    if op in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL") and len(o) >= 3:
        try:
            n = int(o[2])
        except (TypeError, ValueError):
            return None
        if n > 0:
            return dict(type=op, item=o[1], remaining=n)
    return None


def commit(op, item, price, farm, private, market, cap=100):
    shed = private["shed"]
    if op == "SELL":
        if shed.get(item, 0) <= 0:
            return False
        shed[item] -= 1
        farm["money"] += price
        if price > 1:
            market["inventory"][item] += 1
        return True
    if farm["money"] < price:
        return False
    if op == "BUY_PRODUCT":
        if sum(shed.values()) >= cap:
            return False
        farm["money"] -= price
        shed[item] = shed.get(item, 0) + 1
        market["inventory"][item] -= 1
        return True
    if op == "BUY_SEED":
        farm["money"] -= price
        private["seeds"][item] = private["seeds"].get(item, 0) + 1
        return True
    if op == "BUY_ANIMAL":
        if sum(shed.values()) >= cap:
            return False
        farm["money"] -= price
        shed[item] = shed.get(item, 0) + 1
        return True
    return False


def process_market(farms, privates, market, actions, cfg):
    queues = []
    for a in actions:
        q = a.get("market", []) if isinstance(a, dict) else []
        queues.append(q[: cfg["maxMarketOrdersPerTurn"]] if isinstance(q, list) else [])
    for i in range(max(map(len, queues), default=0)):
        orders = [parse_order(q[i]) if i < len(q) else None for q in queues]
        for p, o in enumerate(orders):
            if o is None:
                continue
            if o["type"] == "HIRE":
                do_hire(farms[p], privates[p], cfg["boardSize"], cfg["farmHandCostMult"])
                orders[p] = None
            elif o["type"] == "BUY_LAND":
                do_land(farms[p], cfg["boardSize"])
                orders[p] = None
        for _ in range(99999):
            quoted = [None, None]
            for p, o in enumerate(orders):
                if o is None or o["remaining"] <= 0:
                    continue
                op, item = o["type"], o["item"]
                if op == "SELL" and item in PRODUCTS:
                    price = market_price(item, market["inventory"][item], market.get("params"))
                elif op == "BUY_PRODUCT" and item in ("WHEAT", "FERTILIZER"):
                    price = market_price(item, market["inventory"][item] - 1, market.get("params"))
                elif op == "BUY_SEED" and item in CROPS:
                    price = CROPS[item]["seed"]
                elif op == "BUY_ANIMAL" and item in ANIMALS:
                    price = ANIMALS[item]["cost"]
                else:
                    orders[p] = None
                    continue
                quoted[p] = (op, item, price, o)
            if all(q is None for q in quoted):
                break
            committed = False
            for p, q in enumerate(quoted):
                if q is None:
                    continue
                op, item, price, o = q
                if commit(op, item, price, farms[p], privates[p], market, cfg["shedCapacity"]):
                    o["remaining"] -= 1
                    committed = True
                else:
                    orders[p] = None
            if not committed:
                break
        refresh_prices(market)


DEFAULT_CONFIG = dict(
    boardSize=10,
    startingMoney=3000,
    episodeSteps=720,
    maxMarketOrdersPerTurn=10,
    turnsPerDay=24,
    shedCapacity=100,
    weedSpawnChance=0.005,
    townShopUnlockInterval=3,
    townShopSellInterval=4,
    townCenterSellInterval=24,
    farmHandCostMult=1,
)


class Game:
    def __init__(self, seed=0, **config):
        self.cfg = {**DEFAULT_CONFIG, **config}
        self.seed = seed
        self.step = 0
        self.farms = [new_farm(self.cfg["boardSize"], self.cfg["startingMoney"]) for _ in range(2)]
        self.privates = [new_private() for _ in range(2)]
        self.market = dict(
            inventory={i: 10000 for i in PRODUCTS}, prices={i: p["base"] for i, p in MARKET_PARAMS.items()}
        )
        if self.cfg.get("marketParams"):
            ps = deepcopy(MARKET_PARAMS)
            for item, patch in self.cfg["marketParams"].items():
                ps[item].update(patch)
            self.market["params"] = ps
            self.market["inventory"] = {i: p["I0"] for i, p in ps.items()}
            refresh_prices(self.market)
        self.town = {"unlocked_shops": []}

    @property
    def done(self):
        return self.step >= self.cfg["episodeSteps"] - 1

    def observation(self, p):
        return deepcopy(
            dict(
                player=p,
                step=self.step,
                day=self.step // self.cfg["turnsPerDay"],
                hour=self.step % self.cfg["turnsPerDay"],
                farms=self.farms,
                private=self.privates[p],
                market=self.market,
                town=self.town,
                remainingOverageTime=60,
            )
        )

    def advance(self, actions):
        if self.done:
            raise RuntimeError("advance after game end")
        cfg = self.cfg
        n = cfg["boardSize"]
        tpd = cfg["turnsPerDay"]
        cap = cfg["shedCapacity"]
        day = self.step // tpd
        for p, a in enumerate(actions):
            if not isinstance(a, dict):
                a = {}
            hands = a.get("hands", [])
            hands = hands if isinstance(hands, list) else []
            acts = [a.get("farmer", ["PASS"]), *hands]
            demand = {}
            for ua in acts:
                if isinstance(ua, list) and len(ua) >= 2 and ua[0] == "PLANT":
                    demand[ua[1]] = demand.get(ua[1], 0) + 1
            blocked = {c for c, v in demand.items() if v > self.privates[p]["seeds"].get(c, 0)}
            for j, ua in enumerate(acts):
                if isinstance(ua, list) and len(ua) >= 2 and ua[0] == "PLANT" and ua[1] in blocked:
                    ua = ["PASS"]
                unit_action(self.farms[p], self.privates[p], j, ua, day, n, tpd, cap)
        process_market(self.farms, self.privates, self.market, actions, cfg)
        if self.step % max(1, cfg["townShopSellInterval"]) == 0:
            for s in self.town["unlocked_shops"]:
                items = SHOPS[s]
                for item in items:
                    self.market["inventory"][item] -= 2 if len(items) == 1 else 1
        if self.step % max(1, cfg["townCenterSellInterval"]) == 0:
            for item in PRODUCTS[:-1]:
                self.market["inventory"][item] -= 1
        refresh_prices(self.market)
        for farm in self.farms:
            for row in farm["tiles"]:
                for x, t in enumerate(row):
                    if not isinstance(t, dict) or t.get("kind") != "PLANT":
                        continue
                    mls = t["max_lifespan_step"]
                    if mls < 0 or self.step < mls or (self.step - mls) % 2:
                        continue
                    t["yield_units"] -= 1
                    if t["yield_units"] <= 0:
                        row[x] = {"kind": "WEED"}
        if (self.step + 1) % tpd == 0:
            rng = random.Random((self.seed * 1000003) ^ day)
            for p, farm in enumerate(self.farms):
                private = self.privates[p]
                daily_plants(farm, day, tpd)
                daily_animals(farm, day)
                for row in farm["tiles"]:
                    for x, t in enumerate(row):
                        if t is None and rng.random() < cfg["weedSpawnChance"]:
                            row[x] = {"kind": "WEED"}
                for inv in private["inventories"]:
                    for item, k in list(inv.items()):
                        if k <= 0:
                            del inv[item]
                            continue
                        take_n = min(k, max(0, cap - sum(private["shed"].values())))
                        if take_n > 0:
                            private["shed"][item] = private["shed"].get(item, 0) + take_n
                        del inv[item]
                farm["farmer"] = list(access(n)[0])
                farm["hands"] = []
                farm["hires_today"] = 0
                private["inventories"] = [{}]
            if (day + 1) % cfg["townShopUnlockInterval"] == 0 and len(self.town["unlocked_shops"]) < 8:
                self.town["unlocked_shops"].append(rng.choice(sorted(SHOPS)))
        self.step += 1

# Adapted from msdsm/kaggriculture-solution 84057a0fda4238ccdebc46f9bf5496c6c4b2e00d; see docs/action_ppo_sources.md.
"""C++ finite-horizon production search + multi-start farm-worker route search."""

from __future__ import annotations
import ctypes
import hashlib
import json
import time
from pathlib import Path
from copy import deepcopy
import numpy as np
from .inventory_tracker import OpponentInventoryTracker
from .reference_engine import (
    PRODUCTS,
    ITEMS,
    CROPS,
    ANIMALS,
    SHOPS,
    market_price,
    daily_plants,
    daily_animals,
    Game,
    new_private,
    new_plant,
)

_LOADED_BINARIES = {}


def native_binary_identity():
    """Reject stale builds before public provenance names an executable."""
    binary = Path(__file__).with_name("terminal_search.so")
    receipt = json.loads(binary.with_suffix(".build.json").read_text())
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    if receipt.get("contract") != "owned-season-search-native-wdl-v1" or receipt.get("binary_sha256") != digest:
        raise ValueError("season search binary differs from its build receipt; rebuild owned search")
    source = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(binary.parent.iterdir()) if path.suffix in (".cpp", ".inc")}
    if receipt.get("source_sha256") != source:
        raise ValueError("season search native sources changed; rebuild owned search")
    if str(binary) in _LOADED_BINARIES and _LOADED_BINARIES[str(binary)] != digest:
        raise ValueError("season search binary changed after loading; restart this Python process")
    return binary, digest


def validate_observation(obs):
    """Bound all fixed-size native input arrays before crossing the C ABI."""
    if obs.get("market", {}).get("params"):
        raise ValueError("custom market parameters are unsupported by owned native search")
    if obs.get("player") not in (0, 1) or not 0 <= int(obs["day"]) < 30 or not 0 <= int(obs["hour"]) < 24:
        raise ValueError("season search requires one official seat in the 30-day season")
    if len(obs["farms"]) != 2:
        raise ValueError("season search requires two official farms")
    for farm in obs["farms"]:
        if len(farm["tiles"]) != 10 or any(len(row) != 10 for row in farm["tiles"]):
            raise ValueError("season search requires a 10 by 10 farm")
        positions = [farm["farmer"], *farm["hands"]]
        if len(positions) > 20 or any(len(position) != 2 or
                any(isinstance(coordinate, bool) or int(coordinate) != coordinate or not 0 <= coordinate < 10
                    for coordinate in position) for position in positions):
            raise ValueError("season search requires at most 20 units inside the farm")


OPS = [
    "PASS",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PICKUP",
    "DROP",
    "PLACE",
    "PLANT",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "DIG",
    "FEED",
    "COLLECT_FERTILIZER",
    "CARE",
]
CROP_NAMES = list(CROPS)
ANIMAL_NAMES = list(ANIMALS)


def sale_priority(item, inventory, quantity, opponent_quantity):
    def sell(inv, n):
        value = 0
        for _ in range(n):
            price = market_price(item, inv)
            value += price
            if price > 1:
                inv += 1
        return value, inv

    a, i = sell(inventory, quantity)
    b, _ = sell(i, opponent_quantity)
    x, j = sell(inventory, opponent_quantity)
    y, _ = sell(j, quantity)
    return (a - b) - (y - x) + 1e-6 * quantity * market_price(item, inventory)


def forecast_fertilization(t, d, prices, flow):
    if t.get("watered_today", False) or t.get("fertilized_until_day", -1) >= d:
        return
    c = t["crop"]
    cd = CROPS[c]
    age = d - t["planted_day"]
    if cd["ongoing"]:
        bonus = sum(
            1
            for k in range(3)
            if d + k < 29
            and 0 <= age + k + 1 - cd["first_yield_day"] <= 3 * cd["interval"]
            and (age + k + 1 - cd["first_yield_day"]) % cd["interval"] == 0
        )
    else:
        start = (cd["max_yield_day"] + 1) // 2
        if age < start or age > cd["max_yield_day"]:
            return
        normal = max(0, min(cd["max_yield_day"] - age + 1, 30 - d))
        bonus = max(0, min(3, normal, cd["max_yield"] - t["yield_units"] - normal))
    if bonus * prices[c] > prices["FERTILIZER"] + 8:
        t["fertilized_until_day"] = d + 2
        flow["FERTILIZER"] -= 1


def forecast_prices(obs):
    day = obs["day"]
    farms = deepcopy(obs["farms"])
    inv = dict(obs["market"]["inventory"])
    params = obs["market"].get("params")
    demand = {i: 1 if i != "FERTILIZER" else 0 for i in PRODUCTS}
    for shop in obs["town"]["unlocked_shops"]:
        for item in SHOPS[shop]:
            demand[item] += 12 if len(SHOPS[shop]) == 1 else 6
    out = []
    expected = {i: sum((12 if len(ps) == 1 else 6) for ps in SHOPS.values() if i in ps) / len(SHOPS) for i in PRODUCTS}
    current_count = len(obs["town"]["unlocked_shops"])
    deferred = {i: 0 for i in PRODUCTS}
    for d in range(day, 30):
        future_shops = max(0, min(8, d // 3) - current_count)
        daily_demand = {i: demand[i] + future_shops * expected[i] for i in PRODUCTS}
        q = {i: 0 for i in PRODUCTS}
        feed = 0
        for player, farm in enumerate(farms):
            before_flow = dict(q)
            for row in farm["tiles"]:
                for x, t in enumerate(row):
                    if not isinstance(t, dict):
                        continue
                    if t.get("kind") == "PLANT":
                        cd = CROPS[t["crop"]]
                        age = d - t["planted_day"]
                        c = t["crop"]
                        forecast_fertilization(t, d, obs["market"]["prices"], q)
                        if not t["watered_today"]:
                            t["watered_today"] = True
                            if not cd["ongoing"] and (cd["max_yield_day"] + 1) // 2 <= age <= cd["max_yield_day"]:
                                t["yield_units"] = min(
                                    cd["max_yield"], t["yield_units"] + (2 if t["fertilized_until_day"] >= d else 1)
                                )
                        mature = age >= cd["first_yield_day"]
                        harvest = (
                            cd["ongoing"]
                            or t["yield_units"] >= cd["max_yield"]
                            or age >= cd["max_yield_day"]
                            or d >= 28
                        )
                        if mature and harvest:
                            q[c] += t["yield_units"]
                            t["yield_units"] = 0
                            if not cd["ongoing"]:
                                # The value DP and opponent_flow already renew these staple crops.
                                # Letting them disappear only in the price forecast inflated future
                                # feed costs and could make otherwise profitable animals escape.
                                if c in ("WHEAT", "CARROT") and d + 2 < 30:
                                    row[x] = new_plant(c, d)
                                    row[x]["watered_today"] = True
                                else:
                                    row[x] = None
                    elif "animal" in t:
                        a = ANIMALS[t["animal"]]
                        q[a["product"]] += t["yield_units"]
                        t["yield_units"] = 0
                        q["FERTILIZER"] += int(t.get("fertilizer_available", False))
                        t["fertilizer_available"] = False
                        if d < 29:
                            feed += int(not t["fed_today"])
                            t["fed_today"] = True
                            t["cared_today"] = True
            daily_plants(farm, d)
            daily_animals(farm, d)
            if player == 1 - obs["player"]:
                gross = [max(0, int(q[item] - before_flow[item])) for item in PRODUCTS]
                delayed = (
                    future_sale_quantities(gross, obs.get("_opponent_overnight_fraction", {})) if d < 29 else [0] * 9
                )
                for c, item in enumerate(PRODUCTS):
                    q[item] += deferred[item] - delayed[c]
                    deferred[item] = delayed[c]
        rowprices = []
        for item in PRODUCTS:
            # Public-farm production forecast; future shops and hidden inventory are not used.
            supply = max(0, int(q[item]))
            buy = (feed if item == "WHEAT" else 0) + max(0, -int(q[item]))
            base = inv[item] - buy
            start = base - int(daily_demand[item] * 0.5)
            if supply:
                total = 0.0
                cur = start
                for _ in range(supply):
                    p = market_price(item, cur, params)
                    total += p
                    if p > 1:
                        cur += 1
                val = total / supply
            else:
                val = market_price(item, start, params)
            rowprices.append(max(1.0, val))
            for _ in range(supply):
                if market_price(item, base, params) > 1:
                    base += 1
            inv[item] = base - int(daily_demand[item])
        out.append(rowprices)
    return np.ascontiguousarray(out, dtype=np.float64)


def pack(obs):
    validate_observation(obs)
    day = obs["day"]
    farm = obs["farms"][obs["player"]]
    private = obs["private"]
    positions = [farm["farmer"], *farm["hands"]]
    a = [day, obs["hour"], len(positions), int(farm["money"]), 2500, 2500, day * 104729 + obs["player"] * 97 + 113]
    a.extend(int(private["shed"].get(i, 0)) for i in ITEMS)
    a.extend(int(private["seeds"].get(i, 0)) for i in CROP_NAMES)
    a.extend(x + 10 * y for x, y in positions)
    for j in range(len(positions)):
        inv = private["inventories"][j] if j < len(private["inventories"]) else {}
        a.extend(int(inv.get(i, 0)) for i in ITEMS)
    for row in farm["tiles"]:
        for t in row:
            v = [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1]
            if t == "LOCKED":
                v[0] = 1
            elif isinstance(t, dict):
                if t["kind"] == "WEED":
                    v[0] = 2
                elif t["kind"] == "PLANT":
                    v = [
                        3,
                        CROP_NAMES.index(t["crop"]),
                        day - t["planted_day"],
                        t["yield_units"],
                        t["consecutive_unwatered"],
                        int(t["watered_today"]),
                        t["fertilized_until_day"],
                        0,
                        0,
                        0,
                        t["max_lifespan_step"],
                    ]
                elif "animal" in t:
                    v = [
                        6,
                        5 + ANIMAL_NAMES.index(t["animal"]),
                        day - t["placed_day"],
                        t["yield_units"],
                        t["consecutive_unfed"],
                        int(t["fed_today"]),
                        -1,
                        int(t["cared_today"]),
                        int(t["fertilizer_available"]),
                        t.get("pending_care_bonus", 0),
                        -1,
                    ]
                else:
                    v[0] = 4 if t["kind"] == "COOP" else 5
            a.extend(v)
    a.extend(int(obs["market"]["inventory"][i]) for i in PRODUCTS)
    demand = {i: 0 for i in PRODUCTS}
    for shop in obs["town"]["unlocked_shops"]:
        for item in SHOPS[shop]:
            demand[item] += 2 if len(SHOPS[shop]) == 1 else 1
    a.extend(demand[i] for i in PRODUCTS)
    predicted = np.zeros((24, 9), dtype=np.int32)
    # Prior for hidden starting stock, not access to the opponent's private state.
    for j, item in enumerate(PRODUCTS):
        predicted[0, j] = max(0, int(round(obs.get("_opponent_products", private["shed"]).get(item, 0))))
    for y, row in enumerate(obs["farms"][1 - obs["player"]]["tiles"]):
        for x, t in enumerate(row):
            if not isinstance(t, dict):
                continue
            distance = abs(x - (4 if x <= 4 else 5)) + abs(y - (4 if y <= 4 else 5))
            when = min(22, 7 + 2 * distance)
            if t.get("kind") == "PLANT":
                cd = CROPS[t["crop"]]
                age = day - t["planted_day"]
                if age < cd["first_yield_day"]:
                    continue
                qty = t["yield_units"]
                if (
                    not cd["ongoing"]
                    and not t["watered_today"]
                    and (cd["max_yield_day"] + 1) // 2 <= age <= cd["max_yield_day"]
                ):
                    qty = min(cd["max_yield"], qty + (2 if t["fertilized_until_day"] >= day else 1))
                predicted[when, PRODUCTS.index(t["crop"])] += qty
            elif "animal" in t:
                predicted[when, PRODUCTS.index(ANIMALS[t["animal"]]["product"])] += t.get("yield_units", 0)
                predicted[when, 8] += int(t.get("fertilizer_available", False))
    future = opponent_flow(obs)
    apply_sale_delay(predicted, future, obs.get("_opponent_overnight_fraction", {}), day)
    a.extend(int(x) for x in predicted.flat)
    a.extend(int(x) for x in future.flat)
    ownobs = dict(obs)
    ownobs["player"] = 1 - obs["player"]
    ownobs.pop("_opponent_products", None)
    a.extend(int(x) for x in opponent_flow(ownobs).flat)
    a.append(len(obs["town"]["unlocked_shops"]))
    a.extend((int(obs["farms"][1 - obs["player"]]["money"]),
              max(1, int(round(obs.get("_terminal_uncertainty", 2000.))))))
    return np.ascontiguousarray(a, dtype=np.int32)


def future_sale_quantities(flow, fractions):
    # These products have no BUY_PRODUCT flow, so harvest and feed purchases
    # cannot cancel each other in the signed daily forecast.
    amounts = [0] * 9
    for c in range(1, 8):
        amounts[c] = max(0, int(round(flow[c] * max(0.0, min(1.0, float(fractions.get(PRODUCTS[c], 0.0)))))))
    total = sum(amounts)
    if total > 100:
        prefix = assigned = 0
        for c in range(9):
            prefix += amounts[c]
            next_assigned = 100 * prefix // total
            amounts[c] = next_assigned - assigned
            assigned = next_assigned
    return amounts


def apply_sale_delay(predicted, future, fractions, day):
    # Opening shed stock stays available now. Today's and future harvests may
    # be sold the following day; every shifted unit is conserved.
    if day >= 29 or len(future) < 2:
        return
    original = future.copy()
    quantities = [int(predicted[1:, c].sum()) for c in range(9)]
    wanted = [
        int(round(quantities[c] * max(0.0, min(1.0, float(fractions.get(item, 0.0))))))
        for c, item in enumerate(PRODUCTS)
    ]
    # Returning product must fit the next dawn's 100-unit shed. If forecast
    # production is larger, attribute the excess to same-day deliveries.
    total = sum(wanted)
    if total > 100:
        cumulative = 0
        assigned = 0
        for c in range(9):
            cumulative += wanted[c]
            target = 100 * cumulative // total
            wanted[c] = target - assigned
            assigned = target
    for c in range(9):
        if not quantities[c] or not wanted[c]:
            continue
        cumulative = 0
        shifted = 0
        for h in range(1, 24):
            quantity = int(predicted[h, c])
            cumulative += quantity
            target = int(round(cumulative * wanted[c] / quantities[c]))
            take = target - shifted
            predicted[h, c] -= take
            shifted = target
        future[1, c] += shifted
    # Shift each day's newly harvested quantity once. Incoming overnight stock
    # must remain at this dawn rather than being deferred a second time.
    for d in range(1, len(future) - 1):
        if day + d >= 29:
            break
        shifted = future_sale_quantities(original[d], fractions)
        future[d] -= shifted
        future[d + 1] += shifted


def projected_shed(obs, actions):
    shed = dict(obs["private"]["shed"])
    farm = obs["farms"][obs["player"]]
    positions = [farm["farmer"], *farm["hands"]]
    invs = obs["private"]["inventories"]
    for j, (pos, a) in enumerate(zip(positions, actions)):
        if tuple(pos) not in ((4, 4), (5, 4), (4, 5), (5, 5)):
            continue
        inv = invs[j] if j < len(invs) else {}
        if a[0] == "PICKUP":
            shed[a[1]] = max(0, shed.get(a[1], 0) - a[2])
        elif a[0] == "DROP":
            for item, n in inv.items():
                shed[item] = shed.get(item, 0) + min(n, max(0, 100 - sum(shed.values())))
    return shed


def normalize_water_actions(farm, actions):
    # A crop can disappear through lifespan decay while the daily plan is running.
    # Keep its scheduled waiting turn, without requesting care on a missing crop.
    # An earlier worker's PLANT can create a crop in this same farm phase.
    planted = set()
    for pos, action in zip([farm["farmer"], *farm["hands"]], actions):
        key = tuple(pos)
        if action[0] == "PLANT":
            planted.add(key)
        elif action[0] == "WATER" and key not in planted:
            tile = farm["tiles"][pos[1]][pos[0]]
            if not isinstance(tile, dict) or tile.get("kind") != "PLANT" or tile.get("watered_today", False):
                action[:] = ["PASS"]


def decode_preparation(output):
    start = 32 + 24 * 20 * 3
    count = int(output[start])
    if not 0 <= count <= 96:
        raise RuntimeError("Invalid native market order count")
    result = []
    for k in range(count):
        op, item, n = map(int, output[start + 1 + 3 * k : start + 4 + 3 * k])
        if op == 0:
            order = ["SELL", PRODUCTS[item], n]
        elif op == 1:
            order = ["BUY_SEED", CROP_NAMES[item], n]
        elif op == 2:
            order = ["BUY_PRODUCT", PRODUCTS[item], n]
        elif op == 3:
            order = ["BUY_ANIMAL", ITEMS[item], n]
        elif op == 4:
            order = ["HIRE"]
        elif op == 5:
            order = ["BUY_LAND"]
        else:
            raise RuntimeError("Invalid native market opcode")
        result.append(order)
    return result


class SearchPolicy:
    def __init__(self, seconds=0.7):
        binary, self.binary_sha256 = native_binary_identity()
        self.lib = ctypes.CDLL(str(binary))
        _LOADED_BINARIES[str(binary)] = self.binary_sha256
        ip = ctypes.POINTER(ctypes.c_int)
        dp = ctypes.POINTER(ctypes.c_double)
        self.lib.search_create.argtypes = [ip, dp]
        self.lib.search_create.restype = ctypes.c_void_p
        self.lib.search_advance.argtypes = [ctypes.c_void_p, ctypes.c_double]
        self.lib.search_advance.restype = ctypes.c_int
        self.lib.search_destroy.argtypes = [ctypes.c_void_p]
        self.lib.search_destroy.restype = None
        self.lib.search_import.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.search_import.restype = ctypes.c_int
        self.lib.plan_day_warm.argtypes = [ip, dp, ctypes.c_double, ctypes.c_void_p, ip]
        self.lib.plan_day_warm.restype = ctypes.c_int
        config_path = Path(__file__).with_name("search_config.json")
        self.config = json.loads(config_path.read_text()) if config_path.exists() else {"ponder": True}
        self.warm = None
        self.warm_day = -1
        self.warm_hour = -1
        self.tracker = None
        self.previous = None
        self.previous_action = None
        self.seconds = seconds
        self.plan = None
        self.plan_day = -1

    def make_plan(self, obs):
        inp = pack(obs)
        fc = forecast_prices(obs)
        out = np.zeros(32 + 24 * 20 * 3 + 1 + 96 * 3, dtype=np.int32)
        common = [
            inp.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            fc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self.seconds,
        ]
        tail = [out.ctypes.data_as(ctypes.POINTER(ctypes.c_int))]
        rc = self.lib.plan_day_warm(*common, self.warm if self.warm_day == obs["day"] else None, *tail)
        self.close_warm()
        if rc:
            raise RuntimeError("C++ planner failed")
        self.header = out[:32].copy()
        self.plan = out[32 : 32 + 24 * 20 * 3].reshape(24, 20, 3)
        self.plan_day = obs["day"]
        self.preparation = decode_preparation(out)

    def close_warm(self):
        if self.warm:
            self.lib.search_destroy(self.warm)
        self.warm = None
        self.warm_day = -1
        self.warm_hour = -1

    def __del__(self):
        if getattr(self, "warm", None):
            self.lib.search_destroy(self.warm)

    def predict_dawn(self, obs):
        # Replay our committed actions. Unknown rival actions, weeds and new shops
        # are deliberately omitted; this is only a warm-start proposal, never truth.
        game = Game(weedSpawnChance=0, townShopUnlockInterval=999)
        game.step = int(obs["step"])
        game.farms = deepcopy(obs["farms"])
        game.market = deepcopy(obs["market"])
        game.town = deepcopy(obs["town"])
        player = obs["player"]
        rival = 1 - player
        game.privates = [new_private(), new_private()]
        game.privates[player] = deepcopy(obs["private"])
        # Only the public-state estimator supplies these inventories. In particular,
        # carried goods must survive the automatic overnight DROP into tomorrow's
        # shed; treating them as zero biased the speculative market substantially.
        rival_private = game.privates[rival]
        rival_private["shed"].update({i: max(0, int(round(n))) for i, n in obs.get("_opponent_products", {}).items()})
        estimated = obs.get("_opponent_carried_by_unit", [])
        rival_private["inventories"] = [
            {i: max(0, int(round(n))) for i, n in inv.items() if n > 0} for inv in estimated
        ]
        while len(rival_private["inventories"]) < 1 + len(game.farms[rival]["hands"]):
            rival_private["inventories"].append({})
        while game.step // 24 == obs["day"]:
            actions = [{}, {}]
            actions[player] = self.action_for_plan(game.observation(player))
            actions[rival] = {
                "market": [
                    ["SELL", i, int(rival_private["shed"].get(i, 0))]
                    for i in PRODUCTS
                    if rival_private["shed"].get(i, 0) > 0
                ]
            }
            game.advance(actions)
        predicted = game.observation(player)
        predicted["_opponent_products"] = dict(rival_private["shed"])
        predicted["_opponent_overnight_fraction"] = dict(obs.get("_opponent_overnight_fraction", {}))
        return predicted

    def ponder(self, obs, call_started):
        if not self.config.get("ponder", True) or obs["day"] >= 29 or not 16 <= obs["hour"] <= 23:
            return
        if float(obs.get("remainingOverageTime", 60)) < 3:
            return
        # Re-anchor the next-day snapshot in the latest public observation. Keep
        # the old best action signatures, then recheck all prices and constraints.
        refresh = self.warm_day != obs["day"] + 1 or obs["hour"] - self.warm_hour >= 4 or obs["hour"] == 23
        if refresh:
            predicted = self.predict_dawn(obs)
            inp = pack(predicted)
            fc = forecast_prices(predicted)
            fresh = self.lib.search_create(
                inp.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), fc.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            )
            if not fresh:
                raise RuntimeError("C++ speculative planner failed")
            if self.warm:
                if self.lib.search_import(fresh, self.warm):
                    self.lib.search_destroy(fresh)
                    raise RuntimeError("C++ speculative import failed")
                self.lib.search_destroy(self.warm)
            self.warm = fresh
            self.warm_day = obs["day"] + 1
            self.warm_hour = obs["hour"]
        # Keep the whole call under the 1s actTimeout: 0.90s total target leaves a
        # 0.10s Python/dispatch reserve, and the 0.70s search cap leaves room for
        # C++ overshoot past one iteration.
        seconds = min(0.70, max(0.0, 0.90 - (time.perf_counter() - call_started)))
        if seconds > 0 and self.lib.search_advance(self.warm, seconds):
            raise RuntimeError("C++ speculative search failed")

    def __call__(self, obs):
        validate_observation(obs)
        call_started = time.perf_counter()
        external = "_opponent_products" in obs
        if external:
            est = None
        elif self.tracker is None:
            self.tracker = OpponentInventoryTracker(observer_player=int(obs["player"]))
            est = self.tracker.estimate()
        elif self.previous is not None:
            est = self.tracker.update(self.previous, obs, self.previous_action)
        else:
            est = self.tracker.estimate()
        self.previous = deepcopy(obs)
        obs = dict(obs)
        if not external:
            obs["_opponent_products"] = est.shed
            obs["_opponent_carried_by_unit"] = self.tracker.carried_by_unit
            obs["_opponent_overnight_fraction"] = self.tracker.overnight_sale_fraction()
        if self.plan is None or self.plan_day != obs["day"]:
            self.make_plan(obs)
        result = self.action_for_plan(obs)
        self.ponder(obs, call_started)
        self.previous_action = deepcopy(result)
        return result

    def action_for_plan(self, obs):
        h = obs["hour"]
        farm = obs["farms"][obs["player"]]
        n = 1 + len(farm["hands"])
        a = []
        for u in range(n):
            op, item, qty = (int(x) for x in self.plan[h, u]) if u < 20 else (0, 0, 0)
            if OPS[op] == "PLANT":
                cmd = ["PLANT", CROP_NAMES[item]]
            elif OPS[op] in ("PICKUP", "PLACE"):
                cmd = [OPS[op], ITEMS[item], qty]
            else:
                cmd = [OPS[op]]
            a.append(cmd)
        normalize_water_actions(farm, a)
        shed = projected_shed(obs, a)
        market = []
        # Preserve exactly what scheduled future PICKUP operations require.
        reserve = {i: (int(self.header[10 + PRODUCTS.index(i)]) if i in PRODUCTS else 0) for i in ITEMS}
        for c, item in enumerate(PRODUCTS):
            release = (int(self.header[26 + c // 6]) >> (5 * (c % 6))) & 31
            quantity = (int(self.header[28 + c // 4]) >> (7 * (c % 4))) & 127
            if h < release:
                reserve[item] += quantity
        for hour in range(h + 1, 24):
            for op, item, qty in self.plan[hour, : int(self.header[0])]:
                if op == 5:
                    reserve[ITEMS[int(item)]] += int(qty)
        # Execute the same preparation schedule as the C++ evaluator. Non-prep
        # surplus sales may follow after workers have already started.
        for order in self.preparation[h * 10 : (h + 1) * 10]:
            order = list(order)
            if order[0] == "SELL":
                qty = min(order[2], max(0, shed.get(order[1], 0) - reserve[order[1]]))
                order[2] = qty
                shed[order[1]] = shed.get(order[1], 0) - qty
                if qty == 0:
                    order = ["NOOP"]  # preserve the scheduled, ineffective slot
            market.append(order)
        # Sell in full after this turn's DROP, not just the previous observation's shed.
        quantities = {item: max(0, shed.get(item, 0) - reserve[item]) for item in PRODUCTS}
        rival = obs.get("_opponent_products", {})
        order = sorted(
            (i for i in PRODUCTS if quantities[i] > 0),
            key=lambda i: -sale_priority(
                i, obs["market"]["inventory"][i], quantities[i], max(0, int(round(rival.get(i, 0))))
            ),
        )
        for item in order:
            q = quantities[item]
            if q > 0 and len(market) < 10:
                market.append(["SELL", item, q])
        return dict(farmer=a[0], hands=a[1:], market=market)


def opponent_flow(obs):
    day = obs["day"]
    f = deepcopy(obs["farms"][1 - obs["player"]])
    result = []
    for d in range(day, 30):
        q = {i: 0 for i in PRODUCTS}
        if d == day:
            for i in PRODUCTS:
                q[i] += max(0, int(round(obs.get("_opponent_products", obs["private"]["shed"]).get(i, 0))))
        for row in f["tiles"]:
            for x, t in enumerate(row):
                if not isinstance(t, dict):
                    continue
                if t.get("kind") == "PLANT":
                    c = t["crop"]
                    cd = CROPS[c]
                    age = d - t["planted_day"]
                    forecast_fertilization(t, d, obs["market"]["prices"], q)
                    if not t["watered_today"]:
                        t["watered_today"] = True
                        if not cd["ongoing"] and (cd["max_yield_day"] + 1) // 2 <= age <= cd["max_yield_day"]:
                            t["yield_units"] = min(
                                cd["max_yield"], t["yield_units"] + (2 if t["fertilized_until_day"] >= d else 1)
                            )
                    if age >= cd["first_yield_day"] and (
                        cd["ongoing"] or t["yield_units"] >= cd["max_yield"] or age >= cd["max_yield_day"] or d >= 28
                    ):
                        q[c] += t["yield_units"]
                        t["yield_units"] = 0
                        if not cd["ongoing"]:
                            if c in ("WHEAT", "CARROT") and d + 2 < 30:
                                from .reference_engine import new_plant

                                row[x] = new_plant(c, d)
                                row[x]["watered_today"] = True
                            else:
                                row[x] = None
                elif "animal" in t:
                    c = ANIMALS[t["animal"]]["product"]
                    q[c] += t["yield_units"]
                    t["yield_units"] = 0
                    q["FERTILIZER"] += int(t["fertilizer_available"])
                    t["fertilizer_available"] = False
                    if d < 29:
                        q["WHEAT"] -= int(not t["fed_today"])
                        t["fed_today"] = True
                        t["cared_today"] = True
        daily_plants(f, d)
        daily_animals(f, d)
        result.append([q[i] for i in PRODUCTS])
    return np.array(result, dtype=np.int32)

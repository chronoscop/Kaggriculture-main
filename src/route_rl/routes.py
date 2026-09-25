"""Small, executable, multi-turn routes. No future market-price oracle is used."""
from __future__ import annotations

from dataclasses import dataclass

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = {"GOOSE": ("COOP", 300), "COW": ("PASTURE", 400),
           "SHEEP": ("PASTURE", 500)}
SHEDS = ((4, 4), (5, 4), (4, 5), (5, 5))
FIRST_YIELD = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8,
               "STRAWBERRY": 10, "MELON": 10}


@dataclass(frozen=True)
class Stop:
    pos: tuple[int, int]
    op: tuple


@dataclass(frozen=True)
class Route:
    actor: int
    kind: str
    stops: tuple[Stop, ...]
    buy: tuple | None = None
    sell: str | None = None

    def distance(self, start):
        total, at = 0, start
        for stop in self.stops:
            total += abs(at[0] - stop.pos[0]) + abs(at[1] - stop.pos[1]) + 1
            at = stop.pos
        return total


def _tile(farm, pos):
    return farm["tiles"][pos[1]][pos[0]]


def _animal(tile):
    return isinstance(tile, dict) and tile.get("animal") in ANIMALS


def _ripe(tile, day):
    if not isinstance(tile, dict) or tile.get("yield_units", 0) <= 0:
        return False
    if tile.get("kind") == "PLANT":
        return day - tile.get("planted_day", day) >= FIRST_YIELD.get(tile.get("crop"), 99)
    return _animal(tile)


def _owned(farm):
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            if tile != "LOCKED":
                yield (x, y), tile


def _nearest_shed(pos):
    return min(SHEDS, key=lambda p: abs(pos[0] - p[0]) + abs(pos[1] - p[1]))


def generate(obs, max_routes=24, cash_credit=0.0):
    """Complete 2-4 stop programs; route zero is always the incumbent.

    The actor chooses an entire sequence. Purchase/placement dependencies are
    checked against observed inventory at execution time. Other workers keep
    operating under the incumbent until a route for them is committed.
    """
    farm = obs["farms"][obs["player"]]
    priv = obs["private"]
    pos = [tuple(farm["farmer"])] + [tuple(p) for p in farm["hands"]]
    bags = priv.get("inventories", [])
    shed = priv.get("shed", {})
    seeds = priv.get("seeds", {})
    day = obs["day"]
    money = farm["money"] + cash_credit
    sites = list(_owned(farm))
    empty = [p for p, t in sites if t is None]
    crops = [(p, t) for p, t in sites if isinstance(t, dict) and t.get("kind") == "PLANT"]
    animals = [(p, t) for p, t in sites if _animal(t)]
    routes = [Route(-1, "BASELINE", ())]
    pool = []

    for actor, start in enumerate(pos):
        bag = bags[actor] if actor < len(bags) else {}
        local = []
        # Harvest is followed by a new investment on the same freed plot,
        # then a return to the shed. This includes the cash-turnover path.
        for site, tile in sites:
            if not _ripe(tile, day):
                continue
            product = tile.get("crop") if tile.get("kind") == "PLANT" else {
                "GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}.get(
                    tile["animal"])
            if product is None:
                continue
            base = (Stop(site, ("HARVEST",)),)
            if tile.get("kind") == "PLANT" and tile.get("crop") in ("WHEAT", "CARROT", "MELON"):
                for crop in CROPS:
                    if day + FIRST_YIELD[crop] >= 30:
                        continue
                    buy = None if seeds.get(crop, 0) else ("BUY_SEED", crop, 1)
                    if buy and money < 150 + {"WHEAT": 10, "CARROT": 20,
                                               "TOMATO": 50, "STRAWBERRY": 100,
                                               "MELON": 80}[crop]:
                        continue
                    stops = base + (Stop(site, ("PLANT", crop)), Stop(site, ("WATER",)),
                                    Stop(_nearest_shed(site), ("DROP",)))
                    local.append(Route(actor, "HARVEST_PLANT_" + crop, stops, buy, product))
            local.append(Route(actor, "HARVEST_SELL", base +
                               (Stop(_nearest_shed(site), ("DROP",)),), None, product))

        # One animal and a nearby crop share one physical work trip. Fertilizer
        # is collected before it is applied; no purchased fertilizer is assumed.
        for a_pos, a in animals:
            fert = a.get("fertilizer_available", 0)
            nearby = sorted((p for p, c in crops
                             if c.get("fertilized_until_day", -1) < day),
                            key=lambda p: abs(p[0] - a_pos[0]) + abs(p[1] - a_pos[1]))
            if fert and nearby:
                crop_pos = nearby[0]
                local.append(Route(actor, "MANURE_CROP", (
                    Stop(a_pos, ("COLLECT_FERTILIZER",)),
                    Stop(crop_pos, ("FERTILIZE",)),
                    Stop(_nearest_shed(crop_pos), ("DROP",)))))
            if not a.get("fed_today") and (bag.get("WHEAT", 0) or shed.get("WHEAT", 0)):
                prefix = () if bag.get("WHEAT", 0) else (
                    Stop(_nearest_shed(start), ("PICKUP", "WHEAT", 1)),)
                local.append(Route(actor, "FEED_CARE", prefix + (
                    Stop(a_pos, ("FEED",)), Stop(a_pos, ("CARE",)))))

        # Empty land can receive any crop; select routes, not an irrevocable
        # tile allocation. Expansion into animals includes the transport chain.
        for site in empty:
            for crop in CROPS:
                if day + FIRST_YIELD[crop] >= 30:
                    continue
                buy = None if seeds.get(crop, 0) else ("BUY_SEED", crop, 1)
                if buy and money < 150 + {"WHEAT": 10, "CARROT": 20,
                                           "TOMATO": 50, "STRAWBERRY": 100,
                                           "MELON": 80}[crop]:
                    continue
                local.append(Route(actor, "PLANT_" + crop, (
                    Stop(site, ("PLANT", crop)), Stop(site, ("WATER",))), buy))
            for animal, (structure, cost) in ANIMALS.items():
                if day + {"GOOSE": 4, "COW": 8, "SHEEP": 6}[animal] >= 30:
                    continue
                buy = None if shed.get(animal, 0) or bag.get(animal, 0) else (
                    "BUY_ANIMAL", animal, 1)
                if buy and money < cost + 300:
                    continue
                pickup = () if bag.get(animal, 0) else (
                    Stop(_nearest_shed(site), ("PICKUP", animal, 1)),)
                local.append(Route(actor, "RAISE_" + animal, (
                    Stop(site, ("BUILD_" + structure,)),) + pickup +
                    (Stop(site, ("PLACE", animal)),), buy))

        pool.extend(local)
    # Keep a representative of each route type, then fill by walking cost.
    # This preserves cow/sheep and every crop in the menu even on busy farms.
    pool.sort(key=lambda r: (r.distance(pos[r.actor]), r.kind, r.actor))
    seen = set()
    for route in pool:
        if route.kind not in seen and len(routes) < max_routes:
            routes.append(route)
            seen.add(route.kind)
    for route in pool:
        if len(routes) >= max_routes:
            break
        if route not in routes:
            routes.append(route)
    return routes


def valid_stop(obs, route, index):
    """Check the next job against the real post-step observation."""
    if index >= len(route.stops):
        return False
    farm = obs["farms"][obs["player"]]
    priv = obs["private"]
    stop = route.stops[index]
    op = stop.op[0]
    tile = _tile(farm, stop.pos)
    bag = priv.get("inventories", [])
    bag = bag[route.actor] if route.actor < len(bag) else {}
    if op == "HARVEST":
        return _ripe(tile, obs["day"])
    if op == "PLANT":
        return tile is None and priv.get("seeds", {}).get(stop.op[1], 0) > 0
    if op == "WATER":
        return isinstance(tile, dict) and tile.get("kind") == "PLANT" and not tile.get("watered_today")
    if op == "FERTILIZE":
        return isinstance(tile, dict) and tile.get("kind") == "PLANT" and bag.get("FERTILIZER", 0) > 0
    if op == "COLLECT_FERTILIZER":
        return _animal(tile) and tile.get("fertilizer_available", 0) > 0
    if op == "FEED":
        return _animal(tile) and not tile.get("fed_today") and bag.get("WHEAT", 0) > 0
    if op == "CARE":
        return _animal(tile) and not tile.get("cared_today")
    if op.startswith("BUILD_"):
        return tile is None
    if op == "PICKUP":
        return priv.get("shed", {}).get(stop.op[1], 0) > 0
    if op == "PLACE":
        structure = ANIMALS[stop.op[1]][0]
        return (isinstance(tile, dict) and tile.get("kind") == structure
                and not tile.get("animal") and bag.get(stop.op[1], 0) > 0)
    if op == "DROP":
        return sum(bag.values()) > 0
    return False


def command_toward(current, target):
    x, y = current
    tx, ty = target
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]
    return None

"""Versioned economic action rules, evaluated before sampling.

The production-window and shed rules follow the Final B design described in
msdsm/kaggriculture-solution (84057a0), heuristics/unit_rules.py and shed_patch.py.
This implementation uses the owned prefix resolver's actual projected state;
it never repairs a sampled action or reads an opponent's private inventory.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from kaggle_environments.envs.kaggriculture import kaggriculture as rules

from ..full_action.catalog import EPISODE_STEPS, PRODUCTS, SHED_CAPACITY, TURNS_PER_DAY

LAST_DAY = EPISODE_STEPS // TURNS_PER_DAY - 1
FINAL_ACTION_STEP = EPISODE_STEPS - 2
FINAL_SALE_STEPS = (FINAL_ACTION_STEP - 1, FINAL_ACTION_STEP)


def observation_step(observation: dict[str, Any]) -> int:
    value = observation.get("step")
    return int(value if value is not None else
               int(observation["day"]) * TURNS_PER_DAY + int(observation["hour"]))


def unit_position(farm: dict, index: int) -> tuple[int, int]:
    return tuple(farm["farmer"] if index == 0 else farm["hands"][index - 1])


def unit_tile(farm: dict, index: int) -> dict:
    x, y = unit_position(farm, index)
    tile = farm["tiles"][y][x]
    return tile if isinstance(tile, dict) else {}


def fertilize_gains(tile: dict, day: int) -> bool:
    """Can an uncovered day in this fertilizer's three-day window add yield?"""
    crop = rules.CROPS.get(tile.get("crop")) if tile.get("kind") == "PLANT" else None
    if crop is None:
        return False
    planted = int(tile["planted_day"])
    existing = int(tile.get("fertilized_until_day", -1))
    for date in range(day, min(day + 2, LAST_DAY) + 1):
        if date <= existing:
            continue
        if crop["ongoing"]:
            # Production at this night's end is available the following morning.
            elapsed = date + 1 - planted - crop["first_yield_day"]
            if (date < LAST_DAY and elapsed >= 0 and elapsed % crop["interval"] == 0
                    and elapsed // crop["interval"] < crop["max_yield"]):
                return True
        elif ((crop["max_yield_day"] + 1) // 2 <= date - planted <= crop["max_yield_day"]
              and not (date == day and tile.get("watered_today", False))
              and int(tile.get("yield_units", 0)) < crop["max_yield"] - 1):
            return True
    return False


def care_gains(tile: dict, day: int) -> bool:
    """CARE banks its bonus after tonight's production, so a later night is needed."""
    animal = rules.ANIMALS.get(tile.get("animal"))
    if animal is None or tile.get("cared_today", False):
        return False
    first = int(tile["placed_day"]) + animal["first_yield_day"]
    return any(date + 1 >= first and (date + 1 - first) % animal["interval"] == 0
               for date in range(day + 1, LAST_DAY))


def productive_placement(item: str, day: int) -> bool:
    parameters = rules.CROPS.get(item) or rules.ANIMALS.get(item)
    return parameters is not None and day + parameters["first_yield_day"] <= LAST_DAY


def unit_economic_support(resolver: Any, index: int, support: np.ndarray,
                          vocabulary: tuple) -> np.ndarray:
    result = np.asarray(support, np.bool_).copy()
    if index >= resolver.unit_count:
        return result
    tile = unit_tile(resolver.farm, index)
    fertilizer = fertilize_gains(tile, resolver.day)
    care = care_gains(tile, resolver.day)
    for identity in np.flatnonzero(result):
        action = vocabulary[int(identity)]
        if action[0] == "FERTILIZE":
            result[identity] = fertilizer
        elif action[0] == "CARE":
            result[identity] = care
        elif action[0] == "PLANT" or (action[0] == "PLACE" and action[1] in rules.ANIMALS):
            result[identity] = productive_placement(str(action[1]), resolver.day)
    return result


def pockets(private: dict) -> dict[str, int]:
    held: dict[str, int] = defaultdict(int)
    for inventory in private.get("inventories", []):
        for item, quantity in (inventory or {}).items():
            held[str(item)] += max(0, int(quantity))
    return dict(held)


def forced_drop(resolver: Any, index: int, step: int) -> bool:
    if step != FINAL_ACTION_STEP or index >= resolver.unit_count:
        return False
    farm, private = resolver.farm, resolver.private
    half = len(farm["tiles"]) // 2
    x, y = unit_position(farm, index)
    inventories = private.get("inventories", [])
    return (x in (half - 1, half) and y in (half - 1, half)
            and index < len(inventories)
            and sum(max(0, int(n)) for n in (inventories[index] or {}).values()) > 0)


def forced_sales(resolver: Any, step: int) -> list[tuple[str, str, int]]:
    """Reserve leading market slots before the remaining neural choices.

    The actual post-unit inventory includes harvest, consumption and DROP. Night
    sales retain tomorrow's feed; unsellable animals or pockets alone may still
    exceed capacity. We do not pretend that rules can prevent those losses.
    """
    private, farm, market = resolver.private, resolver.farm, resolver.market
    stock = {item: max(0, int(private["shed"].get(item, 0))) for item in PRODUCTS}
    if step in FINAL_SALE_STEPS:
        quantities = stock
    elif step % TURNS_PER_DAY == TURNS_PER_DAY - 1 and step // TURNS_PER_DAY < LAST_DAY:
        carried = pockets(private)
        overflow = max(0, sum(max(0, int(n)) for n in private["shed"].values())
                       + sum(carried.values()) - SHED_CAPACITY)
        animals = sum(isinstance(tile, dict) and tile.get("animal") in rules.ANIMALS
                      for row in farm["tiles"] for tile in row)
        stock["WHEAT"] = max(0, stock["WHEAT"] - max(0, animals - carried.get("WHEAT", 0)))
        quantities = {item: 0 for item in PRODUCTS}
        for _ in range(overflow):
            available = [item for item in PRODUCTS if stock[item] > quantities[item]]
            if not available:
                break
            def rank(item: str) -> tuple[float, int, str]:
                quote = rules.market_price(item, int(market["inventory"][item]) + quantities[item])
                return (-quote / rules.MARKET_PARAMS[item]["base"], -quote, item)
            quantities[min(available, key=rank)] += 1
    else:
        return []
    order = sorted((item for item in PRODUCTS if quantities[item] > 0),
                   key=lambda item: (-rules.market_price(item, int(market["inventory"][item]))
                                     * quantities[item], item))
    # The official shed contains at most 100 units; reject corrupt states rather
    # than silently truncating the quantity / action identity.
    if any(quantities[item] > SHED_CAPACITY for item in order):
        raise ValueError("forced sale exceeds the absolute SELL vocabulary")
    return [("SELL", item, quantities[item]) for item in order]


def market_economic_support(resolver: Any, step: int, support: np.ndarray,
                            vocabulary: tuple) -> np.ndarray:
    result = np.asarray(support, np.bool_).copy()
    day, hour = divmod(step, TURNS_PER_DAY)
    earliest_use_day = day + int(hour == TURNS_PER_DAY - 1)
    private = resolver.private
    night_room = SHED_CAPACITY - sum(max(0, int(n)) for n in private["shed"].values()) - sum(pockets(private).values())
    for identity in np.flatnonzero(result):
        action = vocabulary[int(identity)]
        operation = action[0]
        if step in FINAL_SALE_STEPS:
            result[identity] = operation == "NOOP"
        elif operation in ("BUY_SEED", "BUY_ANIMAL"):
            result[identity] = productive_placement(str(action[1]), earliest_use_day)
        elif operation == "HIRE":
            # Daily workers hired after the last unit turn expire at that night.
            result[identity] = hour < TURNS_PER_DAY - 1
        elif operation == "BUY_LAND":
            fastest_crop = min(p["first_yield_day"] for p in rules.CROPS.values())
            result[identity] = earliest_use_day + fastest_crop <= LAST_DAY
        if (result[identity] and hour == TURNS_PER_DAY - 1
                and operation in ("BUY_PRODUCT", "BUY_ANIMAL")):
            result[identity] = int(action[2]) <= max(0, night_room)
    return result

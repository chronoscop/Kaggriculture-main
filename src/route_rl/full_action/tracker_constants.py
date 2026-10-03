# Adapted from msdsm/kaggriculture-solution, commit 84057a0fda4238ccdebc46f9bf5496c6c4b2e00d.
# Source: observations.tracker_constants; see docs/action_bc_sources.md.
"""Rule constants required by the self-contained competition tracker."""

from __future__ import annotations

import math

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
PRODUCTS = (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER")
ANIMALS = ("GOOSE", "COW", "SHEEP")
ITEMS = (*PRODUCTS, *ANIMALS)

SEED_COSTS = {
    "WHEAT": 10,
    "CARROT": 20,
    "TOMATO": 50,
    "STRAWBERRY": 100,
    "MELON": 80,
}
ANIMAL_COSTS = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
FIXED_PURCHASE_COSTS = {**SEED_COSTS, **ANIMAL_COSTS}

ANIMAL_PRODUCTS = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
ANIMAL_STRUCTURES = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}
ANIMAL_MAX_HELD = {"GOOSE": 4, "COW": 6, "SHEEP": 6}
ANIMAL_FIRST_YIELD_DAY = {"GOOSE": 4, "COW": 8, "SHEEP": 6}
ANIMAL_INTERVAL = {"GOOSE": 1, "COW": 2, "SHEEP": 3}

CROP_FIRST_YIELD_DAY = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 10}
CROP_MAX_YIELD_DAY = {"WHEAT": 4, "CARROT": 3, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 12}
CROP_INTERVAL = {"WHEAT": 0, "CARROT": 0, "TOMATO": 1, "STRAWBERRY": 2, "MELON": 0}
CROP_MAX_YIELD = {"WHEAT": 6, "CARROT": 4, "TOMATO": 4, "STRAWBERRY": 4, "MELON": 6}
CROP_ONGOING = {"WHEAT": False, "CARROT": False, "TOMATO": True, "STRAWBERRY": True, "MELON": False}

LAND_PRICES = (1_000, 2_000, 4_000)
SHED_CAPACITY = 100
TURNS_PER_DAY = 24

TOWN_CENTER_PRODUCTS = tuple(item for item in PRODUCTS if item != "FERTILIZER")
SHOPS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}

MARKET_I0 = 10_000
PRICE_FLOOR = 1
HINGE_GAIN = 8.0
MARKET_PARAMS = {
    "WHEAT": {
        "base": 25,
        "T": 400,
        "below_func": "sqrt",
        "below_target": 0.80,
        "above_func": "log",
        "above_target": 0.20,
    },
    "CARROT": {
        "base": 35,
        "T": 450,
        "below_func": "hinge",
        "below_target": 1.00,
        "above_func": "sqrt",
        "above_target": 0.70,
    },
    "TOMATO": {
        "base": 60,
        "T": 200,
        "below_func": "hinge",
        "below_target": 0.40,
        "above_func": "sqrt",
        "above_target": 0.60,
    },
    "STRAWBERRY": {
        "base": 120,
        "T": 100,
        "below_func": "sqrt",
        "below_target": 0.70,
        "above_func": "linear",
        "above_target": 1.60,
    },
    "MELON": {
        "base": 250,
        "T": 300,
        "below_func": "log",
        "below_target": 0.20,
        "above_func": "sq",
        "above_target": 3.60,
    },
    "EGG": {
        "base": 50,
        "T": 332,
        "below_func": "hinge",
        "below_target": 0.40,
        "above_func": "log",
        "above_target": 0.20,
    },
    "MILK": {
        "base": 160,
        "T": 122,
        "below_func": "sqrt",
        "below_target": 0.60,
        "above_func": "linear",
        "above_target": 1.60,
    },
    "WOOL": {
        "base": 200,
        "T": 105,
        "below_func": "log",
        "below_target": 0.20,
        "above_func": "sq",
        "above_target": 3.20,
    },
    "FERTILIZER": {
        "base": 100,
        "T": 200,
        "below_func": "linear",
        "below_target": 0.40,
        "above_func": "linear",
        "above_target": 0.40,
    },
}


def _shape(name: str, distance: float, scale: float) -> float:
    distance = max(0.0, distance)
    if name == "linear":
        return distance
    if name == "sq":
        return distance * distance
    if name == "sqrt":
        return math.sqrt(distance)
    if name == "log":
        return math.log1p(distance)
    if name == "hinge":
        normalized = distance / scale
        return normalized + HINGE_GAIN * max(0.0, normalized - 1.0) ** 2
    raise ValueError(f"unknown market shape {name}")


def market_price(item: str, inventory: int) -> int:
    """Match the 1.32.7 market-price function for the default configuration."""
    params = MARKET_PARAMS[item]
    distance = abs(inventory - MARKET_I0)
    below = inventory < MARKET_I0
    side = "below" if below else "above"
    shape = _shape(str(params[f"{side}_func"]), distance, float(params["T"]))
    anchor = _shape(str(params[f"{side}_func"]), float(params["T"]), float(params["T"]))
    amplitude = float(params[f"{side}_target"]) * float(params["base"]) / anchor
    raw_price = float(params["base"]) + (amplitude * shape if below else -amplitude * shape)
    return max(PRICE_FLOOR, round(raw_price))


def hire_cost(hires_before: int, count: int) -> int:
    """Return the cost of `count` successful hires after `hires_before` hires today."""
    a, b = 1, 1
    costs: list[int] = []
    for _ in range(hires_before + count):
        costs.append(a)
        a, b = b, a + b
    return sum(costs[hires_before:])

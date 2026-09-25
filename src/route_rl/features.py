"""Observation-only features for ranking complete route candidates."""
from __future__ import annotations

from .routes import ANIMALS, CROPS

PRODUCTS = CROPS + ("EGG", "MILK", "WOOL", "FERTILIZER")
KINDS = ("BASELINE", "HARVEST", "PLANT", "RAISE", "MANURE", "FEED")
FEATURE_SIZE = 64


def encode(obs, route):
    seat = obs["player"]
    farm, rival = obs["farms"][seat], obs["farms"][1 - seat]
    priv = obs["private"]
    shops = obs["town"].get("unlocked_shops", [])
    tiles = [tile for row in farm["tiles"] for tile in row]
    counts = [sum(isinstance(t, dict) and t.get("crop") == crop for t in tiles)
              for crop in CROPS]
    counts += [sum(isinstance(t, dict) and t.get("animal") == a for t in tiles)
               for a in ANIMALS]
    counts += [sum(t is None for t in tiles), len(farm["hands"])]
    prices = obs["market"].get("prices", {})
    market_stock = obs["market"].get("inventory", {})
    v = [obs["step"] / 719, obs["hour"] / 24,
         farm["money"] / 10000, rival["money"] / 10000,
         len(farm["unlocked_quadrants"]) / 4, len(shops) / 8,
         sum(priv.get("shed", {}).values()) / 100]
    v += [prices.get(p, 0) / 200 for p in PRODUCTS]
    v += [market_stock.get(p, 0) / 1000 for p in PRODUCTS]
    v += [x / 25 for x in counts]
    v += [int(route.kind.startswith(k)) for k in KINDS]
    v += [route.actor / 12, len(route.stops) / 8,
          int(route.buy is not None), int(route.sell is not None)]
    if route.stops:
        start = ([farm["farmer"]] + farm["hands"])[route.actor]
        v += [start[0] / 10, start[1] / 10,
              route.stops[0].pos[0] / 10, route.stops[0].pos[1] / 10,
              route.distance(start) / 40]
    else:
        v += [0] * 5
    v += [int(crop in route.kind) for crop in CROPS]
    v += [int(animal in route.kind) for animal in ANIMALS]
    if len(v) > FEATURE_SIZE:
        raise AssertionError(f"feature vector grew to {len(v)}")
    return v + [0.0] * (FEATURE_SIZE - len(v))


def menu_features(obs, routes, max_routes=24):
    if len(routes) > max_routes:
        raise ValueError("candidate cap exceeded")
    rows = [encode(obs, route) for route in routes]
    mask = [True] * len(rows)
    rows.extend([[0.0] * FEATURE_SIZE for _ in range(max_routes - len(rows))])
    mask.extend([False] * (max_routes - len(mask)))
    return rows, mask

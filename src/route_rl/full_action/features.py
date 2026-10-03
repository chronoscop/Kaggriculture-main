# Adapted from msdsm/kaggriculture-solution, commit 84057a0fda4238ccdebc46f9bf5496c6c4b2e00d.
# Source: observations.features; see docs/action_bc_sources.md.
"""Tokenize the current observation and heuristic opponent inventory belief."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from route_rl.full_action.catalog import (
    ANIMALS,
    BOARD_SIZE,
    CROPS,
    EPISODE_STEPS,
    FARM_ROLES,
    ITEMS,
    MARKET_I0,
    MARKET_SLOTS,
    PRODUCTS,
    QUADRANTS,
    SHED_CAPACITY,
    SHOP_PRODUCTS,
    SHOPS,
    TILE_KINDS,
    TOKEN_TYPES,
    TURNS_PER_DAY,
    UNIT_TYPES,
)
from route_rl.full_action.inventory_tracker import InventoryEstimate


def categorical_names(prefix: str, values: tuple[str, ...]) -> list[str]:
    return [f"{prefix}:{value}" for value in values]


NUMERIC_FEATURES = (
    "plant_age",
    "yield_units",
    "watered",
    "consecutive_unwatered",
    "fertilizer_remaining",
    "has_decay_step",
    "turns_until_decay",
    "animal_age",
    "fed",
    "consecutive_unfed",
    "cared",
    "fertilizer_available",
    "pending_care_bonus",
    "shed_access",
    "unit_shed_access",
    "farmer_occupancy",
    "hand_occupancy",
    "cell_x",
    "cell_y",
    "unit_index",
    "unit_x",
    "unit_y",
    "inventory_visible",
    *(f"inventory:{item}" for item in ITEMS),
    "season_progress",
    "day_progress",
    "day_remaining",
    "season_remaining",
    "shop_tick_progress",
    "town_tick_progress",
    "self_money",
    "opponent_money",
    "self_hires_today",
    "opponent_hires_today",
    "self_unlocked_count",
    "opponent_unlocked_count",
    *(f"self_quadrant:{quadrant}" for quadrant in QUADRANTS),
    *(f"opponent_quadrant:{quadrant}" for quadrant in QUADRANTS),
    *(f"shop_count:{shop}" for shop in SHOPS),
    "product_shed_count",
    "product_seed_count",
    "product_seed_applicable",
    "product_market_inventory",
    "product_market_price",
    "product_market_visible",
    "product_town_demand",
    # Shed capacity is a total across items, so totals and remaining room are decision inputs
    # that per-item counts only imply; they are appended last so older adapters extend by zero rows.
    "shed_total",
    "shed_room",
    "units_inventory_total",
    "shed_overflow_pending",
    "inventory_total",
    "self_hand_count",
    "opponent_hand_count",
)

FEATURE_NAMES = tuple(
    categorical_names("token", TOKEN_TYPES)
    + categorical_names("farm", FARM_ROLES)
    + categorical_names("tile", TILE_KINDS)
    + categorical_names("crop", CROPS)
    + categorical_names("animal", ANIMALS)
    + categorical_names("item", ITEMS)
    + categorical_names("unit", UNIT_TYPES)
    + [f"market_slot:{index}" for index in range(MARKET_SLOTS)]
    + list(NUMERIC_FEATURES)
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
FEATURE_DIM = len(FEATURE_NAMES)
MEMORY_PACK_FEATURE_INDICES = tuple(range(FEATURE_INDEX["tile:EMPTY"], FEATURE_INDEX["tile:EMPTY"] + 50))

GLOBAL_FEATURE_NAMES = (
    "season_progress",
    "day_progress",
    "day_remaining",
    "season_remaining",
    "shop_tick_progress",
    "town_tick_progress",
    "self_money",
    "opponent_money",
    "self_hires_today",
    "opponent_hires_today",
    "self_unlocked_count",
    "opponent_unlocked_count",
    *(f"self_quadrant:{quadrant}" for quadrant in QUADRANTS),
    *(f"opponent_quadrant:{quadrant}" for quadrant in QUADRANTS),
    *(f"shop_count:{shop}" for shop in SHOPS),
    "shed_total",
    "shed_room",
    "units_inventory_total",
    "shed_overflow_pending",
    "self_hand_count",
    "opponent_hand_count",
)
CELL_FEATURE_NAMES = (
    "farm:SELF",
    "farm:OPPONENT",
    *categorical_names("tile", TILE_KINDS),
    *categorical_names("crop", CROPS),
    *categorical_names("animal", ANIMALS),
    "plant_age",
    "yield_units",
    "watered",
    "consecutive_unwatered",
    "fertilizer_remaining",
    "has_decay_step",
    "turns_until_decay",
    "animal_age",
    "fed",
    "consecutive_unfed",
    "cared",
    "fertilizer_available",
    "pending_care_bonus",
    "shed_access",
    "farmer_occupancy",
    "hand_occupancy",
    "cell_x",
    "cell_y",
)
UNIT_FEATURE_NAMES = (
    "farm:SELF",
    "farm:OPPONENT",
    *categorical_names("unit", UNIT_TYPES),
    "unit_shed_access",
    "unit_index",
    "unit_x",
    "unit_y",
    "inventory_visible",
    *(f"inventory:{item}" for item in ITEMS),
    "inventory_total",
)
PRODUCT_FEATURE_NAMES = (
    *categorical_names("item", ITEMS),
    "product_shed_count",
    "product_seed_count",
    "product_seed_applicable",
    "product_market_inventory",
    "product_market_price",
    "product_market_visible",
    "product_town_demand",
)
MARKET_SLOT_FEATURE_NAMES = tuple(f"market_slot:{index}" for index in range(MARKET_SLOTS))
MEMORY_FEATURE_NAMES = (
    *(f"opponent_owned:{item}" for item in ITEMS),
    *(f"opponent_carried:{item}" for item in ITEMS),
    *(f"opponent_owned_uncertainty:{item}" for item in ITEMS),
    *(f"opponent_carried_uncertainty:{item}" for item in ITEMS),
    "opponent_unresolved_cash",
    "opponent_floor_sale_ambiguity",
)
MEMORY_CACHE_VERSION = 1

TOKEN_ADAPTER_FEATURES = {
    "GLOBAL": GLOBAL_FEATURE_NAMES,
    "CELL": CELL_FEATURE_NAMES,
    "UNIT": UNIT_FEATURE_NAMES,
    "PRODUCT": PRODUCT_FEATURE_NAMES,
    "MEMORY": MEMORY_FEATURE_NAMES,
    "MARKET_SLOT": MARKET_SLOT_FEATURE_NAMES,
}

ROPE_GROUP_NONE = 0
ROPE_GROUP_SELF = 1
ROPE_GROUP_OPPONENT = 2


@dataclass(frozen=True)
class EncodedObservation:
    features: np.ndarray
    memory_features: np.ndarray
    coordinates: np.ndarray
    spatial_mask: np.ndarray
    rope_groups: np.ndarray
    own_unit_indices: np.ndarray
    market_slot_indices: np.ndarray


def clipped(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def log_scaled(value: float, reference: float) -> float:
    return math.log1p(max(0.0, value)) / math.log1p(reference)


def set_one_hot(features: list[float], prefix: str, value: str) -> None:
    features[FEATURE_INDEX[f"{prefix}:{value}"]] = 1.0


def set_numeric(features: list[float], name: str, value: float | bool) -> None:
    features[FEATURE_INDEX[name]] = float(value)


def new_token(token_type: str, farm_role: str = "NONE") -> list[float]:
    features = [0.0] * FEATURE_DIM
    set_one_hot(features, "token", token_type)
    set_one_hot(features, "farm", farm_role)
    return features


def observation_step(observation: dict[str, Any]) -> int:
    if "step" in observation:
        return int(observation["step"])
    return int(observation.get("day", 0)) * TURNS_PER_DAY + int(observation.get("hour", 0))


def global_token(observation: dict[str, Any], self_farm: dict[str, Any], opponent_farm: dict[str, Any]) -> list[float]:
    features = new_token("GLOBAL")
    step = observation_step(observation)
    hour = int(observation.get("hour", step % TURNS_PER_DAY))
    set_numeric(features, "season_progress", clipped(step / (EPISODE_STEPS - 1)))
    set_numeric(features, "day_progress", clipped(hour / (TURNS_PER_DAY - 1)))
    set_numeric(features, "day_remaining", clipped((TURNS_PER_DAY - 1 - hour) / TURNS_PER_DAY))
    set_numeric(features, "season_remaining", clipped((EPISODE_STEPS - 1 - step) / EPISODE_STEPS))
    set_numeric(features, "shop_tick_progress", (step % 4) / 3)
    set_numeric(features, "town_tick_progress", (step % TURNS_PER_DAY) / (TURNS_PER_DAY - 1))
    set_numeric(features, "self_money", log_scaled(float(self_farm.get("money", 0.0)), 1_000_000.0))
    set_numeric(features, "opponent_money", log_scaled(float(opponent_farm.get("money", 0.0)), 1_000_000.0))
    set_numeric(features, "self_hires_today", log_scaled(int(self_farm.get("hires_today", 0)), 32.0))
    set_numeric(features, "opponent_hires_today", log_scaled(int(opponent_farm.get("hires_today", 0)), 32.0))

    self_quadrants = set(self_farm.get("unlocked_quadrants", []))
    opponent_quadrants = set(opponent_farm.get("unlocked_quadrants", []))
    set_numeric(features, "self_unlocked_count", len(self_quadrants) / len(QUADRANTS))
    set_numeric(features, "opponent_unlocked_count", len(opponent_quadrants) / len(QUADRANTS))
    for quadrant in QUADRANTS:
        set_numeric(features, f"self_quadrant:{quadrant}", quadrant in self_quadrants)
        set_numeric(features, f"opponent_quadrant:{quadrant}", quadrant in opponent_quadrants)

    shop_counts = {shop: 0 for shop in SHOPS}
    for shop in observation.get("town", {}).get("unlocked_shops", []):
        if shop in shop_counts:
            shop_counts[shop] += 1
    for shop, count in shop_counts.items():
        set_numeric(features, f"shop_count:{shop}", count / 8)

    private = observation.get("private", {})
    shed_total = sum(max(0, int(count)) for count in private.get("shed", {}).values())
    shed_room = max(0, SHED_CAPACITY - shed_total)
    carried = sum(max(0, int(count)) for inventory in private.get("inventories", []) for count in inventory.values())
    set_numeric(features, "shed_total", clipped(shed_total / SHED_CAPACITY))
    set_numeric(features, "shed_room", clipped(shed_room / SHED_CAPACITY))
    # Linear so the model can compare carried items with shed room without inverting a log.
    set_numeric(features, "units_inventory_total", carried / SHED_CAPACITY)
    set_numeric(features, "shed_overflow_pending", clipped(max(0, carried - shed_room) / SHED_CAPACITY))
    set_numeric(features, "self_hand_count", len(self_farm.get("hands", [])) / 32)
    set_numeric(features, "opponent_hand_count", len(opponent_farm.get("hands", [])) / 32)
    return features


def position_counts(farm: dict[str, Any]) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    farmer_counts: dict[tuple[int, int], int] = {}
    hand_counts: dict[tuple[int, int], int] = {}
    farmer = tuple(int(value) for value in farm.get("farmer", (-1, -1)))
    farmer_counts[farmer] = farmer_counts.get(farmer, 0) + 1
    for hand in farm.get("hands", []):
        position = tuple(int(value) for value in hand)
        hand_counts[position] = hand_counts.get(position, 0) + 1
    return farmer_counts, hand_counts


def shed_access(x: int, y: int) -> bool:
    half = BOARD_SIZE // 2
    return (x, y) in {
        (half - 1, half - 1),
        (half, half - 1),
        (half - 1, half),
        (half, half),
    }


def cell_token(
    tile: Any,
    farm_role: str,
    day: int,
    step: int,
    x: int,
    y: int,
    farmer_count: int,
    hand_count: int,
) -> list[float]:
    features = new_token("CELL", farm_role)
    set_numeric(features, "shed_access", shed_access(x, y))
    set_numeric(features, "farmer_occupancy", clipped(farmer_count))
    set_numeric(features, "hand_occupancy", log_scaled(hand_count, 32.0))
    set_numeric(features, "cell_x", clipped(x / (BOARD_SIZE - 1)))
    set_numeric(features, "cell_y", clipped(y / (BOARD_SIZE - 1)))

    if tile is None:
        set_one_hot(features, "tile", "EMPTY")
        return features
    if tile == "LOCKED":
        set_one_hot(features, "tile", "LOCKED")
        return features
    if not isinstance(tile, dict):
        raise TypeError(f"unsupported tile type: {type(tile).__name__}")

    kind = str(tile.get("kind", ""))
    if kind == "WEED":
        set_one_hot(features, "tile", "WEED")
        return features
    if kind == "PLANT":
        set_one_hot(features, "tile", "PLANT")
        crop = str(tile.get("crop"))
        if crop in CROPS:
            set_one_hot(features, "crop", crop)
        set_numeric(features, "plant_age", clipped((day - int(tile.get("planted_day", day))) / 30))
        set_numeric(features, "yield_units", log_scaled(float(tile.get("yield_units", 0)), 8.0))
        set_numeric(features, "watered", bool(tile.get("watered_today", False)))
        set_numeric(features, "consecutive_unwatered", clipped(float(tile.get("consecutive_unwatered", 0)) / 2))
        fertilizer_until = int(tile.get("fertilized_until_day", -1))
        set_numeric(features, "fertilizer_remaining", clipped((fertilizer_until - day + 1) / 3))
        max_lifespan_step = int(tile.get("max_lifespan_step", -1))
        if max_lifespan_step >= 0:
            set_numeric(features, "has_decay_step", 1)
            set_numeric(features, "turns_until_decay", clipped((max_lifespan_step - step) / EPISODE_STEPS, -1, 1))
        return features
    if kind not in {"COOP", "PASTURE"}:
        raise ValueError(f"unsupported tile kind: {kind}")

    set_one_hot(features, "tile", kind)
    animal = tile.get("animal")
    if animal in ANIMALS:
        set_one_hot(features, "animal", animal)
        set_numeric(features, "animal_age", clipped((day - int(tile.get("placed_day", day))) / 30))
        set_numeric(features, "yield_units", log_scaled(float(tile.get("yield_units", 0)), 8.0))
        set_numeric(features, "fed", bool(tile.get("fed_today", False)))
        set_numeric(features, "consecutive_unfed", clipped(float(tile.get("consecutive_unfed", 0)) / 2))
        set_numeric(features, "cared", bool(tile.get("cared_today", False)))
        set_numeric(features, "fertilizer_available", bool(tile.get("fertilizer_available", False)))
        set_numeric(features, "pending_care_bonus", log_scaled(float(tile.get("pending_care_bonus", 0)), 16.0))
    return features


def unit_token(
    farm_role: str,
    unit_type: str,
    unit_index: int,
    x: int,
    y: int,
    inventory: dict[str, Any] | None,
) -> list[float]:
    features = new_token("UNIT", farm_role)
    set_one_hot(features, "unit", unit_type)
    set_numeric(features, "unit_index", clipped(unit_index / 31))
    set_numeric(features, "unit_x", clipped(x / (BOARD_SIZE - 1)))
    set_numeric(features, "unit_y", clipped(y / (BOARD_SIZE - 1)))
    set_numeric(features, "unit_shed_access", shed_access(x, y))
    visible = inventory is not None
    set_numeric(features, "inventory_visible", visible)
    if inventory is not None:
        for item in ITEMS:
            set_numeric(features, f"inventory:{item}", log_scaled(float(inventory.get(item, 0)), SHED_CAPACITY))
        total = sum(max(0, int(inventory.get(item, 0))) for item in ITEMS)
        set_numeric(features, "inventory_total", total / SHED_CAPACITY)
    return features


def town_demand(observation: dict[str, Any]) -> dict[str, int]:
    demand = {item: 0 for item in PRODUCTS}
    for item in PRODUCTS:
        if item != "FERTILIZER":
            demand[item] += 1
    for shop in observation.get("town", {}).get("unlocked_shops", []):
        products = SHOP_PRODUCTS.get(shop, ())
        multiplier = 2 if len(products) == 1 else 1
        for item in products:
            demand[item] += multiplier
    return demand


def product_token(
    observation: dict[str, Any],
    item: str,
    demand: dict[str, int] | None = None,
) -> list[float]:
    features = new_token("PRODUCT")
    set_one_hot(features, "item", item)
    private = observation.get("private", {})
    set_numeric(features, "product_shed_count", log_scaled(float(private.get("shed", {}).get(item, 0)), SHED_CAPACITY))
    if item in CROPS:
        set_numeric(features, "product_seed_applicable", 1)
        set_numeric(features, "product_seed_count", log_scaled(float(private.get("seeds", {}).get(item, 0)), 128.0))
    if item in PRODUCTS:
        market = observation.get("market", {})
        inventory = float(market.get("inventory", {}).get(item, MARKET_I0))
        price = float(market.get("prices", {}).get(item, 0))
        set_numeric(features, "product_market_visible", 1)
        set_numeric(features, "product_market_inventory", math.asinh((inventory - MARKET_I0) / 100.0) / 8)
        set_numeric(features, "product_market_price", log_scaled(price, 10_000.0))
        if demand is None:
            demand = town_demand(observation)
        set_numeric(features, "product_town_demand", demand.get(item, 0) / 20)
    return features


def encode_memory_estimate(estimate: InventoryEstimate) -> np.ndarray:
    values = [
        *(log_scaled(estimate.owned[item], SHED_CAPACITY) for item in ITEMS),
        *(log_scaled(estimate.carried[item], SHED_CAPACITY) for item in ITEMS),
        *(log_scaled(estimate.owned_uncertainty[item], SHED_CAPACITY) for item in ITEMS),
        *(log_scaled(estimate.carried_uncertainty[item], SHED_CAPACITY) for item in ITEMS),
        log_scaled(estimate.unresolved_cash, 1_000_000.0),
        log_scaled(estimate.floor_sale_ambiguity, SHED_CAPACITY),
    ]
    return np.asarray(values, dtype=np.float32)


def memory_token(
    estimate: InventoryEstimate | None,
    encoded_memory: np.ndarray | None = None,
) -> tuple[list[float], np.ndarray]:
    if estimate is not None and encoded_memory is not None:
        raise ValueError("provide an inventory estimate or encoded memory, not both")
    if estimate is not None:
        encoded_memory = encode_memory_estimate(estimate)
    if encoded_memory is None:
        encoded_memory = np.zeros(len(MEMORY_FEATURE_NAMES), dtype=np.float32)
    if encoded_memory.shape != (len(MEMORY_FEATURE_NAMES),):
        raise ValueError(f"encoded memory must have shape ({len(MEMORY_FEATURE_NAMES)},)")
    encoded_memory = np.asarray(encoded_memory, dtype=np.float32)
    features = new_token("MEMORY")
    for feature_index, value in zip(MEMORY_PACK_FEATURE_INDICES, encoded_memory):
        features[feature_index] = float(value)
    return features, encoded_memory


def _encode_observation_reference(
    observation: dict[str, Any],
    opponent_inventory: InventoryEstimate | None = None,
    encoded_opponent_memory: np.ndarray | None = None,
) -> EncodedObservation:
    player = int(observation.get("player", 0))
    farms = observation.get("farms", [])
    if len(farms) != 2 or player not in (0, 1):
        raise ValueError("observation must contain two farms and player 0 or 1")
    self_farm = farms[player]
    opponent_farm = farms[1 - player]
    day = int(observation.get("day", 0))
    step = observation_step(observation)

    feature_rows: list[list[float]] = [global_token(observation, self_farm, opponent_farm)]
    coordinates: list[tuple[int, int]] = [(0, 0)]
    spatial: list[bool] = [False]
    rope_groups: list[int] = [ROPE_GROUP_NONE]

    for farm, role in ((self_farm, "SELF"), (opponent_farm, "OPPONENT")):
        rope_group = ROPE_GROUP_SELF if role == "SELF" else ROPE_GROUP_OPPONENT
        farmer_counts, hand_counts = position_counts(farm)
        tiles = farm.get("tiles", [])
        if len(tiles) != BOARD_SIZE or any(len(row) != BOARD_SIZE for row in tiles):
            raise ValueError("experiment 41 expects a 10 x 10 board")
        for y, row in enumerate(tiles):
            for x, tile in enumerate(row):
                feature_rows.append(
                    cell_token(
                        tile,
                        role,
                        day,
                        step,
                        x,
                        y,
                        farmer_counts.get((x, y), 0),
                        hand_counts.get((x, y), 0),
                    )
                )
                coordinates.append((x, y))
                spatial.append(True)
                rope_groups.append(rope_group)

    own_unit_indices: list[int] = []
    inventories = observation.get("private", {}).get("inventories", [])
    for index, position in enumerate([self_farm.get("farmer", [0, 0]), *self_farm.get("hands", [])]):
        own_unit_indices.append(len(feature_rows))
        inventory = inventories[index] if index < len(inventories) else {}
        x, y = int(position[0]), int(position[1])
        feature_rows.append(unit_token("SELF", "FARMER" if index == 0 else "HAND", index, x, y, inventory))
        coordinates.append((x, y))
        spatial.append(True)
        rope_groups.append(ROPE_GROUP_SELF)

    for index, position in enumerate([opponent_farm.get("farmer", [0, 0]), *opponent_farm.get("hands", [])]):
        x, y = int(position[0]), int(position[1])
        feature_rows.append(unit_token("OPPONENT", "FARMER" if index == 0 else "HAND", index, x, y, None))
        coordinates.append((x, y))
        spatial.append(True)
        rope_groups.append(ROPE_GROUP_OPPONENT)

    demand = town_demand(observation)
    for item in ITEMS:
        feature_rows.append(product_token(observation, item, demand))
        coordinates.append((0, 0))
        spatial.append(False)
        rope_groups.append(ROPE_GROUP_NONE)

    memory_features_row, memory_features = memory_token(opponent_inventory, encoded_opponent_memory)
    feature_rows.append(memory_features_row)
    coordinates.append((0, 0))
    spatial.append(False)
    rope_groups.append(ROPE_GROUP_NONE)

    market_slot_indices: list[int] = []
    for slot in range(MARKET_SLOTS):
        market_slot_indices.append(len(feature_rows))
        features = new_token("MARKET_SLOT")
        features[FEATURE_INDEX[f"market_slot:{slot}"]] = 1.0
        feature_rows.append(features)
        coordinates.append((0, 0))
        spatial.append(False)
        rope_groups.append(ROPE_GROUP_NONE)

    return EncodedObservation(
        features=np.asarray(feature_rows, dtype=np.float32),
        memory_features=memory_features,
        coordinates=np.asarray(coordinates, dtype=np.float32),
        spatial_mask=np.asarray(spatial, dtype=np.bool_),
        rope_groups=np.asarray(rope_groups, dtype=np.int32),
        own_unit_indices=np.asarray(own_unit_indices, dtype=np.int32),
        market_slot_indices=np.asarray(market_slot_indices, dtype=np.int32),
    )


@lru_cache(maxsize=len(FARM_ROLES))
def _cell_templates(farm_role: str) -> np.ndarray:
    templates = np.zeros((BOARD_SIZE * BOARD_SIZE, FEATURE_DIM), dtype=np.float32)
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            row = templates[y * BOARD_SIZE + x]
            row[FEATURE_INDEX["token:CELL"]] = 1.0
            row[FEATURE_INDEX[f"farm:{farm_role}"]] = 1.0
            row[FEATURE_INDEX["shed_access"]] = float(shed_access(x, y))
            row[FEATURE_INDEX["cell_x"]] = clipped(x / (BOARD_SIZE - 1))
            row[FEATURE_INDEX["cell_y"]] = clipped(y / (BOARD_SIZE - 1))
    templates.setflags(write=False)
    return templates


def _fill_cell_dynamic(
    features: np.ndarray,
    tile: Any,
    day: int,
    step: int,
    farmer_count: int,
    hand_count: int,
) -> None:
    features[FEATURE_INDEX["farmer_occupancy"]] = clipped(farmer_count)
    features[FEATURE_INDEX["hand_occupancy"]] = log_scaled(hand_count, 32.0)

    if tile is None:
        features[FEATURE_INDEX["tile:EMPTY"]] = 1.0
        return
    if tile == "LOCKED":
        features[FEATURE_INDEX["tile:LOCKED"]] = 1.0
        return
    if not isinstance(tile, dict):
        raise TypeError(f"unsupported tile type: {type(tile).__name__}")

    kind = str(tile.get("kind", ""))
    if kind == "WEED":
        features[FEATURE_INDEX["tile:WEED"]] = 1.0
        return
    if kind == "PLANT":
        features[FEATURE_INDEX["tile:PLANT"]] = 1.0
        crop = str(tile.get("crop"))
        if crop in CROPS:
            features[FEATURE_INDEX[f"crop:{crop}"]] = 1.0
        features[FEATURE_INDEX["plant_age"]] = clipped((day - int(tile.get("planted_day", day))) / 30)
        features[FEATURE_INDEX["yield_units"]] = log_scaled(float(tile.get("yield_units", 0)), 8.0)
        features[FEATURE_INDEX["watered"]] = float(bool(tile.get("watered_today", False)))
        features[FEATURE_INDEX["consecutive_unwatered"]] = clipped(float(tile.get("consecutive_unwatered", 0)) / 2)
        fertilizer_until = int(tile.get("fertilized_until_day", -1))
        features[FEATURE_INDEX["fertilizer_remaining"]] = clipped((fertilizer_until - day + 1) / 3)
        max_lifespan_step = int(tile.get("max_lifespan_step", -1))
        if max_lifespan_step >= 0:
            features[FEATURE_INDEX["has_decay_step"]] = 1.0
            features[FEATURE_INDEX["turns_until_decay"]] = clipped(
                (max_lifespan_step - step) / EPISODE_STEPS,
                -1,
                1,
            )
        return
    if kind not in {"COOP", "PASTURE"}:
        raise ValueError(f"unsupported tile kind: {kind}")

    features[FEATURE_INDEX[f"tile:{kind}"]] = 1.0
    animal = tile.get("animal")
    if animal not in ANIMALS:
        return
    features[FEATURE_INDEX[f"animal:{animal}"]] = 1.0
    features[FEATURE_INDEX["animal_age"]] = clipped((day - int(tile.get("placed_day", day))) / 30)
    features[FEATURE_INDEX["yield_units"]] = log_scaled(float(tile.get("yield_units", 0)), 8.0)
    features[FEATURE_INDEX["fed"]] = float(bool(tile.get("fed_today", False)))
    features[FEATURE_INDEX["consecutive_unfed"]] = clipped(float(tile.get("consecutive_unfed", 0)) / 2)
    features[FEATURE_INDEX["cared"]] = float(bool(tile.get("cared_today", False)))
    features[FEATURE_INDEX["fertilizer_available"]] = float(bool(tile.get("fertilizer_available", False)))
    features[FEATURE_INDEX["pending_care_bonus"]] = log_scaled(float(tile.get("pending_care_bonus", 0)), 16.0)


def encode_observation(
    observation: dict[str, Any],
    opponent_inventory: InventoryEstimate | None = None,
    encoded_opponent_memory: np.ndarray | None = None,
) -> EncodedObservation:
    """Encode directly into contiguous arrays while preserving the reference layout."""
    player = int(observation.get("player", 0))
    farms = observation.get("farms", [])
    if len(farms) != 2 or player not in (0, 1):
        raise ValueError("observation must contain two farms and player 0 or 1")
    self_farm = farms[player]
    opponent_farm = farms[1 - player]
    own_positions = [self_farm.get("farmer", [0, 0]), *self_farm.get("hands", [])]
    opponent_positions = [opponent_farm.get("farmer", [0, 0]), *opponent_farm.get("hands", [])]
    token_count = (
        1 + 2 * BOARD_SIZE * BOARD_SIZE + len(own_positions) + len(opponent_positions) + len(ITEMS) + 1 + MARKET_SLOTS
    )
    feature_rows = np.zeros((token_count, FEATURE_DIM), dtype=np.float32)
    coordinates = np.zeros((token_count, 2), dtype=np.float32)
    spatial = np.zeros(token_count, dtype=np.bool_)
    rope_groups = np.zeros(token_count, dtype=np.int32)
    day = int(observation.get("day", 0))
    step = observation_step(observation)

    feature_rows[0] = global_token(observation, self_farm, opponent_farm)
    cursor = 1
    for farm, role, rope_group in (
        (self_farm, "SELF", ROPE_GROUP_SELF),
        (opponent_farm, "OPPONENT", ROPE_GROUP_OPPONENT),
    ):
        farmer_counts, hand_counts = position_counts(farm)
        tiles = farm.get("tiles", [])
        if len(tiles) != BOARD_SIZE or any(len(row) != BOARD_SIZE for row in tiles):
            raise ValueError("experiment 41 expects a 10 x 10 board")
        block_end = cursor + BOARD_SIZE * BOARD_SIZE
        feature_rows[cursor:block_end] = _cell_templates(role)
        spatial[cursor:block_end] = True
        rope_groups[cursor:block_end] = rope_group
        for y, row in enumerate(tiles):
            for x, tile in enumerate(row):
                index = cursor + y * BOARD_SIZE + x
                coordinates[index] = (x, y)
                _fill_cell_dynamic(
                    feature_rows[index],
                    tile,
                    day,
                    step,
                    farmer_counts.get((x, y), 0),
                    hand_counts.get((x, y), 0),
                )
        cursor = block_end

    own_unit_indices = np.arange(cursor, cursor + len(own_positions), dtype=np.int32)
    inventories = observation.get("private", {}).get("inventories", [])
    for index, position in enumerate(own_positions):
        inventory = inventories[index] if index < len(inventories) else {}
        x, y = int(position[0]), int(position[1])
        feature_rows[cursor] = unit_token("SELF", "FARMER" if index == 0 else "HAND", index, x, y, inventory)
        coordinates[cursor] = (x, y)
        spatial[cursor] = True
        rope_groups[cursor] = ROPE_GROUP_SELF
        cursor += 1

    for index, position in enumerate(opponent_positions):
        x, y = int(position[0]), int(position[1])
        feature_rows[cursor] = unit_token(
            "OPPONENT",
            "FARMER" if index == 0 else "HAND",
            index,
            x,
            y,
            None,
        )
        coordinates[cursor] = (x, y)
        spatial[cursor] = True
        rope_groups[cursor] = ROPE_GROUP_OPPONENT
        cursor += 1

    demand = town_demand(observation)
    for item in ITEMS:
        feature_rows[cursor] = product_token(observation, item, demand)
        cursor += 1

    memory_features_row, memory_features = memory_token(opponent_inventory, encoded_opponent_memory)
    feature_rows[cursor] = memory_features_row
    cursor += 1

    market_slot_indices = np.arange(cursor, cursor + MARKET_SLOTS, dtype=np.int32)
    for slot in range(MARKET_SLOTS):
        feature_rows[cursor, FEATURE_INDEX["token:MARKET_SLOT"]] = 1.0
        feature_rows[cursor, FEATURE_INDEX["farm:NONE"]] = 1.0
        feature_rows[cursor, FEATURE_INDEX[f"market_slot:{slot}"]] = 1.0
        cursor += 1
    if cursor != token_count:
        raise RuntimeError(f"encoded {cursor} tokens, expected {token_count}")

    return EncodedObservation(
        features=feature_rows,
        memory_features=memory_features,
        coordinates=coordinates,
        spatial_mask=spatial,
        rope_groups=rope_groups,
        own_unit_indices=own_unit_indices,
        market_slot_indices=market_slot_indices,
    )

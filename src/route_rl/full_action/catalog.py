# Adapted from msdsm/kaggriculture-solution, commit 84057a0fda4238ccdebc46f9bf5496c6c4b2e00d.
# Source: actions.catalog; see docs/action_bc_sources.md.
"""Source-free policy vocabularies and rule constants for experiment 41."""

from __future__ import annotations

BOARD_SIZE = 10
TURNS_PER_DAY = 24
EPISODE_STEPS = 720
SHED_CAPACITY = 100
MARKET_I0 = 10_000
PICKUP_QUANTITY_MAX = 20
PLACE_QUANTITY_MAX = 20
BUY_SEED_QUANTITY_MAX = 100
BUY_PRODUCT_QUANTITY_MAX = 100
BUY_ANIMAL_QUANTITY_MAX = 100
SELL_FRACTION_DENOMINATOR = 8
MAX_QUANTITY = "MAX"
MARKET_SLOTS = 10

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
PRODUCTS = (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER")
ANIMALS = ("GOOSE", "COW", "SHEEP")
ITEMS = (*PRODUCTS, *ANIMALS)

SHOPS = (
    "BAKERY",
    "PIZZA_SHOP",
    "BRUNCH_SPOT",
    "YARN_STORE",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "SMOOTHIE_SHOP",
    "FARMERS_MARKET",
)

SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}

TOKEN_TYPES = ("GLOBAL", "CELL", "UNIT", "PRODUCT", "MEMORY", "MARKET_SLOT")
FARM_ROLES = ("NONE", "SELF", "OPPONENT")
TILE_KINDS = ("EMPTY", "LOCKED", "WEED", "PLANT", "COOP", "PASTURE")
UNIT_TYPES = ("FARMER", "HAND")
QUADRANTS = ("NW", "NE", "SW", "SE")

UNIT_OPS = (
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PASS",
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
)

MARKET_OPS = (
    "NOOP",
    "BUY_SEED",
    "BUY_PRODUCT",
    "BUY_ANIMAL",
    "SELL",
    "HIRE",
    "BUY_LAND",
)

UNIT_QUANTITY_OPS = frozenset({"PICKUP", "PLACE"})
MARKET_ITEM_OPS = frozenset({"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"})
MARKET_QUANTITY_OPS = MARKET_ITEM_OPS

UNIT_QUANTITIES_BY_OP = {
    "PICKUP": tuple(range(1, PICKUP_QUANTITY_MAX + 1)),
    "PLACE": tuple(range(1, PLACE_QUANTITY_MAX + 1)),
}

MARKET_ITEMS_BY_OP = {
    "BUY_SEED": CROPS,
    "BUY_PRODUCT": ("WHEAT", "FERTILIZER"),
    "BUY_ANIMAL": ANIMALS,
    "SELL": PRODUCTS,
}
MARKET_QUANTITIES_BY_OP = {
    "BUY_SEED": tuple(range(1, BUY_SEED_QUANTITY_MAX + 1)),
    "BUY_PRODUCT": tuple(range(1, BUY_PRODUCT_QUANTITY_MAX + 1)),
    "BUY_ANIMAL": tuple(range(1, BUY_ANIMAL_QUANTITY_MAX + 1)),
    "SELL": (*range(1, SELL_FRACTION_DENOMINATOR), MAX_QUANTITY),
}


def build_unit_actions() -> tuple[tuple[str | int, ...], ...]:
    actions: list[tuple[str | int, ...]] = []
    for operation in UNIT_OPS:
        if operation == "PLANT":
            actions.extend((operation, item) for item in CROPS)
        elif operation in UNIT_QUANTITY_OPS:
            actions.extend(
                (operation, item, quantity) for item in ITEMS for quantity in UNIT_QUANTITIES_BY_OP[operation]
            )
        else:
            actions.append((operation,))
    return tuple(actions)


def build_market_actions() -> tuple[tuple[str | int, ...], ...]:
    actions: list[tuple[str | int, ...]] = []
    for operation in MARKET_OPS:
        if operation in MARKET_ITEM_OPS:
            actions.extend(
                (operation, item, quantity)
                for item in MARKET_ITEMS_BY_OP[operation]
                for quantity in MARKET_QUANTITIES_BY_OP[operation]
            )
        else:
            actions.append((operation,))
    return tuple(actions)


UNIT_ACTIONS = build_unit_actions()
MARKET_ACTIONS = build_market_actions()
UNIT_ACTION_TO_ID = {action: index for index, action in enumerate(UNIT_ACTIONS)}
MARKET_ACTION_TO_ID = {action: index for index, action in enumerate(MARKET_ACTIONS)}

if len(UNIT_ACTIONS) != 500 or len(MARKET_ACTIONS) != 1_075:
    raise RuntimeError("unexpected flat action-catalog size")

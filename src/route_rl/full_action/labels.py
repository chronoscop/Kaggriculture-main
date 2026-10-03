# Adapted from msdsm/kaggriculture-solution, commit 84057a0fda4238ccdebc46f9bf5496c6c4b2e00d.
# Source: data/prepare_replays.py:label_actions; see docs/action_bc_sources.md.
"""Resolver-filtered full unit and absolute market labels."""
from typing import Any
import numpy as np
from .catalog import MARKET_ACTION_TO_ID, UNIT_ACTION_TO_ID, UNIT_QUANTITY_OPS
from .sell_quantity import ABSOLUTE_START, NON_SELL_IDS, PRODUCTS, QUANTITY_COUNT

UNITS, SLOTS, IGNORE = 20, 10, -100
NON_SELL_MAP = {old: new for new, old in enumerate(NON_SELL_IDS)}

def label_actions(observation: dict, action: dict, legal: Any) -> np.ndarray:
    count = 1 + len(observation["farms"][observation["player"]].get("hands", []))
    hands = action.get("hands", [])
    units = [action.get("farmer", ["PASS"])] + [
        hands[index] if index < len(hands) else ["PASS"] for index in range(count - 1)
    ]
    labels = np.full(UNITS + SLOTS, IGNORE, np.int32)
    for index, order in enumerate(units[:UNITS]):
        if not legal.unit_mask[index]:
            continue
        order = order or ["PASS"]
        operation = order[0]
        try:
            key = (
                (operation, order[1], int(order[2]) if len(order) > 2 else 1)
                if operation in UNIT_QUANTITY_OPS
                else (operation, order[1])
                if operation == "PLANT"
                else (operation,)
            )
            labels[index] = UNIT_ACTION_TO_ID.get(key, IGNORE)
        except (IndexError, TypeError, ValueError):
            pass
    market = action.get("market", [])
    for index in range(SLOTS):
        order = market[index] if index < len(market) else ["NOOP"]
        order = order or ["NOOP"]
        if not legal.market_mask[index]:
            continue
        operation = order[0]
        try:
            if operation == "SELL":
                executed = int(legal.market_executed[index])
                if 1 <= executed <= QUANTITY_COUNT:
                    labels[UNITS + index] = ABSOLUTE_START + PRODUCTS.index(order[1]) * QUANTITY_COUNT + executed - 1
            elif operation in {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"}:
                key = (operation, order[1], int(order[2]) if len(order) > 2 else 1)
                labels[UNITS + index] = NON_SELL_MAP.get(MARKET_ACTION_TO_ID.get(key, IGNORE), IGNORE)
            else:
                labels[UNITS + index] = NON_SELL_MAP.get(MARKET_ACTION_TO_ID.get((operation,), IGNORE), IGNORE)
        except (IndexError, TypeError, ValueError):
            pass
    return labels



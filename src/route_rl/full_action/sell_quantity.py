# Adapted from msdsm/kaggriculture-solution, commit 84057a0fda4238ccdebc46f9bf5496c6c4b2e00d.
# Source: actions.sell_quantity; see docs/action_bc_sources.md.
"""Absolute SELL conditionals with unchanged legacy operation/product marginals."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from route_rl.full_action.quantities import decode_sell_quantity
from route_rl.full_action.catalog import MARKET_ACTIONS, PRODUCTS

QUANTITY_COUNT = 100
FRACTION_COUNT = 8
PRODUCT_COUNT = len(PRODUCTS)
SELL_START = 1001
SELL_END = SELL_START + len(PRODUCTS) * FRACTION_COUNT
NON_SELL_IDS = (*range(SELL_START), *range(SELL_END, len(MARKET_ACTIONS)))
ABSOLUTE_START = len(NON_SELL_IDS)
ABSOLUTE_ACTION_COUNT = ABSOLUTE_START + len(PRODUCTS) * QUANTITY_COUNT
ENGINE_ABSOLUTE_START = 10_000
ENGINE_IDS = (*NON_SELL_IDS, *range(ENGINE_ABSOLUTE_START, ENGINE_ABSOLUTE_START + len(PRODUCTS) * QUANTITY_COUNT))


def compose_market_logits(legacy: jax.Array, quantity: jax.Array) -> jax.Array:
    if legacy.shape[-1] != len(MARKET_ACTIONS) or quantity.shape != (*legacy.shape[:-1], len(PRODUCTS), QUANTITY_COUNT):
        raise ValueError("SELL head requires legacy logits and nine 100-way conditionals")
    fractions = (
        legacy[..., SELL_START:SELL_END]
        .astype(jnp.float32)
        .reshape((*legacy.shape[:-1], len(PRODUCTS), FRACTION_COUNT))
    )
    item_logits = jax.nn.logsumexp(fractions, axis=-1, keepdims=True)
    sell = item_logits + jax.nn.log_softmax(quantity.astype(jnp.float32), axis=-1)
    return jnp.concatenate(
        (legacy[..., :SELL_START], legacy[..., SELL_END:], sell.reshape((*legacy.shape[:-1], -1))), axis=-1
    )


def engine_market_ids(actions: Any, *, absolute: bool, xp: Any = np) -> Any:
    if not absolute:
        return actions
    return xp.asarray(ENGINE_IDS, dtype=xp.int32)[actions]


def decode_engine_market_order(action: int, shed: dict[str, int]) -> list[Any] | None:
    if ENGINE_ABSOLUTE_START <= action < ENGINE_ABSOLUTE_START + len(PRODUCTS) * QUANTITY_COUNT:
        item_id, offset = divmod(action - ENGINE_ABSOLUTE_START, QUANTITY_COUNT)
        item = PRODUCTS[item_id]
        quantity = min(offset + 1, max(0, int(shed.get(item, 0))))
        return ["SELL", item, quantity] if quantity else None
    if not 0 <= action < len(MARKET_ACTIONS):
        raise ValueError("unknown engine market action id")
    encoded = MARKET_ACTIONS[action]
    if encoded[0] == "NOOP":
        return None
    if encoded[0] != "SELL":
        return list(encoded)
    quantity = decode_sell_quantity(encoded[2], int(shed.get(str(encoded[1]), 0)))
    return [encoded[0], encoded[1], quantity] if quantity else None


def teacher_quantity_targets(fractions: jax.Array, stock: jax.Array) -> jax.Array:
    """Push the conditional fraction distribution through the exact integer decoder."""
    stock = stock.astype(jnp.int32)
    numerators = jnp.arange(1, FRACTION_COUNT + 1, dtype=jnp.int32)
    quantities = jnp.maximum(1, (stock[..., None] * numerators + FRACTION_COUNT // 2) // FRACTION_COUNT)
    quantities = jnp.minimum(stock[..., None], quantities)
    bins = jax.nn.one_hot(quantities - 1, QUANTITY_COUNT, dtype=jnp.float32)
    return jnp.sum(fractions[..., None].astype(jnp.float32) * bins, axis=-2)

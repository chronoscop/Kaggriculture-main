# Adapted from msdsm/kaggriculture-solution, commit 84057a0fda4238ccdebc46f9bf5496c6c4b2e00d.
# Source: actions.quantities; see docs/action_bc_sources.md.
"""Shared state-aware quantity semantics for replay labels and inference."""

from __future__ import annotations

from typing import Any

from route_rl.full_action.catalog import (
    ANIMALS,
    BOARD_SIZE,
    MAX_QUANTITY,
    SELL_FRACTION_DENOMINATOR,
    SHED_CAPACITY,
)


SHED_ACCESS = {
    (BOARD_SIZE // 2 - 1, BOARD_SIZE // 2 - 1),
    (BOARD_SIZE // 2, BOARD_SIZE // 2 - 1),
    (BOARD_SIZE // 2 - 1, BOARD_SIZE // 2),
    (BOARD_SIZE // 2, BOARD_SIZE // 2),
}
ANIMAL_STRUCTURE = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}


def sell_fraction_class(quantity: int, sellable_quantity: int) -> int | str:
    """Map a SELL request to the nearest eighth; MAX means all sellable stock."""
    if quantity <= 0:
        raise ValueError("sell quantity must be positive")
    if sellable_quantity <= 0 or quantity >= sellable_quantity:
        return MAX_QUANTITY
    numerator = (2 * SELL_FRACTION_DENOMINATOR * quantity + sellable_quantity) // (2 * sellable_quantity)
    return max(1, min(SELL_FRACTION_DENOMINATOR - 1, numerator))


def decode_sell_quantity(encoded_quantity: int | str, sellable_quantity: int) -> int:
    if sellable_quantity <= 0:
        return 0
    if encoded_quantity == MAX_QUANTITY:
        return sellable_quantity
    numerator = int(encoded_quantity)
    decoded = (sellable_quantity * numerator + SELL_FRACTION_DENOMINATOR // 2) // SELL_FRACTION_DENOMINATOR
    return max(1, min(sellable_quantity, decoded))


def shed_after_unit_actions(observation: dict[str, Any], unit_actions: list[list[Any]]) -> dict[str, int]:
    """Apply deterministic own-unit shed transfers in the engine's pre-Market order."""
    private = observation.get("private", {})
    shed = {str(item): int(quantity) for item, quantity in private.get("shed", {}).items()}
    player = int(observation.get("player", 0))
    farms = observation.get("farms", [])
    if not 0 <= player < len(farms):
        return shed

    farm = farms[player]
    positions = [farm.get("farmer", [-1, -1]), *farm.get("hands", [])]
    inventories = [dict(inventory) for inventory in private.get("inventories", [])]
    shed_total = sum(shed.values())
    for index, order in enumerate(unit_actions[: len(positions)]):
        if not isinstance(order, list) or not order:
            continue
        position = tuple(int(value) for value in positions[index])
        inventory = inventories[index] if index < len(inventories) else {}
        operation = str(order[0])

        if operation == "DROP" and position in SHED_ACCESS:
            for item, held in list(inventory.items()):
                room = max(0, SHED_CAPACITY - shed_total)
                moved = min(int(held), room)
                shed[item] = shed.get(item, 0) + moved
                shed_total += moved
            inventory.clear()
            continue

        if operation == "PICKUP" and position in SHED_ACCESS and len(order) >= 2:
            item = str(order[1])
            requested = int(order[2]) if len(order) >= 3 else 1
            moved = min(max(0, requested), shed.get(item, 0))
            shed[item] = shed.get(item, 0) - moved
            shed_total -= moved
            inventory[item] = int(inventory.get(item, 0)) + moved
            continue

        if operation != "PLACE" or position not in SHED_ACCESS or len(order) < 2:
            continue
        item = str(order[1])
        x, y = position
        tile = farm.get("tiles", [])[y][x]
        animal_pen = (
            item in ANIMALS
            and isinstance(tile, dict)
            and tile.get("kind") == ANIMAL_STRUCTURE[item]
            and "animal" not in tile
        )
        if animal_pen:
            inventory[item] = max(0, int(inventory.get(item, 0)) - 1)
            continue
        requested = int(order[2]) if len(order) >= 3 else 1
        room = max(0, SHED_CAPACITY - shed_total)
        moved = min(max(0, requested), int(inventory.get(item, 0)), room)
        shed[item] = shed.get(item, 0) + moved
        shed_total += moved
        inventory[item] = int(inventory.get(item, 0)) - moved
    return shed

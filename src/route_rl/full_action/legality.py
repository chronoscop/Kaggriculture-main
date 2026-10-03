# Adapted from msdsm/kaggriculture-solution, commit 84057a0fda4238ccdebc46f9bf5496c6c4b2e00d.
# Source: actions.legality; see docs/action_bc_sources.md.
"""Resolver-exact action-effect masks for training labels and policy diagnostics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from kaggle_environments import __version__ as KAGGLE_ENVIRONMENTS_VERSION
from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from route_rl.full_action.catalog import MARKET_SLOTS

EXPECTED_ENGINE_VERSION = "1.32.7"
BOARD_SIZE = 10
TURNS_PER_DAY = 24
SHED_CAPACITY = 100
ENVIRONMENT = SimpleNamespace(
    configuration={
        "boardSize": BOARD_SIZE,
        "maxMarketOrdersPerTurn": MARKET_SLOTS,
        "shedCapacity": SHED_CAPACITY,
    }
)


@dataclass(frozen=True)
class ActionLegality:
    unit_mask: tuple[bool, ...]
    market_mask: tuple[bool, ...]
    unit_operations: tuple[str, ...]
    market_operations: tuple[str | None, ...]
    market_requested: tuple[int, ...]
    market_executed: tuple[int, ...]


def require_expected_engine() -> None:
    if KAGGLE_ENVIRONMENTS_VERSION != EXPECTED_ENGINE_VERSION:
        raise RuntimeError(
            f"legality masks require kaggle-environments=={EXPECTED_ENGINE_VERSION}, "
            f"found {KAGGLE_ENVIRONMENTS_VERSION}"
        )


def normalized_action_dict(action: Any) -> dict[str, Any]:
    return action if isinstance(action, dict) else {}


def normalized_unit_action(action: Any) -> list[Any]:
    return action if isinstance(action, list) and action else ["PASS"]


def ordered_unit_actions(action: Any, unit_count: int) -> list[list[Any]]:
    normalized = normalized_action_dict(action)
    hands = normalized.get("hands", [])
    if not isinstance(hands, list):
        hands = []
    return [
        normalized_unit_action(normalized.get("farmer", ["PASS"])),
        *[normalized_unit_action(hands[index]) if index < len(hands) else ["PASS"] for index in range(unit_count - 1)],
    ]


def market_orders(action: Any) -> list[list[Any]]:
    orders = normalized_action_dict(action).get("market", [])
    if not isinstance(orders, list):
        return []
    return [order if isinstance(order, list) else [] for order in orders[:MARKET_SLOTS]]


def operation(action: list[Any], noop: str) -> str:
    return str(action[0]) if action else noop


def apply_units(
    farms: list[dict[str, Any]],
    privates: list[dict[str, Any]],
    actions: list[Any],
    day: int,
    selected_player: int,
) -> tuple[tuple[bool, ...], tuple[str, ...]]:
    selected_mask: list[bool] = []
    selected_operations: list[str] = []
    for player in range(2):
        farm = farms[player]
        private = privates[player]
        unit_count = 1 + len(farm.get("hands", []))
        units = ordered_unit_actions(actions[player], unit_count)
        plant_demand: dict[str, int] = {}
        for unit_action in units:
            if len(unit_action) >= 2 and unit_action[0] == "PLANT":
                crop = str(unit_action[1])
                plant_demand[crop] = plant_demand.get(crop, 0) + 1
        seeds = private.get("seeds", {})
        blocked_crops = {crop for crop, count in plant_demand.items() if count > seeds.get(crop, 0)}

        for index, original_action in enumerate(units):
            original_operation = operation(original_action, "PASS")
            blocked = (
                len(original_action) >= 2 and original_operation == "PLANT" and str(original_action[1]) in blocked_crops
            )
            resolved_action = ["PASS"] if blocked else original_action
            before_farm = deepcopy(farm)
            before_private = deepcopy(private)
            engine._apply_unit_action(
                farm,
                private,
                index,
                resolved_action,
                BOARD_SIZE,
                day,
                TURNS_PER_DAY,
                SHED_CAPACITY,
            )
            if player != selected_player:
                continue
            selected_operations.append(original_operation)
            selected_mask.append(original_operation == "PASS" or farm != before_farm or private != before_private)
    return tuple(selected_mask), tuple(selected_operations)


def runtime_state(
    farms: list[dict[str, Any]],
    market: dict[str, Any],
    town: dict[str, Any],
    privates: list[dict[str, Any]],
) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            observation=SimpleNamespace(
                farms=farms,
                market=market,
                town=town,
                private=privates[player],
            ),
            action={},
        )
        for player in range(2)
    ]


def clone_runtime_state(state: list[SimpleNamespace]) -> list[SimpleNamespace]:
    observations = [row.observation for row in state]
    return runtime_state(
        deepcopy(observations[0].farms),
        deepcopy(observations[0].market),
        deepcopy(observations[0].town),
        [deepcopy(observation.private) for observation in observations],
    )


def state_snapshot(state: list[SimpleNamespace]) -> tuple[Any, ...]:
    public = state[0].observation
    return (
        deepcopy(public.farms),
        deepcopy(public.market),
        tuple(deepcopy(row.observation.private) for row in state),
    )


def set_market_slot_actions(
    state: list[SimpleNamespace],
    queues: list[list[list[Any]]],
    slot: int,
    omitted_player: int | None = None,
) -> None:
    for player in range(2):
        has_order = player != omitted_player and slot < len(queues[player])
        state[player].action = {"market": [deepcopy(queues[player][slot])]} if has_order else {"market": []}


def requested_quantity(order: list[Any]) -> int:
    if not order:
        return 0
    if order[0] in ("HIRE", "BUY_LAND"):
        return 1
    if len(order) < 3:
        return 0
    try:
        return max(0, int(order[2]))
    except (TypeError, ValueError):
        return 0


def executed_quantity(
    order: list[Any],
    farm_before: dict[str, Any],
    private_before: dict[str, Any],
    farm_after: dict[str, Any],
    private_after: dict[str, Any],
) -> int:
    if not order:
        return 0
    operation_name = str(order[0])
    if operation_name == "HIRE":
        return max(0, len(farm_after.get("hands", [])) - len(farm_before.get("hands", [])))
    if operation_name == "BUY_LAND":
        return max(
            0,
            len(farm_after.get("unlocked_quadrants", [])) - len(farm_before.get("unlocked_quadrants", [])),
        )
    if len(order) < 2:
        return 0
    item = str(order[1])
    if operation_name == "SELL":
        return max(
            0,
            int(private_before.get("shed", {}).get(item, 0)) - int(private_after.get("shed", {}).get(item, 0)),
        )
    if operation_name == "BUY_SEED":
        return max(
            0,
            int(private_after.get("seeds", {}).get(item, 0)) - int(private_before.get("seeds", {}).get(item, 0)),
        )
    if operation_name in ("BUY_PRODUCT", "BUY_ANIMAL"):
        return max(
            0,
            int(private_after.get("shed", {}).get(item, 0)) - int(private_before.get("shed", {}).get(item, 0)),
        )
    return 0


def apply_market(
    state: list[SimpleNamespace],
    actions: list[Any],
    selected_player: int,
) -> tuple[tuple[bool, ...], tuple[str | None, ...], tuple[int, ...], tuple[int, ...]]:
    queues = [market_orders(action) for action in actions]
    masks = [True] * MARKET_SLOTS
    operations: list[str | None] = [None] * MARKET_SLOTS
    requested = [0] * MARKET_SLOTS
    executed = [0] * MARKET_SLOTS

    for slot in range(max((len(queue) for queue in queues), default=0)):
        counterfactual = clone_runtime_state(state)
        before_farm = deepcopy(state[0].observation.farms[selected_player])
        before_private = deepcopy(state[selected_player].observation.private)

        set_market_slot_actions(state, queues, slot)
        set_market_slot_actions(counterfactual, queues, slot, omitted_player=selected_player)
        engine._process_market(state, ENVIRONMENT)
        engine._process_market(counterfactual, ENVIRONMENT)

        if slot >= len(queues[selected_player]):
            continue
        order = queues[selected_player][slot]
        operation_name = operation(order, "NOOP")
        operations[slot] = operation_name
        requested[slot] = requested_quantity(order)
        if operation_name == "NOOP":
            continue
        after_farm = state[0].observation.farms[selected_player]
        after_private = state[selected_player].observation.private
        executed[slot] = executed_quantity(order, before_farm, before_private, after_farm, after_private)
        masks[slot] = state_snapshot(state) != state_snapshot(counterfactual)

    return tuple(masks), tuple(operations), tuple(requested), tuple(executed)


def action_legality(
    observations: list[dict[str, Any]],
    actions: list[Any],
    selected_player: int,
) -> ActionLegality:
    """Evaluate one player's labels under the official simultaneous action resolver."""
    require_expected_engine()
    if len(observations) != 2 or len(actions) != 2:
        raise ValueError("Kaggriculture legality evaluation requires exactly two players")
    if selected_player not in (0, 1):
        raise ValueError(f"selected_player must be 0 or 1, got {selected_player}")

    farms = deepcopy(observations[0]["farms"])
    market = deepcopy(observations[0]["market"])
    town = deepcopy(observations[0]["town"])
    privates = [deepcopy(observation["private"]) for observation in observations]
    day = int(observations[0].get("day", 0))
    unit_mask, unit_operations = apply_units(farms, privates, actions, day, selected_player)
    state = runtime_state(farms, market, town, privates)
    market_mask, market_operations, requested, executed = apply_market(state, actions, selected_player)
    return ActionLegality(
        unit_mask=unit_mask,
        market_mask=market_mask,
        unit_operations=unit_operations,
        market_operations=market_operations,
        market_requested=requested,
        market_executed=executed,
    )

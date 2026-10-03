# Adapted from msdsm/kaggriculture-solution, commit 84057a0fda4238ccdebc46f9bf5496c6c4b2e00d.
# Source: observations.inventory_tracker; see docs/action_bc_sources.md.
"""Observation-only heuristic reconstruction of an opponent's private inventory."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from route_rl.full_action.tracker_constants import (
    ANIMAL_COSTS,
    ANIMAL_FIRST_YIELD_DAY,
    ANIMAL_INTERVAL,
    ANIMAL_MAX_HELD,
    ANIMAL_PRODUCTS,
    ANIMALS,
    CROP_INTERVAL,
    CROP_MAX_YIELD,
    CROP_ONGOING,
    CROPS,
    FIXED_PURCHASE_COSTS,
    ITEMS,
    LAND_PRICES,
    PRODUCTS,
    SHED_CAPACITY,
    SHOPS,
    TOWN_CENTER_PRODUCTS,
    hire_cost,
    market_price,
)

NumberMap = dict[str, float]


def zero_map(names: tuple[str, ...]) -> NumberMap:
    return {name: 0.0 for name in names}


def private_item_totals(private: dict[str, Any]) -> NumberMap:
    totals = {item: float(private.get("shed", {}).get(item, 0)) for item in ITEMS}
    for inventory in private.get("inventories", []):
        for item, quantity in inventory.items():
            if item in totals:
                totals[item] += float(quantity)
    return totals


def aggregate_inventories(private: dict[str, Any]) -> NumberMap:
    totals = zero_map(ITEMS)
    for inventory in private.get("inventories", []):
        for item, quantity in inventory.items():
            if item in totals:
                totals[item] += float(quantity)
    return totals


def farm_positions(farm: dict[str, Any]) -> list[tuple[int, int]]:
    raw_positions = [farm.get("farmer"), *farm.get("hands", [])]
    return [tuple(int(value) for value in position) for position in raw_positions if position is not None]


def is_shed_access(position: tuple[int, int], board_size: int = 10) -> bool:
    half = board_size // 2
    return position in {(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)}


def tile_kind(tile: Any) -> str | None:
    return str(tile.get("kind")) if isinstance(tile, dict) and tile.get("kind") is not None else None


def observation_step(observation: dict[str, Any]) -> int:
    """Recover step for seat-one replay observations where the framework omits it."""
    if "step" in observation:
        return int(observation["step"])
    return int(observation.get("day", 0)) * 24 + int(observation.get("hour", 0))


@dataclass
class PublicFlows:
    seeds: NumberMap = field(default_factory=lambda: zero_map(CROPS))
    items: NumberMap = field(default_factory=lambda: zero_map(ITEMS))
    produced: list[tuple[int | None, str, float]] = field(default_factory=list)
    consumed: list[tuple[int | None, str, float]] = field(default_factory=list)
    ambiguous_fertilizer: float = 0.0
    before_positions: list[tuple[int, int]] = field(default_factory=list)
    after_positions: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class FixedPurchaseSummary:
    mean: NumberMap
    lower: NumberMap
    upper: NumberMap
    explained_budget: int
    residual: float
    solution_count: int


@dataclass
class InventoryEstimate:
    seeds: NumberMap
    shed: NumberMap
    carried: NumberMap
    owned: NumberMap
    owned_uncertainty: NumberMap
    seed_uncertainty: NumberMap
    shed_uncertainty: NumberMap
    carried_uncertainty: NumberMap
    unresolved_cash: float
    floor_sale_ambiguity: float

    @property
    def total(self) -> NumberMap:
        return self.owned


def _stationary_units_by_position(
    before_positions: list[tuple[int, int]],
    after_positions: list[tuple[int, int]],
    end_of_day: bool,
) -> dict[tuple[int, int], int]:
    stationary: dict[tuple[int, int], int] = {}
    for index, position in enumerate(before_positions):
        if not end_of_day and (index >= len(after_positions) or after_positions[index] != position):
            continue
        # Multiple units may share one cell. The previous implementation returned
        # the first unit in farmer/hands order, so preserve that tie break.
        stationary.setdefault(position, index)
    return stationary


def _decay_due(tile: dict[str, Any], step: int) -> bool:
    max_lifespan_step = int(tile.get("max_lifespan_step", -1))
    return max_lifespan_step >= 0 and step >= max_lifespan_step and (step - max_lifespan_step) % 2 == 0


def _harvested_plant_units(before: dict[str, Any], after: Any, step: int, end_of_day: bool) -> float:
    before_yield = int(before.get("yield_units", 0))
    if before_yield <= 0:
        return 0.0
    crop = str(before.get("crop"))
    if after is None and not CROP_ONGOING.get(crop, False):
        return float(before_yield)
    if tile_kind(after) == "WEED":
        return float(before_yield) if before_yield > 1 or not _decay_due(before, step) else 0.0
    if not (isinstance(after, dict) and after.get("kind") == "PLANT" and after.get("crop") == crop):
        return 0.0

    after_yield = int(after.get("yield_units", 0))
    no_action_yield = before_yield - (1 if _decay_due(before, step) else 0)
    if not end_of_day:
        return float(before_yield) if after_yield < max(0, no_action_yield) else 0.0
    if not CROP_ONGOING.get(crop, False):
        return 0.0

    next_day = step // 24 + 1
    days_since_first = (
        next_day
        - int(before.get("planted_day", 0))
        - {
            "TOMATO": 8,
            "STRAWBERRY": 10,
        }.get(crop, 0)
    )
    interval = CROP_INTERVAL[crop]
    scheduled = days_since_first >= 0 and days_since_first % interval == 0
    production = 0
    if scheduled:
        was_watered = bool(before.get("watered_today", False)) or int(after.get("consecutive_unwatered", 1)) == 0
        fertilized = int(before.get("fertilized_until_day", -1)) >= step // 24
        production = 2 if was_watered and fertilized else 1
    no_harvest = min(CROP_MAX_YIELD[crop], max(0, no_action_yield) + production)
    after_harvest = min(CROP_MAX_YIELD[crop], production)
    if after_yield == after_harvest and no_harvest != after_harvest:
        return float(before_yield)
    return 0.0


def _harvested_animal_units(before: dict[str, Any], after: Any, step: int, end_of_day: bool) -> float:
    before_yield = int(before.get("yield_units", 0))
    animal = str(before.get("animal", ""))
    if before_yield <= 0 or animal not in ANIMALS:
        return 0.0
    if not (isinstance(after, dict) and after.get("animal") == animal):
        return 0.0
    after_yield = int(after.get("yield_units", 0))
    if not end_of_day:
        return float(before_yield) if after_yield < before_yield else 0.0

    next_day = step // 24 + 1
    days_since_first = next_day - int(before.get("placed_day", 0)) - ANIMAL_FIRST_YIELD_DAY[animal]
    scheduled = days_since_first >= 0 and days_since_first % ANIMAL_INTERVAL[animal] == 0
    production = 0
    if scheduled:
        was_fed = bool(before.get("fed_today", False)) or int(after.get("consecutive_unfed", 1)) == 0
        production = 1 + int(before.get("pending_care_bonus", 0)) if was_fed else 1
    no_harvest = min(ANIMAL_MAX_HELD[animal], before_yield + production)
    after_harvest = min(ANIMAL_MAX_HELD[animal], production)
    if after_yield == after_harvest and no_harvest != after_harvest:
        return float(before_yield)
    return 0.0


def infer_public_flows(
    before_observation: dict[str, Any], after_observation: dict[str, Any], player: int
) -> PublicFlows:
    """Infer inventory-changing field events without reading actions or private state."""
    before_farm = before_observation["farms"][player]
    after_farm = after_observation["farms"][player]
    before_positions = farm_positions(before_farm)
    after_positions = farm_positions(after_farm)
    flows = PublicFlows(before_positions=before_positions, after_positions=after_positions)
    before_tiles = before_farm["tiles"]
    after_tiles = after_farm["tiles"]
    step = observation_step(before_observation)
    end_of_day = int(after_observation.get("day", 0)) != int(before_observation.get("day", 0))
    stationary_units = _stationary_units_by_position(before_positions, after_positions, end_of_day)

    for y, row in enumerate(before_tiles):
        for x, before in enumerate(row):
            after = after_tiles[y][x]
            if (before is None and after is None) or (before == "LOCKED" and after == "LOCKED"):
                continue
            unit = stationary_units.get((x, y))

            if isinstance(after, dict) and after.get("kind") == "PLANT":
                new_plant = not (
                    isinstance(before, dict)
                    and before.get("kind") == "PLANT"
                    and before.get("crop") == after.get("crop")
                    and before.get("planted_day") == after.get("planted_day")
                )
                if new_plant:
                    crop = str(after["crop"])
                    flows.seeds[crop] -= 1.0

            if isinstance(before, dict) and before.get("kind") == "PLANT":
                crop = str(before["crop"])
                harvested = _harvested_plant_units(before, after, step, end_of_day)
                if harvested > 0:
                    flows.items[crop] += harvested
                    flows.produced.append((unit, crop, harvested))
                if (
                    isinstance(after, dict)
                    and after.get("kind") == "PLANT"
                    and int(after.get("fertilized_until_day", -1)) > int(before.get("fertilized_until_day", -1))
                ):
                    flows.items["FERTILIZER"] -= 1.0
                    flows.consumed.append((unit, "FERTILIZER", 1.0))

            before_animal = before.get("animal") if isinstance(before, dict) else None
            after_animal = after.get("animal") if isinstance(after, dict) else None
            if before_animal is None and after_animal in ANIMALS:
                animal = str(after_animal)
                flows.items[animal] -= 1.0
                flows.consumed.append((unit, animal, 1.0))

            if before_animal in ANIMALS:
                animal = str(before_animal)
                harvested = _harvested_animal_units(before, after, step, end_of_day)
                if harvested > 0:
                    product = ANIMAL_PRODUCTS[animal]
                    flows.items[product] += harvested
                    flows.produced.append((unit, product, harvested))
                same_animal = after_animal == before_animal
                if same_animal and not end_of_day:
                    if bool(before.get("fertilizer_available", False)) and not bool(
                        after.get("fertilizer_available", False)
                    ):
                        flows.items["FERTILIZER"] += 1.0
                        flows.produced.append((unit, "FERTILIZER", 1.0))
                    if not bool(before.get("fed_today", False)) and bool(after.get("fed_today", False)):
                        flows.items["WHEAT"] -= 1.0
                        flows.consumed.append((unit, "WHEAT", 1.0))
                elif same_animal and end_of_day:
                    if not bool(before.get("fed_today", False)) and int(after.get("consecutive_unfed", 1)) == 0:
                        flows.items["WHEAT"] -= 1.0
                        flows.consumed.append((unit, "WHEAT", 1.0))
                    if bool(before.get("fertilizer_available", False)) and unit is not None:
                        flows.ambiguous_fertilizer += 1.0

    return flows


def town_drain(observation: dict[str, Any]) -> dict[str, int]:
    """Return the deterministic town drain applied after this observation's action."""
    step = observation_step(observation)
    drained = {item: 0 for item in PRODUCTS}
    if step % 4 == 0:
        for shop in observation.get("town", {}).get("unlocked_shops", []):
            products = SHOPS.get(str(shop), ())
            multiplier = 2 if len(products) == 1 else 1
            for item in products:
                drained[item] += multiplier
    if step % 24 == 0:
        for item in TOWN_CENTER_PRODUCTS:
            drained[item] += 1
    return drained


def _requested_market_directions(action: dict[str, Any] | None) -> dict[str, set[str]]:
    directions: dict[str, set[str]] = defaultdict(set)
    if not isinstance(action, dict):
        return directions
    for order in action.get("market", [])[:10]:
        if not isinstance(order, list) or len(order) < 3:
            continue
        op, item = str(order[0]), str(order[1])
        if op in {"SELL", "BUY_PRODUCT"} and item in PRODUCTS:
            directions[item].add(op)
    return directions


def infer_own_market_effect(
    before_observation: dict[str, Any],
    after_observation: dict[str, Any],
    own_action: dict[str, Any] | None,
    player: int,
    field_flows: PublicFlows | None = None,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Infer observable own product trades from own private-state deltas."""
    before_total = private_item_totals(before_observation.get("private", {}))
    after_total = private_item_totals(after_observation.get("private", {}))
    if field_flows is None:
        field_flows = infer_public_flows(before_observation, after_observation, player)
    directions = _requested_market_directions(own_action)
    visible_effect = {item: 0 for item in PRODUCTS}
    buys = {item: 0 for item in PRODUCTS}
    sells = {item: 0 for item in PRODUCTS}

    requested_quantities: dict[tuple[str, str], int] = defaultdict(int)
    if isinstance(own_action, dict):
        for order in own_action.get("market", [])[:10]:
            if not isinstance(order, list) or len(order) < 3:
                continue
            try:
                quantity = max(0, int(order[2]))
            except (TypeError, ValueError):
                continue
            requested_quantities[(str(order[0]), str(order[1]))] += quantity

    for item in PRODUCTS:
        net_buy = after_total[item] - before_total[item] - field_flows.items[item]
        item_directions = directions.get(item, set())
        if item_directions == {"BUY_PRODUCT"}:
            buys[item] = min(requested_quantities[("BUY_PRODUCT", item)], max(0, round(net_buy)))
        elif item_directions == {"SELL"}:
            sells[item] = min(requested_quantities[("SELL", item)], max(0, round(-net_buy)))
        elif item_directions == {"BUY_PRODUCT", "SELL"}:
            buy_request = requested_quantities[("BUY_PRODUCT", item)]
            sell_request = requested_quantities[("SELL", item)]
            if net_buy >= 0:
                buys[item] = min(buy_request, round(net_buy))
            else:
                sells[item] = min(sell_request, round(-net_buy))

        before_price = int(before_observation.get("market", {}).get("prices", {}).get(item, 0))
        after_price = int(after_observation.get("market", {}).get("prices", {}).get(item, 0))
        visible_sells = sells[item] if min(before_price, after_price) > 1 else 0
        visible_effect[item] = visible_sells - buys[item]
    return visible_effect, buys, sells


def fixed_purchase_distribution(budget: float, animal_room: int = SHED_CAPACITY) -> FixedPurchaseSummary:
    """Uniform moments over fixed-price purchase bundles nearest to the inferred budget."""
    rounded_budget = max(0, round(budget))
    explained_budget = int(round(rounded_budget / 10) * 10)
    cached = _fixed_purchase_distribution(explained_budget, max(0, animal_room))
    return FixedPurchaseSummary(
        # These cached maps are treated as immutable by the sole runtime caller.
        mean=cached.mean,
        lower=cached.lower,
        upper=cached.upper,
        explained_budget=explained_budget,
        residual=float(budget - explained_budget),
        solution_count=cached.solution_count,
    )


@lru_cache(maxsize=4_096)
def _fixed_purchase_distribution(explained_budget: int, animal_room: int) -> FixedPurchaseSummary:
    """Cacheable integer-budget implementation of the fixed-price bundle DP."""
    names = tuple(FIXED_PURCHASE_COSTS)
    scaled_budget = explained_budget // 10
    scaled_costs = {item: FIXED_PURCHASE_COSTS[item] // 10 for item in names}
    min_animal_cost = min(ANIMAL_COSTS.values()) // 10
    max_animals = min(animal_room, scaled_budget // min_animal_cost)
    ways: dict[tuple[int, int], int] = {(0, 0): 1}
    sums: dict[tuple[int, int], list[int]] = {(0, 0): [0] * len(names)}
    lower: dict[tuple[int, int], list[int]] = {(0, 0): [0] * len(names)}
    upper: dict[tuple[int, int], list[int]] = {(0, 0): [0] * len(names)}

    for item_index, item in enumerate(names):
        cost = scaled_costs[item]
        animal_increment = 1 if item in ANIMALS else 0
        next_ways = dict(ways)
        next_sums = {key: list(value) for key, value in sums.items()}
        next_lower = {key: list(value) for key, value in lower.items()}
        next_upper = {key: list(value) for key, value in upper.items()}
        for spent in range(cost, scaled_budget + 1):
            for animals in range(animal_increment, max_animals + 1):
                previous_key = (spent - cost, animals - animal_increment)
                if previous_key not in next_ways:
                    continue
                key = (spent, animals)
                count = next_ways[previous_key]
                candidate_sums = list(next_sums[previous_key])
                candidate_sums[item_index] += count
                candidate_lower = list(next_lower[previous_key])
                candidate_upper = list(next_upper[previous_key])
                candidate_lower[item_index] += 1
                candidate_upper[item_index] += 1
                if key not in next_ways:
                    next_ways[key] = count
                    next_sums[key] = candidate_sums
                    next_lower[key] = candidate_lower
                    next_upper[key] = candidate_upper
                    continue
                next_ways[key] += count
                for index in range(len(names)):
                    next_sums[key][index] += candidate_sums[index]
                    next_lower[key][index] = min(next_lower[key][index], candidate_lower[index])
                    next_upper[key][index] = max(next_upper[key][index], candidate_upper[index])
        ways, sums, lower, upper = next_ways, next_sums, next_lower, next_upper

    matching = [key for key in ways if key[0] == scaled_budget]
    solution_count = sum(ways[key] for key in matching)
    mean_result = zero_map(names)
    lower_result = zero_map(names)
    upper_result = zero_map(names)
    if solution_count > 0:
        for index, item in enumerate(names):
            total_quantity = sum(sums[key][index] for key in matching)
            mean_result[item] = total_quantity / solution_count
            lower_result[item] = float(min(lower[key][index] for key in matching))
            upper_result[item] = float(max(upper[key][index] for key in matching))
    return FixedPurchaseSummary(
        mean=mean_result,
        lower=lower_result,
        upper=upper_result,
        explained_budget=explained_budget,
        residual=0.0,
        solution_count=solution_count,
    )


class OpponentInventoryTracker:
    """Maintain one heuristic belief using only information visible to `observer_player`."""

    def __init__(self, observer_player: int) -> None:
        if observer_player not in {0, 1}:
            raise ValueError("observer_player must be 0 or 1")
        self.observer_player = observer_player
        self.opponent_player = 1 - observer_player
        self.seeds = zero_map(CROPS)
        self.item_total = zero_map(ITEMS)
        self.shed = zero_map(ITEMS)
        self.carried_by_unit: list[NumberMap] = [zero_map(ITEMS)]
        self.seed_uncertainty = zero_map(CROPS)
        self.item_uncertainty = zero_map(ITEMS)
        self.shed_uncertainty = zero_map(ITEMS)
        self.unresolved_cash = 0.0
        self.floor_sale_ambiguity = 0.0

    def estimate(self, *, copy_maps: bool = True) -> InventoryEstimate:
        carried = zero_map(ITEMS)
        carried_uncertainty = zero_map(ITEMS)
        for inventory in self.carried_by_unit:
            for item in ITEMS:
                carried[item] += inventory[item]
        for item in ITEMS:
            mismatch = abs(self.item_total[item] - self.shed[item] - carried[item])
            carried_uncertainty[item] = self.item_uncertainty[item] + mismatch
        return InventoryEstimate(
            seeds=deepcopy(self.seeds) if copy_maps else self.seeds,
            shed=deepcopy(self.shed) if copy_maps else self.shed,
            carried=carried,
            owned=deepcopy(self.item_total) if copy_maps else self.item_total,
            owned_uncertainty=deepcopy(self.item_uncertainty) if copy_maps else self.item_uncertainty,
            seed_uncertainty=deepcopy(self.seed_uncertainty) if copy_maps else self.seed_uncertainty,
            shed_uncertainty=deepcopy(self.shed_uncertainty) if copy_maps else self.shed_uncertainty,
            carried_uncertainty=carried_uncertainty,
            unresolved_cash=self.unresolved_cash,
            floor_sale_ambiguity=self.floor_sale_ambiguity,
        )

    def update(
        self,
        before_observation: dict[str, Any],
        after_observation: dict[str, Any],
        own_action: dict[str, Any] | None,
        public_flows: dict[int, PublicFlows] | None = None,
    ) -> InventoryEstimate:
        self.advance(before_observation, after_observation, own_action, public_flows)
        return self.estimate()

    def advance(
        self,
        before_observation: dict[str, Any],
        after_observation: dict[str, Any],
        own_action: dict[str, Any] | None,
        public_flows: dict[int, PublicFlows] | None = None,
    ) -> None:
        if observation_step(before_observation) == 0 and observation_step(after_observation) <= 0:
            return
        if public_flows is None:
            opponent_flows = infer_public_flows(before_observation, after_observation, self.opponent_player)
            observer_flows = None
        else:
            opponent_flows = public_flows[self.opponent_player]
            observer_flows = public_flows[self.observer_player]
        end_of_day = int(after_observation.get("day", 0)) != int(before_observation.get("day", 0))
        before_positions = opponent_flows.before_positions
        after_positions = opponent_flows.after_positions
        self._resize_units(len(before_positions))
        self._apply_public_flows(opponent_flows)
        self._infer_shed_transfers(before_positions, after_positions, end_of_day)
        self._apply_market(
            before_observation,
            after_observation,
            own_action,
            end_of_day,
            observer_flows,
            before_positions,
            after_positions,
        )
        if end_of_day:
            self._drop_all_at_end_of_day()
        else:
            self._resize_units(len(after_positions))
        self._reconcile_total()

    def _resize_units(self, count: int) -> None:
        while len(self.carried_by_unit) < count:
            self.carried_by_unit.append(zero_map(ITEMS))
        if len(self.carried_by_unit) > count:
            self.carried_by_unit = self.carried_by_unit[:count]

    def _apply_public_flows(self, flows: PublicFlows) -> None:
        for crop in CROPS:
            self.seeds[crop] += flows.seeds[crop]
            if self.seeds[crop] < 0:
                self.seed_uncertainty[crop] += -self.seeds[crop]
                self.seeds[crop] = 0.0
        for item in ITEMS:
            self.item_total[item] += flows.items[item]
        for unit, item, quantity in flows.produced:
            if unit is None or unit >= len(self.carried_by_unit):
                self.shed_uncertainty[item] += quantity
                continue
            self.carried_by_unit[unit][item] += quantity
        for unit, item, quantity in flows.consumed:
            self._consume_carried(unit, item, quantity)
        if flows.ambiguous_fertilizer > 0:
            estimate = 0.5 * flows.ambiguous_fertilizer
            self.item_total["FERTILIZER"] += estimate
            self.item_uncertainty["FERTILIZER"] += estimate
            self.shed_uncertainty["FERTILIZER"] += estimate

    def _consume_carried(self, unit: int | None, item: str, quantity: float) -> None:
        remaining = quantity
        if unit is not None and unit < len(self.carried_by_unit):
            take = min(remaining, self.carried_by_unit[unit][item])
            self.carried_by_unit[unit][item] -= take
            remaining -= take
        if remaining > 0:
            take = min(remaining, self.shed[item])
            self.shed[item] -= take
            remaining -= take
        if remaining > 0:
            self.item_uncertainty[item] += remaining
        self.item_total[item] = max(0.0, self.item_total[item])

    def _infer_shed_transfers(
        self,
        before_positions: list[tuple[int, int]],
        after_positions: list[tuple[int, int]],
        end_of_day: bool,
    ) -> None:
        if end_of_day:
            return
        for index, before_position in enumerate(before_positions):
            if index >= len(after_positions) or before_position != after_positions[index]:
                continue
            if not is_shed_access(before_position):
                continue
            inventory = self.carried_by_unit[index]
            for item in ITEMS:
                self.shed_uncertainty[item] += min(1.0, self.shed[item]) + inventory[item]

    def _apply_market(
        self,
        before_observation: dict[str, Any],
        after_observation: dict[str, Any],
        own_action: dict[str, Any] | None,
        end_of_day: bool,
        observer_flows: PublicFlows | None,
        before_opponent_positions: list[tuple[int, int]],
        after_opponent_positions: list[tuple[int, int]],
    ) -> None:
        own_effect, _, _ = infer_own_market_effect(
            before_observation,
            after_observation,
            own_action,
            self.observer_player,
            observer_flows,
        )
        drained = town_drain(before_observation)
        opponent_buys = {item: 0 for item in PRODUCTS}
        opponent_sells = {item: 0 for item in PRODUCTS}
        sale_revenue = 0.0
        product_buy_cost = 0.0
        before_opponent_farm = before_observation["farms"][self.opponent_player]
        after_opponent_farm = after_observation["farms"][self.opponent_player]
        stationary_shed_units = self._stationary_shed_units(
            before_opponent_positions,
            after_opponent_positions,
            end_of_day,
        )

        for item in PRODUCTS:
            before_inventory = int(before_observation.get("market", {}).get("inventory", {}).get(item, 0))
            after_inventory = int(after_observation.get("market", {}).get("inventory", {}).get(item, 0))
            joint_effect = after_inventory + drained[item] - before_inventory
            opponent_effect = joint_effect - own_effect[item]
            if item in {"WHEAT", "FERTILIZER"} and opponent_effect < 0:
                opponent_buys[item] = -opponent_effect
            elif opponent_effect > 0:
                opponent_sells[item] = opponent_effect

            if opponent_buys[item] > 0:
                quantity = opponent_buys[item]
                product_buy_cost += sum(market_price(item, before_inventory - offset - 1) for offset in range(quantity))
                self.item_total[item] += quantity
                self.shed[item] += quantity
            if opponent_sells[item] > 0:
                quantity = opponent_sells[item]
                sale_revenue += sum(market_price(item, before_inventory + offset) for offset in range(quantity))
                self.item_total[item] = max(0.0, self.item_total[item] - quantity)
                removed = min(self.shed[item], quantity)
                self.shed[item] -= removed
                remaining = quantity - removed
                for unit in stationary_shed_units:
                    if remaining <= 0:
                        break
                    dropped_and_sold = min(remaining, self.carried_by_unit[unit][item])
                    self.carried_by_unit[unit][item] -= dropped_and_sold
                    remaining -= dropped_and_sold
                if remaining > 0:
                    self.shed_uncertainty[item] += remaining

        fixed_cost = self._public_fixed_cost(before_opponent_farm, after_opponent_farm, end_of_day)
        money_delta = float(after_opponent_farm.get("money", 0)) - float(before_opponent_farm.get("money", 0))
        purchase_budget = sale_revenue - product_buy_cost - fixed_cost - money_delta
        animal_room = max(0, SHED_CAPACITY - round(sum(self.shed.values())))
        summary = fixed_purchase_distribution(purchase_budget, animal_room)
        ambiguous_budget = summary.explained_budget if summary.solution_count > 1 else 0
        self.unresolved_cash = abs(summary.residual) + ambiguous_budget

        for crop in CROPS:
            quantity = summary.lower[crop]
            self.seeds[crop] += quantity
            self.seed_uncertainty[crop] += 0.5 * (summary.upper[crop] - summary.lower[crop])
        for animal in ANIMALS:
            quantity = summary.lower[animal]
            self.item_total[animal] += quantity
            self.shed[animal] += quantity
            self.item_uncertainty[animal] += 0.5 * (summary.upper[animal] - summary.lower[animal])
            self.shed_uncertainty[animal] += 0.5 * (summary.upper[animal] - summary.lower[animal])

        before_prices = before_observation.get("market", {}).get("prices", {})
        after_prices = after_observation.get("market", {}).get("prices", {})
        floor_items = [item for item in PRODUCTS if min(before_prices.get(item, 0), after_prices.get(item, 0)) <= 1]
        unexplained_gain = max(0.0, money_delta + fixed_cost + product_buy_cost - sale_revenue)
        guaranteed_floor_sales = min(round(unexplained_gain), round(sum(self.item_total[item] for item in floor_items)))
        self._remove_floor_sales(floor_items, guaranteed_floor_sales, stationary_shed_units)
        remaining_floor_stock = sum(self.item_total[item] for item in floor_items)
        self.floor_sale_ambiguity = min(remaining_floor_stock, max(0.0, -purchase_budget))

    def _remove_floor_sales(self, floor_items: list[str], quantity: int, stationary_shed_units: list[int]) -> None:
        remaining = float(quantity)
        ordered_items = sorted(floor_items, key=lambda item: (self.shed[item], self.item_total[item]), reverse=True)
        for item in ordered_items:
            if remaining <= 0:
                break
            sold = min(remaining, self.shed[item])
            self.shed[item] -= sold
            self.item_total[item] = max(0.0, self.item_total[item] - sold)
            remaining -= sold
            for unit in stationary_shed_units:
                if remaining <= 0:
                    break
                sold = min(remaining, self.carried_by_unit[unit][item])
                self.carried_by_unit[unit][item] -= sold
                self.item_total[item] = max(0.0, self.item_total[item] - sold)
                remaining -= sold
            if remaining > 0:
                sold = min(remaining, self.item_total[item])
                self.item_total[item] -= sold
                self.item_uncertainty[item] += sold
                remaining -= sold

    def _stationary_shed_units(
        self,
        before_positions: list[tuple[int, int]],
        after_positions: list[tuple[int, int]],
        end_of_day: bool,
    ) -> list[int]:
        if end_of_day:
            return []
        return [
            index
            for index, position in enumerate(before_positions)
            if index < len(after_positions) and position == after_positions[index] and is_shed_access(position)
        ]

    @staticmethod
    def _public_fixed_cost(before_farm: dict[str, Any], after_farm: dict[str, Any], end_of_day: bool) -> int:
        before_land = len(before_farm.get("unlocked_quadrants", []))
        after_land = len(after_farm.get("unlocked_quadrants", []))
        extra_before = max(0, before_land - 1)
        land_count = max(0, after_land - before_land)
        land_cost = sum(LAND_PRICES[extra_before : extra_before + land_count])
        if end_of_day:
            return land_cost
        hires_before = int(before_farm.get("hires_today", 0))
        hires_after = int(after_farm.get("hires_today", 0))
        return land_cost + hire_cost(hires_before, max(0, hires_after - hires_before))

    def _drop_all_at_end_of_day(self) -> None:
        for inventory in self.carried_by_unit:
            for item in ITEMS:
                room = max(0.0, SHED_CAPACITY - sum(self.shed.values()))
                quantity = inventory[item]
                moved = min(quantity, room)
                discarded = quantity - moved
                self.shed[item] += moved
                inventory[item] = 0.0
                if discarded > 0:
                    self.item_total[item] = max(0.0, self.item_total[item] - discarded)
                    self.item_uncertainty[item] += discarded
        self.carried_by_unit = [zero_map(ITEMS)]
        if sum(self.item_total.values()) <= SHED_CAPACITY:
            self.shed = deepcopy(self.item_total)

    def _reconcile_total(self) -> None:
        carried = zero_map(ITEMS)
        for inventory in self.carried_by_unit:
            for item in ITEMS:
                carried[item] += inventory[item]
        for item in ITEMS:
            represented = self.shed[item] + carried[item]
            if represented < self.item_total[item]:
                self.shed_uncertainty[item] += self.item_total[item] - represented
            elif represented > self.item_total[item]:
                excess = represented - self.item_total[item]
                removed = min(excess, self.shed[item])
                self.shed[item] -= removed
                excess -= removed
                for inventory in self.carried_by_unit:
                    if excess <= 0:
                        break
                    removed = min(excess, inventory[item])
                    inventory[item] -= removed
                    excess -= removed
                self.item_uncertainty[item] += represented - self.item_total[item]

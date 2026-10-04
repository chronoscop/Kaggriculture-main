"""Official-rule prefix sampling with stored conditional behavior support.

The fixed-order support and probability recording follow the reference
actions/sequential.py and actions/masks.py. This implementation owns its Python
resolver path and does not import the reference checkout or its Rust extension.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from ..full_action.catalog import (BOARD_SIZE, MARKET_SLOTS, PRODUCTS, SHED_CAPACITY,
                                   TURNS_PER_DAY, UNIT_ACTIONS, UNIT_ACTION_TO_ID)
from ..full_action.inference import MAX_OWN_UNITS, MAX_TOTAL_UNITS
from ..full_action.legality import require_expected_engine
from ..full_action.sell_quantity import (ABSOLUTE_ACTION_COUNT, ENGINE_ABSOLUTE_START,
                                         ENGINE_IDS, QUANTITY_COUNT)

ACTION_SELECTION_CONTRACT = "official-prefix-full-action-ppo-v1"
ECONOMIC_ACTION_SELECTION_CONTRACT = "official-prefix-economic-full-action-ppo-v2"
SUPPORTED_ACTION_SELECTION_CONTRACTS = (ACTION_SELECTION_CONTRACT, ECONOMIC_ACTION_SELECTION_CONTRACT)
EXECUTION_CONTRACT = ACTION_SELECTION_CONTRACT
MASKED_LOGIT = -1e9
PASS_ID = UNIT_ACTION_TO_ID[("PASS",)]


def _market_catalog() -> tuple[tuple[Any, ...], ...]:
    from ..full_action.catalog import MARKET_ACTIONS

    result = []
    for identity in ENGINE_IDS:
        if identity >= ENGINE_ABSOLUTE_START:
            item, quantity = divmod(identity - ENGINE_ABSOLUTE_START, QUANTITY_COUNT)
            result.append(("SELL", PRODUCTS[item], quantity + 1))
        else:
            result.append(MARKET_ACTIONS[identity])
    return tuple(result)


MARKET_ACTIONS = _market_catalog()
NOOP_ID = MARKET_ACTIONS.index(("NOOP",))
MARKET_ACTION_TO_ID = {action: identity for identity, action in enumerate(MARKET_ACTIONS)}
UNIT_STEMS = tuple(dict.fromkeys(action[:2] for action in UNIT_ACTIONS))
UNIT_STEM_INDEX = np.asarray([UNIT_STEMS.index(action[:2]) for action in UNIT_ACTIONS], np.int32)
UNIT_REPRESENTATIVES = tuple(UNIT_ACTIONS[int(np.flatnonzero(UNIT_STEM_INDEX == i)[0])]
                             for i in range(len(UNIT_STEMS)))
MARKET_STEMS = tuple(dict.fromkeys(action[:2] for action in MARKET_ACTIONS))
MARKET_STEM_INDEX = np.asarray([MARKET_STEMS.index(action[:2]) for action in MARKET_ACTIONS], np.int32)
MARKET_REPRESENTATIVES = tuple(MARKET_ACTIONS[int(np.flatnonzero(MARKET_STEM_INDEX == i)[0])]
                               for i in range(len(MARKET_STEMS)))


@dataclass(frozen=True)
class SampledAction:
    action: dict[str, Any]
    unit_action: np.ndarray
    market_action: np.ndarray
    unit_mask: np.ndarray
    market_mask: np.ndarray
    unit_legal_mask: np.ndarray
    market_legal_mask: np.ndarray
    old_unit_log_prob: np.ndarray
    old_market_log_prob: np.ndarray
    old_log_prob: np.float32
    unit_policy_mask: np.ndarray
    market_policy_mask: np.ndarray

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in (
            "unit_action", "market_action", "unit_mask", "market_mask",
            "unit_legal_mask", "market_legal_mask", "old_unit_log_prob",
            "old_market_log_prob", "old_log_prob", "unit_policy_mask", "market_policy_mask")}


def validate_action_selection_contract(name: str) -> str:
    if name not in SUPPORTED_ACTION_SELECTION_CONTRACTS:
        raise ValueError(f"unsupported action selection contract: {name}")
    return name


def validate_observation_capacity(observation: dict[str, Any]) -> int:
    player = int(observation["player"])
    farms = observation["farms"]
    if player not in (0, 1) or len(farms) != 2:
        raise ValueError("PPO requires two farms and player 0 or 1")
    counts = [1 + len(farm.get("hands", [])) for farm in farms]
    if counts[player] > MAX_OWN_UNITS or sum(counts) > MAX_TOTAL_UNITS:
        raise ValueError("PPO observation exceeds fixed unit capacity; truncation is forbidden")
    return counts[player]


class PrefixResolver:
    """Own-action prefix simulation using only the observer's private inventory.

    Market support means an effective request against the visible market with
    the other seat submitting NOOP. Simultaneous opposing orders can still alter
    prices and fills in the real game. The chosen request remains unchanged.
    HIRE support is capped at the model's 20 own-unit representation limit.
    """

    def __init__(self, observation: dict[str, Any]) -> None:
        require_expected_engine()
        self.unit_count = validate_observation_capacity(observation)
        self.player = int(observation["player"])
        self.farms = deepcopy(observation["farms"])
        self.market = deepcopy(observation["market"])
        self.town = deepcopy(observation["town"])
        self.private = deepcopy(observation["private"])
        self.day = int(observation.get("day", 0))

    @property
    def farm(self) -> dict[str, Any]:
        return self.farms[self.player]

    def _apply_unit(self, farm: dict, private: dict, index: int, action: tuple | list) -> None:
        from kaggle_environments.envs.kaggriculture import kaggriculture as rules

        rules._apply_unit_action(farm, private, index, list(action), BOARD_SIZE,
                                 self.day, TURNS_PER_DAY, SHED_CAPACITY)

    def unit_support(self, index: int) -> np.ndarray:
        if not 0 <= index < self.unit_count:
            mask = np.zeros(len(UNIT_ACTIONS), np.bool_)
            mask[PASS_ID] = True
            return mask
        stems = np.zeros(len(UNIT_STEMS), np.bool_)
        for stem_index, action in enumerate(UNIT_REPRESENTATIVES):
            if action[0] == "PASS":
                stems[stem_index] = True
                continue
            farm, private = deepcopy((self.farm, self.private))
            self._apply_unit(farm, private, index, action)
            stems[stem_index] = farm != self.farm or private != self.private
        # All positive PICKUP/PLACE requests have the same effect/no-effect
        # support: the official engine performs quantity limiting at execution.
        return stems[UNIT_STEM_INDEX]

    def apply_unit(self, index: int, identity: int) -> None:
        self._apply_unit(self.farm, self.private, index, UNIT_ACTIONS[identity])

    def _market_state(self, farms: list, market: dict, private: dict) -> list:
        # The opponent's hidden private state is never consulted or copied.
        privates = [{"shed": {}, "seeds": {}, "inventories": []} for _ in range(2)]
        privates[self.player] = private
        return [SimpleNamespace(observation=SimpleNamespace(
            farms=farms, market=market, town=self.town, private=privates[player]),
            action={"market": []}) for player in range(2)]

    def _apply_market(self, farms: list, market: dict, private: dict, action: tuple | list) -> None:
        from kaggle_environments.envs.kaggriculture import kaggriculture as rules

        state = self._market_state(farms, market, private)
        state[self.player].action = {"market": [list(action)]}
        environment = SimpleNamespace(configuration={"boardSize": BOARD_SIZE,
            "maxMarketOrdersPerTurn": MARKET_SLOTS, "shedCapacity": SHED_CAPACITY})
        rules._process_market(state, environment)

    def market_support(self) -> np.ndarray:
        stems = np.zeros(len(MARKET_STEMS), np.bool_)
        for index, action in enumerate(MARKET_REPRESENTATIVES):
            if action[0] == "NOOP":
                stems[index] = True
                continue
            if action[0] == "HIRE" and 1 + len(self.farm["hands"]) >= MAX_OWN_UNITS:
                continue
            farms, market, private = deepcopy((self.farms, self.market, self.private))
            self._apply_market(farms, market, private, action)
            # Price refresh alone does not make an unaffordable request legal.
            # Every effective catalog operation changes own farm or inventory.
            stems[index] = farms[self.player] != self.farm or private != self.private
        mask = stems[MARKET_STEM_INDEX]
        # Restrict absolute SELL quantities before sampling. Decoding is then
        # bijective, with no clamping or removed SELL requests after sampling.
        for index, action in enumerate(MARKET_ACTIONS):
            if action[0] == "SELL":
                mask[index] &= int(action[2]) <= self.private["shed"].get(action[1], 0)
        return mask

    def apply_market(self, identity: int) -> None:
        if identity == NOOP_ID:
            return
        self._apply_market(self.farms, self.market, self.private, MARKET_ACTIONS[identity])


def sample_categorical(logits: np.ndarray, mask: np.ndarray, rng: np.random.Generator | None) -> tuple[int, np.float32]:
    """Greedy when rng is None; otherwise sample the exact masked categorical."""
    logits = np.asarray(logits, np.float32)
    mask = np.asarray(mask, np.bool_)
    if logits.shape != mask.shape or not mask.any() or not np.isfinite(logits[mask]).all():
        raise ValueError("categorical logits require nonempty finite legal support")
    masked = np.where(mask, logits, -np.inf).astype(np.float64)
    shifted = masked - masked.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    if rng is None:
        identity = int(np.argmax(masked))
    else:
        # choice uses the normalized float64 mass; stored logprob differs from
        # accelerator FP32 log_softmax only by ordinary rounding precision.
        identity = int(rng.choice(len(logits), p=probabilities))
    if not mask[identity]:
        raise RuntimeError("categorical selected a masked action")
    log_probability = shifted[identity] - np.log(np.exp(shifted).sum())
    return identity, np.float32(log_probability)


def sample_action(observation: dict[str, Any], outputs: dict[str, Any],
                  rng: np.random.Generator | None, *,
                  action_selection_contract: str = ACTION_SELECTION_CONTRACT,
                  resolver: Any = None) -> SampledAction:
    """Select and execute one fixed-order conditional action without rewriting."""
    validate_action_selection_contract(action_selection_contract)
    economic = action_selection_contract == ECONOMIC_ACTION_SELECTION_CONTRACT
    if economic:
        from .economic_rules import (FINAL_SALE_STEPS, forced_drop, forced_sales, market_economic_support,
                                     observation_step, unit_economic_support)
        step = observation_step(observation)
    resolver = PrefixResolver(observation) if resolver is None else resolver
    unit_logits = np.asarray(outputs["unit_action"])
    market_logits = np.asarray(outputs["market_action"])
    if unit_logits.shape == (1, MAX_OWN_UNITS, len(UNIT_ACTIONS)):
        unit_logits = unit_logits[0]
    if market_logits.shape == (1, MARKET_SLOTS, ABSOLUTE_ACTION_COUNT):
        market_logits = market_logits[0]
    if unit_logits.shape != (MAX_OWN_UNITS, len(UNIT_ACTIONS)) or market_logits.shape != (MARKET_SLOTS, ABSOLUTE_ACTION_COUNT):
        raise ValueError("PPO requires fixed 20-unit and absolute-SELL market logits")
    unit_action = np.full(MAX_OWN_UNITS, PASS_ID, np.int32)
    market_action = np.full(MARKET_SLOTS, NOOP_ID, np.int32)
    unit_mask = np.arange(MAX_OWN_UNITS) < resolver.unit_count
    market_mask = np.ones(MARKET_SLOTS, np.bool_)
    unit_policy_mask = unit_mask.copy()
    market_policy_mask = market_mask.copy()
    unit_legal_mask = np.zeros(unit_logits.shape, np.bool_)
    market_legal_mask = np.zeros(market_logits.shape, np.bool_)
    unit_log_prob = np.zeros(MAX_OWN_UNITS, np.float32)
    market_log_prob = np.zeros(MARKET_SLOTS, np.float32)
    for index in range(MAX_OWN_UNITS):
        unit_legal_mask[index] = resolver.unit_support(index)
        if economic:
            unit_legal_mask[index] = unit_economic_support(resolver, index, unit_legal_mask[index], UNIT_ACTIONS)
        if unit_mask[index]:
            if economic and forced_drop(resolver, index, step):
                identity, probability = UNIT_ACTION_TO_ID[("DROP",)], np.float32(0)
                # DROP may lose inventory when the shed is full. It is still a
                # deterministic controller action, with its exact singleton support.
                unit_legal_mask[index].fill(False)
                unit_legal_mask[index, identity] = True
                unit_policy_mask[index] = False
            else:
                identity, probability = sample_categorical(unit_logits[index], unit_legal_mask[index], rng)
            unit_action[index], unit_log_prob[index] = identity, probability
            resolver.apply_unit(index, identity)
    sales = forced_sales(resolver, step) if economic else []
    if len(sales) > MARKET_SLOTS:
        raise ValueError("forced sales exceed the market slot capacity")
    support = None
    for slot in range(MARKET_SLOTS):
        if slot < len(sales):
            identity, probability = MARKET_ACTION_TO_ID[sales[slot]], np.float32(0)
            if not resolver.market_support()[identity]:
                raise RuntimeError("forced sale is outside the actual prefix support")
            market_legal_mask[slot, identity] = True
            market_policy_mask[slot] = False
        else:
            if support is None:
                support = resolver.market_support()
                if economic:
                    support = market_economic_support(resolver, step, support, MARKET_ACTIONS)
            market_legal_mask[slot] = support
            identity, probability = sample_categorical(market_logits[slot], market_legal_mask[slot], rng)
            # The final liquidation controller owns every remaining market
            # slot, including deterministic NOOPs; no neural purchases survive.
            if economic and step in FINAL_SALE_STEPS:
                market_policy_mask[slot] = False
        market_action[slot], market_log_prob[slot] = identity, probability
        resolver.apply_market(identity)
        if identity != NOOP_ID:
            support = None
    units = [list(UNIT_ACTIONS[int(identity)]) for identity in unit_action[:resolver.unit_count]]
    # Explicit NOOP occupies its sampled slot in the simultaneous resolver.
    market = [list(MARKET_ACTIONS[int(identity)]) for identity in market_action]
    action = {"farmer": units[0], "hands": units[1:], "market": market}
    joint = np.float32(unit_log_prob[unit_policy_mask].sum(dtype=np.float64)
                       + market_log_prob[market_policy_mask].sum(dtype=np.float64))
    return SampledAction(action, unit_action, market_action, unit_mask, market_mask,
                         unit_legal_mask, market_legal_mask, unit_log_prob, market_log_prob, joint,
                         unit_policy_mask, market_policy_mask)


def external_action_sample(action: dict[str, Any], observation: dict[str, Any]) -> SampledAction:
    """Record a controller's executed catalog actions without inventing neural probabilities."""
    count = validate_observation_capacity(observation)
    units = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
    if len(units) > count or len(action.get("market", [])) > MARKET_SLOTS:
        raise ValueError("external controller action exceeds the actual unit or market capacity")
    units.extend([["PASS"] for _ in range(count - len(units))])
    markets = list(action.get("market", []))
    markets.extend([["NOOP"] for _ in range(MARKET_SLOTS - len(markets))])
    unit_ids = np.full(MAX_OWN_UNITS, PASS_ID, np.int32)
    market_ids = np.full(MARKET_SLOTS, NOOP_ID, np.int32)
    try:
        unit_ids[:count] = [UNIT_ACTION_TO_ID[tuple(order)] for order in units]
        market_ids[:] = [MARKET_ACTION_TO_ID[tuple(order)] for order in markets]
    except (KeyError, TypeError) as error:
        raise ValueError("external controller action is outside the full action vocabulary") from error
    unit_support = np.zeros((MAX_OWN_UNITS, len(UNIT_ACTIONS)), np.bool_)
    market_support = np.zeros((MARKET_SLOTS, ABSOLUTE_ACTION_COUNT), np.bool_)
    unit_support[np.arange(MAX_OWN_UNITS), unit_ids] = True
    market_support[np.arange(MARKET_SLOTS), market_ids] = True
    normalized = {"farmer": list(units[0]), "hands": [list(order) for order in units[1:]],
                  "market": [list(order) for order in markets]}
    return SampledAction(normalized, unit_ids, market_ids,
                         np.arange(MAX_OWN_UNITS) < count, np.ones(MARKET_SLOTS, np.bool_),
                         unit_support, market_support, np.zeros(MAX_OWN_UNITS, np.float32),
                         np.zeros(MARKET_SLOTS, np.float32), np.float32(0),
                         np.zeros(MAX_OWN_UNITS, np.bool_), np.zeros(MARKET_SLOTS, np.bool_))

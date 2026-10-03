# Adapted from msdsm/kaggriculture-solution, commit 84057a0fda4238ccdebc46f9bf5496c6c4b2e00d.
# Source: agents/neural.py (fixed batch and decoders); see docs/action_bc_sources.md.
"""Candidate full-action inference using the project's BC encoder and model."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
from .catalog import BOARD_SIZE, ITEMS, MARKET_SLOTS, UNIT_ACTIONS
from .features import FEATURE_DIM, MEMORY_FEATURE_NAMES, EncodedObservation, encode_observation, observation_step
from .inventory_tracker import OpponentInventoryTracker
from .quantities import shed_after_unit_actions
from .sell_quantity import ABSOLUTE_ACTION_COUNT, decode_engine_market_order, engine_market_ids
from .model import JaxModelConfig, policy_forward
from .checkpoints import load_training_source

MAX_TOTAL_UNITS = 40
MAX_OWN_UNITS = 20
FIXED_TOKEN_COUNT = 264
CELL_REGION_END = 201

@dataclass(frozen=True)
class FixedBatch:
    arrays: dict[str, np.ndarray]
    actual_own_units: int
    encoded_own_units: int


def _numpy(tensor: Any) -> np.ndarray:
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def prepare_fixed_batch(encoded: EncodedObservation, observation: dict[str, Any]) -> FixedBatch:
    """Pad once and truncate Units only outside the observed-data safety envelope."""
    player = int(observation.get("player", 0))
    farms = observation.get("farms", [])
    if player not in (0, 1) or len(farms) != 2:
        raise ValueError("observation must contain two farms and player 0 or 1")

    actual_own_units = 1 + len(farms[player].get("hands", []))
    actual_opponent_units = 1 + len(farms[1 - player].get("hands", []))
    encoded_own_units = min(actual_own_units, MAX_OWN_UNITS)
    encoded_opponent_units = min(actual_opponent_units, MAX_TOTAL_UNITS - encoded_own_units)

    own_start = CELL_REGION_END
    opponent_start = own_start + actual_own_units
    tail_start = opponent_start + actual_opponent_units
    selected_indices = [
        *range(CELL_REGION_END),
        *range(own_start, own_start + encoded_own_units),
        *range(opponent_start, opponent_start + encoded_opponent_units),
        *range(tail_start, int(encoded.features.shape[0])),
    ]
    selected_count = len(selected_indices)
    if selected_count > FIXED_TOKEN_COUNT:
        raise RuntimeError(f"selected token count {selected_count} exceeds fixed capacity {FIXED_TOKEN_COUNT}")

    old_to_new = {old: new for new, old in enumerate(selected_indices)}
    features = np.zeros((1, FIXED_TOKEN_COUNT, FEATURE_DIM), dtype=np.float32)
    coordinates = np.zeros((1, FIXED_TOKEN_COUNT, 2), dtype=np.float32)
    spatial_mask = np.zeros((1, FIXED_TOKEN_COUNT), dtype=np.bool_)
    rope_groups = np.zeros((1, FIXED_TOKEN_COUNT), dtype=np.int32)
    token_mask = np.zeros((1, FIXED_TOKEN_COUNT), dtype=np.bool_)
    features[0, :selected_count] = _numpy(encoded.features)[selected_indices]
    coordinates[0, :selected_count] = _numpy(encoded.coordinates)[selected_indices]
    spatial_mask[0, :selected_count] = _numpy(encoded.spatial_mask)[selected_indices]
    rope_groups[0, :selected_count] = _numpy(encoded.rope_groups)[selected_indices]
    token_mask[0, :selected_count] = True

    unit_indices = np.zeros((1, MAX_OWN_UNITS), dtype=np.int32)
    old_unit_indices = _numpy(encoded.own_unit_indices).astype(np.int64, copy=False)
    for output_index, old_index in enumerate(old_unit_indices[:encoded_own_units]):
        unit_indices[0, output_index] = old_to_new[int(old_index)]
    market_indices = np.asarray(
        [[old_to_new[int(old_index)] for old_index in _numpy(encoded.market_slot_indices)]],
        dtype=np.int32,
    )
    return FixedBatch(
        arrays={
            "features": features,
            "memory_features": _numpy(encoded.memory_features).astype(np.float32, copy=False)[None],
            "coordinates": coordinates,
            "spatial_mask": spatial_mask,
            "rope_groups": rope_groups,
            "token_mask": token_mask,
            "unit_indices": unit_indices,
            "market_indices": market_indices,
        },
        actual_own_units=actual_own_units,
        encoded_own_units=encoded_own_units,
    )


def dummy_fixed_batch() -> dict[str, np.ndarray]:
    result = {
        "features": np.zeros((1, FIXED_TOKEN_COUNT, FEATURE_DIM), dtype=np.float32),
        "memory_features": np.zeros((1, len(MEMORY_FEATURE_NAMES)), dtype=np.float32),
        "coordinates": np.zeros((1, FIXED_TOKEN_COUNT, 2), dtype=np.float32),
        "spatial_mask": np.zeros((1, FIXED_TOKEN_COUNT), dtype=np.bool_),
        "rope_groups": np.zeros((1, FIXED_TOKEN_COUNT), dtype=np.int32),
        "token_mask": np.zeros((1, FIXED_TOKEN_COUNT), dtype=np.bool_),
        "unit_indices": np.zeros((1, MAX_OWN_UNITS), dtype=np.int32),
        "market_indices": np.zeros((1, MARKET_SLOTS), dtype=np.int32),
    }
    return result


def decode_unit_action(logits: np.ndarray) -> list[Any]:
    return list(UNIT_ACTIONS[int(np.argmax(logits))])


def decode_market_order(logits: np.ndarray, sellable_shed: dict[str, int]) -> list[Any] | None:
    selected = engine_market_ids(int(np.argmax(logits)), absolute=logits.shape[-1] == ABSOLUTE_ACTION_COUNT)
    return decode_engine_market_order(int(selected), sellable_shed)


class GreedyPolicy:
    def __init__(self, payload: dict, *, warm: bool = True) -> None:
        self.config = JaxModelConfig(**payload["model_config"])
        self.config.validate()
        self.params = jax.tree.map(jnp.asarray, payload["params"])
        self.tracker = None
        self.previous_observation = None
        self.previous_action = None
        self.forward = jax.jit(lambda batch: policy_forward(self.params, batch, self.config, dtype=jnp.float32))
        if warm:
            jax.block_until_ready(self.forward(jax.tree.map(jnp.asarray, dummy_fixed_batch())))

    def __call__(self, observation: dict) -> dict:
        player, step = int(observation["player"]), observation_step(observation)
        previous = observation_step(self.previous_observation) if self.previous_observation is not None else None
        if (self.tracker is None or self.tracker.observer_player != player or previous is None
                or step < previous or step > previous + 1):
            self.tracker = OpponentInventoryTracker(observer_player=player)
        elif step == previous + 1:
            self.tracker.update(self.previous_observation, observation, self.previous_action)
        fixed = prepare_fixed_batch(encode_observation(observation, self.tracker.estimate()), observation)
        outputs = jax.device_get(self.forward(jax.tree.map(jnp.asarray, fixed.arrays)))
        units = [decode_unit_action(outputs["unit_action"][0, index]) for index in range(fixed.encoded_own_units)]
        units.extend([["PASS"] for _ in range(fixed.actual_own_units - fixed.encoded_own_units)])
        sellable = shed_after_unit_actions(observation, units)
        market = []
        for logits in outputs["market_action"][0]:
            order = decode_market_order(logits, sellable)
            if order is None:
                continue
            market.append(order)
            if order[0] == "SELL":
                item = str(order[1])
                sellable[item] = max(0, sellable.get(item, 0) - int(order[2]))
        action = {"farmer": units[0], "hands": units[1:], "market": market}
        self.previous_observation, self.previous_action = deepcopy(observation), deepcopy(action)
        return action


def load_policy(path: str | Path, *, warm: bool = True) -> GreedyPolicy:
    return GreedyPolicy(load_training_source(Path(path)), warm=warm)

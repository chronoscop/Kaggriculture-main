"""Checkpoint-specific PPO inference using the collection action contract."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..full_action.checkpoints import load_training_source
from ..full_action.features import encode_observation, observation_step
from ..full_action.inference import GreedyPolicy, dummy_fixed_batch, prepare_fixed_batch
from ..full_action.inventory_tracker import OpponentInventoryTracker
from ..full_action.model import JaxModelConfig, policy_forward
from ..replay_rules import require_pinned_rules
from .provenance import OBJECTIVE_CONTRACT
from .sampling import (ACTION_SELECTION_CONTRACT, sample_action, validate_observation_capacity,
                       validate_action_selection_contract)


def compute_type(name: str) -> Any:
    if name not in ("float32", "float16", "bfloat16"):
        raise ValueError("compute_dtype must be float32, float16, or bfloat16")
    return getattr(jnp, name)


class PolicyHistory:
    """One independent observer tracker; neither seat can share hidden history."""

    def __init__(self, player: int) -> None:
        self.player = player
        self.tracker = OpponentInventoryTracker(observer_player=player)
        self.previous_observation = None
        self.previous_action = None

    def observe(self, observation: dict[str, Any]) -> None:
        """Advance public observer memory without duplicating native encoding."""
        player, step = int(observation["player"]), observation_step(observation)
        if player != self.player:
            raise ValueError("a PPO history belongs to exactly one player")
        previous = (observation_step(self.previous_observation)
                    if self.previous_observation is not None else None)
        if previous is not None:
            if step == previous + 1:
                self.tracker.update(self.previous_observation, observation, self.previous_action)
            elif step != previous:
                self.tracker = OpponentInventoryTracker(observer_player=player)
                self.previous_observation = self.previous_action = None
        validate_observation_capacity(observation)

    def encode(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        self.observe(observation)
        encoded = encode_observation(observation, self.tracker.estimate())
        fixed = prepare_fixed_batch(encoded, observation)
        if fixed.actual_own_units != fixed.encoded_own_units:
            raise ValueError("PPO cannot truncate unit actions")
        return fixed.arrays

    def remember(self, observation: dict, action: dict) -> None:
        self.previous_observation, self.previous_action = deepcopy(observation), deepcopy(action)


class PPOPolicy:
    def __init__(self, payload: dict, *, warm: bool = True) -> None:
        self.action_selection_contract = validate_action_selection_contract(payload.get("action_selection_contract"))
        if payload.get("learning_objective_contract") != OBJECTIVE_CONTRACT:
            raise ValueError("PPO checkpoint has an unsupported learning objective contract")
        require_pinned_rules()
        self.config = JaxModelConfig(**payload["model_config"])
        self.config.validate()
        self.params = jax.tree.map(jnp.asarray, payload["params"])
        self.history = None
        self.controller_config = payload.get('season_controller')
        self.controller = None
        if self.controller_config is not None:
            from .controllers import contract_identity
            if payload.get('controller_identity') != contract_identity(self.controller_config):
                raise ValueError('season controller source/binary/config changed; create a new candidate')
        self.compute_dtype = payload.get("compute_dtype", "float32")
        dtype = compute_type(self.compute_dtype)
        self.forward = jax.jit(lambda batch: policy_forward(self.params, batch, self.config,
                                                           dtype=dtype, training=False))
        if warm:
            jax.block_until_ready(self.forward(jax.tree.map(jnp.asarray, dummy_fixed_batch())))

    def __call__(self, observation: dict[str, Any]) -> dict[str, Any]:
        player = int(observation["player"])
        reset = (self.history is None or self.history.player != player
                 or observation_step(observation) == 0 and self.history.previous_observation is not None)
        if reset:
            self.history = PolicyHistory(player)
            if self.controller_config is not None:
                from .controllers import FinalDayController
                self.controller = FinalDayController(self.controller_config, observer_player=player)
        batch = self.history.encode(observation)
        if self.controller is not None:
            action = self.controller.choose_action(observation, history=self.history)
            if action is not None:
                self.history.remember(observation, action)
                return action
        outputs = jax.device_get(self.forward(jax.tree.map(jnp.asarray, batch)))
        selected = sample_action(observation, outputs, rng=None,
                                 action_selection_contract=self.action_selection_contract)
        self.history.remember(observation, selected.action)
        return selected.action


def load_policy(path: str | Path, *, warm: bool = True) -> PPOPolicy | GreedyPolicy:
    payload = load_training_source(Path(path))
    marker = payload.get("action_selection_contract")
    if marker is None:
        return GreedyPolicy(payload, warm=warm)
    validate_action_selection_contract(marker)
    return PPOPolicy(payload, warm=warm)

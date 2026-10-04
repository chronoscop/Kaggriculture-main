"""Frozen-policy self play in complete games, with explicit Python/Rust backends.

Borrowed concepts: reference training/rollout.py fixed batches, independent
observer histories, behavior log probabilities, and terminal GAE interface.
The owned native collector batches both seats of all games synchronously. Both
backends use complete seasons and terminal match-score rewards.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable
import time

import jax
import jax.numpy as jnp
import numpy as np

from ..full_action.catalog import EPISODE_STEPS
from ..full_action.features import observation_step
from ..full_action.model import JaxModelConfig, policy_forward
from ..replay_rules import require_pinned_rules
from .inference import PolicyHistory, compute_type
from .sampling import ACTION_SELECTION_CONTRACT, sample_action

TRANSITIONS = EPISODE_STEPS - 1


@dataclass(frozen=True)
class Rollout:
    arrays: dict[str, np.ndarray]
    values: np.ndarray
    terminal_scores: np.ndarray
    metadata: dict[str, Any]

    @property
    def transitions(self) -> int:
        return int(self.values.size)


def terminal_scores(rewards: list[float] | np.ndarray) -> np.ndarray:
    rewards = np.asarray(rewards, np.float64)
    if rewards.shape != (2,) or not np.isfinite(rewards).all():
        raise ValueError("complete two-player terminal rewards are required")
    first = 1.0 if rewards[0] > rewards[1] else 0.5 if rewards[0] == rewards[1] else 0.0
    return np.asarray([first, 1.0 - first], np.float32)


def _observation(environment: Any, player: int) -> dict[str, Any]:
    observation = deepcopy(dict(environment.state[player].observation))
    observation.setdefault("step", observation_step(observation))
    return observation


def _collect_official(params: dict, model_config: JaxModelConfig | dict, seeds: list[int],
                    sampling_seed: int, compute_dtype: str = "float32", *,
                    forward: Callable | None = None,
                    excluded_seeds: set[int] | None = None,
                    action_selection_contract: str = ACTION_SELECTION_CONTRACT,
                    season_controller: dict | None = None) -> Rollout:
    """Collect both seats on-policy, with every game ending at recorded step 719.

    The optional forward hook is for deterministic execution tests. Arrays store
    float32 encoder inputs exactly as consumed during collection; compute dtype
    is a separate contract and is used again by the updater. Held-out exclusions
    can be supplied in addition to the run-level reserved-range validation.
    """
    if not seeds or len(seeds) != len(set(seeds)) or any(not 0 < seed < 2**31 for seed in seeds):
        raise ValueError("rollout seeds must be distinct nonzero signed-32-bit seeds")
    if set(seeds) & set(excluded_seeds or ()):
        raise ValueError("rollout seeds overlap held-out or reserved seeds")
    if isinstance(model_config, dict):
        model_config = JaxModelConfig(**model_config)
    model_config.validate()
    if model_config.dropout != 0:
        raise ValueError("PPO behavior/update parity requires dropout=0")
    dtype = compute_type(compute_dtype)
    require_pinned_rules()
    from kaggle_environments import make

    rng = np.random.default_rng(sampling_seed)
    if forward is None:
        frozen = jax.tree.map(jnp.asarray, params)
        compiled = jax.jit(lambda batch: policy_forward(frozen, batch, model_config,
                                                        dtype=dtype, training=False))
        forward = lambda batch: jax.device_get(compiled(jax.tree.map(jnp.asarray, batch)))
    arrays: dict[str, np.ndarray] = {}
    values = np.empty((TRANSITIONS, len(seeds), 2), np.float32)
    scores = np.empty((len(seeds), 2), np.float32)
    records = []

    def store(step: int, game: int, player: int, row: dict[str, Any]) -> None:
        for name, value in row.items():
            value = np.asarray(value)
            if name not in arrays:
                arrays[name] = np.empty((TRANSITIONS, len(seeds), 2, *value.shape), value.dtype)
            arrays[name][step, game, player] = value

    for game, seed in enumerate(seeds):
        environment = make("kaggriculture", configuration={"seed": int(seed)}, debug=False)
        if environment.info.get("seed") != seed:
            raise ValueError("official environment did not retain the requested game seed")
        if environment.configuration.episodeSteps != EPISODE_STEPS:
            raise ValueError("rollout requires the official complete 720-recorded-step game")
        histories = [PolicyHistory(0), PolicyHistory(1)]
        controllers = _controllers(season_controller)
        for step in range(TRANSITIONS):
            if environment.done:
                raise ValueError("official game ended before the complete horizon")
            observations = [_observation(environment, player) for player in range(2)]
            if any(observation_step(observation) != step for observation in observations):
                raise ValueError("official rollout pre-action observation alignment mismatch")
            batches = [history.encode(observation) for history, observation in zip(histories, observations)]
            paired = {name: np.concatenate([batch[name] for batch in batches], axis=0) for name in batches[0]}
            outputs = forward(paired)
            if "value" not in outputs:
                raise ValueError("PPO rollout requires the initialized score value head")
            paired_values = np.asarray(jax.nn.sigmoid(jnp.asarray(outputs["value"])), np.float32)
            if paired_values.shape != (2,) or not np.isfinite(paired_values).all():
                raise ValueError("PPO critic must emit one finite score logit per seat")
            values[step, game] = paired_values
            actions = []
            for player in range(2):
                seat_outputs = {name: np.asarray(output)[player] for name, output in outputs.items()}
                selected = _select(observations[player], seat_outputs, rng,
                    action_selection_contract, histories[player], controllers[player])
                row = {name: batch_value[0] for name, batch_value in batches[player].items()}
                row.update(selected.arrays())
                row["done"] = np.asarray(step == TRANSITIONS - 1, np.bool_)
                store(step, game, player, row)
                histories[player].remember(observations[player], selected.action)
                actions.append(selected.action)
            environment.step(actions)
        statuses = [state.status for state in environment.state]
        if not environment.done or statuses != ["DONE", "DONE"] or len(environment.steps) != EPISODE_STEPS:
            raise ValueError("rollout did not finish a normal complete 720-step game")
        final = [_observation(environment, player) for player in range(2)]
        if any(observation_step(observation) != EPISODE_STEPS - 1 for observation in final):
            raise ValueError("terminal observation does not match the official complete horizon")
        rewards = [state.reward for state in environment.state]
        scores[game] = terminal_scores(rewards)
        record = {"seed": int(seed), "turns": len(environment.steps), "statuses": statuses,
                  "rewards": rewards, "terminal_scores": scores[game].tolist()}
        if season_controller is not None:
            record["controller_diagnostics"] = [deepcopy(controller.diagnostics) for controller in controllers]
        records.append(record)
    reward = np.zeros(values.shape, np.float32)
    reward[-1] = scores
    arrays["reward"] = reward
    return Rollout(arrays, values, scores, {
        "action_selection_contract": action_selection_contract,
        "seeds": list(map(int, seeds)), "sampling_seed": int(sampling_seed),
        "compute_dtype": compute_dtype, "storage_dtype": "float32",
        "transitions_per_seat": TRANSITIONS, "recorded_steps": EPISODE_STEPS,
        "market_support": "own-prefix-other-seat-noop-public-state",
        "unit_capacity": 20, "array_bytes": sum(array.nbytes for array in arrays.values()),
        "games": records, "collection": _backend_identity("official-python"),
        "controller": _controller_identity(season_controller),
        "controller_identity": _controller_identity(season_controller)})


def _backend_identity(name):
    from .native_backend import backend_identity
    return backend_identity(name)


def _controller_identity(config):
    if config is None:
        return None
    from .controllers import contract_identity
    return contract_identity(config)


def _controllers(config):
    if config is None:
        return [None, None]
    from .controllers import FinalDayController
    return [FinalDayController(config, observer_player=player) for player in range(2)]


def _select(observation, outputs, rng, contract, history, controller, resolver=None):
    if controller is not None:
        action = controller.choose_action(observation, history=history)
        if action is not None:
            from .sampling import external_action_sample
            return external_action_sample(action, observation)
    return sample_action(observation, outputs, rng,
                         action_selection_contract=contract, resolver=resolver)


def collect_rollout(params: dict, model_config: JaxModelConfig | dict, seeds: list[int],
                    sampling_seed: int, compute_dtype: str = "float32", *,
                    forward: Callable | None = None, excluded_seeds: set[int] | None = None,
                    collection_backend: str = "official-python",
                    action_selection_contract: str = ACTION_SELECTION_CONTRACT,
                    season_controller: dict | None = None) -> Rollout:
    """Complete-horizon collection with an explicit execution and backend identity.

    The legacy official path retains its exact sequential-game ordering. The
    native path advances all games synchronously, forwarding 2*games seats at
    once and recording the actual conditional supports and executed requests.
    """
    started = time.perf_counter()
    if collection_backend == "official-python":
        rollout = _collect_official(params, model_config, seeds, sampling_seed, compute_dtype,
            forward=forward, excluded_seeds=excluded_seeds,
            action_selection_contract=action_selection_contract, season_controller=season_controller)
        return _timed_rollout(rollout, time.perf_counter() - started)
    if collection_backend != "rust-batch":
        raise ValueError(f"unsupported collection backend: {collection_backend}")
    if not seeds or len(seeds) != len(set(seeds)) or any(not 0 < seed < 2**31 for seed in seeds):
        raise ValueError("rollout seeds must be distinct nonzero signed-32-bit seeds")
    if set(seeds) & set(excluded_seeds or ()):
        raise ValueError("rollout seeds overlap held-out or reserved seeds")
    if isinstance(model_config, dict):
        model_config = JaxModelConfig(**model_config)
    model_config.validate()
    if model_config.dropout != 0:
        raise ValueError("PPO behavior/update parity requires dropout=0")
    dtype = compute_type(compute_dtype)
    require_pinned_rules()
    from .native_backend import RustBatchBackend
    backend = RustBatchBackend(seeds)
    rng = np.random.default_rng(sampling_seed)
    if forward is None:
        frozen = jax.tree.map(jnp.asarray, params)
        compiled = jax.jit(lambda batch: policy_forward(frozen, batch, model_config,
                                                       dtype=dtype, training=False))
        forward = lambda batch: jax.device_get(compiled(jax.tree.map(jnp.asarray, batch)))
    arrays: dict[str, np.ndarray] = {}
    values = np.empty((TRANSITIONS, len(seeds), 2), np.float32)
    controllers = [_controllers(season_controller) for _ in seeds]
    histories = [[PolicyHistory(0), PolicyHistory(1)] for _ in seeds] if season_controller is not None else None
    controller_identity = _controller_identity(season_controller)
    for step in range(TRANSITIONS):
        observations = backend.observe()
        if any(observation_step(o) != step for game in observations for o in game):
            raise ValueError("Rust rollout pre-action observation alignment mismatch")
        batch = backend.encode()
        outputs = forward(batch)
        if "value" not in outputs:
            raise ValueError("PPO rollout requires the initialized score value head")
        paired_values = np.asarray(jax.nn.sigmoid(jnp.asarray(outputs["value"])), np.float32)
        if paired_values.shape != (len(seeds)*2,) or not np.isfinite(paired_values).all():
            raise ValueError("PPO critic must emit one finite score logit per seat")
        values[step] = paired_values.reshape(len(seeds), 2)
        selected_actions, external = [], False
        for game in range(len(seeds)):
            for player in range(2):
                index = game*2+player
                observation = observations[game][player]
                history = histories[game][player] if histories is not None else None
                if history is not None:
                    history.observe(observation)
                controller = controllers[game][player]
                seat_outputs = {name: np.asarray(output)[index] for name, output in outputs.items()}
                selected = _select(observation, seat_outputs, rng, action_selection_contract,
                    history, controller, resolver=backend.resolver(game, player))
                # Presence masks are independent of whether the action came from
                # the actor. Fully external requests must reach the engine intact.
                if controller is not None and not selected.unit_policy_mask.any() and not selected.market_policy_mask.any():
                    external = True
                selected_actions.append(selected)
                row = {name: array[index] for name, array in batch.items()}
                row.update(selected.arrays())
                row["done"] = np.asarray(step == TRANSITIONS-1, np.bool_)
                for name, value in row.items():
                    value = np.asarray(value)
                    if name not in arrays:
                        arrays[name] = np.empty((TRANSITIONS, len(seeds), 2, *value.shape), value.dtype)
                    arrays[name][step, game, player] = value
                if history is not None:
                    history.remember(observation, selected.action)
        backend.advance(selected_actions, external=external)
    final = backend.observe()
    if any(observation_step(o) != EPISODE_STEPS-1 for game in final for o in game):
        raise ValueError("Rust terminal observation does not match the complete horizon")
    rewards = backend.rewards()
    scores = np.stack([terminal_scores(reward) for reward in rewards])
    reward = np.zeros(values.shape, np.float32)
    reward[-1] = scores
    arrays["reward"] = reward
    records = [{"seed": int(seed), "turns": EPISODE_STEPS, "statuses": ["DONE", "DONE"],
                "rewards": list(rewards[game]), "terminal_scores": scores[game].tolist()}
               for game, seed in enumerate(seeds)]
    if season_controller is not None:
        for game, record in enumerate(records):
            record["controller_diagnostics"] = [deepcopy(controller.diagnostics) for controller in controllers[game]]
    result = Rollout(arrays, values, scores, {
        "action_selection_contract": action_selection_contract,
        "seeds": list(map(int, seeds)), "sampling_seed": int(sampling_seed),
        "compute_dtype": compute_dtype, "storage_dtype": "float32",
        "transitions_per_seat": TRANSITIONS, "recorded_steps": EPISODE_STEPS,
        "market_support": "own-prefix-other-seat-noop-public-state", "unit_capacity": 20,
        "array_bytes": sum(array.nbytes for array in arrays.values()), "games": records,
        "collection": backend.identity, "controller": controller_identity,
        "controller_identity": controller_identity})
    return _timed_rollout(result, time.perf_counter() - started)


def _timed_rollout(rollout: Rollout, seconds: float) -> Rollout:
    metadata = {**rollout.metadata, 'collection_seconds': seconds,
                'collection_env_steps': rollout.transitions,
                'collection_env_steps_per_second': rollout.transitions / max(seconds, 1e-9),
                'collection_timing_scope': 'complete collection including setup/JIT, support, inference, storage and optional search'}
    return Rollout(rollout.arrays, rollout.values, rollout.terminal_scores, metadata)


def flatten_rollout(rollout: Rollout, advantages: np.ndarray,
                    returns: np.ndarray) -> dict[str, np.ndarray]:
    """Flatten time/game/seat in that order, preserving adjacent opponent pairs."""
    shape = rollout.values.shape
    advantages, returns = np.asarray(advantages, np.float32), np.asarray(returns, np.float32)
    if advantages.shape != shape or returns.shape != shape:
        raise ValueError("GAE arrays must match rollout time/game/seat shape")
    samples = int(np.prod(shape))
    batch = {name: array.reshape((samples, *array.shape[3:])) for name, array in rollout.arrays.items()}
    batch.update({"old_value": rollout.values.reshape(samples), "value": rollout.values.reshape(samples),
                  "advantage": advantages.reshape(samples), "return": returns.reshape(samples),
                  "sample_mask": np.ones(samples, np.float32)})
    return batch

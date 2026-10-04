"""Bounded single-device PPO updates and strict, separate training checkpoints.

BC artifacts remain inputs. Resumption restores the optimizer, schedule counters,
frozen teacher, dtype and execution/objective identities together.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from ..full_action.checkpoints import POLICY_CONTRACT, atomic_pickle, policy_hash
from ..full_action.model import JaxModelConfig, add_zero_value_head, policy_forward
from .objective import (PPOConfig, make_optimizer, make_update_step, make_critic_update_step,
                        score_values, behavior_outputs, joint_log_prob)
from .provenance import OBJECTIVE_CONTRACT, PIPELINE_CONTRACT

STATE_CONTRACT = 'route-rl-full-action-ppo-training-state-v2'
STATE_CONTRACTS = (STATE_CONTRACT, 'route-rl-full-action-ppo-training-state-v1')
# Default retains the interpretation of accepted v1 policies.
EXECUTION_CONTRACT = 'official-prefix-full-action-ppo-v1'


def _host(tree):
    return jax.tree.map(lambda x: np.asarray(jax.device_get(x)), tree)


def _finite(tree):
    return all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(jax.device_get(tree)))


def _lineage_hash(identity, metadata):
    return hashlib.sha256(json.dumps({'identity': identity, 'metadata': metadata}, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def _load_checkpoint(path: Path) -> dict:
    # Checkpoints are trusted local training artifacts, not arbitrary untrusted pickles.
    with Path(path).open('rb') as stream:
        payload = pickle.load(stream)
    if payload.get('contract') not in STATE_CONTRACTS:
        raise ValueError('not an owned PPO training state; incompatible checkpoints cannot resume')
    from .sampling import validate_action_selection_contract
    validate_action_selection_contract(payload.get('execution'))
    if payload.get('objective') != OBJECTIVE_CONTRACT:
        raise ValueError('PPO objective/execution changed; use a new run directory')
    if payload.get('execution') != payload['identity'].get('execution', EXECUTION_CONTRACT):
        raise ValueError('PPO state execution differs from its run identity')
    if payload.get('season_controller') != payload['identity'].get('settings', {}).get('season_controller'):
        raise ValueError('PPO state continuation controller differs from its run identity')
    if payload.get('controller_identity') != payload['identity'].get('controller_identity'):
        raise ValueError('PPO state controller provenance mismatch')
    if policy_hash(payload['params']) != payload.get('policy_sha256'):
        raise ValueError('PPO state parameter checksum mismatch')
    if policy_hash(payload['teacher']) != payload.get('teacher_sha256'):
        raise ValueError('frozen BC teacher checksum mismatch')
    if policy_hash(payload['optimizer_state']) != payload.get('optimizer_sha256'):
        raise ValueError('PPO optimizer state checksum mismatch')
    if policy_hash(payload['critic_optimizer_state']) != payload.get('critic_optimizer_sha256'):
        raise ValueError('critic optimizer state checksum mismatch')
    if _lineage_hash(payload['identity'], payload['metadata']) != payload.get('lineage_sha256'):
        raise ValueError('PPO seed/lineage metadata checksum mismatch')
    return payload


def checkpoint_metadata(path: Path, expected_identity: dict | None = None) -> dict:
    """Read lineage, optionally bind it to the requesting run's integration."""
    saved = _load_checkpoint(path)
    if expected_identity is not None and saved['identity'] != expected_identity:
        raise ValueError('training state belongs to a different run integration')
    return dict(saved['metadata'])


class PPOTrainer:
    def __init__(self, model_config: JaxModelConfig, config: PPOConfig,
                 params: dict, teacher: dict, *, compute_dtype='float32',
                 action_selection_contract=EXECUTION_CONTRACT,
                 season_controller=None, controller_identity=None):
        from .sampling import validate_action_selection_contract
        self.action_selection_contract = validate_action_selection_contract(action_selection_contract)
        self.season_controller = season_controller
        self.controller_identity = controller_identity
        if jax.process_count() != 1 or len(jax.local_devices()) != 1:
            raise ValueError('PPOTrainer supports one process with one visible JAX device')
        self.model_config = model_config if isinstance(model_config, JaxModelConfig) else JaxModelConfig(**model_config)
        self.model_config.validate()
        if self.model_config.dropout:
            raise ValueError('PPO sampling/update parity requires dropout=0')
        if compute_dtype not in ('float32', 'bfloat16'):
            raise ValueError('compute_dtype must be float32 or bfloat16')
        self.config, self.compute_dtype = config, compute_dtype
        self.dtype = jnp.bfloat16 if compute_dtype == 'bfloat16' else jnp.float32
        # Zero-initialized critic adds no new policy behavior and predicts neutral score .5.
        self.params = jax.tree.map(jnp.asarray, add_zero_value_head(params, self.model_config))
        self.teacher = jax.tree.map(jnp.asarray, teacher)
        self.teacher_sha256 = policy_hash(self.teacher)
        self.optimizer = make_optimizer(config)
        self.critic_optimizer = make_optimizer(config, critic=True)
        self.optimizer_state = self.optimizer.init(self.params)
        self.critic_optimizer_state = self.critic_optimizer.init(self.params['value'])
        self.completed_updates = 0
        self.completed_critic_updates = 0
        self._update = make_update_step(self.model_config, config, self.dtype, self.optimizer)
        self._critic_update = make_critic_update_step(self.model_config, config, self.dtype, self.critic_optimizer)
        self._log_probs = jax.jit(lambda p, b: joint_log_prob(
            behavior_outputs(policy_forward(p, b, self.model_config, dtype=self.dtype, training=False), b),
            b['unit_action'], b['market_action'],
            b.get('unit_policy_mask', b['unit_mask']), b.get('market_policy_mask', b['market_mask'])))
        self._values = jax.jit(lambda p, b: score_values(policy_forward(p, b, self.model_config,
                                                                       dtype=self.dtype, training=False)))

    def _validate_batch(self, batch: dict, *, critic=False):
        required = {'features', 'return', 'sample_mask'}
        if not critic:
            required |= {'unit_action', 'market_action', 'unit_mask', 'market_mask', 'unit_legal_mask',
                         'market_legal_mask', 'old_log_prob', 'advantage'}
            if self.action_selection_contract != EXECUTION_CONTRACT or self.season_controller is not None:
                required |= {'unit_policy_mask', 'market_policy_mask'}
        if required - batch.keys():
            raise ValueError(f'missing learning batch fields: {sorted(required - batch.keys())}')
        count = len(batch['features'])
        if count < 1 or any(np.asarray(value).ndim < 1 or len(value) != count for value in batch.values()):
            raise ValueError('learning arrays require one consistent leading sample dimension')
        if not _finite(batch):
            raise ValueError('learning arrays must be finite')
        weights = np.asarray(batch['sample_mask'])
        if weights.shape != (count,) or not np.isin(weights, [0., 1.]).all() or weights.sum() < 1:
            raise ValueError('sample_mask requires at least one valid binary row')
        if np.asarray(batch['return']).shape != (count,):
            raise ValueError('value returns must be one scalar per transition')
        if critic:
            return
        for name in ('old_log_prob', 'advantage'):
            if np.asarray(batch[name]).shape != (count,):
                raise ValueError(f'{name} must be one scalar per transition')
        for head in ('unit', 'market'):
            actions = np.asarray(batch[f'{head}_action'])
            active = np.asarray(batch.get(f'{head}_mask', np.ones_like(actions, dtype=bool)))
            if actions.ndim != 2 or active.shape != actions.shape or not np.isin(active, [0, 1]).all():
                raise ValueError(f'{head} action and active mask shapes differ')
            policy_active = np.asarray(batch.get(f'{head}_policy_mask', active))
            if (policy_active.shape != active.shape or not np.isin(policy_active, [0, 1]).all()
                    or np.any(policy_active > active)):
                raise ValueError(f'{head} policy mask must be binary and contained in the presence mask')
            valid = (active > 0) & (weights[:, None] > 0)
            if np.any(actions[valid] < 0) or not np.issubdtype(actions.dtype, np.integer):
                raise ValueError('active sampled actions must be nonnegative integer ids')
            legal = batch.get(f'{head}_legal_mask')
            if legal is not None:
                legal = np.asarray(legal)
                if legal.shape[:-1] != actions.shape or legal.ndim != 3:
                    raise ValueError(f'{head} conditional legal mask shape differs from actions')
                if not np.isin(legal, [0, 1]).all() or not np.all(legal.any(-1)):
                    raise ValueError('each categorical slot must admit at least one legal action')
                if np.any(actions[valid] >= legal.shape[-1]):
                    raise ValueError('sampled action id outside categorical action space')
                safe_actions = np.clip(actions, 0, legal.shape[-1] - 1)
                selected = np.take_along_axis(legal, safe_actions[..., None], -1)[..., 0]
                if not np.all(selected[valid]):
                    raise ValueError('sampled action not legal under its recorded prefix mask')

    def update(self, batch: dict) -> dict:
        self._validate_batch(batch)
        params, state, metrics = self._update(self.params, self.optimizer_state, self.teacher,
                                             jax.tree.map(jnp.asarray, batch))
        if not _finite((params, state, metrics)):
            raise RuntimeError('nonfinite PPO update; current parameters and optimizer remain unchanged')
        self.params, self.optimizer_state = params, state
        self.completed_updates += 1
        return {key: float(np.asarray(value)) for key, value in jax.device_get(metrics).items()}

    def warmup(self, batch: dict) -> dict:
        self._validate_batch(batch, critic=True)
        actor = {key: value for key, value in self.params.items() if key != 'value'}
        value, state, metrics = self._critic_update(self.params['value'], self.critic_optimizer_state, actor,
                                                   jax.tree.map(jnp.asarray, batch))
        if not _finite((value, state, metrics)):
            raise RuntimeError('nonfinite critic update; current parameters and optimizer remain unchanged')
        # Exactly the original actor leaves; no zero-gradient Adam or decay update touches them.
        self.params = {**actor, 'value': value}
        self.critic_optimizer_state = state
        self.completed_critic_updates += 1
        return {key: float(np.asarray(value)) for key, value in jax.device_get(metrics).items()}

    @staticmethod
    def _batches(batch, minibatch_size, *, epochs=1, seed=None):
        if minibatch_size < 2 or minibatch_size % 2 or epochs < 1:
            raise ValueError('minibatch size must be positive even and epochs positive')
        arrays = {key: np.asarray(value) for key, value in batch.items()}
        count = len(arrays['features'])
        if count < 2 or count % 2:
            raise ValueError('learning rollout must contain adjacent seat pairs')
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            pair_indices = np.arange(count // 2) if seed is None else rng.permutation(count // 2)
            indices = (pair_indices[:, None] * 2 + np.arange(2)[None, :]).reshape(-1)
            for start in range(0, count, minibatch_size):
                chosen = indices[start:start + minibatch_size]
                size = len(chosen)
                if size < minibatch_size:
                    chosen = np.pad(chosen, (0, minibatch_size - size), mode='wrap')
                result = {key: value[chosen] for key, value in arrays.items()}
                result['sample_mask'] = result.get('sample_mask', np.ones(minibatch_size, np.float32)).copy()
                result['sample_mask'][size:] = 0
                yield result

    def verify_behavior(self, batch, minibatch_size):
        """Reject stale or mismatched on-policy labels before any PPO parameter update."""
        self._validate_batch(batch)
        maximum_error = 0.
        for minibatch in self._batches(batch, minibatch_size):
            logs = np.asarray(jax.device_get(self._log_probs(self.params, jax.tree.map(jnp.asarray, minibatch))))
            valid = minibatch['sample_mask'] > 0
            maximum_error = max(maximum_error, float(np.max(np.abs(logs[valid] - minibatch['old_log_prob'][valid]))))
        # Sampling computes float64 log-softmax from the same FP32 logits. The
        # accumulated FP32 joint probability has only a small rounding allowance.
        if not np.isfinite(maximum_error) or maximum_error > 2e-4:
            raise ValueError(f'collected behavior log probability mismatch: {maximum_error:.6g}')
        return maximum_error

    def _learn_rollout(self, batch, minibatch_size, epochs, seed, *, critic):
        self._validate_batch(batch, critic=critic)
        behavior_error = None if critic else self.verify_behavior(batch, minibatch_size)
        if not critic and self.config.normalize_advantages:
            from .objective import actor_sample_mask
            mask = np.asarray(actor_sample_mask(batch), np.float64)
            advantages = np.asarray(batch['advantage'], np.float64)
            count_valid = max(1., mask.sum())
            mean = float(np.sum(advantages * mask) / count_valid)
            deviation = float(np.sqrt(np.sum(np.square(advantages - mean) * mask) / count_valid))
            count = len(mask)
            batch = {**batch, 'advantage_normalization_mean': np.full(count, mean, np.float32),
                     'advantage_normalization_standard_deviation': np.full(count, deviation, np.float32)}
        sums, samples, updates = {}, 0., 0
        for minibatch in self._batches(batch, minibatch_size, epochs=epochs, seed=seed):
            row = self.warmup(minibatch) if critic else self.update(minibatch)
            count = row.pop('sample_count')
            samples += count
            updates += 1
            for key, value in row.items():
                sums[key] = sums.get(key, 0.) + value * count
        result = {**{key: value / samples for key, value in sums.items()}, 'samples': int(samples), 'updates': updates}
        if behavior_error is not None:
            result['initial_behavior_log_prob_max_error'] = behavior_error
        return result

    def update_rollout(self, batch, minibatch_size, epochs, seed):
        return self._learn_rollout(batch, minibatch_size, epochs, seed, critic=False)

    def warmup_rollout(self, batch, minibatch_size, epochs, seed):
        return self._learn_rollout(batch, minibatch_size, epochs, seed, critic=True)

    def validate_values(self, batch, minibatch_size):
        self._validate_batch(batch, critic=True)
        squared_error, count, prediction = 0., 0., 0.
        for minibatch in self._batches(batch, minibatch_size):
            values = np.asarray(jax.device_get(self._values(self.params, jax.tree.map(jnp.asarray, minibatch))))
            mask = minibatch['sample_mask']
            squared_error += float(np.sum(np.square(values - minibatch['return']) * mask))
            prediction += float(np.sum(values * mask))
            count += float(mask.sum())
        return {'value_mse': squared_error / count, 'value_mean': prediction / count, 'samples': int(count)}

    def save(self, path: Path, identity: dict, metadata: dict) -> None:
        if policy_hash(self.teacher) != self.teacher_sha256:
            raise RuntimeError('frozen BC teacher changed')
        params, teacher = _host(self.params), _host(self.teacher)
        optimizer, critic_optimizer = _host(self.optimizer_state), _host(self.critic_optimizer_state)
        payload = {'contract': STATE_CONTRACT, 'pipeline': PIPELINE_CONTRACT, 'objective': OBJECTIVE_CONTRACT,
                   'execution': self.action_selection_contract, 'identity': identity, 'metadata': dict(metadata),
                   'season_controller': self.season_controller, 'controller_identity': self.controller_identity,
                   'lineage_sha256': _lineage_hash(identity, metadata),
                   'params': params, 'teacher': teacher, 'policy_sha256': policy_hash(params),
                   'teacher_sha256': self.teacher_sha256, 'optimizer_state': optimizer,
                   'critic_optimizer_state': critic_optimizer, 'optimizer_sha256': policy_hash(optimizer),
                   'critic_optimizer_sha256': policy_hash(critic_optimizer),
                   'model_config': self.model_config.to_dict(), 'config': asdict(self.config),
                   'compute_dtype': self.compute_dtype, 'completed_updates': self.completed_updates,
                   'completed_critic_updates': self.completed_critic_updates}
        atomic_pickle(Path(path), payload)

    @classmethod
    def load(cls, path: Path, model_config, config: PPOConfig, expected_identity: dict):
        saved = _load_checkpoint(path)
        model = model_config if isinstance(model_config, JaxModelConfig) else JaxModelConfig(**model_config)
        if saved['identity'] != expected_identity or saved['model_config'] != model.to_dict():
            raise ValueError('PPO resume source/model/seed/lineage identity mismatch; use a new run directory')
        if saved['config'] != asdict(config):
            raise ValueError('PPO resume optimizer/loss configuration mismatch; use a new run directory')
        expected_dtype = expected_identity.get('settings', {}).get('compute_dtype', saved['compute_dtype'])
        if saved['compute_dtype'] != expected_dtype:
            raise ValueError('PPO resume dtype mismatch')
        trainer = cls(model, config, saved['params'], saved['teacher'], compute_dtype=saved['compute_dtype'],
                      action_selection_contract=saved['execution'],
                      season_controller=saved.get('season_controller'),
                      controller_identity=saved.get('controller_identity'))
        trainer.optimizer_state = jax.tree.map(jnp.asarray, saved['optimizer_state'])
        trainer.critic_optimizer_state = jax.tree.map(jnp.asarray, saved['critic_optimizer_state'])
        expected_states = (trainer.optimizer.init(trainer.params),
                           trainer.critic_optimizer.init(trainer.params['value']))
        restored_states = (trainer.optimizer_state, trainer.critic_optimizer_state)
        if jax.tree.structure(restored_states) != jax.tree.structure(expected_states):
            raise ValueError('PPO optimizer tree changed; incompatible state cannot resume')
        for restored, expected in zip(jax.tree.leaves(restored_states), jax.tree.leaves(expected_states)):
            if restored.shape != expected.shape or restored.dtype != expected.dtype:
                raise ValueError('PPO optimizer leaf shape/dtype changed; incompatible state cannot resume')
        trainer.completed_updates = int(saved['completed_updates'])
        trainer.completed_critic_updates = int(saved['completed_critic_updates'])
        return trainer, dict(saved['metadata'])

    def export_policy(self, path: Path, metadata: dict | None = None):
        from .sampling import validate_action_selection_contract
        validate_action_selection_contract(self.action_selection_contract)
        params = _host(self.params)
        payload = {**(metadata or {}), 'contract': POLICY_CONTRACT, 'params': params,
                   'model_config': self.model_config.to_dict(), 'training_backend': 'jax',
                   'compute_dtype': self.compute_dtype,
                   'rl_completed_updates': self.completed_updates, 'policy_sha256': policy_hash(params),
                   'action_selection_contract': self.action_selection_contract,
                   'season_controller': self.season_controller,
                   'controller_identity': self.controller_identity,
                   'learning_objective_contract': OBJECTIVE_CONTRACT, 'deployment': 'candidate_only'}
        atomic_pickle(Path(path), payload)

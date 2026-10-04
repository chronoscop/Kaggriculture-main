# Adapted from msdsm/kaggriculture-solution, commit 84057a0fda4238ccdebc46f9bf5496c6c4b2e00d.
# Source: training/objectives.py and training/rollout.py; project objective and mask contract differ.
"""Single-device clipped PPO with a frozen BC anchor and terminal match score.

Every probability, entropy and KL uses the collected action's active slots and
prefix-conditioned legal masks. Cash is deliberately absent from this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..full_action.model import JaxModelConfig, policy_forward
from .provenance import OBJECTIVE_CONTRACT

MASKED_LOGIT = -1e9


@dataclass(frozen=True)
class PPOConfig:
    clip: float = 0.2
    teacher_kl: float = 0.2
    unit_entropy: float = 0.0015
    market_entropy: float = 0.0015
    value_coefficient: float = 2.0
    value_huber_delta: float = 1.0
    learning_rate: float = 5e-5
    adam_epsilon: float = 1e-5
    gradient_norm: float = 5.0
    gamma: float = 1.0
    gae_lambda: float = 0.97
    normalize_advantages: bool = True
    warmup_steps: int = 240
    lr_decay_steps: int = 150_000
    lr_min_ratio: float = 0.25

    def __post_init__(self):
        numeric = (self.clip, self.teacher_kl, self.unit_entropy, self.market_entropy,
                   self.value_coefficient, self.value_huber_delta, self.learning_rate,
                   self.adam_epsilon, self.gradient_norm, self.gamma, self.gae_lambda,
                   self.lr_min_ratio)
        if not all(math.isfinite(x) for x in numeric):
            raise ValueError('PPO coefficients must be finite')
        if not 0 < self.clip < 1:
            raise ValueError('PPO clip must be between zero and one')
        if min(self.teacher_kl, self.unit_entropy, self.market_entropy, self.value_coefficient) < 0:
            raise ValueError('PPO loss coefficients cannot be negative')
        if min(self.value_huber_delta, self.learning_rate, self.adam_epsilon, self.gradient_norm) <= 0:
            raise ValueError('PPO optimizer and Huber parameters must be positive')
        if self.gamma != 1.0:
            raise ValueError('terminal match score objective requires gamma=1')
        if not 0 <= self.gae_lambda <= 1 or not 0 < self.lr_min_ratio <= 1:
            raise ValueError('invalid GAE lambda or minimum learning rate ratio')
        if self.warmup_steps < 0 or self.lr_decay_steps < 1:
            raise ValueError('invalid optimizer schedule length')


def masked_logits(logits, legal):
    values = logits.astype(jnp.float32)
    legal_maximum = jnp.max(jnp.where(legal, values, -jnp.inf), axis=-1, keepdims=True)
    legal_maximum = jnp.where(jnp.any(legal, axis=-1, keepdims=True), legal_maximum, 0.)
    return jnp.where(legal, values - legal_maximum, jnp.float32(MASKED_LOGIT))


def behavior_outputs(outputs: dict, batch: dict) -> dict:
    """Rebuild the exact collected conditional categorical distributions."""
    result = dict(outputs)
    for head in ('unit', 'market'):
        legal = batch.get(f'{head}_legal_mask')
        if legal is not None:
            if legal.shape != outputs[f'{head}_action'].shape:
                raise ValueError(f'{head} recorded legal mask must exactly match policy logits')
            result[f'{head}_action'] = masked_logits(outputs[f'{head}_action'], legal)
    return result


def selected_log_prob(logits, actions):
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    # Inactive padding actions may be -1; those factors are excluded by the mask.
    return jnp.take_along_axis(log_probs, jnp.maximum(actions, 0)[..., None], axis=-1)[..., 0]


def joint_log_prob(outputs, unit_actions, market_actions, unit_mask, market_mask=None):
    if outputs['unit_action'].shape[:-1] != unit_actions.shape or unit_mask.shape != unit_actions.shape:
        raise ValueError('unit action/mask must exactly match categorical policy factors')
    if outputs['market_action'].shape[:-1] != market_actions.shape:
        raise ValueError('market actions must exactly match categorical policy factors')
    if market_mask is not None and market_mask.shape != market_actions.shape:
        raise ValueError('market mask must exactly match categorical policy factors')
    unit = selected_log_prob(outputs['unit_action'], unit_actions)
    market = selected_log_prob(outputs['market_action'], market_actions)
    if market_mask is None:
        market_mask = jnp.ones_like(market, dtype=jnp.bool_)
    return jnp.sum(jnp.where(unit_mask, unit, 0.), axis=-1) + jnp.sum(jnp.where(market_mask, market, 0.), axis=-1)


def categorical_entropy(logits):
    logs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    return -jnp.sum(jnp.exp(logs) * logs, axis=-1)


def categorical_teacher_kl(logits, teacher_logits):
    """KL(frozen BC teacher || candidate) after identical conditional masks."""
    policy_logs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    teacher_logs = jax.lax.stop_gradient(jax.nn.log_softmax(teacher_logits.astype(jnp.float32), axis=-1))
    return jnp.sum(jnp.exp(teacher_logs) * (teacher_logs - policy_logs), axis=-1)


def joint_entropy_components(outputs, unit_mask, market_mask=None):
    unit = categorical_entropy(outputs['unit_action'])
    market = categorical_entropy(outputs['market_action'])
    if market_mask is None:
        market_mask = jnp.ones_like(market, dtype=jnp.bool_)
    return jnp.sum(jnp.where(unit_mask, unit, 0.), -1), jnp.sum(jnp.where(market_mask, market, 0.), -1)


def joint_teacher_kl(outputs, teacher_outputs, unit_mask, market_mask=None):
    unit = categorical_teacher_kl(outputs['unit_action'], teacher_outputs['unit_action'])
    market = categorical_teacher_kl(outputs['market_action'], teacher_outputs['market_action'])
    if market_mask is None:
        market_mask = jnp.ones_like(market, dtype=jnp.bool_)
    return jnp.sum(jnp.where(unit_mask, unit, 0.), -1) + jnp.sum(jnp.where(market_mask, market, 0.), -1)


def masked_mean(values, mask):
    # where, rather than multiply by zero, also protects inactive padding rows.
    return jnp.sum(jnp.where(mask > 0, values * mask, 0.)) / jnp.maximum(jnp.sum(mask), 1.)


def actor_sample_mask(batch):
    units = batch.get('unit_policy_mask', batch['unit_mask'])
    market = batch.get('market_policy_mask', batch.get('market_mask'))
    chosen = jnp.any(units, axis=-1)
    if market is not None:
        chosen = chosen | jnp.any(market, axis=-1)
    else:
        chosen = jnp.ones_like(chosen)
    return batch['sample_mask'].astype(jnp.float32) * chosen.astype(jnp.float32)


def normalize_masked_advantages(advantages, sample_mask, epsilon=1e-8):
    mean = masked_mean(advantages, sample_mask)
    variance = masked_mean(jnp.square(advantages - mean), sample_mask)
    return jnp.where(sample_mask > 0, (advantages - mean) / (jnp.sqrt(jnp.maximum(variance, 0.)) + epsilon), 0.)


def clipped_surrogate(log_prob, old_log_prob, advantages, clip):
    ratio = jnp.exp(log_prob - jax.lax.stop_gradient(old_log_prob))
    clipped = jnp.clip(ratio, 1. - clip, 1. + clip)
    return jnp.minimum(ratio * advantages, clipped * advantages), ratio


def score_values(outputs):
    """Critic predicts expected terminal win/draw/loss score in [0, 1]."""
    return jax.nn.sigmoid(outputs['value'].astype(jnp.float32))


def generalized_advantage_estimate(values, terminal_scores, *, gamma=1.0, gae_lambda=0.97):
    """Full-game GAE: [time, game, seat], terminal reward only, no bootstrapping.

    The collector must call this only after normal complete games. Short smoke
    rollouts do not have a terminal outcome and must not create learning labels.
    """
    values = np.asarray(values, dtype=np.float32)
    scores = np.asarray(terminal_scores, dtype=np.float32)
    if gamma != 1.0 or not 0 <= gae_lambda <= 1:
        raise ValueError('terminal score GAE requires gamma=1 and lambda in [0,1]')
    if values.ndim < 2 or values.shape[0] < 1 or scores.shape != values.shape[1:]:
        raise ValueError('GAE expects values [time, ...] and terminal scores [...]')
    if not np.isfinite(values).all() or not np.isfinite(scores).all():
        raise ValueError('GAE inputs must be finite')
    if not np.isin(scores, [0., .5, 1.]).all():
        raise ValueError('terminal rewards must be actual win=1/draw=.5/loss=0 outcomes')
    advantages = np.empty_like(values)
    next_advantage = np.zeros_like(scores)
    for t in range(len(values) - 1, -1, -1):
        next_value = np.zeros_like(scores) if t == len(values) - 1 else values[t + 1]
        reward = scores if t == len(values) - 1 else np.zeros_like(scores)
        delta = reward + gamma * next_value - values[t]
        next_advantage = delta + gamma * gae_lambda * next_advantage
        advantages[t] = next_advantage
    return advantages, advantages + values


def learning_rate_schedule(config: PPOConfig):
    decay = optax.linear_schedule(config.learning_rate, config.learning_rate * config.lr_min_ratio,
                                 config.lr_decay_steps)
    if not config.warmup_steps:
        return decay
    warmup = optax.linear_schedule(0., config.learning_rate, config.warmup_steps)
    return optax.join_schedules([warmup, decay], [config.warmup_steps])


def make_optimizer(config: PPOConfig, *, critic=False):
    # Critic head warmup uses a separate optimizer; it cannot alter actor Adam state.
    rate = config.learning_rate if critic else learning_rate_schedule(config)
    return optax.chain(optax.clip_by_global_norm(config.gradient_norm), optax.adam(rate, eps=config.adam_epsilon))


def loss_and_metrics(params, teacher, batch, model, config, dtype):
    raw = policy_forward(params, batch, model, dtype=dtype, training=False)
    outputs = behavior_outputs(raw, batch)
    teacher_outputs = behavior_outputs(jax.lax.stop_gradient(policy_forward(teacher, batch, model, dtype=dtype,
                                                                           training=False)), batch)
    # Presence masks belong to the encoder. Forced rules and external planners
    # are real transitions, but are not categorical choices made by the actor.
    unit_mask = batch.get('unit_policy_mask', batch['unit_mask'])
    market_mask = batch.get('market_policy_mask', batch.get('market_mask'))
    mask = batch['sample_mask'].astype(jnp.float32)
    actor_mask = actor_sample_mask(batch)
    advantage = jax.lax.stop_gradient(batch['advantage'])
    if config.normalize_advantages:
        mean = batch.get('advantage_normalization_mean')
        deviation = batch.get('advantage_normalization_standard_deviation')
        advantage = (normalize_masked_advantages(advantage, actor_mask) if mean is None or deviation is None
                     else (advantage - mean) / (deviation + 1e-8))
    logs = joint_log_prob(outputs, batch['unit_action'], batch['market_action'], unit_mask, market_mask)
    surrogate, ratio = clipped_surrogate(logs, batch['old_log_prob'], advantage, config.clip)
    policy_loss = -masked_mean(surrogate, actor_mask)
    values = score_values(raw)
    target = jax.lax.stop_gradient(batch['return'])
    value_loss = masked_mean(optax.huber_loss(values, target, delta=config.value_huber_delta), mask)
    unit_entropy, market_entropy = joint_entropy_components(outputs, unit_mask, market_mask)
    anchor = masked_mean(joint_teacher_kl(outputs, teacher_outputs, unit_mask, market_mask), actor_mask)
    terms = {'policy_loss': policy_loss, 'value_loss': value_loss, 'teacher_kl': anchor,
             'unit_entropy': masked_mean(unit_entropy, actor_mask), 'market_entropy': masked_mean(market_entropy, actor_mask)}
    total = policy_loss + config.value_coefficient * value_loss + config.teacher_kl * anchor
    total -= config.unit_entropy * terms['unit_entropy'] + config.market_entropy * terms['market_entropy']
    return total, {**terms, 'loss': total, 'sample_count': jnp.sum(mask), 'actor_sample_count': jnp.sum(actor_mask),
                   'value_mean': masked_mean(values, mask), 'ratio_mean': masked_mean(ratio, actor_mask),
                   'clip_fraction': masked_mean((jnp.abs(ratio - 1.) > config.clip).astype(jnp.float32), actor_mask),
                   'approx_old_kl': masked_mean(batch['old_log_prob'] - logs, actor_mask),
                   'max_log_prob_error': jnp.max(jnp.where(mask > 0, jnp.abs(logs - batch['old_log_prob']), 0.))}


def make_update_step(model: JaxModelConfig, config: PPOConfig, dtype, optimizer):
    differentiate = jax.value_and_grad(lambda p, t, b: loss_and_metrics(p, t, b, model, config, dtype), has_aux=True)

    @jax.jit
    def update(params, state, teacher, batch):
        (_, metrics), gradients = differentiate(params, teacher, batch)
        updates, state = optimizer.update(gradients, state, params)
        return optax.apply_updates(params, updates), state, {**metrics, 'gradient_norm': optax.global_norm(gradients)}
    return update


def make_critic_update_step(model: JaxModelConfig, config: PPOConfig, dtype, optimizer):
    """Differentiate only value parameters; the entire actor/trunk remains fixed."""
    def loss(value, actor, batch):
        params = {**actor, 'value': value}
        values = score_values(policy_forward(params, batch, model, dtype=dtype, training=False))
        mask = batch['sample_mask'].astype(jnp.float32)
        mse = masked_mean(jnp.square(values - batch['return']), mask)
        huber = masked_mean(optax.huber_loss(values, batch['return'], delta=config.value_huber_delta), mask)
        return config.value_coefficient * huber, {'loss': config.value_coefficient * huber,
                                                 'value_loss': huber, 'value_mse': mse,
                                                 'sample_count': jnp.sum(mask)}
    differentiate = jax.value_and_grad(loss, has_aux=True)

    @jax.jit
    def update(value, state, actor, batch):
        (_, metrics), gradients = differentiate(value, actor, batch)
        updates, state = optimizer.update(gradients, state, value)
        return optax.apply_updates(value, updates), state, {**metrics, 'gradient_norm': optax.global_norm(gradients)}
    return update

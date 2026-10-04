"""PPO probability, terminal targets, freezing and exact-resume invariants."""
from dataclasses import replace
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from route_rl.full_action.checkpoints import policy_hash
from route_rl.full_action.model import JaxModelConfig
from route_rl.ppo.objective import (PPOConfig, behavior_outputs, categorical_teacher_kl,
                                   clipped_surrogate, generalized_advantage_estimate,
                                   joint_log_prob, masked_logits, normalize_masked_advantages)
from route_rl.ppo.trainer import PPOTrainer, checkpoint_metadata


class PPOMath(unittest.TestCase):
    def test_joint_ratio_uses_only_active_sampled_slots(self):
        outputs = {'unit_action': jnp.log(jnp.array([[[.7, .3], [.2, .8]]])),
                   'market_action': jnp.log(jnp.array([[[.4, .6], [.9, .1]]]))}
        logs = joint_log_prob(outputs, jnp.array([[0, 1]]), jnp.array([[0, 1]]),
                              jnp.array([[True, False]]), jnp.array([[True, False]]))
        np.testing.assert_allclose(np.asarray(logs), np.log(.7 * .4), rtol=1e-6)
        def loss(unit_logits):
            return joint_log_prob({**outputs, 'unit_action': unit_logits}, jnp.array([[0, 1]]),
                                  jnp.array([[0, 1]]), jnp.array([[True, False]]),
                                  jnp.array([[True, False]])).sum()
        gradients = jax.grad(loss)(outputs['unit_action'])
        np.testing.assert_array_equal(gradients[0, 1], [0., 0.])

    def test_legal_support_remains_exact_for_very_negative_logits(self):
        logits = jnp.array([[[-2e10, -2e10, 3.]]])
        legal = jnp.array([[[True, True, False]]])
        probabilities = jax.nn.softmax(masked_logits(logits, legal), -1)
        np.testing.assert_array_equal(probabilities, [[[.5, .5, 0.]]])
        outputs = {'unit_action': logits, 'market_action': logits}
        reconstructed = behavior_outputs(outputs, {'unit_legal_mask': legal, 'market_legal_mask': legal})
        logs = joint_log_prob(reconstructed, jnp.zeros((1, 1), dtype=int), jnp.zeros((1, 1), dtype=int),
                              jnp.ones((1, 1), dtype=bool), jnp.ones((1, 1), dtype=bool))
        self.assertAlmostEqual(float(logs[0]), np.log(.25), places=6)

    def test_clipping_is_signed_and_padding_does_not_change_normalization(self):
        advantage = jnp.array([1., -1., 99.])
        ratio = jnp.array([2., .5, 1.])
        surrogate, actual = clipped_surrogate(jnp.log(ratio), jnp.zeros(3), advantage, .2)
        np.testing.assert_allclose(surrogate[:2], [1.2, -.8], rtol=1e-6)
        np.testing.assert_allclose(actual, ratio, rtol=1e-6)
        normalized = normalize_masked_advantages(jnp.array([1., 3., 1000.]), jnp.array([1., 1., 0.]))
        np.testing.assert_allclose(normalized, [-1., 1., 0.], atol=1e-6)

    def test_teacher_is_frozen_and_kl_zero_for_same_distribution(self):
        logits = jnp.array([[1., 2., 3.]])
        np.testing.assert_allclose(categorical_teacher_kl(logits, logits), 0., atol=1e-6)
        teacher_gradient = jax.grad(lambda teacher: categorical_teacher_kl(-logits, teacher).sum())(logits)
        np.testing.assert_array_equal(teacher_gradient, np.zeros_like(logits))
        self.assertGreater(float(categorical_teacher_kl(-logits, logits)[0]), 0.)

    def test_gae_uses_only_actual_terminal_score_and_return_order(self):
        values = np.array([[[.2, .8]], [[.4, .6]], [[.7, .3]]], np.float32)
        scores = np.array([[1., 0.]], np.float32)
        advantages, returns = generalized_advantage_estimate(values, scores, gae_lambda=1.)
        np.testing.assert_allclose(returns, np.broadcast_to(scores, values.shape), atol=1e-6)
        np.testing.assert_allclose(advantages, returns - values, atol=1e-6)
        a0, r0 = generalized_advantage_estimate(values, scores, gae_lambda=0.)
        np.testing.assert_allclose(r0[:-1], values[1:], atol=1e-6)
        np.testing.assert_allclose(r0[-1], scores, atol=1e-6)
        draw_a, draw_r = generalized_advantage_estimate(values, np.full((1, 2), .5), gae_lambda=1.)
        np.testing.assert_allclose(draw_r, .5, atol=1e-6)
        with self.assertRaisesRegex(ValueError, 'actual win'):
            generalized_advantage_estimate(values, np.array([[1234., 5678.]]))
        with self.assertRaises(ValueError):
            PPOConfig(gamma=.99)


def tiny_forward(params, batch, model, *, dtype=jnp.float32, training=False):
    """Small differentiable shared-trunk fixture isolates optimizer wiring."""
    del model, dtype, training
    h = batch['features'][:, 0] @ params['trunk']['kernel']
    units = (h @ params['unit_action']['kernel'])[:, None, :]
    markets = (h @ params['market_action']['kernel'])[:, None, :]
    result = {'unit_action': units, 'market_action': markets}
    if 'value' in params:
        v = jax.nn.gelu(h @ params['value']['hidden']['kernel'] + params['value']['hidden']['bias'])
        result['value'] = (v @ params['value']['output']['kernel'] + params['value']['output']['bias'])[:, 0]
    return result


def tiny_fixture():
    params = {'trunk': {'kernel': jnp.array([[.3, .1], [.1, .4]])},
              'unit_action': {'kernel': jnp.array([[.4, -.2], [.1, .3]])},
              'market_action': {'kernel': jnp.array([[.2, -.1], [.3, .1]])},
              'value': {'hidden': {'kernel': jnp.eye(2), 'bias': jnp.zeros(2)},
                        'output': {'kernel': jnp.zeros((2, 1)), 'bias': jnp.zeros(1)}}}
    batch = {'features': np.array([[[1., .2]], [[.3, 1.]], [[1.2, .1]], [[.1, .8]]], np.float32),
             'unit_action': np.array([[0], [1], [0], [1]], np.int32),
             'market_action': np.zeros((4, 1), np.int32),
             'unit_mask': np.ones((4, 1), bool), 'market_mask': np.ones((4, 1), bool),
             'unit_legal_mask': np.ones((4, 1, 2), bool), 'market_legal_mask': np.ones((4, 1, 2), bool),
             'advantage': np.array([.5, -.5, .5, -.5], np.float32),
             'return': np.array([1., 0., 1., 0.], np.float32), 'sample_mask': np.ones(4, np.float32)}
    outputs = tiny_forward(params, batch, None)
    batch['old_log_prob'] = np.asarray(joint_log_prob(outputs, batch['unit_action'], batch['market_action'],
                                                    batch['unit_mask'], batch['market_mask']))
    return params, batch


class OptimizerContracts(unittest.TestCase):
    def setUp(self):
        self.patches = [patch('route_rl.ppo.objective.policy_forward', tiny_forward),
                        patch('route_rl.ppo.trainer.policy_forward', tiny_forward)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.params, self.batch = tiny_fixture()
        self.model = JaxModelConfig(d_model=4, layers=1, heads=1, ffn_dim=8, rope_dim=4)
        self.config = PPOConfig(warmup_steps=0, learning_rate=.01)

    def learner(self):
        return PPOTrainer(self.model, self.config, self.params, self.params)

    def test_warmup_changes_only_critic_and_preserves_actor_optimizer(self):
        learner = self.learner()
        actor_before = policy_hash({k: v for k, v in learner.params.items() if k != 'value'})
        teacher_before = policy_hash(learner.teacher)
        optimizer_before = policy_hash(learner.optimizer_state)
        value_before = policy_hash(learner.params['value'])
        learner.warmup(self.batch)
        self.assertEqual(actor_before, policy_hash({k: v for k, v in learner.params.items() if k != 'value'}))
        self.assertEqual(optimizer_before, policy_hash(learner.optimizer_state))
        self.assertEqual(teacher_before, policy_hash(learner.teacher))
        self.assertNotEqual(value_before, policy_hash(learner.params['value']))
        metrics = learner.update(self.batch)
        self.assertNotEqual(actor_before, policy_hash({k: v for k, v in learner.params.items() if k != 'value'}))
        self.assertEqual(teacher_before, policy_hash(learner.teacher))
        self.assertTrue(np.isfinite(metrics['loss']))

    def test_pair_padding_and_whole_rollout_advantage_statistics(self):
        batches = list(PPOTrainer._batches(self.batch, 6, epochs=1, seed=19))
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]['sample_mask'].sum(), 4)
        for offset in (0, 2):
            first = batches[0]['features'][offset, 0]
            second = batches[0]['features'][offset + 1, 0]
            index = np.flatnonzero(np.all(self.batch['features'][:, 0] == first, axis=-1))[0]
            self.assertEqual(index % 2, 0)
            np.testing.assert_array_equal(second, self.batch['features'][index + 1, 0])
        learner = self.learner()
        observed = []
        def capture(batch):
            observed.append(batch)
            return {'sample_count': batch['sample_mask'].sum(), 'loss': 0.}
        with patch.object(learner, 'update', capture):
            learner.update_rollout(self.batch, 2, 1, 3)
        for batch in observed:
            np.testing.assert_allclose(batch['advantage_normalization_mean'], 0.)
            np.testing.assert_allclose(batch['advantage_normalization_standard_deviation'], .5)
        with self.assertRaisesRegex(ValueError, 'even'):
            list(PPOTrainer._batches(self.batch, 3))

    def test_resume_restores_teacher_adam_counters_and_lineage(self):
        learner = self.learner()
        learner.warmup(self.batch)
        learner.update(self.batch)
        identity = {'settings': {'compute_dtype': 'float32'}, 'source': 'fixture'}
        meta = {'training_seeds': [101], 'next_seed': 102}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pkl'
            learner.save(path, identity, meta)
            restored, metadata = PPOTrainer.load(path, self.model, self.config, identity)
            self.assertEqual(metadata, meta)
            self.assertEqual(checkpoint_metadata(path, expected_identity=identity), meta)
            with self.assertRaisesRegex(ValueError, 'different run integration'):
                checkpoint_metadata(path, expected_identity={**identity, 'source': 'other-run'})
            self.assertEqual(restored.completed_updates, 1)
            self.assertEqual(restored.completed_critic_updates, 1)
            learner.update(self.batch)
            restored.update(self.batch)
            self.assertEqual(policy_hash(learner.params), policy_hash(restored.params))
            self.assertEqual(policy_hash(learner.optimizer_state), policy_hash(restored.optimizer_state))
            with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
                PPOTrainer.load(path, self.model, replace(self.config, teacher_kl=.3), identity)
            with path.open('rb') as stream:
                saved = pickle.load(stream)
            saved['metadata']['next_seed'] = 900
            with path.open('wb') as stream:
                pickle.dump(saved, stream)
            with self.assertRaisesRegex(ValueError, 'lineage metadata checksum'):
                checkpoint_metadata(path)

    def test_stale_behavior_is_rejected_before_any_update(self):
        learner = self.learner()
        original = policy_hash(learner.params)
        bad = {**self.batch, 'old_log_prob': self.batch['old_log_prob'] + .1}
        with self.assertRaisesRegex(ValueError, 'behavior log probability mismatch'):
            learner.update_rollout(bad, 2, 1, 4)
        self.assertEqual(learner.completed_updates, 0)
        self.assertEqual(policy_hash(learner.params), original)

    def test_export_preserves_dtype_and_does_not_export_training_state(self):
        from route_rl.full_action.checkpoints import load_training_source, POLICY_CONTRACT
        learner = self.learner()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.pkl'
            learner.export_policy(path, {'source_policy_sha256': 'source', 'contract': 'override'})
            policy = load_training_source(path)
            self.assertEqual(policy['contract'], POLICY_CONTRACT)
            self.assertEqual(policy['compute_dtype'], 'float32')
            self.assertEqual(policy['action_selection_contract'], 'official-prefix-full-action-ppo-v1')
            self.assertEqual(policy['learning_objective_contract'], 'terminal-match-score-1-0.5-0-v1')
            self.assertEqual(policy['source_policy_sha256'], 'source')
            self.assertNotIn('optimizer_state', policy)
            self.assertNotIn('teacher', policy)

    def test_real_prefix_mask_labels_are_required(self):
        learner = self.learner()
        with self.assertRaisesRegex(ValueError, 'missing learning batch'):
            learner.update({k: v for k, v in self.batch.items() if k != 'market_legal_mask'})
        batch = {**self.batch, 'unit_legal_mask': self.batch['unit_legal_mask'].copy()}
        batch['unit_legal_mask'][0, 0, 0] = False
        with self.assertRaisesRegex(ValueError, 'recorded prefix mask'):
            learner.update(batch)


if __name__ == '__main__':
    unittest.main()

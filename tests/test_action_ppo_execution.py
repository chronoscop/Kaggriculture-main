"""CPU execution-contract and actor/value ownership regression checks."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from kaggle_environments import make

from route_rl.full_action.checkpoints import atomic_pickle, POLICY_CONTRACT, policy_hash
from route_rl.full_action.model import JaxModelConfig
from route_rl.ppo.inference import load_policy
from route_rl.ppo.objective import PPOConfig, joint_log_prob, loss_and_metrics
from route_rl.ppo.pipeline import flatten_for_learning
from route_rl.ppo.provenance import OBJECTIVE_CONTRACT
from route_rl.ppo.rollout import Rollout
from route_rl.ppo.sampling import (ACTION_SELECTION_CONTRACT, ECONOMIC_ACTION_SELECTION_CONTRACT,
                                   MARKET_ACTIONS, NOOP_ID, PASS_ID, sample_action)
from route_rl.ppo.trainer import PPOTrainer, checkpoint_metadata


def factor_forward(params, batch, model, *, dtype=jnp.float32, training=False):
    """Independent actor factors and a critic isolate gradient ownership."""
    del model, dtype, training
    count = batch['features'].shape[0]
    return {'unit_action': jnp.broadcast_to(params['unit'], (count, 1, 2)),
            'market_action': jnp.broadcast_to(params['market'], (count, 1, 2)),
            'value': params['value']['bias'] + params['value']['slope'] * batch['features'][:, 0, 0]}


def fixture():
    params = {'unit': jnp.array([.6, -.2]), 'market': jnp.array([-.3, .4]),
              'value': {'bias': jnp.array(0.), 'slope': jnp.array(0.)}}
    teacher = {**params, 'unit': -params['unit'], 'market': -params['market']}
    batch = {'features': np.array([[[1.]], [[-1.]], [[.5]], [[-.5]]], np.float32),
             'unit_action': np.array([[0], [1], [0], [1]], np.int32),
             'market_action': np.array([[1], [0], [1], [0]], np.int32),
             'unit_mask': np.ones((4, 1), bool), 'market_mask': np.ones((4, 1), bool),
             'unit_policy_mask': np.zeros((4, 1), bool), 'market_policy_mask': np.zeros((4, 1), bool),
             'unit_legal_mask': np.ones((4, 1, 2), bool), 'market_legal_mask': np.ones((4, 1, 2), bool),
             'old_log_prob': np.zeros(4, np.float32), 'advantage': np.array([.5, -.5, 40., -30.], np.float32),
             'return': np.ones(4, np.float32), 'sample_mask': np.ones(4, np.float32)}
    return params, teacher, batch


class ActorAndCriticOwnership(unittest.TestCase):
    def setUp(self):
        for target in ('route_rl.ppo.objective.policy_forward', 'route_rl.ppo.trainer.policy_forward'):
            replacement = patch(target, factor_forward)
            replacement.start()
            self.addCleanup(replacement.stop)
        self.params, self.teacher, self.batch = fixture()
        self.model = JaxModelConfig(d_model=4, layers=1, heads=1, ffn_dim=8, rope_dim=4)
        self.config = PPOConfig(warmup_steps=0, learning_rate=.01)

    def gradients(self, batch):
        return jax.value_and_grad(lambda params: loss_and_metrics(
            params, self.teacher, batch, self.model, self.config, jnp.float32), has_aux=True)(self.params)

    def learner(self):
        return PPOTrainer(self.model, self.config, self.params, self.teacher,
                          action_selection_contract=ECONOMIC_ACTION_SELECTION_CONTRACT)

    def test_forced_external_rows_have_no_actor_kl_entropy_gradient_but_train_critic(self):
        (_, metrics), gradient = self.gradients(self.batch)
        np.testing.assert_array_equal(gradient['unit'], [0., 0.])
        np.testing.assert_array_equal(gradient['market'], [0., 0.])
        self.assertNotEqual(float(gradient['value']['bias']), 0)
        self.assertEqual(float(metrics['policy_loss']), 0)
        self.assertEqual(float(metrics['teacher_kl']), 0)
        self.assertEqual(float(metrics['unit_entropy']), 0)
        self.assertEqual(float(metrics['market_entropy']), 0)
        self.assertGreater(float(metrics['value_loss']), 0)
        learner = self.learner()
        actor_before = policy_hash({name: learner.params[name] for name in ('unit', 'market')})
        critic_before = policy_hash(learner.params['value'])
        learner.update(self.batch)
        self.assertEqual(actor_before, policy_hash({name: learner.params[name] for name in ('unit', 'market')}))
        self.assertNotEqual(critic_before, policy_hash(learner.params['value']))

    def test_external_advantages_do_not_change_other_actor_gradients_or_normalization(self):
        mixed = deepcopy(self.batch)
        mixed['unit_policy_mask'][:2] = True
        mixed['market_policy_mask'][:2] = True
        outputs = factor_forward(self.params, mixed, self.model)
        mixed['old_log_prob'] = np.asarray(joint_log_prob(outputs, mixed['unit_action'], mixed['market_action'],
                                                         mixed['unit_policy_mask'], mixed['market_policy_mask']))
        altered = {**mixed, 'advantage': mixed['advantage'].copy()}
        altered['advantage'][2:] = [1000., -1000.]
        (_, _), normal_gradient = self.gradients(mixed)
        (_, _), altered_gradient = self.gradients(altered)
        for name in ('unit', 'market'):
            np.testing.assert_allclose(normal_gradient[name], altered_gradient[name], atol=1e-7)
            self.assertGreater(float(jnp.linalg.norm(normal_gradient[name])), 0)
        learner = self.learner()
        captured = []
        def capture(batch):
            captured.append(batch)
            return {'sample_count': batch['sample_mask'].sum(), 'loss': 0.}
        with patch.object(learner, 'update', capture):
            learner.update_rollout(altered, 2, 1, 97)
        for batch in captured:
            np.testing.assert_allclose(batch['advantage_normalization_mean'], 0.)
            np.testing.assert_allclose(batch['advantage_normalization_standard_deviation'], .5)

    def test_malformed_policy_masks_and_missing_v2_ownership_are_rejected(self):
        learner = self.learner()
        for head in ('unit', 'market'):
            for invalid in (np.full((4, 1), .5), np.ones((4, 2), bool)):
                with self.subTest(head=head, invalid=invalid.shape), self.assertRaisesRegex(ValueError, 'policy mask'):
                    learner._validate_batch({**self.batch, f'{head}_policy_mask': invalid})
            absent = {**self.batch, f'{head}_mask': np.zeros((4, 1), bool),
                      f'{head}_policy_mask': np.ones((4, 1), bool)}
            with self.assertRaisesRegex(ValueError, 'presence mask'):
                learner._validate_batch(absent)
            without = {key: value for key, value in self.batch.items() if key != f'{head}_policy_mask'}
            with self.assertRaisesRegex(ValueError, 'policy'):
                learner._validate_batch(without)
        legacy = PPOTrainer(self.model, self.config, self.params, self.teacher)
        legacy._validate_batch({key: value for key, value in self.batch.items() if not key.endswith('_policy_mask')})

    def test_training_state_rejects_mixed_execution_or_continuation_identity(self):
        learner = self.learner()
        identity = {'execution': ECONOMIC_ACTION_SELECTION_CONTRACT,
                    'settings': {'compute_dtype': 'float32'}, 'controller_identity': None}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pkl'
            learner.save(path, identity, {'training_seeds': [918]})
            checkpoint_metadata(path, expected_identity=identity)
            import pickle
            saved = pickle.loads(path.read_bytes())
            for field, replacement in (('execution', ACTION_SELECTION_CONTRACT),
                                       ('season_controller', {'start_day': 29}),
                                       ('controller_identity', {'contract': 'other-controller'})):
                with self.subTest(field=field):
                    atomic_pickle(path, {**saved, field: replacement})
                    with self.assertRaisesRegex(ValueError, 'execution|continuation|provenance'):
                        checkpoint_metadata(path)


class ContinuationContracts(unittest.TestCase):
    @staticmethod
    def rollout(contract, controller=None):
        shape = (2, 1, 2)
        return Rollout({'features': np.zeros((*shape, 1, 1), np.float32)},
                       np.full(shape, .5, np.float32), np.array([[1., 0.]], np.float32),
                       {'action_selection_contract': contract, 'controller': controller})

    def test_labels_cannot_mix_action_or_controller_continuations(self):
        config = {'gamma': 1., 'gae_lambda': 1., 'action_selection_contract': ECONOMIC_ACTION_SELECTION_CONTRACT}
        accepted = flatten_for_learning(self.rollout(ECONOMIC_ACTION_SELECTION_CONTRACT), config)
        np.testing.assert_allclose(accepted['return'], [1., 0., 1., 0.])
        with self.assertRaisesRegex(ValueError, 'execution|continuation|contract'):
            flatten_for_learning(self.rollout(ACTION_SELECTION_CONTRACT), config)
        hybrid = {**config, 'season_controller': {'start_day': 29}}
        with patch('route_rl.ppo.controllers.contract_identity', return_value={'contract': 'expected-controller'}):
            with self.assertRaisesRegex(ValueError, 'controller|continuation'):
                flatten_for_learning(self.rollout(ECONOMIC_ACTION_SELECTION_CONTRACT,
                                                 {'contract': 'other-controller'}), hybrid)

    def test_old_v1_checkpoint_inference_keeps_unpatched_final_actions(self):
        from route_rl.full_action.catalog import MARKET_SLOTS, UNIT_ACTIONS
        from route_rl.full_action.inference import MAX_OWN_UNITS
        environment = make('kaggriculture', configuration={'seed': 180721}, debug=False)
        observation = deepcopy(dict(environment.state[0].observation))
        observation.update(day=29, hour=22, step=718)
        observation['private']['shed'] = {'WHEAT': 2}
        observation['private']['inventories'] = [{'CARROT': 3}]
        outputs = {'unit_action': np.full((1, MAX_OWN_UNITS, len(UNIT_ACTIONS)), -100., np.float32),
                   'market_action': np.full((1, MARKET_SLOTS, len(MARKET_ACTIONS)), -100., np.float32)}
        outputs['unit_action'][:, :, PASS_ID] = 0
        outputs['unit_action'][0, 0, UNIT_ACTIONS.index(('NORTH',))] = 1
        outputs['market_action'][:, :, NOOP_ID] = 0
        model = JaxModelConfig(d_model=4, layers=1, heads=1, ffn_dim=8, rope_dim=4)
        base = {'contract': POLICY_CONTRACT, 'params': {}, 'policy_sha256': policy_hash({}),
                'model_config': model.to_dict(), 'learning_objective_contract': OBJECTIVE_CONTRACT}
        def fixed_forward(*args, **kwargs):
            return {key: jnp.asarray(value) for key, value in outputs.items()}
        with tempfile.TemporaryDirectory() as directory, patch('route_rl.ppo.inference.policy_forward', fixed_forward):
            path = Path(directory) / 'policy.pkl'
            atomic_pickle(path, {**base, 'action_selection_contract': ACTION_SELECTION_CONTRACT})
            old = load_policy(path, warm=False)
            self.assertEqual(old(observation), sample_action(observation, outputs, None).action)
            self.assertEqual(old(observation)['farmer'], ['NORTH'])
            self.assertTrue(all(order == ['NOOP'] for order in old(observation)['market']))
            atomic_pickle(path, {**base, 'action_selection_contract': ECONOMIC_ACTION_SELECTION_CONTRACT})
            new = load_policy(path, warm=False)
            self.assertEqual(new(observation)['farmer'], ['DROP'])
            self.assertTrue(any(order[0] == 'SELL' for order in new(observation)['market']))


if __name__ == '__main__':
    unittest.main()

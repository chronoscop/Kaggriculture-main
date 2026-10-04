"""Small injected collector fixture tests the critic-to-PPO artifact handoff.

This is an orchestration/learning connectivity test, not measured gameplay data.
The separate collector test runs the actual complete official horizon.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from route_rl.action_bc import initialize, source_identity as bc_source
from route_rl.ppo.pipeline import warmup, train
from route_rl.ppo.provenance import file_hash


class StageHandoff(unittest.TestCase):
    def test_frozen_critic_artifact_initializes_ppo_and_resumes_cursor(self):
        from kaggle_environments import make
        from route_rl.ppo.inference import PolicyHistory
        from route_rl.ppo.rollout import Rollout
        from route_rl.ppo.sampling import PASS_ID, NOOP_ID, MARKET_ACTIONS
        from route_rl.full_action.catalog import UNIT_ACTIONS
        from route_rl.ppo.trainer import checkpoint_metadata

        environment = make('kaggriculture', configuration={'seed': 40001}, debug=False)
        observations = [deepcopy(dict(s.observation)) for s in environment.state]
        pairs = [PolicyHistory(p).encode(observations[p]) for p in (0, 1)]

        def fixture_collect(params, model_config, seeds, sampling_seed, compute_dtype, **options):
            arrays = {key: np.stack([value[key][0] for value in pairs])[None, None]
                      for key in pairs[0]}
            arrays = {key: np.repeat(np.repeat(value, 2, axis=0), len(seeds), axis=1)
                      for key, value in arrays.items()}
            shape = (2, len(seeds), 2)
            arrays['unit_action'] = np.full((*shape, 20), PASS_ID, np.int32)
            arrays['market_action'] = np.full((*shape, 10), NOOP_ID, np.int32)
            arrays['unit_mask'] = np.zeros((*shape, 20), bool)
            arrays['unit_mask'][..., 0] = True
            arrays['market_mask'] = np.ones((*shape, 10), bool)
            arrays['unit_policy_mask'] = arrays['unit_mask'].copy()
            arrays['market_policy_mask'] = arrays['market_mask'].copy()
            arrays['unit_legal_mask'] = np.zeros((*shape, 20, len(UNIT_ACTIONS)), bool)
            arrays['unit_legal_mask'][..., PASS_ID] = True
            arrays['market_legal_mask'] = np.zeros((*shape, 10, len(MARKET_ACTIONS)), bool)
            arrays['market_legal_mask'][..., NOOP_ID] = True
            arrays['old_log_prob'] = np.zeros(shape, np.float32)
            return Rollout(arrays, np.full(shape, .5, np.float32),
                           np.tile(np.array([[1., 0.]], np.float32), (len(seeds), 1)),
                           {'fixture_only': True, 'seeds': list(seeds),
                            'action_selection_contract': options.get('action_selection_contract', 'official-prefix-full-action-ppo-v1')})

        with tempfile.TemporaryDirectory() as directory, patch('route_rl.ppo.rollout.collect_rollout', side_effect=fixture_collect):
            root = Path(directory)
            bc = root / 'bc'
            bc.mkdir()
            initial = bc / 'final_student_jax.pkl'
            initialize('smoke', initial, 0)
            (bc / 'integration.json').write_text(json.dumps({
                'contract': 'public-full-action-bc-v3', 'source': bc_source(),
                'initial_sha256': file_hash(initial), 'cache': {'fixture': True},
                'preparation': {'input': {'known_game_seeds': [200], 'games_without_seed': 0}}}))
            critic_config = {'collection_backend': 'official-python',
                             'action_selection_contract': 'official-prefix-economic-full-action-ppo-v2',
                             'season_controller': None,
                             'games': 1, 'validation_games': 1, 'minibatch_size': 4,
                             'epochs': 1, 'compute_dtype': 'float32', 'seed': 51,
                             'environment_seed_start': 200, 'training_seed_limit': 250,
                             'validation_seed_start': 300, 'validation_seed_limit': 350,
                             'patience': 8, 'learning_rate': 1e-4, 'gamma': 1., 'gae_lambda': .97}
            critic = root / 'critic'
            warmup(NS(initial=initial, bc_run=bc, out=critic, updates=1), critic_config)
            self.assertEqual(checkpoint_metadata(critic / 'latest_critic_state.pkl')['training_seeds'], [201])
            self.assertEqual(checkpoint_metadata(critic / 'latest_critic_state.pkl')['best_update'], 0)
            ppo_config = json.loads(Path('src/route_rl/ppo/configs/ppo.json').read_text())
            ppo_config.update(collection_backend='official-python', games=1, minibatch_size=4, compute_dtype='float32',
                              environment_seed_start=100, training_seed_limit=150, warmup_steps=0)
            ppo = root / 'ppo'
            args = NS(initial=critic / 'policy_with_critic.pkl', teacher=initial, critic_run=critic,
                      out=ppo, updates=1)
            train(args, ppo_config)
            self.assertTrue((ppo / 'policy-update-1.pkl').exists())
            args.updates = 2
            train(args, ppo_config)
            metadata = checkpoint_metadata(ppo / 'latest_ppo_state.pkl')
            self.assertEqual(metadata['training_seeds'], [100, 101])
            self.assertEqual(metadata['env_steps'], 2 * 1438)
            self.assertEqual(metadata['completed_updates'], 2)
            self.assertEqual(json.loads((critic / 'receipt.json').read_text())['actor_unchanged'], True)
            bad = deepcopy(ppo_config)
            bad['teacher_kl'] = .3
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                train(args, bad)


if __name__ == '__main__':
    unittest.main()

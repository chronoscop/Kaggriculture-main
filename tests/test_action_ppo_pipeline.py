"""Seed, provenance and paired comparison invariants for the new branch."""
import json
from pathlib import Path
import tempfile
import unittest

from route_rl.ppo.evaluation import check_panel, score, freeze_inputs
from route_rl.ppo.provenance import (PIPELINE_CONTRACT, bc_provenance, check_seed_ranges,
                                     establish_run, next_seeds, require_training_state)


class PipelineInvariants(unittest.TestCase):
    def test_paired_inputs_freeze_the_bytes_used_by_every_match(self):
        from route_rl.ppo.provenance import file_hash
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            originals = [root / name for name in ('candidate.pkl', 'baseline.pkl', 'opponent.py')]
            for index, path in enumerate(originals):
                path.write_bytes(f'original-{index}'.encode())
            snapshots = root / 'frozen'
            snapshots.mkdir()
            frozen, hashes = freeze_inputs(*originals, snapshots)
            for original, name in zip(originals, ('policy', 'baseline', 'opponent')):
                self.assertEqual(file_hash(frozen[name]), hashes[f'{name}_sha256'])
                original.write_bytes(b'replaced-by-live-training')
                self.assertNotEqual(file_hash(original), hashes[f'{name}_sha256'])
                self.assertEqual(file_hash(frozen[name]), hashes[f'{name}_sha256'])

    def test_missing_optimizer_state_does_not_restart_completed_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            require_training_state(out, 'latest_ppo_state.pkl', 'ppo')
            artifact = out / 'policy-update-1.pkl'
            artifact.write_bytes(b'completed-candidate')
            with self.assertRaisesRegex(ValueError, 'restore the state'):
                require_training_state(out, 'latest_ppo_state.pkl', 'ppo')
            self.assertEqual(artifact.read_bytes(), b'completed-candidate')
            (out / 'latest_ppo_state.pkl').write_bytes(b'optimizer-state')
            require_training_state(out, 'latest_ppo_state.pkl', 'ppo')

    def test_seed_cursor_skips_demonstrations_and_is_resume_deterministic(self):
        first, cursor = next_seeds(1, 3, 20, {1, 4, 8})
        self.assertEqual(first, [2, 3, 5])
        second, final = next_seeds(cursor, 2, 20, {1, 4, 8})
        self.assertEqual(first + second, next_seeds(1, 5, 20, {1, 4, 8})[0])
        self.assertEqual(final, 8)
        with self.assertRaisesRegex(ValueError, 'exhausted'):
            next_seeds(1, 3, 4, {2})

    def test_critic_training_validation_and_ppo_ranges_cannot_overlap(self):
        check_seed_ranges({'ppo': (1, 10), 'critic': (10, 20), 'validation': (20, 30)})
        for ranges in ({'ppo': (1, 11), 'critic': (10, 20)}, {'ppo': (0, 10)}, {'ppo': (10, 10)}):
            with self.subTest(ranges=ranges), self.assertRaises(ValueError):
                check_seed_ranges(ranges)

    def test_panel_excludes_actual_data_and_future_training_and_screening(self):
        identity = {'bc': {'known_game_seeds': [100]},
                    'critic_training_seeds': [200], 'critic_validation_seeds': [300],
                    'reserved_training_ranges': [(1, 50)]}
        metadata = {'training_seeds': [400]}
        for seed in (100, 200, 300, 400, 20):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                check_panel([seed], identity, metadata, None)
        with self.assertRaisesRegex(ValueError, 'independent'):
            check_panel([500], identity, metadata, {'seeds': [500]})
        check_panel([501, 502], identity, metadata, {'seeds': [500]})

    def test_positive_cash_difference_is_only_converted_to_final_match_score(self):
        self.assertEqual(score(3, 2), 1)
        self.assertEqual(score(30_000, 20_000), 1)
        self.assertEqual(score(0, 0), .5)
        self.assertEqual(score(2, 3), 0)
        with self.assertRaises(ValueError):
            score(None, 2)

    def test_resume_requires_identical_identity_and_cumulative_target(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            identity = {'contract': PIPELINE_CONTRACT, 'teacher': 'fixed', 'objective': 'terminal-score'}
            establish_run(out, identity, {'batch': 2}, 3)
            establish_run(out, identity, {'batch': 2}, 4)
            with self.assertRaisesRegex(ValueError, 'decrease'):
                establish_run(out, identity, {'batch': 2}, 2)
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                establish_run(out, {**identity, 'teacher': 'changed'}, {'batch': 2}, 5)

    def test_bc_lineage_cannot_silently_use_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'integration.json').write_text(json.dumps({'contract': 'public-full-action-bc-v3',
                                                            'source': {'files_sha256': 'wrong'}}))
            with self.assertRaisesRegex(ValueError, 'BC source/contract changed'):
                bc_provenance(run)


class BCPolicyMembership(unittest.TestCase):
    @staticmethod
    def fixture(run):
        from route_rl.action_bc import CONTRACT, source_identity
        from route_rl.full_action.checkpoints import save_params_payload
        import numpy as np

        run.mkdir()
        (run / 'integration.json').write_text(json.dumps({
            'contract': CONTRACT, 'source': source_identity(), 'initial_sha256': 'initial-source',
            'cache': {'index_sha256': 'cache-index'},
            'preparation': {'input': {'known_game_seeds': [73], 'games_without_seed': 0}}}))
        save_params_payload(run / 'epoch-1-policy.pkl', {'weights': np.array([1., 2.])}, {}, 0)

    def test_byte_identical_external_copy_belongs_to_the_named_run(self):
        from route_rl.ppo.provenance import file_hash, verify_bc_policy
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'bc'
            self.fixture(run)
            copied = root / 'frozen.pkl'
            copied.write_bytes((run / 'epoch-1-policy.pkl').read_bytes())
            result = verify_bc_policy(run, copied)
            self.assertEqual(result['policy_file_sha256'], file_hash(copied))
            self.assertEqual(result['bc_integration_sha256'], file_hash(run / 'integration.json'))
            copied.write_bytes(b'another BC run policy')
            with self.assertRaisesRegex(ValueError, 'specified BC run'):
                verify_bc_policy(run, copied)

    def test_only_final_and_numeric_epoch_artifacts_prove_membership(self):
        from route_rl.ppo.provenance import verify_bc_policy
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'bc'
            self.fixture(run)
            original = (run / 'epoch-1-policy.pkl').read_bytes()
            (run / 'epoch-1-policy.pkl').unlink()
            copied = root / 'frozen.pkl'
            copied.write_bytes(original)
            for name in ('initial.pkl', 'epoch-latest-policy.pkl', 'epoch-2-state.pkl'):
                (run / name).write_bytes(original)
            with self.assertRaisesRegex(ValueError, 'specified BC run'):
                verify_bc_policy(run, copied)

    def test_final_receipt_checks_policy_initial_and_cache_identities(self):
        from route_rl.full_action.checkpoints import load_training_source
        from route_rl.ppo.provenance import verify_bc_policy
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'bc'
            self.fixture(run)
            final = run / 'final_student_jax.pkl'
            final.write_bytes((run / 'epoch-1-policy.pkl').read_bytes())
            payload = load_training_source(final)
            receipt = {'initial_sha256': 'initial-source', 'cache_sha256': 'cache-index',
                       'policy_sha256': payload['policy_sha256']}
            path = run / 'receipt.json'
            path.write_text(json.dumps(receipt))
            verify_bc_policy(run, final)
            for field in ('initial_sha256', 'cache_sha256', 'policy_sha256'):
                path.write_text(json.dumps({**receipt, field: 'wrong'}))
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'receipt'):
                    verify_bc_policy(run, final)

    def test_old_epoch_membership_remains_stable_after_a_new_final_appears(self):
        from route_rl.full_action.checkpoints import load_training_source, save_params_payload
        from route_rl.ppo.provenance import verify_bc_policy
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'bc'
            self.fixture(run)
            old_epoch = run / 'epoch-1-policy.pkl'
            before = verify_bc_policy(run, old_epoch)
            final = run / 'final_student_jax.pkl'
            save_params_payload(final, {'weights': np.array([3., 4.])}, {}, 0)
            (run / 'receipt.json').write_text(json.dumps({
                'initial_sha256': 'initial-source', 'cache_sha256': 'cache-index',
                'policy_sha256': load_training_source(final)['policy_sha256']}))
            self.assertEqual(before, verify_bc_policy(run, old_epoch))


if __name__ == '__main__':
    unittest.main()

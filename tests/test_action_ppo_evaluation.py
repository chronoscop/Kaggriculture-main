"""Paired report bookkeeping uses frozen inputs and independent confirmation."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

from route_rl.ppo.evaluation import evaluate, AgentProcess
from route_rl.ppo.provenance import (PIPELINE_CONTRACT, OBJECTIVE_CONTRACT, file_hash,
                                     source_identity, write_json)


class PairedEvaluation(unittest.TestCase):
    def test_real_worker_uses_frozen_entry_with_original_resource_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / 'main.py'
            original.write_text("raise RuntimeError('live original was replaced')\n")
            (root / 'helper.py').write_text("operation = 'PASS'\n")
            (root / 'resource.txt').write_text('PASS')
            snapshots = root / 'frozen'
            snapshots.mkdir()
            frozen = snapshots / 'opponent.py'
            frozen.write_text("from pathlib import Path\nfrom helper import operation\n"
                              "def agent(observation, configuration):\n"
                              "    assert (Path(__file__).parent / 'resource.txt').read_text() == operation\n"
                              "    return {'farmer': [operation], 'hands': [], 'market': []}\n")
            worker = AgentProcess('entry', frozen, source_context=original)
            try:
                self.assertEqual(worker({}, {})['farmer'], ['PASS'])
            finally:
                worker.close()

    def test_frozen_pairs_confirmation_and_changed_inputs_are_rejected(self):
        # Environment and workers are injected: real complete engine/entry
        # execution is separately tested in sampler and policy bundle checks.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'run'
            run.mkdir()
            policy, baseline, opponent = [root / name for name in ('candidate.pkl', 'baseline.pkl', 'main.py')]
            policy.write_bytes(b'frozen-candidate')
            baseline.write_bytes(b'frozen-baseline')
            opponent.write_bytes(b'frozen-opponent-entry')
            identity = {'contract': PIPELINE_CONTRACT, 'phase': 'ppo', 'source': source_identity(),
                        'bc': {'known_game_seeds': [], 'games_without_seed': 0},
                        'teacher_sha256': file_hash(baseline), 'initial_sha256': 'critic-start',
                        'reserved_training_ranges': [[1, 10]]}
            write_json(run / 'integration.json', identity)
            receipt_hash = file_hash(run / 'integration.json')
            seen = []

            class Worker:
                def __init__(self, kind, path, source_context=None):
                    self.kind, self.name = kind, path.stem
                    self.data = path.read_bytes()
                    seen.append((kind, self.name, self.data))
                def close(self):
                    pass

            class Environment:
                def run(self, workers):
                    seat = 0 if workers[0].kind == 'policy' else 1
                    own = workers[seat]
                    rewards = [0., 0.]
                    if own.name == 'policy':
                        rewards[seat] = 1.
                    self.steps = [None] * 719 + [[NS(status='DONE', reward=v) for v in rewards]]

            args = NS(run=run, policy=policy, baseline=baseline, opponent=opponent,
                      games=2, seed=100, phase='screen', screen_report=None, out=root / 'screen.json')
            with patch('route_rl.ppo.trainer.checkpoint_metadata', return_value={'training_seeds': []}), \
                 patch('route_rl.full_action.checkpoints.load_training_source', return_value={'run_integration_sha256': receipt_hash}), \
                 patch('route_rl.ppo.evaluation.AgentProcess', Worker), \
                 patch('kaggle_environments.make', side_effect=lambda *a, **kw: Environment()):
                screen = evaluate(args)
                self.assertEqual(screen['total_games'], 4)
                self.assertEqual(screen['paired_score_delta'], .5)
                self.assertEqual(screen['objective'], OBJECTIVE_CONTRACT)
                self.assertFalse(screen['eligible_for_manual_review'])
                self.assertEqual([row['seat'] for row in screen['games']], [0, 0, 1, 1])
                args.phase = 'confirmation'
                args.screen_report = root / 'screen.json'
                args.out = root / 'confirm.json'
                with self.assertRaisesRegex(ValueError, 'independent'):
                    evaluate(args)
                args.seed = 200
                confirmed = evaluate(args)
                self.assertTrue(confirmed['eligible_for_manual_review'])
                self.assertEqual(confirmed['deployment'], 'candidate_only')
                args.out = root / 'changed.json'
                policy.write_bytes(b'changed-candidate')
                with self.assertRaisesRegex(ValueError, 'changed since screening'):
                    evaluate(args)
                self.assertFalse(args.out.exists())
                # A controller wrapper preserves the parent run hash but must
                # never be silently reported as a neural learning comparison.
                with patch('route_rl.full_action.checkpoints.load_training_source', return_value={
                    'run_integration_sha256': receipt_hash,
                    'candidate_origin': 'frozen-controller-adaptation-v1'}):
                    with self.assertRaisesRegex(ValueError, '--comparison controller'):
                        evaluate(args)
                with patch('route_rl.full_action.checkpoints.load_training_source', return_value={
                    'run_integration_sha256': receipt_hash,
                    'action_selection_contract': 'different-execution'}):
                    with self.assertRaisesRegex(ValueError, 'execution/continuation'):
                        evaluate(args)
            self.assertTrue(all(data == b'frozen-candidate' for kind, name, data in seen if name == 'policy'))
            self.assertTrue(all(data == b'frozen-baseline' for kind, name, data in seen if name == 'baseline'))
            self.assertTrue(all(data == b'frozen-opponent-entry' for kind, name, data in seen if kind == 'entry'))


if __name__ == '__main__':
    unittest.main()

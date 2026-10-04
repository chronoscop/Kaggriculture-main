"""Frozen-model controller comparisons and deployment asset integrity."""
from pathlib import Path
from types import SimpleNamespace as NS
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest

import jax
import numpy as np

from route_rl.action_bc import source_identity as bc_source
from route_rl.full_action.checkpoints import load_training_source, save_params_payload
from route_rl.full_action.model import JaxModelConfig, initialize_params
from route_rl.packaging import package_submission
from route_rl.ppo.candidate import create_candidate
from route_rl.ppo.controllers import ASSET_ROOT
from route_rl.ppo.evaluation import controller_comparison_lineage, check_panel
from route_rl.ppo.provenance import file_hash, write_json
from route_rl.ppo.sampling import ECONOMIC_ACTION_SELECTION_CONTRACT

ROOT = Path(__file__).resolve().parents[1]


class FrozenControllerCandidates(unittest.TestCase):
    def fixture(self, root):
        run = root / 'bc'
        run.mkdir()
        base = run / 'epoch-1-policy.pkl'
        config = JaxModelConfig(d_model=16, layers=1, heads=1, ffn_dim=32, rope_dim=16)
        params = initialize_params(jax.random.PRNGKey(613), config)
        save_params_payload(base, params, config.to_dict(), 0)
        identity = {'contract': 'public-full-action-bc-v3', 'source': bc_source(),
                    'initial_sha256': 'fixture-initial', 'cache': {},
                    'preparation': {'input': {'known_game_seeds': [11, 12], 'games_without_seed': 0}}}
        write_json(run / 'integration.json', identity)
        return run, base, identity

    def test_candidate_keeps_weights_and_requires_same_owned_frozen_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, base, identity = self.fixture(root)
            original = base.read_bytes()
            candidate = root / 'economic.pkl'
            create_candidate(base, candidate, action_selection_contract=ECONOMIC_ACTION_SELECTION_CONTRACT)
            self.assertEqual(base.read_bytes(), original)
            payload = load_training_source(candidate)
            self.assertEqual(payload['policy_sha256'], load_training_source(base)['policy_sha256'])
            frozen = {'policy': candidate, 'baseline': base}
            hashes = {'baseline_sha256': file_hash(base)}
            lineage, metadata = controller_comparison_lineage(NS(run=run), identity, frozen, hashes)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                check_panel([11], lineage, metadata, None)
            with self.assertRaises(FileExistsError):
                create_candidate(base, candidate, action_selection_contract=ECONOMIC_ACTION_SELECTION_CONTRACT)
            unowned = root / 'unowned.pkl'
            save_params_payload(unowned, initialize_params(jax.random.PRNGKey(614),
                                JaxModelConfig(**payload['model_config'])), payload['model_config'], 0)
            with self.assertRaisesRegex(ValueError, 'identical frozen weights'):
                controller_comparison_lineage(NS(run=run), identity,
                                             {'policy': candidate, 'baseline': unowned}, hashes)

    @unittest.skipUnless((ASSET_ROOT / 'terminal_search.so').exists(), 'build owned search first')
    def test_packaged_hybrid_executes_the_same_final_day_controller_in_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, base, identity = self.fixture(root)
            candidate = root / 'hybrid.pkl'
            create_candidate(base, candidate, action_selection_contract=ECONOMIC_ACTION_SELECTION_CONTRACT,
                             season_controller={'search_seconds': .001})
            archive = root / 'hybrid.tar.gz'
            manifest = package_submission(candidate, archive, root=ROOT)
            self.assertIn('route_rl/season_search/terminal_search.so', manifest['files'])
            self.assertFalse(any('kaggriculture-solution' in path for path in manifest['files']))
            bundle = root / 'bundle'
            bundle.mkdir()
            with tarfile.open(archive) as packed:
                packed.extractall(bundle, filter='data')
            script = root / 'isolated.py'
            script.write_text('''import json, runpy, sys
from pathlib import Path
root = Path(sys.argv[1]); sys.path.insert(0, str(root))
entry = runpy.run_path(str(root / "main.py"))
from kaggle_environments import make
import route_rl
assert str(root) in route_rl.__file__
environment = make("kaggriculture", configuration={"seed": 614}, debug=False)
# Advance with actual official actions; only invoke the packaged agent near
# handoff so this isolates deployment assets rather than measuring policy.
for step in range(719):
    observation = dict(environment.state[0].observation)
    own = entry["agent"](observation, {"__raw_path__": str(root / "main.py")}) if step >= 695 else {"farmer": ["PASS"]}
    environment.step([own, {"farmer": ["PASS"]}])
policy = entry["agent"].__globals__["_policy"]
assert policy.controller.diagnostics["handoff_actions"] == 23, policy.controller.diagnostics
assert policy.controller.diagnostics["fallback_actions"] == 0, policy.controller.diagnostics
assert len(environment.steps) == 720
assert [state.status for state in environment.state] == ["DONE", "DONE"]
print(json.dumps({"isolated_hybrid": True, "handoff_actions": 23}))
''')
            env = {**os.environ, 'JAX_PLATFORMS': 'cpu', 'CUDA_VISIBLE_DEVICES': '',
                   'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'OMP_NUM_THREADS': '2'}
            env.pop('PYTHONPATH', None)
            result = subprocess.run([sys.executable, '-I', str(script), str(bundle)],
                                    cwd=root, env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('"isolated_hybrid": true', result.stdout)


if __name__ == '__main__':
    unittest.main()

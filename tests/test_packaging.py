"""Release isolation, integrity and actual packaged-entrypoint checks."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from route_rl.packaging import (MANIFEST_NAME, PPO_ACTION_CONTRACT, _policy_module,
                                package_submission, source_release, verify_archive)

PROJECT = Path(__file__).resolve().parents[1]


class SourceReleaseTests(unittest.TestCase):
    def fixture(self, root):
        contents = {
            "pyproject.toml": "[project]\nname='test'\nversion='0.1.0'\n",
            "README.md": "Owned pipeline\n", "AGENTS.md": "Preserve accepted policies\n",
            "src/route_rl/__init__.py": "", "src/route_rl/ppo/new_pipeline.py": "NEW = 1\n",
            "tools/package_pipeline.py": "", "docs/action_ppo_sources.md": "Attribution\n",
            "native/Cargo.toml": "", "native/src/lib.rs": "", "native/LICENSE.txt": "Apache\n",
            "third_party/kaggriculture-simulation/LICENSE": "Apache\n",
            "third_party/kaggriculture-simulation/src-rust/kagg-engine/src/lib.rs": "",
            "runs/private.json": "private", "data/replays.json": "private",
            "kaggriculture-solution/python/secret.py": "private", "src/route_rl/.env": "private",
            "src/route_rl/credentials.json": "private", "src/route_rl/secrets/key.json": "private",
            "native/target/compiled.json": "private", "tools/.venv/lib/hidden.py": "private",
            "src/route_rl/__pycache__/bad.py": "private", "src/route_rl/checkpoint.pkl": "private",
        }
        for name, data in contents.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data)

    def test_deterministic_owned_source_includes_future_pipeline_and_licenses(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            source_release(root, root / "first.tar.gz")
            source_release(root, root / "second.tar.gz")
            self.assertEqual((root / "first.tar.gz").read_bytes(), (root / "second.tar.gz").read_bytes())
            manifest = verify_archive(root / "first.tar.gz")
            inventory = manifest["files"]
            self.assertIn("src/route_rl/ppo/new_pipeline.py", inventory)
            self.assertIn("AGENTS.md", inventory)
            self.assertIn("native/LICENSE.txt", inventory)
            self.assertIn("third_party/kaggriculture-simulation/LICENSE", inventory)
            self.assertFalse(manifest["reference_checkout_required"])
            self.assertFalse(any("private" in data for data in self.archive_contents(root / "first.tar.gz").values()))
            with self.assertRaises(FileExistsError):
                source_release(root, root / "first.tar.gz")
            self.assertEqual((root / "first.tar.gz").read_bytes(), (root / "second.tar.gz").read_bytes())

    @staticmethod
    def archive_contents(path):
        with tarfile.open(path, "r:gz") as archive:
            return {member.name: archive.extractfile(member).read().decode() for member in archive.getmembers()}

    def test_symlink_cannot_pull_an_outside_file_into_a_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            (root / "src/route_rl/leak.py").symlink_to(root / "runs/private.json")
            with self.assertRaisesRegex(ValueError, "symlink"):
                source_release(root, root / "release.tar.gz")
            self.assertFalse((root / "release.tar.gz").exists())

    def test_tampered_archive_fails_hash_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            source_release(root, root / "original.tar.gz")
            with tarfile.open(root / "original.tar.gz", "r:gz") as original:
                files = {member.name: original.extractfile(member).read() for member in original.getmembers()}
            files["README.md"] = b"tampered"
            with tarfile.open(root / "tampered.tar.gz", "w:gz") as archive:
                for name, data in files.items():
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_archive(root / "tampered.tar.gz")

    def test_policy_execution_routes_by_contract_and_rejects_unknown_contract(self):
        self.assertEqual(_policy_module({}), "route_rl.full_action.inference")
        self.assertEqual(_policy_module({"action_selection_contract": PPO_ACTION_CONTRACT,
                                        "learning_objective_contract": "terminal-match-score-1-0.5-0-v1"}),
                         "route_rl.ppo.inference")
        with self.assertRaisesRegex(ValueError, "learning objective contract"):
            _policy_module({"action_selection_contract": PPO_ACTION_CONTRACT})
        with self.assertRaisesRegex(ValueError, "learning objective contract"):
            _policy_module({"action_selection_contract": PPO_ACTION_CONTRACT,
                            "learning_objective_contract": "cash-margin-v0"})
        with self.assertRaisesRegex(ValueError, "unsupported"):
            _policy_module({"action_selection_contract": "future-unknown-v1"})


class SubmissionTests(unittest.TestCase):
    def test_packaged_bc_policy_executes_from_an_isolated_directory(self):
        self.assert_isolated_execution(ppo=False)

    def test_packaged_ppo_keeps_compute_dtype_and_executes_its_own_contract(self):
        self.assert_isolated_execution(ppo=True)

    def assert_isolated_execution(self, *, ppo):
        # A deliberately tiny random policy tests deployment connectivity only.
        import jax
        from route_rl.full_action.checkpoints import atomic_pickle, load_training_source, save_params_payload
        from route_rl.full_action.model import JaxModelConfig, initialize_params
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = JaxModelConfig(d_model=16, layers=1, heads=1, ffn_dim=32, rope_dim=16)
            params = initialize_params(jax.random.PRNGKey(73), config)
            save_params_payload(root / "smoke-policy.pkl", params, config.to_dict(), 0)
            if ppo:
                payload = load_training_source(root / "smoke-policy.pkl")
                payload.update(action_selection_contract=PPO_ACTION_CONTRACT, compute_dtype="bfloat16",
                               learning_objective_contract="terminal-match-score-1-0.5-0-v1")
                atomic_pickle(root / "smoke-policy.pkl", payload)
            original = (root / "smoke-policy.pkl").read_bytes()
            manifest = package_submission(root / "smoke-policy.pkl", root / "candidate.tar.gz", root=PROJECT)
            self.assertEqual(original, (root / "smoke-policy.pkl").read_bytes())
            self.assertEqual(manifest["deployment"], "candidate_only")
            expected_module = "route_rl.ppo.inference" if ppo else "route_rl.full_action.inference"
            self.assertEqual(manifest["policy_execution_module"], expected_module)
            self.assertNotIn("kaggriculture-solution", " ".join(manifest["files"]))
            bundle = root / "bundle"
            bundle.mkdir()
            with tarfile.open(root / "candidate.tar.gz", "r:gz") as archive:
                archive.extractall(bundle, filter="data")
            import pickle
            exported = pickle.loads((bundle / "policy.pkl").read_bytes())
            self.assertNotIn("state", exported)
            if ppo:
                self.assertEqual(exported["compute_dtype"], "bfloat16")
                self.assertEqual(exported["action_selection_contract"], PPO_ACTION_CONTRACT)
            script = root / "isolated.py"
            script.write_text('''import json, runpy, sys\nfrom pathlib import Path\nroot = Path(sys.argv[1])\nsys.path.insert(0, str(root))\nentry = runpy.run_path(str(root / "main.py"))\nfrom kaggle_environments import make\nimport route_rl\nassert str(root) in route_rl.__file__, route_rl.__file__\nenvironment = make("kaggriculture", configuration={"seed": 73}, debug=False)\nobservation = dict(environment.state[0].observation)\naction = entry["agent"](observation, {"__raw_path__": str(root / "main.py")})\nassert set(action) == {"farmer", "hands", "market"}\nenvironment.step([action, {"farmer": ["PASS"], "hands": [], "market": []}])\nassert environment.state[0].status == "ACTIVE", environment.state[0].status\nprint(json.dumps({"isolated": True, "player_status": environment.state[0].status}))\n''')
            environment = {**os.environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "",
                           "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
            environment.pop("PYTHONPATH", None)
            result = subprocess.run([sys.executable, "-I", str(script), str(bundle)], cwd=root,
                                    env=environment, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('"isolated": true', result.stdout)


if __name__ == "__main__":
    unittest.main()

"""Export and relocated stdlib-only inference tests."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from route_rl.check import fixture
from route_rl.features import SCHEMA, menu_features, state_features
from route_rl.routes import PlanningState
from route_rl.submission import export, MAX_ARCHIVE
from route_rl.submission_check import unpack, Probe
from route_rl.deployment import NativePolicy

CAN_EXPORT = (importlib.util.find_spec("torch") is not None and shutil.which("cc")
              and sys.platform == "linux" and platform.machine() == "x86_64")

@unittest.skipUnless(CAN_EXPORT, "export tests require PyTorch and a Linux x86_64 C compiler")
class SubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from route_rl.training import Network
        torch.set_num_threads(1)
        torch.manual_seed(511)
        cls.temporary = tempfile.TemporaryDirectory(prefix="route-export-test-")
        cls.root = Path(cls.temporary.name)
        cls.model = Network(torch)
        cls.checkpoint = cls.root / "checkpoint.pt"
        torch.save(dict(network=cls.model.module.state_dict(), schema=SCHEMA,
            engine="kaggle-environments==1.32.7", takeover=0, horizon=24,
            replan_interval=6, iteration=1), cls.checkpoint)
        cls.archive = cls.root / "submission.tar.gz"
        cls.report = export(cls.checkpoint, cls.archive)
        cls.package = cls.root / "relocated path" / "kaggle_simulations" / "agent"
        unpack(cls.archive, cls.package)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_small_policy_only_archive(self):
        self.assertLess(self.report["compressed_bytes"], MAX_ARCHIVE)
        with tarfile.open(self.archive) as tar:
            names = tar.getnames()
        self.assertIn("main.py", names)
        self.assertIn("_route_submission/weights.bin", names)
        self.assertNotIn("baseline.py", names)
        self.assertFalse(any("torch" in name or name.endswith(".pt") or "simulator" in name for name in names))
        metadata = json.loads((self.package / "_route_submission/metadata.json").read_text())
        self.assertEqual(metadata["parameters"], 471681)
        self.assertEqual(metadata["planning_seconds"], 0.65)

    def test_native_matches_pytorch_on_actual_planning_states(self):
        import torch
        native = NativePolicy(self.package / "_route_submission")
        plan = PlanningState(fixture())
        for _ in range(3):
            jobs = plan.candidates(0)
            features, mask = menu_features(plan, jobs)
            state = state_features(plan)
            with torch.no_grad():
                expected, _ = self.model(torch.tensor([features]),
                                         torch.tensor([mask]), torch.tensor([state]))
            actual = torch.tensor(native.scores(features, state))
            self.assertTrue(torch.allclose(expected[0], actual, atol=2e-5, rtol=2e-5))
            self.assertEqual(int(expected.argmax()), int(actual.argmax()))
            next_job = next(j for j in jobs if j.op[0] == "BUY_SEED")
            plan.commit(next_job)

    def test_relocated_no_site_packages_agent_and_episode_reset(self):
        with Probe(self.package / "main.py", self.root) as probe:
            self.assertEqual(probe.startup["imported_heavy_libraries"], [])
            self.assertEqual(probe.startup["cpu_affinity"], 1)
            first = probe.act(fixture())
            again = probe.act(fixture())
            self.assertEqual(first["action"], again["action"])
            self.assertEqual(set(first["action"]), {"farmer", "hands", "market"})

    def test_kaggle_style_exec_without_dunder_file(self):
        # The repository's Kaggle-compatible loader executes without __file__.
        # It places the submission directory on sys.path.
        code = (
            "import json,sys; from pathlib import Path; "
            "root=Path(sys.argv[1]); sys.path.insert(0,str(root)); ns={}; "
            "exec(compile((root/'main.py').read_text(),str(root/'main.py'),'exec'),ns); "
            "entry=[v for v in ns.values() if callable(v)][-1]; "
            "print(json.dumps(entry(json.loads(sys.argv[2]))))")
        result = subprocess.run([sys.executable, "-I", "-S", "-c", code,
            str(self.package), json.dumps(fixture())], cwd=self.root,
            check=True, capture_output=True, text=True)
        self.assertEqual(set(json.loads(result.stdout)), {"farmer", "hands", "market"})

    def test_invalid_weights_do_not_replace_existing_output(self):
        import torch
        checkpoint = torch.load(self.checkpoint, weights_only=False)
        checkpoint["network"]["encoder.0.weight"][0, 0] = float("nan")
        bad = self.root / "bad.pt"
        torch.save(checkpoint, bad)
        before = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        with self.assertRaises(ValueError):
            export(bad, self.archive)
        self.assertEqual(before, hashlib.sha256(self.archive.read_bytes()).hexdigest())

    def test_legacy_and_warmup_checkpoints_rejected(self):
        import torch
        for change in ({"schema": "v1"}, {"takeover": 144}):
            checkpoint = torch.load(self.checkpoint, weights_only=False)
            checkpoint.update(change)
            bad = self.root / "unsupported.pt"
            torch.save(checkpoint, bad)
            with self.assertRaises(ValueError):
                export(bad, self.root / "unsupported.tar.gz")

    def test_weight_corruption_is_detected(self):
        target = self.root / "corrupt"
        shutil.copytree(self.package / "_route_submission", target)
        raw = bytearray((target / "weights.bin").read_bytes())
        raw[0] ^= 1
        (target / "weights.bin").write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "checksum"):
            NativePolicy(target)

    def test_planning_budget_ends_extension(self):
        native = NativePolicy(self.package / "_route_submission")
        native.deadline = -1
        plan = PlanningState(fixture())
        self.assertEqual(native.choose(plan, plan.candidates(0)), 0)
        self.assertEqual(native.budget_cutoffs, 1)

class ArchiveSafetyTests(unittest.TestCase):
    def test_unsafe_archive_entries_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.tar.gz"
            with tarfile.open(path, "w:gz") as tar:
                for name in ("main.py", "../escape.py"):
                    member = tarfile.TarInfo(name)
                    member.size = 1
                    tar.addfile(member, io.BytesIO(b"x"))
            with self.assertRaises(ValueError):
                unpack(path, Path(tmp) / "unpacked")
            self.assertFalse((Path(tmp) / "escape.py").exists())

if __name__ == "__main__":
    unittest.main()

"""Bounded-memory loading preserves complete trajectories and validation isolation."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from route_rl.action_bc import (CONTRACT, COMPATIBLE_CACHE_ENCODINGS,
                                cache_compatibility, source_identity, train)
from route_rl.full_action.dataset import ReplayDataset
from route_rl.full_action.memory import host_memory_status


def make_cache(root: Path, lengths=(1, 5, 2, 7, 3)) -> int:
    root.mkdir(exist_ok=True)
    rows, start = [], 0
    for game, length in enumerate(lengths):
        features = np.zeros((length, 264, 124), np.float16)
        labels = np.full((length, 30), -100, np.int16)
        features[:, 0, 0] = np.arange(start, start + length)
        labels[:, 0] = labels[:, 20] = np.arange(start, start + length)
        np.savez_compressed(root / f"{game}.npz", features=features, labels=labels)
        rows.append({"key": str(game), "split": "train", "samples": length})
        start += length
    (root / "index.json").write_text(json.dumps({"episodes": rows}))
    return start


def epoch_rows(dataset: ReplayDataset, seed=None, size=4):
    rng = np.random.default_rng(seed) if seed is not None else None
    ids, counts, maximum = [], [], 0
    for batch, count in dataset.batches(size, rng):
        ids.extend(map(int, batch["features"][:count, 0, 0]))
        counts.append(count)
        maximum = max(maximum, dataset.resident_trajectories)
        np.testing.assert_array_equal(batch["features"][:count, 0, 0], batch["unit_action"][:count, 0])
        np.testing.assert_array_equal(batch["features"][:count, 0, 0], batch["market_action"][:count, 0])
    return ids, counts, maximum


class StreamingReplayChecks(unittest.TestCase):
    def test_initialization_reads_headers_without_loading_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            count = make_cache(root)
            with patch("route_rl.full_action.dataset.np.load", side_effect=AssertionError("full array load")):
                dataset = ReplayDataset(root, "train", 1, 0, window=2)
            self.assertEqual(dataset.size, count)
            self.assertEqual(dataset.resident_trajectories, 0)

    def test_shuffled_windows_visit_every_row_once_and_only_pad_the_epoch_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            count = make_cache(root)
            dataset = ReplayDataset(root, "train", 1, 0, window=2)
            ids, counts, maximum = epoch_rows(dataset, seed=23)
            self.assertEqual(sorted(ids), list(range(count)))
            self.assertEqual(counts, [4, 4, 4, 4, 2])
            self.assertLessEqual(maximum, 4)
            self.assertEqual(dataset.resident_trajectories, 0)
            self.assertEqual(epoch_rows(dataset, seed=23)[0], ids)
            self.assertNotEqual(epoch_rows(dataset, seed=24)[0], ids)

    def test_sequential_validation_and_random_batch_keep_feature_label_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            count = make_cache(root)
            dataset = ReplayDataset(root, "train", 1, 0, window=1)
            ids, _, maximum = epoch_rows(dataset)
            self.assertEqual(ids, list(range(count)))
            self.assertLessEqual(maximum, 2)
            selected = np.array([10, 0, 7, 1, 17, 1])
            expected = np.resize(selected, 8)
            batch = dataset.batch(selected, 8)
            np.testing.assert_array_equal(batch["features"][:, 0, 0], expected)
            np.testing.assert_array_equal(batch["unit_action"][:, 0], expected)
            np.testing.assert_array_equal(batch["market_action"][:, 0], expected)
            self.assertLessEqual(dataset.resident_trajectories, 2)

    def test_closing_an_interrupted_epoch_releases_resident_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_cache(root)
            dataset = ReplayDataset(root, "train", 1, 0, window=2)
            batches = dataset.batches(4)
            next(batches)
            self.assertGreater(dataset.resident_trajectories, 0)
            batches.close()
            self.assertEqual(dataset.resident_trajectories, 0)

    def test_old_cache_reuse_is_explicit_and_rejects_encoding_changes(self):
        digest = next(iter(COMPATIBLE_CACHE_ENCODINGS))
        preparation = {"contract": CONTRACT, "source": {
            "implementation": "route_rl.full_action", "reference_checkout_required": False, "files_sha256": digest}}
        with self.assertRaisesRegex(ValueError, "reuse-compatible-cache"):
            cache_compatibility(preparation, False)
        self.assertEqual(cache_compatibility(preparation, True)["mode"], "explicit-compatible-cache-reuse")
        with patch("route_rl.action_bc.encoding_identity", return_value="changed-labels"):
            with self.assertRaises(ValueError):
                cache_compatibility(preparation, True)
        preparation["contract"] = "older-contract"
        with self.assertRaisesRegex(ValueError, "incompatible cache contract"):
            cache_compatibility(preparation, True)

    def test_training_checks_effective_container_memory_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            (cache / "pipeline.json").write_text(json.dumps({"contract": CONTRACT, "source": source_identity()}))
            args = NS(cache=cache, initial=root / "initial.pkl", out=root / "run", epochs=1,
                      batch_size=320, compute_dtype="bfloat16", shuffle_window=16, reuse_compatible_cache=False)
            memory = {"host_available_bytes": 500 * 2**30, "effective_available_bytes": 2**30,
                      "cgroup_limits": []}
            with patch("route_rl.action_bc.audit_cache", return_value={"decoded_bytes": 87 * 2**30,
                                                                       "largest_trajectory_bytes": 47 * 2**20}), \
                    patch("route_rl.full_action.memory.host_memory_status", return_value=memory), \
                    patch("route_rl.action_bc.run_training") as launch:
                with self.assertRaisesRegex(ValueError, "effective available memory"):
                    train(args)
                launch.assert_not_called()


class ContainerMemoryChecks(unittest.TestCase):
    def test_v1_limit_overrides_large_host_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc, cgroup = root / "proc", root / "cgroup"
            (proc / "self").mkdir(parents=True)
            (cgroup / "memory").mkdir(parents=True)
            (proc / "meminfo").write_text("MemAvailable: 500000000 kB\n")
            (proc / "self/cgroup").write_text("11:memory:/docker/test\n")
            (cgroup / "memory/memory.limit_in_bytes").write_text("50000000000\n")
            (cgroup / "memory/memory.usage_in_bytes").write_text("2000000000\n")
            self.assertEqual(host_memory_status(proc, cgroup)["effective_available_bytes"], 48000000000)

    def test_v2_limit_and_unlimited_sentinel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc, cgroup = root / "proc", root / "cgroup"
            (proc / "self").mkdir(parents=True)
            cgroup.mkdir()
            (proc / "meminfo").write_text("MemAvailable: 500000000 kB\n")
            (proc / "self/cgroup").write_text("0::/\n")
            (cgroup / "memory.max").write_text("8000000000\n")
            (cgroup / "memory.current").write_text("3000000000\n")
            self.assertEqual(host_memory_status(proc, cgroup)["effective_available_bytes"], 5000000000)
            (cgroup / "memory.max").write_text("max\n")
            self.assertEqual(host_memory_status(proc, cgroup)["effective_available_bytes"], 500000000 * 1024)


if __name__ == "__main__":
    unittest.main()

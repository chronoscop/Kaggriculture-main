"""Owned native search, final-day handoff, and actual-action teacher contracts."""
from copy import deepcopy
import ctypes
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from route_rl.action_teacher import (combine, generate, run_exclusions, select_seeds, teacher_episode)
from route_rl.action_bc import inspect_index, validation_episode
from route_rl.ppo.controllers import (ASSET_ROOT, DEFAULTS, OBJECTIVE, FinalDayController,
                                     contract_identity, validate_config)
from route_rl.ppo.inference import PolicyHistory
from route_rl.ppo.sampling import external_action_sample
from route_rl.replay_download import read_replay
from route_rl.season_search.search_policy import _LOADED_BINARIES, pack


def game(seed=59101):
    from kaggle_environments import make
    return make("kaggriculture", configuration={"seed": seed}, debug=False)


def observe(environment, player=0):
    return deepcopy(dict(environment.state[player].observation))


@unittest.skipUnless((ASSET_ROOT / "terminal_search.so").exists(), "build owned native search before native checks")
class NativeSeasonContracts(unittest.TestCase):
    def test_expected_terminal_score_cpp_is_bounded_symmetric_and_versioned(self):
        lib = ctypes.CDLL(str(ASSET_ROOT / "terminal_search.so"))
        scorer = lib.expected_terminal_score
        scorer.argtypes = [ctypes.c_double, ctypes.c_double]
        scorer.restype = ctypes.c_double
        self.assertEqual(scorer(0, 2000), .5)
        self.assertGreater(scorer(1000, 2000), .5)
        self.assertLess(scorer(-1000, 2000), .5)
        self.assertAlmostEqual(scorer(1000, 2000) + scorer(-1000, 2000), 1.)
        self.assertGreaterEqual(scorer(-1e10, 1), 0.)
        self.assertLessEqual(scorer(1e10, 1), 1.)
        identity = contract_identity({})
        self.assertEqual(identity["objective"], OBJECTIVE)
        self.assertEqual(identity["binary_sha256"], hashlib.sha256((ASSET_ROOT / "terminal_search.so").read_bytes()).hexdigest())
        with self.assertRaises(ValueError):
            validate_config({"objective": "money"})
        binary = str(ASSET_ROOT / "terminal_search.so")
        with patch.dict(_LOADED_BINARIES, {binary: "old-loaded-native-code"}):
            with self.assertRaisesRegex(ValueError, "restart"):
                contract_identity({})
        malformed = observe(game())
        malformed["farms"][0]["tiles"].pop()
        with self.assertRaisesRegex(ValueError, "10 by 10"):
            pack(malformed)

    def test_final_day_all_23_handoffs_use_actual_observations_and_no_actor_factors(self):
        environment = game()
        histories = [PolicyHistory(0), PolicyHistory(1)]
        controllers = [FinalDayController({"search_seconds": .03}, player) for player in (0, 1)]
        searched_operations = set()
        for step in range(719):
            actions = []
            for player in (0, 1):
                obs = observe(environment, player)
                if step >= 695:
                    histories[player].encode(obs)
                action = controllers[player].choose_action(obs, histories[player])
                if step < 696:
                    self.assertIsNone(action)
                    action = {"farmer": ["PASS"], "market": []}
                    # Establish real live production through official actions.
                    # The final-day controller must harvest, return, and sell
                    # this crop, rather than pass an empty-farm smoke only.
                    if step == 624:
                        action["market"] = [["BUY_SEED", "WHEAT", 1]]
                    elif step == 625:
                        action["farmer"] = ["PLANT", "WHEAT"]
                    elif step in (626, 648, 672):
                        action["farmer"] = ["WATER"]
                else:
                    self.assertIsNotNone(action, controllers[player].diagnostics)
                    selected = external_action_sample(action, obs)
                    self.assertFalse(selected.unit_policy_mask.any())
                    self.assertFalse(selected.market_policy_mask.any())
                    self.assertEqual(selected.old_log_prob, 0)
                    self.assertEqual(selected.unit_mask.sum(), 1 + len(obs["farms"][player]["hands"]))
                    searched_operations.add(action["farmer"][0])
                    searched_operations.update(order[0] for order in action.get("market", []))
                histories[player].remember(obs, action)
                actions.append(action)
            environment.step(actions)
        self.assertTrue(environment.done)
        self.assertEqual(len(environment.steps), 720)
        self.assertEqual([state.status for state in environment.state], ["DONE", "DONE"])
        self.assertTrue({"HARVEST", "DROP", "SELL"} <= searched_operations, searched_operations)
        for controller in controllers:
            self.assertEqual(controller.diagnostics["handoff_actions"], 23)
            self.assertEqual(controller.diagnostics["fallback_actions"], 0)
        # Reusing the same object cannot carry a failed/started controller into
        # a new game, even without the PPOPolicy wrapper doing a reset.
        self.assertIsNone(controllers[0].choose_action(observe(game(59102)), histories[0]))
        self.assertFalse(controllers[0].started)
        self.assertFalse(controllers[0].failed)

    def test_handoff_uses_existing_belief_and_error_falls_back_to_neural(self):
        obs = observe(game())
        obs.update(step=696, day=29, hour=0)
        history = PolicyHistory(0)
        history.tracker.shed["WHEAT"] = 7
        history.tracker.carried_by_unit[0]["WHEAT"] = 3
        seen = []
        class Planner:
            def __init__(self, seconds):
                self.config = {}
                self.header = np.zeros(32, np.int32)
                self.header[9] = 500000
            def __call__(self, supplied):
                seen.append(supplied)
                return {"farmer": ["PASS"], "hands": [], "market": []}
        with patch("route_rl.season_search.search_policy.SearchPolicy", Planner):
            controller = FinalDayController({}, 0)
            self.assertIsNotNone(controller.choose_action(obs, history))
        self.assertEqual(seen[0]["_opponent_products"]["WHEAT"], 7)
        self.assertEqual(seen[0]["_opponent_carried_by_unit"][0]["WHEAT"], 3)
        packed = pack(seen[0])
        self.assertEqual(packed[-2], obs["farms"][1]["money"])
        self.assertEqual(packed[-1], DEFAULTS["uncertainty_scale"])
        with patch("route_rl.season_search.search_policy.SearchPolicy", side_effect=RuntimeError("native test failure")):
            controller = FinalDayController({}, 0)
            self.assertIsNone(controller.choose_action(obs, history))
            self.assertTrue(controller.failed)
            self.assertIn("native test failure", controller.diagnostics["last_error"])
        obs["remainingOverageTime"] = 2
        controller = FinalDayController({}, 0)
        self.assertIsNone(controller.choose_action(obs, history))
        self.assertFalse(controller.started)

    def test_teacher_full_games_have_actual_labels_and_unchanged_bc_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(seed=[59121, 59122], source_replay=[], seed_start=None, games=None,
                exclude_seeds=None, exclude_run=[], holdout_games=1, split_seed=51,
                seconds=.05, uncertainty_scale=2000., out=root / "teachers")
            receipt = generate(args)
            self.assertTrue(receipt["complete"])
            self.assertFalse(receipt["training_started"])
            self.assertEqual(receipt["teacher_seats"], 4)
            self.assertEqual(receipt["planner"]["config"]["search_seconds"], args.seconds)
            inventory = inspect_index(args.out / "teacher-seats.jsonl")
            self.assertEqual(inventory["splits"], {"train": 2, "validation": 2})
            rows = [json.loads(line) for line in (args.out / "teacher-seats.jsonl").read_text().splitlines()]
            for row in rows:
                replay = read_replay(args.out / row["path"])
                self.assertEqual(len(replay["steps"]), 720)
                self.assertEqual(validation_episode(row["episode_id"]), row["split"] == "validation")
                self.assertEqual(replay["info"]["seed"], row["seed"])
                self.assertIsInstance(replay["steps"][1][row["seat"]]["action"], dict)
                operations = {state[row["seat"]]["action"].get("farmer", ["PASS"])[0]
                              for state in replay["steps"][1:]}
                self.assertTrue({"PLANT", "HARVEST"} <= operations, operations)
                self.assertTrue(any(isinstance(tile, dict) and tile.get("yield_units", 0) > 0
                    for states in replay["steps"] for farm in states[row["seat"]]["observation"]["farms"]
                    for tiles in farm["tiles"] for tile in tiles))
            # Verify the generated archive enters the original label encoder,
            # rather than only resembling its JSON shape.
            from route_rl.full_action.labels import label_actions
            from route_rl.full_action.legality import action_legality
            replay = read_replay(args.out / rows[0]["path"])
            observations = [state["observation"] for state in replay["steps"][0]]
            actions = [state["action"] for state in replay["steps"][1]]
            legal = action_legality(observations, actions, 0)
            labels = label_actions(observations[0], actions[0], legal)
            self.assertEqual(labels.shape, (30,))
            self.assertGreaterEqual(labels[0], 0)
            public = root / "public"
            public.mkdir()
            identities = {}
            public_rows = []
            for row in rows:
                ep = row["episode_id"]
                if ep not in identities:
                    identities[ep] = next(number for number in range(1, 1000)
                        if validation_episode(number) == (row["split"] == "validation")
                        and number not in identities.values())
                copied = {**row, "episode_id": identities[ep], "submission_id": 321,
                          "path": str((args.out / row["path"]).resolve())}
                copied.pop("teacher_contract")
                public_rows.append(copied)
            public_index = public / "teacher-seats.jsonl"
            public_index.write_text("".join(json.dumps(row) + "\n" for row in public_rows))
            merge = SimpleNamespace(public_index=public_index, teacher_index=args.out / "teacher-seats.jsonl",
                                    out=root / "mixed")
            merged = combine(merge)
            self.assertEqual(merged["combined_inventory"]["teacher_seats"], 8)
            self.assertEqual(merged["public_holdout_seeds"], receipt["heldout_seeds"])
            merged_rows = [json.loads(line) for line in (merge.out / "teacher-seats.jsonl").read_text().splitlines()]
            self.assertTrue(all(Path(row["path"]).is_absolute() for row in merged_rows))
            self.assertTrue(all(Path(row["path"]).exists() for row in merged_rows))
            with self.assertRaisesRegex(ValueError, "immutable"):
                combine(merge)
            # Swap both public split roles so both splits still exist. Combining
            # must detect the teacher trying to train on a public held-out seed.
            leak_rows = []
            swapped = {}
            for row in public_rows:
                seed = row["seed"]
                validation = row["split"] != "validation"
                swapped.setdefault(seed, next(number for number in range(2000, 3000)
                    if validation_episode(number) == validation and number not in swapped.values()))
                leak_rows.append({**row, "episode_id": swapped[seed],
                                  "split": "validation" if validation else "train"})
            public_index.write_text("".join(json.dumps(row) + "\n" for row in leak_rows))
            merge.out = root / "leaking-mixed"
            with self.assertRaisesRegex(ValueError, "public holdout"):
                combine(merge)
            self.assertFalse(merge.out.exists())


class TeacherSelectionContracts(unittest.TestCase):
    def test_split_ids_group_each_seed_and_run_exclusions_protect_reserved_ranges(self):
        self.assertTrue(validation_episode(teacher_episode(531, True)))
        self.assertFalse(validation_episode(teacher_episode(532, False)))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "integration.json").write_text(json.dumps({"bc": {"known_game_seeds": [1, 2]},
                "critic_validation_seeds": [3], "reserved_training_ranges": [[100, 200]]}))
            (path / "receipt.json").write_text(json.dumps({"training_seeds": [4]}))
            seeds, ranges, records = run_exclusions([path])
            self.assertEqual(seeds, {1, 2, 3, 4})
            self.assertEqual(ranges, [(100, 200)])
            self.assertEqual(len(records), 2)
            with self.assertRaisesRegex(ValueError, "reserved"):
                select_seeds([3], [], None, None, seeds)


if __name__ == "__main__":
    unittest.main()

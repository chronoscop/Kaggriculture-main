"""Native/Python execution equivalence, including real prefix supports and history.

These are correctness checks, not throughput or playing-strength experiments.
"""
from copy import deepcopy
from pathlib import Path
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from route_rl.full_action.model import JaxModelConfig
from route_rl.ppo.inference import PolicyHistory
from route_rl.ppo.native_backend import RustBatchBackend, backend_identity, native_module
from route_rl.ppo.rollout import collect_rollout
from route_rl.ppo.sampling import (ACTION_SELECTION_CONTRACT, ECONOMIC_ACTION_SELECTION_CONTRACT,
                                   MARKET_ACTIONS, PASS_ID, NOOP_ID, PrefixResolver, sample_action)


def official(seed):
    from kaggle_environments import make
    return make("kaggriculture", configuration={"seed": seed}, debug=False)


def observation(env, player):
    result = deepcopy(dict(env.state[player].observation))
    result.setdefault("step", int(result["day"])*24+int(result["hour"]))
    return result


def pass_logits(rows=1):
    result = {"unit_action": np.full((rows, 20, 500), -1000., np.float32),
              "market_action": np.full((rows, 10, len(MARKET_ACTIONS)), -1000., np.float32)}
    result["unit_action"][..., PASS_ID] = 0.
    result["market_action"][..., NOOP_ID] = 0.
    return result


class NativeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from route_rl.ppo import native_backend
        if not Path(native_backend.__file__).with_name("_native").joinpath("build.json").exists():
            raise unittest.SkipTest("build owned native backend with tools/build_action_native.py")
        # A present but stale/tampered build must fail validation, not silently
        # turn a broken release into skipped tests.
        native_module()

    def test_observations_features_inventory_history_and_conditional_supports(self):
        seed = 370019
        env, backend = official(seed), RustBatchBackend([seed])
        histories = [PolicyHistory(0), PolicyHistory(1)]
        schedule = {
            0: {"farmer": ["PASS"], "market": [["HIRE"], ["HIRE"], ["BUY_SEED", "WHEAT", 3], ["BUY_PRODUCT", "FERTILIZER", 3], ["BUY_PRODUCT", "WHEAT", 4]]},
            1: {"farmer": ["PICKUP", "FERTILIZER", 1], "hands": [["WEST"], ["NORTH"]]},
            2: {"farmer": ["PLANT", "WHEAT"], "hands": [["PLANT", "WHEAT"], ["WEST"]]},
            3: {"farmer": ["FERTILIZE"], "hands": [["WATER"], ["PLANT", "WHEAT"]]},
            4: {"farmer": ["WATER"], "hands": [["EAST"], ["WATER"]]},
            6: {"farmer": ["PASS"], "market": [["NOOP"], ["SELL", "WHEAT", 2], ["SELL", "WHEAT", 1]]},
            24: {"farmer": ["PASS"], "market": [["HIRE"], ["BUY_SEED", "TOMATO", 1]]},
            25: {"farmer": ["WEST"], "hands": [["NORTH"]]},
            26: {"farmer": ["HARVEST"], "hands": [["HARVEST"]]},
            27: {"farmer": ["EAST"], "hands": [["SOUTH"]]},
            28: {"farmer": ["DROP"], "hands": [["DROP"]]},
            29: {"farmer": ["PLANT", "TOMATO"]},
        }
        random = np.random.default_rng(115)
        for step in range(52):
            native_obs = backend.observe()[0]
            own_obs = [observation(env, player) for player in range(2)]
            native_batch = backend.encode()
            for player in range(2):
                for key in ("step", "day", "hour", "farms", "market", "town", "private"):
                    self.assertEqual(native_obs[player][key], own_obs[player][key], (step, player, key))
                expected = histories[player].encode(own_obs[player])
                for name, array in expected.items():
                    np.testing.assert_allclose(native_batch[name][player], array[0], rtol=0, atol=2e-7,
                                               err_msg=f"step {step} player {player} feature {name}")
                logits = {"unit_action": random.normal(size=(20, 500)).astype(np.float32),
                          "market_action": random.normal(size=(10, len(MARKET_ACTIONS))).astype(np.float32)}
                for contract in (ACTION_SELECTION_CONTRACT, ECONOMIC_ACTION_SELECTION_CONTRACT):
                    python_sample = sample_action(own_obs[player], logits, np.random.default_rng(82), action_selection_contract=contract)
                    rust_sample = sample_action(native_obs[player], logits, np.random.default_rng(82),
                        action_selection_contract=contract, resolver=backend.resolver(0, player))
                    self.assertEqual(python_sample.action, rust_sample.action, (step, player, contract))
                    for name, array in python_sample.arrays().items():
                        np.testing.assert_array_equal(array, rust_sample.arrays()[name], err_msg=f"{step} {player} {contract} {name}")
            actions = [deepcopy(schedule.get(step, {"farmer": ["PASS"]})) for _ in range(2)]
            backend.environment.step_owned_actions(actions)
            backend.step += 1
            env.step(actions)
            for player in range(2):
                histories[player].remember(own_obs[player], actions[player])
        self.assertEqual(len(backend.observe()[0][0]["farms"][0]["hands"]), 0)

    def test_exact_decoder_preserves_market_noop_positions_and_absolute_quantities(self):
        env, backend = official(48003), RustBatchBackend([48003])
        buy = [{"farmer": ["PASS"], "market": [["BUY_PRODUCT", "WHEAT", 4]]}, {"farmer": ["PASS"]}]
        env.step(buy)
        backend.environment.step_owned_actions(buy)
        backend.step += 1
        logits = pass_logits(2)
        for slot, amount in ((1, 1), (3, 2)):
            logits["market_action"][0, slot, MARKET_ACTIONS.index(("SELL", "WHEAT", amount))] = 1.
        samples = [sample_action(backend.observe()[0][player], {name: a[player] for name, a in logits.items()}, None,
                                resolver=backend.resolver(0, player)) for player in range(2)]
        self.assertEqual(samples[0].action["market"][:4], [["NOOP"], ["SELL", "WHEAT", 1], ["NOOP"], ["SELL", "WHEAT", 2]])
        backend.advance(samples)
        env.step([sample.action for sample in samples])
        for player in range(2):
            self.assertEqual(backend.observe()[0][player]["market"], observation(env, player)["market"])
            self.assertEqual(backend.observe()[0][player]["private"], observation(env, player)["private"])

    def test_two_game_complete_horizon_is_synchronous_and_old_support_probabilities_recorded(self):
        sizes = []
        def forward(batch):
            rows = len(batch["features"])
            sizes.append(rows)
            return {**pass_logits(rows), "value": np.zeros(rows, np.float32)}
        config = JaxModelConfig(d_model=8, layers=1, heads=1, ffn_dim=12, rope_dim=4)
        rollout = collect_rollout({}, config, [41001, 41002], 4, forward=forward, collection_backend="rust-batch")
        self.assertEqual(sizes, [4]*719)
        self.assertEqual(rollout.values.shape, (719, 2, 2))
        self.assertTrue(rollout.arrays["done"][-1].all())
        self.assertFalse(rollout.arrays["done"][:-1].any())
        self.assertTrue((rollout.arrays["old_log_prob"] == 0).all())
        np.testing.assert_array_equal(rollout.terminal_scores, [[.5, .5], [.5, .5]])
        self.assertEqual(rollout.metadata["collection"], backend_identity("rust-batch"))
        self.assertTrue(all(game["turns"] == 720 for game in rollout.metadata["games"]))

    def test_native_prefix_economic_forced_overflow_and_terminal_components(self):
        backend = RustBatchBackend([42008])
        # An explicit initial-money fixture makes capacity reachable through
        # actual requests. No future inventory/workers are injected.
        backend.environment = native_module().RustBatchEnv([42008], config_overrides={"startingMoney": 100000})
        noop = {"farmer": ["PASS"]}
        def step(action):
            backend.environment.step_owned_actions([action, noop])
            backend.step += 1
        step({"farmer": ["PASS"], "market": [["BUY_PRODUCT", "FERTILIZER", 100]]})
        step({"farmer": ["PICKUP", "FERTILIZER", 1], "market": [["BUY_PRODUCT", "WHEAT", 1]]})
        while backend.step < 23:
            step(noop)
        for requested_step in (23, 717):
            if requested_step == 717:
                # Real purchase/pickup earlier in the final day yields a carried
                # item to liquidate, without inventing a worker or future stock.
                while backend.step < 716:
                    step(noop)
                step({"farmer": ["PICKUP", "FERTILIZER", 1]})
            obs = backend.observe()[0][0]
            logits = {name: array[0] for name, array in pass_logits().items()}
            expected = sample_action(obs, logits, None, action_selection_contract=ECONOMIC_ACTION_SELECTION_CONTRACT)
            actual = sample_action(obs, logits, None, action_selection_contract=ECONOMIC_ACTION_SELECTION_CONTRACT,
                                   resolver=backend.resolver(0, 0))
            self.assertEqual(actual.action, expected.action)
            for name, array in expected.arrays().items():
                np.testing.assert_array_equal(array, actual.arrays()[name])
            self.assertFalse(actual.market_policy_mask.all())
            self.assertTrue(any(order[0] == "SELL" for order in actual.action["market"]))

    def test_complete_native_hybrid_records_actual_external_actions_without_actor_probability(self):
        def forward(batch):
            rows = len(batch["features"])
            return {**pass_logits(rows), "value": np.zeros(rows, np.float32)}
        config = JaxModelConfig(d_model=8, layers=1, heads=1, ffn_dim=12, rope_dim=4)
        rollout = collect_rollout({}, config, [54003], 4, forward=forward,
            collection_backend="rust-batch", action_selection_contract=ECONOMIC_ACTION_SELECTION_CONTRACT,
            season_controller={"search_seconds": .001})
        self.assertTrue(rollout.arrays["unit_policy_mask"][:696, ..., 0].all())
        self.assertFalse(rollout.arrays["unit_policy_mask"][696:].any())
        self.assertFalse(rollout.arrays["market_policy_mask"][696:].any())
        self.assertTrue((rollout.arrays["old_log_prob"][696:] == 0).all())
        self.assertEqual(rollout.metadata["controller"]["contract"], "owned-final-day-search-expected-score-v1")

    def test_complete_search_teacher_tape_matches_full_state_and_public_history(self):
        from route_rl.action_teacher import generate_game
        seed = 58109
        # A small search budget creates a real productive action tape; identical
        # requests are then replayed in Rust. This is a correctness fixture and
        # its terminal money is not a policy-selection or speed benchmark.
        replay = generate_game(seed, seconds=.05, uncertainty_scale=2000.)
        self.assertEqual(len(replay["steps"]), 720)
        backend = RustBatchBackend([seed])
        histories = [PolicyHistory(0), PolicyHistory(1)]
        production_seen, harvesting_seen, animal_seen, worker_seen = False, False, False, False
        crop_yield_seen, animal_yield_seen = False, False
        operations = set()
        for step, states in enumerate(replay["steps"]):
            native_observations = backend.observe()[0]
            native_batch = backend.encode()
            observations = []
            for player in (0, 1):
                expected = deepcopy(states[player]["observation"])
                expected.setdefault("step", expected["day"]*24+expected["hour"])
                observations.append(expected)
                for key in ("step", "day", "hour", "farms", "private", "market", "town"):
                    self.assertEqual(native_observations[player][key], expected[key], (step, player, key))
                encoded = histories[player].encode(expected)
                for name, array in encoded.items():
                    np.testing.assert_allclose(native_batch[name][player], array[0], rtol=0, atol=2e-7,
                                               err_msg=f"teacher step {step} player {player} {name}")
                farm = expected["farms"][player]
                worker_seen |= bool(farm["hands"])
                for row in farm["tiles"]:
                    for tile in row:
                        if isinstance(tile, dict):
                            production_seen |= tile.get("kind") == "PLANT"
                            animal_seen |= tile.get("animal") in ("GOOSE", "COW", "SHEEP")
                            crop_yield_seen |= tile.get("kind") == "PLANT" and tile.get("yield_units", 0) > 0
                            animal_yield_seen |= bool(tile.get("animal")) and tile.get("yield_units", 0) > 0
            if step == 719:
                break
            actions = [deepcopy(replay["steps"][step+1][player]["action"]) for player in (0, 1)]
            for action in actions:
                for request in [action.get("farmer", []), *action.get("hands", []), *action.get("market", [])]:
                    if request:
                        operations.add(request[0])
                        harvesting_seen |= request[0] == "HARVEST"
            done = backend.environment.step_owned_actions(actions)
            backend.step += 1
            self.assertEqual(done, [step == 718])
            for player in (0, 1):
                histories[player].remember(observations[player], actions[player])
        self.assertTrue(production_seen, operations)
        self.assertTrue(harvesting_seen, operations)
        self.assertTrue(worker_seen, operations)
        self.assertTrue(animal_seen, operations)
        self.assertTrue(crop_yield_seen, operations)
        self.assertTrue(animal_yield_seen, operations)
        np.testing.assert_array_equal(backend.rewards()[0], [state["reward"] for state in replay["steps"][-1]])

    def test_native_identity_rejects_binary_tampering_and_stale_source(self):
        from route_rl.ppo import native_backend
        source = Path(native_backend.__file__).with_name("_native")
        receipt = json.loads((source / "build.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_module = root / "src/route_rl/ppo/native_backend.py"
            artifacts = fake_module.with_name("_native")
            artifacts.mkdir(parents=True)
            shutil.copy2(source / receipt["binary"], artifacts / receipt["binary"])
            bad = {**receipt, "binary_sha256": "0"*64}
            (artifacts / "build.json").write_text(json.dumps(bad))
            with patch.object(native_backend, "__file__", str(fake_module)):
                with self.assertRaisesRegex(RuntimeError, "binary does not match"):
                    backend_identity("rust-batch")
                (artifacts / "build.json").write_text(json.dumps(receipt))
                crate = root / "native/action_engine"
                (crate / "src").mkdir(parents=True)
                for name in ("Cargo.toml", "Cargo.lock", "pyproject.toml", "src/lib.rs"):
                    (crate / name).write_text("changed source fixture\n")
                with self.assertRaisesRegex(RuntimeError, "source changed after building"):
                    backend_identity("rust-batch")


if __name__ == "__main__":
    unittest.main()

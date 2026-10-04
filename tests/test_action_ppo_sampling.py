"""Official execution and behavior probability checks; no strength claims."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from route_rl.full_action.catalog import MARKET_SLOTS, UNIT_ACTIONS, UNIT_ACTION_TO_ID
from route_rl.full_action.checkpoints import atomic_pickle, save_params_payload
from route_rl.full_action.inference import GreedyPolicy, MAX_OWN_UNITS
from route_rl.full_action.model import JaxModelConfig, add_zero_value_head, initialize_params, policy_forward
from route_rl.full_action.sell_quantity import ABSOLUTE_ACTION_COUNT
from route_rl.ppo.inference import PPOPolicy, PolicyHistory, load_policy
from route_rl.ppo.objective import behavior_outputs, generalized_advantage_estimate, joint_log_prob
from route_rl.ppo.rollout import collect_rollout, flatten_rollout, terminal_scores
from route_rl.ppo.provenance import OBJECTIVE_CONTRACT
from route_rl.ppo.sampling import (ACTION_SELECTION_CONTRACT, MARKET_ACTIONS, NOOP_ID,
                                   PASS_ID, PrefixResolver, sample_action, sample_categorical)


def game(seed=1701):
    from kaggle_environments import make
    return make("kaggriculture", configuration={"seed": seed}, debug=False)


def observation(environment, player=0):
    return deepcopy(dict(environment.state[player].observation))


def preferred_logits(units=None, markets=None):
    result = {"unit_action": np.full((MAX_OWN_UNITS, len(UNIT_ACTIONS)), -1000., np.float32),
              "market_action": np.full((MARKET_SLOTS, ABSOLUTE_ACTION_COUNT), -1000., np.float32)}
    result["unit_action"][:, PASS_ID] = -1.
    result["market_action"][:, NOOP_ID] = -1.
    for index, action in (units or {}).items():
        result["unit_action"][index, UNIT_ACTION_TO_ID[tuple(action)]] = 0.
    for index, action in (markets or {}).items():
        result["market_action"][index, MARKET_ACTIONS.index(tuple(action))] = 0.
    return result


class PrefixSamplingContracts(unittest.TestCase):
    def test_seeds_and_shared_tiles_use_the_actual_unit_prefix(self):
        environment = game()
        environment.step([{"farmer": ["PASS"], "market": [["HIRE"], ["HIRE"], ["BUY_SEED", "WHEAT", 2]]},
                          {"farmer": ["PASS"]}])
        environment.step([{"farmer": ["PASS"], "hands": [["WEST"], ["NORTH"]]},
                          {"farmer": ["PASS"]}])
        before = observation(environment)
        self.assertEqual(before["farms"][0]["hands"], [[4, 4], [4, 4]])
        wheat = UNIT_ACTION_TO_ID[("PLANT", "WHEAT")]
        logits = preferred_logits({0: ["PLANT", "WHEAT"], 1: ["PLANT", "WHEAT"], 2: ["PLANT", "WHEAT"]})
        selected = sample_action(before, logits, rng=None)
        self.assertEqual(selected.action["farmer"], ["PLANT", "WHEAT"])
        self.assertEqual(selected.action["hands"], [["PASS"], ["PASS"]])
        self.assertTrue(selected.unit_legal_mask[0, wheat])
        self.assertFalse(selected.unit_legal_mask[1:, wheat].any())
        self.assertEqual(selected.unit_mask.sum(), 3)
        environment.step([selected.action, {"farmer": ["PASS"]}])
        after = observation(environment)
        self.assertEqual(after["private"]["seeds"]["WHEAT"], 1)
        self.assertEqual(after["farms"][0]["tiles"][4][4]["crop"], "WHEAT")
        # A second plant on a different empty tile is blocked by actual depleted
        # seed count, not by a same-tile heuristic alone.
        resolver = PrefixResolver(before)
        resolver.private["seeds"]["WHEAT"] = 1
        resolver.farm["hands"][0] = [3, 4]
        resolver.apply_unit(0, wheat)
        self.assertFalse(resolver.unit_support(1)[wheat])

    def test_absolute_sell_is_never_clamped_and_noop_slots_survive(self):
        environment = game()
        environment.step([{"farmer": ["PASS"], "market": [["BUY_PRODUCT", "WHEAT", 2]]},
                          {"farmer": ["PASS"]}])
        before = observation(environment)
        logits = preferred_logits(markets={1: ["SELL", "WHEAT", 1], 2: ["SELL", "WHEAT", 1],
                                            3: ["SELL", "WHEAT", 3]})
        selected = sample_action(before, logits, rng=None)
        self.assertEqual(len(selected.action["market"]), MARKET_SLOTS)
        self.assertEqual(selected.action["market"][:4], [["NOOP"], ["SELL", "WHEAT", 1],
                                                        ["SELL", "WHEAT", 1], ["NOOP"]])
        sell_two = MARKET_ACTIONS.index(("SELL", "WHEAT", 2))
        sell_one = MARKET_ACTIONS.index(("SELL", "WHEAT", 1))
        self.assertTrue(selected.market_legal_mask[1, sell_two])
        self.assertFalse(selected.market_legal_mask[2, sell_two])
        self.assertTrue(selected.market_legal_mask[2, sell_one])
        self.assertFalse(selected.market_legal_mask[3, sell_one])
        for slot, identity in enumerate(selected.market_action):
            self.assertEqual(selected.action["market"][slot], list(MARKET_ACTIONS[int(identity)]))
        environment.step([selected.action, {"farmer": ["PASS"]}])
        self.assertEqual(observation(environment)["private"]["shed"]["WHEAT"], 0)

    def test_post_unit_drop_changes_market_support_before_sampling(self):
        environment = game()
        environment.step([{"farmer": ["PASS"], "market": [["BUY_PRODUCT", "WHEAT", 2]]},
                          {"farmer": ["PASS"]}])
        environment.step([{"farmer": ["PICKUP", "WHEAT", 2]}, {"farmer": ["PASS"]}])
        selected = sample_action(observation(environment),
                                preferred_logits({0: ["DROP"]}, {0: ["SELL", "WHEAT", 2]}), None)
        self.assertEqual(selected.action["farmer"], ["DROP"])
        self.assertEqual(selected.action["market"][0], ["SELL", "WHEAT", 2])
        environment.step([selected.action, {"farmer": ["PASS"]}])
        self.assertEqual(observation(environment)["private"]["shed"]["WHEAT"], 0)

    def test_expired_workers_are_absent_until_real_rehire_observation(self):
        environment = game()
        environment.step([{"farmer": ["PASS"], "market": [["HIRE"]]}, {"farmer": ["PASS"]}])
        for _ in range(23):
            environment.step([{"farmer": ["PASS"]}, {"farmer": ["PASS"]}])
        expired = sample_action(observation(environment), preferred_logits(), None)
        self.assertEqual(expired.unit_mask.sum(), 1)
        self.assertTrue((expired.unit_action[1:] == PASS_ID).all())
        environment.step([{"farmer": ["PASS"], "market": [["HIRE"]]}, {"farmer": ["PASS"]}])
        hired = sample_action(observation(environment), preferred_logits({1: ["WEST"]}), None)
        self.assertEqual(hired.unit_mask.sum(), 2)
        self.assertEqual(hired.action["hands"], [["WEST"]])

    def test_hire_support_respects_capacity_and_oversized_observations_fail(self):
        before = observation(game())
        before["farms"][0]["hands"] = [[4, 4]] * 19
        before["private"]["inventories"] = [{} for _ in range(20)]
        hire = MARKET_ACTIONS.index(("HIRE",))
        self.assertFalse(PrefixResolver(before).market_support()[hire])
        before["farms"][0]["hands"].append([4, 4])
        with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
            PrefixResolver(before)

    def test_stored_prefix_masks_reproduce_joint_and_per_slot_probabilities(self):
        before = observation(game())
        rng = np.random.default_rng(22)
        outputs = {"unit_action": rng.normal(size=(20, 500)).astype(np.float32),
                   "market_action": rng.normal(size=(10, ABSOLUTE_ACTION_COUNT)).astype(np.float32)}
        selected = sample_action(before, outputs, rng)
        batch = {name: value[None] for name, value in selected.arrays().items()}
        raw = {name: jnp.asarray(value[None]) for name, value in outputs.items()}
        masked = behavior_outputs(raw, batch)
        joint = joint_log_prob(masked, batch["unit_action"], batch["market_action"],
                               batch["unit_mask"], batch["market_mask"])
        np.testing.assert_allclose(np.asarray(joint)[0], selected.old_log_prob, atol=2e-5)
        for head in ("unit", "market"):
            probabilities = jax.nn.log_softmax(masked[f"{head}_action"], axis=-1)
            chosen = np.take_along_axis(np.asarray(probabilities), batch[f"{head}_action"][..., None], -1)[0, :, 0]
            active = selected.arrays()[f"{head}_mask"]
            np.testing.assert_allclose(chosen[active], selected.arrays()[f"old_{head}_log_prob"][active], atol=2e-6)
            self.assertTrue(np.asarray(jax.nn.softmax(masked[f"{head}_action"]))[~batch[f"{head}_legal_mask"]].sum() == 0)
        self.assertAlmostEqual(float(selected.old_log_prob),
                               float(selected.old_unit_log_prob.sum() + selected.old_market_log_prob.sum()), places=4)

    def test_masked_categories_have_zero_mass_even_below_old_sentinel(self):
        logits = np.asarray([-2e9, 3e9], np.float32)
        mask = np.asarray([True, False])
        for rng in (None, np.random.default_rng(3)):
            identity, probability = sample_categorical(logits, mask, rng)
            self.assertEqual(identity, 0)
            self.assertEqual(probability, 0)

    def test_market_price_refresh_does_not_make_ineffective_orders_legal(self):
        before = observation(game())
        before["farms"][0]["money"] = 0
        before["market"]["prices"]["WHEAT"] = 999999
        mask = PrefixResolver(before).market_support()
        self.assertTrue(mask[NOOP_ID])
        self.assertEqual(mask.sum(), 1)


class RolloutConnectivity(unittest.TestCase):
    def test_real_tiny_model_and_checkpoint_specific_greedy_inference(self):
        config = JaxModelConfig(d_model=16, layers=1, heads=1, ffn_dim=24, rope_dim=4)
        params = add_zero_value_head(initialize_params(jax.random.PRNGKey(7), config), config)
        before = observation(game())
        history = PolicyHistory(0)
        arrays = history.encode(before)
        outputs = jax.device_get(jax.jit(lambda batch: policy_forward(params, batch, config))(
            jax.tree.map(jnp.asarray, arrays)))
        selected = sample_action(before, outputs, None)
        ppo_payload = {"model_config": config.to_dict(), "params": params,
                       "action_selection_contract": ACTION_SELECTION_CONTRACT,
                       "learning_objective_contract": OBJECTIVE_CONTRACT}
        own = PPOPolicy(ppo_payload, warm=False)
        self.assertEqual(own(before), selected.action)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pkl"
            save_params_payload(path, params, config.to_dict(), 0)
            self.assertIsInstance(load_policy(path, warm=False), GreedyPolicy)
            import pickle
            payload = pickle.loads(path.read_bytes())
            payload["action_selection_contract"] = ACTION_SELECTION_CONTRACT
            payload["learning_objective_contract"] = OBJECTIVE_CONTRACT
            atomic_pickle(path, payload)
            self.assertIsInstance(load_policy(path, warm=False), PPOPolicy)
            payload["action_selection_contract"] = "unsupported"
            atomic_pickle(path, payload)
            with self.assertRaisesRegex(ValueError, "unsupported"):
                load_policy(path, warm=False)

    def test_full_official_game_terminal_score_and_flatten_alignment(self):
        # A deterministic tiny forward hook checks all 719 real transitions and
        # both independent histories without running an optimization experiment.
        template = preferred_logits()
        template["unit_action"][:, PASS_ID] = 0.
        template["market_action"][:, NOOP_ID] = 0.

        def forward(batch):
            self.assertEqual(batch["features"].shape[0], 2)
            return {**{name: np.broadcast_to(value, (2, *value.shape)) for name, value in template.items()},
                    "value": np.asarray([0., 0.], np.float32)}

        config = JaxModelConfig(d_model=8, layers=1, heads=1, ffn_dim=12, rope_dim=4)
        rollout = collect_rollout({}, config.to_dict(), [194713], 11, forward=forward)
        self.assertEqual(rollout.values.shape, (719, 1, 2))
        self.assertEqual(rollout.metadata["games"][0]["turns"], 720)
        self.assertEqual(rollout.metadata["games"][0]["statuses"], ["DONE", "DONE"])
        np.testing.assert_array_equal(rollout.terminal_scores, [[.5, .5]])
        self.assertFalse(rollout.arrays["done"][:-1].any())
        self.assertTrue(rollout.arrays["done"][-1].all())
        self.assertFalse(rollout.arrays["reward"][:-1].any())
        np.testing.assert_array_equal(rollout.arrays["reward"][-1], rollout.terminal_scores)
        self.assertTrue((rollout.arrays["unit_mask"].sum(-1) == 1).all())
        self.assertTrue((rollout.arrays["old_log_prob"] == 0).all())
        advantages, returns = generalized_advantage_estimate(rollout.values, rollout.terminal_scores)
        flat = flatten_rollout(rollout, advantages, returns)
        self.assertEqual(len(flat["return"]), 1438)
        np.testing.assert_array_equal(flat["features"][:2], rollout.arrays["features"][0, 0])
        np.testing.assert_array_equal(flat["return"].reshape(719, 1, 2), returns)
        np.testing.assert_array_equal(flat["advantage"].reshape(719, 1, 2), advantages)
        self.assertTrue((returns == .5).all())
        self.assertTrue((advantages == 0).all())

    def test_seeds_are_validated_before_environment_creation(self):
        config = JaxModelConfig(d_model=8, layers=1, heads=1, ffn_dim=12, rope_dim=4)
        for seeds in ([], [0], [1, 1], [-1]):
            with self.assertRaisesRegex(ValueError, "seeds"):
                collect_rollout({}, config, seeds, 1)
        with self.assertRaisesRegex(ValueError, "held-out"):
            collect_rollout({}, config, [1], 1, excluded_seeds={1})
        for rewards, scores in (([10, 5], [1., 0.]), ([5, 10], [0., 1.]), ([5, 5], [.5, .5])):
            np.testing.assert_array_equal(terminal_scores(rewards), scores)


if __name__ == "__main__":
    unittest.main()

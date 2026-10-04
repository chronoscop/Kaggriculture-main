"""Execution / probability invariants for economic PPO; no strength claims."""
from copy import deepcopy
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as rules

from route_rl.full_action.catalog import MARKET_SLOTS, UNIT_ACTIONS, UNIT_ACTION_TO_ID
from route_rl.full_action.inference import MAX_OWN_UNITS
from route_rl.full_action.sell_quantity import ABSOLUTE_ACTION_COUNT
from route_rl.ppo.economic_rules import care_gains, fertilize_gains
from route_rl.ppo.objective import behavior_outputs, joint_log_prob
from route_rl.ppo.sampling import (
    ACTION_SELECTION_CONTRACT, ECONOMIC_ACTION_SELECTION_CONTRACT,
    MARKET_ACTIONS, MARKET_ACTION_TO_ID, NOOP_ID, PASS_ID, PrefixResolver,
    external_action_sample, sample_action,
)


def observation(day=0, hour=0):
    environment = make("kaggriculture", configuration={"seed": 180701}, debug=False)
    result = deepcopy(dict(environment.state[0].observation))
    result.update(day=day, hour=hour, step=day * 24 + hour)
    return result


def logits(units=(), markets=()):
    result = {"unit_action": np.full((MAX_OWN_UNITS, len(UNIT_ACTIONS)), -20., np.float32),
              "market_action": np.full((MARKET_SLOTS, ABSOLUTE_ACTION_COUNT), -20., np.float32)}
    result["unit_action"][:, PASS_ID] = 0
    result["market_action"][:, NOOP_ID] = 0
    for index, order in units:
        result["unit_action"][index, UNIT_ACTION_TO_ID[tuple(order)]] = 10
    for index, order in markets:
        result["market_action"][index, MARKET_ACTION_TO_ID[tuple(order)]] = 10
    return result


def economic(before, outputs=None, rng=None):
    return sample_action(before, outputs or logits(), rng,
                         action_selection_contract=ECONOMIC_ACTION_SELECTION_CONTRACT)


class EconomicSamplingContracts(unittest.TestCase):
    def test_useful_fertilizer_windows_and_caps(self):
        plant = rules._new_plant("WHEAT", 0, 24)
        self.assertTrue(fertilize_gains(plant, 2))
        plant["fertilized_until_day"] = 4
        self.assertFalse(fertilize_gains(plant, 2))
        plant["fertilized_until_day"] = -1
        plant["yield_units"] = rules.CROPS["WHEAT"]["max_yield"] - 1
        self.assertFalse(fertilize_gains(plant, 2))
        tomato = rules._new_plant("TOMATO", 21, 24)
        self.assertTrue(fertilize_gains(tomato, 28))
        self.assertFalse(fertilize_gains(tomato, 29))
        before = observation(29)
        before["farms"][0]["tiles"][4][4] = tomato
        before["private"]["inventories"] = [{"FERTILIZER": 1}]
        outputs = logits([(0, ["FERTILIZE"])])
        old = sample_action(before, outputs, None)
        new = economic(before, outputs)
        self.assertEqual(old.action["farmer"], ["FERTILIZE"])
        self.assertEqual(new.action["farmer"], ["PASS"])
        self.assertFalse(new.unit_legal_mask[0, UNIT_ACTION_TO_ID[("FERTILIZE",)]])

    def test_care_requires_a_later_collectable_production_night(self):
        animal = rules._new_animal("GOOSE", 20)
        self.assertTrue(care_gains(animal, 27))
        self.assertFalse(care_gains(animal, 28))
        before = observation(28)
        before["farms"][0]["tiles"][4][4] = animal
        selected = economic(before, logits([(0, ["CARE"])]))
        self.assertEqual(selected.action["farmer"], ["PASS"])
        self.assertFalse(selected.unit_legal_mask[0, UNIT_ACTION_TO_ID[("CARE",)]])

    def test_late_production_and_same_night_hiring_are_masked(self):
        before = observation(28, 23)
        before["private"]["seeds"] = {"WHEAT": 1}
        outputs = logits([(0, ["PLANT", "WHEAT"])],
                         [(0, ["BUY_SEED", "WHEAT", 1]), (1, ["HIRE"]), (2, ["BUY_LAND"])])
        selected = economic(before, outputs)
        self.assertEqual(selected.action["farmer"], ["PASS"])
        self.assertTrue((selected.market_action == NOOP_ID).all())
        self.assertFalse(selected.market_legal_mask[:, MARKET_ACTION_TO_ID[("HIRE",)]].any())
        self.assertFalse(selected.market_legal_mask[:, MARKET_ACTION_TO_ID[("BUY_LAND",)]].any())
        # A first harvest on the final morning is allowed, rather than rejecting
        # every late investment indiscriminately.
        earlier = observation(27, 1)
        earlier["private"]["seeds"] = {"WHEAT": 1}
        self.assertEqual(economic(earlier, outputs).action["farmer"], ["PLANT", "WHEAT"])

    def test_final_drop_and_sales_are_executed_with_zero_policy_factors(self):
        before = observation(29, 22)
        before["private"]["shed"] = {"WHEAT": 2}
        before["private"]["inventories"] = [{"CARROT": 3}]
        selected = economic(before, logits([(0, ["NORTH"])], [(0, ["HIRE"])]))
        self.assertEqual(selected.action["farmer"], ["DROP"])
        self.assertEqual({tuple(order) for order in selected.action["market"] if order[0] == "SELL"},
                         {("SELL", "WHEAT", 2), ("SELL", "CARROT", 3)})
        self.assertTrue(selected.unit_mask[0])
        self.assertFalse(selected.unit_policy_mask.any())
        self.assertFalse(selected.market_policy_mask.any())
        self.assertEqual(float(selected.old_log_prob), 0)
        self.assertTrue((selected.unit_legal_mask.sum(-1) == 1).all())
        self.assertTrue((selected.market_legal_mask.sum(-1) == 1).all())
        resolver = PrefixResolver(before)
        resolver.apply_unit(0, int(selected.unit_action[0]))
        for identity in selected.market_action:
            resolver.apply_market(int(identity))
        self.assertEqual(sum(resolver.private["shed"].values()), 0)
        self.assertEqual(sum(resolver.private["inventories"][0].values()), 0)

    def test_night_sales_use_actual_post_unit_inventory_and_preserve_feed(self):
        before = observation(15, 23)
        before["private"]["shed"] = {"WHEAT": 1, "CARROT": 99}
        before["private"]["inventories"] = [{"WHEAT": 1}]
        before["farms"][0]["tiles"][0][0] = rules._new_animal("GOOSE", 0)
        before["farms"][0]["tiles"][0][1] = rules._new_animal("GOOSE", 0)
        selected = economic(before, logits(markets=[(1, ["BUY_PRODUCT", "WHEAT", 1])]))
        self.assertEqual(selected.action["market"][0], ["SELL", "CARROT", 1])
        self.assertFalse(selected.market_policy_mask[0])
        self.assertTrue(selected.market_policy_mask[1:].all())
        self.assertEqual(selected.action["market"][1], ["NOOP"])
        self.assertFalse(selected.market_legal_mask[1, MARKET_ACTION_TO_ID[("BUY_PRODUCT", "WHEAT", 1)]])
        resolver = PrefixResolver(before)
        for identity in selected.market_action:
            resolver.apply_market(int(identity))
        self.assertEqual(resolver.private["shed"]["WHEAT"], 1)
        self.assertEqual(sum(resolver.private["shed"].values()) + 1, 100)
        # A harvest creates new carried goods that were absent at observation
        # time; the liquidation decision must include the selected unit prefix.
        harvest = observation(15, 23)
        plant = rules._new_plant("WHEAT", 10, 24)
        plant["yield_units"] = 6
        harvest["farms"][0]["tiles"][4][4] = plant
        harvest["private"]["shed"] = {"CARROT": 100}
        harvested = economic(harvest, logits([(0, ["HARVEST"])]))
        self.assertEqual(harvested.action["market"][0], ["SELL", "CARROT", 6])

    def test_forced_slots_are_excluded_from_probability_recomputation(self):
        before = observation(10, 23)
        before["private"]["shed"] = {"CARROT": 100}
        before["private"]["inventories"] = [{"WHEAT": 2}]
        outputs = logits()
        selected = economic(before, outputs, np.random.default_rng(904))
        arrays = {name: jnp.asarray(value[None]) for name, value in selected.arrays().items()}
        model = {name: jnp.asarray(value[None]) for name, value in outputs.items()}
        masked = behavior_outputs(model, arrays)
        recomputed = joint_log_prob(masked, arrays["unit_action"], arrays["market_action"],
                                    arrays["unit_policy_mask"], arrays["market_policy_mask"])
        np.testing.assert_allclose(jax.device_get(recomputed), [selected.old_log_prob], atol=2e-5)
        self.assertEqual(selected.old_market_log_prob[0], 0)
        self.assertFalse(selected.market_policy_mask[0])

    def test_v1_keeps_its_original_unpatched_action(self):
        before = observation(29, 22)
        before["private"]["shed"] = {"WHEAT": 2}
        before["private"]["inventories"] = [{"CARROT": 3}]
        output = logits([(0, ["NORTH"])])
        default = sample_action(before, output, None)
        explicit = sample_action(before, output, None, action_selection_contract=ACTION_SELECTION_CONTRACT)
        self.assertEqual(default.action, explicit.action)
        self.assertEqual(default.action["farmer"], ["NORTH"])
        self.assertTrue((default.market_action == NOOP_ID).all())
        np.testing.assert_array_equal(default.unit_policy_mask, default.unit_mask)
        np.testing.assert_array_equal(default.market_policy_mask, default.market_mask)

    def test_economic_sampling_uses_real_workers_after_rehire_and_shared_seeds(self):
        environment = make("kaggriculture", configuration={"seed": 180711}, debug=False)
        environment.step([{"market": [["HIRE"], ["BUY_SEED", "WHEAT", 1]]}, {}])
        before = deepcopy(dict(environment.state[0].observation))
        chosen = economic(before)
        self.assertEqual(chosen.unit_mask.sum(), 2)
        for _ in range(23):
            environment.step([{}, {}])
        expired = deepcopy(dict(environment.state[0].observation))
        self.assertEqual(economic(expired).unit_mask.sum(), 1)
        self.assertFalse(economic(expired).unit_policy_mask[1:].any())
        environment.step([{"market": [["HIRE"]]}, {}])
        rehired = deepcopy(dict(environment.state[0].observation))
        self.assertEqual(economic(rehired).unit_mask.sum(), 2)
        # Use that actual rehired unit on a separate tile: its PLANT cannot
        # borrow the seed already consumed by the farmer's chosen prefix.
        rehired["farms"][0]["hands"][0] = [3, 4]
        selected = economic(rehired, logits([(0, ["PLANT", "WHEAT"]),
                                             (1, ["PLANT", "WHEAT"])]))
        self.assertEqual(selected.action["farmer"], ["PLANT", "WHEAT"])
        self.assertEqual(selected.action["hands"], [["PASS"]])
        self.assertFalse(selected.unit_legal_mask[1, UNIT_ACTION_TO_ID[("PLANT", "WHEAT")]])

    def test_economic_windows_read_the_selected_unit_prefix(self):
        environment = make("kaggriculture", configuration={"seed": 180711}, debug=False)
        environment.step([{"market": [["HIRE"]]}, {}])
        before = deepcopy(dict(environment.state[0].observation))
        before.update(day=4, hour=0, step=96)
        before["farms"][0]["hands"][0] = [4, 4]
        plant = rules._new_plant("WHEAT", 0, 24)
        before["farms"][0]["tiles"][4][4] = plant
        before["private"]["inventories"] = [{}, {"FERTILIZER": 1}]
        output = logits([(0, ["WATER"]), (1, ["FERTILIZE"])])
        selected = economic(before, output)
        self.assertEqual(selected.action["farmer"], ["WATER"])
        self.assertEqual(selected.action["hands"], [["PASS"]])
        # Before WATER, fertilizing still has a useful same-day watering window.
        self.assertTrue(fertilize_gains(plant, 4))
        self.assertFalse(selected.unit_legal_mask[1, UNIT_ACTION_TO_ID[("FERTILIZE",)]])

    def test_external_controller_records_actions_without_a_policy_probability(self):
        before = observation()
        external = external_action_sample({"farmer": ["NORTH"], "market": [["SELL", "WHEAT", 3]]}, before)
        self.assertEqual(external.action["farmer"], ["NORTH"])
        self.assertTrue(external.unit_mask[0])
        self.assertFalse(external.unit_policy_mask.any())
        self.assertFalse(external.market_policy_mask.any())
        self.assertEqual(external.old_log_prob, 0)
        self.assertTrue((external.unit_legal_mask.sum(-1) == 1).all())
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            external_action_sample({"market": [["SELL", "WHEAT", 101]]}, before)
        with self.assertRaisesRegex(ValueError, "capacity"):
            external_action_sample({"hands": [["PASS"]]}, before)


if __name__ == "__main__":
    unittest.main()

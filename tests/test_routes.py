"""Reservation, dependency, execution and season-return regressions."""
import copy
import hashlib
import importlib.util
import unittest
from route_rl.check import fixture, structural
from route_rl.controller import RouteController, receipt_matches
from route_rl.features import FEATURE_SIZE, STATE_SIZE, menu_features, state_features
from route_rl.paths import BASELINE
from route_rl.routes import PlanningState, Job, CROPS, farm_of, project, positions, legal
from route_rl.training import _advantages

def take(plan, actor, op, pos=None):
    job = next(j for j in plan.candidates(actor) if j.op == op and (pos is None or j.pos == pos))
    plan.commit(job)
    return plan.jobs[-1]

def scripted(sequences):
    def choose(plan, jobs):
        actor = jobs[0].actor
        index = sum(j.actor == actor for j in plan.jobs)
        sequence = sequences.get(actor, [])
        if index >= len(sequence):
            return 0
        op, pos = sequence[index]
        return next(i for i, j in enumerate(jobs) if j.op == op and (pos is None or j.pos == pos))
    return choose

def tick(obs, action):
    """Test-only immediate transitions; Rust integration separately verifies rules."""
    after = copy.deepcopy(obs)
    for actor, op in enumerate([action["farmer"]] + action["hands"]):
        if op == ["PASS"]:
            continue
        p = positions(after)[actor]
        if op[0] in ("NORTH", "SOUTH", "EAST", "WEST"):
            dx, dy = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}[op[0]]
            if actor == 0:
                farm_of(after)["farmer"] = [p[0] + dx, p[1] + dy]
            else:
                farm_of(after)["hands"][actor - 1] = [p[0] + dx, p[1] + dy]
        else:
            job = Job(actor, p, tuple(op))
            if legal(after, job):
                project(after, job)
    for op in action["market"]:
        job = Job(0, positions(after)[0], tuple(op))
        if legal(after, job):
            project(after, job)
    after["step"] += 1
    after["hour"] += 1
    return after

class DynamicRoutesTests(unittest.TestCase):
    def test_repository_assets_and_check(self):
        self.assertEqual(hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
            "aa623df1a03567d4a1ac40fb7113295838189d19ebb4b9832e9374bb8fc86ff9")
        self.assertGreater(structural()["candidate_count"], 24)

    def test_construct_mixed_route_one_job_at_a_time(self):
        obs = fixture()
        obs["private"]["seeds"]["STRAWBERRY"] = 1
        plan = PlanningState(obs)
        sequence = [
            (("PICKUP", "WHEAT", 1), (4, 4)), (("FEED",), (2, 3)),
            (("CARE",), (2, 3)), (("COLLECT_FERTILIZER",), (2, 3)),
            (("FERTILIZE",), (3, 3)), (("WATER",), (3, 3)),
            (("HARVEST",), (3, 3)), (("PLANT", "STRAWBERRY"), (3, 3)),
            (("WATER",), (3, 3)), (("DROP",), (4, 4))]
        for op, pos in sequence:
            take(plan, 0, op, pos)
        self.assertEqual(farm_of(plan.obs)["tiles"][3][3]["crop"], "STRAWBERRY")
        self.assertGreater(plan.obs["private"]["shed"]["WHEAT"], 8)
        self.assertEqual(farm_of(obs)["tiles"][3][3]["crop"], "WHEAT")
        self.assertEqual(len(plan.jobs), 10)
        self.assertLessEqual(plan.elapsed[0], 24)

    def test_no_profit_or_topk_pruning(self):
        obs = fixture()
        obs.update(day=29, step=696, hour=0)
        obs["private"]["seeds"] = dict.fromkeys(CROPS, 1)
        plan = PlanningState(obs)
        jobs = plan.candidates(0)
        self.assertGreater(len(jobs), 24)
        for crop in CROPS:
            self.assertTrue(any(j.op == ("PLANT", crop) and j.pos == (0, 0) for j in jobs))

    def test_harvest_seed_stock_and_cash_reserved(self):
        obs = fixture()
        obs["private"]["seeds"] = {"WHEAT": 1}
        obs["private"]["shed"] = {"WHEAT": 1}
        farm_of(obs)["money"] = 10
        plan = PlanningState(obs)
        take(plan, 0, ("HARVEST",), (3, 3))
        self.assertFalse(any(j.op == ("HARVEST",) and j.pos == (3, 3) for j in plan.candidates(1)))
        take(plan, 0, ("PLANT", "WHEAT"), (3, 3))
        self.assertFalse(any(j.op == ("PLANT", "WHEAT") for j in plan.candidates(1)))
        take(plan, 0, ("PICKUP", "WHEAT", 1), (4, 4))
        self.assertFalse(any(j.op[0] == "PICKUP" for j in plan.candidates(1)))
        take(plan, 0, ("BUY_SEED", "WHEAT", 1))
        self.assertFalse(any(j.op[0].startswith("BUY") for j in plan.candidates(1)))

    def test_cross_worker_sale_purchase_dependencies(self):
        obs = fixture()
        farm_of(obs)["money"] = 0
        obs["market"]["prices"]["WHEAT"] = 100
        plan = PlanningState(obs)
        harvest = take(plan, 0, ("HARVEST",), (3, 3))
        drop = take(plan, 0, ("DROP",), (4, 4))
        sell = take(plan, 0, ("SELL", "WHEAT", plan.obs["private"]["shed"]["WHEAT"]))
        build = take(plan, 1, ("BUILD_PASTURE",), (4, 3))
        buy = take(plan, 1, ("BUY_ANIMAL", "SHEEP", 1))
        pickup = take(plan, 1, ("PICKUP", "SHEEP", 1), (4, 4))
        place = take(plan, 1, ("PLACE", "SHEEP"), (4, 3))
        self.assertIn(harvest.id, drop.deps)
        self.assertIn(drop.id, sell.deps)
        self.assertIn(sell.id, buy.deps)
        self.assertIn(buy.id, pickup.deps)
        self.assertIn(build.id, place.deps)
        self.assertIn(pickup.id, place.deps)
        self.assertGreater(buy.finish, sell.finish)

    def test_land_can_change_from_crop_to_livestock_and_back(self):
        obs = fixture()
        obs["private"]["inventories"][0] = {"COW": 1}
        plan = PlanningState(obs)
        take(plan, 0, ("HARVEST",), (3, 3))
        take(plan, 0, ("BUILD_PASTURE",), (3, 3))
        take(plan, 0, ("PLACE", "COW"), (3, 3))
        self.assertFalse(any(j.op == ("DIG",) and j.pos == (3, 3) for j in plan.candidates(0)))
        # The engine removes an unfed animal after two days; only then DIG.
        farm_of(obs)["tiles"][3][3] = {"kind": "PASTURE"}
        plan = PlanningState(obs)
        take(plan, 0, ("DIG",), (3, 3))
        take(plan, 0, ("PLANT", "WHEAT"), (3, 3))

    def test_work_time_includes_wait_for_other_worker(self):
        plan = PlanningState(fixture(), horizon=3)
        self.assertFalse(any(j.pos == (0, 0) and j.op[0] == "PLANT" for j in plan.candidates(0)))
        obs = fixture()
        obs["private"]["seeds"] = {"WHEAT": 1}
        plan = PlanningState(obs)
        buy = take(plan, 0, ("BUY_SEED", "STRAWBERRY", 1))
        plant = take(plan, 1, ("PLANT", "STRAWBERRY"), (4, 3))
        self.assertIn(buy.id, plant.deps)
        self.assertGreater(plant.finish, buy.finish)

    def test_controller_never_calls_baseline_after_takeover(self):
        def forbidden(*args):
            self.fail("baseline must not allocate work")
        c = RouteController(forbidden, choose=lambda *_: 0)
        action = c.act(fixture())
        self.assertEqual(action, {"farmer": ["PASS"], "hands": [["PASS"]], "market": []})

    def test_executor_completes_harvest_replant_route(self):
        obs = fixture()
        obs["private"]["seeds"]["STRAWBERRY"] = 1
        choose = scripted({0: [(("HARVEST",), (3, 3)),
                               (("PLANT", "STRAWBERRY"), (3, 3)),
                               (("WATER",), (3, 3)), (("DROP",), (4, 4))]})
        c = RouteController(choose=choose, replan_interval=24)
        for _ in range(8):
            obs = tick(obs, c.act(obs))
        self.assertEqual(farm_of(obs)["tiles"][3][3]["crop"], "STRAWBERRY")
        self.assertTrue(farm_of(obs)["tiles"][3][3]["watered_today"])
        self.assertEqual(c.stats["aborts"], 0)
        self.assertGreater(obs["private"]["shed"]["WHEAT"], 8)

    def test_purchase_receipt_gates_other_worker_pickup(self):
        obs = fixture()
        choose = scripted({0: [(("BUY_ANIMAL", "SHEEP", 1), None)],
                           1: [(("PICKUP", "SHEEP", 1), (4, 4))]})
        c = RouteController(choose=choose, replan_interval=24)
        first = c.act(obs)
        self.assertEqual(first["market"], [["BUY_ANIMAL", "SHEEP", 1]])
        self.assertNotEqual(first["hands"][0][0], "PICKUP")
        obs = tick(obs, first)
        second = c.act(obs)
        self.assertEqual(second["hands"][0], ["PICKUP", "SHEEP", 1])

    def test_existing_inventory_does_not_fake_purchase_receipt(self):
        obs = fixture()
        obs["private"]["shed"]["SHEEP"] = 1
        job = Job(0, (4, 4), ("BUY_ANIMAL", "SHEEP", 1))
        self.assertFalse(receipt_matches(obs, copy.deepcopy(obs), job))
        after = copy.deepcopy(obs)
        project(after, job)
        self.assertTrue(receipt_matches(obs, after, job))

    def test_sale_shortfall_blocks_purchase_and_releases_plan(self):
        obs = fixture()
        farm_of(obs)["money"] = 0
        obs["market"]["prices"]["WHEAT"] = 100
        choose = scripted({0: [(("SELL", "WHEAT", 8), None)],
                           1: [(("BUY_ANIMAL", "SHEEP", 1), None)]})
        c = RouteController(choose=choose, replan_interval=24)
        first = c.act(obs)
        self.assertEqual(first["market"], [["SELL", "WHEAT", 8]])
        after = tick(obs, first)
        farm_of(after)["money"] = 100  # Actual sale yielded less than projected.
        second = c.act(after)
        self.assertEqual(second["market"], [])
        self.assertIsNone(c.plan)

    def test_daily_reset_discards_worker_reservations(self):
        obs = fixture()
        c = RouteController(choose=lambda *_: 0)
        c.act(obs)
        obs.update(day=7, hour=0, step=168)
        farm_of(obs)["hands"] = []
        obs["private"]["inventories"] = [{}]
        action = c.act(obs)
        self.assertEqual(action["hands"], [])
        self.assertEqual(len(c.plan.elapsed), 1)

    def test_trace_distinguishes_plan_execution_and_receipt(self):
        obs, events = fixture(), []
        c = RouteController(choose=scripted({0: [(("BUY_SEED", "STRAWBERRY", 1), None)]}),
                            trace=events.append)
        first = c.act(obs)
        c.choose = lambda *_: 0
        c.act(tick(obs, first))
        self.assertTrue(any(e["event"] == "plan" and e["jobs"] for e in events))
        self.assertTrue(any(e["event"] == "action" and e["action"]["market"] for e in events))
        self.assertTrue(any(e["event"] == "receipt" and e["success"] for e in events))

    def test_shed_overflow_is_not_silently_discarded(self):
        obs = fixture()
        obs["private"]["shed"] = {"WHEAT": 100}
        obs["private"]["inventories"][0] = {"WOOL": 6}
        plan = PlanningState(obs)
        self.assertFalse(any(j.op[0] == "DROP" for j in plan.candidates(0)))
        sale = take(plan, 1, ("SELL", "WHEAT", 100))
        drop = take(plan, 0, ("DROP",), (4, 4))
        self.assertIn(sale.id, drop.deps)

    def test_planning_limits_are_validated(self):
        with self.assertRaises(ValueError):
            RouteController(horizon=0)
        with self.assertRaises(ValueError):
            RouteController(replan_interval=0)

    def test_features_include_changed_planning_state(self):
        plan = PlanningState(fixture())
        before = state_features(plan)
        take(plan, 0, ("HARVEST",), (3, 3))
        self.assertNotEqual(before, state_features(plan))
        self.assertEqual(len(before), STATE_SIZE)
        jobs = plan.candidates(0)
        feats, mask = menu_features(plan, jobs)
        self.assertTrue(all(len(f) == FEATURE_SIZE for f in feats))
        self.assertEqual(len(mask), len(jobs))

    def test_late_cash_reaches_earlier_investment(self):
        rows = [{"reward": -1.0, "value": 0.2},
                {"reward": 0.0, "value": -0.1},
                {"reward": 4.0, "value": 0.8}]
        _advantages([rows])
        self.assertAlmostEqual(rows[0]["return"], 3.0)
        self.assertAlmostEqual(rows[1]["return"], 4.0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional")
    def test_ppo_accepts_variable_menus_and_updates(self):
        import torch
        from route_rl.training import Network, update
        torch.set_num_threads(1)
        torch.manual_seed(1)
        net = Network(torch)
        plan, rows = PlanningState(fixture()), []
        for _ in range(2):
            jobs = plan.candidates(0)
            feat, mask = menu_features(plan, jobs)
            state = state_features(plan)
            with torch.no_grad():
                logits, value = net(torch.tensor([feat]), torch.tensor([mask]),
                                    torch.tensor([state]))
                dist = torch.distributions.Categorical(logits=logits)
            idx = next(i for i, j in enumerate(jobs) if j.op[0] == "HARVEST")
            rows.append(dict(features=feat, mask=mask, state=state, action=idx,
                             logp=dist.log_prob(torch.tensor([idx])).item(),
                             value=value.item(), reward=float(len(rows))))
            plan.commit(jobs[idx])
        self.assertNotEqual(len(rows[0]["features"]), len(rows[1]["features"]))
        before = next(net.module.parameters()).detach().clone()
        result = update(net, torch.optim.Adam(net.module.parameters()), torch, [rows], "cpu")
        self.assertEqual(result["samples"], 2)
        self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))
        self.assertFalse(torch.equal(before, next(net.module.parameters())))

if __name__ == "__main__":
    unittest.main()

"""Optional real-Rust regressions; run automatically when kagg is built."""
import copy
import importlib.util
import unittest
from route_rl.check import fixture
from route_rl.controller import RouteController
from route_rl.paths import add_kaggsim
from test_routes import scripted
add_kaggsim()
from kaggsim.binary import find_kagg
from kaggsim.serve import Serve, obs_for

try:
    find_kagg()
    HAVE_ENGINE = True
except FileNotFoundError:
    HAVE_ENGINE = False

IDLE = {"farmer": ["PASS"], "hands": [], "market": []}

@unittest.skipUnless(HAVE_ENGINE, "build Rust simulator to run integration tests")
class EngineTests(unittest.TestCase):
    def state(self, srv):
        state = srv.reset(42)
        obs = fixture()
        state["step"] = obs["step"]
        state["farms"][0] = copy.deepcopy(obs["farms"][0])
        state["private"][0] = copy.deepcopy(obs["private"])
        return state

    def test_mixed_route_on_real_engine(self):
        with Serve() as srv:
            state = self.state(srv)
            state["private"][0]["seeds"]["STRAWBERRY"] = 1
            state = srv.load_state(state, 42)
            sequence = [
                (("PICKUP", "WHEAT", 1), (4, 4)), (("FEED",), (2, 3)),
                (("CARE",), (2, 3)), (("COLLECT_FERTILIZER",), (2, 3)),
                (("FERTILIZE",), (3, 3)), (("WATER",), (3, 3)),
                (("HARVEST",), (3, 3)), (("PLANT", "STRAWBERRY"), (3, 3)),
                (("WATER",), (3, 3)), (("DROP",), (4, 4))]
            controller = RouteController(choose=scripted({0: sequence}), replan_interval=24)
            actions = []
            for _ in range(24):
                action = controller.act(obs_for(state, 0))
                actions.append(action["farmer"][0])
                state = srv.step2(action, IDLE)
                if action["farmer"][0] == "DROP":
                    controller.choose = lambda *_: 0
                    controller.act(obs_for(state, 0))
                    break
            tile = state["farms"][0]["tiles"][3][3]
            self.assertEqual(tile["crop"], "STRAWBERRY")
            self.assertTrue(tile["watered_today"])
            self.assertGreater(state["private"][0]["shed"]["WHEAT"], 8)
            self.assertEqual(controller.stats["aborts"], 0)
            self.assertIn("COLLECT_FERTILIZER", actions)
            self.assertIn("FERTILIZE", actions)

    def test_sale_finances_other_workers_livestock(self):
        with Serve() as srv:
            state = self.state(srv)
            state["farms"][0]["money"] = 0
            state["farms"][0]["farmer"] = [2, 3]
            state = srv.load_state(state, 42)
            choose = scripted({
                0: [(("HARVEST",), (2, 3)), (("DROP",), (4, 4)),
                    (("SELL", "WOOL", 6), None)],
                1: [(("HARVEST",), (3, 3)), (("BUILD_PASTURE",), (3, 3)),
                    (("BUY_ANIMAL", "SHEEP", 1), None),
                    (("PICKUP", "SHEEP", 1), (4, 4)), (("PLACE", "SHEEP"), (3, 3))]})
            controller = RouteController(choose=choose, replan_interval=24)
            sale_step = buy_step = pickup_step = None
            # Stop after final placement, before a fresh scripted plan is requested.
            for step in range(24):
                action = controller.act(obs_for(state, 0))
                if action["market"]:
                    if action["market"][0][0] == "SELL":
                        sale_step = step
                    if action["market"][0][0] == "BUY_ANIMAL":
                        buy_step = step
                if action["hands"][0][0] == "PICKUP":
                    pickup_step = step
                state = srv.step2(action, IDLE)
                if (state["farms"][0]["tiles"][3][3] or {}).get("animal") == "SHEEP":
                    break
            self.assertEqual(state["farms"][0]["tiles"][3][3]["animal"], "SHEEP")
            self.assertLess(sale_step, buy_step)
            self.assertLess(buy_step, pickup_step)
            self.assertGreater(state["farms"][0]["money"], 0)
            self.assertEqual(controller.stats["aborts"], 0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional")
    def test_full_season_cash_is_conserved_in_training_trajectory(self):
        import torch
        from route_rl.training import episode
        torch.set_num_threads(1)
        class EndPolicy:
            def __call__(self, features, mask, state):
                logits = torch.full(mask.shape, -1e9)
                logits[:, 0] = 0  # One END decision per planning window.
                return logits, torch.zeros(mask.shape[0])
        with Serve() as srv:
            rows, banks, stats = episode(srv, EndPolicy(), torch, 51031, 0, "cpu")
        self.assertGreater(len(rows), 1)
        self.assertAlmostEqual(sum(r["reward"] for r in rows) * 10000,
                               banks[0] - banks[1], places=6)
        self.assertEqual(stats["purchases"], 0)

    def test_animal_exit_and_structure_reuse(self):
        with Serve() as srv:
            state = self.state(srv)
            state["farms"][0]["farmer"] = [2, 3]
            state["farms"][0]["hands"] = []
            state["private"][0]["inventories"] = [{}]
            state = srv.load_state(state, 42)
            for _ in range(48):
                state = srv.step2(IDLE, IDLE)
            self.assertNotIn("animal", state["farms"][0]["tiles"][3][2])
            state["farms"][0]["farmer"] = [2, 3]
            state = srv.load_state(state, 42)
            controller = RouteController(choose=scripted({0: [
                (("DIG",), (2, 3)), (("PLANT", "WHEAT"), (2, 3)), (("WATER",), (2, 3))]}),
                replan_interval=24)
            for _ in range(3):
                state = srv.step2(controller.act(obs_for(state, 0)), IDLE)
            self.assertEqual(state["farms"][0]["tiles"][3][2]["crop"], "WHEAT")

if __name__ == "__main__":
    unittest.main()

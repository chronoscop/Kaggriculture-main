"""Fast checks that do not require Rust, PyTorch, or the official runner."""
import copy
import hashlib
import unittest

from route_rl.check import fixture
from route_rl.controller import RouteController
from route_rl.features import FEATURE_SIZE, menu_features
from route_rl.paths import BASELINE, SIM_ROOT
from route_rl.routes import generate


def idle(observation, configuration=None):
    return {"farmer": ["PASS"], "hands": [["PASS"]], "market": []}


class RoutePipelineTests(unittest.TestCase):
    def test_repository_assets(self):
        self.assertTrue((SIM_ROOT / "src-rust" / "Cargo.toml").is_file())
        self.assertEqual(hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
                         "aa623df1a03567d4a1ac40fb7113295838189d19ebb4b9832e9374bb8fc86ff9")

    def test_candidate_menu_covers_crop_animal_and_manure(self):
        obs = fixture()
        routes = generate(obs)
        kinds = {route.kind for route in routes}
        self.assertIn("BASELINE", kinds)
        self.assertIn("MANURE_CROP", kinds)
        self.assertIn("RAISE_COW", kinds)
        self.assertIn("RAISE_SHEEP", kinds)
        for crop in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"):
            self.assertIn("PLANT_" + crop, kinds)
        features, mask = menu_features(obs, routes)
        self.assertEqual((len(features), len(mask)), (24, 24))
        self.assertTrue(all(len(row) == FEATURE_SIZE for row in features))
        self.assertEqual(sum(mask), len(routes))

    def test_purchase_is_confirmed_before_planting(self):
        obs = fixture()

        def choose(_obs, routes):
            return next(i for i, route in enumerate(routes)
                        if route.kind == "PLANT_STRAWBERRY")

        controller = RouteController(idle, choose, takeover=144)
        first = controller.act(obs)
        self.assertIn(["BUY_SEED", "STRAWBERRY", 1], first["market"])
        self.assertEqual(first["farmer"], ["PASS"])
        self.assertEqual(first["hands"], [["PASS"]])

        received = copy.deepcopy(obs)
        received["step"] = 145
        received["hour"] = 1
        received["private"]["seeds"]["STRAWBERRY"] = 1
        second = controller.act(received)
        self.assertNotIn(["BUY_SEED", "STRAWBERRY", 1], second["market"])
        self.assertTrue(second["farmer"] != ["PASS"] or second["hands"] != [["PASS"]])

    def test_failed_receipt_aborts_route(self):
        obs = fixture()
        controller = RouteController(idle, lambda _obs, routes:
            next(i for i, route in enumerate(routes) if route.kind == "PLANT_STRAWBERRY"),
            takeover=144)
        controller.act(obs)
        missing = copy.deepcopy(obs)
        missing["step"] = 145
        missing["hour"] = 1
        controller.act(missing)
        self.assertIsNone(controller.active)
        self.assertEqual(controller.stats["aborts"], 1)

    def test_daily_route_cap_preserves_incumbent(self):
        obs = fixture()
        controller = RouteController(idle, lambda *_: self.fail("should not choose"),
                                     takeover=144, max_routes_per_day=0)
        self.assertEqual(controller.act(obs), idle(obs))
        self.assertEqual(controller.stats["decisions"], 0)


if __name__ == "__main__":
    unittest.main()

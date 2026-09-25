"""Cheap structural checks; --official-steps also exercises the pinned game."""
from __future__ import annotations

import argparse
import copy
import json
from .paths import BASELINE, add_kaggsim

add_kaggsim()

from kaggsim import official
from kaggsim.fidelity import digest_json, digest_official
from kaggsim.serve import Serve, call_agent, load_agent, obs_for

from .controller import RouteController
from .features import FEATURE_SIZE, menu_features
from .routes import generate


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def fixture():
    tiles = [[None if x < 5 and y < 5 else "LOCKED" for x in range(10)]
             for y in range(10)]
    tiles[3][3] = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
                   "yield_units": 6, "watered_today": False}
    tiles[2][3] = {"kind": "PLANT", "crop": "CARROT", "planted_day": 2,
                   "yield_units": 1, "watered_today": False,
                   "fertilized_until_day": -1}
    tiles[3][2] = {"kind": "PASTURE", "animal": "SHEEP", "yield_units": 6,
                   "fertilizer_available": 1, "fed_today": False,
                   "cared_today": False}
    farm = {"money": 3000, "farmer": [4, 4], "hands": [[4, 3]],
            "unlocked_quadrants": ["NW"], "tiles": tiles}
    rival = copy.deepcopy(farm)
    return {"step": 144, "day": 6, "hour": 0, "player": 0,
            "farms": [farm, rival], "market": {"prices": {"WOOL": 100},
                    "inventory": {}}, "town": {"unlocked_shops": []},
            "private": {"shed": {"WHEAT": 8},
                        "seeds": {"WHEAT": 4, "CARROT": 4},
                        "inventories": [{}, {}]}}


def structural():
    obs = fixture()
    routes = generate(obs)
    kinds = {r.kind for r in routes}
    assert "BASELINE" in kinds
    assert any(k.startswith("HARVEST_PLANT") for k in kinds)
    assert "MANURE_CROP" in kinds
    assert "RAISE_COW" in kinds and "RAISE_SHEEP" in kinds
    feats, mask = menu_features(obs, routes)
    assert len(feats) == len(mask) == 24
    assert all(len(row) == FEATURE_SIZE for row in feats)
    assert sum(mask) == len(routes)

    # A work route is actually committed across multiple observations.
    target = next(i for i, r in enumerate(routes) if r.kind == "MANURE_CROP")
    baseline = lambda observation, configuration=None: {
        "farmer": ["PASS"], "hands": [["PASS"]], "market": []}
    controller = RouteController(baseline, lambda _obs, _routes: target)
    action = controller.act(obs)
    assert action["farmer"][0] in ("NORTH", "WEST") or action["hands"][0][0] in ("NORTH", "WEST")
    assert controller.active is not None
    return {"route_count": len(routes), "kinds": sorted(kinds),
            "first_action": action}


def official_steps(n, takeover):
    engine = official.engine_module()
    config = AttrDict(dict(official.DEFAULT_CONFIGURATION, seed=42))
    env = AttrDict(configuration=config, info={}, done=False)
    state = [AttrDict(observation=AttrDict(step=0), action={},
                      status="ACTIVE", reward=0) for _ in range(2)]
    engine.interpreter(state, env)
    baseline = load_agent(str(BASELINE))
    opponent = load_agent(str(BASELINE))
    agent_config = AttrDict(dict(official.DEFAULT_CONFIGURATION))
    controller = RouteController(baseline, lambda obs, routes: min(1, len(routes) - 1),
                                 takeover=takeover)
    actions, digests = [], []
    for _ in range(n):
        obs0 = copy.deepcopy(state[0].observation)
        obs1 = copy.deepcopy(state[1].observation)
        state[0].action = controller.act(obs0, agent_config)
        state[1].action = opponent(obs1, agent_config)
        actions.append((copy.deepcopy(state[0].action), copy.deepcopy(state[1].action)))
        engine.interpreter(state, env)
        for row in state:
            row.observation.step += 1
        digests.append(digest_official(state))
    return {"steps": n, "step": state[0].observation.step,
            "cash": [s.observation.farms[i]["money"] for i, s in enumerate(state)],
            "controller": controller.stats}, actions, digests


def rust_steps(n, takeover):
    baseline = load_agent(str(BASELINE))
    opponent = load_agent(str(BASELINE))
    controller = RouteController(baseline, lambda obs, routes: min(1, len(routes) - 1),
                                 takeover=takeover)
    with Serve() as srv:
        state = srv.reset(42)
        for _ in range(n):
            own = call_agent(controller.act, obs_for(state, 0))
            other = call_agent(opponent, obs_for(state, 1))
            state = srv.step2(own, other)
    return {"steps": n, "step": state["step"],
            "cash": [state["farms"][i]["money"] for i in range(2)],
            "controller": controller.stats}


def rust_replay(actions, expected_digests):
    with Serve() as srv:
        state = srv.reset(42)
        for step, ((action0, action1), want) in enumerate(zip(actions, expected_digests), 1):
            state = srv.step2(action0, action1)
            if digest_json(state) != want:
                raise AssertionError(f"official/Rust full-state divergence at step {step}")
    return {"steps": len(actions), "full_state_equal": True,
            "cash": [state["farms"][i]["money"] for i in range(2)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-steps", type=int, default=0)
    parser.add_argument("--rust-steps", type=int, default=0)
    parser.add_argument("--takeover", type=int, default=144)
    args = parser.parse_args()
    result = {"structural": structural()}
    if args.official_steps:
        report, actions, digests = official_steps(args.official_steps, args.takeover)
        result["official"] = report
    if args.rust_steps:
        if args.official_steps == args.rust_steps:
            result["rust"] = rust_replay(actions, digests)
        else:
            result["rust"] = rust_steps(args.rust_steps, args.takeover)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

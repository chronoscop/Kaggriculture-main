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
from .routes import PlanningState, CROPS


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
    tiles[3][3] = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 3, "max_lifespan_step": 192,
                   "yield_units": 6, "watered_today": False}
    tiles[2][3] = {"kind": "PLANT", "crop": "CARROT", "planted_day": 4, "max_lifespan_step": 192,
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
    plan = PlanningState(obs)
    routes = plan.candidates(0)
    ops = {r.op[0] for r in routes}
    assert {"END", "WAIT", "HARVEST", "CARE", "BUY_ANIMAL"} <= ops
    assert {r.op[1] for r in routes if r.op[0] == "BUY_SEED"} == set(CROPS)
    feats, mask = menu_features(plan, routes)
    assert len(feats) == len(mask) == len(routes)
    assert all(len(row) == FEATURE_SIZE for row in feats)
    controller = RouteController(choose=lambda p, jobs: next(
        (i for i, j in enumerate(jobs) if j.op[0] == "HARVEST"), 0))
    action = controller.act(obs)
    assert controller.plan.jobs
    return {"candidate_count": len(routes), "ops": sorted(ops),
            "first_action": action}


def probe_choice(plan, jobs):
    # Diagnostic policy only; training never uses these priorities.
    for op in ("HARVEST", "COLLECT_FERTILIZER", "WATER", "CARE", "DROP", "SELL"):
        for i, job in enumerate(jobs):
            if job.op[0] == op:
                return i
    return 0


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
    controller = RouteController(baseline, probe_choice,
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
    controller = RouteController(baseline, probe_choice,
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
    parser.add_argument("--takeover", type=int, default=0)
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

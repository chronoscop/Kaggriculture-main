"""Regression (arity, configuration, per-seat isolation): per-seat
observation isolation and official
invocation. A seat that vandalises its observation must not change what
the other seat sees, nor the game."""
import json

import pytest

from kaggsim import official
from kaggsim.policies import ScriptedFarmer
from kaggsim.serve import Struct, call_agent, obs_for, run_match


class Vandal:
    """Plays like ScriptedFarmer, then mutates every part of its obs."""

    def __init__(self, seed):
        self.inner = ScriptedFarmer(seed)

    def __call__(self, obs, configuration):
        # Attribute access on the configuration, as on the official runner.
        assert configuration.episodeSteps == 720
        a = self.inner(obs)
        obs.farms[0]["money"] = -1
        obs.farms[1]["tiles"][0][0] = "LOCKED"
        obs.market["prices"].clear()
        obs.market.inventory["WHEAT"] = 0
        obs.town["unlocked_shops"].append("NOWHERE")
        obs.private["shed"]["WHEAT"] = 999
        obs.step = -5
        return a


class Witness:
    def __init__(self, seed):
        self.inner, self.seen = ScriptedFarmer(seed), []

    def __call__(self, obs, configuration=None):
        self.seen.append(json.dumps(obs, sort_keys=True))
        return self.inner(obs)


def test_seat_views_are_independent_copies():
    js = {"step": 3, "day": 0, "hour": 3,
          "farms": [{"money": 1}, {"money": 2}], "market": {"prices": {}},
          "town": {"unlocked_shops": []},
          "private": [{"shed": {"A": 1}}, {"shed": {"B": 2}}]}
    o0, o1 = obs_for(js, 0), obs_for(js, 1)
    assert o0.private == {"shed": {"A": 1}} and o1.private["shed"] == {"B": 2}
    assert o0.player == 0 and o1.player == 1
    o0.farms[0]["money"] = 99
    assert o1.farms[0]["money"] == 1 and js["farms"][0]["money"] == 1


def test_call_agent_respects_arity():
    # Functions get args truncated to co_argcount; other callables (no
    # __code__) get both, exactly as the official runner does.
    assert call_agent(lambda o: "one", {}) == "one"
    assert isinstance(call_agent(lambda o, c: c, {}), Struct)

    class Both:
        def __call__(self, o, c):
            return c

    assert isinstance(call_agent(Both(), {}), Struct)


@pytest.mark.official
def test_vandal_changes_nothing(official_mod, serve):
    seed = 77
    w_clean = Witness(2)
    clean = run_match(ScriptedFarmer(1), w_clean, seed, serve)
    w_vandal = Witness(2)
    vandal = run_match(Vandal(1), w_vandal, seed, serve)
    assert vandal == clean
    assert w_vandal.seen == w_clean.seen
    off, env = official.run_agents(Vandal(1), Witness(2), seed)
    assert [s["status"] for s in env.steps[-1]] == ["DONE", "DONE"]
    assert off == vandal

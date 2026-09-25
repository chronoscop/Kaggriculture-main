"""replay_state: continue a recorded game from any step on the Rust engine."""
import json

import pytest

from kaggsim import official
from kaggsim.policies import RandomPolicy, ScriptedFarmer
from kaggsim.tape import action_to_line, replay_actions, replay_state

pytestmark = pytest.mark.official


def test_resume_mid_game_reproduces_the_recorded_ending(official_mod, serve):
    _, env = official.run_agents(ScriptedFarmer(4), RandomPolicy(5), 21)
    rep = json.loads(json.dumps(env.toJSON()))
    final = [float(x) for x in rep["rewards"]]
    for t in (0, 150, 400, 700):
        st = replay_state(rep, t)
        loaded = serve.load_state(st)
        assert loaded["step"] == t
        rest0 = [action_to_line(a) for a in replay_actions(rep, 0)[t:]]
        rest1 = [action_to_line(a) for a in replay_actions(rep, 1)[t:]]
        out = serve.rollout(st, 719 - t, rest0, rest1)
        assert out["step"] == 719
        assert [out["farms"][0]["money"], out["farms"][1]["money"]] == final
    with pytest.raises(ValueError):
        replay_state(rep, 10_000)

"""Regression (episode length): an episode is 719 actions (steps 0..718) and the final
bank is read from the step-719 state. A driver that also applied a step-719
action (and the day-30 end-of-day it triggers) would bank differently for a
policy that acts on the last turn."""
import os

import pytest

from kaggsim import official
from kaggsim.batch import run_batch
from kaggsim.constants import FINAL_STEP
from kaggsim.policies import LastTurnSeller, RandomPolicy
from kaggsim.serve import run_match
from kaggsim.tape import action_to_line, write_tape

pytestmark = pytest.mark.official


class Counting:
    def __init__(self, inner):
        self.inner, self.steps = inner, []

    def __call__(self, obs, configuration=None):
        self.steps.append(obs["step"])
        return self.inner(obs, configuration)


def test_last_turn_policy_banks_identically(official_mod, serve, tmp_path):
    seed = 424
    off0, off1 = Counting(LastTurnSeller(3)), Counting(RandomPolicy(4))
    (ob0, ob1), env = official.run_agents(off0, off1, seed)
    # The official runner solicits exactly 719 actions: steps 0..718.
    assert off0.steps == list(range(FINAL_STEP))
    # The policy WOULD act on step 719 if asked: a driver that stepped once
    # more would change its bank.
    last_obs = env.steps[-1][0]["observation"]
    assert last_obs["step"] == FINAL_STEP
    assert LastTurnSeller(3)(last_obs)["market"], "control: acts at 719"

    rs0, rs1 = Counting(LastTurnSeller(3)), Counting(RandomPolicy(4))
    rust = run_match(rs0, rs1, seed, serve)
    assert rs0.steps == list(range(FINAL_STEP))
    assert rust == (ob0, ob1)

    # kagg batch on the recorded streams, with an EXTRA step-719 line that
    # buys: batch must ignore it and reproduce the official bank.
    acts = [s[0]["action"] for s in env.steps[1:]]
    opp = [s[1]["action"] for s in env.steps[1:]]
    extra = action_to_line({"market": [["BUY_SEED", "WHEAT", 5]]})
    ta, tb = str(tmp_path / "a.tape"), str(tmp_path / "b.tape")
    write_tape(ta, seed, [action_to_line(a) for a in acts] + [extra])
    write_tape(tb, seed, [action_to_line(a) for a in opp] + [extra])
    [(s, b0, b1)] = run_batch([(None, ta, tb)])
    assert (s, b0, b1) == (seed, ob0, ob1)
    assert os.path.exists(ta)


def test_serve_refuses_to_step_past_terminal(serve):
    js = serve.reset(1)
    for _ in range(FINAL_STEP):
        js = serve.step2({"market": [["BUY_SEED", "WHEAT", 1]]}, {})
    assert js["step"] == FINAL_STEP and js["done"]
    again = serve.step2({"market": [["BUY_SEED", "WHEAT", 1]]}, {})
    assert again == js

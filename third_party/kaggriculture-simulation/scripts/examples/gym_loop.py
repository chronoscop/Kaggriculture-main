"""Minimal gym-style loop over `kagg serve`."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src-python"))

from kaggsim.env import Env  # noqa: E402
from kaggsim.policies import RandomPolicy, ScriptedFarmer  # noqa: E402

p0, p1 = ScriptedFarmer(1), RandomPolicy(2)
with Env() as env:
    obs0, obs1 = env.reset(seed=11)
    done, ret = False, [0.0, 0.0]
    while not done:
        (obs0, obs1), (r0, r1), done, info = env.step(p0(obs0), p1(obs1))
        ret[0] += r0
        ret[1] += r1
print("steps", info["step"], "banks", info["banks"], "returns", ret)

"""Tape agent, batch, env, economy, gauntlet and forced-world tools."""
import json
import os

import pytest

from kaggsim import economy, forced, gauntlet, official
from kaggsim.batch import run_batch
from kaggsim.constants import FINAL_STEP
from kaggsim.env import Env
from kaggsim.forced import patched, selftest
from kaggsim.policies import RandomPolicy, ScriptedFarmer
from kaggsim.serve import load_agent, run_match
from kaggsim.tape import (action_to_line, tape_to_agent, validate_tape,
                          write_tape)


def recorded_tapes(serve, tmp_path, seed=12):
    _, trace = run_match(ScriptedFarmer(8), RandomPolicy(9), seed, serve,
                         record=True)
    paths = []
    for s in (0, 1):
        p = str(tmp_path / f"seat{s}.tape")
        write_tape(p, seed, [action_to_line(t["actions"][s]) for t in trace])
        paths.append(p)
    return paths


def test_batch_equals_serve(serve, tmp_path):
    a, b = recorded_tapes(serve, tmp_path)
    [(seed, b0, b1)] = run_batch([(None, a, b)])
    _, trace = run_match(ScriptedFarmer(8), RandomPolicy(9), 12, serve,
                         record=True)
    last = trace[-1]["state"]["farms"]
    assert (b0, b1) == (last[0]["money"], last[1]["money"])


@pytest.mark.official
def test_tape_agents_reproduce_batch_on_official(official_mod, serve,
                                                 tmp_path):
    a, b = recorded_tapes(serve, tmp_path)
    [(seed, b0, b1)] = run_batch([(None, a, b)])
    ma = tape_to_agent(a, str(tmp_path / "a_main.py"))
    mb = tape_to_agent(b, str(tmp_path / "b_main.py"))
    off, _ = official.run_agents(load_agent(ma), load_agent(mb), seed)
    assert off == (b0, b1)


def test_validate_recorded_tape(serve, tmp_path):
    a, _ = recorded_tapes(serve, tmp_path)
    assert validate_tape(a) == {}


def test_env_rewards_sum_to_bank_change(kagg):
    pol0, pol1 = ScriptedFarmer(2), RandomPolicy(3)
    with Env(kagg) as env:
        o0, o1 = env.reset(seed=4)
        assert "private" in o0 and o0["player"] == 0 and o1["player"] == 1
        total, done, n = [0.0, 0.0], False, 0
        while not done:
            (o0, o1), (r0, r1), done, info = env.step(pol0(o0), pol1(o1))
            total[0] += r0
            total[1] += r1
            n += 1
    assert n == FINAL_STEP
    assert info["banks"] == (3000 + total[0], 3000 + total[1])


@pytest.mark.official
def test_economy_reconciles_every_step(official_mod):
    _, env = official.run_agents(ScriptedFarmer(1), ScriptedFarmer(6), 11)
    rows = economy.analyze_replay(json.loads(json.dumps(env.toJSON())))
    for r in rows:
        assert r["steps_reconciled"] == r["steps_total"] == FINAL_STEP
        spent = sum(r["spend"].values())
        assert r["bank"] == pytest.approx(3000 + r["revenue_total"] - spent)
    assert economy.summarise(rows)["players"] == 2


def test_gauntlet_resumable_and_compare(kagg, tmp_path):
    cand = tmp_path / "cand.py"
    base = tmp_path / "base.py"
    opp = tmp_path / "opp.py"
    body = ("from kaggsim.policies import {cls}\n"
            "_p = {cls}({seed})\n"
            "def agent(obs):\n    return _p(obs)\n")
    cand.write_text(body.format(cls="ScriptedFarmer", seed=1))
    base.write_text(body.format(cls="RandomPolicy", seed=1))
    opp.write_text(body.format(cls="RandomPolicy", seed=5))
    out = str(tmp_path / "gauntlets")
    s1 = gauntlet.run_gauntlet(str(cand), [("opp", str(opp))], [1, 2], "cand",
                       out_dir=out, workers=2)
    assert s1["games"] == 4 and s1["errors"] == 0
    assert s1["focus"]["overall"]["games"] == 4
    res_a = os.path.join(s1["out_dir"], "results.jsonl")
    lines = open(res_a, encoding="utf-8").read().splitlines()
    s1b = gauntlet.run_gauntlet(str(cand), [("opp", str(opp))], [1, 2], "cand",
                        out_dir=out)
    assert s1b["games"] == 4                     # resumed, nothing replayed
    assert open(res_a, encoding="utf-8").read().splitlines() == lines
    s2 = gauntlet.run_gauntlet(str(base), [("opp", str(opp))], [1, 2], "base",
                       out_dir=out)
    res_b = os.path.join(s2["out_dir"], "results.jsonl")
    res = gauntlet.compare(res_a, res_b, "cand", "base")
    assert res["n_pairs"] == 4
    assert gauntlet.main(["compare", res_a, res_b, "--agent-a", "cand",
                      "--agent-b", "base"]) == 0


@pytest.mark.official
def test_forced_world_selftest(official_mod):
    assert selftest()
    with patched(forced_shops=["YARN_STORE", "PET_CAFE"]):
        # fidelity tools refuse the modified game ...
        with pytest.raises(official.EngineMismatch):
            official.make(3)
        # ... the forced helpers run it
        env = forced.make(0, episodeSteps=160, actTimeout=60,
                          runTimeout=10 ** 5)
        env.run([lambda o: {}, lambda o: {}])
        shops = env.state[0].observation.town["unlocked_shops"]
    # seed 0 (a small per-day key) is patched too
    assert shops[:2] == ["YARN_STORE", "PET_CAFE"]
    # undone: the engine module is back to the real RNG and end of day
    assert official_mod.random.__name__ == "random"
    official.make(3)

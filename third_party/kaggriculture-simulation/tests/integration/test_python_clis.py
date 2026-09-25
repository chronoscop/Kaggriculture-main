"""Command-line entry points and remaining helpers of the Python package."""
import json
import os
import sys

import pytest

from kaggsim import binary, economy, fidelity, official, policies, worlds
from kaggsim import benchmark, forced, selfplay, tape, tournament
from kaggsim.serve import Serve, run_match
from kaggsim.tape import action_to_line, read_tape, write_tape


def test_find_kagg(monkeypatch, tmp_path, kagg):
    assert binary.find_kagg() == kagg
    assert binary.find_kagg(kagg) == kagg
    monkeypatch.setenv("KAGG_BIN", kagg)
    assert binary.find_kagg() == kagg
    with pytest.raises(FileNotFoundError):
        monkeypatch.delenv("KAGG_BIN")
        monkeypatch.setattr(binary, "_REPO", str(tmp_path))
        monkeypatch.delenv("CARGO_TARGET_DIR", raising=False)
        monkeypatch.setenv("PATH", str(tmp_path))
        binary.find_kagg()


def test_policies_are_deterministic():
    obs = {"player": 0, "step": 0, "farms": [
        {"money": 3000, "hands": [], "farmer": [4, 4],
         "tiles": [[None] * 10 for _ in range(10)]}] * 2,
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]}}
    for kind in policies.POLICIES:
        a, b = policies.make_policy(kind, 3), policies.make_policy(kind, 3)
        assert [a(obs) for _ in range(5)] == [b(obs) for _ in range(5)]


def test_serve_state_commands(serve):
    js = serve.reset(4)
    js = serve.step2({"market": [["HIRE"]]}, {})
    loaded = serve.load_state(js, seed=4)
    assert loaded["step"] == js["step"]
    out = serve.rollout(js, 5, [], [])
    assert out["step"] == js["step"] + 5
    g = serve.gengame(4, [], [])
    assert g["final"]["step"] == 719 and len(g["days"]) == 29


def test_tape_cli(tmp_path, serve):
    _, trace = run_match(policies.ScriptedFarmer(1), policies.RandomPolicy(2),
                         5, serve, record=True)
    steps = [[{"action": {}}, {"action": {}}]] + [
        [{"action": t["actions"][0]}, {"action": t["actions"][1]}]
        for t in trace]
    rep = tmp_path / "r.json"
    rep.write_text(json.dumps({"info": {"seed": 5}, "steps": steps}))
    t = str(tmp_path / "s0.tape")
    assert tape.main(["from-replay", str(rep), "0", t]) == 0
    seed, lines = read_tape(t)
    assert seed == 5 and lines[0] == action_to_line(trace[0]["actions"][0])
    assert tape.main(["validate", t]) == 0
    bad = str(tmp_path / "bad.tape")
    write_tape(bad, 1, ["FLY\t\t"])
    assert tape.main(["validate", bad]) == 1
    assert tape.main(["to-agent", t, str(tmp_path / "main.py"),
                      "--align-hands"]) == 0
    ns = {}
    exec((tmp_path / "main.py").read_text(), ns)
    obs = {"step": 0, "player": 0, "farms": [{"hands": [[4, 4]]}]}
    assert len(ns["agent"](obs)["hands"]) == 1
    with pytest.raises(ValueError):
        tape.replay_seed({"info": {}})


def test_worlds_cli(tmp_path, capsys):
    assert worlds.main(["catalog", "--seeds", "0-5", "--out",
                        str(tmp_path / "c.json")]) == 0
    cat = json.loads((tmp_path / "c.json").read_text())
    assert len(cat["worlds"]) == 6
    assert worlds.main(["catalog", "--seeds", "0-1", "--a", "chaos:1",
                        "--b", "random:2"]) == 0
    from kaggsim._util import parse_seeds
    assert parse_seeds("1-3,7,3,-2,-4..-3") == [1, 2, 3, 7, -2, -4, -3]
    with pytest.raises(ValueError):
        parse_seeds("5-1")
    with pytest.raises(ValueError):
        parse_seeds("x")


def test_screen_cli(tmp_path, serve):
    lines = [[], []]
    _, trace = run_match(policies.ScriptedFarmer(1), policies.RandomPolicy(2),
                         8, serve, record=True)
    for s in (0, 1):
        lines[s] = [action_to_line(t["actions"][s]) for t in trace]
    paths = []
    for s in (0, 1):
        p = str(tmp_path / f"t{s}.tape")
        write_tape(p, 8, lines[s])
        paths.append(p)
    out = str(tmp_path / "rt")
    assert worlds.main(["screen", paths[0], "--prefix", paths[0],
                        "--opp", paths[1], "--seeds", "0-3", "--out",
                        out]) == 0
    assert os.path.exists(out + ".json") and os.path.exists(out + ".md")


def test_template_clis(capsys):
    assert tournament.main(["template"]) == 0
    assert json.loads(capsys.readouterr().out)["schedule"] == "gauntlet"
    assert selfplay.main(["template"]) == 0
    assert json.loads(capsys.readouterr().out)["opponents"]["mode"] == \
        "mirror"


def test_tournament_cli_run(tmp_path, capsys):
    cfg = tournament.template("tournament")
    cfg.update(name="cli", worlds={"strategy": "range", "start": 0,
                                   "count": 1},
               output={"dir": str(tmp_path), "resume": False})
    p = tmp_path / "t.json"
    p.write_text(json.dumps(cfg))
    assert tournament.main(["run", str(p), "--set", "workers=1"]) == 0
    assert json.loads(capsys.readouterr().out)["games"] == 4


def test_official_override_errors(monkeypatch, tmp_path):
    monkeypatch.setenv(official.ENV_VAR, str(tmp_path))
    with pytest.raises(official.EngineMismatch):
        official._prepend_override()


# ------------------------------------------------ official-engine CLIs --

@pytest.mark.official
def test_fidelity_cli(official_mod, tmp_path):
    out = str(tmp_path / "c.json")
    assert fidelity.main(["certify", "--episodes", "1", "--json", out]) == 0
    assert json.loads(open(out).read())[0]["ok"]
    a = tmp_path / "a.py"
    a.write_text("from kaggsim.policies import ScriptedFarmer\n"
                 "_p = ScriptedFarmer(2)\n"
                 "def agent(obs):\n    return _p(obs)\n")
    assert fidelity.main(["diverge", str(a), str(a), "--seed", "4"]) == 0
    # Deterministic non-determinism: each load of this file acts
    # differently (a counter in the environment), so the official run and
    # the Rust run of seat 0 disagree at step 0.
    nd = tmp_path / "nd.py"
    nd.write_text("import os\n"
                  "N = int(os.environ.get('KAGGSIM_TEST_LOADS', '0'))\n"
                  "os.environ['KAGGSIM_TEST_LOADS'] = str(N + 1)\n"
                  "def agent(obs):\n"
                  "    return {'farmer': ['NORTH' if N == 0 else 'SOUTH']}\n")
    os.environ.pop("KAGGSIM_TEST_LOADS", None)
    try:
        assert fidelity.main(["diverge", str(nd), str(nd), "--seed",
                              "4"]) == 1
    finally:
        os.environ.pop("KAGGSIM_TEST_LOADS", None)


@pytest.mark.official
def test_economy_cli(official_mod, tmp_path, capsys):
    _, env = official.run_agents(policies.ScriptedFarmer(1),
                                 policies.RandomPolicy(2), 7)
    p = tmp_path / "r.json"
    p.write_text(json.dumps(env.toJSON()))
    assert economy.main([str(p), "--out", str(tmp_path / "rows.json")]) == 0
    rows = json.loads((tmp_path / "rows.json").read_text())
    assert rows[0]["steps_reconciled"] == 719


@pytest.mark.official
def test_forced_cli(official_mod):
    assert forced.main(["--selftest"]) == 0
    assert forced.main([]) == 0


@pytest.mark.official
def test_benchmark_small(official_mod, tmp_path):
    out = str(tmp_path / "b.json")
    assert benchmark.main(["--games", "2", "--workers", "2",
                           "--json", out]) == 0
    res = json.loads(open(out).read())
    assert res["all_banks_identical"]
    paths = {(r["path"], r["workers"]) for r in res["rows"]}
    assert {("official", 1), ("official", 2), ("rust-tournament", 2),
            ("rust-batch", 2), ("rust-builtin", 2)} <= paths
    assert "speed-up" in benchmark.markdown(res)


@pytest.mark.official
def test_gauntlet_verify_official(official_mod, tmp_path):
    from kaggsim import gauntlet
    a = tmp_path / "a.py"
    a.write_text("from kaggsim.policies import RandomPolicy\n"
                 "_p = RandomPolicy(3)\n"
                 "def agent(obs):\n    return _p(obs)\n")
    assert all(r["exact"] for r in gauntlet.verify_official(str(a), str(a),
                                                             [2]))
    assert gauntlet.main(["verify-official", str(a), str(a), "--seeds",
                          "3"]) == 0
    with pytest.raises(SystemExit):
        gauntlet.main(["run", str(a), "--opp", "no-equals-sign"])


def test_serve_error_paths(kagg):
    from kaggsim.serve import ServeError
    with Serve(kagg) as srv:
        with pytest.raises(ServeError):
            srv.cmd("STEP PASS\t\t")          # no episode yet
    assert sys.executable

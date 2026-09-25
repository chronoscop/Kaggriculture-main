"""Rust orchestration (kagg tournament / selfplay) driven from Python:
the agent host, processors/hooks, wrappers, and official fidelity of games
that involve Python agents."""
import io
import json
import os
import sys

import pytest

from kaggsim import official
from kaggsim.host import Host, loads_struct, serve as host_serve
from kaggsim.processor import JsonlWriter, Processor, basic_features, run
from kaggsim.processor import main as processor_main
from kaggsim.selfplay import run_selfplay
from kaggsim.serve import Serve, Struct, seat_view
from kaggsim.tournament import (compare, load_results, run_tournament,
                                template)

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def helper_path(monkeypatch):
    """Make tests/integration importable in kagg-spawned processes."""
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        [HERE] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    return HERE


def write_agent(path, body):
    path.write_text(body)
    return str(path)


# ------------------------------------------------------------------ host --

def test_loads_struct_is_attribute_accessible_and_fresh():
    a = loads_struct('{"x": {"y": [{"z": 1}]}, "items": 3}')
    assert isinstance(a, Struct) and a.x.y[0].z == 1
    assert "items" not in a                  # official Struct drops "items"
    b = loads_struct('{"x": {"y": [{"z": 1}]}}')
    a.x.y[0].z = 5
    assert b.x.y[0].z == 1


def test_host_protocol_in_process(tmp_path):
    agent = write_agent(tmp_path / "a.py",
                        "def agent(obs, configuration):\n"
                        "    print('chatty agent')\n"
                        "    assert configuration.episodeSteps == 720\n"
                        "    assert configuration.seed is None\n"
                        "    return {'farmer': ['WATER'], 'hands': [[]],\n"
                        "            'market': [[], ['HIRE']]}\n")
    with Serve() as srv:
        obs = json.dumps(seat_view(srv.reset(3), 0))
    msgs = [
        {"cmd": "ping"},
        {"cmd": "load", "slot": 0, "spec": {"type": "python", "path": agent},
         "echo": True},
        {"cmd": "load", "slot": 1, "spec": {"type": "pypolicy",
                                            "kind": "random", "seed": 1}},
        "ACT", "ACT2",
        {"cmd": "load", "slot": 0, "spec": {"type": "nope"}},
        {"cmd": "act", "slot": 5, "obs": {}},
        {"cmd": "bogus"},
        {"cmd": "quit"},
    ]
    lines = []
    for m in msgs:
        if m == "ACT":
            lines.append('{"cmd": "act", "slot": 0, "obs": ' + obs + "}")
        elif m == "ACT2":
            lines.append('{"cmd": "act2", "obs": [' + obs + ", " + obs + "]}")
        else:
            lines.append(json.dumps(m))
    lines.append("not json")
    out = io.StringIO()
    host_serve(io.StringIO("\n".join(lines) + "\n"), out)
    replies = [json.loads(x) for x in out.getvalue().splitlines()]
    assert replies[0]["ok"] and replies[1]["ok"] and replies[2]["ok"]
    assert replies[3]["line"] == "WATER\t\t;HIRE"
    assert replies[3]["action"]["market"] == [[], ["HIRE"]]
    assert replies[4]["lines"][0] == "WATER\t\t;HIRE"
    assert "error" in replies[5] and "error" in replies[6]
    assert "error" in replies[7]
    assert len(replies) == 8                 # quit stops before "not json"


def test_host_factory_and_errors(helper_path):
    h = Host()
    assert h.load({"slot": 0, "spec": {"type": "factory",
                                       "ref": "procfix:make_agent",
                                       "kwargs": {"bias": 1}},
                   "game_seed": 4})["ok"]
    with Serve() as srv:
        obs = loads_struct(json.dumps(seat_view(srv.reset(1), 0)))
    assert "line" in h.handle({"cmd": "act", "slot": 0, "obs": obs})
    assert "error" in h.handle({"cmd": "act", "slot": 1, "obs": obs})
    with pytest.raises(Exception):
        h.load({"slot": 0, "spec": {"type": "factory",
                                    "ref": "procfix:feats_missing"}})


# ------------------------------------------------------------ processor --

def test_processor_jsonl_writer(tmp_path, helper_path):
    recs = [
        {"record": "game", "game_id": "g"},
        {"record": "sample", "game_id": "g", "step": 3, "obs":
         {"town": {"unlocked_shops": ["A"]}}, "features": {"x": 1},
         "labels": {"outcome": 1}},
        {"record": "sample", "game_id": "g", "step": 4, "features": None,
         "labels": {"outcome": 0}},
    ]
    stream = [json.dumps(r) for r in recs] + ["", "garbage"]
    out = tmp_path / "o.jsonl"
    w = JsonlWriter(str(out), "procfix:feats", "procfix:labs",
                    keep="game_id,step,features,labels", records="sample,game")
    st = run(w, stream)
    assert st == {"games": 1, "samples": 2, "bad_lines": 1}
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert rows[0] == {"game_id": "g"}
    assert rows[1]["features"] == {"x": 1, "n_shops": 1, "custom_step": 3}
    assert rows[1]["labels"]["doubled"] == 2
    assert "obs" not in rows[1]
    assert rows[2]["features"]["custom_step"] == 4

    class Count(Processor):
        n = 0

        def on_sample(self, record):
            Count.n += 1

    run(Count(), stream)
    assert Count.n == 2
    assert processor_main(["--out", str(tmp_path / "p.jsonl")],
                          stdin=io.StringIO("\n".join(stream))) == 0


def test_basic_features_match_rust(tmp_path):
    cfg = {"name": "feat", "candidate": {"name": "a", "type": "builtin",
                                         "kind": "random", "seed": 1},
           "panel": [{"name": "b", "type": "builtin", "kind": "chaos"}],
           "seats": "seat0", "worlds": {"strategy": "list", "seeds": [9]},
           "workers": 1, "output": {"dir": str(tmp_path), "resume": False},
           "samples": {"include_obs": True, "stride": 97, "seats": [0]},
           "sinks": [{"type": "jsonl", "path": str(tmp_path / "s.jsonl"),
                      "records": ["sample"]}]}
    run_tournament(cfg)
    for line in (tmp_path / "s.jsonl").read_text().splitlines():
        rec = json.loads(line)
        py = basic_features(rec["obs"])
        assert py == rec["features"], rec["step"]


# ----------------------------------------------------- kagg front ends --

def test_tournament_wrapper_and_compare(tmp_path):
    cfg = template("tournament")
    cfg.update(name="wrap", workers=2,
               worlds={"strategy": "stratified", "pool": [0, 120],
                       "per_world": 1, "key_depth": 1},
               output={"dir": str(tmp_path), "resume": True})
    s = run_tournament(cfg)
    assert s["games"] > 0 and s["errors"] == 0
    assert len(s["standings"]) == 3
    res = os.path.join(s["out_dir"], "results.jsonl")
    rows = load_results(res)
    assert len(rows) == s["games"]
    assert all(r["target_world"] for r in rows)
    c = compare(res, res, "mine")
    assert c["n_pairs"] == s["games"] and c["p_value"] == 1.0
    again = run_tournament(cfg, overrides=["workers=1"])
    assert again["games"] == s["games"]
    with pytest.raises(RuntimeError):
        run_tournament({"name": "bad", "panel": []})


def test_selfplay_with_command_hook(tmp_path, helper_path):
    feats = tmp_path / "feats.jsonl"
    cfg = {"name": "sp", "agent": {"name": "me", "type": "pypolicy",
                                   "kind": "random", "seed": "per_game"},
           "opponents": {"mode": "pool", "p_mirror": 0.5, "rng_seed": 2,
                         "pool": [{"name": "c", "type": "builtin",
                                   "kind": "chaos", "seed": "per_game"}]},
           "worlds": {"strategy": "range", "start": 0, "count": 4},
           "seats": "both", "workers": 2,
           "samples": {"stride": 180, "include_obs": True,
                       "features": ["money"],
                       "labels": ["outcome", "return_to_go", "steps_left"],
                       "gamma": 0.99},
           "sinks": [{"type": "command", "records": ["sample"],
                      "argv": [sys.executable, "-m", "kaggsim.processor",
                               "--features", "procfix:feats",
                               "--labels", "procfix:labs",
                               "--out", str(feats)]},
                     {"type": "jsonl", "path": str(tmp_path / "g.jsonl"),
                      "records": ["game"]}],
           "output": {"dir": str(tmp_path), "resume": False}}
    s = run_selfplay(cfg)
    assert s["games"] == 8 and s["errors"] == 0
    rows = [json.loads(x) for x in feats.read_text().splitlines()]
    assert len(rows) == 8 * 4 * 2          # steps 0,180,360,540 x 2 seats
    r = rows[0]
    assert {"money_me", "n_shops", "custom_step"} <= set(r["features"])
    assert set(r["labels"]) == {"outcome", "return_to_go", "steps_left",
                                "doubled"}
    assert "obs" not in r
    games = (tmp_path / "g.jsonl").read_text().splitlines()
    assert len(games) == 8


# ------------------------------------------------------ official parity --

VANDAL = """
from kaggsim.policies import ScriptedFarmer
_p = ScriptedFarmer(5)
def agent(obs, configuration):
    a = _p(obs, configuration)
    obs.farms[0]["money"] = -1
    obs.market["prices"].clear()
    obs.private["shed"]["WHEAT"] = 999
    return a
"""

ONE_ARG = """
from kaggsim.policies import LastTurnSeller
_p = LastTurnSeller(6)
def agent(obs):
    return _p(obs)
"""


@pytest.mark.official
def test_python_hosted_games_match_official(official_mod, tmp_path):
    a = write_agent(tmp_path / "vandal.py", VANDAL)
    b = write_agent(tmp_path / "onearg.py", ONE_ARG)
    seeds = [3, 4]
    cfg = {"name": "parity",
           "candidate": {"name": "vandal", "type": "python", "path": a},
           "panel": [{"name": "onearg", "type": "python", "path": b}],
           "seats": "both", "worlds": {"strategy": "list", "seeds": seeds},
           "workers": 2, "output": {"dir": str(tmp_path), "resume": False}}
    s = run_tournament(cfg)
    rows = load_results(os.path.join(s["out_dir"], "results.jsonl"))
    assert len(rows) == 4
    from kaggsim.serve import load_agent
    for r in rows:
        paths = {"vandal": a, "onearg": b}
        off, env = official.run_agents(load_agent(paths[r["agents"][0]]),
                                       load_agent(paths[r["agents"][1]]),
                                       r["seed"])
        assert [st["status"] for st in env.steps[-1]] == ["DONE", "DONE"]
        assert list(off) == r["banks"], r


# ------------------------------------------------ host robustness / limits --

def test_host_contains_system_exit_and_keeps_items_kwargs(helper_path):
    import io as _io
    from kaggsim.host import serve as _serve
    lines = [
        json.dumps({"cmd": "load", "slot": 0,
                    "spec": {"type": "factory", "ref": "procfix:make_agent",
                             "kwargs": {"bias": 1}}, "game_seed": 1}),
        json.dumps({"cmd": "load", "slot": 1,
                    "spec": {"type": "factory", "ref": "procfix:exiting"}}),
        json.dumps({"cmd": "act", "slot": 1, "obs": {"step": 0}}),
        json.dumps({"cmd": "ping"}),
    ]
    out = _io.StringIO()
    _serve(_io.StringIO("\n".join(lines) + "\n"), out)
    replies = [json.loads(x) for x in out.getvalue().splitlines()]
    assert replies[0]["ok"] and replies[1]["ok"]
    assert "SystemExit" in replies[2]["error"] and replies[2]["slot"] == 1
    assert replies[3]["ok"]                  # the host is still alive


SLOW = """
import time
def agent(obs):
    if obs["step"] == 5:
        time.sleep(0.6)
    return {"farmer": ["PASS"]}
"""


def test_time_limits_forfeit_a_slow_agent(tmp_path):
    slow = write_agent(tmp_path / "slow.py", SLOW)
    cfg = {"name": "limits",
           "candidate": {"name": "slow", "type": "python", "path": slow},
           "panel": [{"name": "idle", "type": "builtin", "kind": "idle"}],
           "seats": "seat0", "worlds": {"strategy": "list", "seeds": [1]},
           "workers": 1, "time_limits": {"act_s": 0.1, "overage_s": 0.2},
           "output": {"dir": str(tmp_path), "resume": False}}
    s = run_tournament(cfg)
    row = load_results(os.path.join(s["out_dir"], "results.jsonl"))[0]
    assert row["error"]["seat"] == 0 and "timeout" in row["error"]["message"]
    assert row["forfeit"] and row["scores"] == [0, 1]
    assert row["banks"] is None
    # without limits the same agent finishes normally
    cfg["time_limits"] = None
    cfg["name"] = "nolimits"
    s = run_tournament(cfg)
    row = load_results(os.path.join(s["out_dir"], "results.jsonl"))[0]
    assert row["error"] is None and row["steps"] == 719


def test_processor_rejects_opts_without_custom_processor():
    with pytest.raises(SystemExit):
        processor_main(["--opt", "x=1"], stdin=io.StringIO(""))

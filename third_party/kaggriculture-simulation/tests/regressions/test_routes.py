"""Tape splicing and per-world continuation screening (synthetic tapes)."""
import json
import os

import pytest

from kaggsim.constants import FIRST_SHOP_STEP, N_ACTIONS, SECOND_SHOP_STEP
from kaggsim.policies import RandomPolicy, ScriptedFarmer
from kaggsim.serve import run_match
from kaggsim.tape import (TapeError, action_to_line, main as tape_main,
                          read_tape, splice_tapes, write_tape)
from kaggsim.worlds import (realized_world, screen_continuations,
                            screening_markdown)


def record(serve, p0, p1, seed):
    _, trace = run_match(p0, p1, seed, serve, record=True)
    return ([action_to_line(t["actions"][0]) for t in trace],
            [action_to_line(t["actions"][1]) for t in trace])


def fake_replay(path, seed, l0, l1):
    """A minimal replay in the official schema (actions only)."""
    from kaggsim.tape import line_to_action
    steps = [[{"action": {}}, {"action": {}}]]
    steps += [[{"action": line_to_action(a)}, {"action": line_to_action(b)}]
              for a, b in zip(l0, l1)]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"info": {"seed": seed}, "steps": steps}, fh)
    return path


def test_splice_composition_and_seed(tmp_path):
    pre, suf, out = (str(tmp_path / n) for n in ("p.tape", "s.tape",
                                                   "o.tape"))
    write_tape(pre, 11, ["NORTH\t\t"] * 200)
    write_tape(suf, 99, ["WATER\t\t"] * N_ACTIONS)
    rep = splice_tapes(pre, suf, 150, out)
    seed, lines = read_tape(out)
    assert seed == 11 and len(lines) == N_ACTIONS
    assert lines[:150] == ["NORTH\t\t"] * 150
    assert lines[150:] == ["WATER\t\t"] * (N_ACTIONS - 150)
    assert rep["world_kept"] == "both shops"
    assert splice_tapes(pre, suf, FIRST_SHOP_STEP, out)["world_kept"] == \
        "first shop"
    # short suffix: missing steps are empty actions
    splice_tapes(suf, pre, 300, out)
    assert read_tape(out)[1][350] == "PASS\t\t"


def test_splice_validates_both_inputs(tmp_path):
    good, bad, out = (str(tmp_path / n) for n in ("g.tape", "b.tape",
                                                   "o.tape"))
    write_tape(good, 1, ["WATER\t\t"] * N_ACTIONS)
    write_tape(bad, 1, ["FLY\t\t"] * N_ACTIONS)
    with pytest.raises(TapeError):
        splice_tapes(good, bad, 144, out)
    with pytest.raises(TapeError):
        splice_tapes(bad, good, 144, out)
    # bad steps outside the used range are ignored
    splice_tapes(good, bad, N_ACTIONS, out)
    with pytest.raises(TapeError):
        splice_tapes(good, good, 900, out)
    assert tape_main(["splice", good, bad, "144", out]) == 1
    assert tape_main(["splice", good, good, "144", out]) == 0


def test_splice_after_split_keeps_the_world(serve, tmp_path):
    a0, b0 = record(serve, ScriptedFarmer(1), RandomPolicy(2), 5)
    a1, _ = record(serve, ScriptedFarmer(7), RandomPolicy(8), 6)
    pa, pb, other, out = (str(tmp_path / n) for n in
                          ("a.tape", "b.tape", "x.tape", "o.tape"))
    write_tape(pa, 5, a0)
    write_tape(pb, 5, b0)
    write_tape(other, 6, a1)
    base = realized_world(serve, 5, pa, pb, k=2)
    splice_tapes(pa, other, SECOND_SHOP_STEP, out, validate=False)
    assert realized_world(serve, 5, out, pb, k=2) == base


def test_screen_continuations(serve, tmp_path):
    reps = []
    for i in range(4):
        l0, l1 = record(serve, ScriptedFarmer(10 + i), RandomPolicy(20 + i),
                        30 + i)
        reps.append(fake_replay(str(tmp_path / f"r{i}.json"), 30 + i, l0, l1))
    op0, _ = record(serve, ScriptedFarmer(3), RandomPolicy(4), 1)
    prefix = str(tmp_path / "prefix.tape")
    write_tape(prefix, 1, op0)
    opps = []
    for i in range(2):
        _, l1 = record(serve, RandomPolicy(40 + i), ScriptedFarmer(50 + i),
                       2 + i)
        p = str(tmp_path / f"opp{i}.tape")
        write_tape(p, 2 + i, l1)
        opps.append(p)
    single = str(tmp_path / "single.tape")
    write_tape(single, 30, read_tape(prefix)[1])
    out = str(tmp_path / "table")
    res = screen_continuations(reps + [single], prefix, opps, range(0, 40),
                            at_step=SECOND_SHOP_STEP, threads=2, top_k=3,
                            out=out, srv=serve)
    meta = res["meta"]
    assert meta["candidates"] == 9 and meta["world_key_depth"] == 2
    assert sum(meta["cells_by_world"].values()) == 2 * 40 * 2
    assert res["worlds"], "at least one world has candidates"
    evaluated = 0
    for world, e in res["worlds"].items():
        assert world.count("|") == 1
        scores = [r["score"] for r in e["ranked"]]
        assert scores == sorted(scores, reverse=True)
        assert len(e["ranked"]) <= 3
        for r in e["ranked"]:
            assert r["n"] == e["cells"]
            assert r["wins"] + r["draws"] + r["losses"] == r["n"]
            evaluated += 1
        if e["candidates"] >= 2 and e["cells"]:
            assert "leader_vs_runner_up" in e
    assert evaluated > 0
    assert os.path.exists(out + ".json") and os.path.exists(out + ".md")
    md = open(out + ".md", encoding="utf-8").read()
    assert "Continuation screening (open loop)" in md
    assert md == screening_markdown(res)


def test_screening_refuses_split_breaking_step(serve, tmp_path):
    t = str(tmp_path / "t.tape")
    write_tape(t, 1, [])
    with pytest.raises(ValueError):
        screen_continuations([t], t, [t], [1], at_step=50, srv=serve)

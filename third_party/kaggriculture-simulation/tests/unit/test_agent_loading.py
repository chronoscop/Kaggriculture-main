"""Agent loading: compiled once per process, executed fresh per game."""
import os
import time

import pytest

from kaggsim import serve

COUNTER = """
calls = 0
def agent(obs):
    global calls
    calls += 1
    return {"farmer": ["PASS"], "calls": calls, "tag": TAG}
TAG = "v1"
"""


def test_fresh_state_per_load_with_cached_code(tmp_path):
    p = tmp_path / "main.py"
    p.write_text(COUNTER)
    a = serve.load_agent(str(p))
    a({})
    a({})
    assert a({})["calls"] == 3
    b = serve.load_agent(str(p))
    assert b({})["calls"] == 1             # fresh module state
    assert a is not b
    key = os.path.abspath(str(p))
    code = serve._CODE_CACHE[key][1]
    serve.load_agent(str(p))
    assert serve._CODE_CACHE[key][1] is code   # compiled once


def test_cache_invalidates_on_change(tmp_path):
    p = tmp_path / "main.py"
    p.write_text(COUNTER)
    assert serve.load_agent(str(p))({})["tag"] == "v1"
    time.sleep(0.01)
    p.write_text(COUNTER.replace('"v1"', '"v2-longer"'))
    assert serve.load_agent(str(p))({})["tag"] == "v2-longer"


def test_missing_agent_function(tmp_path):
    p = tmp_path / "bad.py"
    p.write_text("x = 1\n")
    with pytest.raises(ValueError):
        serve.load_agent(str(p))


# `agent` is bound first, re-bound last; `helper` sits in between, so the
# LAST callable by insertion order is `helper`, not the final `agent`.
REBOUND = """
def agent(obs):
    return {"farmer": ["PASS"], "who": "first"}
def helper(obs, configuration=None):
    return {"farmer": ["PASS"], "who": "helper"}
def agent(obs):
    return {"farmer": ["PASS"], "who": "last-agent"}
"""


def test_entry_point_is_last_callable_like_the_official_runner(tmp_path):
    p = tmp_path / "main.py"
    p.write_text(REBOUND)
    assert serve.load_agent(str(p))({})["who"] == "helper"


def test_entry_point_matches_official_get_last_callable(tmp_path):
    agent_mod = pytest.importorskip("kaggle_environments.agent")
    p = tmp_path / "main.py"
    p.write_text(REBOUND)
    official = agent_mod.get_last_callable(REBOUND, path=str(p))
    assert serve.load_agent(str(p))({}) == official({})

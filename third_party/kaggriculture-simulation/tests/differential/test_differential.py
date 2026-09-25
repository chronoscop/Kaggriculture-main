"""Rust `kagg episode` vs the official interpreter, full-state digest after
EVERY step, over synthetic episodes (chaos / random / scripted / last-turn
policies in both seats).

    KAGGSIM_DIFF_EPISODES=500 pytest tests/differential     # nightly size
"""
import os

import pytest

from kaggsim import fidelity

N = int(os.environ.get("KAGGSIM_DIFF_EPISODES", "50"))
BASE_SEED = int(os.environ.get("KAGGSIM_DIFF_BASE_SEED", "1000"))
PLAN = fidelity.episode_plan(N, BASE_SEED)

pytestmark = pytest.mark.official


@pytest.mark.slow
@pytest.mark.parametrize("name,seed,p0,p1", PLAN, ids=[p[0] for p in PLAN])
def test_episode_bit_identical(official_mod, kagg, name, seed, p0, p1):
    res = fidelity.compare_episode(name, seed, p0, p1, kagg, log=None)
    assert res["ok"], res
    assert res["steps"] == 719

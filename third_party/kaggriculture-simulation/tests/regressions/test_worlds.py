"""World generator: realized worlds, split points, serve == official."""
import random

import pytest

from kaggsim import official
from kaggsim.constants import FIRST_SHOP_STEP, SECOND_SHOP_STEP
from kaggsim.policies import ChaosPolicy, ScriptedFarmer, chaos_action
from kaggsim.serve import run_match
from kaggsim.tape import action_to_line, line_to_action
from kaggsim.worlds import (build_catalog, build_map, idle_catalog,
                            realized_world)


def stream(policy_seed, seed, serve):
    """Record a closed-loop scripted stream as open-loop lines."""
    _, trace = run_match(ScriptedFarmer(policy_seed),
                         ScriptedFarmer(policy_seed + 50), seed, serve,
                         record=True)
    return ([action_to_line(t["actions"][0]) for t in trace],
            [action_to_line(t["actions"][1]) for t in trace])


def mutate_from(lines, start, rng):
    return lines[:start] + [action_to_line(chaos_action(rng))
                            for _ in lines[start:]]


def test_split_points(serve):
    changed_first = changed_second = 0
    for seed in range(8):
        a, b = stream(seed, seed, serve)
        base1 = realized_world(serve, seed, a, b, k=1)
        base2 = realized_world(serve, seed, a, b, k=2)
        rng = random.Random(seed)
        # Anything at/after the split cannot move that split's key.
        assert realized_world(serve, seed, mutate_from(a, FIRST_SHOP_STEP,
                              rng), b, k=1) == base1
        assert realized_world(serve, seed, a, mutate_from(
            b, SECOND_SHOP_STEP, rng), k=2) == base2
        # ...while changes before it can (and, over 8 seeds, do).
        changed_first += realized_world(
            serve, seed, mutate_from(a, 0, rng), b, k=1) != base1
        changed_second += realized_world(
            serve, seed, a, mutate_from(b, FIRST_SHOP_STEP, rng),
            k=2) != base2
    assert changed_first > 0 and changed_second > 0


def test_world_depends_on_actions_not_just_seed(serve):
    idle = idle_catalog(range(12))["worlds"]
    played = build_catalog(lambda: ChaosPolicy(1), lambda: ChaosPolicy(2),
                           range(12), srv=serve)["worlds"]
    assert any(idle[str(s)] != played[str(s)] for s in range(12))


def test_build_map_seats(serve):
    a, b = stream(3, 5, serve)
    m = build_map(a, [b], seeds=[5, 6], srv=serve)
    assert set(m) == {(0, 5, 0), (0, 5, 1), (0, 6, 0), (0, 6, 1)}
    assert m[(0, 5, 0)] == realized_world(serve, 5, a, b)


@pytest.mark.official
def test_realized_world_matches_official(official_mod, serve):
    a, b = stream(4, 21, serve)
    it0, it1 = iter(a), iter(b)
    _, env = official.run_agents(lambda o: line_to_action(next(it0)),
                                 lambda o: line_to_action(next(it1)), 21)
    shops = env.steps[-1][0]["observation"]["town"]["unlocked_shops"]
    assert realized_world(serve, 21, a, b) == "|".join(shops[:2])

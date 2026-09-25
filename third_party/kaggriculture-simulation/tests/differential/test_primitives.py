"""Pure engine primitives vs CPython / the official module."""
import json
import random
import struct
import subprocess

import pytest

from kaggsim.constants import SHOPS_SORTED

pytestmark = pytest.mark.official


def run(kagg, *args):
    return subprocess.run([kagg, *map(str, args)], capture_output=True,
                          text=True, check=True).stdout


def f64(bits):
    return struct.unpack("<d", struct.pack("<Q", int(bits)))[0]


@pytest.mark.parametrize("seed,day", [(0, 0), (1, 5), (2 ** 31 - 1, 29),
                                      (123456789, 17)])
def test_rng_probe_matches_cpython(kagg, seed, day):
    out = json.loads(run(kagg, "rng-probe", seed, day))
    r = random.Random((seed * 1_000_003) ^ day)
    assert [f64(b) for b in out["first8_random"]] == [r.random()
                                                      for _ in range(8)]
    r = random.Random((seed * 1_000_003) ^ day)
    assert out["getrandbits32_first4"] == [r.getrandbits(32)
                                           for _ in range(4)]
    r = random.Random((seed * 1_000_003) ^ day)
    assert out["choice_first4"] == [r.choice(SHOPS_SORTED) for _ in range(4)]


def test_weed_draws_match_cpython(kagg):
    out = json.loads(run(kagg, "weeds", 77, 3, 200))
    r = random.Random((77 * 1_000_003) ^ 3)
    assert [f64(b) for b in out] == [r.random() for _ in range(200)]


def test_rule_tables_match_official(kagg, official_mod):
    m = official_mod
    rules = json.loads(run(kagg, "rules"))
    assert rules["fib"] == [m._fib(n) for n in range(25)]
    assert rules["hire_cost"] == [m._hire_cost(n) for n in range(15)]
    for name, c in m.CROPS.items():
        r = rules["crops"][name]
        assert (r["seed"], r["first_yield_day"], r["max_yield_day"],
                r["interval"], r["max_yield"], r["ongoing"]) == (
            c["seed"], c["first_yield_day"], c["max_yield_day"],
            c["interval"], c["max_yield"], c["ongoing"])
    for name, a in m.ANIMALS.items():
        r = rules["animals"][name]
        assert (r["cost"], r["structure"], r["product"]) == (
            a["cost"], a["structure"], a["product"])
    assert rules["max_shop_instances"] == m.MAX_SHOP_INSTANCES
    assert rules["land_order"] == m.LAND_ORDER
    assert rules["land_prices"] == m.LAND_PRICES
    assert rules["quadrants"] == [m._quadrant_of(x, y, 10)
                                  for y in range(10) for x in range(10)]

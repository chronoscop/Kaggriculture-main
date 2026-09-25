"""Regression: 1.32.7 prices CARROT/TOMATO/EGG scarcity on the "hinge"
curve. Older engines (1.32.6 and earlier) used log/linear curves there; a
stale engine on sys.path silently answers a different question."""
import json
import subprocess

import pytest

from kaggsim.constants import PRODUCTS

pytestmark = pytest.mark.official


def sweep(kagg, item, lo, hi, step):
    out = subprocess.run([kagg, "price-sweep", item, lo, hi, step],
                         capture_output=True, text=True, check=True).stdout
    return {int(inv): q for inv, _, q in json.loads(out)}


@pytest.mark.parametrize("item", PRODUCTS)
def test_quoted_price_curve_matches_official(kagg, official_mod, item):
    got = sweep(kagg, item, "0", "20000", "7")
    for inv, q in got.items():
        assert q == official_mod.market_price(item, inv), (item, inv)


def test_carrot_is_on_the_hinge(kagg, official_mod):
    m = official_mod
    assert m.MARKET_PARAMS["CARROT"]["below_func"] == "hinge"
    # base 35, target 1.0, T 450: 1 capacity short -> 70, 2 short -> 385.
    assert m.market_price("CARROT", 10000 - 450) == 70
    assert m.market_price("CARROT", 10000 - 900) == 385
    got = sweep(kagg, "CARROT", "9100", "9550", "450")
    assert got == {9100: 385, 9550: 70}

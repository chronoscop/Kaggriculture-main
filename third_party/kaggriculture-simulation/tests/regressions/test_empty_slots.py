"""Regression: empty `[]` market orders and
empty hand slots must KEEP their position.

The official interpreter settles the market one ORDER INDEX at a time in
lockstep across both seats and applies hands[i] to hand i. Each test checks
the positional encoding against the official engine step by step AND a
negative control: the compacted encoding (empties dropped) must diverge,
proving the scenario actually exercises the rule.
"""
import pytest

from kaggsim import fidelity
from kaggsim.tape import action_to_line

pytestmark = pytest.mark.official


def compact(action):
    return {k: ([x for x in v if x] if k in ("hands", "market") else v)
            for k, v in action.items()}


def check(seed, acts0, acts1):
    l0 = [action_to_line(a) for a in acts0]
    l1 = [action_to_line(a) for a in acts1]
    off, off_banks = fidelity.run_official_lines(seed, l0, l1)
    rs, rs_banks = fidelity.run_rust(seed, l0, l1)
    assert rs == off, "positional encoding diverged from official"
    c0 = [action_to_line(compact(a)) for a in acts0]
    c1 = [action_to_line(compact(a)) for a in acts1]
    rs_c, _ = fidelity.run_rust(seed, c0, c1)
    assert rs_c != off, "negative control: compaction should diverge"
    return off_banks


def idle(n):
    return [{"farmer": ["PASS"], "hands": [], "market": []}] * n


def test_interior_empty_market_order_keeps_its_index(official_mod, kagg):
    # Seat 0 buys WHEAT at order index 1 (after an empty slot); seat 1 buys at
    # index 0. Positional: seat 1's buys move the price before seat 0 quotes.
    # Compacted: both quote the same pre-commit inventory unit by unit.
    a0 = [{"farmer": ["PASS"], "hands": [],
           "market": [[], ["BUY_PRODUCT", "WHEAT", 30],
                      [], ["SELL", "WHEAT", 5]]}] * 6 + idle(10)
    a1 = [{"farmer": ["PASS"], "hands": [],
           "market": [["BUY_PRODUCT", "WHEAT", 30], ["SELL", "WHEAT", 7],
                      ["BUY_PRODUCT", "FERTILIZER", 2]]}] * 6 + idle(10)
    check(31, a0, a1)


def test_leading_empty_order_changes_hire_and_market_order(official_mod,
                                                           kagg):
    a0 = [{"farmer": ["PASS"], "hands": [],
           "market": [[], [], ["BUY_PRODUCT", "FERTILIZER", 40]]}] * 3
    a1 = [{"farmer": ["PASS"], "hands": [],
           "market": [["BUY_PRODUCT", "FERTILIZER", 40]]}] * 3
    check(5, a0 + idle(5), a1 + idle(5))


def test_empty_hand_slot_keeps_hand_alignment(official_mod, kagg):
    hire3 = {"farmer": ["PASS"], "hands": [],
             "market": [["HIRE"], ["HIRE"], ["HIRE"]]}
    # Hand 0 idles via an empty slot, hands 1 and 2 walk away. Compacted,
    # hands 0 and 1 would walk instead.
    walk = {"farmer": ["PASS"], "hands": [[], ["WEST"], ["NORTH"]],
            "market": []}
    walk1 = {"farmer": ["EAST"], "hands": [["SOUTH"], [], ["EAST"]],
             "market": []}
    acts0 = [hire3] + [walk] * 4
    acts1 = [hire3] + [walk1] * 4
    check(9, acts0, acts1)

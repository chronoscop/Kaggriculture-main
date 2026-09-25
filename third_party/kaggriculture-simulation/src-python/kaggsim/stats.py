"""Paired win statistics -- the currency of the competition is WINS.

The leaderboard scores win / draw / loss, so $1 and $10,000 of margin are
paid identically. A mean-margin difference converts to wins only through the
density of the margin distribution near zero: two changes with the same mean
margin can be worth very different amounts of win rate. Decide with:

* :func:`score` / :func:`expected_score` -- 1 win, 0.5 draw, 0 loss.
* :func:`win_value` -- the EXACT win value of a per-game effect.
* :func:`paired_test` -- same seeds/seats: McNemar's exact test on the
  discordant pairs, plus a normal-approximation CI on the score difference.
* :func:`min_detectable` -- planning: how many paired games you need.
* :func:`margin_shift_to_wins` -- only for restating a margin claim in
  wins; it assumes a uniform per-game shift and is therefore an upper
  bound.
"""
from __future__ import annotations

import math

WIN, DRAW, LOSS = 1.0, 0.5, 0.0


def score(bank, opp_bank) -> float:
    a, b = float(bank), float(opp_bank)
    if a > b:
        return WIN
    if a < b:
        return LOSS
    return DRAW


def expected_score(scores) -> float:
    scores = list(scores)
    return sum(scores) / len(scores) if scores else float("nan")


def summarise(pairs):
    """(wins, draws, losses, expected_score) for (bank, opp_bank) pairs."""
    w = d = l = 0
    for bank, opp in pairs:
        s = score(bank, opp)
        if s == WIN:
            w += 1
        elif s == DRAW:
            d += 1
        else:
            l += 1
    n = w + d + l
    return w, d, l, ((w + 0.5 * d) / n if n else float("nan"))


def win_value(baseline_margins, deltas):
    """(gained, lost, net score change per game) of applying ``deltas[i]`` to
    the game whose margin was ``baseline_margins[i]``."""
    gained = lost = n = 0
    net = 0.0
    for base, d in zip(baseline_margins, deltas):
        before, after = score(base, 0.0), score(base + d, 0.0)
        net += after - before
        gained += after > before
        lost += after < before
        n += 1
    return gained, lost, (net / n if n else 0.0)


def margin_shift_to_wins(margins, dollars) -> float:
    """Win-rate value of a UNIFORM +``dollars``/game shift over ``margins``."""
    margins = [float(m) for m in margins]
    if not margins:
        return float("nan")
    n = len(margins)
    if dollars >= 0:
        return sum(1 for m in margins if -dollars < m <= 0) / n
    return -sum(1 for m in margins if 0 < m <= -dollars) / n


def mcnemar_exact(better_a: int, better_b: int) -> float:
    """Two-sided exact binomial p-value on the discordant pairs."""
    disc = better_a + better_b
    if disc == 0:
        return 1.0
    k = min(better_a, better_b)
    tail = sum(math.comb(disc, i) for i in range(k + 1)) / (2 ** disc)
    return min(1.0, 2 * tail)


def paired_test(a_scores, b_scores, alpha: float = 0.05) -> dict:
    """Paired comparison of two builds' per-game scores on the same games."""
    a, b = list(a_scores), list(b_scores)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    win_a = sum(1 for x, y in zip(a, b) if x > y)
    win_b = sum(1 for x, y in zip(a, b) if y > x)
    p = mcnemar_exact(win_a, win_b)
    diff = (sum(a) - sum(b)) / n if n else 0.0
    if n > 1:
        var = sum(((x - y) - diff) ** 2 for x, y in zip(a, b)) / (n - 1)
        half = 1.959964 * math.sqrt(var / n)
    else:
        half = float("nan")
    return {"n_pairs": n, "score_a": expected_score(a),
            "score_b": expected_score(b), "score_diff": diff,
            "better_a": win_a, "better_b": win_b,
            "discordant": win_a + win_b, "p_value": p,
            "ci95": (diff - half, diff + half),
            "significant": bool(p < alpha)}


def min_detectable(n_pairs: int, base_rate: float = 0.6,
                   discordance: float | None = None) -> float:
    """Smallest score difference ``n_pairs`` paired games detect (alpha 0.05,
    power 0.8). A PLANNING tool: a real paired test can be significant below
    it when the effect is one-directional. Pass the observed ``discordance``
    (fraction of pairs where the builds disagree) when known."""
    if n_pairs < 2:
        return float("nan")
    z_a, z_b = 1.959964, 0.8416212
    pdisc = (max(1.0 / n_pairs, float(discordance))
             if discordance is not None
             else max(0.05, 2 * base_rate * (1 - base_rate)))
    return (z_a + z_b) * math.sqrt(pdisc / n_pairs)

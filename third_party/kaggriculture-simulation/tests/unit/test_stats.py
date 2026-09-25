import math

from kaggsim import stats


def test_score_and_summary():
    assert stats.score(2, 1) == 1.0
    assert stats.score(1, 1) == 0.5
    assert stats.score(0, 1) == 0.0
    assert stats.summarise([(3, 1), (1, 1), (0, 5)]) == (1, 1, 1, 0.5)


def test_mcnemar_exact_matches_binomial():
    # 6 better vs 0 better: p = 2 * 0.5**6 = 0.03125
    assert stats.mcnemar_exact(6, 0) == 0.03125
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(3, 3) == 1.0


def test_paired_test():
    a = [1, 1, 1, 1, 1, 1, 0, 0.5]
    b = [0, 0, 0, 0, 0, 0, 0, 0.5]
    r = stats.paired_test(a, b)
    assert r["better_a"] == 6 and r["better_b"] == 0
    assert r["significant"] and math.isclose(r["p_value"], 0.03125)


def test_win_value_and_margin_shift():
    assert stats.win_value([-500, 2000, -5000], [1000, 1000, 1000])[:2] == (1, 0)
    assert stats.margin_shift_to_wins([-500, 2000, -5000, 0], 1000) == 0.5


def test_min_detectable_shrinks():
    assert stats.min_detectable(400) < stats.min_detectable(100)

//! Win statistics: the competition pays win / draw / loss, not margin.

/// 1 win, 0.5 draw, 0 loss.
pub fn score(bank: f64, opp: f64) -> f64 {
    if bank > opp {
        1.0
    } else if bank < opp {
        0.0
    } else {
        0.5
    }
}

pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        f64::NAN
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// Normal-approximation 95% CI of the mean.
pub fn ci95(xs: &[f64]) -> (f64, f64) {
    let n = xs.len();
    if n < 2 {
        return (f64::NAN, f64::NAN);
    }
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1) as f64;
    let h = 1.959964 * (var / n as f64).sqrt();
    (m - h, m + h)
}

fn ln_choose(n: u64, k: u64) -> f64 {
    ln_fact(n) - ln_fact(k) - ln_fact(n - k)
}

fn ln_fact(n: u64) -> f64 {
    (1..=n).map(|i| (i as f64).ln()).sum()
}

/// Two-sided exact McNemar p-value on the discordant pairs.
pub fn mcnemar_exact(better_a: u64, better_b: u64) -> f64 {
    let d = better_a + better_b;
    if d == 0 {
        return 1.0;
    }
    let k = better_a.min(better_b);
    let ln2 = (2f64).ln() * d as f64;
    let tail: f64 = (0..=k).map(|i| (ln_choose(d, i) - ln2).exp()).sum();
    (2.0 * tail).min(1.0)
}

/// Paired comparison of two per-game score lists.
#[derive(Debug, Clone)]
pub struct Paired {
    pub n: usize,
    pub better_a: u64,
    pub better_b: u64,
    pub p_value: f64,
    pub diff: f64,
    pub ci95: (f64, f64),
}

pub fn paired(a: &[f64], b: &[f64]) -> Paired {
    let n = a.len().min(b.len());
    let (mut wa, mut wb) = (0u64, 0u64);
    let mut d = Vec::with_capacity(n);
    for i in 0..n {
        if a[i] > b[i] {
            wa += 1;
        } else if b[i] > a[i] {
            wb += 1;
        }
        d.push(a[i] - b[i]);
    }
    Paired {
        n,
        better_a: wa,
        better_b: wb,
        p_value: mcnemar_exact(wa, wb),
        diff: if n > 0 { mean(&d) } else { 0.0 },
        ci95: ci95(&d),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scores() {
        assert_eq!(score(2.0, 1.0), 1.0);
        assert_eq!(score(1.0, 1.0), 0.5);
        assert_eq!(score(0.0, 1.0), 0.0);
    }

    #[test]
    fn mcnemar_matches_binomial() {
        assert!((mcnemar_exact(6, 0) - 0.03125).abs() < 1e-12);
        assert_eq!(mcnemar_exact(0, 0), 1.0);
        assert_eq!(mcnemar_exact(3, 3), 1.0);
        // large counts stay finite
        let p = mcnemar_exact(600, 400);
        assert!(p > 0.0 && p < 1e-8);
    }

    #[test]
    fn paired_and_ci() {
        let r = paired(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], &[0.0; 7]);
        assert_eq!((r.better_a, r.better_b), (6, 0));
        assert!((r.p_value - 0.03125).abs() < 1e-12);
        let (lo, hi) = ci95(&[0.0, 1.0, 0.0, 1.0]);
        assert!(lo < 0.5 && hi > 0.5);
        assert!(ci95(&[1.0]).0.is_nan());
        assert!(mean(&[]).is_nan());
    }
}

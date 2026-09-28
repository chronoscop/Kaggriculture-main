//! Bounded confirmation of a frozen candidate using terminal match score only.
//!
//! Each observation is the mean candidate-minus-incumbent score difference for
//! ONE independent seed, with both seats and the same frozen opponent roster.
//! Seats/opponents must not be treated as independent observations. The caller
//! supplies cumulative, unused confirmation seeds at predeclared looks and must
//! not train or choose another candidate between those looks.
//!
//! The one-sided paired sign-flip test assumes independent seeds and symmetric
//! paired differences under the null. Bonferroni covers the bounded looks for
//! one proposal under that assumption, NOT all proposals in a long training run.
//! This evidence filter does not prove that a deployed strategy cannot regress.
use kagg_engine::json::Json;

pub const PROPOSAL_ALPHA: f64 = 0.05;
const EXACT_LIMIT: usize = 16;
const MONTE_CARLO_DRAWS: usize = 32_767;

/// One-based `look`; cumulative seed counts may be N, 2N, 4N at three looks.
/// No absolute cash threshold or minimum score-improvement size is imposed.
pub fn assess(deltas: &[f64], look: usize, max_looks: usize) -> Result<Json, String> {
    if deltas.is_empty() {
        return Err("confirmation requires at least one paired independent seed".into());
    }
    if max_looks == 0 || look == 0 || look > max_looks {
        return Err("confirmation look must be in 1..=max_looks".into());
    }
    if deltas.iter().any(|d| !d.is_finite() || d.abs() > 1.) {
        return Err("confirmation requires finite terminal score differences in [-1, 1]".into());
    }
    // Sorting makes Monte Carlo invariant to worker result collection order.
    let mut values = deltas.to_vec();
    values.sort_by(f64::total_cmp);
    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;
    let standard_error = if n > 1 {
        (values.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n * (n - 1)) as f64).sqrt()
    } else {
        0.
    };
    let alpha = PROPOSAL_ALPHA / max_looks as f64;
    let exact = n <= EXACT_LIMIT;
    let tolerance = 64. * f64::EPSILON * values.iter().map(|d| d.abs()).sum::<f64>();

    // A signed sum >= the observed sum means the flipped subset sum <= 0.
    // For a location null mu it means subset_mean <= mu, so subset averages
    // also invert the SAME test into one-sided bounds under its assumption.
    // The empty subset always ties, even if sample variance is zero.
    let mut subset_means = Vec::with_capacity(if exact { 1 << n } else { MONTE_CARLO_DRAWS + 1 });
    let mut tail = 0usize;
    let mut empty = 0usize;
    let mut record = |sum: f64, count: usize| {
        if sum <= tolerance {
            tail += 1;
        }
        if count == 0 {
            empty += 1;
        } else {
            subset_means.push(sum / count as f64);
        }
    };
    if exact {
        for mask in 0usize..(1 << n) {
            let mut sum = 0.;
            let mut count = 0;
            for (i, d) in values.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    sum += d;
                    count += 1;
                }
            }
            record(sum, count);
        }
    } else {
        // Identity plus random sign vectors gives (tail + 1)/(draws + 1),
        // avoiding spuriously zero Monte Carlo p-values. No training RNG used.
        record(0., 0);
        let mut rng = 0xa076_1d64_78bd_642fu64;
        for _ in 0..MONTE_CARLO_DRAWS {
            let mut sum = 0.;
            let mut count = 0;
            let mut bits = 0;
            for (i, d) in values.iter().enumerate() {
                if i % 64 == 0 {
                    rng = rng.wrapping_add(0x9e37_79b9_7f4a_7c15);
                    bits = rng;
                    bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                    bits ^= bits >> 31;
                }
                if bits & (1u64 << (i % 64)) != 0 {
                    sum += d;
                    count += 1;
                }
            }
            record(sum, count);
        }
    }
    let samples = subset_means.len() + empty;
    let p = tail as f64 / samples as f64;
    subset_means.sort_by(f64::total_cmp);
    let rank = (alpha * samples as f64).floor() as usize;
    let (lower, upper) = if rank < empty {
        (-1., 1.)
    } else {
        let k = (rank - empty).min(subset_means.len() - 1);
        (subset_means[k], subset_means[subset_means.len() - 1 - k])
    };
    let qualifies = mean > 0. && p <= alpha;
    let decision = if qualifies {
        "promote"
    } else if mean <= 0. {
        // Early futility is allowed: it cannot increase false promotions,
        // but may miss a genuine improvement on a noisy first batch.
        "reject"
    } else if look < max_looks {
        "extend"
    } else {
        "inconclusive"
    };
    Ok(Json::Obj(vec![
        (
            "method".into(),
            Json::Str("paired_seed_sign_flip_v1".into()),
        ),
        ("seed_count".into(), Json::Num(n as f64)),
        (
            "nonzero_seeds".into(),
            Json::Num(values.iter().filter(|d| **d != 0.).count() as f64),
        ),
        ("mean_score_gain".into(), Json::Num(mean)),
        ("standard_error".into(), Json::Num(standard_error)),
        ("lower_bound".into(), Json::Num(lower)),
        ("upper_bound".into(), Json::Num(upper)),
        (
            "bounds_method".into(),
            Json::Str("inverted_one_sided_sign_flip_location_bounds".into()),
        ),
        ("p_value".into(), Json::Num(p)),
        ("alpha_per_look".into(), Json::Num(alpha)),
        ("proposal_alpha".into(), Json::Num(PROPOSAL_ALPHA)),
        ("look".into(), Json::Num(look as f64)),
        ("max_looks".into(), Json::Num(max_looks as f64)),
        ("exact".into(), Json::Bool(exact)),
        ("sign_vectors".into(), Json::Num(samples as f64)),
        ("decision".into(), Json::Str(decision.into())),
        ("qualifies".into(), Json::Bool(qualifies)),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_and_tiny_identical_samples_do_not_fake_certainty() {
        let tied = assess(&[0.; 8], 1, 3).unwrap();
        assert_eq!(tied.get("decision").str(), "reject");
        assert_eq!(tied.get("p_value").f64(), 1.);
        let small = assess(&[0.25; 2], 1, 3).unwrap();
        assert_eq!(small.get("standard_error").f64(), 0.);
        assert_eq!(small.get("decision").str(), "extend");
        assert_eq!(small.get("lower_bound").f64(), -1.);
    }

    #[test]
    fn consistent_small_score_gains_can_promote_without_effect_floor() {
        for delta in [0.125, 0.001, 1e-10] {
            let result = assess(&[delta; 8], 1, 3).unwrap();
            assert_eq!(result.get("decision").str(), "promote");
            assert_eq!(result.get("p_value").f64(), 1. / 256.);
        }
    }

    #[test]
    fn three_changed_seeds_are_insufficient_and_extension_is_bounded() {
        let deltas = [0.5, -0.25, 0., 0., 0., 0., 0.25, 0.];
        assert_eq!(
            assess(&deltas, 1, 3).unwrap().get("decision").str(),
            "extend"
        );
        assert_eq!(
            assess(&deltas, 3, 3).unwrap().get("decision").str(),
            "inconclusive"
        );
    }

    #[test]
    fn admission_is_not_a_vote_by_sign_of_seeds() {
        let mut deltas = vec![0.5; 8];
        deltas.extend([-0.001; 9]);
        assert_eq!(
            assess(&deltas, 1, 3).unwrap().get("decision").str(),
            "promote"
        );
        assert_eq!(
            assess(&[-0.25; 8], 1, 3).unwrap().get("decision").str(),
            "reject"
        );
    }

    #[test]
    fn monte_carlo_is_deterministic_and_collection_order_independent() {
        let mut deltas: Vec<f64> = (0..32)
            .map(|i| if i % 4 == 0 { -0.25 } else { 0.5 })
            .collect();
        let first = assess(&deltas, 2, 3).unwrap();
        deltas.reverse();
        assert_eq!(first, assess(&deltas, 2, 3).unwrap());
        assert_eq!(
            first.get("sign_vectors").f64(),
            (MONTE_CARLO_DRAWS + 1) as f64
        );
    }

    #[test]
    fn invalid_score_inputs_and_unbounded_looks_are_rejected() {
        assert!(assess(&[], 1, 3).is_err());
        assert!(assess(&[f64::NAN], 1, 3).is_err());
        assert!(assess(&[1.1], 1, 3).is_err());
        assert!(assess(&[0.5], 0, 3).is_err());
        assert!(assess(&[0.5], 4, 3).is_err());
        assert!(assess(&[0.5], 1, 0).is_err());
    }
}

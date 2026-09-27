//! Complete training-game outcomes only. Validation is never used for PFSP.
use kagg_engine::json::Json;

pub const HALF_LIFE: f64 = 50.;
pub const PRIOR_GAMES: f64 = 8.;
pub const UNIFORM_MIX: f64 = 0.2;
pub const MAX_SHARE: f64 = 0.4;

#[derive(Clone, Debug, Default)]
pub struct Results {
    pub games: f64,
    pub points: f64,
    pub cash: f64,
    pub margin: f64,
    pub total_games: u64,
    pub last_iteration: u64,
}
impl Results {
    fn decay(&self, iteration: u64) -> f64 {
        2_f64.powf(-(iteration.saturating_sub(self.last_iteration) as f64) / HALF_LIFE)
    }
    pub fn effective_games(&self, iteration: u64) -> f64 {
        self.games * self.decay(iteration)
    }
    pub fn score(&self, iteration: u64) -> f64 {
        let d = self.decay(iteration);
        (self.points * d + PRIOR_GAMES * 0.5) / (self.games * d + PRIOR_GAMES)
    }
    pub fn observe(&mut self, iteration: u64, cash: f64, rival_cash: f64) {
        let d = self.decay(iteration);
        let margin = cash - rival_cash;
        self.games = self.games * d + 1.;
        self.points = self.points * d
            + if margin > 0. {
                1.
            } else if margin == 0. {
                0.5
            } else {
                0.
            };
        self.cash = self.cash * d + cash;
        self.margin = self.margin * d + margin;
        self.total_games += 1;
        self.last_iteration = iteration;
    }
    pub fn json(&self) -> Json {
        Json::Obj(vec![
            ("effective_games".into(), Json::Num(self.games)),
            ("points".into(), Json::Num(self.points)),
            ("cash_sum".into(), Json::Num(self.cash)),
            ("margin_sum".into(), Json::Num(self.margin)),
            (
                "total_games".into(),
                Json::Str(self.total_games.to_string()),
            ),
            (
                "last_iteration".into(),
                Json::Str(self.last_iteration.to_string()),
            ),
        ])
    }
    pub fn restore(j: &Json) -> Result<Self, String> {
        if j.is_null() {
            return Ok(Self::default());
        }
        let number = |key| -> Result<f64, String> {
            if let Json::Num(n) = j.get(key) {
                if n.is_finite() {
                    return Ok(*n);
                }
            }
            Err("invalid opponent outcome statistics".into())
        };
        let s = Self {
            games: number("effective_games")?,
            points: number("points")?,
            cash: number("cash_sum")?,
            margin: number("margin_sum")?,
            total_games: j
                .get("total_games")
                .str()
                .parse()
                .map_err(|_| "invalid outcome count")?,
            last_iteration: j
                .get("last_iteration")
                .str()
                .parse()
                .map_err(|_| "invalid outcome iteration")?,
        };
        if s.games < 0.
            || s.points < 0.
            || s.points > s.games
            || s.games > s.total_games as f64 + 1e-6
        {
            return Err("inconsistent opponent outcomes".into());
        }
        Ok(s)
    }
    pub fn report(&self, iteration: u64) -> Json {
        let mut j = self.json();
        j.set_path(
            "effective_games_now",
            Json::Num(self.effective_games(iteration)),
        );
        j.set_path("smoothed_score", Json::Num(self.score(iteration)));
        j.set_path(
            "mean_cash",
            if self.games > 0. {
                Json::Num(self.cash / self.games)
            } else {
                Json::Null
            },
        );
        j.set_path(
            "mean_margin",
            if self.games > 0. {
                Json::Num(self.margin / self.games)
            } else {
                Json::Null
            },
        );
        j
    }
}
#[derive(Clone, Debug, Default)]
pub struct OpponentResults {
    pub sampled: Results,
    pub greedy: Results,
    pub focused: Results,
    pub broad: Results,
}
impl OpponentResults {
    pub fn json(&self) -> Json {
        Json::Obj(vec![
            ("sampled".into(), self.sampled.json()),
            ("greedy".into(), self.greedy.json()),
            ("focused".into(), self.focused.json()),
            ("broad".into(), self.broad.json()),
        ])
    }
    pub fn restore(j: &Json) -> Result<Self, String> {
        if !j.is_null() && !j.is_obj() {
            return Err("invalid opponent results".into());
        }
        Ok(Self {
            sampled: Results::restore(j.get("sampled"))?,
            greedy: Results::restore(j.get("greedy"))?,
            focused: Results::restore(j.get("focused"))?,
            broad: Results::restore(j.get("broad"))?,
        })
    }
}
/// Linear PFSP with prior shrinkage, uniform coverage and a cap.
/// With fewer than three members the cap must be >= 1/n.
pub fn probabilities(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() {
        return vec![];
    }
    let n = scores.len();
    let weights: Vec<_> = scores.iter().map(|p| (1. - p).clamp(0., 1.)).collect();
    let sum: f64 = weights.iter().sum();
    let mut p: Vec<_> = weights
        .iter()
        .map(|w| {
            UNIFORM_MIX / n as f64
                + (1. - UNIFORM_MIX) * if sum > 0. { w / sum } else { 1. / n as f64 }
        })
        .collect();
    let cap = MAX_SHARE.max(1. / n as f64);
    for _ in 0..n {
        let excess: f64 = p.iter().map(|x| (x - cap).max(0.)).sum();
        if excess < 1e-12 {
            break;
        }
        let free = p.iter().filter(|&&x| x < cap).count();
        for x in &mut p {
            if *x > cap {
                *x = cap;
            } else if *x < cap && free > 0 {
                *x += excess / free as f64;
            }
        }
    }
    p
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn outcomes_decay_shrink_and_round_trip() {
        let mut s = Results::default();
        s.observe(10, 100., 100.);
        assert_eq!(s.score(10), 0.5);
        s.observe(10, 200., 100.);
        assert!(s.score(10) > 0.5 && s.score(10) < 0.75);
        assert_eq!(s.effective_games(60), 1.);
        assert!((s.score(1010) - 0.5).abs() < 1e-6);
        assert_eq!(Results::restore(&s.json()).unwrap().json(), s.json());
        let mut bad = s.json();
        bad.set_path("points", Json::Num(3.));
        assert!(Results::restore(&bad).is_err());
    }
    #[test]
    fn pfsp_has_coverage_cap_and_handles_unknown_or_small_pools() {
        for scores in [
            vec![0.],
            vec![0., 1.],
            vec![0., 1., 1., 1., 1., 1.],
            vec![1.; 8],
            vec![0.5; 8],
        ] {
            let p = probabilities(&scores);
            assert!((p.iter().sum::<f64>() - 1.).abs() < 1e-10);
            assert!(p.iter().all(|&x| x >= UNIFORM_MIX / p.len() as f64 - 1e-10
                && x <= MAX_SHARE.max(1. / p.len() as f64) + 1e-10));
        }
        let p = probabilities(&[0.2, 0.6, 0.9, 0.5]);
        assert!(p[0] > p[1] && p[1] > p[2]);
    }
}

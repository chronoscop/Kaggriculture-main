//! Self-generated profitable episodes. Separate from the PPO on-policy stream.
use super::policy::{Rng, Sample};
use kagg_engine::json::Json;
pub const CAPACITY: usize = 16;
pub const ROWS_PER_EPISODE: usize = 32;
#[derive(Clone)]
pub struct Experience {
    pub seed: i64,
    pub seat: usize,
    pub profit: f64,
    pub style: u32,
    pub rows: Vec<Sample>,
}
impl Experience {
    pub fn from_episode(
        seed: i64,
        seat: usize,
        profit: f64,
        harvested: i64,
        rows: &[Sample],
    ) -> Option<Self> {
        if profit <= 0. || harvested <= 0 {
            return None;
        }
        let mut style = 0u32;
        for r in rows {
            let f = &r.features[r.action];
            if f[2] > 0. {
                for j in 9..17 {
                    if f[j] > 0. {
                        style |= 1 << (j - 9);
                    }
                }
            }
            if f[31] == 11. || f[31] == 14. {
                style |= 1 << 8;
            }
        }
        let eligible: Vec<_> = rows
            .iter()
            .filter(|r| r.mc_return > 0. && r.features[r.action][31] != 0.)
            .collect();
        if eligible.is_empty() {
            return None;
        }
        let stride = eligible.len().div_ceil(ROWS_PER_EPISODE);
        let rows = eligible
            .into_iter()
            .step_by(stride)
            .take(ROWS_PER_EPISODE)
            .cloned()
            .collect();
        Some(Self {
            seed,
            seat,
            profit,
            style,
            rows,
        })
    }
    pub fn json(&self) -> Json {
        Json::Obj(vec![
            ("seed".into(), Json::Str(self.seed.to_string())),
            ("seat".into(), Json::Num(self.seat as f64)),
            ("profit".into(), Json::Num(self.profit)),
            ("style".into(), Json::Num(self.style as f64)),
            (
                "rows".into(),
                Json::Arr(self.rows.iter().map(Sample::json).collect()),
            ),
        ])
    }
}
#[derive(Default)]
pub struct ExperienceBank {
    pub episodes: Vec<Experience>,
}
impl ExperienceBank {
    pub fn insert(&mut self, episode: Experience) {
        if self
            .episodes
            .iter()
            .any(|e| e.seed == episode.seed && e.seat == episode.seat)
        {
            return;
        }
        self.episodes.push(episode);
        self.episodes.sort_by(|a, b| {
            b.profit
                .total_cmp(&a.profit)
                .then(a.seed.cmp(&b.seed))
                .then(a.seat.cmp(&b.seat))
        });
        let mut styles = std::collections::BTreeMap::new();
        self.episodes.retain(|e| {
            let n = styles.entry(e.style).or_insert(0);
            *n += 1;
            *n <= 2
        });
        self.episodes.truncate(CAPACITY);
    }
    pub fn sample(&self, count: usize, rng: &mut Rng) -> Vec<Sample> {
        if self.episodes.is_empty() {
            return vec![];
        }
        (0..count)
            .map(|_| {
                let episode = &self.episodes[(rng.uniform() * self.episodes.len() as f64) as usize];
                episode.rows[(rng.uniform() * episode.rows.len() as f64) as usize].clone()
            })
            .collect()
    }
    pub fn json(&self) -> Json {
        Json::Arr(self.episodes.iter().map(Experience::json).collect())
    }
    pub fn restore(j: &Json) -> Result<Self, String> {
        if !j.is_arr() || j.arr().len() > CAPACITY {
            return Err("invalid experience bank".into());
        }
        let mut bank = Self::default();
        for e in j.arr() {
            let rows = e
                .get("rows")
                .arr()
                .iter()
                .map(Sample::parse)
                .collect::<Result<Vec<_>, _>>()?;
            let profit = e.get("profit").f64();
            if rows.is_empty()
                || rows.len() > ROWS_PER_EPISODE
                || !profit.is_finite()
                || profit <= 0.
                || rows
                    .iter()
                    .any(|r| r.mc_return <= 0. || r.features[r.action][31] == 0.)
            {
                return Err("invalid profitable experience".into());
            }
            bank.episodes.push(Experience {
                seed: e
                    .get("seed")
                    .str()
                    .parse()
                    .map_err(|_| "invalid experience seed")?,
                seat: e.get("seat").i64() as usize,
                profit,
                style: e.get("style").i64() as u32,
                rows,
            });
        }
        Ok(bank)
    }
}

//! Self-generated profitable episodes. Separate from the PPO on-policy stream.
use super::policy::{Rng, Sample};
use kagg_engine::json::Json;
pub const CAPACITY: usize = 128;
pub const PER_SOURCE: usize = 32;
pub const PER_STYLE: usize = 4;
pub const ROWS_PER_EPISODE: usize = 32;
#[derive(Clone)]
pub struct Experience {
    pub seed: i64,
    pub seat: usize,
    pub profit: f64,
    pub style: u32,
    pub rows: Vec<Sample>,
    pub opponent: String,
    pub learner_seat: Option<usize>,
    pub collected_iteration: u64,
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
            opponent: "legacy".into(),
            learner_seat: None,
            collected_iteration: 0,
        })
    }
    pub fn json(&self) -> Json {
        Json::Obj(vec![
            ("seed".into(), Json::Str(self.seed.to_string())),
            ("seat".into(), Json::Num(self.seat as f64)),
            ("profit".into(), Json::Num(self.profit)),
            ("style".into(), Json::Num(self.style as f64)),
            ("opponent".into(), Json::Str(self.opponent.clone())),
            (
                "learner_seat".into(),
                self.learner_seat
                    .map(|v| Json::Num(v as f64))
                    .unwrap_or(Json::Null),
            ),
            (
                "collected_iteration".into(),
                Json::Str(self.collected_iteration.to_string()),
            ),
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
        // A seed runs twice with swapped learner seats; these are distinct trajectories.
        if self.episodes.iter().any(|e| {
            e.seed == episode.seed
                && e.seat == episode.seat
                && e.learner_seat == episode.learner_seat
        }) {
            return;
        }
        self.episodes.push(episode);
        self.retain_balanced();
    }
    fn retain_balanced(&mut self) {
        use std::collections::BTreeMap;
        let mut groups: BTreeMap<String, BTreeMap<u32, Vec<Experience>>> = BTreeMap::new();
        for e in self.episodes.drain(..) {
            groups
                .entry(e.opponent.clone())
                .or_default()
                .entry(e.style)
                .or_default()
                .push(e);
        }
        // Retire provenance-free legacy rows once fresh coverage exists in every
        // live source. They cannot be assigned to an opponent retrospectively.
        if ["heuristic", "current", "historical"].iter().all(|source| {
            groups
                .get(*source)
                .is_some_and(|styles| styles.values().map(Vec::len).sum::<usize>() >= 8)
        }) {
            groups.remove("legacy");
        }
        for (_, styles) in groups {
            let mut retained = Vec::new();
            for (_, mut episodes) in styles {
                episodes.sort_by(|a, b| {
                    b.profit
                        .total_cmp(&a.profit)
                        .then(b.collected_iteration.cmp(&a.collected_iteration))
                        .then(b.seed.cmp(&a.seed))
                });
                let best: Vec<_> = episodes
                    .drain(..episodes.len().min(PER_STYLE / 2))
                    .collect();
                episodes.sort_by(|a, b| {
                    b.collected_iteration
                        .cmp(&a.collected_iteration)
                        .then(b.seed.cmp(&a.seed))
                        .then(b.profit.total_cmp(&a.profit))
                });
                let recent: Vec<_> = episodes.into_iter().take(PER_STYLE / 2).collect();
                // When source capacity binds, offer a recent episode before the
                // second profit leader, not only after both profit leaders.
                let mut chosen = Vec::new();
                for rank in 0..PER_STYLE / 2 {
                    if let Some(e) = best.get(rank) {
                        chosen.push(e.clone());
                    }
                    if let Some(e) = recent.get(rank) {
                        chosen.push(e.clone());
                    }
                }
                retained.push(chosen);
            }
            // Keep coverage across styles; when there are too many, favor recently seen styles.
            retained.sort_by_key(|v| {
                std::cmp::Reverse(
                    v.iter()
                        .map(|e| (e.collected_iteration, e.seed))
                        .max()
                        .unwrap(),
                )
            });
            let mut count = 0;
            for rank in 0..PER_STYLE {
                for style in &retained {
                    if count < PER_SOURCE {
                        if let Some(e) = style.get(rank) {
                            self.episodes.push(e.clone());
                            count += 1;
                        }
                    }
                }
            }
        }
        debug_assert!(self.episodes.len() <= CAPACITY);
    }
    pub fn sample(&self, count: usize, rng: &mut Rng) -> Vec<Sample> {
        use std::collections::BTreeMap;
        let mut groups: BTreeMap<&str, BTreeMap<u32, Vec<&Experience>>> = BTreeMap::new();
        for e in &self.episodes {
            groups
                .entry(&e.opponent)
                .or_default()
                .entry(e.style)
                .or_default()
                .push(e);
        }
        let sources: Vec<Vec<Vec<&Experience>>> = groups
            .into_values()
            .map(|s| s.into_values().collect())
            .collect();
        if sources.is_empty() {
            return vec![];
        }
        (0..count)
            .map(|_| {
                let styles = &sources[(rng.uniform() * sources.len() as f64) as usize];
                let episodes = &styles[(rng.uniform() * styles.len() as f64) as usize];
                let e = episodes[(rng.uniform() * episodes.len() as f64) as usize];
                e.rows[(rng.uniform() * e.rows.len() as f64) as usize].clone()
            })
            .collect()
    }
    pub fn summary(&self) -> Json {
        use std::collections::{BTreeMap, BTreeSet};
        let mut sources: BTreeMap<&str, Vec<&Experience>> = BTreeMap::new();
        for e in &self.episodes {
            sources.entry(&e.opponent).or_default().push(e);
        }
        Json::Obj(
            sources
                .into_iter()
                .map(|(source, es)| {
                    (
                        source.into(),
                        Json::Obj(vec![
                            ("episodes".into(), Json::Num(es.len() as f64)),
                            (
                                "rows".into(),
                                Json::Num(es.iter().map(|e| e.rows.len()).sum::<usize>() as f64),
                            ),
                            (
                                "styles".into(),
                                Json::Num(
                                    es.iter().map(|e| e.style).collect::<BTreeSet<_>>().len()
                                        as f64,
                                ),
                            ),
                            (
                                "oldest_iteration".into(),
                                Json::Num(
                                    es.iter().map(|e| e.collected_iteration).min().unwrap() as f64
                                ),
                            ),
                            (
                                "newest_iteration".into(),
                                Json::Num(
                                    es.iter().map(|e| e.collected_iteration).max().unwrap() as f64
                                ),
                            ),
                        ]),
                    )
                })
                .collect(),
        )
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
            let opponent = if e.get("opponent").is_null() {
                "legacy"
            } else {
                e.get("opponent").str()
            };
            if !["legacy", "heuristic", "current", "historical"].contains(&opponent) {
                return Err("invalid experience opponent source".into());
            }
            let learner_seat = if e.get("learner_seat").is_null() {
                None
            } else if matches!(e.get("learner_seat").f64(), 0. | 1.) {
                Some(e.get("learner_seat").i64() as usize)
            } else {
                return Err("invalid experience learner seat".into());
            };
            let collected_iteration = if e.get("collected_iteration").is_null() {
                0
            } else {
                e.get("collected_iteration")
                    .str()
                    .parse::<u64>()
                    .map_err(|_| "invalid experience iteration")?
            };
            bank.episodes.push(Experience {
                opponent: opponent.into(),
                learner_seat,
                collected_iteration,
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

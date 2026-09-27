//! Self-generated profitable episodes. Separate from the PPO on-policy stream.
use super::policy::{Rng, Sample};
use kagg_engine::json::Json;
pub const CAPACITY: usize = 192;
pub const PER_SOURCE: usize = 48;
pub const PER_BUCKET: usize = 16;
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
    pub retention_bucket: String,
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
            retention_bucket: "unassigned".into(),
        })
    }
    pub fn json(&self) -> Json {
        Json::Obj(vec![
            ("seed".into(), Json::Str(self.seed.to_string())),
            ("seat".into(), Json::Num(self.seat as f64)),
            ("profit".into(), Json::Num(self.profit)),
            (
                "retention_bucket".into(),
                Json::Str(self.retention_bucket.clone()),
            ),
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
        let mut sources: BTreeMap<String, Vec<Experience>> = BTreeMap::new();
        for e in self.episodes.drain(..) {
            sources.entry(e.opponent.clone()).or_default().push(e);
        }
        if ["heuristic", "current", "historical"]
            .iter()
            .all(|s| sources.get(*s).is_some_and(|v| v.len() >= 8))
        {
            sources.remove("legacy");
        }
        for (_, mut remaining) in sources {
            // Protected elite capacity: neither recency nor new style IDs can evict it.
            remaining.sort_by(|a, b| {
                b.profit
                    .total_cmp(&a.profit)
                    .then(a.collected_iteration.cmp(&b.collected_iteration))
                    .then(a.seed.cmp(&b.seed))
                    .then(a.seat.cmp(&b.seat))
                    .then(a.learner_seat.cmp(&b.learner_seat))
            });
            for mut e in remaining.drain(..remaining.len().min(PER_BUCKET)) {
                e.retention_bucket = "elite".into();
                self.episodes.push(e);
            }
            remaining.sort_by(|a, b| {
                b.collected_iteration
                    .cmp(&a.collected_iteration)
                    .then(b.seed.cmp(&a.seed))
                    .then(b.profit.total_cmp(&a.profit))
                    .then(a.seat.cmp(&b.seat))
                    .then(a.learner_seat.cmp(&b.learner_seat))
            });
            for mut e in remaining.drain(..remaining.len().min(PER_BUCKET)) {
                e.retention_bucket = "recent".into();
                self.episodes.push(e);
            }
            // Diversity has its own capacity and does not compete with elites.
            remaining.sort_by(|a, b| {
                b.profit
                    .total_cmp(&a.profit)
                    .then(a.seed.cmp(&b.seed))
                    .then(a.seat.cmp(&b.seat))
                    .then(a.learner_seat.cmp(&b.learner_seat))
            });
            let mut styles = Vec::<u32>::new();
            for _ in 0..PER_BUCKET.min(remaining.len()) {
                let i = (0..remaining.len())
                    .max_by_key(|&i| {
                        let d = styles
                            .iter()
                            .map(|s| (s ^ remaining[i].style).count_ones())
                            .min()
                            .unwrap_or(32);
                        (d, std::cmp::Reverse(i))
                    })
                    .unwrap();
                let mut e = remaining.remove(i);
                styles.push(e.style);
                e.retention_bucket = "diverse".into();
                self.episodes.push(e);
            }
        }
        debug_assert!(self.episodes.len() <= CAPACITY);
    }
    pub fn sample(&self, count: usize, rng: &mut Rng) -> Vec<Sample> {
        use std::collections::BTreeMap;
        let mut groups: BTreeMap<&str, BTreeMap<&str, Vec<&Experience>>> = BTreeMap::new();
        for e in &self.episodes {
            groups
                .entry(&e.opponent)
                .or_default()
                .entry(&e.retention_bucket)
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
                                "buckets".into(),
                                Json::Obj(
                                    ["elite", "recent", "diverse"]
                                        .iter()
                                        .map(|&bucket| {
                                            let rows: Vec<_> = es
                                                .iter()
                                                .filter(|e| e.retention_bucket == bucket)
                                                .collect();
                                            (
                                                bucket.into(),
                                                Json::Obj(vec![
                                                    (
                                                        "episodes".into(),
                                                        Json::Num(rows.len() as f64),
                                                    ),
                                                    (
                                                        "oldest_iteration".into(),
                                                        Json::Num(
                                                            rows.iter()
                                                                .map(|e| e.collected_iteration)
                                                                .min()
                                                                .unwrap_or(0)
                                                                as f64,
                                                        ),
                                                    ),
                                                    (
                                                        "newest_iteration".into(),
                                                        Json::Num(
                                                            rows.iter()
                                                                .map(|e| e.collected_iteration)
                                                                .max()
                                                                .unwrap_or(0)
                                                                as f64,
                                                        ),
                                                    ),
                                                ]),
                                            )
                                        })
                                        .collect(),
                                ),
                            ),
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
                retention_bucket: e.get("retention_bucket").str().into(),
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
        if j.arr().iter().any(|e| e.get("retention_bucket").is_null()) {
            bank.retain_balanced();
        } else {
            use std::collections::{BTreeMap, BTreeSet};
            let mut counts = BTreeMap::new();
            let mut seen = BTreeSet::new();
            for e in &bank.episodes {
                if !["elite", "recent", "diverse"].contains(&e.retention_bucket.as_str())
                    || !seen.insert((e.seed, e.seat, e.learner_seat))
                {
                    return Err("invalid retained experience".into());
                }
                let n = counts
                    .entry((&e.opponent, &e.retention_bucket))
                    .or_insert(0);
                *n += 1;
                if *n > PER_BUCKET {
                    return Err("experience bucket exceeds capacity".into());
                }
            }
        }
        Ok(bank)
    }
}

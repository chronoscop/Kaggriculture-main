//! World-selection strategies: which seeds (worlds) a run plays.
//!
//! ```json
//! {"strategy": "list", "seeds": [1, 2, 3]}
//! {"strategy": "range", "start": 0, "count": 100}
//! {"strategy": "stratified", "pool": [0, 5000], "per_world": 4,
//!  "key_depth": 2, "include": [], "exclude": []}
//! {"strategy": "weighted", "pool": [0, 5000], "count": 200, "key_depth": 1,
//!  "weights": {"SHOP_A": 3, "*": 1}, "rng_seed": 7}
//! ```
//!
//! (`SHOP_A` stands for any shop name; weights are illustrative.)
//!
//! The realized world depends on play (weeds and shops share the per-day
//! RNG), so `stratified` / `weighted` TARGET worlds through an idle-drive
//! label computed here in Rust (microseconds per seed). Every game records
//! the world it actually realized; summaries group by that.
//!
//! A plan never repeats a seed: `list` drops duplicates, and `weighted`
//! samples seeds without replacement (a world whose pool seeds run out
//! stops being drawn; the plan is shorter than `count` if every world with
//! a positive weight runs out).

use crate::util::{f64_or, i64_or, str_or, strings};
use kagg_engine::json::Json;
use kagg_engine::policies::Rng;
use kagg_engine::world::idle_world;
use std::collections::{BTreeMap, HashSet, VecDeque};

pub const STRATEGIES: [&str; 4] = ["list", "range", "stratified", "weighted"];

/// Largest `key_depth` accepted (shops drawn before the last step).
pub const MAX_KEY_DEPTH: usize = 8;

/// (seed, targeted world)
pub type SeedPlan = Vec<(i64, Option<String>)>;

fn pool(j: &Json) -> Result<Vec<i64>, String> {
    let p = j.get("pool");
    let (lo, hi) = if p.is_arr() {
        (p.idx(0).i64(), p.idx(1).i64())
    } else {
        (0, 2000)
    };
    if hi <= lo {
        return Err(format!("worlds.pool [{lo}, {hi}] is empty"));
    }
    Ok((lo..hi).collect())
}

/// Idle-drive world labels for the given seeds, computed in parallel.
pub fn idle_catalog(seeds: &[i64], k: usize, threads: usize) -> BTreeMap<i64, Option<String>> {
    let k = k.clamp(1, MAX_KEY_DEPTH);
    let threads = threads.max(1).min(seeds.len().max(1));
    let chunk = seeds.len().div_ceil(threads).max(1);
    let mut out = BTreeMap::new();
    std::thread::scope(|s| {
        let hs: Vec<_> = seeds
            .chunks(chunk)
            .map(|c| s.spawn(move || c.iter().map(|&x| (x, idle_world(x, k))).collect::<Vec<_>>()))
            .collect();
        for h in hs {
            out.extend(h.join().expect("catalog thread"));
        }
    });
    out
}

pub fn select(j: &Json, threads: usize) -> Result<SeedPlan, String> {
    let strat = str_or(j, "strategy", "range");
    match strat {
        "list" => {
            let mut seen = HashSet::new();
            Ok(j.get("seeds")
                .arr()
                .iter()
                .map(|v| v.i64())
                .filter(|s| seen.insert(*s))
                .map(|s| (s, None))
                .collect())
        }
        "range" => {
            let start = i64_or(j, "start", 0);
            let count = i64_or(j, "count", 20).max(0);
            Ok((start..start + count).map(|s| (s, None)).collect())
        }
        "stratified" | "weighted" => {
            let k = i64_or(j, "key_depth", 2).clamp(1, MAX_KEY_DEPTH as i64) as usize;
            let include = strings(j.get("include"));
            let exclude = strings(j.get("exclude"));
            let mut by_world: BTreeMap<String, VecDeque<i64>> = BTreeMap::new();
            for (s, w) in idle_catalog(&pool(j)?, k, threads) {
                let Some(w) = w else { continue };
                if exclude.contains(&w) || (!include.is_empty() && !include.contains(&w)) {
                    continue;
                }
                by_world.entry(w).or_default().push_back(s);
            }
            if strat == "stratified" {
                let per = i64_or(j, "per_world", 4).max(1) as usize;
                return Ok(by_world
                    .iter()
                    .flat_map(|(w, ss)| ss.iter().take(per).map(move |&s| (s, Some(w.clone()))))
                    .collect());
            }
            let weights = j.get("weights");
            let default = f64_or(weights, "*", 1.0);
            let worlds: Vec<String> = by_world.keys().cloned().collect();
            let mut ws: Vec<f64> = worlds
                .iter()
                .map(|w| f64_or(weights, w, default).max(0.0))
                .collect();
            let mut rng = Rng::new(i64_or(j, "rng_seed", 0) as u64);
            let count = i64_or(j, "count", 100).max(0);
            let mut out = Vec::new();
            for _ in 0..count {
                let total: f64 = ws.iter().sum();
                if total <= 0.0 {
                    break;
                }
                let mut r = rng.unit() * total;
                // the last world with a positive weight absorbs rounding
                let mut pick = ws.iter().rposition(|&w| w > 0.0).expect("positive weight");
                for (i, w) in ws.iter().enumerate() {
                    if *w > 0.0 && r < *w {
                        pick = i;
                        break;
                    }
                    r -= w;
                }
                let queue = by_world.get_mut(&worlds[pick]).expect("world");
                let seed = queue.pop_front().expect("non-empty");
                if queue.is_empty() {
                    ws[pick] = 0.0;
                }
                out.push((seed, Some(worlds[pick].clone())));
            }
            Ok(out)
        }
        other => Err(format!(
            "unknown world strategy {other:?}; choose from {STRATEGIES:?}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kagg_engine::json;

    fn sel(src: &str) -> SeedPlan {
        select(&json::parse(src).unwrap(), 2).unwrap()
    }

    #[test]
    fn list_and_range() {
        assert_eq!(
            sel(r#"{"strategy": "list", "seeds": [5, 9]}"#),
            vec![(5, None), (9, None)]
        );
        assert_eq!(
            sel(r#"{"strategy": "range", "start": 3, "count": 2}"#).len(),
            2
        );
        assert!(select(&json::parse(r#"{"strategy": "zzz"}"#).unwrap(), 1).is_err());
    }

    #[test]
    fn stratified_targets_worlds() {
        let plan =
            sel(r#"{"strategy": "stratified", "pool": [0, 300], "per_world": 2, "key_depth": 1}"#);
        let mut per: BTreeMap<String, usize> = BTreeMap::new();
        for (s, w) in &plan {
            let w = w.clone().unwrap();
            assert_eq!(idle_world(*s, 1).unwrap(), w);
            *per.entry(w).or_default() += 1;
        }
        assert!(per.len() >= 5, "several first shops appear in 300 seeds");
        assert!(per.values().all(|&n| n <= 2));
        let only = sel(
            r#"{"strategy": "stratified", "pool": [0, 300], "per_world": 3, "key_depth": 1, "include": ["BAKERY"]}"#,
        );
        assert!(only.iter().all(|(_, w)| w.as_deref() == Some("BAKERY")));
    }

    #[test]
    fn weighted_respects_weights_without_replacement() {
        let plan = sel(
            r#"{"strategy": "weighted", "pool": [0, 300], "count": 200, "key_depth": 1, "weights": {"BAKERY": 1, "*": 0}}"#,
        );
        assert!(!plan.is_empty());
        assert!(plan.iter().all(|(_, w)| w.as_deref() == Some("BAKERY")));
        // never more than the pool holds, never a repeated seed
        let seeds: HashSet<i64> = plan.iter().map(|x| x.0).collect();
        assert_eq!(seeds.len(), plan.len());
        assert!(plan.len() < 200);
        let none =
            sel(r#"{"strategy": "weighted", "pool": [0, 50], "count": 5, "weights": {"*": 0}}"#);
        assert!(none.is_empty());
        let mixed =
            sel(r#"{"strategy": "weighted", "pool": [0, 400], "count": 150, "key_depth": 1}"#);
        let uniq: HashSet<i64> = mixed.iter().map(|x| x.0).collect();
        assert_eq!(uniq.len(), mixed.len());
    }

    #[test]
    fn list_dedupes_and_bad_pool_errors() {
        assert_eq!(
            sel(r#"{"strategy": "list", "seeds": [3, 3, 4]}"#),
            vec![(3, None), (4, None)]
        );
        assert!(select(
            &json::parse(r#"{"strategy": "stratified", "pool": [5, 5]}"#).unwrap(),
            1
        )
        .is_err());
    }

    #[test]
    fn catalog_is_parallel_safe() {
        let seeds: Vec<i64> = (0..40).collect();
        let a = idle_catalog(&seeds, 2, 1);
        let b = idle_catalog(&seeds, 20, 4); // k is clamped
        let c = idle_catalog(&seeds, 2, 4);
        assert_eq!(b.len(), 40);
        assert_eq!(a, c);
    }
}

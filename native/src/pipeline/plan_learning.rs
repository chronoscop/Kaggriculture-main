//! State-dependent, concrete investment/renewal choices; no parameter search.
use super::{
    encoding,
    executor::*,
    plan_prototype::{Agent, Config, PlanningCache},
    planner::{Choice, Problem},
};
use crate::learning::policy::{Policy, Rng, Sample};
use kagg_engine::{
    engine::PlayerAction,
    rules,
    state::{Cell, ANIMAL_NAMES, CROP_NAMES, PRODUCTS},
};

pub const CONTRACT: &str = "plan-policy-320x32-v1";
#[derive(Clone)]
pub struct Candidate {
    pub next: Agent,
    pub orders: Vec<Vec<String>>,
    pub features: Vec<f32>,
}

/// Candidate construction runs on copies. Only the chosen copy commits resources.
pub fn candidates(agent: &Agent, o: &Observation) -> Vec<Candidate> {
    candidates_impl(agent, o, true)
}
fn candidates_impl(agent: &Agent, o: &Observation, compact: bool) -> Vec<Candidate> {
    let cache = PlanningCache::new(agent, o);
    let mut specs = vec![
        (None, 4, true, false),
        (None, 0, false, false),
        (None, 0, true, false),
    ];
    for kind in CROP_NAMES
        .iter()
        .map(|s| Production::Crop((*s).into()))
        .chain(ANIMAL_NAMES.iter().map(|s| Production::Animal((*s).into())))
    {
        for size in [1, 4] {
            specs.push((Some(kind.clone()), size, true, false));
        }
        specs.push((Some(kind), 4, true, true));
    }
    let reserved = &cache.reserved_sites;
    let mut seen = Vec::new();
    let mut result = Vec::new();
    for (kind, size, land, rotate) in specs {
        let mut released = Vec::new();
        if rotate {
            // Revise only confirmed, now-empty plots with no running route. Never
            // destroy a growing crop or cancel an in-flight material obligation.
            for (&pos, p) in &agent.executor.projects {
                if released.len() >= size {
                    break;
                }
                if p.confirmed
                    && kind.as_ref().is_some_and(|k| *k != p.production)
                    && matches!(
                        tile(&o.farm, pos),
                        Cell::Empty | Cell::Structure { animal: None, .. }
                    )
                    && !reserved.contains(&pos)
                {
                    released.push((pos, p.clone()));
                }
            }
            if released.is_empty() {
                continue;
            }
        }
        let mut next = if compact {
            agent.planning_copy()
        } else {
            agent.clone()
        };
        for (pos, _) in &released {
            next.executor.projects.remove(pos);
        }
        let orders = next.plan_batch_cached(o, kind.as_ref(), size, land, &cache);
        // A failed replacement retains the old commitment; no speculative exit.
        for (pos, old) in &released {
            if !next.executor.projects.contains_key(pos) {
                next.executor.projects.insert(*pos, old.clone());
            }
        }
        let changes: Vec<_> = next
            .executor
            .projects
            .iter()
            .filter(|(pos, p)| {
                agent
                    .executor
                    .projects
                    .get(pos)
                    .is_none_or(|old| old.production != p.production)
            })
            .map(|(pos, p)| (*pos, p.production.clone()))
            .collect();
        if seen
            .iter()
            .any(|(old_orders, old_changes)| old_orders == &orders && old_changes == &changes)
        {
            continue;
        }
        seen.push((orders.clone(), changes.clone()));
        let mut f = vec![0.; 32];
        f[0] = f32::from(changes.is_empty());
        f[1] = changes.len() as f32 / 4.;
        let mut cost = 0.;
        let mut hire = o.farm.hires_today as u32;
        for order in &orders {
            let qty = order
                .get(2)
                .and_then(|s| s.parse::<i64>().ok())
                .unwrap_or(1);
            let item = order.get(1).map(String::as_str).unwrap_or("");
            match order[0].as_str() {
                "BUY_SEED" => cost += qty as f64 * rules::crop(item).unwrap().seed_cost as f64,
                "BUY_ANIMAL" => cost += qty as f64 * rules::animal(item).unwrap().cost as f64,
                "BUY_PRODUCT" => {
                    cost += -super::trading::quote(item, o.market.inventory.get(item) - 10, -qty).0
                }
                "HIRE" => {
                    cost += rules::hire_cost(hire, 1) as f64;
                    hire += 1;
                    f[4] += 0.2;
                }
                "BUY_LAND" => {
                    cost += rules::next_land(o.farm.unlocked_quadrants.len() - 1)
                        .map(|(_, c)| c as f64)
                        .unwrap_or(0.);
                    f[5] = 1.;
                }
                _ => {}
            }
        }
        f[2] = cost as f32 / 10000.;
        f[3] = (o.farm.money - cost) as f32 / 10000.;
        f[6] = orders.len() as f32 / 6.;
        for (pos, kind) in &changes {
            let (product, delay, units, service, feed) = match kind {
                Production::Crop(c) => {
                    let r = rules::crop(c).unwrap();
                    (
                        c.as_str(),
                        r.first_yield_day,
                        if r.ongoing { 4. } else { 3. },
                        r.first_yield_day as f64,
                        0.,
                    )
                }
                Production::Animal(a) => {
                    let r = rules::animal(a).unwrap();
                    (
                        r.product,
                        r.first_yield_day,
                        (29 - o.day() - r.first_yield_day).max(0) as f64 / r.interval as f64,
                        (29 - o.day()) as f64 * 2.,
                        (29 - o.day()) as f64,
                    )
                }
                Production::Vacant => continue,
            };
            if let Some(i) = PRODUCTS.iter().position(|p| *p == product) {
                f[7 + i] += units as f32 / 40.;
            }
            f[16] += delay as f32 / 120.;
            f[17] += service as f32 / 200.;
            f[18] += feed as f32 / 120.;
            f[19] += (cache.price(product) * units) as f32 / 10000.;
            f[20] += distance(*pos, home(*pos)) as f32 / 40.;
            f[21] += f32::from(agent.executor.projects.contains_key(pos)) / 4.;
            f[22] += ((29 - o.day() - delay).max(0)) as f32 / 120.;
            f[23] += o.market.prices.get(product) as f32 / 2000.;
        }
        f[24] = agent
            .executor
            .projects
            .values()
            .filter(|p| !p.confirmed)
            .count() as f32
            / 25.;
        f[25] = (o.farm.hands.len() + 1) as f32 / 15.;
        f[26] = (100 - o.private.shed.sum()) as f32 / 100.;
        f[27] = (29 - o.day()) as f32 / 30.;
        // All plans share one category: greedy selection compares whole plans,
        // not category mass. Feature 30 below is the explicit planner prior.
        f[31] = 1.;
        result.push(Candidate {
            next,
            orders,
            features: f,
        });
    }
    if result.len() > 1 {
        // Baseline plan starts with 75% total sampling probability; all alternatives
        // together have 25%. Greedy initialization is exactly the planner, while
        // PPO can change every probability through learned residual logits.
        result[0].features[30] = (3. * (result.len() - 1) as f32).ln();
    }
    result
}

pub fn sample(agent: &Agent, o: &Observation, choices: &[Candidate]) -> Sample {
    let (mut context, _) = encoding::encode(
        o,
        &agent.executor,
        &Problem {
            actor: None,
            choices: vec![Choice::Continue],
        },
    );
    // Replace legacy two-project budget indicators with real plan obligations.
    context[4] = (o.farm.hands.len() + 1) as f32 / 15.;
    Sample {
        context: std::mem::take(&mut context),
        features: choices.iter().map(|c| c.features.clone()).collect(),
        step: o.step,
        cash: o.farm.money as f32,
        ..Sample::default()
    }
}

#[derive(Default)]
pub struct Timings {
    pub candidates: f64,
    pub inference: f64,
    pub execution: f64,
}
pub struct Learner {
    pub agent: Agent,
    pub rows: Vec<Sample>,
    pub decisions: usize,
    pub investment_choices: usize,
    pub rotation_choices: usize,
    pub timings: Timings,
}
impl Learner {
    pub fn new(config: Config) -> Self {
        Self {
            agent: Agent::new(config),
            rows: vec![],
            decisions: 0,
            investment_choices: 0,
            rotation_choices: 0,
            timings: Timings::default(),
        }
    }
    pub fn action(
        &mut self,
        o: &Observation,
        policy: &Policy,
        greedy: bool,
        teacher: bool,
        record: bool,
        rng: &mut Rng,
    ) -> Result<PlayerAction, String> {
        let start = std::time::Instant::now();
        self.agent.observe(o);
        self.timings.execution += start.elapsed().as_secs_f64();
        if !Agent::planning_due(o) {
            let start = std::time::Instant::now();
            let action = self.agent.execute(o, vec![]);
            self.timings.execution += start.elapsed().as_secs_f64();
            return Ok(action);
        }
        let start = std::time::Instant::now();
        let mut choices = candidates(&self.agent, o);
        if choices.is_empty() {
            return Err("no executable plan candidate".into());
        }
        let mut row = sample(&self.agent, o, &choices);
        self.timings.candidates += start.elapsed().as_secs_f64();
        let start = std::time::Instant::now();
        let action = if teacher || choices.len() == 1 {
            0
        } else {
            let d = policy
                .infer(std::slice::from_ref(&row), greedy, rng)?
                .remove(0);
            row.action = d.action;
            row.logp = d.logp;
            row.value = d.value;
            d.action
        };
        self.timings.inference += start.elapsed().as_secs_f64();
        if choices.len() > 1 {
            self.decisions += 1;
            if record {
                self.rows.push(row);
            }
        }
        let selected = choices.swap_remove(action);
        self.investment_choices += usize::from(selected.features[1] > 0.);
        self.rotation_choices += usize::from(selected.features[21] > 0.);
        self.agent.commit_plan(selected.next);
        let start = std::time::Instant::now();
        let action = self.agent.execute(o, selected.orders);
        self.timings.execution += start.elapsed().as_secs_f64();
        Ok(action)
    }
}

/// Undiscounted terminal win/loss, with GAE decay in actual environment steps.
/// Forecast profit and work counts never enter the reward.
pub fn finish(rows: &mut [Sample], final_step: i64, outcome: f32) {
    let mut next_step = final_step;
    let mut next_value = 0.;
    let mut trace = 0.;
    let mut terminal = outcome;
    for row in rows.iter_mut().rev() {
        row.elapsed = next_step - row.step;
        row.advantage =
            terminal + next_value - row.value + 0.997f32.powi(row.elapsed as i32) * trace;
        row.reward = row.value + row.advantage;
        row.mc_return = outcome;
        next_step = row.step;
        next_value = row.value;
        trace = row.advantage;
        terminal = 0.;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kagg_engine::state::State;
    #[test]
    fn proposals_do_not_commit_and_cover_species_and_batch_size() {
        let mut s = State::new(17);
        s.farms[0].money = 10000.;
        s.step = 7; // Worker hiring no longer consumes the opening order slots.
        let o = Observation::from_state(&s, 0);
        let mut cfg = Config::default();
        cfg.plots_per_worker = 4.;
        let mut a = Agent::new(cfg);
        a.observe(&o);
        let before = a.executor.projects.len();
        let cs = candidates(&a, &o);
        assert_eq!(a.executor.projects.len(), before);
        for crop in CROP_NAMES {
            assert!(
                cs.iter().any(|c| c
                    .next
                    .executor
                    .projects
                    .values()
                    .any(|p| p.production == Production::Crop(crop.into()))),
                "{crop}"
            );
        }
        assert!(cs.iter().any(|c| c.features[1] == 0.25));
        assert!(cs.iter().any(|c| c.features[1] == 1.));
        assert!(cs.iter().all(|c| c.features[3] >= 0.));
        let mut hidden = s.clone();
        hidden.private[1].shed.add("WHEAT", 9999);
        assert_eq!(
            sample(&a, &o, &cs).context,
            sample(&a, &Observation::from_state(&hidden, 0), &cs).context
        );
    }
    #[test]
    fn zero_residual_greedy_matches_executable_planner_and_starts_production() {
        crate::learning::tensor::threads(1);
        let model = Policy::plans(-1, 991, 1e-4).unwrap();
        let cfg = Config::default();
        let mut learner = Learner::new(cfg.clone());
        let mut baseline = Agent::new(cfg);
        let mut state = State::new(810005);
        let mut rng = Rng(77);
        for _ in 0..49 {
            let o = Observation::from_state(&state, 0);
            let learned = learner
                .action(&o, &model, true, false, false, &mut rng)
                .unwrap();
            let expected = baseline.action(&o);
            assert_eq!(action_json(&learned), action_json(&expected));
            kagg_engine::engine::step(&mut state, &[learned, Default::default()]);
        }
        assert!(learner.agent.opening_plots >= 12);
    }
    #[test]
    fn compact_plans_match_full_copies_with_running_routes() {
        let mut state = State::new(810005);
        let mut agent = Agent::new(Config::default());
        let mut compact_seconds = 0.;
        let mut full_seconds = 0.;
        let mut compared = 0;
        for _ in 0..180 {
            let o = Observation::from_state(&state, 0);
            agent.observe(&o);
            if Agent::planning_due(&o) && o.step % 4 == 0 {
                let t = std::time::Instant::now();
                let full = candidates_impl(&agent, &o, false);
                full_seconds += t.elapsed().as_secs_f64();
                let t = std::time::Instant::now();
                let compact = candidates_impl(&agent, &o, true);
                compact_seconds += t.elapsed().as_secs_f64();
                assert_eq!(full.len(), compact.len());
                for (a, b) in full.iter().zip(&compact) {
                    assert_eq!(a.orders, b.orders);
                    assert_eq!(a.features, b.features);
                    assert_eq!(
                        format!("{:?}", a.next.executor.projects),
                        format!("{:?}", b.next.executor.projects)
                    );
                    assert_eq!(a.next.batches, b.next.batches);
                    assert_eq!(
                        a.next.executor.stats.projects_requested,
                        b.next.executor.stats.projects_requested
                    );
                }
                // Commit must preserve active routes, inventory receipts and market history.
                let mut expected = full[0].next.clone();
                let mut actual = agent.clone();
                actual.commit_plan(compact[0].next.clone());
                assert_eq!(
                    action_json(&expected.execute(&o, full[0].orders.clone())),
                    action_json(&actual.execute(&o, compact[0].orders.clone()))
                );
                compared += 1;
            }
            let action = agent.action(&o);
            kagg_engine::engine::step(&mut state, &[action, Default::default()]);
        }
        eprintln!("candidate_copy_comparison decisions={compared} full_seconds={full_seconds:.4} compact_seconds={compact_seconds:.4}");
    }
    #[test]
    fn returns_use_elapsed_time_and_only_terminal_result() {
        let mut rows = vec![
            Sample {
                step: 0,
                ..Default::default()
            },
            Sample {
                step: 10,
                ..Default::default()
            },
        ];
        finish(&mut rows, 20, 1.);
        assert_eq!(rows[1].reward, 1.);
        assert!((rows[0].reward - 0.997f32.powi(10)).abs() < 1e-6);
        assert_eq!(rows[0].elapsed, 10);
    }
}

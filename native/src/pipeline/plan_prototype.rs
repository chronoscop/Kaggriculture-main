//! Small independent, deterministic planning prototype. No opponent private state,
//! baseline tapes, learned micro-trading or changes to legacy policy contracts.
use super::{executor::*, planner, trading};
use kagg_engine::{
    engine::PlayerAction,
    json::Json,
    market, rules,
    state::{Cell, OMap, ANIMAL_NAMES, CROP_NAMES, PRODUCTS},
};

#[derive(Clone, Debug)]
pub struct Config {
    pub plots_per_worker: f64,
    pub cash_reserve: f64,
    pub animal_share: f64,
    pub short_share: f64,
    pub expansion_fill: f64,
    pub forecast_days: f64,
    pub renewal_lead_days: i64,
    pub economic_routes: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            plots_per_worker: 3.5,
            cash_reserve: 180.,
            animal_share: 0.20,
            short_share: 0.45,
            expansion_fill: 0.75,
            forecast_days: 6.,
            renewal_lead_days: 0,
            economic_routes: true,
        }
    }
}
impl Config {
    pub fn json(&self) -> Json {
        Json::Obj(vec![
            (
                "schema".into(),
                Json::Str("production-plan-prototype-v1".into()),
            ),
            ("plots_per_worker".into(), Json::Num(self.plots_per_worker)),
            ("cash_reserve".into(), Json::Num(self.cash_reserve)),
            ("animal_share".into(), Json::Num(self.animal_share)),
            ("short_share".into(), Json::Num(self.short_share)),
            ("expansion_fill".into(), Json::Num(self.expansion_fill)),
            ("forecast_days".into(), Json::Num(self.forecast_days)),
            (
                "renewal_lead_days".into(),
                Json::Num(self.renewal_lead_days as f64),
            ),
            ("economic_routes".into(), Json::Bool(self.economic_routes)),
        ])
    }
    pub fn parse(j: &Json) -> Result<Self, String> {
        if j.get("schema").str() != "production-plan-prototype-v1" {
            return Err("invalid prototype configuration schema".into());
        }
        let x = Self {
            plots_per_worker: j.get("plots_per_worker").f64(),
            cash_reserve: j.get("cash_reserve").f64(),
            animal_share: j.get("animal_share").f64(),
            short_share: j.get("short_share").f64(),
            expansion_fill: j.get("expansion_fill").f64(),
            forecast_days: j.get("forecast_days").f64(),
            renewal_lead_days: j.get("renewal_lead_days").i64(),
            economic_routes: matches!(j.get("economic_routes"), Json::Bool(true)),
        };
        for (v, lo, hi) in [
            (x.plots_per_worker, 2., 6.),
            (x.cash_reserve, 0., 1500.),
            (x.animal_share, 0., 0.5),
            (x.short_share, 0., 1.),
            (x.expansion_fill, 0.4, 1.),
            (x.forecast_days, 0., 16.),
            (x.renewal_lead_days as f64, 0., 2.),
        ] {
            if !v.is_finite() || v < lo || v > hi {
                return Err("prototype parameter out of range".into());
            }
        }
        Ok(x)
    }
}

/// Read-only data shared by alternative plans for one observation. Confirmed
/// empty-plot rotations cannot change these route reservations or pending supply.
pub struct PlanningCache {
    pub reserved_sites: std::collections::BTreeSet<Pos>,
    pub reserved_seeds: OMap,
    pub reserved_pickups: OMap,
    prices: [f64; 9],
}
impl PlanningCache {
    pub fn new(agent: &Agent, o: &Observation) -> Self {
        let (reserved_sites, reserved_seeds, reserved_pickups) =
            agent.executor.reserved(usize::MAX);
        Self {
            reserved_sites,
            reserved_seeds,
            reserved_pickups,
            prices: std::array::from_fn(|i| {
                forecast_price(o, &agent.executor, PRODUCTS[i], agent.config.forecast_days)
            }),
        }
    }
    pub fn price(&self, item: &str) -> f64 {
        self.prices[PRODUCTS
            .iter()
            .position(|p| *p == item)
            .expect("production product")]
    }
}

#[derive(Clone)]
pub struct Agent {
    pub executor: Executor,
    pub config: Config,
    pub batches: usize,
    pub peak_plots: usize,
    pub peak_workers: usize,
    pub opening_plots: usize,
    pub rejected_cash: usize,
    pub rejected_capacity: usize,
    pub supply_sites: std::collections::BTreeSet<Pos>,
}
impl Agent {
    pub fn new(config: Config) -> Self {
        let mut executor = Executor::new();
        executor.market_mode = trading::MarketMode::Rule;
        Self {
            executor,
            config,
            batches: 0,
            peak_plots: 0,
            peak_workers: 0,
            opening_plots: 0,
            rejected_cash: 0,
            rejected_capacity: 0,
            supply_sites: Default::default(),
        }
    }
    pub fn planning_copy(&self) -> Self {
        let mut copy = Self::new(self.config.clone());
        copy.executor.projects = self.executor.projects.clone();
        copy.executor.stats.projects_requested = self.executor.stats.projects_requested;
        copy.supply_sites = self.supply_sites.clone();
        copy.batches = self.batches;
        copy.rejected_cash = self.rejected_cash;
        copy.rejected_capacity = self.rejected_capacity;
        copy
    }
    pub fn commit_plan(&mut self, plan: Self) {
        self.executor.projects = plan.executor.projects;
        self.executor.stats.projects_requested = plan.executor.stats.projects_requested;
        self.batches = plan.batches;
        self.rejected_cash = plan.rejected_cash;
        self.rejected_capacity = plan.rejected_capacity;
    }
    pub fn observe(&mut self, o: &Observation) {
        self.executor.observe(o);
        let plots = o
            .farm
            .tiles
            .iter()
            .flatten()
            .filter(|t| {
                matches!(
                    t,
                    Cell::Plant { .. }
                        | Cell::Structure {
                            animal: Some(_),
                            ..
                        }
                )
            })
            .count();
        self.peak_plots = self.peak_plots.max(plots);
        self.peak_workers = self.peak_workers.max(o.farm.hands.len() + 1);
        if o.step == 48 {
            self.opening_plots = plots;
        }
    }
    pub fn action(&mut self, o: &Observation) -> PlayerAction {
        self.observe(o);
        let orders = self.plan(o);
        self.execute(o, orders)
    }
    pub fn execute(&mut self, o: &Observation, orders: Vec<Vec<String>>) -> PlayerAction {
        for actor in 0..o.private.inventories.len() {
            if self
                .executor
                .routes
                .get(actor)
                .and_then(Option::as_ref)
                .is_some_and(|r| !r.steps.is_empty())
            {
                continue;
            }
            let p = if self.config.economic_routes {
                planner::economic_route_problem(o, &self.executor, actor)
            } else {
                planner::route_problem(o, &self.executor, actor)
            };
            let i = if self.config.economic_routes {
                p.choices
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        let score = |c: &planner::Choice| {
                            if let planner::Choice::Route { route, .. } = c {
                                let urgency: f64 = route
                                    .sites
                                    .iter()
                                    .filter_map(|s| self.executor.service_deadlines.get(s))
                                    .map(|due| if *due <= o.step + 24 { 20. } else { 0. })
                                    .sum();
                                planner::route_value(route) + urgency
                            } else {
                                0.
                            }
                        };
                        score(a).total_cmp(&score(b))
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            } else {
                p.heuristic()
            };
            p.select(i, &mut self.executor, o).expect("generated route");
        }
        self.executor.action(o, orders)
    }
    fn plan(&mut self, o: &Observation) -> Vec<Vec<String>> {
        self.plan_batch(o, None, 4, true)
    }
    /// Build actual orders and persistent projects. Forced species bypasses the
    /// heuristic mix preference, but not cash, material, capacity or season checks.
    pub fn plan_batch(
        &mut self,
        o: &Observation,
        forced: Option<&Production>,
        batch: usize,
        land: bool,
    ) -> Vec<Vec<String>> {
        if !Self::planning_due(o) {
            return vec![];
        }
        let cache = PlanningCache::new(self, o);
        self.plan_batch_cached(o, forced, batch, land, &cache)
    }
    /// Only projects, project request count and admission counters are mutated.
    pub fn plan_batch_cached(
        &mut self,
        o: &Observation,
        forced: Option<&Production>,
        batch: usize,
        land: bool,
        cache: &PlanningCache,
    ) -> Vec<Vec<String>> {
        let hour = o.step % 24;
        let day = o.day();
        if !Self::planning_due(o) {
            return vec![];
        }
        let mut orders = Vec::new();
        let mut spend = 0.;
        let mut stock = o.private.shed.clone();
        let mut seeds = o.private.seeds.clone();
        let mut pending = Vec::new();
        let reserved_seeds = &cache.reserved_seeds;
        let reserved_pickups = &cache.reserved_pickups;
        let animals = self
            .executor
            .projects
            .values()
            .filter(|p| matches!(p.production, Production::Animal(_)))
            .count();
        let feed_reserve = if day < 28 {
            animals as f64 * o.market.prices.get("WHEAT") as f64
        } else {
            0.
        };
        let available = (o.farm.money - self.config.cash_reserve - feed_reserve).max(0.);
        let mut order = |op: &str, item: &str, n: i64, cost: f64, mandatory: bool| -> bool {
            // Existing production is serviced before a not-yet-started successor.
            // Optional expansion respects future reservations; maintenance never borrows
            // fictitious income and its actual orders are debited before successor orders.
            let limit = if mandatory { o.farm.money } else { available };
            if orders.len() >= 6 || spend + cost > limit || n <= 0 {
                return false;
            }
            spend += cost;
            orders.push(if item.is_empty() {
                vec![op.into()]
            } else {
                vec![op.into(), item.into(), n.to_string()]
            });
            true
        };
        // Labor is paid per day. Admit only affordable workers, never pretend an
        // order has already produced a worker in today's route simulation.
        let weighted = self
            .executor
            .projects
            .values()
            .map(|p| match p.production {
                Production::Animal(_) => 1.5,
                Production::Crop(_) => 1.,
                _ => 0.,
            })
            .sum::<f64>();
        let desired = if day < 2 {
            6usize
        } else {
            (weighted / self.config.plots_per_worker).ceil().max(1.) as usize
        }
        .min(15);
        let mut workers = o.farm.hands.len() + 1;
        if hour <= 6 {
            while workers < desired {
                let cost = rules::hire_cost(
                    o.farm.hires_today as u32 + (workers - o.farm.hands.len() - 1) as u32,
                    1,
                ) as f64;
                if !order("HIRE", "", 1, cost, true) {
                    break;
                }
                workers += 1;
            }
        }
        // Feed needs are derived from active obligations; purchases wait for
        // receipts before the route builder can pick them up.
        let carried = o
            .private
            .inventories
            .iter()
            .map(|i| i.get("WHEAT"))
            .sum::<i64>();
        let feed_target = if day < 29 { animals as i64 * 2 } else { 0 };
        let q = (feed_target - stock.get("WHEAT") - carried)
            .max(0)
            .min(30)
            .min(100 - stock.sum());
        if q > 0 {
            let cost = -trading::quote("WHEAT", o.market.inventory.get("WHEAT") - 10, -q).0;
            if order("BUY_PRODUCT", "WHEAT", q, cost, true) {
                stock.add("WHEAT", q);
            }
        }
        // Just-in-time renewal stock, including seeds promised to active routes.
        for name in CROP_NAMES {
            if day + rules::crop(name).unwrap().first_yield_day >= 30 {
                continue;
            }
            let need = self
                .executor
                .projects
                .iter()
                .filter(|(p, pr)| {
                    pr.production == Production::Crop(name.into())
                        && !self.supply_sites.contains(p)
                        && match tile(&o.farm, **p) {
                            Cell::Empty | Cell::Weed | Cell::Structure { animal: None, .. } => true,
                            Cell::Plant {
                                crop, planted_day, ..
                            } => {
                                !rules::crop(crop).unwrap().ongoing
                                    && day - planted_day
                                        >= rules::crop(crop).unwrap().first_yield_day
                                            - self.config.renewal_lead_days
                            }
                            _ => false,
                        }
                })
                .count() as i64;
            let q = (need.max(reserved_seeds.get(name)) - seeds.get(name))
                .max(0)
                .min(12);
            if q > 0
                && order(
                    "BUY_SEED",
                    name,
                    q,
                    q as f64 * rules::crop(name).unwrap().seed_cost as f64,
                    true,
                )
            {
                seeds.add(name, q);
            }
        }
        let occupied = self
            .executor
            .projects
            .values()
            .filter(|p| p.production != Production::Vacant)
            .count();
        let unlocked = o.farm.unlocked_quadrants.len() * 25;
        if land
            && day >= 5
            && day < 22
            && occupied as f64 >= unlocked as f64 * self.config.expansion_fill
        {
            if let Some((_, cost)) = rules::next_land(o.farm.unlocked_quadrants.len() - 1) {
                order("BUY_LAND", "", 1, cost as f64, false);
            }
        }
        let mut sites = Vec::new();
        for y in 0..10 {
            for x in 0..10 {
                let p = (x, y);
                if self
                    .executor
                    .projects
                    .get(&p)
                    .is_some_and(|q| q.production != Production::Vacant)
                {
                    continue;
                }
                if matches!(
                    tile(&o.farm, p),
                    Cell::Empty | Cell::Weed | Cell::Structure { animal: None, .. }
                ) {
                    sites.push(p);
                }
            }
        }
        // Choose spatially compact additions, with neighbor service costs.
        sites.sort_by_key(|p| (distance(*p, home(*p)), *p));
        let limit = (workers as f64 * self.config.plots_per_worker).floor() as usize;
        let mut total = occupied;
        let mut animal_count = animals;
        let mut short = self
            .executor
            .projects
            .values()
            .filter(|p| matches!(&p.production,Production::Crop(c) if c=="WHEAT"||c=="CARROT"))
            .count();
        let mut additions = 0;
        for site in sites {
            if additions >= batch || day >= 27 {
                break;
            }
            if total >= limit {
                self.rejected_capacity += 1;
                break;
            }
            let want_animal = (animal_count as f64)
                < (total.max(8) as f64 * self.config.animal_share)
                && day < 17;
            let want_short = (short as f64)
                < ((total - animal_count).max(1) as f64 * self.config.short_share)
                || day >= 18;
            let mut kinds = Vec::new();
            for name in CROP_NAMES {
                let c = rules::crop(name).unwrap();
                if day + c.first_yield_day >= 29 {
                    continue;
                }
                if forced.is_some_and(|k| *k != Production::Crop(name.into())) {
                    continue;
                }
                if forced.is_none() && want_short && !matches!(name, "WHEAT" | "CARROT") {
                    continue;
                }
                let price = cache.price(name);
                let duration = if c.ongoing {
                    c.first_yield_day + 3 * c.interval
                } else {
                    c.max_yield_day
                };
                let units = if c.ongoing {
                    4.
                } else {
                    (1 + c.max_yield_day - (c.max_yield_day + 1) / 2 + 1).min(c.max_yield) as f64
                };
                let gain = (units * price - c.seed_cost as f64) / (duration as f64 + 1.);
                kinds.push((gain, Production::Crop(name.into()), c.seed_cost as f64));
            }
            if want_animal || (day < 17 && matches!(forced, Some(Production::Animal(_)))) {
                for name in ANIMAL_NAMES {
                    if forced.is_some_and(|k| *k != Production::Animal(name.into())) {
                        continue;
                    }
                    let a = rules::animal(name).unwrap();
                    let productive = (29 - day - a.first_yield_day).max(0);
                    let price = cache.price(a.product);
                    let revenue = productive as f64 * 2.0 / a.interval as f64 * price;
                    let feed = (29 - day) as f64 * o.market.prices.get("WHEAT") as f64;
                    let gain = (revenue - a.cost as f64 - feed) / (29 - day).max(1) as f64;
                    kinds.push((gain, Production::Animal(name.into()), a.cost as f64));
                }
            }
            kinds.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut selected = None;
            for (gain, kind, cost) in kinds {
                if gain <= 0. && forced.is_none() {
                    continue;
                }
                let name = kind.name();
                let ok = match &kind {
                    Production::Crop(_) => order("BUY_SEED", name, 1, cost, false),
                    Production::Animal(_) => {
                        stock.sum() + reserved_pickups.sum() < 95
                            && order("BUY_ANIMAL", name, 1, cost, false)
                    }
                    _ => false,
                };
                if ok {
                    selected = Some(kind);
                    break;
                }
            }
            if let Some(kind) = selected {
                if matches!(kind, Production::Animal(_)) {
                    animal_count += 1;
                    stock.add(kind.name(), 1);
                }
                if matches!(kind.name(), "WHEAT" | "CARROT") {
                    short += 1;
                }
                pending.push((site, kind));
                total += 1;
                additions += 1;
            } else {
                self.rejected_cash += 1;
                break;
            }
        }
        if !pending.is_empty() {
            self.batches += 1;
        }
        for (p, kind) in pending {
            self.executor.stats.projects_requested += 1;
            self.executor.projects.insert(
                p,
                Project {
                    production: kind,
                    requested: o.step,
                    confirmed: false,
                    failures: 0,
                },
            );
        }
        orders
    }
    pub fn planning_due(o: &Observation) -> bool {
        o.step % 24 <= 8 || o.step % 4 == 0
    }
    pub fn report(&self) -> Json {
        Json::Obj(vec![
            ("peak_plots".into(), Json::Num(self.peak_plots as f64)),
            ("opening_plots".into(), Json::Num(self.opening_plots as f64)),
            ("peak_workers".into(), Json::Num(self.peak_workers as f64)),
            ("plan_batches".into(), Json::Num(self.batches as f64)),
            ("work".into(), Json::Num(self.executor.stats.work as f64)),
            (
                "harvested_units".into(),
                Json::Num(self.executor.stats.harvested_units as f64),
            ),
            (
                "projects_started".into(),
                Json::Num(self.executor.stats.projects_started as f64),
            ),
            (
                "expired_projects".into(),
                Json::Num(self.executor.stats.expired_projects as f64),
            ),
            (
                "invalidated".into(),
                Json::Num(self.executor.stats.invalidated as f64),
            ),
            (
                "receipt_failures".into(),
                Json::Num(self.executor.stats.receipt_failures as f64),
            ),
            (
                "mixed_routes".into(),
                Json::Num(self.executor.stats.mixed_routes as f64),
            ),
        ])
    }
}

/// Scenario estimate from visible production and currently unlocked shops only.
pub fn forecast_price(o: &Observation, e: &Executor, item: &str, days: f64) -> f64 {
    let mut supply = 0.;
    for farm in [&o.farm, &o.rival] {
        for t in farm.tiles.iter().flatten() {
            match t {
                Cell::Plant { crop, .. } if crop == item => {
                    let c = rules::crop(crop).unwrap();
                    supply += if c.ongoing {
                        1. / c.interval as f64
                    } else {
                        3. / (c.max_yield_day + 1) as f64
                    };
                }
                Cell::Structure {
                    animal: Some(a), ..
                } if rules::animal(&a.animal).unwrap().product == item => {
                    supply += 1.5 / rules::animal(&a.animal).unwrap().interval as f64;
                }
                _ => {}
            }
        }
    }
    // Pending own commitments contribute supply, without assuming opponent sales times.
    for p in e.projects.values().filter(|p| !p.confirmed) {
        match &p.production {
            Production::Crop(c) if c == item => supply += 0.5,
            Production::Animal(a) if rules::animal(a).unwrap().product == item => supply += 0.5,
            _ => {}
        }
    }
    let demand = 1.
        + 6. * o
            .shops
            .iter()
            .filter(|s| kagg_engine::engine::shop_products(s).contains(&item))
            .count() as f64;
    let current = o.market.prices.get(item).max(1) as f64;
    (market::price(
        market::param(item).unwrap(),
        o.market.inventory.get(item) as f64 + days * (supply - demand),
    ) as f64)
        .clamp(current * 0.5, current * 1.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kagg_engine::{engine, state::State};
    #[test]
    fn identical_observations_ignore_hidden_opponent_inventory_and_rng() {
        let a = State::new(1);
        let mut b = a.clone();
        b.seed = 999999;
        b.private[1].shed.add("WHEAT", 9000);
        b.private[1].seeds.add("STRAWBERRY", 500);
        let mut aa = Agent::new(Config::default());
        let mut bb = Agent::new(Config::default());
        assert_eq!(
            action_json(&aa.action(&Observation::from_state(&a, 0))),
            action_json(&bb.action(&Observation::from_state(&b, 0)))
        );
    }
    #[test]
    fn opening_batch_delivers_production_without_inventing_stock() {
        let mut state = State::new(810005);
        let mut agent = Agent::new(Config::default());
        for _ in 0..49 {
            let o = Observation::from_state(&state, 0);
            let a = agent.action(&o);
            assert!(a.market.len() <= 10);
            engine::step(&mut state, &[a, Default::default()]);
            assert!(state.farms[0].money >= 0.);
            assert!(state.private[0].shed.sum() <= 100);
            assert!(state.private[0].seeds.0.iter().all(|(_, n)| *n >= 0));
        }
        assert!(
            agent.opening_plots >= 12,
            "approved investments must become actual production"
        );
        assert_eq!(agent.executor.stats.invalidated, 0);
        assert_eq!(agent.executor.stats.receipt_failures, 0);
        assert_eq!(agent.executor.stats.expired_projects, 0);
    }
    #[test]
    fn renewal_seeds_arrive_before_maturity_and_old_configs_keep_zero_lead() {
        let mut state = State::new(7);
        state.step = 24;
        for row in &mut state.farms[0].tiles {
            for t in row {
                *t = Cell::Locked;
            }
        }
        state.farms[0].tiles[4][3] = Cell::Plant {
            crop: "WHEAT".into(),
            planted_day: 0,
            watered_today: true,
            consecutive_unwatered: 0,
            yield_units: 1,
            max_lifespan_step: 120,
            fertilized_until_day: -1,
        };
        state.private[0].seeds = kagg_engine::state::OMap::seeded(&CROP_NAMES);
        let mut cfg = Config::default();
        cfg.renewal_lead_days = 1;
        let mut early = Agent::new(cfg.clone());
        let a = early.action(&Observation::from_state(&state, 0));
        assert!(a
            .market
            .iter()
            .any(|r| r[0] == "BUY_SEED" && r[1] == "WHEAT"));
        engine::step(&mut state, &[a, Default::default()]);
        assert!(state.private[0].seeds.get("WHEAT") > 0);
        assert!(state.step / 24 < 2);
        let mut old = cfg.json();
        if let Json::Obj(ref mut v) = old {
            v.retain(|(k, _)| k != "renewal_lead_days");
        }
        assert_eq!(Config::parse(&old).unwrap().renewal_lead_days, 0);
    }
    #[test]
    fn economic_route_selection_changes_with_product_prices() {
        for high in ["WHEAT", "CARROT"] {
            let mut s = State::new(9);
            s.step = 28 * 24 + 20;
            for (x, y, crop) in [(3, 4, "WHEAT"), (4, 3, "CARROT")] {
                s.farms[0].tiles[y][x] = Cell::Plant {
                    crop: crop.into(),
                    planted_day: 24,
                    watered_today: true,
                    consecutive_unwatered: 0,
                    yield_units: 2,
                    max_lifespan_step: 9999,
                    fertilized_until_day: -1,
                };
            }
            for name in ["WHEAT", "CARROT"] {
                s.market.prices.add(
                    name,
                    if name == high {
                        1000 - s.market.prices.get(name)
                    } else {
                        1 - s.market.prices.get(name)
                    },
                );
            }
            let o = Observation::from_state(&s, 0);
            let mut e = Executor::new();
            e.observe(&o);
            let p = planner::economic_route_problem(&o, &e, 0);
            let best = p
                .choices
                .iter()
                .filter_map(|c| {
                    if let planner::Choice::Route { route, .. } = c {
                        Some(route)
                    } else {
                        None
                    }
                })
                .max_by(|a, b| planner::route_value(a).total_cmp(&planner::route_value(b)))
                .unwrap();
            assert_eq!(
                best.harvested_products.get(high),
                2,
                "four remaining hours must prefer the valuable delivery"
            );
        }
    }
}

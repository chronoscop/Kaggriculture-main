//! Bounded route insertion over actual production, with explicit inventory simulation.
use super::executor::*;
use kagg_engine::{
    engine, rules,
    state::{Cell, Farm, Private, ANIMAL_NAMES, CROP_NAMES},
};
use std::collections::BTreeSet;
#[derive(Clone, Debug)]
pub enum Choice {
    Continue,
    Trade(super::trading::Trade),
    Route {
        actor: usize,
        route: Route,
    },
    Invest {
        site: Option<Pos>,
        production: Option<Production>,
        orders: Vec<Vec<String>>,
        cost: f64,
    },
}
impl Choice {
    /// Stable category ids are part of the policy/checkpoint contract.
    pub fn category(&self, e: &Executor) -> usize {
        match self {
            Self::Continue => 0,
            Self::Trade(t) => {
                if t.quantity == 0 {
                    16
                } else if t.quantity > 0 {
                    17
                } else {
                    18
                }
            }
            Self::Route { route, .. } => {
                if route.crop_jobs > 0 && route.animal_jobs > 0 {
                    if route.reused_fertilizer > 0 {
                        14
                    } else {
                        11
                    }
                } else if route.replants > 0 {
                    15
                } else if route.harvested > 0 {
                    if route.animal_jobs > 0 {
                        13
                    } else {
                        12
                    }
                } else if route.crop_jobs > 0 {
                    9
                } else if route.animal_jobs > 0 {
                    10
                } else {
                    8
                }
            }
            Self::Invest {
                site,
                production,
                orders,
                ..
            } => match production {
                Some(Production::Vacant) => 5,
                Some(kind) => {
                    if site.and_then(|p| e.projects.get(&p)).is_some_and(|p| {
                        p.production != Production::Vacant && &p.production != kind
                    }) {
                        4
                    } else if matches!(kind, Production::Crop(_)) {
                        2
                    } else {
                        3
                    }
                }
                None if orders.iter().any(|o| o[0] == "HIRE") => 6,
                None if orders.iter().any(|o| o[0] == "BUY_LAND") => 7,
                None => 1,
            },
        }
    }
}
#[derive(Clone)]
pub struct Problem {
    pub actor: Option<usize>,
    pub choices: Vec<Choice>,
}
impl Problem {
    pub fn select(
        &self,
        index: usize,
        e: &mut Executor,
        o: &Observation,
    ) -> Result<Vec<Vec<String>>, String> {
        let choice = self
            .choices
            .get(index)
            .ok_or("candidate index out of range")?;
        match choice {
            Choice::Continue => {}
            Choice::Trade(t) => {
                e.stats.trade_decisions += 1;
                if t.quantity == 0 {
                    e.stats.trade_holds += 1;
                } else {
                    if t.quantity > 0 {
                        e.stats.sell_orders += 1;
                    } else {
                        e.stats.buy_orders += 1;
                    }
                    return Ok(vec![vec![
                        if t.quantity > 0 {
                            "SELL".into()
                        } else {
                            "BUY_PRODUCT".into()
                        },
                        t.item.clone(),
                        t.quantity.abs().to_string(),
                    ]]);
                }
            }
            Choice::Route { actor, route } => e.assign(*actor, route.clone()),
            Choice::Invest {
                site,
                production,
                orders,
                cost,
            } => {
                if let (Some(p), Some(kind)) = (site, production) {
                    if *kind != Production::Vacant {
                        e.expansion_spent += cost;
                        e.new_projects_today += 1;
                        e.stats.projects_requested += 1;
                    }
                    e.projects.insert(
                        *p,
                        Project {
                            production: kind.clone(),
                            requested: o.step,
                            confirmed: false,
                            failures: 0,
                        },
                    );
                }
                e.stats.investments += 1;
                return Ok(orders.clone());
            }
        }
        Ok(vec![])
    }
    pub fn heuristic(&self) -> usize {
        self.choices
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| {
                heuristic(a)
                    .total_cmp(&heuristic(b))
                    .then_with(|| ib.cmp(ia))
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}
fn heuristic(c: &Choice) -> f64 {
    match c {
        Choice::Continue => 0.,
        Choice::Trade(t) => t.cash_delta,
        Choice::Route { route, .. } => {
            route.work as f64 * 3.
                + route.harvested as f64 * 2.
                + route.reused_fertilizer as f64 * 2.
                - route.walking as f64 * 0.2
        }
        Choice::Invest {
            production: Some(Production::Crop(c)),
            cost,
            ..
        } => {
            if c == "WHEAT" {
                8. - cost / 100.
            } else {
                5. - cost / 100.
            }
        }
        Choice::Invest {
            production: Some(Production::Animal(_)),
            cost,
            ..
        } => 3. - cost / 300.,
        Choice::Invest { orders, .. } => {
            if orders.iter().any(|o| o[0] == "HIRE") {
                7.
            } else {
                1.
            }
        }
    }
}
#[derive(Clone)]
struct Builder {
    farm: Farm,
    private: Private,
    actor: usize,
    t: i64,
    end: i64,
    route: Route,
    collected: i64,
    prices: kagg_engine::state::OMap,
}
impl Builder {
    fn new(o: &Observation, e: &Executor, actor: usize) -> Self {
        let (_, seeds, pickups) = e.reserved(actor);
        let mut private = o.private.clone();
        for (s, n) in seeds.0 {
            private.seeds.sub(&s, n.min(private.seeds.get(&s)));
        }
        for (s, n) in pickups.0 {
            private.shed.sub(&s, n.min(private.shed.get(&s)));
        }
        Self {
            farm: o.farm.clone(),
            private,
            actor,
            t: o.step,
            end: o.end(),
            route: Route::default(),
            collected: 0,
            prices: o.market.prices.clone(),
        }
    }
    fn command(&mut self, op: &str, item: &str, n: i64) -> bool {
        if self.t > self.end {
            return false;
        }
        // apply_unit_action never changes money, other workers, or remote tiles.
        // Movement changes only this actor's position; other successful commands
        // change the standing tile and/or this actor's inventory. PLANT also uses
        // a seed, but necessarily changes the standing tile on success.
        #[cfg(test)]
        let full_before = (self.farm.clone(), self.private.clone());
        let p = pos(&self.farm, self.actor);
        let movement = matches!(op, "NORTH" | "SOUTH" | "EAST" | "WEST");
        let initial = (!movement).then(|| tile(&self.farm, p).clone());
        let before_inventory = (!movement).then(|| self.private.inventories[self.actor].clone());
        let a = unit(op, item, n);
        engine::apply_unit_action(
            &mut self.farm,
            &mut self.private,
            self.actor,
            &a,
            self.t / 24,
        );
        let changed = if movement {
            pos(&self.farm, self.actor) != p
        } else {
            tile(&self.farm, p) != initial.as_ref().unwrap()
                || &self.private.inventories[self.actor] != before_inventory.as_ref().unwrap()
        };
        #[cfg(test)]
        assert_eq!(
            changed,
            self.farm != full_before.0 || self.private != full_before.1,
            "local action check: {op}"
        );
        if !changed {
            return false;
        }
        // Estimated value orders feasible routes; only actual game results train policies.
        let price = |name: &str| self.prices.get(name).max(1) as f64;
        let mut value = 0.;
        if op == "HARVEST" {
            for name in kagg_engine::state::PRODUCTS {
                let q = (self.private.inventories[self.actor].get(name)
                    - before_inventory.as_ref().unwrap().get(name))
                .max(0);
                if q > 0 {
                    self.route.harvested_products.add(name, q);
                }
                value += q as f64 * price(name);
            }
        }
        if let Some(Cell::Plant {
            crop,
            planted_day,
            consecutive_unwatered,
            ..
        }) = initial.as_ref()
        {
            let spec = rules::crop(crop).unwrap();
            if op == "WATER" {
                let survival = if *consecutive_unwatered > 0 { 1.5 } else { 0.7 };
                value += price(crop) * survival;
            }
            if op == "FERTILIZE" {
                let age = self.t / 24 - planted_day;
                if age >= spec.max_yield_day / 2 {
                    value += price(crop) * 0.65;
                }
            }
        }
        if let Some(Cell::Structure {
            animal: Some(a), ..
        }) = initial.as_ref()
        {
            let spec = rules::animal(&a.animal).unwrap();
            let remaining = 29 - self.t / 24;
            if op == "FEED" && remaining > 0 {
                value += price(spec.product) * (if a.consecutive_unfed > 0 { 2.0 } else { 1.0 })
                    / spec.interval as f64
                    + spec.cost as f64 / 25.;
            }
            if op == "CARE" && remaining > 0 {
                value += price(spec.product) * 0.65 / spec.interval as f64;
            }
            if op == "COLLECT_FERTILIZER" {
                value += price("FERTILIZER").min(80.) * 0.3;
            }
        }
        if op == "PLANT" {
            let spec = rules::crop(item).unwrap();
            value += (price(item) * spec.max_yield as f64 - spec.seed_cost as f64).max(0.)
                / (spec.first_yield_day as f64 + 2.)
                + 15.;
        }
        if op == "PLACE" && rules::animal(item).is_some() {
            value += 120.;
        }
        if op.starts_with("BUILD_") {
            value += 30.;
        }
        if op == "DIG" {
            value += 8.;
        }
        self.route.economic_value += value;
        self.route.walking += usize::from(movement);
        self.route.work += usize::from(!movement);
        if op == "PLANT" {
            self.route.seeds.add(item, 1);
        }
        if op == "PICKUP" {
            self.route.pickups.add(item, n);
        }
        if op == "COLLECT_FERTILIZER" {
            self.collected += 1;
        }
        if op == "FERTILIZE" && self.collected > 0 {
            self.collected -= 1;
            self.route.reused_fertilizer += 1;
        }
        if op == "HARVEST" {
            self.route.harvested += (self.private.inventories[self.actor].sum()
                - before_inventory.as_ref().unwrap().sum())
            .max(0);
        }
        self.route.steps.push_back(Scheduled {
            at: self.t,
            position: p,
            action: a,
        });
        engine::decay_plants(&mut self.farm, self.t);
        self.t += 1;
        true
    }
    fn walk(&mut self, target: Pos) -> bool {
        while pos(&self.farm, self.actor) != target {
            let op = movement(pos(&self.farm, self.actor), target);
            if !self.command(op, "", 0) {
                return false;
            }
        }
        true
    }
    fn pickup(&mut self, item: &str, quantity: i64) -> bool {
        let need = (quantity - self.private.inventories[self.actor].get(item)).max(0);
        if need == 0 {
            return true;
        }
        if self.private.shed.get(item) < need {
            return false;
        }
        if !self.walk(home(pos(&self.farm, self.actor))) {
            return false;
        }
        self.command("PICKUP", item, need)
    }
    fn finish(&self) -> Option<Route> {
        let mut b = self.clone();
        if b.private.inventories[b.actor].sum() > 0 {
            if !b.walk(home(pos(&b.farm, b.actor))) {
                return None;
            }
            let cargo = b.private.inventories[b.actor].sum();
            let room = 100 - b.private.shed.sum();
            if cargo <= room {
                if !b.command("DROP", "", 0) {
                    return None;
                }
            } else {
                let entries = b.private.inventories[b.actor].0.clone();
                let mut deposited = false;
                for (item, q) in entries {
                    let n = q.min(100 - b.private.shed.sum());
                    if n > 0 && b.command("PLACE", &item, n) {
                        deposited = true;
                    }
                }
                if !deposited {
                    return None;
                }
            }
        }
        (!b.route.steps.is_empty()).then_some(b.route)
    }
    fn service(&self, o: &Observation, e: &Executor, target: Pos, mode: usize) -> Option<Self> {
        self.service_mode(o, e, target, mode, true).or_else(|| {
            e.harvest_successors
                .contains_key(&target)
                .then(|| self.service_mode(o, e, target, mode, false))
                .flatten()
        })
    }
    fn service_mode(
        &self,
        o: &Observation,
        e: &Executor,
        target: Pos,
        mode: usize,
        allow_handoff: bool,
    ) -> Option<Self> {
        let mut b = self.clone();
        let mut desired = e.projects.get(&target).map(|p| p.production.clone());
        let successor = if allow_handoff {
            e.harvest_successors.get(&target).cloned()
        } else {
            None
        };
        let handoff_animal = successor.as_ref().and_then(|p| {
            if let Production::Animal(a) = p {
                Some(a)
            } else {
                None
            }
        });
        if let Some(animal) = handoff_animal {
            if !b.pickup(animal, 1) {
                return None;
            }
        }

        let initial = tile(&b.farm, target).clone();
        // Reserve confirmed seed stock for earlier approved projects before optional renewal.
        let seed_for_target = |name: &str, available: i64| {
            let own = e.projects.get(&target);
            let priority = own
                .map(|p| (!p.confirmed, p.requested))
                .unwrap_or((false, i64::MAX));
            let reserved = e
                .projects
                .iter()
                .filter(|(p, project)| {
                    **p != target
                        && !project.confirmed
                        && project.production == Production::Crop(name.into())
                        && (!priority.0 || (project.requested, **p) < (priority.1, target))
                })
                .count() as i64;
            available > reserved
        };

        if let Some(Production::Animal(ref name)) = desired {
            if !matches!(
                initial,
                Cell::Structure {
                    animal: Some(_),
                    ..
                }
            ) && !b.pickup(name, 1)
            {
                return None;
            }
        }
        if let Cell::Structure {
            animal: Some(ref animal),
            ..
        } = initial
        {
            if desired != Some(Production::Vacant)
                && !animal.fed_today
                && b.private.inventories[b.actor].get("WHEAT") == 0
            {
                // Take a bounded tour's feed once, then share the trip across animals and crops.
                let need = e
                    .projects
                    .values()
                    .filter(|p| matches!(p.production, Production::Animal(_)))
                    .count()
                    .clamp(1, 4) as i64;
                let available = b.private.shed.get("WHEAT");
                if available > 0 {
                    b.pickup("WHEAT", need.min(available));
                }
            }
        }
        if !b.walk(target) {
            return None;
        }
        let work_before = b.route.work;
        let retiring = desired == Some(Production::Vacant);
        match initial {
            Cell::Plant {
                ref crop,
                planted_day,
                ..
            } => {
                let spec = rules::crop(crop)?;
                // Conversion of mature ongoing crops is an explicit investment decision.
                let replacing = desired.as_ref().is_some_and(|d| d.name() != crop)
                    && match &desired {
                        Some(Production::Crop(c)) => seed_for_target(c, b.private.seeds.get(c)),
                        _ => true,
                    };
                if replacing {
                    if matches!(tile(&b.farm,target),Cell::Plant{yield_units,..} if *yield_units>0&&o.day()-planted_day>=spec.first_yield_day)
                    {
                        b.command("HARVEST", "", 0);
                    }
                    b.command("DIG", "", 0);
                } else {
                    if mode != 2
                        && matches!(tile(&b.farm,target),Cell::Plant{fertilized_until_day,..} if *fertilized_until_day<o.day())
                        && b.private.inventories[b.actor].get("FERTILIZER") > 0
                    {
                        b.command("FERTILIZE", "", 0);
                    }
                    if o.day() < 29
                        && matches!(
                            tile(&b.farm, target),
                            Cell::Plant {
                                watered_today: false,
                                ..
                            }
                        )
                    {
                        b.command("WATER", "", 0);
                    }
                    if let Cell::Plant { yield_units, .. } = tile(&b.farm, target) {
                        if *yield_units > 0
                            && o.day() - planted_day >= spec.first_yield_day
                            && (mode == 1
                                || spec.ongoing
                                || o.day() - planted_day >= spec.max_yield_day
                                || *yield_units >= spec.max_yield
                                || o.day() == 29)
                        {
                            b.command("HARVEST", "", 0);
                        }
                    }
                }
                b.route.crop_jobs += usize::from(b.route.work > work_before);
            }
            Cell::Structure {
                animal: Some(a), ..
            } => {
                if !retiring && !a.fed_today && b.private.inventories[b.actor].get("WHEAT") > 0 {
                    b.command("FEED", "", 0);
                }
                if !retiring && mode != 2 && !a.cared_today {
                    b.command("CARE", "", 0);
                }
                if a.yield_units > 0 {
                    b.command("HARVEST", "", 0);
                }
                if a.fertilizer_available && mode != 2 {
                    b.command("COLLECT_FERTILIZER", "", 0);
                }
                b.route.animal_jobs += usize::from(b.route.work > work_before);
            }
            Cell::Weed => {
                if desired.is_some() {
                    b.command("DIG", "", 0);
                }
            }
            _ => {}
        }
        if let Some(next) = &successor {
            // Actual simulated HARVEST must precede the successor. Never clear an
            // immature crop just because a batch owns a future promise.
            if !b
                .route
                .steps
                .iter()
                .any(|s| s.position == target && s.action.op == "HARVEST")
            {
                return None;
            }
            if matches!(tile(&b.farm, target), Cell::Plant { .. }) && !b.command("DIG", "", 0) {
                return None;
            }
            desired = Some(next.clone());
        }
        match desired {
            Some(Production::Crop(c)) => {
                if matches!(tile(&b.farm, target), Cell::Structure { animal: None, .. }) {
                    b.command("DIG", "", 0);
                }
                if *tile(&b.farm, target) == Cell::Empty
                    && seed_for_target(&c, b.private.seeds.get(&c))
                    && o.day() + rules::crop(&c)?.first_yield_day < 30
                {
                    let harvested = b
                        .route
                        .steps
                        .iter()
                        .any(|s| s.position == target && s.action.op == "HARVEST");
                    if b.command("PLANT", &c, 1) {
                        b.route.replants += usize::from(harvested);
                        b.command("WATER", "", 0);
                        b.route.crop_jobs += 1;
                    }
                }
            }
            Some(Production::Animal(name)) => {
                let animal = rules::animal(&name)?;
                if matches!(tile(&b.farm,target),Cell::Structure{animal:None,kind} if kind!=animal.structure)
                {
                    b.command("DIG", "", 0);
                }
                if *tile(&b.farm, target) == Cell::Empty {
                    b.command(&format!("BUILD_{}", animal.structure), "", 0);
                }
                if matches!(tile(&b.farm,target),Cell::Structure{animal:None,kind} if kind==animal.structure)
                    && b.private.inventories[b.actor].get(&name) > 0
                {
                    if b.command("PLACE", &name, 1) {
                        b.route.animal_jobs += 1;
                    }
                }
            }
            Some(Production::Vacant) => {
                if matches!(
                    tile(&b.farm, target),
                    Cell::Weed | Cell::Structure { animal: None, .. }
                ) {
                    b.command("DIG", "", 0);
                }
            }
            None => {}
        }
        if let Some(next) = successor {
            let placed = b.route.steps.iter().any(|s| {
                s.position == target
                    && s.action.item == next.name()
                    && matches!(s.action.op.as_str(), "PLANT" | "PLACE")
            });
            if !placed {
                return None;
            }
            if e.harvest_successor_deadlines
                .get(&target)
                .is_some_and(|deadline| {
                    b.route.steps.iter().any(|s| {
                        s.position == target
                            && s.action.item == next.name()
                            && matches!(s.action.op.as_str(), "PLANT" | "PLACE")
                            && s.at > *deadline
                    })
                })
            {
                return None;
            }
            b.route.production_handoffs.insert(target, next);
        }
        if b.route.work == work_before {
            return None;
        }
        b.route.sites.insert(target);
        b.finish()?;
        Some(b)
    }
}
pub fn route_problem(o: &Observation, e: &Executor, actor: usize) -> Problem {
    route_problem_impl(o, e, actor, false)
}
/// Opt-in prototype: existing checkpoint candidate ordering remains unchanged.
pub fn economic_route_problem(o: &Observation, e: &Executor, actor: usize) -> Problem {
    route_problem_impl(o, e, actor, true)
}
pub fn route_value(route: &Route) -> f64 {
    route.economic_value / (route.steps.len().max(1) as f64).powf(0.65)
        - route.walking as f64 * 0.25
}
fn route_problem_impl(o: &Observation, e: &Executor, actor: usize, economic: bool) -> Problem {
    let mut choices = vec![Choice::Continue];
    let (reserved, _, _) = e.reserved(actor);
    let start = Builder::new(o, e, actor);
    if let Some(route) = start.finish() {
        choices.push(Choice::Route { actor, route });
    }
    let mut targets: BTreeSet<_> = e.projects.keys().copied().collect();
    for y in 0..10 {
        for x in 0..10 {
            if matches!(
                tile(&o.farm, (x, y)),
                Cell::Plant { .. }
                    | Cell::Structure {
                        animal: Some(_),
                        ..
                    }
            ) {
                targets.insert((x, y));
            }
        }
    }
    targets.retain(|p| !reserved.contains(p));
    let mut seen = BTreeSet::new();
    for mode in 0..3 {
        let mut beam = vec![start.clone()];
        for _ in 0..4 {
            let mut next = Vec::new();
            for b in &beam {
                let mut nearby: Vec<_> = targets
                    .iter()
                    .copied()
                    .filter(|p| !b.route.sites.contains(p))
                    .collect();
                nearby.sort_by_key(|p| {
                    (
                        if economic
                            && e.service_deadlines
                                .get(p)
                                .is_some_and(|t| *t <= o.step + 24)
                        {
                            0
                        } else {
                            1
                        },
                        distance(pos(&b.farm, actor), *p),
                        *p,
                    )
                });
                for p in nearby.into_iter().take(8) {
                    if let Some(b) = b.service(o, e, p, mode) {
                        next.push(b);
                    }
                }
            }
            next.sort_by(|a, b| {
                if economic {
                    let urgency = |r: &Route| {
                        r.sites
                            .iter()
                            .filter(|site| {
                                e.service_deadlines
                                    .get(site)
                                    .is_some_and(|t| *t <= o.step + 24)
                            })
                            .count() as f64
                            * 20.
                    };
                    return (route_value(&b.route) + urgency(&b.route))
                        .total_cmp(&(route_value(&a.route) + urgency(&a.route)))
                        .then_with(|| a.route.sites.cmp(&b.route.sites));
                }
                let score = |x: &Builder| {
                    x.route.work as i64 * 4
                        + x.route.harvested * 2
                        + x.route.reused_fertilizer as i64 * 3
                        - x.route.walking as i64
                };
                score(b)
                    .cmp(&score(a))
                    .then_with(|| a.route.sites.cmp(&b.route.sites))
            });
            next.truncate(3);
            for b in &next {
                if let Some(route) = b.finish() {
                    let key = route
                        .steps
                        .iter()
                        .map(|s| {
                            format!("{}:{}:{}:{}", s.at, s.action.op, s.action.item, s.action.n)
                        })
                        .collect::<Vec<_>>()
                        .join("|");
                    if seen.insert(key) {
                        choices.push(Choice::Route { actor, route });
                    }
                }
            }
            beam = next;
            if beam.is_empty() {
                break;
            }
        }
    }
    // An approved project is a commitment: choose how to service it, not whether
    // to silently abandon it. Continue remains legal only when no route can run.
    // Investment decisions still permit waiting, switching production and exiting.
    if choices.len() > 1 {
        choices.remove(0);
    }
    Problem {
        actor: Some(actor),
        choices,
    }
}
fn order(op: &str, item: &str, n: i64) -> Vec<String> {
    vec![op.into(), item.into(), n.to_string()]
}
pub fn investment_problem(o: &Observation, e: &Executor) -> Problem {
    let mut choices = vec![Choice::Continue];
    let cash = o.farm.money;
    let reserved: BTreeSet<_> = e
        .routes
        .iter()
        .flatten()
        .flat_map(|r| r.sites.iter().copied())
        .collect();
    let funding = |orders: &[Vec<String>]| -> Option<f64> {
        let mut total = 0.;
        let mut hires = o.farm.hires_today as u32;
        for row in orders {
            match row[0].as_str() {
                "HIRE" => {
                    total += rules::hire_cost(hires, 1) as f64;
                    hires += 1;
                }
                "BUY_LAND" => {
                    total += rules::next_land(o.farm.unlocked_quadrants.len() - 1)?.1 as f64
                }
                "BUY_SEED" => {
                    total += rules::crop(&row[1])?.seed_cost as f64 * row[2].parse::<f64>().ok()?
                }
                "BUY_ANIMAL" => {
                    total += rules::animal(&row[1])?.cost as f64 * row[2].parse::<f64>().ok()?
                }
                "BUY_PRODUCT" => {
                    let n = row[2].parse::<i64>().ok()?;
                    let p = kagg_engine::market::param(&row[1])?;
                    total += (0..n)
                        .map(|i| {
                            kagg_engine::market::price(
                                p,
                                (o.market.inventory.get(&row[1]) - i - 10) as f64,
                            )
                        })
                        .sum::<i64>() as f64;
                }
                _ => {}
            }
        }
        (total <= cash).then_some(total)
    };
    let pending_projects = e
        .projects
        .values()
        .filter(|p| !p.confirmed && p.production != Production::Vacant)
        .count();
    let mut add = |site, production: Option<Production>, orders: Vec<Vec<String>>| {
        if orders.len() <= 10 {
            if let Some(cost) = funding(&orders) {
                if production
                    .as_ref()
                    .is_some_and(|p| *p != Production::Vacant)
                    && (pending_projects >= 2
                        || e.new_projects_today >= 2
                        || e.expansion_spent + cost > e.expansion_budget)
                {
                    return;
                }
                choices.push(Choice::Invest {
                    site,
                    production,
                    orders,
                    cost,
                });
            }
        }
    };
    let load = e.labor_load();
    let potential_workers = load.div_ceil(14).clamp(1, 8);
    if o.step % 24 <= 8 && o.farm.hands.len() + 1 < potential_workers && load > 0 {
        add(None, None, vec![vec!["HIRE".into()]]);
    }
    if o.day() < 24 && e.projects.len() >= 12 && o.farm.unlocked_quadrants.len() < 4 {
        add(None, None, vec![vec!["BUY_LAND".into()]]);
    }
    // Replenishment is proposed from aggregate project obligations, not guessed receipts.
    for item in CROP_NAMES {
        let required = e
            .projects
            .values()
            .filter(|p| matches!(&p.production,Production::Crop(c) if c==item))
            .count() as i64;
        let need = (required - o.private.seeds.get(item)).max(0).min(4);
        if need > 0 && o.day() + rules::crop(item).unwrap().first_yield_day < 30 {
            add(None, None, vec![order("BUY_SEED", item, need)]);
        }
    }
    let animals = e
        .projects
        .values()
        .filter(|p| matches!(p.production, Production::Animal(_)))
        .count() as i64;
    let wheat = o.private.shed.get("WHEAT")
        + o.private
            .inventories
            .iter()
            .map(|i| i.get("WHEAT"))
            .sum::<i64>();
    let need = (animals * 2 - wheat)
        .max(0)
        .min(8)
        .min((100 - o.private.shed.sum()).max(0));
    if need > 0 && e.market_mode == super::trading::MarketMode::Rule {
        add(None, None, vec![order("BUY_PRODUCT", "WHEAT", need)]);
    }
    let mut sites = Vec::new();
    for y in 0..10 {
        for x in 0..10 {
            let p = (x, y);
            if reserved.contains(&p)
                || e.projects.get(&p).is_some_and(|pr| {
                    !pr.confirmed
                        && pr.production != Production::Vacant
                        && o.step - pr.requested <= 24
                })
            {
                continue;
            }
            match tile(&o.farm, p) {
                Cell::Empty | Cell::Weed | Cell::Structure { animal: None, .. }
                    if !e.projects.contains_key(&p)
                        || e.projects
                            .get(&p)
                            .is_some_and(|p| p.production == Production::Vacant) =>
                {
                    sites.push(p)
                }
                Cell::Plant {
                    crop,
                    planted_day,
                    yield_units,
                    ..
                } if *yield_units > 0
                    && o.day() - planted_day >= rules::crop(crop).unwrap().first_yield_day =>
                {
                    sites.push(p)
                }
                _ => {}
            }
        }
    }
    sites.sort_by_key(|p| (distance(*p, home(*p)), *p));
    // Keep location alternatives alongside all crop/species alternatives.
    for p in sites.into_iter().take(3) {
        let existing = e.projects.get(&p);
        let expansion =
            existing.is_none() || existing.is_some_and(|p| p.production == Production::Vacant);
        if expansion && load + 2 > potential_workers * 16 {
            continue;
        }
        for name in CROP_NAMES {
            if o.day() + rules::crop(name).unwrap().first_yield_day >= 30
                || existing.is_some_and(|p| p.production == Production::Crop(name.into()))
            {
                continue;
            }
            let required = e
                .projects
                .values()
                .filter(|p| p.production == Production::Crop(name.into()))
                .count() as i64
                + 1;
            let orders = if o.private.seeds.get(name) < required {
                vec![order("BUY_SEED", name, 1)]
            } else {
                vec![]
            };
            add(Some(p), Some(Production::Crop(name.into())), orders);
        }
        for name in ANIMAL_NAMES {
            if o.day() + rules::animal(name).unwrap().first_yield_day >= 30
                || load + 4 > potential_workers * 16
            {
                continue;
            }
            let animal_stock = o.private.shed.get(name)
                + o.private
                    .inventories
                    .iter()
                    .map(|i| i.get(name))
                    .sum::<i64>();
            let pending = e
                .projects
                .iter()
                .filter(|(q, v)| {
                    v.production == Production::Animal(name.into())
                        && !matches!(
                            tile(&o.farm, **q),
                            Cell::Structure {
                                animal: Some(_),
                                ..
                            }
                        )
                })
                .count() as i64;
            let mut orders = if animal_stock > pending {
                vec![]
            } else if o.private.shed.sum() < 98 {
                vec![order("BUY_ANIMAL", name, 1)]
            } else {
                continue;
            };
            let feed = (2 - wheat).max(0);
            if feed > 0 && e.market_mode == super::trading::MarketMode::Rule {
                orders.push(order("BUY_PRODUCT", "WHEAT", feed));
            }
            add(Some(p), Some(Production::Animal(name.into())), orders);
        }
    }
    for (p, project) in &e.projects {
        if !reserved.contains(p)
            && project.production != Production::Vacant
            && (project.confirmed || o.step - project.requested > 24)
        {
            add(Some(*p), Some(Production::Vacant), vec![]);
        }
    }
    choices.truncate(96);
    Problem {
        actor: None,
        choices,
    }
}

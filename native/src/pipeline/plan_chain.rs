//! Executable two-stage production commitments. Only public/own observations enter proposals.
//! Purchases use received cash, transitions use observed stock and harvest receipts.
use super::{
    executor::*,
    plan_learning,
    plan_prototype::{Agent, Config, PlanningCache},
};
use crate::learning::policy::{Policy, Rng, Sample};
use kagg_engine::{
    engine::PlayerAction,
    json::Json,
    rules,
    state::{Cell, ANIMAL_NAMES, CROP_NAMES},
};
use std::collections::BTreeMap;

pub const CONTRACT: &str = "plan-chain-320x32-v2";
pub const SCHEMA: &str = "plan-comparison-v2";
fn kinds() -> Vec<Production> {
    CROP_NAMES
        .iter()
        .map(|s| Production::Crop((*s).into()))
        .chain(ANIMAL_NAMES.iter().map(|s| Production::Animal((*s).into())))
        .collect()
}
fn index(p: &Production) -> usize {
    kinds().iter().position(|k| k == p).unwrap()
}
fn cost(p: &Production) -> f64 {
    match p {
        Production::Crop(c) => rules::crop(c).unwrap().seed_cost as f64,
        Production::Animal(a) => rules::animal(a).unwrap().cost as f64,
        _ => 0.,
    }
}
fn duration(p: &Production) -> i64 {
    match p {
        Production::Crop(c) => rules::crop(c).unwrap().max_yield_day,
        Production::Animal(a) => rules::animal(a).unwrap().first_yield_day,
        _ => 30,
    }
}
fn actual(t: &Cell) -> Option<(Production, i64)> {
    match t {
        Cell::Plant {
            crop, planted_day, ..
        } => Some((Production::Crop(crop.clone()), *planted_day)),
        Cell::Structure {
            animal: Some(a), ..
        } => Some((Production::Animal(a.animal.clone()), a.placed_day)),
        _ => None,
    }
}
fn product(p: &Production) -> &str {
    match p {
        Production::Crop(c) => c,
        Production::Animal(a) => rules::animal(a).unwrap().product,
        _ => "",
    }
}
#[derive(Clone, Debug)]
pub struct Link {
    pub site: Pos,
    pub first: Production,
    pub next: Production,
    pub cycles: usize,
}
impl Link {
    pub fn json(&self) -> Json {
        Json::Obj(vec![
            (
                "site".into(),
                Json::Arr(vec![
                    Json::Num(self.site.0 as f64),
                    Json::Num(self.site.1 as f64),
                ]),
            ),
            ("first".into(), Json::Str(self.first.name().into())),
            ("next".into(), Json::Str(self.next.name().into())),
            (
                "harvests_before_switch".into(),
                Json::Num(self.cycles as f64),
            ),
        ])
    }
}
#[derive(Clone)]
pub struct Choice {
    pub base: Option<plan_learning::Candidate>,
    pub links: Vec<Link>,
    pub features: Vec<f32>,
}
impl Choice {
    pub fn json(&self) -> Json {
        Json::Obj(vec![
            (
                "links".into(),
                Json::Arr(self.links.iter().map(Link::json).collect()),
            ),
            (
                "orders".into(),
                Json::Arr(
                    self.base
                        .as_ref()
                        .map(|c| {
                            c.orders
                                .iter()
                                .map(|o| Json::Arr(o.iter().cloned().map(Json::Str).collect()))
                                .collect()
                        })
                        .unwrap_or_default(),
                ),
            ),
            (
                "features".into(),
                Json::Arr(self.features.iter().map(|x| Json::Num(*x as f64)).collect()),
            ),
        ])
    }
}
#[derive(Clone)]
pub struct Progress {
    pub link: Link,
    pub requested: i64,
    pub first_started: bool,
    pub first_marker: i64,
    pub first_harvests: usize,
    pub expected_ready: i64,
    pub armed: bool,
    pub retiring: bool,
    pub successor_started: bool,
    pub successor_yielded: bool,
    pub failed: bool,
    pub superseded: bool,
    pub waiting_material_steps: usize,
}
impl Progress {
    pub fn json(&self) -> Json {
        let mut j = self.link.json();
        for (k, v) in [
            ("requested", Json::Num(self.requested as f64)),
            ("first_started", Json::Bool(self.first_started)),
            ("first_harvests", Json::Num(self.first_harvests as f64)),
            ("expected_ready", Json::Num(self.expected_ready as f64)),
            ("successor_started", Json::Bool(self.successor_started)),
            ("successor_yielded", Json::Bool(self.successor_yielded)),
            ("failed", Json::Bool(self.failed && !self.superseded)),
            ("superseded", Json::Bool(self.superseded)),
            (
                "waiting_material_steps",
                Json::Num(self.waiting_material_steps as f64),
            ),
        ] {
            j.set_path(k, v);
        }
        j
    }
}
#[derive(Clone)]
struct Receipt {
    site: Pos,
    actor: usize,
    item: String,
    before: i64,
    successor: bool,
}
#[derive(Clone)]
pub struct Controller {
    pub agent: Agent,
    pub progress: Vec<Progress>,
    active: BTreeMap<Pos, usize>,
    receipts: Vec<Receipt>,
    pub decisions: usize,
    pub batches: Vec<super::plan_resources::BatchPlan>,
    pub batch_revisions: usize,
    pub batch_cancellations: usize,
    pub hold_transitions: bool,
}
impl Controller {
    pub fn new(config: Config) -> Self {
        Self {
            agent: Agent::new(config),
            progress: vec![],
            active: BTreeMap::new(),
            receipts: vec![],
            decisions: 0,
            batches: vec![],
            batch_revisions: 0,
            batch_cancellations: 0,
            hold_transitions: false,
        }
    }
    pub fn event_mode(&self) -> bool {
        self.batches.iter().any(|b| {
            !b.cancelled
                && b.stage.links.iter().any(|id| {
                    let p = &self.progress[*id];
                    self.is_active(*id) && !p.failed && !p.successor_yielded
                })
        })
    }
    pub fn pending_at(&self, site: Pos) -> bool {
        self.active
            .get(&site)
            .is_some_and(|i| !self.progress[*i].successor_started && !self.progress[*i].failed)
    }
    pub fn is_active(&self, id: usize) -> bool {
        self.active.values().any(|i| *i == id)
    }
    pub fn pending_id(&self, site: Pos) -> Option<usize> {
        self.active.get(&site).copied()
    }
    pub fn editable(&self, o: &Observation, site: Pos) -> bool {
        if self.agent.executor.reserved(usize::MAX).0.contains(&site) {
            return false;
        }
        (matches!(tile(&o.farm, site), Cell::Plant { .. })
            || (matches!(tile(&o.farm, site), Cell::Empty) && self.pending_at(site)))
            && self.active.get(&site).is_none_or(|i| {
                let p = &self.progress[*i];
                !p.armed && !p.successor_started && !p.retiring && !p.failed
            })
    }
    /// Atomic suffix replacement. Already planted crops and stock are never erased.
    pub fn revise_batch(
        &mut self,
        o: &Observation,
        sites: &[Pos],
        next: Option<Production>,
        cycles: usize,
        lead: i64,
        floor: f64,
    ) -> Result<usize, String> {
        self.revise_batch_mode(o, sites, next, cycles, lead, floor, false)
    }
    pub fn conditional_batch(&self, id: usize) -> bool {
        self.batches
            .get(id)
            .is_some_and(|b| !b.cancelled && b.stage.conditional)
    }
    pub fn revise_batch_mode(
        &mut self,
        o: &Observation,
        sites: &[Pos],
        next: Option<Production>,
        cycles: usize,
        lead: i64,
        floor: f64,
        conditional: bool,
    ) -> Result<usize, String> {
        if sites.is_empty() || sites.len() > 4 || sites.iter().any(|p| !self.editable(o, *p)) {
            return Err("batch is no longer editable".into());
        }
        if ![0, 12, 24, 48].contains(&lead)
            || !(1..=2).contains(&cycles)
            || !floor.is_finite()
            || floor < 0.
        {
            return Err("invalid batch funding/timing".into());
        }
        let unique: std::collections::BTreeSet<_> = sites.iter().copied().collect();
        if unique.len() != sites.len() {
            return Err("duplicate batch sites".into());
        }
        if let Some(ref kind) = next {
            if sites.iter().any(|site| {
                super::plan_resources::revision_ready(self, o, *site, cycles)
                    + super::plan_resources::duration(kind) * 24
                    + 24
                    >= 719
            }) {
                return Err("current batch and successor cannot finish this season".into());
            }
            let ledger = super::plan_resources::Schedule::build(self, o);
            let replacement_work = sites
                .iter()
                .map(|s| {
                    2. * distance(*s, home(*s)) as f64
                        + if matches!(kind, Production::Animal(_)) {
                            10.
                        } else {
                            5.
                        }
                })
                .sum::<f64>();
            let removed_work = ledger
                .needs
                .iter()
                .filter(|n| sites.contains(&n.site) && n.ready - n.lead <= o.step)
                .map(|n| n.work)
                .sum::<f64>();
            if !conditional && ledger.work_due - removed_work + replacement_work > ledger.free_work
            {
                return Err("batch exceeds available service windows".into());
            }
        }
        let mut ids = vec![];
        for &site in sites {
            let old = self.active.remove(&site);
            let prior = old.map(|id| self.progress[id].clone());
            if let Some(old) = old {
                self.progress[old].failed = true;
                self.progress[old].superseded = true;
            }
            if let Some(ref kind) = next {
                let empty = matches!(tile(&o.farm, site), Cell::Empty);
                let (first, marker) = actual(tile(&o.farm, site))
                    .or_else(|| {
                        prior
                            .as_ref()
                            .map(|p| (p.link.first.clone(), p.first_marker))
                    })
                    .ok_or("missing first production")?;
                let id = self.progress.len();
                let link = Link {
                    site,
                    first,
                    next: kind.clone(),
                    cycles,
                };
                let expected_ready =
                    super::plan_resources::ready_step(o, &link, usize::from(empty));
                self.progress.push(Progress {
                    link,
                    requested: o.step,
                    first_started: true,
                    first_marker: marker,
                    first_harvests: usize::from(empty),
                    expected_ready,
                    armed: false,
                    retiring: false,
                    successor_started: false,
                    successor_yielded: false,
                    failed: false,
                    superseded: false,
                    waiting_material_steps: 0,
                });
                self.active.insert(site, id);
                ids.push(id);
            }
        }
        // Remove only revised sites from earlier reservation owners; unaffected sites keep working.
        for b in &mut self.batches {
            let previous_len = b.stage.links.len();
            b.stage
                .links
                .retain(|id| !sites.contains(&self.progress[*id].link.site));
            b.sites.retain(|s| !sites.contains(s));
            if previous_len != b.stage.links.len() {
                b.revision += 1;
            }
            if b.stage.links.is_empty() {
                b.cancelled = true;
            }
        }
        let id = self.batches.len();
        let deadline = next
            .as_ref()
            .map(|k| 718 - super::plan_resources::duration(k) * 24)
            .unwrap_or(718);
        self.batches.push(super::plan_resources::BatchPlan {
            id,
            revision: 0,
            sites: sites.to_vec(),
            stage: super::plan_resources::Stage {
                route_handoff: false,
                conditional,
                links: ids,
                lead_steps: lead,
                cash_floor: floor,
                deadline,
            },
            cancelled: next.is_none(),
        });
        self.batch_revisions += 1;
        if next.is_none() {
            self.batch_cancellations += 1;
        }
        Ok(id)
    }
    pub fn observe(&mut self, o: &Observation) {
        self.agent.observe(o);
        for r in self.receipts.drain(..) {
            if o.private
                .inventories
                .get(r.actor)
                .is_some_and(|i| i.get(&r.item) > r.before)
            {
                if let Some(id) = self.active.get(&r.site) {
                    let p = &mut self.progress[*id];
                    if r.successor {
                        p.successor_yielded = true;
                    } else {
                        p.first_harvests += 1;
                        // Commit only after the real harvest receipt. The route
                        // already owns its prepared successor; stale/cancelled
                        // routes cannot prematurely change the current project.
                        let handoff = self
                            .agent
                            .executor
                            .routes
                            .get(r.actor)
                            .and_then(Option::as_ref)
                            .and_then(|route| route.production_handoffs.get(&r.site));
                        if handoff == Some(&p.link.next) && p.first_harvests >= p.link.cycles {
                            p.armed = true;
                            self.agent.executor.projects.insert(
                                r.site,
                                Project {
                                    production: p.link.next.clone(),
                                    requested: o.step,
                                    confirmed: false,
                                    failures: 0,
                                },
                            );
                        }
                    }
                }
            }
        }
        // Resource arrival may happen AFTER a normal harvest/renewal route was
        // assigned. At the confirmed empty boundary revise just that renewal,
        // not the worker's other committed tasks or existing reservations.
        if !self.hold_transitions && self.event_mode() {
            let ledger = super::plan_resources::Schedule::build(self, o);
            let mut work = 0.;
            for n in &ledger.needs {
                let p = &self.progress[n.id];
                let routed = self.batches.iter().any(|b| {
                    !b.cancelled && b.stage.route_handoff && b.stage.links.contains(&n.id)
                });
                if routed
                    && matches!(tile(&o.farm, n.site), Cell::Empty)
                    && p.first_harvests >= p.link.cycles
                    && n.stocked
                    && n.feed_stocked
                    && o.farm.money >= n.cash_floor
                    && o.step <= n.deadline
                    && work + n.work <= ledger.free_work
                {
                    if self
                        .agent
                        .executor
                        .release_harvested_renewal(n.site, &p.link.first)
                    {
                        work += n.work;
                    }
                }
            }
        }
        let reserved = self.agent.executor.reserved(usize::MAX).0;
        let schedule = self
            .event_mode()
            .then(|| super::plan_resources::Schedule::build(self, o));
        let mut admitted_work = 0.;
        for (&site, &id) in &self.active {
            let p = &mut self.progress[id];
            let a = actual(tile(&o.farm, site));
            if !p.first_started && a.as_ref().is_some_and(|(k, _)| *k == p.link.first) {
                p.first_started = true;
                p.first_marker = a.as_ref().unwrap().1;
                p.expected_ready = super::plan_resources::ready_step(o, &p.link, p.first_harvests);
            }
            if p.armed
                && a.as_ref().is_some_and(|(k, day)| {
                    *k == p.link.next && (p.link.next != p.link.first || *day != p.first_marker)
                })
            {
                p.successor_started = true;
            }
            if p.successor_yielded || p.failed {
                continue;
            }
            // An unfulfilled purchase/placement or a season-infeasible successor is explicit failure.
            if o.step >= 719
                || (!p.first_started && o.step - p.requested > 72)
                || (!p.successor_started && o.day() + duration(&p.link.next) >= 30)
            {
                p.failed = true;
                if !reserved.contains(&site) {
                    if let Some((k, _)) = a {
                        if let Some(pr) = self.agent.executor.projects.get_mut(&site) {
                            pr.production = k;
                            pr.confirmed = true;
                        }
                    }
                }
                continue;
            }
            if p.armed
                && !p.successor_started
                && self
                    .agent
                    .executor
                    .projects
                    .get(&site)
                    .is_none_or(|pr| pr.production != p.link.next)
            {
                p.armed = false;
            }
            if self.hold_transitions
                || p.successor_started
                || !p.first_started
                || reserved.contains(&site)
            {
                continue;
            }
            let final_harvest_ready = match tile(&o.farm, site) {
                Cell::Plant {
                    crop,
                    planted_day,
                    yield_units,
                    ..
                } => {
                    *yield_units > 0
                        && o.day() - planted_day >= rules::crop(crop).unwrap().max_yield_day
                        && p.first_harvests + 1 >= p.link.cycles
                }
                _ => false,
            };
            let empty = actual(tile(&o.farm, site)).is_none();
            let finished = p.first_harvests >= p.link.cycles;
            let harvested_ongoing = matches!(tile(&o.farm,site), Cell::Plant{crop,planted_day,..}
                if rules::crop(crop).unwrap().ongoing && finished && o.day()-planted_day>=rules::crop(crop).unwrap().max_yield_day);
            let animal_finished = matches!(
                tile(&o.farm, site),
                Cell::Structure {
                    animal: Some(_),
                    ..
                }
            ) && finished;
            // A route may have already replanted the original crop while a successor
            // purchase was waiting. Preserve that new immature crop until harvest.
            let revised_suffix = self
                .batches
                .iter()
                .any(|b| !b.cancelled && b.stage.links.contains(&id));
            // A harvested empty plot still needs the promised extra first-stage
            // cycle before conversion. Preserve legacy links' original behavior.
            let empty_ready = empty && (!revised_suffix || finished);
            if !(final_harvest_ready || harvested_ongoing || animal_finished || empty_ready) {
                continue;
            }
            // Retire animals only after the selected number of observed harvests.
            // Existing engine retirement is starvation/exit; continue collecting held output.
            if matches!(
                tile(&o.farm, site),
                Cell::Structure {
                    animal: Some(_),
                    ..
                }
            ) {
                if finished {
                    p.retiring = true;
                    if let Some(pr) = self.agent.executor.projects.get_mut(&site) {
                        pr.production = Production::Vacant;
                    }
                }
                continue;
            }
            let stock = match &p.link.next {
                Production::Crop(c) => o.private.seeds.get(c),
                Production::Animal(a) => {
                    o.private.shed.get(a)
                        + o.private.inventories.iter().map(|i| i.get(a)).sum::<i64>()
                }
                _ => 0,
            };
            // Account for all other placements and route reservations before binding this site.
            let obligations = self
                .agent
                .executor
                .projects
                .iter()
                .filter(|(s, pr)| **s != site && !pr.confirmed && pr.production == p.link.next)
                .count() as i64;
            let (_, seeds, pickups) = self.agent.executor.reserved(usize::MAX);
            let in_routes = if matches!(p.link.next, Production::Crop(_)) {
                seeds.get(p.link.next.name())
            } else {
                pickups.get(p.link.next.name())
            };
            let approved = if let Some(s) = &schedule {
                if let Some(n) = s.needs.iter().find(|n| n.id == id) {
                    let ready = n.stocked
                        && (!n.conditional || (n.feed_stocked && o.farm.money >= n.cash_floor))
                        && o.step <= n.deadline
                        && n.work + admitted_work <= s.free_work;
                    if ready {
                        admitted_work += n.work;
                    }
                    ready
                } else {
                    stock > obligations.max(in_routes)
                }
            } else {
                stock > obligations.max(in_routes)
            };
            if approved {
                if !p.armed {
                    self.agent.executor.projects.insert(
                        site,
                        Project {
                            production: p.link.next.clone(),
                            requested: o.step,
                            confirmed: false,
                            failures: 0,
                        },
                    );
                    p.first_marker = actual(tile(&o.farm, site))
                        .map(|(_, day)| day)
                        .unwrap_or(p.first_marker);
                    p.armed = true;
                }
            }
        }
        self.active
            .retain(|_, id| !self.progress[*id].successor_yielded && !self.progress[*id].failed);
    }
    /// Stock just before the switching window. No projected sale proceeds are spent.
    fn supplies(&mut self, o: &Observation, mut orders: Vec<Vec<String>>) -> Vec<Vec<String>> {
        if self.event_mode() {
            let ledger = super::plan_resources::Schedule::build(self, o);
            let extra = ledger.orders(o, &orders);
            orders.extend(extra);
            return orders;
        }
        let mut spend = order_cost(o, &orders);
        let mut budget = (o.farm.money - self.agent.config.cash_reserve - spend).max(0.);
        let mut wanted: BTreeMap<String, (Production, i64)> = BTreeMap::new();
        for id in self.active.values() {
            let p = &self.progress[*id];
            if !p.first_started || p.successor_started {
                continue;
            }
            let near = match tile(&o.farm, p.link.site) {
                Cell::Plant {
                    crop, planted_day, ..
                } => {
                    o.day() - planted_day >= rules::crop(crop).unwrap().max_yield_day - 1
                        && p.first_harvests + 1 >= p.link.cycles
                }
                Cell::Structure {
                    animal: Some(_), ..
                } => p.first_harvests >= p.link.cycles,
                _ => true,
            };
            if near {
                wanted
                    .entry(p.link.next.name().into())
                    .or_insert((p.link.next.clone(), 0))
                    .1 += 1;
            }
        }
        let (_, rs, rp) = self.agent.executor.reserved(usize::MAX);
        for (name, (kind, n)) in wanted {
            let held = match &kind {
                Production::Crop(_) => o.private.seeds.get(&name),
                _ => {
                    o.private.shed.get(&name)
                        + o.private
                            .inventories
                            .iter()
                            .map(|i| i.get(&name))
                            .sum::<i64>()
                }
            };
            let reserved = if matches!(kind, Production::Crop(_)) {
                rs.get(&name)
            } else {
                rp.get(&name)
            };
            let pending = self
                .agent
                .executor
                .projects
                .iter()
                .filter(|(site, p)| {
                    !self.active.contains_key(site) && !p.confirmed && p.production == kind
                })
                .count() as i64;
            let ordered = orders
                .iter()
                .filter(|v| {
                    v.get(1) == Some(&name) && matches!(v[0].as_str(), "BUY_SEED" | "BUY_ANIMAL")
                })
                .map(|v| v[2].parse::<i64>().unwrap_or(0))
                .sum::<i64>();
            let need = (n.max(reserved) + pending - held - ordered).max(0);
            let mut q = need.min((budget / cost(&kind)).floor() as i64);
            if matches!(kind, Production::Animal(_)) {
                let incoming = orders
                    .iter()
                    .filter(|v| matches!(v[0].as_str(), "BUY_ANIMAL" | "BUY_PRODUCT"))
                    .map(|v| v[2].parse::<i64>().unwrap_or(0))
                    .sum::<i64>();
                q = q.min((100 - o.private.shed.sum() - incoming).max(0));
            }
            if q > 0 && orders.len() < 6 {
                orders.push(vec![
                    if matches!(kind, Production::Crop(_)) {
                        "BUY_SEED".into()
                    } else {
                        "BUY_ANIMAL".into()
                    },
                    name.clone(),
                    q.to_string(),
                ]);
                spend += q as f64 * cost(&kind);
                budget = (o.farm.money - self.agent.config.cash_reserve - spend).max(0.);
            } else if need > 0 {
                for id in self.active.values() {
                    if self.progress[*id].link.next == kind {
                        self.progress[*id].waiting_material_steps += 1;
                    }
                }
            }
        }
        orders
    }
    pub fn proposals(&self, o: &Observation) -> Vec<Choice> {
        let mut base_agent = self.agent.clone();
        // Upcoming successor material is budgeted before optional NEW expansion.
        // Mandatory servicing can still spend actual cash; there is no fictitious credit.
        let upcoming: f64 = self
            .active
            .values()
            .filter_map(|id| {
                let p = &self.progress[*id];
                if !p.first_started || p.successor_started {
                    return None;
                }
                let near = match tile(&o.farm, p.link.site) {
                    Cell::Plant {
                        crop, planted_day, ..
                    } => {
                        o.day() - planted_day >= rules::crop(crop).unwrap().max_yield_day - 1
                            && p.first_harvests + 1 >= p.link.cycles
                    }
                    Cell::Structure {
                        animal: Some(_), ..
                    } => p.first_harvests >= p.link.cycles,
                    _ => true,
                };
                near.then(|| cost(&p.link.next))
            })
            .sum();
        if !self.event_mode() {
            base_agent.config.cash_reserve += upcoming;
        } else {
            let ledger = super::plan_resources::Schedule::build(self, o);
            base_agent.config.cash_reserve = ledger.cash_floor + ledger.material_cash;
            base_agent.supply_sites = self
                .active
                .iter()
                .filter(|(_, id)| {
                    let p = &self.progress[**id];
                    p.first_harvests + 1 >= p.link.cycles
                        && !(actual(tile(&o.farm, p.link.site)).is_none()
                            && p.first_harvests < p.link.cycles)
                })
                .map(|(s, _)| *s)
                .collect();
        }
        // Active commitments cannot be overwritten by new rotations. Their projects
        // are already marked non-confirmed while waiting for successor placement.
        let raw = plan_learning::candidates(&base_agent, o);
        let maintenance = raw
            .iter()
            .find(|c| c.features[1] == 0. && c.features[5] == 0.)
            .cloned();
        let cache = PlanningCache::new(&base_agent, o);
        let mut out = Vec::new();
        for c in raw {
            if self.active.keys().any(|site| {
                c.next.executor.projects.get(site).map(|p| &p.production)
                    != self
                        .agent
                        .executor
                        .projects
                        .get(site)
                        .map(|p| &p.production)
            }) {
                continue;
            }
            let changes: Vec<_> = c
                .next
                .executor
                .projects
                .iter()
                .filter(|(site, p)| {
                    self.agent
                        .executor
                        .projects
                        .get(site)
                        .is_none_or(|a| a.production != p.production)
                })
                .map(|(site, p)| (*site, p.production.clone()))
                .collect();
            let f = features(self, o, Some(&c), &[], &cache);
            out.push(Choice {
                base: Some(c.clone()),
                links: vec![],
                features: f,
            });
            // New short-crop batch -> any feasible successor; current crop pays for the next stage.
            if changes.len() > 0
                && changes
                    .iter()
                    .all(|(_, k)| matches!(k,Production::Crop(s) if s=="WHEAT"||s=="CARROT"))
                && changes.len() > 1
            {
                for next in kinds() {
                    if o.day() + duration(&changes[0].1) + duration(&next) + 2 >= 30 {
                        continue;
                    }
                    let links = changes
                        .iter()
                        .map(|(site, first)| Link {
                            site: *site,
                            first: first.clone(),
                            next: next.clone(),
                            cycles: 1,
                        })
                        .collect::<Vec<_>>();
                    out.push(Choice {
                        features: features(self, o, Some(&c), &links, &cache),
                        base: Some(c.clone()),
                        links,
                    });
                }
            }
        }
        // Same local cohort for every successor alternative: a controlled change
        // in the production connection, not a simultaneous global config mutation.
        let reserved = self.agent.executor.reserved(usize::MAX).0;
        let mut available: Vec<_> = self
            .agent
            .executor
            .projects
            .iter()
            .filter_map(|(site, p)| {
                if self.active.contains_key(site) || reserved.contains(site) || !p.confirmed {
                    return None;
                }
                let (kind, start) = actual(tile(&o.farm, *site))?;
                Some((*site, kind, start))
            })
            .collect();
        available.sort_by_key(|(site, k, start)| {
            (start + duration(k), distance(*site, home(*site)), *site)
        });
        if let Some((_, first, _)) = available.first() {
            let cohort: Vec<_> = available
                .iter()
                .filter(|(_, k, _)| k == first)
                .take(4)
                .collect();
            for cycles in [1, 2] {
                for next in kinds() {
                    if &next == first
                        && matches!(first,Production::Crop(c) if rules::crop(c).unwrap().ongoing)
                    {
                        continue;
                    }
                    if o.day() + duration(&next) + 2 >= 30 {
                        continue;
                    }
                    let links = cohort
                        .iter()
                        .map(|(site, k, _)| Link {
                            site: *site,
                            first: k.clone(),
                            next: next.clone(),
                            cycles,
                        })
                        .collect::<Vec<_>>();
                    if links
                        .iter()
                        .any(|l| o.day() + remaining_first(o, l) + duration(&l.next) + 2 >= 30)
                    {
                        continue;
                    }
                    out.push(Choice {
                        features: features(self, o, maintenance.as_ref(), &links, &cache),
                        base: maintenance.clone(),
                        links,
                    });
                }
            }
        }
        // Default remains first only as an exact tie-break at zero initialization;
        // no probability/logit prior restricts subsequent deterministic selection.
        out.truncate(120);
        out
    }
    pub fn sample(&self, o: &Observation, cs: &[Choice]) -> Sample {
        let dummy = plan_learning::sample(&self.agent, o, &[]);
        Sample {
            features: cs.iter().map(|c| c.features.clone()).collect(),
            ..dummy
        }
    }
    pub fn choose(&self, o: &Observation, cs: &[Choice], p: &Policy) -> Result<usize, String> {
        if cs.len() == 1 {
            return Ok(0);
        }
        Ok(p.infer(&[self.sample(o, cs)], true, &mut Rng(0))?[0].action)
    }
    pub fn execute_choice(&mut self, o: &Observation, c: Choice) -> PlayerAction {
        self.decisions += 1;
        self.agent.supply_sites.clear();
        if self.event_mode() {
            self.agent.supply_sites = self
                .active
                .iter()
                .filter(|(_, id)| {
                    let p = &self.progress[**id];
                    p.first_harvests + 1 >= p.link.cycles
                        && !(actual(tile(&o.farm, p.link.site)).is_none()
                            && p.first_harvests < p.link.cycles)
                })
                .map(|(s, _)| *s)
                .collect();
        }
        let mut orders = if let Some(b) = c.base {
            self.agent.commit_plan(b.next);
            b.orders
        } else {
            self.agent.plan_batch(o, None, 0, false)
        };
        for link in c.links {
            if self.active.contains_key(&link.site) {
                continue;
            }
            let a = actual(tile(&o.farm, link.site));
            let id = self.progress.len();
            self.active.insert(link.site, id);
            self.progress.push(Progress {
                first_started: a.as_ref().is_some_and(|(k, _)| *k == link.first),
                first_marker: a.map(|(_, d)| d).unwrap_or(-1),
                expected_ready: super::plan_resources::ready_step(o, &link, 0),
                link,
                requested: o.step,
                first_harvests: 0,
                armed: false,
                retiring: false,
                successor_started: false,
                successor_yielded: false,
                failed: false,
                superseded: false,
                waiting_material_steps: 0,
            });
        }
        orders = self.supplies(o, orders);
        self.execute(o, orders)
    }
    fn execute(&mut self, o: &Observation, orders: Vec<Vec<String>>) -> PlayerAction {
        self.agent.executor.service_deadlines.clear();
        self.agent.executor.harvest_successors.clear();
        self.agent.executor.harvest_successor_deadlines.clear();
        self.agent.executor.plan_wheat_reserve = 0;
        if self.event_mode() {
            let ledger = super::plan_resources::Schedule::build(self, o);
            self.agent.executor.plan_wheat_reserve = ledger.feed_keep;
            let mut allocated_work = 0.;
            for n in &ledger.needs {
                let p = &self.progress[n.id];
                let routed = self.batches.iter().any(|b| {
                    !b.cancelled && b.stage.route_handoff && b.stage.links.contains(&n.id)
                });
                let harvestable = matches!(tile(&o.farm,n.site),Cell::Plant{crop,planted_day,yield_units,..}
                    if *yield_units>0 && o.day()-planted_day>=rules::crop(crop).unwrap().first_yield_day);
                if routed
                    && p.first_harvests + 1 >= p.link.cycles
                    && harvestable
                    && n.stocked
                    && n.feed_stocked
                    && o.farm.money >= n.cash_floor
                    && o.step <= n.deadline
                    && allocated_work + n.work <= ledger.free_work
                {
                    self.agent
                        .executor
                        .harvest_successors
                        .insert(n.site, n.production.clone());
                    self.agent
                        .executor
                        .harvest_successor_deadlines
                        .insert(n.site, n.deadline);
                    allocated_work += n.work;
                }
            }
            for (&site, &id) in &self.active {
                let p = &self.progress[id];
                if !p.failed && !p.successor_yielded {
                    self.agent.executor.service_deadlines.insert(
                        site,
                        super::plan_resources::ready_step(o, &p.link, p.first_harvests) + 24,
                    );
                }
            }
        }
        let a = self.agent.execute(o, orders);
        for (actor, cmd) in std::iter::once(&a.farmer).chain(a.hands.iter()).enumerate() {
            if cmd.op == "HARVEST" {
                let site = pos(&o.farm, actor);
                if let Some(id) = self.active.get(&site) {
                    let p = &self.progress[*id];
                    let k = if p.successor_started {
                        &p.link.next
                    } else {
                        &p.link.first
                    };
                    self.receipts.push(Receipt {
                        site,
                        actor,
                        item: product(k).into(),
                        before: o.private.inventories[actor].get(product(k)),
                        successor: p.successor_started,
                    });
                }
            }
        }
        a
    }
    /// Continue already committed work after the caller reconciled this observation.
    pub fn continue_action(&mut self, o: &Observation) -> PlayerAction {
        self.execute(o, vec![])
    }
    /// Caller observes exactly once. Forks are taken after observe, before select/action.
    pub fn act_observed(&mut self, o: &Observation, p: &Policy) -> Result<PlayerAction, String> {
        if Agent::planning_due(o) {
            let mut cs = self.proposals(o);
            let i = self.choose(o, &cs, p)?;
            Ok(self.execute_choice(o, cs.swap_remove(i)))
        } else {
            Ok(self.execute(o, vec![]))
        }
    }
    pub fn action(&mut self, o: &Observation, p: &Policy) -> Result<PlayerAction, String> {
        self.observe(o);
        self.act_observed(o, p)
    }
    pub fn report(&self) -> Json {
        let mut j = self.agent.report();
        j.set_path("batch_revisions", Json::Num(self.batch_revisions as f64));
        j.set_path(
            "batch_cancellations",
            Json::Num(self.batch_cancellations as f64),
        );
        j.set_path(
            "batch_plans",
            Json::Arr(self.batches.iter().map(|b| b.json(self)).collect()),
        );
        for (k, n) in [
            ("plan_decisions", self.decisions),
            ("links_requested", self.progress.len()),
            (
                "links_successor_started",
                self.progress.iter().filter(|p| p.successor_started).count(),
            ),
            (
                "links_successor_yielded",
                self.progress.iter().filter(|p| p.successor_yielded).count(),
            ),
            (
                "links_failed",
                self.progress
                    .iter()
                    .filter(|p| p.failed && !p.superseded)
                    .count(),
            ),
        ] {
            j.set_path(k, Json::Num(n as f64));
        }
        j
    }
}
pub(crate) fn order_cost(o: &Observation, orders: &[Vec<String>]) -> f64 {
    let mut hire = o.farm.hires_today as u32;
    orders
        .iter()
        .map(|v| {
            let n = v.get(2).and_then(|s| s.parse::<i64>().ok()).unwrap_or(1);
            let name = v.get(1).map(String::as_str).unwrap_or("");
            match v[0].as_str() {
                "BUY_SEED" => n as f64 * rules::crop(name).unwrap().seed_cost as f64,
                "BUY_ANIMAL" => n as f64 * rules::animal(name).unwrap().cost as f64,
                "BUY_PRODUCT" => {
                    -super::trading::quote(name, o.market.inventory.get(name) - 10, -n).0
                }
                "HIRE" => {
                    let c = rules::hire_cost(hire, 1);
                    hire += 1;
                    c as f64
                }
                "BUY_LAND" => rules::next_land(o.farm.unlocked_quadrants.len() - 1)
                    .map(|(_, c)| c as f64)
                    .unwrap_or(0.),
                _ => 0.,
            }
        })
        .sum()
}
fn remaining_first(o: &Observation, l: &Link) -> i64 {
    match tile(&o.farm, l.site) {
        Cell::Plant {
            crop, planted_day, ..
        } => {
            let c = rules::crop(crop).unwrap();
            (c.max_yield_day - (o.day() - planted_day)).max(0)
                + (l.cycles as i64 - 1)
                    * if c.ongoing {
                        c.interval.max(1)
                    } else {
                        c.max_yield_day + 1
                    }
        }
        Cell::Structure {
            animal: Some(a), ..
        } => {
            let spec = rules::animal(&a.animal).unwrap();
            (spec.first_yield_day - (o.day() - a.placed_day)).max(0)
                + (l.cycles as i64 - 1) * spec.interval
                + 2
        }
        _ => duration(&l.first),
    }
}
fn features(
    c: &Controller,
    o: &Observation,
    b: Option<&plan_learning::Candidate>,
    links: &[Link],
    cache: &PlanningCache,
) -> Vec<f32> {
    let mut f = vec![0.; 32];
    let changes: Vec<_> = b
        .map(|b| {
            b.next
                .executor
                .projects
                .iter()
                .filter(|(s, p)| {
                    c.agent
                        .executor
                        .projects
                        .get(s)
                        .is_none_or(|v| v.production != p.production)
                })
                .map(|(s, p)| (*s, p.production.clone()))
                .collect()
        })
        .unwrap_or_default();
    f[0] = f32::from(changes.is_empty() && links.is_empty());
    f[1] = changes.len().max(links.len()) as f32 / 4.;
    let spend = b.map(|b| order_cost(o, &b.orders)).unwrap_or(0.);
    f[2] = spend as f32 / 10000.;
    f[3] = (o.farm.money - spend) as f32 / 10000.;
    if let Some(b) = b {
        f[4] = b.features[4];
        f[5] = b.features[5];
    }
    for (site, k) in &changes {
        f[6 + index(k)] += 0.25;
        f[26] += distance(*site, home(*site)) as f32 / 40.;
    }
    for l in links {
        if changes.is_empty() {
            f[6 + index(&l.first)] += 0.25;
            f[26] += distance(l.site, home(l.site)) as f32 / 40.;
        }
        f[14 + index(&l.next)] += 0.25;
        f[22] += (remaining_first(o, l) + duration(&l.next)) as f32 / 120.;
        f[23] += l.cycles as f32 / 8.;
        f[24] += f32::from(l.next != l.first) / 4.;
        f[25] += if matches!(l.next, Production::Animal(_)) {
            0.5
        } else {
            0.25
        };
        f[27] += cost(&l.next) as f32 / 10000.;
        f[29] += cache.price(product(&l.next)) as f32 / 8000.;
    }
    f[28] = c.active.len() as f32 / 40.;
    f[30] = 0.;
    f[31] = 1.;
    f
}

#[cfg(test)]
mod tests {
    use super::*;
    use kagg_engine::{engine, state::State};
    fn wheat_state(step: i64) -> State {
        let mut s = State::new(91);
        s.step = step;
        s.farms[0].money = 2000.;
        s.farms[0].farmer = (4, 4);
        s.farms[0].tiles[4][3] = Cell::Plant {
            crop: "WHEAT".into(),
            planted_day: 0,
            watered_today: true,
            consecutive_unwatered: 0,
            yield_units: 4,
            max_lifespan_step: 144,
            fertilized_until_day: -1,
        };
        s
    }
    fn link_choice() -> Choice {
        Choice {
            base: None,
            links: vec![Link {
                site: (3, 4),
                first: Production::Crop("WHEAT".into()),
                next: Production::Crop("CARROT".into()),
                cycles: 1,
            }],
            features: vec![0.; 32],
        }
    }
    #[test]
    fn crop_chain_preserves_immature_crop_and_uses_observed_stock() {
        let mut s = wheat_state(24);
        let mut c = Controller::new(Config::default());
        let o = Observation::from_state(&s, 0);
        c.observe(&o);
        c.execute_choice(&o, link_choice());
        c.agent.executor.routes.clear();
        c.receipts.clear();
        s.private[0].seeds.add("CARROT", 1);
        c.observe(&Observation::from_state(&s, 0));
        assert_eq!(
            c.agent.executor.projects[&(3, 4)].production,
            Production::Crop("WHEAT".into())
        );
        s.step = 96;
        s.private[0].seeds.sub("CARROT", 1);
        c.observe(&Observation::from_state(&s, 0));
        assert!(!c.progress[0].armed);
        s.private[0].seeds.add("CARROT", 1);
        c.observe(&Observation::from_state(&s, 0));
        assert!(c.progress[0].armed);
        assert_eq!(
            c.agent.executor.projects[&(3, 4)].production,
            Production::Crop("CARROT".into())
        );
    }
    #[test]
    fn chain_executes_harvest_replant_and_successor_yield_in_real_engine() {
        let mut s = wheat_state(96);
        s.private[0].seeds.add("CARROT", 1);
        // Preserve one walkable farm quadrant; prevent optional expansion during this contract test.
        let mut c = Controller::new(Config::default());
        let o = Observation::from_state(&s, 0);
        c.observe(&o);
        let a = c.execute_choice(&o, link_choice());
        engine::step(&mut s, &[a, Default::default()]);
        while s.step < 220 {
            let o = Observation::from_state(&s, 0);
            c.observe(&o);
            let orders = c.agent.plan_batch(&o, None, 0, false);
            let orders = c.supplies(&o, orders);
            let a = c.execute(&o, orders);
            engine::step(&mut s, &[a, Default::default()]);
        }
        c.observe(&Observation::from_state(&s, 0));
        assert!(
            c.progress[0].successor_started,
            "{}",
            c.progress[0].json().dump()
        );
        assert!(
            c.progress[0].successor_yielded,
            "{}",
            c.progress[0].json().dump()
        );
        assert_eq!(c.agent.executor.stats.receipt_failures, 0);
    }
    #[test]
    fn future_revenue_never_pays_current_order_and_proposals_ignore_hidden_state() {
        let mut s = wheat_state(72);
        s.farms[0].money = 0.;
        let mut c = Controller::new(Config::default());
        let o = Observation::from_state(&s, 0);
        c.observe(&o);
        let a = c.execute_choice(&o, link_choice());
        assert!(a.market.iter().all(|v| !v[0].starts_with("BUY")));
        assert_eq!(s.private[0].seeds.get("CARROT"), 0);
        let cs = c.proposals(&o);
        let row = c.sample(&o, &cs);
        assert!(cs.iter().all(|p| p.features[30] == 0.));
        s.private[1].shed.add("WHEAT", 9000);
        s.seed = 123456;
        let hidden = Observation::from_state(&s, 0);
        let cs2 = c.proposals(&hidden);
        assert_eq!(row.json(), c.sample(&hidden, &cs2).json());
    }
}

use kagg_engine::{
    engine::{self, PlayerAction, UnitAction},
    state::{Cell, Farm, Market, OMap, Private, State},
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
pub type Pos = (i64, i64);
pub fn unit(op: &str, item: &str, n: i64) -> UnitAction {
    UnitAction {
        op: op.into(),
        item: item.into(),
        n,
        has_n: !item.is_empty(),
    }
}
pub fn pos(f: &Farm, i: usize) -> Pos {
    if i == 0 {
        f.farmer
    } else {
        f.hands[i - 1]
    }
}
pub fn distance(a: Pos, b: Pos) -> i64 {
    (a.0 - b.0).abs() + (a.1 - b.1).abs()
}
pub fn home(p: Pos) -> Pos {
    *[(4, 4), (5, 4), (4, 5), (5, 5)]
        .iter()
        .min_by_key(|q| distance(p, **q))
        .unwrap()
}
pub fn tile(f: &Farm, p: Pos) -> &Cell {
    &f.tiles[p.1 as usize][p.0 as usize]
}
pub fn movement(p: Pos, q: Pos) -> &'static str {
    if p.0 < q.0 {
        "EAST"
    } else if p.0 > q.0 {
        "WEST"
    } else if p.1 < q.1 {
        "SOUTH"
    } else if p.1 > q.1 {
        "NORTH"
    } else {
        "PASS"
    }
}
#[derive(Clone)]
pub struct Observation {
    pub step: i64,
    pub seat: usize,
    pub farm: Farm,
    pub private: Private,
    pub market: Market,
    pub rival: Farm,
    pub shops: Vec<String>,
}
impl Observation {
    // This is the only full-state boundary. Policies cannot access rival private state or RNG seed.
    pub fn from_state(s: &State, seat: usize) -> Self {
        Self {
            step: s.step,
            seat,
            farm: s.farms[seat].clone(),
            private: s.private[seat].clone(),
            market: s.market.clone(),
            rival: s.farms[1 - seat].clone(),
            shops: s.town.unlocked_shops.clone(),
        }
    }
    pub fn day(&self) -> i64 {
        self.step / 24
    }
    pub fn end(&self) -> i64 {
        ((self.day() + 1) * 24 - 1).min(718)
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Production {
    Crop(String),
    Animal(String),
    Vacant,
}
impl Production {
    pub fn name(&self) -> &str {
        match self {
            Self::Crop(s) | Self::Animal(s) => s,
            Self::Vacant => "",
        }
    }
}
#[derive(Clone, Debug)]
pub struct Project {
    pub production: Production,
    pub requested: i64,
    pub confirmed: bool,
    pub failures: u32,
}
#[derive(Clone, Debug)]
pub struct Scheduled {
    pub at: i64,
    pub position: Pos,
    pub action: UnitAction,
}
#[derive(Clone, Debug, Default)]
pub struct Route {
    pub steps: VecDeque<Scheduled>,
    pub sites: BTreeSet<Pos>,
    pub seeds: OMap,
    pub pickups: OMap,
    pub walking: usize,
    pub work: usize,
    pub crop_jobs: usize,
    pub animal_jobs: usize,
    pub reused_fertilizer: usize,
    pub harvested: i64,
    pub replants: usize,
}
impl Route {
    pub fn remaining(&self) -> usize {
        self.steps.len()
    }
}
#[derive(Clone)]
struct Receipt {
    actor: usize,
    position: Pos,
    inventory: OMap,
    tile_position: Pos,
    tile: Option<Cell>,
    at: i64,
}
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub routes: u64,
    pub mixed_routes: u64,
    pub material_routes: u64,
    pub replants: u64,
    pub walking: u64,
    pub work: u64,
    pub idle: u64,
    pub invalidated: u64,
    pub receipt_failures: u64,
    pub completed: u64,
    pub investments: u64,
    pub projects_requested: u64,
    pub projects_started: u64,
    pub expired_projects: u64,
    pub harvested_units: i64,
    pub sold_units: i64,
    pub fertilizer_used: u64,
    pub trade_sessions: u64,
    pub trade_events: [u64; super::trading::EVENT_COUNT],
    pub trade_decisions: u64,
    pub trade_holds: u64,
    pub sell_orders: u64,
    pub buy_orders: u64,
}
#[derive(Clone, Default)]
pub struct Executor {
    pub projects: BTreeMap<Pos, Project>,
    pub routes: Vec<Option<Route>>,
    pub stats: Stats,
    pub day: i64,
    pub last_observed: Option<i64>,
    receipts: Vec<Receipt>,
    pub last_market: i64,
    pub last_cash: f64,
    pub expansion_budget: f64,
    pub expansion_spent: f64,
    pub new_projects_today: usize,
    pub market_mode: super::trading::MarketMode,
    pub market_history: VecDeque<super::trading::MarketPoint>,
    pub rival_supply: super::public_supply::History,
    pub trade_anchor: Option<super::trading::MarketAnchor>,
    pub trade_events: [bool; super::trading::EVENT_COUNT],
    pub trade_remaining: [bool; 9],
    pub trade_slots: usize,
    pub trade_reserved_cash: f64,
    pub trade_start_cash: f64,
    pub trade_reserved_space: i64,
}
impl Executor {
    pub fn new() -> Self {
        Self {
            day: -1,
            last_market: -10,
            ..Self::default()
        }
    }
    pub fn observe(&mut self, o: &Observation) {
        if self.last_observed == Some(o.step) {
            return;
        }
        self.last_observed = Some(o.step);
        self.rival_supply.observe(&o.rival, o.step);
        self.trade_events = [false; super::trading::EVENT_COUNT];
        self.trade_remaining = [false; 9];
        self.trade_slots = 0;
        self.trade_reserved_cash = 0.;
        self.trade_reserved_space = 0;
        self.trade_start_cash = o.farm.money;
        self.market_history.push_back(super::trading::MarketPoint {
            step: o.step,
            prices: o.market.prices.clone(),
            inventory: o.market.inventory.clone(),
        });
        while self
            .market_history
            .front()
            .is_some_and(|p| p.step < o.step - 24)
        {
            self.market_history.pop_front();
        }
        for r in self.receipts.drain(..) {
            if r.at % 24 == 23 {
                continue;
            }
            if r.actor >= o.private.inventories.len() {
                continue;
            }
            if pos(&o.farm, r.actor) != r.position
                || o.private.inventories[r.actor] != r.inventory
                || r.tile
                    .as_ref()
                    .is_some_and(|expected| tile(&o.farm, r.tile_position) != expected)
            {
                self.stats.receipt_failures += 1;
                if let Some(x) = self.routes.get_mut(r.actor) {
                    *x = None;
                }
            }
        }
        if self.day != o.day() {
            self.day = o.day();
            self.routes.clear();
            self.expansion_budget = o.farm.money.max(0.) * 0.25;
            self.expansion_spent = 0.;
            self.new_projects_today = 0;
        }
        self.routes.resize_with(o.farm.hands.len() + 1, || None);
        // Discover actual production; approved projects preserve their desired successor.
        for y in 0..10 {
            for x in 0..10 {
                let p = (x, y);
                let actual = match tile(&o.farm, p) {
                    Cell::Plant { crop, .. } => Some(Production::Crop(crop.clone())),
                    Cell::Structure {
                        animal: Some(a), ..
                    } => Some(Production::Animal(a.animal.clone())),
                    _ => None,
                };
                if let Some(actual) = actual {
                    let entry = self.projects.entry(p).or_insert(Project {
                        production: actual.clone(),
                        requested: o.step,
                        confirmed: true,
                        failures: 0,
                    });
                    if entry.production == actual {
                        if !entry.confirmed {
                            self.stats.projects_started += 1;
                        }
                        entry.confirmed = true;
                    }
                }
            }
        }
        // A failed purchase cannot occupy a plot indefinitely. No invented receipt.
        self.projects.retain(|p, project| {
            let stocked = match &project.production {
                Production::Crop(c) => o.private.seeds.get(c) > 0,
                Production::Animal(a) => {
                    o.private.shed.get(a) > 0 || o.private.inventories.iter().any(|i| i.get(a) > 0)
                }
                Production::Vacant => true,
            };
            let expired = !project.confirmed
                && !stocked
                && o.step - project.requested > 24
                && matches!(
                    tile(&o.farm, *p),
                    Cell::Empty | Cell::Weed | Cell::Structure { animal: None, .. }
                );
            if expired {
                self.stats.expired_projects += 1;
            }
            !expired
        });
        for r in &mut self.routes {
            if r.as_ref().is_some_and(|r| r.steps.is_empty()) {
                *r = None;
                self.stats.completed += 1;
            }
        }
    }
    pub fn reserved(&self, except: usize) -> (BTreeSet<Pos>, OMap, OMap) {
        let mut sites = BTreeSet::new();
        let mut seeds = OMap::default();
        let mut pickups = OMap::default();
        for (i, r) in self.routes.iter().enumerate() {
            if i == except {
                continue;
            }
            if let Some(r) = r {
                sites.extend(r.sites.iter().copied());
                for s in &r.steps {
                    let a = &s.action;
                    if a.op == "PLANT" {
                        seeds.add(&a.item, 1);
                    }
                    if a.op == "PICKUP" {
                        pickups.add(&a.item, a.n);
                    }
                }
            }
        }
        (sites, seeds, pickups)
    }
    pub fn assign(&mut self, actor: usize, route: Route) {
        self.routes.resize_with(actor + 1, || None);
        self.stats.routes += 1;
        self.stats.mixed_routes += u64::from(route.crop_jobs > 0 && route.animal_jobs > 0);
        self.stats.material_routes += u64::from(route.reused_fertilizer > 0);
        self.routes[actor] = Some(route);
    }
    pub fn action(&mut self, o: &Observation, market: Vec<Vec<String>>) -> PlayerAction {
        self.project_action(o, market).0
    }
    /// Project own legal unit actions once, before simultaneous market settlement.
    pub fn project_action(
        &mut self,
        o: &Observation,
        market: Vec<Vec<String>>,
    ) -> (PlayerAction, Observation) {
        let mut farm = o.farm.clone();
        let mut private = o.private.clone();
        let mut units = Vec::new();
        let mut receipts = Vec::new();
        for actor in 0..o.private.inventories.len() {
            let mut cmd = unit("PASS", "", 0);
            if let Some(r) = self.routes.get_mut(actor).and_then(Option::as_mut) {
                if let Some(next) = r.steps.front() {
                    if next.at == o.step && next.position == pos(&farm, actor) {
                        cmd = next.action.clone();
                    } else {
                        r.steps.clear();
                        self.stats.invalidated += 1;
                    }
                }
            }
            // DROP discards overflow in the engine. Recheck against all earlier workers now.
            if cmd.op == "DROP" && private.inventories[actor].sum() + private.shed.sum() > 100 {
                self.routes[actor] = None;
                let room = (100 - private.shed.sum()).max(0);
                cmd = private.inventories[actor]
                    .0
                    .iter()
                    .find(|(_, q)| *q > 0 && room > 0)
                    .map(|(item, q)| unit("PLACE", item, (*q).min(room)))
                    .unwrap_or_else(|| unit("PASS", "", 0));
            }
            let before_pos = pos(&farm, actor);
            let before_farm = farm.clone();
            let before_private = private.clone();
            engine::apply_unit_action(&mut farm, &mut private, actor, &cmd, o.day());
            let changed = farm != before_farm || private != before_private;
            if cmd.op != "PASS" && !changed {
                self.stats.invalidated += 1;
                self.routes[actor] = None;
                cmd = unit("PASS", "", 0);
            } else if let Some(r) = self.routes.get_mut(actor).and_then(Option::as_mut) {
                r.steps.pop_front();
            }
            match cmd.op.as_str() {
                "NORTH" | "SOUTH" | "EAST" | "WEST" => self.stats.walking += 1,
                "PASS" => self.stats.idle += 1,
                _ => self.stats.work += 1,
            }
            if cmd.op == "HARVEST" {
                self.stats.harvested_units += (private.inventories[actor].sum()
                    - before_private.inventories[actor].sum())
                .max(0);
            }
            if cmd.op == "FERTILIZE" {
                self.stats.fertilizer_used += 1;
            }
            if cmd.op != "PASS" {
                receipts.push(Receipt {
                    actor,
                    position: pos(&farm, actor),
                    inventory: private.inventories[actor].clone(),
                    tile_position: before_pos,
                    tile: (tile(&farm, before_pos) != tile(&before_farm, before_pos))
                        .then(|| tile(&farm, before_pos).clone()),
                    at: o.step,
                });
            }
            units.push(cmd);
        }
        // Receipts use the final own-action projection (another unit can legitimately touch the same tile).
        let mut expected = farm.clone();
        engine::decay_plants(&mut expected, o.step);
        for r in &mut receipts {
            if r.tile.is_some() {
                r.tile = Some(tile(&expected, r.tile_position).clone());
            }
        }
        self.receipts = receipts;
        let mut orders = Vec::new();
        let animals = self
            .projects
            .values()
            .filter(|p| matches!(p.production, Production::Animal(_)))
            .count() as i64;
        for item in kagg_engine::state::PRODUCTS {
            if self.market_mode != super::trading::MarketMode::Rule {
                break;
            }
            let reserve = if o.step >= 694 {
                0
            } else if item == "WHEAT" {
                animals * 2
            } else if item == "FERTILIZER" {
                self.projects.len().min(8) as i64
            } else {
                0
            };
            let q = (private.shed.get(item) - reserve).max(0);
            if q > 0 {
                orders.push(vec!["SELL".into(), item.into(), q.to_string()]);
            }
        }
        // Purchasing proposals reserve their own order slots first; never truncate a selected investment.
        orders.truncate(10usize.saturating_sub(market.len()));
        orders.extend(market);
        let mut projected = o.clone();
        projected.farm = farm;
        projected.private = private;
        (
            PlayerAction {
                farmer: units.remove(0),
                hands: units,
                market: orders,
            },
            projected,
        )
    }
    pub fn labor_load(&self) -> usize {
        self.projects
            .values()
            .map(|p| match p.production {
                Production::Crop(_) => 2,
                Production::Animal(_) => 4,
                Production::Vacant => 0,
            })
            .sum()
    }
    pub fn market_due(&self, o: &Observation) -> bool {
        o.step - self.last_market >= 6 || o.step % 24 == 0 || o.farm.money - self.last_cash > 100.
    }
}
pub fn action_json(a: &PlayerAction) -> kagg_engine::json::Json {
    use kagg_engine::json::Json;
    let u = |u: &UnitAction| {
        let mut v = vec![Json::Str(u.op.clone())];
        if !u.item.is_empty() {
            v.push(Json::Str(u.item.clone()));
            if u.has_n {
                v.push(Json::Num(u.n as f64));
            }
        }
        Json::Arr(v)
    };
    Json::Obj(vec![
        ("farmer".into(), u(&a.farmer)),
        ("hands".into(), Json::Arr(a.hands.iter().map(u).collect())),
        (
            "market".into(),
            Json::Arr(
                a.market
                    .iter()
                    .map(|o| {
                        Json::Arr(
                            o.iter()
                                .enumerate()
                                .map(|(i, s)| {
                                    if i == 2 {
                                        Json::Num(s.parse::<f64>().unwrap_or(0.))
                                    } else {
                                        Json::Str(s.clone())
                                    }
                                })
                                .collect(),
                        )
                    })
                    .collect(),
            ),
        ),
    ])
}

//! Shared reservations for pending production. Forecasts never become spendable cash.
use super::{
    executor::*,
    plan_chain::{Controller, Link},
};
use kagg_engine::{json::Json, rules};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct Stage {
    pub conditional: bool,
    pub links: Vec<usize>,
    pub lead_steps: i64,
    pub cash_floor: f64,
    pub deadline: i64,
}
#[derive(Clone, Debug)]
pub struct BatchPlan {
    pub id: usize,
    pub revision: u64,
    pub sites: Vec<Pos>,
    pub stage: Stage,
    pub cancelled: bool,
}
impl BatchPlan {
    pub fn json(&self, c: &Controller) -> Json {
        Json::Obj(vec![
            ("id".into(), Json::Num(self.id as f64)),
            ("conditional".into(), Json::Bool(self.stage.conditional)),
            ("revision".into(), Json::Num(self.revision as f64)),
            ("lead_steps".into(), Json::Num(self.stage.lead_steps as f64)),
            ("cash_floor".into(), Json::Num(self.stage.cash_floor)),
            ("deadline".into(), Json::Num(self.stage.deadline as f64)),
            ("cancelled".into(), Json::Bool(self.cancelled)),
            (
                "stages".into(),
                Json::Arr(
                    self.stage
                        .links
                        .iter()
                        .map(|i| c.progress[*i].json())
                        .collect(),
                ),
            ),
        ])
    }
}
#[derive(Clone, Debug)]
pub struct Need {
    pub conditional: bool,
    pub feed_stocked: bool,
    pub cash_floor: f64,
    pub id: usize,
    pub site: Pos,
    pub production: Production,
    pub ready: i64,
    /// Anchored at acceptance; delayed forecasts must not move the deadline.
    pub service_due: i64,
    pub deadline: i64,
    pub lead: i64,
    pub cost: f64,
    pub stocked: bool,
    pub work: f64,
}
#[derive(Clone, Debug, Default)]
pub struct Schedule {
    pub needs: Vec<Need>,
    pub cash_floor: f64,
    pub material_cash: f64,
    pub free_cash: f64,
    pub work_due: f64,
    pub free_work: f64,
    pub feed_keep: i64,
    pub conditional_feed: i64,
    pub feed_cash: f64,
    pub outstanding_units: i64,
}
pub fn production_cost(p: &Production) -> f64 {
    match p {
        Production::Crop(k) => rules::crop(k).unwrap().seed_cost as f64,
        Production::Animal(k) => rules::animal(k).unwrap().cost as f64,
        _ => 0.,
    }
}
pub fn duration(p: &Production) -> i64 {
    match p {
        Production::Crop(k) => rules::crop(k).unwrap().max_yield_day,
        Production::Animal(k) => rules::animal(k).unwrap().first_yield_day,
        _ => 30,
    }
}
pub fn ready_step(o: &Observation, link: &Link, harvests: usize) -> i64 {
    let remaining = link.cycles.saturating_sub(harvests);
    if remaining == 0 {
        return o.step;
    }
    let next_yield = |first: i64, interval: i64, held: bool| {
        if first > o.step {
            first
        } else if held {
            o.step
        } else {
            first + ((o.step - first) / (interval.max(1) * 24) + 1) * interval.max(1) * 24
        }
    };
    match tile(&o.farm, link.site) {
        kagg_engine::state::Cell::Plant {
            crop,
            planted_day,
            yield_units,
            ..
        } => {
            let r = rules::crop(crop).unwrap();
            if r.ongoing {
                next_yield(
                    (*planted_day + r.first_yield_day) * 24,
                    r.interval,
                    *yield_units > 0,
                ) + remaining.saturating_sub(1) as i64 * r.interval.max(1) * 24
            } else {
                ((*planted_day + r.max_yield_day) * 24).max(o.step)
                    + remaining.saturating_sub(1) as i64 * (r.max_yield_day + 1) * 24
            }
        }
        kagg_engine::state::Cell::Structure {
            animal: Some(a), ..
        } => {
            let r = rules::animal(&a.animal).unwrap();
            next_yield(
                (a.placed_day + r.first_yield_day) * 24,
                r.interval,
                a.yield_units > 0,
            ) + (remaining.saturating_sub(1) as i64 * r.interval + 2) * 24
        }
        _ => match &link.first {
            // A harvested empty plot with another cycle still owed must first be
            // replanted. Its next harvest is not available at the current step.
            Production::Crop(k) => {
                o.step + remaining as i64 * (rules::crop(k).unwrap().max_yield_day + 1) * 24
            }
            Production::Animal(k) => {
                let r = rules::animal(k).unwrap();
                o.step
                    + (r.first_yield_day + remaining.saturating_sub(1) as i64 * r.interval + 2) * 24
            }
            _ => o.step,
        },
    }
}
/// The candidate generator and execution guard use exactly the same suffix timing.
pub fn revision_ready(c: &Controller, o: &Observation, site: Pos, cycles: usize) -> i64 {
    let empty = matches!(tile(&o.farm, site), kagg_engine::state::Cell::Empty);
    let first = match tile(&o.farm, site) {
        kagg_engine::state::Cell::Plant { crop, .. } => Production::Crop(crop.clone()),
        _ => c
            .pending_id(site)
            .map(|id| c.progress[id].link.first.clone())
            .unwrap_or(Production::Vacant),
    };
    ready_step(
        o,
        &Link {
            site,
            first: first.clone(),
            next: first,
            cycles,
        },
        usize::from(empty),
    )
}

impl Schedule {
    pub fn build(c: &Controller, o: &Observation) -> Self {
        let e = &c.agent.executor;
        let (_, rs, rp) = e.reserved(usize::MAX);
        let mut seed = o.private.seeds.clone();
        let mut shed = o.private.shed.clone();
        let mut seed_claims = rs;
        let mut shed_claims = rp;
        let mut placement_work = 0.;
        for (site, p) in &e.projects {
            if !p.confirmed
                && (!c.pending_at(*site)
                    || c.pending_id(*site).is_some_and(|id| c.progress[id].armed))
            {
                // Merge only the same site/kind placement, not aggregate counts.
                let routed = e.routes.iter().flatten().any(|r| {
                    r.steps.iter().any(|s| {
                        s.position == *site
                            && s.action.item == p.production.name()
                            && matches!(s.action.op.as_str(), "PLANT" | "PLACE")
                    })
                });
                if routed {
                    continue;
                }
                match &p.production {
                    Production::Crop(k) => seed_claims.add(k, 1),
                    Production::Animal(k) => shed_claims.add(k, 1),
                    _ => (),
                }
                placement_work += 2. * distance(*site, home(*site)) as f64
                    + if matches!(p.production, Production::Animal(_)) {
                        10.
                    } else {
                        5.
                    };
            }
        }
        for k in kagg_engine::state::CROP_NAMES {
            seed.add(k, -seed_claims.get(k));
        }
        for k in kagg_engine::state::ANIMAL_NAMES {
            shed.add(k, -shed_claims.get(k));
        }
        let animals = e
            .projects
            .values()
            .filter(|p| matches!(p.production, Production::Animal(_)))
            .count() as i64;
        let floor = c
            .batches
            .iter()
            .filter(|b| {
                !b.cancelled
                    && b.stage
                        .links
                        .iter()
                        .any(|i| !c.progress[*i].successor_started && !c.progress[*i].failed)
            })
            .map(|b| b.stage.cash_floor)
            .fold(c.agent.config.cash_reserve, f64::max);
        let feed_keep = if o.day() < 29 { animals * 2 } else { 0 };
        let feed_cash =
            (feed_keep - shed.get("WHEAT")).max(0) as f64 * o.market.prices.get("WHEAT") as f64;
        let mut out = Self {
            cash_floor: floor + feed_cash,
            feed_keep,
            ..Default::default()
        };
        let free = e
            .routes
            .iter()
            .filter_map(|r| r.as_ref())
            .map(|r| r.steps.len().min(48) as f64)
            .sum::<f64>();
        // Includes background care/harvest and route travel; still a conservative estimate.
        let background = e
            .projects
            .values()
            .map(|p| {
                if matches!(p.production, Production::Animal(_)) {
                    5.
                } else {
                    2.
                }
            })
            .sum::<f64>();
        out.free_work =
            ((o.farm.hands.len() + 1) as f64 * 48. - free - background - placement_work).max(0.);
        for (id, p) in c.progress.iter().enumerate() {
            if p.failed || p.successor_started || !p.first_started || p.armed || !c.is_active(id) {
                continue;
            }
            let meta = c
                .batches
                .iter()
                .find(|b| !b.cancelled && b.stage.links.contains(&id));
            let ready = ready_step(o, &p.link, p.first_harvests);
            let deadline = meta
                .map(|b| b.stage.deadline)
                .unwrap_or(718 - duration(&p.link.next) * 24);
            let lead = meta.map(|b| b.stage.lead_steps).unwrap_or(24);
            let travel = 2. * distance(p.link.site, home(p.link.site)) as f64;
            let service = if matches!(p.link.next, Production::Animal(_)) {
                10.
            } else {
                5.
            };
            out.needs.push(Need {
                conditional: meta.is_some_and(|b| b.stage.conditional),
                feed_stocked: true,
                cash_floor: floor,
                id,
                site: p.link.site,
                production: p.link.next.clone(),
                ready,
                service_due: p.expected_ready + 24,
                deadline,
                lead,
                cost: production_cost(&p.link.next),
                stocked: false,
                work: travel + service,
            });
        }
        out.needs
            .sort_by_key(|n| (n.ready - n.lead, n.deadline, n.id));
        // Feed for a pending conditional animal has its own reservation. Existing
        // animals and route pickups own their stock first; no future harvest is stock.
        let conditional_feed = out
            .needs
            .iter()
            .filter(|n| {
                n.conditional
                    && matches!(n.production, Production::Animal(_))
                    && o.step >= n.ready - n.lead
                    && o.step <= n.deadline
            })
            .count() as i64
            * 2;
        if conditional_feed > 0 {
            out.conditional_feed = conditional_feed;
            out.feed_keep += conditional_feed;
            let missing =
                (out.feed_keep + shed_claims.get("WHEAT") - o.private.shed.get("WHEAT")).max(0);
            out.feed_cash =
                -super::trading::quote("WHEAT", o.market.inventory.get("WHEAT") - 10, -missing).0;
            out.cash_floor = floor + out.feed_cash;
        }
        let mut free_feed =
            (o.private.shed.get("WHEAT") - shed_claims.get("WHEAT") - animals * 2).max(0);
        for n in &mut out.needs {
            if n.conditional && matches!(n.production, Production::Animal(_)) {
                n.feed_stocked = free_feed >= 2;
                free_feed = (free_feed - 2).max(0);
            }
            let pool = if matches!(n.production, Production::Crop(_)) {
                &mut seed
            } else {
                &mut shed
            };
            n.stocked = pool.get(n.production.name()) > 0;
            if n.stocked {
                pool.add(n.production.name(), -1);
            } else {
                out.outstanding_units += 1;
            }
            if n.ready - n.lead <= o.step {
                if !n.stocked {
                    out.material_cash += n.cost;
                }
                out.work_due += n.work;
            }
        }
        out.free_cash = (o.farm.money - out.cash_floor - out.material_cash).max(0.);
        out
    }
    pub fn allows_work(&self, extra: f64) -> bool {
        self.work_due + extra <= self.free_work
    }
    /// Allocate due obligations in the same order as orders; no future sale cash.
    pub fn can_fund(&self, ids: &[usize], money: f64, step: i64) -> bool {
        let mut budget = (money - self.cash_floor).max(0.);
        let mut found = false;
        let mut affordable = true;
        for n in &self.needs {
            let target = ids.contains(&n.id);
            found |= target;
            if target && n.conditional && money < self.cash_floor {
                affordable = false;
            }
            if n.stocked {
                continue;
            }
            let due = step >= n.ready - n.lead && step <= n.deadline;
            let funded = due && budget >= n.cost;
            if funded {
                budget -= n.cost;
            }
            if target && !funded {
                affordable = false;
            }
        }
        found && affordable
    }
    /// Purchased material remains pending until the next observation confirms it.
    pub fn orders(&self, o: &Observation, already: &[Vec<String>]) -> Vec<Vec<String>> {
        let mut budget =
            (o.farm.money - self.cash_floor - super::plan_chain::order_cost(o, already)).max(0.);
        let mut room = (100 - o.private.shed.sum()).max(0);
        let mut incoming: BTreeMap<String, i64> = BTreeMap::new();
        for a in already {
            if a.len() > 2 && a[0] == "BUY_PRODUCT" {
                room -= a[2].parse::<i64>().unwrap_or(0);
            }
            if a.len() > 2 && matches!(a[0].as_str(), "BUY_SEED" | "BUY_ANIMAL") {
                *incoming.entry(a[1].clone()).or_default() += a[2].parse::<i64>().unwrap_or(0);
                if a[0] == "BUY_ANIMAL" {
                    room -= a[2].parse::<i64>().unwrap_or(0);
                }
            }
        }
        let mut buys: BTreeMap<(String, String), i64> = BTreeMap::new();
        let mut feed_order = vec![];
        if self.conditional_feed > 0 && already.len() < 6 {
            let bought = already
                .iter()
                .filter(|a| a.len() > 2 && a[0] == "BUY_PRODUCT" && a[1] == "WHEAT")
                .map(|a| a[2].parse::<i64>().unwrap_or(0))
                .sum::<i64>();
            let target = (self.feed_keep - o.private.shed.get("WHEAT") - bought)
                .max(0)
                .min(room)
                .min((o.market.inventory.get("WHEAT") - 10).max(0));
            let available = (o.farm.money
                - (self.cash_floor - self.feed_cash)
                - super::plan_chain::order_cost(o, already))
            .max(0.)
            .min(self.feed_cash);
            for q in (1..=target).rev() {
                let cost =
                    -super::trading::quote("WHEAT", o.market.inventory.get("WHEAT") - 10, -q).0;
                if cost <= available {
                    feed_order.push(vec!["BUY_PRODUCT".into(), "WHEAT".into(), q.to_string()]);
                    room -= q;
                    break;
                }
            }
        }
        for need in &self.needs {
            if need.stocked || o.step < need.ready - need.lead || o.step > need.deadline {
                continue;
            }
            let available = incoming.entry(need.production.name().into()).or_default();
            if *available > 0 {
                *available -= 1;
                continue;
            }
            if budget < need.cost {
                continue;
            }
            let animal = matches!(need.production, Production::Animal(_));
            if animal && room <= 0 {
                continue;
            }
            let op = if animal { "BUY_ANIMAL" } else { "BUY_SEED" };
            let key = (op.into(), need.production.name().into());
            if !buys.contains_key(&key) && buys.len() + already.len() + feed_order.len() >= 6 {
                continue;
            }
            *buys.entry(key).or_default() += 1;
            budget -= need.cost;
            if animal {
                room -= 1;
            }
        }
        feed_order.extend(
            buys.into_iter()
                .map(|((op, k), n)| vec![op, k, n.to_string()]),
        );
        feed_order
    }
    pub fn json(&self) -> Json {
        Json::Obj(vec![
            ("cash_floor".into(), Json::Num(self.cash_floor)),
            ("material_cash".into(), Json::Num(self.material_cash)),
            ("free_cash".into(), Json::Num(self.free_cash)),
            ("work_due".into(), Json::Num(self.work_due)),
            ("free_work".into(), Json::Num(self.free_work)),
            ("unstocked".into(), Json::Num(self.outstanding_units as f64)),
        ])
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::{plan_events, plan_prototype::Config};
    use super::*;
    use kagg_engine::{
        engine,
        state::{Cell, State},
    };
    pub fn fixture(step: i64) -> (State, Controller) {
        let mut s = State::new(91);
        s.step = step;
        s.farms[0].money = 2000.;
        s.farms[0].farmer = (4, 4);
        for x in [2, 3] {
            s.farms[0].tiles[4][x] = Cell::Plant {
                crop: "WHEAT".into(),
                planted_day: 0,
                watered_today: true,
                consecutive_unwatered: 0,
                yield_units: 4,
                max_lifespan_step: 144,
                fertilized_until_day: -1,
            };
        }
        let mut c = Controller::new(Config::default());
        c.observe(&Observation::from_state(&s, 0));
        (s, c)
    }
    #[test]
    fn reservations_allocate_stock_once_and_do_not_spend_future_sales() {
        let (mut s, mut c) = fixture(72);
        let o = Observation::from_state(&s, 0);
        for site in [(2, 4), (3, 4)] {
            c.revise_batch(
                &o,
                &[site],
                Some(Production::Crop("CARROT".into())),
                1,
                24,
                180.,
            )
            .unwrap();
        }
        s.private[0].seeds.add("CARROT", 1);
        s.farms[0].money = 0.;
        let o = Observation::from_state(&s, 0);
        let ledger = Schedule::build(&c, &o);
        assert_eq!(ledger.needs.iter().filter(|n| n.stocked).count(), 1);
        assert!(ledger.orders(&o, &[]).is_empty());
        s.farms[0].money = 2000.;
        let o = Observation::from_state(&s, 0);
        let orders = Schedule::build(&c, &o).orders(&o, &[]);
        assert_eq!(
            orders
                .iter()
                .filter(|a| a[1] == "CARROT")
                .map(|a| a[2].parse::<i64>().unwrap())
                .sum::<i64>(),
            1
        );
        assert_eq!(
            s.private[0].seeds.get("CARROT"),
            1,
            "issuing an order must not manufacture inventory"
        );
    }
    #[test]
    fn revise_and_cancel_preserve_real_crop_inventory_and_other_commitments() {
        let (mut s, mut c) = fixture(48);
        s.private[0].seeds.add("CARROT", 2);
        let o = Observation::from_state(&s, 0);
        c.revise_batch(
            &o,
            &[(2, 4), (3, 4)],
            Some(Production::Crop("CARROT".into())),
            1,
            24,
            180.,
        )
        .unwrap();
        c.revise_batch(
            &o,
            &[(2, 4)],
            Some(Production::Animal("COW".into())),
            1,
            48,
            430.,
        )
        .unwrap();
        assert_eq!(
            c.progress[c.pending_id((3, 4)).unwrap()].link.next.name(),
            "CARROT"
        );
        assert_eq!(
            c.progress[c.pending_id((2, 4)).unwrap()].link.next.name(),
            "COW"
        );
        c.revise_batch(&o, &[(2, 4)], None, 1, 0, 0.).unwrap();
        assert!(!c.pending_at((2, 4)));
        assert!(c.pending_at((3, 4)));
        assert_eq!(s.private[0].seeds.get("CARROT"), 2);
        assert!(matches!(tile(&o.farm,(2,4)),Cell::Plant{crop,..} if crop=="WHEAT"));
    }
    #[test]
    fn actual_harvest_uses_prepared_stock_and_successor_yields() {
        let (mut s, mut c) = fixture(96);
        s.private[0].seeds.add("CARROT", 1);
        let o = Observation::from_state(&s, 0);
        c.revise_batch(
            &o,
            &[(3, 4)],
            Some(Production::Crop("CARROT".into())),
            1,
            24,
            180.,
        )
        .unwrap();
        while s.step < 240 {
            let o = Observation::from_state(&s, 0);
            c.observe(&o);
            // No optional expansion; use the real maintenance/purchase/executor path.
            let choice = super::super::plan_chain::Choice {
                base: None,
                links: vec![],
                features: vec![],
            };
            let a = if super::super::plan_prototype::Agent::planning_due(&o) {
                c.execute_choice(&o, choice)
            } else {
                c.continue_action(&o)
            };
            engine::step(&mut s, &[a, Default::default()]);
        }
        c.observe(&Observation::from_state(&s, 0));
        assert!(c.progress[0].successor_started, "{}", c.report().dump());
        assert!(c.progress[0].successor_yielded, "{}", c.report().dump());
    }
    #[test]
    fn plan_commitments_are_visible_to_network() {
        let (s, mut c) = fixture(48);
        let o = Observation::from_state(&s, 0);
        let event = plan_events::Event::capture(
            "test".into(),
            plan_events::EventKind::Harvest,
            vec![(3, 4)],
            None,
            &c,
            &o,
        );
        let before = plan_events::sample(&c, &o, &event, &[]);
        c.revise_batch(
            &o,
            &[(3, 4)],
            Some(Production::Animal("COW".into())),
            1,
            48,
            430.,
        )
        .unwrap();
        let after = plan_events::sample(&c, &o, &event, &[]);
        assert_ne!(before.context, after.context);
        assert_eq!(o.farm.money, 2000.);
    }
}

#[cfg(test)]
mod transition_tests {
    use super::*;
    use kagg_engine::state::Cell;
    #[test]
    fn successor_reservations_do_not_starve_current_renewal() {
        let (mut s, mut c) = tests::fixture(96);
        s.farms[0].money = rules::crop("WHEAT").unwrap().seed_cost as f64 + 1.;
        let o = Observation::from_state(&s, 0);
        c.revise_batch(
            &o,
            &[(3, 4)],
            Some(Production::Animal("COW".into())),
            1,
            48,
            180.,
        )
        .unwrap();
        assert!(Schedule::build(&c, &o).material_cash > o.farm.money);
        let a = c.execute_choice(
            &o,
            super::super::plan_chain::Choice {
                base: None,
                links: vec![],
                features: vec![],
            },
        );
        assert!(
            a.market
                .iter()
                .any(|a| a[0] == "BUY_SEED" && a[1] == "WHEAT"),
            "{:?}",
            a.market
        );
        assert!(!a.market.iter().any(|a| a[0] == "BUY_ANIMAL"));
    }
    #[test]
    fn empty_plot_can_revise_pending_successor_without_requiring_another_harvest() {
        let (mut s, mut c) = tests::fixture(96);
        let o = Observation::from_state(&s, 0);
        c.revise_batch(
            &o,
            &[(3, 4)],
            Some(Production::Crop("CARROT".into())),
            1,
            24,
            180.,
        )
        .unwrap();
        s.farms[0].tiles[4][3] = Cell::Empty;
        s.private[0].seeds.add("TOMATO", 1);
        let o = Observation::from_state(&s, 0);
        c.revise_batch(
            &o,
            &[(3, 4)],
            Some(Production::Crop("TOMATO".into())),
            1,
            24,
            180.,
        )
        .unwrap();
        c.observe(&o);
        let id = c.pending_id((3, 4)).unwrap();
        assert!(c.progress[id].armed);
        assert_eq!(
            c.agent.executor.projects[&(3, 4)].production.name(),
            "TOMATO"
        );
    }
    #[test]
    fn armed_unrouted_placement_keeps_ownership_of_its_seed() {
        let (mut s, mut c) = tests::fixture(96);
        let o = Observation::from_state(&s, 0);
        c.revise_batch(
            &o,
            &[(2, 4), (3, 4)],
            Some(Production::Crop("CARROT".into())),
            1,
            24,
            180.,
        )
        .unwrap();
        s.private[0].seeds.add("CARROT", 1);
        let o = Observation::from_state(&s, 0);
        c.observe(&o);
        assert_eq!(c.progress.iter().filter(|p| p.armed).count(), 1);
        let schedule = Schedule::build(&c, &o);
        assert_eq!(schedule.needs.iter().filter(|n| n.stocked).count(), 0);
        c.observe(&o);
        assert_eq!(c.progress.iter().filter(|p| p.armed).count(), 1);
    }
}

#[cfg(test)]
mod audit_tests {
    use super::*;
    use kagg_engine::{
        engine::{self, UnitAction},
        state::Cell,
    };

    #[test]
    fn service_deadline_stays_anchored_after_harvest_and_missing_material() {
        let (mut state, mut c) = tests::fixture(72);
        let o = Observation::from_state(&state, 0);
        c.revise_batch(
            &o,
            &[(3, 4)],
            Some(Production::Animal("COW".into())),
            1,
            24,
            180.,
        )
        .unwrap();
        let promised = Schedule::build(&c, &o).needs[0].service_due;
        state.step = promised + 3;
        state.farms[0].tiles[4][3] = Cell::Empty;
        c.progress[0].first_harvests = 1;
        let o = Observation::from_state(&state, 0);
        let ledger = Schedule::build(&c, &o);
        assert_eq!(ledger.needs[0].service_due, promised);
        assert!(o.step > ledger.needs[0].service_due);
    }
    #[test]
    fn ongoing_crop_forecast_uses_next_yield_after_actual_harvest() {
        let (mut state, _) = tests::fixture(300);
        state.farms[0].tiles[4][3] = Cell::Plant {
            crop: "TOMATO".into(),
            planted_day: 0,
            watered_today: true,
            consecutive_unwatered: 0,
            yield_units: 0,
            max_lifespan_step: -1,
            fertilized_until_day: -1,
        };
        let link = Link {
            site: (3, 4),
            first: Production::Crop("TOMATO".into()),
            next: Production::Animal("COW".into()),
            cycles: 2,
        };
        assert_eq!(
            ready_step(&Observation::from_state(&state, 0), &link, 1),
            312
        );
        if let Cell::Plant { yield_units, .. } = &mut state.farms[0].tiles[4][3] {
            *yield_units = 4;
        }
        assert_eq!(
            ready_step(&Observation::from_state(&state, 0), &link, 0),
            324
        );
    }
    #[test]
    fn separate_route_and_project_do_not_share_one_seed() {
        let (mut state, mut c) = tests::fixture(72);
        c.revise_batch(
            &Observation::from_state(&state, 0),
            &[(3, 4)],
            Some(Production::Crop("CARROT".into())),
            1,
            24,
            180.,
        )
        .unwrap();
        state.private[0].seeds.add("CARROT", 2);
        let project = Project {
            production: Production::Crop("CARROT".into()),
            requested: 72,
            confirmed: false,
            failures: 0,
        };
        c.agent.executor.projects.insert((1, 4), project.clone());
        let mut route = Route::default();
        route.steps.push_back(Scheduled {
            at: 73,
            position: (0, 4),
            action: UnitAction::parse(&["PLANT", "CARROT"]),
        });
        c.agent.executor.assign(0, route);
        let o = Observation::from_state(&state, 0);
        assert!(
            !Schedule::build(&c, &o).needs[0].stocked,
            "two distinct placements own both seeds"
        );
        c.agent.executor.projects.remove(&(1, 4));
        c.agent.executor.projects.insert((0, 4), project);
        assert!(
            Schedule::build(&c, &o).needs[0].stocked,
            "the same placement counts only once"
        );
    }
    #[test]
    fn funding_event_uses_shared_due_cash_and_releases_future_claims() {
        let (mut state, mut c) = tests::fixture(72);
        let o = Observation::from_state(&state, 0);
        c.revise_batch(
            &o,
            &[(2, 4)],
            Some(Production::Crop("CARROT".into())),
            1,
            24,
            180.,
        )
        .unwrap();
        c.revise_batch(
            &o,
            &[(3, 4)],
            Some(Production::Crop("CARROT".into())),
            1,
            24,
            180.,
        )
        .unwrap();
        let mut ledger = Schedule::build(&c, &o);
        let money = ledger.cash_floor + production_cost(&Production::Crop("CARROT".into()));
        assert!(ledger.can_fund(&[0], money, o.step));
        assert!(!ledger.can_fund(&[1], money, o.step));
        state.farms[0].money = money;
        let o = Observation::from_state(&state, 0);
        assert_eq!(
            ledger
                .orders(&o, &[])
                .iter()
                .map(|a| a[2].parse::<i64>().unwrap())
                .sum::<i64>(),
            1
        );
        ledger.needs[0].ready += 100;
        assert!(
            ledger.can_fund(&[1], money, o.step),
            "future procurement cannot take current cash"
        );
    }
    #[test]
    fn extra_cycle_is_replanted_before_successor_in_real_engine() {
        let (mut state, mut c) = tests::fixture(96);
        state.private[0].seeds.add("CARROT", 1);
        c.revise_batch(
            &Observation::from_state(&state, 0),
            &[(3, 4)],
            Some(Production::Crop("CARROT".into())),
            2,
            24,
            180.,
        )
        .unwrap();
        while state.step < 360 {
            let o = Observation::from_state(&state, 0);
            c.observe(&o);
            assert!(!c.progress[0].armed || c.progress[0].first_harvests >= 1);
            assert!(
                !c.progress[0].successor_started || c.progress[0].first_harvests >= 2,
                "successor started before two real harvests: {}",
                c.progress[0].json().dump()
            );
            let a = if super::super::plan_prototype::Agent::planning_due(&o) {
                c.execute_choice(
                    &o,
                    super::super::plan_chain::Choice {
                        base: None,
                        links: vec![],
                        features: vec![],
                    },
                )
            } else {
                c.continue_action(&o)
            };
            engine::step(&mut state, &[a, Default::default()]);
        }
        c.observe(&Observation::from_state(&state, 0));
        assert!(c.progress[0].successor_yielded, "{}", c.report().dump());
        assert!(c.progress[0].first_harvests >= 2);
    }
}

#[cfg(test)]
mod conditional_tests {
    use super::*;
    use kagg_engine::{engine, state::Cell};
    fn conditional(c: &mut Controller, o: &Observation, cycles: usize) {
        c.revise_batch_mode(
            o,
            &[(2, 4), (3, 4)],
            Some(Production::Animal("SHEEP".into())),
            cycles,
            24,
            180.,
            true,
        )
        .unwrap();
    }
    #[test]
    fn conditional_waits_for_confirmed_feed_and_cash_without_erasing_crop() {
        let (mut state, mut c) = tests::fixture(96);
        conditional(&mut c, &Observation::from_state(&state, 0), 1);
        state.private[0].shed.add("SHEEP", 2);
        state.private[0]
            .shed
            .add("WHEAT", -state.private[0].shed.get("WHEAT"));
        c.observe(&Observation::from_state(&state, 0));
        assert!(c.progress.iter().all(|p| !p.armed && !p.failed));
        assert!(c
            .agent
            .executor
            .projects
            .values()
            .all(|p| p.production.name() == "WHEAT"));
        state.private[0].shed.add("WHEAT", 4);
        state.farms[0].money = 0.;
        c.observe(&Observation::from_state(&state, 0));
        assert!(c.progress.iter().all(|p| !p.armed && !p.failed));
        state.farms[0].money = 2000.;
        c.observe(&Observation::from_state(&state, 0));
        assert_eq!(c.progress.iter().filter(|p| p.armed).count(), 2);
    }
    #[test]
    fn conditional_menu_preserves_delayed_two_site_animal_with_no_current_cash() {
        let (mut state, c) = tests::fixture(72);
        state.farms[0].money = 0.;
        let o = Observation::from_state(&state, 0);
        let e = super::super::plan_events::Event::capture(
            "test".into(),
            super::super::plan_events::EventKind::Harvest,
            vec![(2, 4), (3, 4)],
            None,
            &c,
            &o,
        );
        let raw = super::super::plan_events::choices(&c, &o, &e);
        let menu = super::super::plan_menu::build_conditional(&c, &o, &e, &raw, 0).unwrap();
        assert!(menu.iter().any(|p| p.conditional
            && p.cycles == 2
            && p.sites.len() == 2
            && matches!(p.next, Some(Production::Animal(_)))));
        assert!(menu.iter().all(|p| {
            let mut fork = c.clone();
            p.apply(&mut fork, &o).is_ok()
        }));
    }
    #[test]
    fn conditional_can_wait_for_work_but_cannot_arm_without_it() {
        use super::super::executor::{unit, Route, Scheduled};
        let (mut state, mut c) = tests::fixture(96);
        let mut route = Route::default();
        for _ in 0..48 {
            route.steps.push_back(Scheduled {
                at: 100,
                position: (4, 4),
                action: unit("WAIT", "", 0),
            });
        }
        c.agent.executor.assign(0, route);
        let o = Observation::from_state(&state, 0);
        assert!(c
            .revise_batch(
                &o,
                &[(2, 4), (3, 4)],
                Some(Production::Animal("SHEEP".into())),
                1,
                24,
                180.
            )
            .is_err());
        conditional(&mut c, &o, 1);
        state.private[0].shed.add("SHEEP", 2);
        state.private[0].shed.add("WHEAT", 4);
        c.observe(&Observation::from_state(&state, 0));
        assert!(c.progress.iter().all(|p| !p.armed && !p.failed));
        c.agent.executor.routes.clear();
        c.observe(&Observation::from_state(&state, 0));
        assert_eq!(c.progress.iter().filter(|p| p.armed).count(), 2);
    }
    #[test]
    fn conditional_shortage_keeps_real_renewal_and_cancel_available() {
        let (mut state, mut c) = tests::fixture(96);
        state.farms[0].money = 1.;
        state.private[0].seeds.add("WHEAT", 2);
        // Reserve a deliberately unavailable operating floor; no future sale can fund it.
        c.revise_batch_mode(
            &Observation::from_state(&state, 0),
            &[(2, 4), (3, 4)],
            Some(Production::Animal("SHEEP".into())),
            1,
            24,
            100000.,
            true,
        )
        .unwrap();
        let o = Observation::from_state(&state, 0);
        let e = super::super::plan_events::Event::capture(
            "shortage".into(),
            super::super::plan_events::EventKind::Review,
            vec![(2, 4), (3, 4)],
            Some(0),
            &c,
            &o,
        );
        let raw = super::super::plan_events::choices(&c, &o, &e);
        let menu = super::super::plan_menu::build_conditional(&c, &o, &e, &raw, 0).unwrap();
        assert!(menu.iter().any(|p| !p.keep && p.next.is_none()));
        let mut renewed = false;
        while state.step < 210 {
            let o = Observation::from_state(&state, 0);
            c.observe(&o);
            assert!(c.progress.iter().all(|p| !p.armed && !p.failed));
            renewed |= c.progress.iter().any(|p|
                matches!(super::tile(&o.farm,p.link.site),Cell::Plant{crop,planted_day,..} if crop=="WHEAT" && *planted_day>0));
            let a = if super::super::plan_prototype::Agent::planning_due(&o) {
                c.execute_choice(
                    &o,
                    super::super::plan_chain::Choice {
                        base: None,
                        links: vec![],
                        features: vec![],
                    },
                )
            } else {
                c.continue_action(&o)
            };
            engine::step(&mut state, &[a, Default::default()]);
        }
        assert!(renewed, "waiting must keep actual crop renewal");
        assert!(c.progress.iter().all(|p| p.first_harvests > 0));
    }
    #[test]
    fn conditional_real_engine_renews_then_converts_two_sheep() {
        let (mut state, mut c) = tests::fixture(96);
        state.farms[0].money = 10000.;
        conditional(&mut c, &Observation::from_state(&state, 0), 2);
        let mut renewed = false;
        while state.step < 510 {
            let o = Observation::from_state(&state, 0);
            c.observe(&o);
            for p in &c.progress {
                assert!(
                    !p.successor_started || p.first_harvests >= 2,
                    "{}",
                    p.json().dump()
                );
                if matches!(super::tile(&o.farm,p.link.site),Cell::Plant{crop,planted_day,..} if crop=="WHEAT" && *planted_day>0)
                {
                    renewed = true;
                }
            }
            let a = if super::super::plan_prototype::Agent::planning_due(&o) {
                c.execute_choice(
                    &o,
                    super::super::plan_chain::Choice {
                        base: None,
                        links: vec![],
                        features: vec![],
                    },
                )
            } else {
                c.continue_action(&o)
            };
            engine::step(&mut state, &[a, Default::default()]);
        }
        c.observe(&Observation::from_state(&state, 0));
        assert!(renewed);
        assert_eq!(
            c.progress.iter().filter(|p| p.successor_yielded).count(),
            2,
            "{}",
            c.report().dump()
        );
    }
    #[test]
    fn feed_orders_share_actual_cash_and_wait_for_confirmation() {
        let (mut state, mut c) = tests::fixture(96);
        conditional(&mut c, &Observation::from_state(&state, 0), 1);
        state.private[0]
            .shed
            .add("WHEAT", -state.private[0].shed.get("WHEAT"));
        let o = Observation::from_state(&state, 0);
        let ledger = Schedule::build(&c, &o);
        let orders = ledger.orders(&o, &[]);
        assert!(
            orders
                .iter()
                .any(|a| a[0] == "BUY_PRODUCT" && a[1] == "WHEAT"),
            "{:?}",
            orders
        );
        assert!(super::super::plan_chain::order_cost(&o, &orders) <= o.farm.money - 180.);
        assert!(ledger.needs.iter().all(|n| !n.feed_stocked));
        c.observe(&o);
        assert!(c.progress.iter().all(|p| !p.armed));
    }
}

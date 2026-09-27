//! Market choices use public quotes and our own projected actions, never rival private state.
use super::{
    executor::*,
    planner::{Choice, Problem},
};
use kagg_engine::{
    engine, market, rules,
    state::{Cell, OMap, MAX_MARKET_ORDERS, PRODUCTS, SHED_CAP},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MarketMode {
    #[default]
    Learned,
    Rule,
}
impl MarketMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Learned => "learned",
            Self::Rule => "rule",
        }
    }
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "learned" => Ok(Self::Learned),
            "rule" => Ok(Self::Rule),
            _ => Err("market-mode must be learned or rule".into()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Trade {
    pub item: String,
    /// Positive means sell, negative means buy, zero means hold.
    pub quantity: i64,
    pub cash_delta: f64,
    pub marginal_price: i64,
    pub cash_after: f64,
    pub stock_after: i64,
    pub quantity_limit: i64,
}

#[derive(Clone)]
pub struct MarketPoint {
    pub step: i64,
    pub prices: OMap,
    pub inventory: OMap,
}

/// Total proceeds/cost along the price curve, assuming no simultaneous rival orders.
/// Buys quote post-buy inventory; sells quote pre-sell inventory, including the $1 floor rule.
pub fn quote(item: &str, inventory: i64, quantity: i64) -> (f64, i64, i64) {
    let p = market::param(item).expect("known product");
    let mut inv = inventory;
    let mut total = 0.;
    let mut last = market::price(p, inv as f64);
    for _ in 0..quantity.abs() {
        if quantity < 0 {
            inv -= 1;
        }
        last = market::price(p, inv as f64);
        total += last as f64 * if quantity < 0 { -1. } else { 1. };
        if quantity > 0 && last > 1 {
            inv += 1;
        }
    }
    (total, inv, last)
}

pub fn operating_need(o: &Observation, e: &Executor) -> (f64, i64) {
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
    let feed = (2 * animals - wheat).max(0).min(SHED_CAP);
    let mut cost = -quote("WHEAT", o.market.inventory.get("WHEAT"), -feed).0;
    for name in kagg_engine::state::CROP_NAMES {
        let count = e
            .projects
            .values()
            .filter(|p| p.production == Production::Crop(name.into()))
            .count() as i64;
        cost += (count - o.private.seeds.get(name)).max(0) as f64
            * rules::crop(name).unwrap().seed_cost as f64;
    }
    (cost, feed)
}

pub fn committed(e: &Executor, item: &str) -> i64 {
    e.routes
        .iter()
        .flatten()
        .flat_map(|r| &r.steps)
        .filter(|s| s.action.op == "PICKUP" && s.action.item == item)
        .map(|s| s.action.n)
        .sum()
}

/// Events grant a decision opportunity, never force an order.
pub const EVENT_COUNT: usize = 9;
pub const EVENT_NAMES: [&str; EVENT_COUNT] = [
    "periodic",
    "market_change",
    "delivery",
    "cash_change",
    "material_shortage",
    "capacity",
    "endgame",
    "investment",
    "rival_supply",
];

#[derive(Clone)]
pub struct MarketAnchor {
    point: MarketPoint,
    cash: f64,
    stock: OMap,
    cash_need: f64,
    shortages: [i64; 2],
    rival_supply: Option<super::public_supply::Snapshot>,
}
impl MarketAnchor {
    fn capture(o: &Observation, e: &Executor) -> Self {
        let (cash_need, feed) = operating_need(o, e);
        Self {
            point: MarketPoint {
                step: o.step,
                prices: o.market.prices.clone(),
                inventory: o.market.inventory.clone(),
            },
            cash: o.farm.money,
            stock: o.private.shed.clone(),
            cash_need,
            rival_supply: e.rival_supply.points.back().cloned(),
            shortages: [
                feed.max(committed(e, "WHEAT") - o.private.shed.get("WHEAT")),
                (committed(e, "FERTILIZER") - o.private.shed.get("FERTILIZER")).max(0),
            ],
        }
    }
}

pub fn events(
    o: &Observation,
    projected: &Observation,
    e: &Executor,
    investment: bool,
) -> [bool; EVENT_COUNT] {
    let now = MarketAnchor::capture(projected, e);
    let mut flags = [false; EVENT_COUNT];
    flags[0] = e.trade_anchor.is_none() || o.step % 4 == 0;
    flags[2] = PRODUCTS
        .iter()
        .any(|item| projected.private.shed.get(item) > o.private.shed.get(item));
    flags[6] = o.step >= 696; // Last day: all remaining opportunities to realize cash.
    flags[7] = investment;
    if let Some(old) = &e.trade_anchor {
        flags[1] = PRODUCTS.iter().any(|item| {
            // Ignore prices of products we cannot currently trade.
            let relevant =
                now.stock.get(item) > committed(e, item) || matches!(*item, "WHEAT" | "FERTILIZER");
            let price_move = (now.point.prices.get(item) - old.point.prices.get(item)).abs();
            let threshold = ((old.point.prices.get(item) as f64 * 0.03).ceil() as i64).max(1);
            relevant
                && (price_move >= threshold
                    || (now.point.inventory.get(item) - old.point.inventory.get(item)).abs() >= 4)
        });
        flags[2] |= PRODUCTS
            .iter()
            .any(|item| now.stock.get(item) > old.stock.get(item));
        flags[3] = now.cash > old.cash
            || (now.cash - old.cash).abs() >= 50.
            || (now.cash >= now.cash_need) != (old.cash >= old.cash_need);
        flags[4] = now.shortages.iter().zip(old.shortages).any(|(a, b)| *a > b)
            || now.cash_need > old.cash_need + 50.;
        flags[5] = now.stock.sum() >= 90 && old.stock.sum() < 90;
        flags[8] = now
            .rival_supply
            .as_ref()
            .zip(old.rival_supply.as_ref())
            .is_some_and(|(a, b)| a.changed(b));
    } else {
        flags[4] = now.shortages.iter().any(|q| *q > 0);
        flags[5] = now.stock.sum() >= 90;
    }
    flags
}

fn quantity_bins(limit: i64) -> Vec<i64> {
    if limit <= 0 {
        return vec![];
    }
    let mut out = vec![1];
    for percent in [10, 25, 50, 75, 100] {
        out.push((limit * percent + 99) / 100);
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Integrate the real nonlinear buy curve, bounded by shared money and warehouse space.
fn affordable(item: &str, inventory: i64, room: i64, cash: f64) -> i64 {
    let param = market::param(item).unwrap();
    let mut spent = 0.;
    let mut limit = 0;
    for n in 1..=room.max(0) {
        spent += market::price(param, (inventory - n) as f64) as f64;
        if spent > cash {
            break;
        }
        limit = n;
    }
    limit
}

pub struct Trading {
    pub obs: Observation,
    used: [bool; 9],
    stopped: bool,
    pub orders: Vec<Vec<String>>,
    investments: Vec<Vec<String>>,
    reserved_cash: f64,
    start_cash: f64,
    reserved_space: i64,
}
impl Trading {
    pub fn new(obs: Observation, investments: Vec<Vec<String>>) -> Self {
        let mut reserved_cash = 0.;
        let mut reserved_space = 0;
        let mut hires = obs.farm.hires_today as u32;
        for row in &investments {
            let n = row.get(2).and_then(|n| n.parse::<i64>().ok()).unwrap_or(1);
            match row[0].as_str() {
                "BUY_SEED" => {
                    reserved_cash += rules::crop(&row[1]).unwrap().seed_cost as f64 * n as f64
                }
                "BUY_ANIMAL" => {
                    reserved_cash += rules::animal(&row[1]).unwrap().cost as f64 * n as f64;
                    reserved_space += n;
                }
                "HIRE" => {
                    reserved_cash += rules::hire_cost(hires, 1) as f64;
                    hires += 1;
                }
                "BUY_LAND" => {
                    reserved_cash += rules::next_land(obs.farm.unlocked_quadrants.len() - 1)
                        .map(|x| x.1 as f64)
                        .unwrap_or(0.)
                }
                _ => panic!("learned market must own all product purchases"),
            }
        }
        Self {
            start_cash: obs.farm.money,
            obs,
            used: [false; 9],
            stopped: false,
            orders: vec![],
            investments,
            reserved_cash,
            reserved_space,
        }
    }
    /// Shared by collection and submission so triggers cannot drift apart.
    pub fn begin_if_due(
        o: &Observation,
        projected: Observation,
        investments: &mut Vec<Vec<String>>,
        e: &mut Executor,
    ) -> Option<Self> {
        if e.market_mode != MarketMode::Learned {
            return None;
        }
        let flags = events(o, &projected, e, !investments.is_empty());
        if !flags.iter().any(|x| *x) {
            return None;
        }
        e.trade_anchor = Some(MarketAnchor::capture(&projected, e));
        e.trade_events = flags;
        e.stats.trade_sessions += 1;
        for (count, active) in e.stats.trade_events.iter_mut().zip(flags) {
            *count += u64::from(active);
        }
        Some(Self::new(projected, std::mem::take(investments)))
    }

    pub fn next(&mut self, e: &mut Executor) -> Option<Problem> {
        e.trade_start_cash = self.start_cash;
        e.trade_reserved_cash = self.reserved_cash;
        e.trade_reserved_space = self.reserved_space;
        e.trade_remaining = self.used.map(|used| !used);
        e.trade_slots =
            MAX_MARKET_ORDERS.saturating_sub(self.orders.len() + self.investments.len());
        if self.stopped || e.trade_slots == 0 {
            return None;
        }
        // A single global stop holds all remaining products. All other choices jointly
        // select the next product, direction and size; list order imposes no priority.
        let mut choices = vec![Choice::Trade(Trade {
            item: String::new(),
            quantity: 0,
            cash_delta: 0.,
            marginal_price: 0,
            cash_after: self.obs.farm.money - self.reserved_cash,
            stock_after: self.obs.private.shed.sum(),
            quantity_limit: 0,
        })];
        let (cash_need, feed) = operating_need(&self.obs, e);
        for (j, item) in PRODUCTS.iter().enumerate() {
            if self.used[j] {
                continue;
            }
            let stock = self.obs.private.shed.get(item);
            let available = (stock - committed(e, item)).max(0);
            let inv = self.obs.market.inventory.get(item);
            let mut sales = quantity_bins(available);
            // Offer the smallest sale covering working capital as well as percentage bins.
            let gap = (cash_need + self.reserved_cash - self.obs.farm.money).max(0.);
            if gap > 0. {
                let mut supply = inv;
                let mut proceeds = 0.;
                for n in 1..=available {
                    let price = market::price(market::param(item).unwrap(), supply as f64);
                    proceeds += price as f64;
                    if price > 1 {
                        supply += 1;
                    }
                    if proceeds >= gap {
                        sales.push(n);
                        break;
                    }
                }
            }
            let mut quantities: Vec<(i64, i64)> =
                sales.into_iter().map(|q| (q, available)).collect();
            if matches!(*item, "WHEAT" | "FERTILIZER") {
                let room = (SHED_CAP - self.obs.private.shed.sum() - self.reserved_space).max(0);
                let limit = affordable(item, inv, room, self.obs.farm.money - self.reserved_cash);
                let mut buys = quantity_bins(limit);
                let need =
                    (committed(e, item) - stock).max(if *item == "WHEAT" { feed } else { 0 });
                if need > 0 && limit > 0 {
                    buys.push(need.min(limit));
                }
                quantities.extend(buys.into_iter().map(|q| (-q, limit)));
            }
            quantities.sort_unstable();
            quantities.dedup();
            for (quantity, quantity_limit) in quantities {
                let (cash_delta, _, marginal_price) = quote(item, inv, quantity);
                choices.push(Choice::Trade(Trade {
                    item: (*item).into(),
                    quantity,
                    cash_delta,
                    marginal_price,
                    cash_after: self.obs.farm.money + cash_delta - self.reserved_cash,
                    stock_after: stock - quantity,
                    quantity_limit,
                }));
            }
        }
        if choices.len() == 1 {
            return None;
        }
        Some(Problem {
            actor: None,
            choices,
        })
    }
    pub fn apply(&mut self, orders: Vec<Vec<String>>) {
        assert!(!self.stopped);
        if orders.is_empty() {
            self.stopped = true;
            return;
        }
        assert_eq!(
            orders.len(),
            1,
            "one model selection adds exactly one order"
        );
        assert!(self.orders.len() + self.investments.len() < MAX_MARKET_ORDERS);
        for row in orders {
            let j = PRODUCTS
                .iter()
                .position(|item| *item == row[1])
                .expect("known product");
            assert!(!self.used[j], "one transaction per product in a phase");
            self.used[j] = true;
            let n = row[2].parse::<i64>().expect("generated quantity");
            let q = if row[0] == "SELL" { n } else { -n };
            let (delta, inv, _) = quote(&row[1], self.obs.market.inventory.get(&row[1]), q);
            self.obs.farm.money += delta;
            self.obs.private.shed.sub(&row[1], q);
            let old = self.obs.market.inventory.get(&row[1]);
            self.obs.market.inventory.add(&row[1], inv - old);
            self.orders.push(row);
        }
        engine::refresh_prices(&mut self.obs.market);
    }
    pub fn finish(mut self) -> Vec<Vec<String>> {
        self.orders.append(&mut self.investments);
        assert!(self.orders.len() <= MAX_MARKET_ORDERS);
        self.orders
    }
}

/// Public production state, not a claim about guaranteed future receipts.
pub fn production_features(farm: &kagg_engine::state::Farm, day: i64) -> Vec<f32> {
    let mut out = vec![];
    for name in kagg_engine::state::CROP_NAMES {
        let plants: Vec<_> = farm
            .tiles
            .iter()
            .flatten()
            .filter_map(|t| match t {
                Cell::Plant {
                    crop,
                    planted_day,
                    yield_units,
                    ..
                } if crop == name => Some((*planted_day, *yield_units)),
                _ => None,
            })
            .collect();
        out.push(plants.len() as f32 / 25.);
        out.push(
            plants
                .iter()
                .filter(|(d, _)| day - d >= rules::crop(name).unwrap().first_yield_day)
                .map(|(_, q)| q)
                .sum::<i64>() as f32
                / 100.,
        );
    }
    for name in kagg_engine::state::ANIMAL_NAMES {
        let animals: Vec<_> = farm
            .tiles
            .iter()
            .flatten()
            .filter_map(|t| match t {
                Cell::Structure {
                    animal: Some(a), ..
                } if a.animal == name => Some(a),
                _ => None,
            })
            .collect();
        out.push(animals.len() as f32 / 20.);
        out.push(animals.iter().map(|a| a.yield_units).sum::<i64>() as f32 / 100.);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use kagg_engine::{engine::PlayerAction, state::State};

    #[test]
    fn full_order_quotes_match_real_engine_including_buy_curve_and_floor() {
        for (item, q, inv) in [
            ("MILK", 40, 10000),
            ("WHEAT", -16, 9600),
            ("FERTILIZER", -8, 10100),
            ("MILK", 40, 12000),
            ("FERTILIZER", 30, 10490),
        ] {
            let mut s = State::new(33);
            s.farms[0].money = 100000.;
            let old = s.market.inventory.get(item);
            s.market.inventory.add(item, inv - old);
            if q > 0 {
                s.private[0].shed.add(item, q);
            }
            let expected = quote(item, inv, q);
            let a = PlayerAction {
                market: vec![vec![
                    if q > 0 {
                        "SELL".into()
                    } else {
                        "BUY_PRODUCT".into()
                    },
                    item.into(),
                    q.abs().to_string(),
                ]],
                ..Default::default()
            };
            engine::process_market(&mut s, &[a, Default::default()]);
            assert_eq!(s.farms[0].money - 100000., expected.0);
            assert_eq!(s.market.inventory.get(item), expected.1);
        }
    }

    #[test]
    fn hold_is_respected_and_rule_mode_still_sells() {
        let mut s = State::new(9);
        s.private[0].shed.add("MILK", 8);
        let o = Observation::from_state(&s, 0);
        let mut e = Executor::new();
        e.observe(&o);
        let (mut action, projected) = e.project_action(&o, vec![]);
        assert!(action.market.is_empty());
        let mut t = Trading::new(projected, vec![]);
        let mut saw_milk = false;
        while let Some(p) = t.next(&mut e) {
            let choices: Vec<_> = p
                .choices
                .iter()
                .filter_map(|c| {
                    if let Choice::Trade(t) = c {
                        Some(t)
                    } else {
                        None
                    }
                })
                .collect();
            if choices.iter().any(|t| t.item == "MILK") {
                saw_milk = true;
                assert!(choices.iter().any(|t| t.quantity == 2));
                assert!(choices.iter().any(|t| t.quantity == 8));
            }
            let hold = p
                .choices
                .iter()
                .position(|c| matches!(c,Choice::Trade(t) if t.quantity==0))
                .unwrap();
            t.apply(p.select(hold, &mut e, &t.obs.clone()).unwrap());
        }
        assert!(saw_milk);
        action.market = t.finish();
        engine::step(&mut s, &[action, Default::default()]);
        assert_eq!(s.private[0].shed.get("MILK"), 8);
        let o = Observation::from_state(&s, 0);
        e.market_mode = MarketMode::Rule;
        assert!(e
            .action(&o, vec![])
            .market
            .iter()
            .any(|r| r[0] == "SELL" && r[1] == "MILK"));
    }

    #[test]
    fn budgets_capacity_committed_materials_and_order_slots_are_respected() {
        let mut s = State::new(8);
        s.farms[0].money = 100000.;
        s.private[0].shed.add("WHEAT", 10);
        s.private[0].shed.add("MILK", 88);
        let o = Observation::from_state(&s, 0);
        let mut e = Executor::new();
        e.observe(&o);
        e.routes = vec![Some(Route {
            steps: std::collections::VecDeque::from([Scheduled {
                at: 1,
                position: (4, 4),
                action: unit("PICKUP", "WHEAT", 7),
            }]),
            ..Default::default()
        })];
        let mut t = Trading::new(
            o.clone(),
            vec![vec!["BUY_ANIMAL".into(), "COW".into(), "1".into()]],
        );
        if let Some(p) = t.next(&mut e) {
            for c in &p.choices {
                if let Choice::Trade(tr) = c {
                    assert!(
                        !(tr.quantity < 0 && !matches!(tr.item.as_str(), "WHEAT" | "FERTILIZER"))
                    );
                    if tr.quantity < 0 {
                        assert!(tr.quantity >= -1);
                    }
                    if tr.item == "WHEAT" && tr.quantity > 0 {
                        assert!(tr.quantity <= 3);
                    }
                }
            }
        }
        let mut poor = o.clone();
        poor.farm.money = rules::animal("COW").unwrap().cost as f64;
        let mut t = Trading::new(
            poor,
            vec![vec!["BUY_ANIMAL".into(), "COW".into(), "1".into()]],
        );
        if let Some(p) = t.next(&mut e) {
            assert!(p
                .choices
                .iter()
                .all(|c| matches!(c,Choice::Trade(t) if t.quantity>=0)));
        }
        let investments = vec![vec!["BUY_SEED".into(), "WHEAT".into(), "1".into()]; 10];
        let mut t = Trading::new(o, investments);
        assert!(t.next(&mut e).is_none());
        assert_eq!(t.finish().len(), 10);
    }

    #[test]
    fn sequential_purchases_share_cash_and_inventory_budget() {
        let mut s = State::new(6);
        s.farms[0].money = 120.;
        let o = Observation::from_state(&s, 0);
        let mut e = Executor::new();
        e.observe(&o);
        let mut t = Trading::new(o, vec![]);
        while let Some(p) = t.next(&mut e) {
            let i = p
                .choices
                .iter()
                .enumerate()
                .min_by_key(|(_, c)| {
                    if let Choice::Trade(t) = c {
                        t.quantity
                    } else {
                        0
                    }
                })
                .unwrap()
                .0;
            t.apply(p.select(i, &mut e, &t.obs.clone()).unwrap());
            assert!(t.obs.farm.money >= 0.);
            assert!(t.obs.private.shed.sum() <= 100);
        }
        let predicted = t.obs.farm.money;
        let a = PlayerAction {
            market: t.finish(),
            ..Default::default()
        };
        engine::process_market(&mut s, &[a, Default::default()]);
        assert_eq!(s.farms[0].money, predicted);
    }

    #[test]
    fn same_turn_deposit_can_be_sold_without_waiting_another_turn() {
        let mut s = State::new(7);
        s.step = 13;
        s.farms[0].farmer = (4, 4);
        s.private[0].inventories[0].add("MILK", 3);
        let o = Observation::from_state(&s, 0);
        let mut e = Executor::new();
        e.observe(&o);
        e.assign(
            0,
            Route {
                steps: std::collections::VecDeque::from([Scheduled {
                    at: 13,
                    position: (4, 4),
                    action: unit("DROP", "", 0),
                }]),
                ..Default::default()
            },
        );
        let (mut a, projected) = e.project_action(&o, vec![]);
        assert_eq!(projected.private.shed.get("MILK"), 3);
        assert!(events(&o, &projected, &e, false)[2]);
        let mut t = Trading::new(projected, vec![]);
        while let Some(p) = t.next(&mut e) {
            let i = p
                .choices
                .iter()
                .position(|c| matches!(c,Choice::Trade(t) if t.item=="MILK"&&t.quantity==3))
                .unwrap_or_else(|| {
                    p.choices
                        .iter()
                        .position(|c| matches!(c,Choice::Trade(t) if t.quantity==0))
                        .unwrap()
                });
            t.apply(p.select(i, &mut e, &t.obs.clone()).unwrap());
        }
        let expected = t.obs.farm.money;
        a.market = t.finish();
        engine::step(&mut s, &[a, Default::default()]);
        assert_eq!(s.farms[0].money, expected);
        assert_eq!(s.private[0].shed.get("MILK"), 0);
        assert_eq!(s.private[0].inventories[0].get("MILK"), 0);
    }

    #[test]
    fn sales_can_fund_feed_in_the_same_market_phase() {
        let mut s = State::new(71);
        s.farms[0].money = 0.;
        s.private[0].shed.add("MILK", 3);
        let o = Observation::from_state(&s, 0);
        let mut e = Executor::new();
        e.observe(&o);
        let mut t = Trading::new(o, vec![]);
        while let Some(p) = t.next(&mut e) {
            let i = p
                .choices
                .iter()
                .position(|c| matches!(c,Choice::Trade(t) if t.item=="MILK" && t.quantity==3))
                .or_else(|| {
                    p.choices.iter().position(
                        |c| matches!(c,Choice::Trade(t) if t.item=="WHEAT" && t.quantity == -1),
                    )
                })
                .unwrap_or_else(|| {
                    p.choices
                        .iter()
                        .position(|c| matches!(c,Choice::Trade(t) if t.quantity==0))
                        .unwrap()
                });
            t.apply(p.select(i, &mut e, &t.obs.clone()).unwrap());
        }
        let expected = t.obs.farm.money;
        let orders = t.finish();
        assert_eq!(
            orders.iter().map(|o| o[0].as_str()).collect::<Vec<_>>(),
            vec!["SELL", "BUY_PRODUCT"]
        );
        engine::process_market(
            &mut s,
            &[
                PlayerAction {
                    market: orders,
                    ..Default::default()
                },
                Default::default(),
            ],
        );
        assert_eq!(s.private[0].shed.get("WHEAT"), 1);
        assert_eq!(s.farms[0].money, expected);
    }

    fn choose_trade(t: &mut Trading, e: &mut Executor, item: &str, quantity: i64) {
        let p = t.next(e).unwrap();
        let i = p
            .choices
            .iter()
            .position(
                |c| matches!(c, Choice::Trade(tr) if tr.item == item && tr.quantity == quantity),
            )
            .unwrap();
        t.apply(p.select(i, e, &t.obs.clone()).unwrap());
    }

    #[test]
    fn quantities_cover_capacity_money_curve_and_small_needs() {
        let mut s = State::new(91);
        s.farms[0].money = 100000.;
        let o = Observation::from_state(&s, 0);
        let mut e = Executor::new();
        let mut t = Trading::new(o.clone(), vec![]);
        let p = t.next(&mut e).unwrap();
        for item in ["WHEAT", "FERTILIZER"] {
            for n in [1, 10, 25, 50, 75, 100] {
                assert!(p.choices.iter().any(
                    |c| matches!(c, Choice::Trade(tr) if tr.item == item && tr.quantity == -n)
                ));
            }
            for inv in [9500, 10000, 10499, 11000] {
                let max_cash = -quote(item, inv, -37).0;
                assert_eq!(affordable(item, inv, 100, max_cash), 37);
                assert_eq!(affordable(item, inv, 100, max_cash - 1.), 36);
                assert_eq!(affordable(item, inv, 12, max_cash), 12);
                assert_eq!(affordable(item, inv, 100, 0.), 0);
            }
        }
        choose_trade(&mut t, &mut e, "WHEAT", -100);
        assert!(t.next(&mut e).is_none()); // Used product cannot be sold again; warehouse is full.
        engine::process_market(
            &mut s,
            &[
                PlayerAction {
                    market: t.finish(),
                    ..Default::default()
                },
                Default::default(),
            ],
        );
        assert_eq!(s.private[0].shed.get("WHEAT"), 100);

        s.private[0].shed = OMap::default();
        e.projects.insert(
            (3, 3),
            Project {
                production: Production::Animal("COW".into()),
                requested: 0,
                confirmed: true,
                failures: 0,
            },
        );
        let mut t = Trading::new(Observation::from_state(&s, 0), vec![]);
        let p = t.next(&mut e).unwrap();
        assert!(p
            .choices
            .iter()
            .any(|c| matches!(c, Choice::Trade(tr) if tr.item == "WHEAT" && tr.quantity == -2)));
    }

    #[test]
    fn model_selects_product_priority_and_can_stop_with_other_trades_available() {
        for first in ["WHEAT", "FERTILIZER"] {
            let mut s = State::new(92);
            s.farms[0].money = 100000.;
            let mut e = Executor::new();
            let mut t = Trading::new(Observation::from_state(&s, 0), vec![]);
            choose_trade(&mut t, &mut e, first, -75);
            let p = t.next(&mut e).unwrap();
            assert!(p
                .choices
                .iter()
                .any(|c| matches!(c, Choice::Trade(tr) if tr.quantity == -25)));
            assert!(p
                .choices
                .iter()
                .all(|c| matches!(c, Choice::Trade(tr) if tr.item != first && tr.quantity >= -25)));
            choose_trade(&mut t, &mut e, "", 0);
            assert!(t.next(&mut e).is_none());
            let orders = t.finish();
            assert_eq!(orders.len(), 1);
            assert_eq!(orders[0][1], first);
        }
        // The old fixed order cannot do WHEAT first then MILK; the new policy can.
        let mut s = State::new(93);
        s.private[0].shed.add("MILK", 8);
        let mut e = Executor::new();
        let mut t = Trading::new(Observation::from_state(&s, 0), vec![]);
        choose_trade(&mut t, &mut e, "WHEAT", -1);
        choose_trade(&mut t, &mut e, "MILK", 6); // 75% sale bin.
        choose_trade(&mut t, &mut e, "", 0);
        let predicted = t.obs.farm.money;
        let orders = t.finish();
        assert_eq!(
            orders.iter().map(|r| r[1].as_str()).collect::<Vec<_>>(),
            vec!["WHEAT", "MILK"]
        );
        engine::process_market(
            &mut s,
            &[
                PlayerAction {
                    market: orders,
                    ..Default::default()
                },
                Default::default(),
            ],
        );
        assert_eq!(s.farms[0].money, predicted);
        assert_eq!(s.private[0].shed.get("MILK"), 2);
    }

    #[test]
    fn event_triggers_use_observed_changes_and_hold_does_not_spin() {
        let mut s = State::new(94);
        s.step = 1;
        let mut e = Executor::new();
        let o = Observation::from_state(&s, 0);
        let mut t = Trading::begin_if_due(&o, o.clone(), &mut vec![], &mut e).unwrap();
        choose_trade(&mut t, &mut e, "", 0);
        assert!(t.finish().is_empty());
        s.step = 2;
        let o = Observation::from_state(&s, 0);
        assert_eq!(events(&o, &o, &e, false), [false; EVENT_COUNT]);
        assert!(Trading::begin_if_due(&o, o.clone(), &mut vec![], &mut e).is_none());
        let mut changed = o.clone();
        changed.market.inventory.sub("WHEAT", 4);
        engine::refresh_prices(&mut changed.market);
        assert!(events(&changed, &changed, &e, false)[1]);
        changed = o.clone();
        changed.farm.money += 1.;
        assert!(events(&changed, &changed, &e, false)[3]);
        changed = o.clone();
        changed.private.shed.add("MILK", 90);
        let flags = events(&o, &changed, &e, false);
        assert!(flags[2] && flags[5]);
        e.projects.insert(
            (3, 3),
            Project {
                production: Production::Animal("COW".into()),
                requested: 2,
                confirmed: true,
                failures: 0,
            },
        );
        assert!(events(&o, &o, &e, false)[4]);
        assert!(events(&o, &o, &e, true)[7]);
        changed = o.clone();
        changed.step = 697;
        assert!(events(&changed, &changed, &e, false)[6]);
        e.market_mode = MarketMode::Rule;
        let mut orders = vec![vec!["BUY_SEED".into(), "WHEAT".into(), "1".into()]];
        assert!(Trading::begin_if_due(&o, o.clone(), &mut orders, &mut e).is_none());
        assert_eq!(orders.len(), 1);
        assert_eq!(e.stats.trade_sessions, 1);
    }

    #[cfg(feature = "train")]
    #[test]
    fn success_bank_keeps_production_and_trade_coverage_and_realized_trading_episodes() {
        use crate::learning::{experience::Experience, policy::Sample};
        let make = |group: usize, step: i64| {
            let mut f = vec![0.; 32];
            f[31] = group as f32;
            Sample {
                context: vec![0.; super::super::CONTEXT],
                features: vec![f],
                action: 0,
                mc_return: 2.,
                step,
                ..Default::default()
            }
        };
        let mut rows: Vec<_> = (0..1000).map(|i| make(17, i)).collect();
        rows.extend((0..40).map(|i| make(12, i * 10)));
        let e = Experience::from_episode(1, 0, 100., 2, &rows).unwrap();
        assert_eq!(e.rows.len(), 32);
        assert_eq!(
            e.rows
                .iter()
                .filter(|r| r.features[r.action][31] < 16.)
                .count(),
            16
        );
        assert!(Experience::from_episode(2, 0, 100., 0, &[make(17, 0)]).is_some());
        assert!(Experience::from_episode(2, 0, -100., 0, &[make(17, 0)]).is_none());
    }

    #[test]
    fn observed_history_and_public_rival_production_affect_features_but_private_does_not() {
        let mut s = State::new(11);
        let mut e = Executor::new();
        e.observe(&Observation::from_state(&s, 0));
        s.step = 4;
        s.market.inventory.sub("MILK", 20);
        engine::refresh_prices(&mut s.market);
        let o = Observation::from_state(&s, 0);
        e.observe(&o);
        e.observe(&o);
        assert_eq!(e.market_history.len(), 2);
        let p = Problem {
            actor: None,
            choices: vec![Choice::Continue],
        };
        let a = super::super::encoding::encode(&o, &e, &p);
        let mut fresh = Executor::new();
        fresh.observe(&o);
        let b = super::super::encoding::encode(&o, &fresh, &p);
        assert_ne!(a.0, b.0);
        s.private[1].shed.add("FERTILIZER", 99);
        s.private[1].seeds.add("MELON", 99);
        let secret = super::super::encoding::encode(&Observation::from_state(&s, 0), &e, &p);
        assert_eq!(a, secret);
        s.farms[1].tiles[4][4] = Cell::Plant {
            crop: "WHEAT".into(),
            planted_day: 0,
            watered_today: false,
            consecutive_unwatered: 0,
            yield_units: 5,
            max_lifespan_step: 900,
            fertilized_until_day: -1,
        };
        assert_ne!(
            a.0,
            super::super::encoding::encode(&Observation::from_state(&s, 0), &e, &p).0
        );
    }

    #[cfg(feature = "train")]
    #[test]
    fn final_trade_rows_use_real_cash_not_projected_order_proceeds() {
        use super::super::rollout::{Game, Opponent};
        use crate::learning::policy::Decision;
        let mut g = Game::new(17, 0, Opponent::Heuristic, false);
        g.state.step = 718;
        g.state.private[0].shed.add("WHEAT", 4);
        g.state.private[0].shed.add("MILK", 3);
        let initial = g.state.farms[0].money;
        for seat in 0..2 {
            let o = Observation::from_state(&g.state, seat);
            g.agents[seat].observe(&o);
            g.agents[seat].last_market = 718;
            g.agents[seat].last_cash = o.farm.money;
        }
        let mut market_rows = 0;
        while let Some(row) = g.prepare() {
            let trade = row.features[0][31] >= 16.;
            if trade {
                market_rows += 1;
                assert_eq!(row.cash, (initial / 10000.) as f32);
            }
            let i = row
                .features
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a[3].total_cmp(&b[3]))
                .unwrap()
                .0;
            g.accept(Decision {
                action: i,
                logp: 0.,
                value: 0.,
                wait_probability: 0.,
            })
            .unwrap();
        }
        assert!(market_rows >= 2);
        let final_cash = g.state.farms[0].money;
        let rows = g.terminal_rows();
        assert!(
            (rows.iter().map(|r| r.cash_delta).sum::<f32>()
                - (final_cash - initial) as f32 / 10000.)
                .abs()
                < 1e-5
        );
        assert_eq!(g.state.step, 719);
        assert_eq!(g.state.private[0].shed.get("MILK"), 0);
    }

    #[cfg(feature = "train")]
    #[test]
    fn evaluation_accepts_realized_profitable_trading_but_not_idle_or_losses() {
        use super::super::{
            league::Score,
            rollout::{Collection, Game, Opponent},
        };
        let mut g = Game::new(9, 0, Opponent::Heuristic, false);
        g.state.farms[0].money = 3100.;
        let mut c = Collection {
            games: vec![g],
            samples: vec![],
            experiences: vec![],
            seconds: 0.,
            inference_seconds: 0.,
            inference_calls: 0,
            mean_batch: 0.,
        };
        assert_eq!(Score::from_collection(&c).inactive_games, 1);
        c.games[0].trade_stats[0].bought_units[0] = 4;
        c.games[0].trade_stats[0].units[0] = 4;
        assert_eq!(Score::from_collection(&c).inactive_games, 0);
        c.games[0].state.farms[0].money = 2999.;
        assert_eq!(Score::from_collection(&c).inactive_games, 1);
    }

    #[cfg(feature = "train")]
    #[test]
    fn market_exploration_likelihood_matches_ppo_and_allows_holding() {
        use crate::learning::{
            policy::{exploration_proposal, Batch, Policy, Rng, Sample},
            tensor,
        };
        tensor::threads(1);
        let mut s = State::new(7);
        s.private[0].shed.add("WHEAT", 8);
        let o = Observation::from_state(&s, 0);
        let mut e = Executor::new();
        e.observe(&o);
        let mut t = Trading::new(o, vec![]);
        let p = t.next(&mut e).unwrap();
        let (context, features) = super::super::encoding::encode(&t.obs, &e, &p);
        let proposal = exploration_proposal(&features);
        assert!(features
            .iter()
            .zip(&proposal)
            .any(|(f, q)| f[31] == 16. && *q > 0.));
        let devices = if std::env::var_os("ROUTE_RL_TEST_CUDA").is_some() {
            vec![-1, 0]
        } else {
            vec![-1]
        };
        for device in devices {
            let mut policy = Policy::mixed_routes(device, 19, 1e-4).unwrap();
            let mut row = Sample {
                context: context.clone(),
                features: features.clone(),
                exploration: 0.2,
                ..Default::default()
            };
            let d = policy
                .infer(&[row.clone()], false, &mut Rng(13))
                .unwrap()
                .remove(0);
            row.action = d.action;
            row.logp = d.logp;
            row.value = d.value;
            row.advantage = 1.;
            row.reward = 1.;
            let b = Batch::new(&[row.clone()], device).unwrap();
            let (lp, _) = policy.forward(&b).unwrap();
            let behavior = policy.behavior(&lp, &b).unwrap().data().unwrap();
            assert!((behavior[row.action] - row.logp).abs() < 1e-5);
            assert_eq!(policy.update(&[row], 1, 1, &mut Rng(9)).unwrap().updates, 1);
        }
    }
}

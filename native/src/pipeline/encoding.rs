//! Public observation and candidate-specific material/route features. No future state.
use super::{
    executor::*,
    planner::{Choice, Problem},
};
use kagg_engine::state::{Cell, ANIMAL_NAMES, CROP_NAMES, PRODUCTS};
pub fn encode(o: &Observation, e: &Executor, p: &Problem) -> (Vec<f32>, Vec<Vec<f32>>) {
    let mut c = vec![
        o.step as f32 / 719.,
        (o.step % 24) as f32 / 24.,
        o.farm.money as f32 / 10000.,
        o.rival.money as f32 / 10000.,
        o.farm.hands.len() as f32 / 8.,
        o.farm.unlocked_quadrants.len() as f32 / 4.,
        e.projects.len() as f32 / 40.,
        e.labor_load() as f32 / 100.,
    ];
    for item in PRODUCTS {
        c.extend([
            o.market.prices.get(item) as f32 / 200.,
            o.market.inventory.get(item) as f32 / 10000.,
            o.private.shed.get(item) as f32 / 100.,
        ]);
    }
    for item in CROP_NAMES {
        c.push(o.private.seeds.get(item) as f32 / 20.);
        c.push(
            o.farm
                .tiles
                .iter()
                .flatten()
                .filter(|t| matches!(t,Cell::Plant{crop,..} if crop==item))
                .count() as f32
                / 25.,
        );
    }
    for item in ANIMAL_NAMES {
        c.push(
            o.farm
                .tiles
                .iter()
                .flatten()
                .filter(|t| matches!(t,Cell::Structure{animal:Some(a),..} if a.animal==item))
                .count() as f32
                / 20.,
        );
    }
    for shop in kagg_engine::engine::SHOPS_SORTED {
        c.push(f32::from(o.shops.iter().any(|s| s == shop)));
    }
    let actor_start = c.len();
    if let Some(actor) = p.actor {
        let at = pos(&o.farm, actor);
        c.extend([1., at.0 as f32 / 10., at.1 as f32 / 10.]);
        for item in PRODUCTS {
            c.push(o.private.inventories[actor].get(item) as f32 / 30.);
        }
    }
    c.resize(actor_start + 3 + PRODUCTS.len(), 0.);
    // Remaining maturity and committed resources make future cash distinguishable.
    for name in CROP_NAMES {
        let first = kagg_engine::rules::crop(name).unwrap().first_yield_day;
        let remaining = o
            .farm
            .tiles
            .iter()
            .flatten()
            .filter_map(|t| match t {
                Cell::Plant {
                    crop, planted_day, ..
                } if crop == name => Some((first - (o.day() - planted_day)).max(0)),
                _ => None,
            })
            .min()
            .unwrap_or(30);
        c.push(remaining as f32 / 30.);
    }
    for name in ANIMAL_NAMES {
        let first = kagg_engine::rules::animal(name).unwrap().first_yield_day;
        let remaining = o
            .farm
            .tiles
            .iter()
            .flatten()
            .filter_map(|t| match t {
                Cell::Structure {
                    animal: Some(a), ..
                } if a.animal == name => Some((first - (o.day() - a.placed_day)).max(0)),
                _ => None,
            })
            .min()
            .unwrap_or(30);
        c.push(remaining as f32 / 30.);
    }
    c.extend([
        (e.expansion_budget - e.expansion_spent).max(0.) as f32 / 10000.,
        e.projects
            .values()
            .filter(|p| !p.confirmed && p.production != Production::Vacant)
            .count() as f32
            / 2.,
        e.new_projects_today as f32 / 2.,
    ]);
    c.resize(96, 0.);
    // Trends are computed from observations actually seen by this seat.
    for item in PRODUCTS {
        let param = kagg_engine::market::param(item).unwrap();
        c.push((o.market.prices.get(item) as f64 / param.base) as f32);
        c.push(((o.market.inventory.get(item) as f64 - param.i0) / param.t) as f32);
        for horizon in [4, 24] {
            let past = e
                .market_history
                .iter()
                .rev()
                .find(|p| p.step <= o.step - horizon)
                .or_else(|| e.market_history.front());
            let dp = past
                .map(|p| o.market.prices.get(item) - p.prices.get(item))
                .unwrap_or(0);
            let di = past
                .map(|p| o.market.inventory.get(item) - p.inventory.get(item))
                .unwrap_or(0);
            c.extend([
                (dp as f64 / param.base) as f32,
                (di as f64 / param.t) as f32,
            ]);
        }
        c.push(
            o.private
                .inventories
                .iter()
                .map(|i| i.get(item))
                .sum::<i64>() as f32
                / 100.,
        );
        c.push(super::trading::committed(e, item) as f32 / 20.);
        let demand: i64 = o
            .shops
            .iter()
            .map(|shop| {
                let products = kagg_engine::engine::shop_products(shop);
                if products.contains(&item) {
                    if products.len() == 1 {
                        2
                    } else {
                        1
                    }
                } else {
                    0
                }
            })
            .sum();
        c.push((6 * demand + i64::from(item != "FERTILIZER")) as f32 / 24.);
    }
    c.extend(super::trading::production_features(&o.farm, o.day()));
    c.extend(super::trading::production_features(&o.rival, o.day()));
    let (operating_cash, feed) = super::trading::operating_need(o, e);
    c.extend([
        operating_cash as f32 / 10000.,
        feed as f32 / 20.,
        (100 - o.private.shed.sum()) as f32 / 100.,
        e.trade_reserved_cash as f32 / 10000.,
        e.trade_reserved_space as f32 / 100.,
        (o.farm.money - e.trade_start_cash) as f32 / 10000.,
        f32::from(p.choices.iter().any(|c| matches!(c, Choice::Trade(_)))),
        e.market_history
            .front()
            .map(|h| (o.step - h.step) as f32 / 24.)
            .unwrap_or(0.),
    ]);
    c.extend(e.trade_remaining.map(f32::from));
    c.push(e.trade_slots as f32 / 10.);
    c.extend(e.trade_events.map(f32::from));
    c.extend(e.rival_supply.features());
    assert!(c.len() <= super::CONTEXT);
    c.resize(super::CONTEXT, 0.);
    let features = p
        .choices
        .iter()
        .map(|choice| {
            let mut f = vec![0.; 32];
            f[31] = choice.category(e) as f32;
            match choice {
                Choice::Continue => f[0] = 1.,
                Choice::Trade(t) => {
                    // Empty product identifies a global end-of-trading decision.
                    let base = kagg_engine::market::param(&t.item)
                        .map(|p| p.base)
                        .unwrap_or(1.);
                    f[0] = f32::from(t.quantity == 0);
                    f[3] = t.quantity as f32 / 100.;
                    f[4] = t.cash_delta as f32 / 10000.;
                    f[5] = if t.quantity == 0 {
                        0.
                    } else {
                        (t.cash_delta.abs() / t.quantity.abs() as f64 / base) as f32
                    };
                    f[6] = t.cash_after as f32 / 10000.;
                    f[7] = t.stock_after as f32 / 100.;
                    f[8] = super::trading::committed(e, &t.item) as f32 / 20.;
                    for (j, name) in PRODUCTS.iter().enumerate() {
                        f[9 + j] = f32::from(t.item == *name);
                    }
                    f[18] = (o.market.prices.get(&t.item) as f64 / base) as f32;
                    f[19] = (t.marginal_price as f64 / base) as f32;
                    f[20] = operating_cash as f32 / 10000.;
                    f[21] = (100 - o.private.shed.sum() + t.quantity - e.trade_reserved_space)
                        as f32
                        / 100.;
                    f[22] = t.quantity.abs() as f32 / t.quantity_limit.max(1) as f32;
                    f[23] = (t.cash_after - operating_cash) as f32 / 10000.;
                }
                Choice::Route { actor, route } => {
                    f[1] = 1.;
                    f[3] = route.steps.len() as f32 / 24.;
                    f[4] = route.walking as f32 / 24.;
                    f[5] = route.work as f32 / 24.;
                    f[6] = route.sites.len() as f32 / 8.;
                    f[7] = route.crop_jobs as f32 / 8.;
                    f[8] = route.animal_jobs as f32 / 8.;
                    f[9] = route.reused_fertilizer as f32 / 4.;
                    f[10] = route.harvested as f32 / 30.;
                    f[11] = route.replants as f32 / 4.;
                    f[12] = route.pickups.sum() as f32 / 20.;
                    f[13] = route.seeds.sum() as f32 / 8.;
                    f[14] = *actor as f32 / 8.;
                    if let Some(last) = route.steps.back() {
                        f[15] = (last.at - o.step) as f32 / 24.;
                        f[16] = last.position.0 as f32 / 10.;
                        f[17] = last.position.1 as f32 / 10.;
                    }
                    for (j, name) in [
                        "FEED",
                        "CARE",
                        "HARVEST",
                        "COLLECT_FERTILIZER",
                        "FERTILIZE",
                        "PLANT",
                        "WATER",
                        "DROP",
                        "PLACE",
                        "PICKUP",
                        "DIG",
                        "BUILD_PASTURE",
                        "BUILD_COOP",
                    ]
                    .iter()
                    .enumerate()
                    {
                        f[18 + j] =
                            route.steps.iter().filter(|s| s.action.op == *name).count() as f32 / 8.;
                    }
                }
                Choice::Invest {
                    site,
                    production,
                    orders,
                    cost,
                } => {
                    f[2] = 1.;
                    f[3] = *cost as f32 / 1000.;
                    if let Some(p) = site {
                        f[4] = p.0 as f32 / 10.;
                        f[5] = p.1 as f32 / 10.;
                        f[6] = distance(*p, home(*p)) as f32 / 10.;
                        let mut near_crops = 0;
                        let mut near_animals = 0;
                        for (q, pr) in &e.projects {
                            if distance(*p, *q) <= 3 {
                                match pr.production {
                                    Production::Crop(_) => near_crops += 1,
                                    Production::Animal(_) => near_animals += 1,
                                    _ => {}
                                }
                            }
                        }
                        f[7] = near_crops as f32 / 10.;
                        f[8] = near_animals as f32 / 10.;
                    }
                    if let Some(kind) = production {
                        for (j, name) in CROP_NAMES.iter().chain(ANIMAL_NAMES.iter()).enumerate() {
                            f[9 + j] = f32::from(kind.name() == *name);
                        }
                        f[17] = f32::from(*kind == Production::Vacant);
                        let product = match kind.name() {
                            "COW" => "MILK",
                            "SHEEP" => "WOOL",
                            "GOOSE" => "EGG",
                            x => x,
                        };
                        f[18] = o.market.prices.get(product) as f32 / 200.;
                    }
                    f[19] = orders.iter().filter(|r| r[0] == "HIRE").count() as f32;
                    f[20] = f32::from(orders.iter().any(|r| r[0] == "BUY_LAND"));
                    f[21] = orders
                        .iter()
                        .filter_map(|r| r.get(2).and_then(|v| v.parse::<f32>().ok()))
                        .sum::<f32>()
                        / 20.;
                    for (j, name) in ["BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"].iter().enumerate() {
                        f[22 + j] = f32::from(orders.iter().any(|r| r[0] == *name));
                    }
                }
            }
            f
        })
        .collect();
    (c, features)
}

//! Versioned small executable menus. Generation is deterministic, uses visible
//! resources only, and never consults the learner's scores or rollout outcomes.
use super::{
    executor::*,
    plan_chain::Controller,
    plan_events::{Choice, Event},
    plan_resources::{self, Schedule},
};
use kagg_engine::{
    json::Json,
    rules,
    state::{ANIMAL_NAMES, CROP_NAMES},
};
pub const ENCODING: &str = "event-menu-resource-ranked-v3";
pub const MAX_CONTEXTUAL_CHOICES: usize = 6;
pub const MAX_CHOICES: usize = 4;
fn kind_index(p: &Production) -> usize {
    CROP_NAMES
        .iter()
        .position(|k| *k == p.name())
        .unwrap_or_else(|| 5 + ANIMAL_NAMES.iter().position(|k| *k == p.name()).unwrap())
}
/// Economic signature includes remaining cycles, readiness, service deadlines,
/// arming and funding. Identical command parameters alone are NOT equivalence.
fn obligations(c: &Controller, o: &Observation) -> Vec<String> {
    let mut out = vec![];
    for (i, p) in c
        .progress
        .iter()
        .enumerate()
        .filter(|(i, p)| c.is_active(*i) && !p.failed)
    {
        let b = c
            .batches
            .iter()
            .find(|b| !b.cancelled && b.stage.links.contains(&i));
        out.push(format!(
            "{:?}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
            p.link.site,
            p.link.first.name(),
            p.link.next.name(),
            p.link.cycles.saturating_sub(p.first_harvests),
            plan_resources::ready_step(o, &p.link, p.first_harvests),
            p.expected_ready,
            p.first_marker,
            p.armed,
            p.successor_started,
            p.first_started,
            p.retiring,
            p.successor_yielded,
            b.map(|b| b.stage.lead_steps).unwrap_or(24),
            b.map(|b| b.stage.cash_floor)
                .unwrap_or(c.agent.config.cash_reserve),
            b.map(|b| b.stage.deadline).unwrap_or(718)
        ));
    }
    out.sort();
    out
}
struct Projection {
    choice: Choice,
    key: Vec<String>,
    rank: f64,
    family: usize,
}
fn project(
    c: &Controller,
    o: &Observation,
    e: &Event,
    p: &Choice,
    before: &Schedule,
) -> Result<Projection, String> {
    project_mode(c, o, e, p, before, false)
}
fn project_mode(
    c: &Controller,
    o: &Observation,
    e: &Event,
    p: &Choice,
    before: &Schedule,
    contextual: bool,
) -> Result<Projection, String> {
    let mut after = c.clone();
    p.apply(&mut after, o)?;
    let ledger = Schedule::build(&after, o);
    let mut f = vec![0.; 32];
    f[0] = f32::from(p.keep);
    f[1] = p.sites.len() as f32 / 4.;
    let mut price = 0.;
    let mut family = 3;
    if let Some(k) = &p.next {
        f[2 + kind_index(k)] = 1.;
        family = usize::from(matches!(k, Production::Animal(_)));
        let product = match k {
            Production::Animal(a) => rules::animal(a).unwrap().product,
            _ => k.name(),
        };
        price = o.market.prices.get(product) as f64;
    }
    let total = |s: &Schedule| {
        s.needs
            .iter()
            .filter(|n| !n.stocked)
            .map(|n| n.cost)
            .sum::<f64>()
    };
    let ready = |s: &Schedule| {
        s.needs
            .iter()
            .filter(|n| e.sites.contains(&n.site))
            .map(|n| n.ready)
            .max()
            .unwrap_or(o.step)
    };
    let due = ledger
        .needs
        .iter()
        .filter(|n| p.sites.contains(&n.site))
        .map(|n| n.ready - n.lead)
        .min()
        .unwrap_or(o.step);
    let reset = p
        .sites
        .iter()
        .filter_map(|s| c.pending_id(*s))
        .map(|i| c.progress[i].first_harvests)
        .sum::<usize>();
    f[10] = ((total(&ledger) - total(before)) / 10000.) as f32;
    f[11] = ((o.farm.money - ledger.cash_floor - ledger.material_cash) / 10000.) as f32;
    f[12] = ((ledger.work_due - before.work_due) / 100.) as f32;
    f[13] = ((ledger.free_work - ledger.work_due) / 720.) as f32;
    f[14] = (due - o.step) as f32 / 719.;
    f[15] = (ready(&ledger) - o.step) as f32 / 719.;
    f[16] = (ledger.needs.len() as f32 - before.needs.len() as f32) / 8.;
    f[17] = if p.keep { 0. } else { reset as f32 / 8. };
    f[18] = (ready(&ledger) - ready(before)) as f32 / 288.;
    f[19] =
        ledger.needs.iter().filter(|n| n.stocked).count() as f32 / ledger.needs.len().max(1) as f32;
    f[20] = ((ledger.cash_floor - before.cash_floor) / 2000.) as f32;
    f[21 + e.kind as usize] = 1.;
    f[25] = p
        .sites
        .iter()
        .filter(|s| {
            c.pending_id(**s)
                .is_some_and(|i| Some(&c.progress[i].link.next) != p.next.as_ref())
        })
        .count() as f32
        / 4.;
    f[26] = o.rival.money as f32 / 10000.;
    f[27] = (price / 2000.) as f32;
    f[28] = p.cycles as f32 / 2.;
    f[29] = p.lead as f32 / 48.;
    f[31] = 1.;
    let mut choice = p.clone();
    choice.features = f;
    // A shortlist heuristic, never a training reward. It values executable
    // near-term output and penalizes material/work shortages at the actual state.
    let duration = p
        .next
        .as_ref()
        .map(plan_resources::duration)
        .unwrap_or(1)
        .max(1) as f64;
    let shortage = (ledger.cash_floor + total(&ledger) - o.farm.money).max(0.);
    let work_short = (ledger.work_due - ledger.free_work).max(0.);
    let rank = price * p.sites.len() as f64 / duration
        - shortage * 0.05
        - work_short * 5.
        - (ready(&ledger) - o.step).max(0) as f64 * 0.02;
    let rank = if contextual {
        contextual_rank(c, o, p, before, &ledger)
    } else {
        rank
    };
    let mut key = obligations(&after, o);
    if contextual && p.route_handoff {
        key.push("harvest-route-v1".into());
    }
    Ok(Projection {
        choice,
        key,
        rank,
        family,
    })
}
pub fn build(
    c: &Controller,
    o: &Observation,
    e: &Event,
    raw: &[Choice],
    anchor: usize,
) -> Result<Vec<Choice>, String> {
    if anchor >= raw.len() {
        return Err("invalid menu anchor".into());
    }
    let before = Schedule::build(c, o);
    let current = obligations(c, o);
    let base = project(c, o, e, &raw[anchor], &before)?;
    let mut selected = vec![base];
    // Preserve the exact legacy anchor and explicit Keep (token lifecycle may
    // differ), but do not fill exploration slots with redundant reaffirmations.
    if anchor != 0 {
        selected.push(project(c, o, e, &raw[0], &before)?);
    }
    let mut pool = vec![];
    for (i, p) in raw.iter().enumerate() {
        if i == anchor || i == 0 {
            continue;
        }
        let q = project(c, o, e, p, &before)?;
        if q.key == current
            || selected.iter().any(|v| v.key == q.key)
            || pool.iter().any(|v: &Projection| v.key == q.key)
        {
            continue;
        }
        pool.push(q);
    }
    pool.sort_by(|a, b| {
        b.rank
            .total_cmp(&a.rank)
            .then_with(|| a.choice.json().dump().cmp(&b.choice.json().dump()))
    });
    // Cover crop and livestock when feasible; under resource stress also expose
    // cancellation. Missing families are filled with a distinct timing/quantity.
    let stressed = before.free_cash <= 0. || before.work_due > before.free_work;
    for family in if stressed {
        vec![3, 0, 1]
    } else {
        vec![0, 1, 3]
    } {
        if selected.len() >= MAX_CHOICES {
            break;
        }
        if let Some(i) = pool.iter().position(|q| q.family == family) {
            selected.push(pool.remove(i));
        }
    }
    while selected.len() < MAX_CHOICES && !pool.is_empty() {
        selected.push(pool.remove(0));
    }
    Ok(selected.into_iter().map(|p| p.choice).collect())
}
pub fn annotate_context(row: &mut crate::learning::policy::Sample, o: &Observation) {
    // 283..293 already contain crop/livestock counts; 294 is reserved.
    row.context[294] = o.seat as f32;
}
pub fn signature(c: &Controller, o: &Observation) -> Json {
    Json::Arr(obligations(c, o).into_iter().map(Json::Str).collect())
}

/// Four measured choices, preserving temporal alternatives before economic ranking.
/// Old menu construction remains available for accepted v9 deployments.
pub fn build_conditional(
    c: &Controller,
    o: &Observation,
    e: &Event,
    raw: &[Choice],
    anchor: usize,
) -> Result<Vec<Choice>, String> {
    build_conditional_mode(c, o, e, raw, anchor, false)
}
pub fn build_conditional_mode(
    c: &Controller,
    o: &Observation,
    e: &Event,
    raw: &[Choice],
    anchor: usize,
    route_handoff: bool,
) -> Result<Vec<Choice>, String> {
    let before = Schedule::build(c, o);
    let mut selected = vec![project(c, o, e, &raw[anchor], &before)?.choice];
    if anchor != 0 {
        selected.push(project(c, o, e, &raw[0], &before)?.choice);
    }
    let mut pool: Vec<_> = super::plan_events::choices_mode(c, o, e, true)
        .iter()
        .filter(|p| !p.keep)
        .filter_map(|p| {
            let mut p = p.clone();
            p.route_handoff = route_handoff && p.next.is_some();
            project(c, o, e, &p, &before).ok()
        })
        .collect();
    // Public shops contribute demand. Rotate species using visible season time;
    // this is coverage, not an outcome label or a hidden-seed feature.
    let desired = ANIMAL_NAMES[(o.day() as usize) % ANIMAL_NAMES.len()];
    pool.sort_by(|a, b| {
        let preference = |p: &Projection| {
            let name = p.choice.next.as_ref().map(Production::name).unwrap_or("");
            let product = rules::animal(name).map(|a| a.product).unwrap_or(name);
            let shops = o
                .shops
                .iter()
                .filter(|s| kagg_engine::engine::shop_products(s).contains(&product))
                .count();
            (
                shops,
                usize::from(name == desired),
                usize::from(p.choice.sites.len() == e.sites.len().min(2)),
            )
        };
        preference(b)
            .cmp(&preference(a))
            .then_with(|| b.rank.total_cmp(&a.rank))
            .then_with(|| a.choice.json().dump().cmp(&b.choice.json().dump()))
    });
    // Under genuine funding/work shortages cancellation must remain available.
    // It releases only the future promise, never planted crops or purchased stock.
    if e.batch.is_some() && (before.free_cash <= 0. || before.work_due > before.free_work) {
        if let Some(i) = pool.iter().position(|q| q.family == 3) {
            let q = pool.remove(i).choice;
            if !selected.iter().any(|p| p.json() == q.json()) {
                selected.push(q);
            }
        }
    }
    // Compare now versus one renewal of the SAME production and quantity.
    // Delayed candidates survive even when current work/cash is insufficient.
    if let Some(i) = pool
        .iter()
        .position(|q| q.family == 1 && q.choice.cycles == 2)
    {
        let delayed = pool.remove(i).choice;
        let immediate = pool.iter().position(|q| {
            q.choice.next == delayed.next && q.choice.sites == delayed.sites && q.choice.cycles == 1
        });
        if selected.len() < MAX_CHOICES {
            selected.push(delayed);
        }
        if selected.len() < MAX_CHOICES {
            if let Some(i) = immediate {
                selected.push(pool.remove(i).choice);
            }
        }
    }
    // At an owned pending batch anchor=Keep. Expose a smaller/cancelled suffix
    // or a crop alternative as well, without creating an unrelated territory.
    while selected.len() < MAX_CHOICES && !pool.is_empty() {
        let i = pool
            .iter()
            .position(|q| {
                e.batch.is_some()
                    && e.sites.len() > 1
                    && q.family == 1
                    && q.choice.sites.len() == 1
                    && q.choice.cycles == 1
            })
            .or_else(|| pool.iter().position(|q| q.family == 3 && e.batch.is_some()))
            .or_else(|| {
                pool.iter()
                    .position(|q| q.family == 0 && q.choice.cycles == 2)
            })
            .unwrap_or(0);
        let q = pool.remove(i).choice;
        if !selected.iter().any(|p| p.json() == q.json()) {
            selected.push(q);
        }
    }
    annotate_conditional(&mut selected, o);
    Ok(selected)
}

fn annotate_conditional(selected: &mut [Choice], o: &Observation) {
    for p in selected {
        p.features[17] = f32::from(p.conditional);
        let item = p.next.as_ref().map(|k| match k {
            Production::Animal(a) => rules::animal(a).unwrap().product,
            _ => k.name(),
        });
        p.features[27] = item
            .map(|item| {
                o.shops
                    .iter()
                    .filter(|s| kagg_engine::engine::shop_products(s).contains(&item))
                    .count() as f32
                    / 8.
            })
            .unwrap_or(0.);
    }
}

/// Conservative proposal estimate, NOT a reward, value target or promotion gate.
/// Compare replacement production with renewing the current crop over the SAME
/// post-harvest horizon. All observations are public market/own resources.
fn production_estimate(
    c: &Controller,
    o: &Observation,
    k: &Production,
    days: f64,
    scale: f64,
) -> (f64, f64) {
    use kagg_engine::market;
    let (product, units, capital, service, feed) = match k {
        Production::Crop(name) => {
            let r = rules::crop(name).unwrap();
            let cycles = if r.ongoing {
                if days < r.first_yield_day as f64 {
                    0.
                } else {
                    1. + ((days - r.first_yield_day as f64) / r.interval.max(1) as f64).floor()
                }
            } else {
                if days < r.max_yield_day as f64 {
                    0.
                } else {
                    1. + ((days - r.max_yield_day as f64) / (r.max_yield_day + 1) as f64).floor()
                }
            };
            // Do not assume every future visit obtains maximum yield/fertilizer.
            let units = cycles * if r.ongoing { 1. } else { 3. };
            (
                r.name,
                units,
                r.seed_cost as f64 * if r.ongoing { 1. } else { cycles.max(1.) },
                days + cycles * 2.,
                0.,
            )
        }
        Production::Animal(name) => {
            let r = rules::animal(name).unwrap();
            let units = if days < r.first_yield_day as f64 {
                0.
            } else {
                1. + ((days - r.first_yield_day as f64) / r.interval.max(1) as f64).floor()
            };
            let feed_price =
                super::plan_prototype::forecast_price(o, &c.agent.executor, "WHEAT", days)
                    .max(o.market.prices.get("WHEAT") as f64);
            (
                r.product,
                units,
                r.cost as f64,
                days * 2. + units + 3.,
                days * feed_price,
            )
        }
        Production::Vacant => return (0., 0.),
    };
    let price = super::plan_prototype::forecast_price(o, &c.agent.executor, product, days)
        .min(o.market.prices.get(product).max(1) as f64)
        .min(market::price(
            market::param(product).unwrap(),
            o.market.inventory.get(product) as f64 + units * scale,
        ) as f64);
    (units * price - capital - feed, service)
}
fn current_production(c: &Controller, o: &Observation, site: Pos) -> Option<Production> {
    match tile(&o.farm, site) {
        kagg_engine::state::Cell::Plant { crop, .. } => Some(Production::Crop(crop.clone())),
        kagg_engine::state::Cell::Structure {
            animal: Some(a), ..
        } => Some(Production::Animal(a.animal.clone())),
        _ => c
            .agent
            .executor
            .projects
            .get(&site)
            .map(|p| p.production.clone()),
    }
}
fn contextual_rank(
    c: &Controller,
    o: &Observation,
    p: &Choice,
    before: &Schedule,
    after: &Schedule,
) -> f64 {
    let Some(next) = &p.next else {
        return 0.;
    };
    let mut gain = 0.;
    let mut extra_service = 0.;
    let mut delay = 0.;
    for site in &p.sites {
        let Some(n) = after.needs.iter().find(|n| n.site == *site) else {
            continue;
        };
        let days = ((719 - n.ready.max(o.step)) as f64 / 24.).clamp(0., 12.);
        let (value, service) = production_estimate(c, o, next, days, p.sites.len() as f64);
        let (old, old_service) = current_production(c, o, *site)
            .map(|k| production_estimate(c, o, &k, days, p.sites.len() as f64))
            .unwrap_or((0., 0.));
        gain += value - old;
        extra_service += (service - old_service).max(0.) + n.work;
        delay += (n.ready - o.step).max(0) as f64 / 24.;
    }
    let cash_gap = |s: &Schedule| (s.cash_floor + s.material_cash - o.farm.money).max(0.);
    let work_gap = |s: &Schedule| (s.work_due - s.free_work).max(0.);
    // Costs express opportunity/resource pressure only in shortlist ordering.
    // Negative estimates stay eligible for measured exploration within families.
    gain / (1. + extra_service / 24.)
        - (cash_gap(after) - cash_gap(before)).max(0.) * 0.1
        - (work_gap(after) - work_gap(before)).max(0.) * 2.
        - delay
}

/// Same frozen anchor plus distinct resource-conditioned production alternatives.
/// Six is a bound on ACTUAL compared/executed choices, never a sampled subset
/// of a larger action set offered to the network at deployment.
pub fn build_contextual(
    c: &Controller,
    o: &Observation,
    e: &Event,
    raw: &[Choice],
    anchor: usize,
) -> Result<Vec<Choice>, String> {
    if anchor >= raw.len() {
        return Err("invalid contextual anchor".into());
    }
    let before = Schedule::build(c, o);
    let mut selected = vec![project(c, o, e, &raw[anchor], &before)?];
    if anchor != 0 {
        selected.push(project(c, o, e, &raw[0], &before)?);
    }
    let mut pool = Vec::new();
    for mut p in super::plan_events::choices_mode(c, o, e, true) {
        if p.keep {
            continue;
        }
        p.route_handoff = p.next.is_some();
        let Ok(q) = project_mode(c, o, e, &p, &before, true) else {
            continue; // This variant cannot revise the current live commitment.
        };
        if pool.iter().any(|v: &Projection| v.key == q.key) {
            continue;
        }
        pool.push(q);
    }
    pool.sort_by(|a, b| {
        b.rank
            .total_cmp(&a.rank)
            .then_with(|| a.choice.json().dump().cmp(&b.choice.json().dump()))
    });
    fn take(
        selected: &mut Vec<Projection>,
        pool: &mut Vec<Projection>,
        pred: impl Fn(&Projection) -> bool,
    ) {
        if selected.len() >= MAX_CONTEXTUAL_CHOICES {
            return;
        }
        if let Some(i) = pool
            .iter()
            .position(|q| pred(q) && !selected.iter().any(|s| s.key == q.key))
        {
            selected.push(pool.remove(i));
        }
    }
    // Pending commitment: cancel only the future suffix, leaving real production
    // and purchases intact. Otherwise explicitly offer renewing current crops.
    let has_pending = e.sites.iter().any(|s| c.pending_at(*s));
    if has_pending {
        take(&mut selected, &mut pool, |q| q.choice.next.is_none());
    } else {
        take(&mut selected, &mut pool, |q| {
            q.family == 0
                && q.choice.cycles == 1
                && q.choice
                    .sites
                    .iter()
                    .all(|s| current_production(c, o, *s) == q.choice.next)
        });
    }
    take(&mut selected, &mut pool, |q| {
        q.family == 0
            && q.choice.cycles == 1
            && q.choice
                .sites
                .iter()
                .any(|s| current_production(c, o, *s) != q.choice.next)
    });
    // One plot measures a limited livestock investment; additional scale must
    // earn another menu slot through the resource-aware estimate.
    take(&mut selected, &mut pool, |q| {
        q.family == 1 && q.choice.cycles == 1 && q.choice.sites.len() == 1
    });
    // Compare immediate vs one more renewal for the best-ranked CHANGE actually
    // present, rather than rotating to an unrelated animal by calendar day.
    let delayed_target = selected
        .iter()
        .filter(|q| q.choice.conditional && q.choice.next.is_some() && q.choice.cycles == 1)
        .filter(|q| {
            q.choice
                .sites
                .iter()
                .any(|s| current_production(c, o, *s) != q.choice.next)
        })
        .max_by(|a, b| a.rank.total_cmp(&b.rank))
        .map(|q| (q.choice.next.clone(), q.choice.sites.clone()));
    if let Some((next, sites)) = delayed_target {
        take(&mut selected, &mut pool, |q| {
            q.choice.next == next && q.choice.sites == sites && q.choice.cycles == 2
        });
    }
    // If the paired delay is season-infeasible, try another feasible delayed plan.
    if !selected
        .iter()
        .any(|q| q.choice.conditional && q.choice.cycles == 2)
    {
        take(&mut selected, &mut pool, |q| {
            q.choice.next.is_some() && q.choice.cycles == 2
        });
    }
    while selected.len() < MAX_CONTEXTUAL_CHOICES && !pool.is_empty() {
        let before_len = selected.len();
        take(&mut selected, &mut pool, |_| true);
        if selected.len() == before_len {
            break;
        }
    }
    let mut choices: Vec<_> = selected.into_iter().map(|q| q.choice).collect();
    annotate_conditional(&mut choices, o);
    Ok(choices)
}

#[cfg(test)]
mod contextual_tests {
    use super::super::plan_events::{choices, EventKind};
    use super::super::plan_resources::tests::fixture;
    use super::*;
    fn menu(c: &Controller, o: &Observation, batch: Option<usize>) -> Vec<Choice> {
        let event = Event::capture(
            "contextual".into(),
            EventKind::Harvest,
            vec![(2, 4), (3, 4)],
            batch,
            c,
            o,
        );
        let raw = choices(c, o, &event);
        build_contextual(c, o, &event, &raw, 0).unwrap()
    }
    #[test]
    fn contextual_menu_covers_continuation_crop_livestock_and_paired_delay() {
        let (state, c) = fixture(72);
        let o = Observation::from_state(&state, 0);
        let m = menu(&c, &o, None);
        assert!(m[0].keep);
        assert!(m.len() <= MAX_CONTEXTUAL_CHOICES);
        assert!(m
            .iter()
            .any(|p| p.next == Some(Production::Crop("WHEAT".into()))));
        assert!(m
            .iter()
            .any(|p| matches!(&p.next,Some(Production::Crop(k)) if k!="WHEAT")));
        assert!(m
            .iter()
            .any(|p| matches!(p.next, Some(Production::Animal(_))) && p.sites.len() == 1));
        assert!(m.iter().any(|p| p.cycles == 2
            && m.iter()
                .any(|q| q.cycles == 1 && q.next == p.next && q.sites == p.sites)));
        assert!(m
            .iter()
            .skip(1)
            .all(|p| !p.keep && (p.next.is_none() || p.route_handoff)));
        assert!(m.iter().flat_map(|p| &p.features).all(|v| v.is_finite()));
    }
    #[test]
    fn contextual_menu_changes_crop_with_visible_prices_not_hidden_state() {
        let (state, c) = fixture(72);
        let mut low = Observation::from_state(&state, 0);
        // Abundant grain and animal products; only one alternative crop is scarce.
        for item in ["CARROT", "TOMATO", "STRAWBERRY", "MELON"] {
            let n = low.market.inventory.get(item);
            low.market.inventory.add(item, 14000 - n);
            let p = low.market.prices.get(item);
            low.market.prices.add(item, 1 - p);
        }
        let mut carrot = low.clone();
        let mut melon = low;
        for (o, item) in [(&mut carrot, "CARROT"), (&mut melon, "MELON")] {
            let n = o.market.inventory.get(item);
            o.market.inventory.add(item, 8000 - n);
            let p = o.market.prices.get(item);
            o.market.prices.add(item, 10000 - p);
        }
        let first_crop = |m: Vec<Choice>| {
            m.into_iter()
                .find_map(|p| match p.next {
                    Some(Production::Crop(k)) if k != "WHEAT" => Some(k),
                    _ => None,
                })
                .unwrap()
        };
        assert_eq!(first_crop(menu(&c, &carrot, None)), "CARROT");
        assert_eq!(first_crop(menu(&c, &melon, None)), "MELON");
        let mut hidden = state.clone();
        hidden.seed += 1;
        hidden.private[1].shed.add("WOOL", 10000);
        let a = menu(&c, &Observation::from_state(&state, 0), None);
        let b = menu(&c, &Observation::from_state(&hidden, 0), None);
        assert_eq!(
            a.iter().map(Choice::json).collect::<Vec<_>>(),
            b.iter().map(Choice::json).collect::<Vec<_>>()
        );
    }
    #[test]
    fn contextual_menu_keeps_cancellation_and_real_resource_shortage() {
        let (mut state, mut c) = fixture(72);
        let batch = c
            .revise_batch_mode(
                &Observation::from_state(&state, 0),
                &[(2, 4), (3, 4)],
                Some(Production::Animal("SHEEP".into())),
                2,
                24,
                180.,
                true,
            )
            .unwrap();
        state.farms[0].money = 0.;
        state.private[0]
            .shed
            .add("WHEAT", -state.private[0].shed.get("WHEAT"));
        let o = Observation::from_state(&state, 0);
        let m = menu(&c, &o, Some(batch));
        assert!(m[0].keep);
        assert!(m.iter().any(|p| !p.keep && p.next.is_none()));
        for p in m.iter().filter(|p| p.next.is_some()) {
            let mut branch = c.clone();
            p.apply(&mut branch, &o).unwrap();
            branch.observe(&o);
            assert!(branch
                .progress
                .iter()
                .filter(|p| !p.superseded)
                .all(|p| !p.armed));
        }
    }
}

/// Diagnostics only; these categories never control rewards or replay admission.
pub fn family_name(c: &Controller, o: &Observation, p: &Choice) -> &'static str {
    if p.keep {
        "keep"
    } else if p.next.is_none() {
        "cancel_future"
    } else if p.cycles == 2 {
        "delay_one_renewal"
    } else if matches!(p.next, Some(Production::Animal(_))) {
        "livestock"
    } else if p
        .sites
        .iter()
        .all(|s| current_production(c, o, *s) == p.next)
    {
        "renew_crop"
    } else {
        "switch_crop"
    }
}

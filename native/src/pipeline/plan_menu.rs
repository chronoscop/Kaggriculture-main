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
pub const ENCODING: &str = "event-menu-conditional-resources-v2";
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
    Ok(Projection {
        choice,
        key: obligations(&after, o),
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
    let before = Schedule::build(c, o);
    let mut selected = vec![project(c, o, e, &raw[anchor], &before)?.choice];
    if anchor != 0 {
        selected.push(project(c, o, e, &raw[0], &before)?.choice);
    }
    let mut pool: Vec<_> = super::plan_events::choices_mode(c, o, e, true)
        .iter()
        .filter(|p| !p.keep)
        .filter_map(|p| project(c, o, e, p, &before).ok())
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
    for p in &mut selected {
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
    Ok(selected)
}

//! Identity-bearing business events and concrete, amendable next-batch choices.
use super::{
    executor::*,
    plan_chain::Controller,
    plan_resources::{self, Schedule},
};
use crate::learning::policy::Sample;
use kagg_engine::{
    json::Json,
    rules,
    state::{Cell, ANIMAL_NAMES, CROP_NAMES, PRODUCTS},
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub const CONTRACT: &str = "event-batch-context320-actions32-season-pair-v4";
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    Harvest = 0,
    Material = 1,
    Funding = 2,
    Review = 3,
}
#[derive(Clone, Debug)]
struct SiteIdentity {
    site: Pos,
    link: Option<usize>,
    crop_generation: Option<(String, i64)>,
}
#[derive(Clone, Debug)]
pub struct Event {
    pub key: String,
    pub kind: EventKind,
    pub sites: Vec<Pos>,
    pub batch: Option<usize>,
    /// When the business condition changed, rather than when it became editable.
    pub observed_step: i64,
    batch_revision: Option<u64>,
    identities: Vec<SiteIdentity>,
    claimed_slot: Option<usize>,
}
impl Event {
    pub fn capture(
        key: String,
        kind: EventKind,
        sites: Vec<Pos>,
        batch: Option<usize>,
        c: &Controller,
        o: &Observation,
    ) -> Self {
        let identities = sites
            .iter()
            .map(|&site| SiteIdentity {
                site,
                link: c.pending_id(site),
                // A resource event belongs to a pending link even if its current crop renews.
                // A harvest event belongs to exactly the observed crop generation.
                crop_generation: if kind == EventKind::Harvest {
                    match tile(&o.farm, site) {
                        Cell::Plant {
                            crop, planted_day, ..
                        } => Some((crop.clone(), *planted_day)),
                        _ => None,
                    }
                } else {
                    None
                },
            })
            .collect();
        Self {
            key,
            kind,
            sites,
            batch,
            observed_step: o.step,
            batch_revision: batch.map(|id| c.batches[id].revision),
            identities,
            claimed_slot: None,
        }
    }
    fn live_sites(&self, c: &Controller, o: &Observation) -> Vec<Pos> {
        if let Some(id) = self.batch {
            if c.batches
                .get(id)
                .is_none_or(|b| b.cancelled || Some(b.revision) != self.batch_revision)
            {
                return vec![];
            }
        }
        self.identities
            .iter()
            .filter(|i| {
                self.sites.contains(&i.site)
                    && c.pending_id(i.site) == i.link
                    && i.crop_generation.as_ref().is_none_or(|(kind, day)| {
                        matches!(tile(&o.farm, i.site), Cell::Plant { crop, planted_day, .. }
                        if crop == kind && planted_day == day)
                    })
                    && self.batch.is_none_or(|id| {
                        c.batches[id].sites.contains(&i.site)
                            && i.link
                                .is_some_and(|link| c.batches[id].stage.links.contains(&link))
                    })
            })
            .map(|i| i.site)
            .collect()
    }
    pub fn json(&self) -> Json {
        Json::Obj(vec![
            ("key".into(), Json::Str(self.key.clone())),
            ("kind".into(), Json::Num(self.kind as usize as f64)),
            ("observed_step".into(), Json::Num(self.observed_step as f64)),
            (
                "batch".into(),
                self.batch
                    .map(|x| Json::Num(x as f64))
                    .unwrap_or(Json::Null),
            ),
            (
                "batch_revision".into(),
                self.batch_revision
                    .map(|x| Json::Num(x as f64))
                    .unwrap_or(Json::Null),
            ),
            (
                "sites".into(),
                Json::Arr(
                    self.sites
                        .iter()
                        .map(|p| Json::Arr(vec![Json::Num(p.0 as f64), Json::Num(p.1 as f64)]))
                        .collect(),
                ),
            ),
            (
                "identities".into(),
                Json::Arr(
                    self.identities
                        .iter()
                        .filter(|i| self.sites.contains(&i.site))
                        .map(|i| {
                            Json::Obj(vec![
                                (
                                    "site".into(),
                                    Json::Arr(vec![
                                        Json::Num(i.site.0 as f64),
                                        Json::Num(i.site.1 as f64),
                                    ]),
                                ),
                                (
                                    "link".into(),
                                    i.link.map(|v| Json::Num(v as f64)).unwrap_or(Json::Null),
                                ),
                                (
                                    "crop_generation".into(),
                                    i.crop_generation
                                        .as_ref()
                                        .map(|(k, d)| {
                                            Json::Arr(vec![
                                                Json::Str(k.clone()),
                                                Json::Num(*d as f64),
                                            ])
                                        })
                                        .unwrap_or(Json::Null),
                                ),
                            ])
                        })
                        .collect(),
                ),
            ),
        ])
    }
}
#[derive(Clone, Default)]
pub struct Tracker {
    known: BTreeMap<String, (bool, bool, bool, i64)>,
    seen_harvest: BTreeSet<(Pos, String, i64)>,
    pending: VecDeque<Event>,
    pub counts: [usize; 4],
    pub claimed: [bool; 16],
    last_step: Option<i64>,
}
impl Tracker {
    pub fn refresh_after_edit(&mut self, c: &Controller, o: &Observation) {
        self.last_step = None;
        self.observe(c, o);
    }
    pub fn observe(&mut self, c: &Controller, o: &Observation) {
        if self.last_step == Some(o.step) {
            return;
        }
        self.last_step = Some(o.step);
        let ledger = Schedule::build(c, o);
        let mut cohorts: BTreeMap<(String, i64, Pos, Option<usize>), Vec<Pos>> = BTreeMap::new();
        for (site, p) in &c.agent.executor.projects {
            if !p.confirmed
                || c.pending_id(*site).is_some_and(|id| {
                    let p = &c.progress[id];
                    p.armed || p.successor_started || p.retiring || p.failed
                })
            {
                continue;
            }
            if let Cell::Plant {
                crop, planted_day, ..
            } = tile(&o.farm, *site)
            {
                let due = (planted_day + rules::crop(crop).unwrap().max_yield_day) * 24;
                if o.step >= due - 48
                    && o.step <= due + 24
                    && self
                        .seen_harvest
                        .insert((*site, crop.clone(), *planted_day))
                {
                    let batch = c.pending_id(*site).and_then(|id| {
                        c.batches
                            .iter()
                            .find(|b| !b.cancelled && b.stage.links.contains(&id))
                            .map(|b| b.id)
                    });
                    cohorts
                        .entry((crop.clone(), *planted_day, home(*site), batch))
                        .or_default()
                        .push(*site);
                }
            }
        }
        for ((crop, day, _, batch), mut sites) in cohorts {
            sites.sort();
            for group in sites.chunks(4) {
                self.pending.push_back(Event::capture(
                    format!("crop:{crop}:{day}:{group:?}"),
                    EventKind::Harvest,
                    group.to_vec(),
                    batch,
                    c,
                    o,
                ));
            }
        }
        // Legacy foundation links also carry commitments. Give them event identity without
        // creating BatchPlans: merely observing them must not enable the new executor mode.
        let mut groups: BTreeMap<String, (Option<usize>, Vec<usize>)> = BTreeMap::new();
        for n in &ledger.needs {
            let batch = c
                .batches
                .iter()
                .find(|b| !b.cancelled && b.stage.links.contains(&n.id));
            let key = batch
                .map(|b| format!("batch:{}:{}", b.id, b.revision))
                .unwrap_or_else(|| format!("link:{}", n.id));
            groups
                .entry(key)
                .or_insert_with(|| (batch.map(|b| b.id), vec![]))
                .1
                .push(n.id);
        }
        for (key, (batch, ids)) in groups {
            let needs: Vec<_> = ledger
                .needs
                .iter()
                .filter(|n| ids.contains(&n.id))
                .collect();
            let material = needs.iter().all(|n| n.stocked);
            let affordable = ledger.can_fund(&ids, o.farm.money, o.step);
            let blocked =
                needs.iter().any(|n| o.step > n.service_due) || ledger.work_due > ledger.free_work;
            let price = needs
                .first()
                .map(|n| {
                    let product = match &n.production {
                        Production::Animal(k) => rules::animal(k).unwrap().product,
                        _ => n.production.name(),
                    };
                    o.market.prices.get(product)
                })
                .unwrap_or(0);
            let old = self.known.get(&key).copied();
            let (kind, anchor) = if let Some((m, a, bl, anchor)) = old {
                let kind = if material && !m {
                    Some(EventKind::Material)
                } else if affordable && !a {
                    Some(EventKind::Funding)
                } else if (blocked && !bl)
                    || (price - anchor).abs() as f64 > (anchor.abs() as f64 * 0.15).max(5.)
                {
                    Some(EventKind::Review)
                } else {
                    None
                };
                (kind, if kind.is_some() { price } else { anchor })
            } else {
                (
                    if blocked {
                        Some(EventKind::Review)
                    } else {
                        None
                    },
                    price,
                )
            };
            if let Some(kind) = kind {
                self.pending.push_back(Event::capture(
                    format!("{key}:{}:{kind:?}", o.step),
                    kind,
                    needs.iter().map(|n| n.site).collect(),
                    batch,
                    c,
                    o,
                ));
            }
            self.known
                .insert(key, (material, affordable, blocked, anchor));
        }
    }
    pub fn defer(&mut self, mut event: Event) {
        if let Some(slot) = event.claimed_slot.take() {
            self.claimed[slot] = false;
            self.counts[event.kind as usize] -= 1;
        }
        self.pending.push_back(event);
    }
    pub fn take(&mut self, c: &Controller, o: &Observation) -> Option<(Event, usize)> {
        self.take_excluding_batches(c, o, &[])
    }
    /// Armed paired decisions own their batch events while routes temporarily
    /// prevent an edit. Ordinary scope exhaustion must not discard those events.
    pub fn take_excluding_batches(
        &mut self,
        c: &Controller,
        o: &Observation,
        protected_batches: &[usize],
    ) -> Option<(Event, usize)> {
        // One real event of each kind in each quarter of the season. An early burst
        // cannot spend opportunities reserved for later renewals and conversions.
        let phase = (o.step.max(0) / 180).min(3) as usize;
        let n = self.pending.len();
        for _ in 0..n {
            let event = self.pending.pop_front()?;
            if event
                .batch
                .is_some_and(|id| protected_batches.contains(&id))
            {
                if !event.live_sites(c, o).is_empty() {
                    self.pending.push_back(event);
                }
                continue;
            }
            let kind = event.kind as usize;
            let slot = kind * 4 + phase;
            if self.claimed[slot] {
                continue;
            }
            let live = event.live_sites(c, o);
            let sites: Vec<_> = live.iter().copied().filter(|s| c.editable(o, *s)).collect();
            if sites.is_empty() {
                // Retain only a live commitment temporarily occupied by a route. Never let
                // an expired crop/batch event attach to a newly planted crop or replacement.
                if !live.is_empty()
                    && live.iter().any(|s| {
                        matches!(tile(&o.farm, *s), Cell::Plant { .. }) || c.pending_at(*s)
                    })
                {
                    self.pending.push_back(event);
                }
                continue;
            }
            let mut event = event;
            event.sites = sites;
            event.claimed_slot = Some(slot);
            self.claimed[slot] = true;
            self.counts[kind] += 1;
            return Some((event, slot));
        }
        None
    }
    /// A bounded second decision requires a new event on the first decision's batch.
    /// It neither spends another normal season scope nor borrows another batch.
    pub fn take_for_batch(
        &mut self,
        c: &Controller,
        o: &Observation,
        batch: usize,
        not_before: i64,
    ) -> Option<Event> {
        if o.step <= not_before {
            return None;
        }
        let n = self.pending.len();
        for _ in 0..n {
            let mut event = self.pending.pop_front()?;
            if event.batch != Some(batch) || event.observed_step <= not_before {
                self.pending.push_back(event);
                continue;
            }
            let live = event.live_sites(c, o);
            let sites: Vec<_> = live.iter().copied().filter(|s| c.editable(o, *s)).collect();
            if sites.is_empty() {
                if !live.is_empty()
                    && live.iter().any(|s| {
                        matches!(tile(&o.farm, *s), Cell::Plant { .. }) || c.pending_at(*s)
                    })
                {
                    self.pending.push_back(event);
                }
                continue;
            }
            event.sites = sites;
            event.claimed_slot = None;
            return Some(event);
        }
        None
    }
}
#[derive(Clone)]
pub struct Choice {
    pub sites: Vec<Pos>,
    pub next: Option<Production>,
    pub cycles: usize,
    pub lead: i64,
    pub floor: f64,
    pub keep: bool,
    pub features: Vec<f32>,
}
impl Choice {
    pub fn json(&self) -> Json {
        Json::Obj(vec![
            ("keep".into(), Json::Bool(self.keep)),
            (
                "sites".into(),
                Json::Arr(
                    self.sites
                        .iter()
                        .map(|p| Json::Arr(vec![Json::Num(p.0 as f64), Json::Num(p.1 as f64)]))
                        .collect(),
                ),
            ),
            (
                "next".into(),
                self.next
                    .as_ref()
                    .map(|k| Json::Str(k.name().into()))
                    .unwrap_or(Json::Null),
            ),
            ("cycles".into(), Json::Num(self.cycles as f64)),
            ("lead_steps".into(), Json::Num(self.lead as f64)),
            ("cash_floor".into(), Json::Num(self.floor)),
        ])
    }
    pub fn apply(&self, c: &mut Controller, o: &Observation) -> Result<(), String> {
        if !self.keep {
            c.revise_batch(
                o,
                &self.sites,
                self.next.clone(),
                self.cycles,
                self.lead,
                self.floor,
            )?;
        }
        Ok(())
    }
}
fn kinds() -> Vec<Production> {
    CROP_NAMES
        .iter()
        .map(|k| Production::Crop((*k).into()))
        .chain(ANIMAL_NAMES.iter().map(|k| Production::Animal((*k).into())))
        .collect()
}
fn index(k: &Production) -> usize {
    kinds().iter().position(|v| v == k).unwrap_or(0)
}
fn features(c: &Controller, o: &Observation, e: &Event, p: &Choice, s: &Schedule) -> Vec<f32> {
    let mut f = vec![0.; 32];
    f[0] = f32::from(p.keep);
    f[1] = p.sites.len() as f32 / 4.;
    if let Some(k) = &p.next {
        f[2 + index(k)] = 1.;
        f[10] = plan_resources::production_cost(k) as f32 * p.sites.len() as f32 / 10000.;
        f[11] = plan_resources::duration(k) as f32 / 30.;
        let product = match k {
            Production::Animal(a) => rules::animal(a).unwrap().product,
            _ => k.name(),
        };
        f[12] = o.market.prices.get(product) as f32 / 2000.;
    }
    f[13] = p.cycles as f32 / 2.;
    f[14] = p.lead as f32 / 48.;
    f[15] = p.floor as f32 / 2000.;
    f[16] = s.free_cash as f32 / 10000.;
    f[17] = (s.free_work - s.work_due) as f32 / 720.;
    f[18] = f32::from(!p.keep && p.next.is_none());
    f[19] = f32::from(e.sites.iter().any(|site| c.pending_at(*site)));
    f[20 + e.kind as usize] = 1.;
    for site in &p.sites {
        f[24] += distance(*site, home(*site)) as f32 / 40.;
        if let Cell::Plant {
            crop, planted_day, ..
        } = tile(&o.farm, *site)
        {
            f[25] += ((planted_day + rules::crop(crop).unwrap().max_yield_day) * 24 - o.step)
                as f32
                / 288.;
        }
        if let Some(id) = c.pending_id(*site) {
            f[26] += 1. / 4.;
            f[27] += c.progress[id].first_harvests as f32 / 8.;
        }
    }
    f[28] = (s.material_cash
        + p.sites.len() as f64
            * p.next
                .as_ref()
                .map(plan_resources::production_cost)
                .unwrap_or(0.)) as f32
        / 10000.;
    f[29] = (718 - o.step) as f32 / 719.;
    f[31] = 1.;
    f
}
pub fn choices(c: &Controller, o: &Observation, e: &Event) -> Vec<Choice> {
    let ledger = Schedule::build(c, o);
    let mut out = vec![Choice {
        sites: vec![],
        next: None,
        cycles: 1,
        lead: 0,
        floor: 0.,
        keep: true,
        features: vec![],
    }];
    for kind in kinds() {
        for count in [1, 2, 4] {
            if count > e.sites.len() {
                continue;
            }
            // Three coherent funding/timing variants, not independently sampled trade commands.
            for (cycles, lead, extra_reserve) in [(1, 24, 0.), (1, 48, 250.), (2, 12, 0.)] {
                let sites = e.sites[..count].to_vec();
                let latest = sites
                    .iter()
                    .map(|site| plan_resources::revision_ready(c, o, *site, cycles))
                    .max()
                    .unwrap_or(o.step);
                if latest.max(o.step) + plan_resources::duration(&kind) * 24 + 24 >= 719 {
                    continue;
                }
                let p = Choice {
                    sites,
                    next: Some(kind.clone()),
                    cycles,
                    lead,
                    floor: c.agent.config.cash_reserve + extra_reserve,
                    keep: false,
                    features: vec![],
                };
                let extra = p
                    .sites
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
                let released = ledger
                    .needs
                    .iter()
                    .filter(|n| p.sites.contains(&n.site) && n.ready - n.lead <= o.step)
                    .map(|n| n.work)
                    .sum::<f64>();
                if ledger.work_due - released + extra <= ledger.free_work {
                    out.push(p);
                }
            }
        }
    }
    if e.sites.iter().any(|s| c.pending_at(*s)) {
        out.push(Choice {
            sites: e.sites.clone(),
            next: None,
            cycles: 1,
            lead: 0,
            floor: 0.,
            keep: false,
            features: vec![],
        });
    }
    for p in &mut out {
        p.features = features(c, o, e, p, &ledger);
    }
    out
}
/// Explicit public market/own resource features plus a per-cell future commitment map.
/// New outer checkpoint contract prevents interpreting old 320-dimensional weights as this layout.
pub fn sample(c: &Controller, o: &Observation, e: &Event, choices: &[Choice]) -> Sample {
    let s = Schedule::build(c, o);
    let mut x = vec![0.; 320];
    x[0] = o.step as f32 / 719.;
    x[1] = o.farm.money as f32 / 10000.;
    x[2] = s.cash_floor as f32 / 10000.;
    x[3] = s.material_cash as f32 / 10000.;
    x[4] = s.free_cash as f32 / 10000.;
    x[5] = s.free_work as f32 / 720.;
    x[6] = s.work_due as f32 / 720.;
    x[7] = (o.farm.hands.len() + 1) as f32 / 15.;
    x[8] = o.private.shed.sum() as f32 / 100.;
    x[9] = s.feed_keep as f32 / 30.;
    x[10] = s.outstanding_units as f32 / 20.;
    x[11] = c.batches.len() as f32 / 16.;
    x[12 + e.kind as usize] = 1.;
    for (i, k) in PRODUCTS.iter().enumerate() {
        x[16 + 3 * i] = o.market.prices.get(k) as f32 / 2000.;
        x[17 + 3 * i] = o.market.inventory.get(k) as f32 / 100.;
        x[18 + 3 * i] = o.private.shed.get(k) as f32 / 100.;
    }
    let indices: Vec<usize> = (43..48).chain(283..296).collect();
    for (offset, farm) in [(0, &o.farm), (8, &o.rival)] {
        for t in farm.tiles.iter().flatten() {
            let k = match t {
                Cell::Plant { crop, .. } => Some(Production::Crop(crop.clone())),
                Cell::Structure {
                    animal: Some(a), ..
                } => Some(Production::Animal(a.animal.clone())),
                _ => None,
            };
            if let Some(k) = k {
                x[indices[offset + index(&k)]] += 0.025;
            }
        }
    }
    for (id, p) in c.progress.iter().enumerate() {
        if !c.is_active(id) || p.failed {
            continue;
        }
        let at = (p.link.site.1 * 10 + p.link.site.0) as usize;
        x[48 + at] = (1 + index(&p.link.next)) as f32 / 8.;
        x[148 + at] = if p.successor_started {
            -1.
        } else {
            (plan_resources::ready_step(o, &p.link, p.first_harvests) - o.step) as f32 / 719.
        };
        let k = index(&p.link.next);
        x[248 + k * 4] += 0.05;
        x[249 + k * 4] += f32::from(p.armed) / 20.;
        x[250 + k * 4] += f32::from(p.successor_started) / 20.;
    }
    for n in &s.needs {
        if !n.stocked {
            x[251 + index(&n.production) * 4] += 0.05;
        }
    }
    if let Some(b) = e.batch.map(|i| &c.batches[i]) {
        x[280] = b.stage.lead_steps as f32 / 48.;
        x[281] = b.stage.cash_floor as f32 / 2000.;
        x[282] = (b.stage.deadline - o.step) as f32 / 719.;
    }
    for (i, site) in e.sites.iter().take(4).enumerate() {
        let b = 296 + i * 6;
        x[b] = site.0 as f32 / 10.;
        x[b + 1] = site.1 as f32 / 10.;
        if let Cell::Plant {
            crop,
            planted_day,
            yield_units,
            ..
        } = tile(&o.farm, *site)
        {
            x[b + 2] = (1 + index(&Production::Crop(crop.clone()))) as f32 / 8.;
            x[b + 3] = (o.day() - planted_day) as f32 / 30.;
            x[b + 4] = *yield_units as f32 / 10.;
        }
        x[b + 5] = f32::from(c.pending_at(*site));
    }
    Sample {
        context: x,
        features: choices.iter().map(|p| p.features.clone()).collect(),
        step: o.step,
        cash: o.farm.money as f32,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::super::plan_resources::tests::fixture;
    use super::*;
    #[test]
    fn normal_scopes_reserve_one_opportunity_per_kind_per_season_phase() {
        let (mut state, c) = fixture(72);
        let mut tracker = Tracker::default();
        for phase in 0..4 {
            state.step = phase * 180 + 72;
            let o = Observation::from_state(&state, 0);
            for kind in [
                EventKind::Harvest,
                EventKind::Material,
                EventKind::Funding,
                EventKind::Review,
            ] {
                for attempt in 0..3 {
                    tracker.pending.push_back(Event::capture(
                        format!("phase:{phase}:{kind:?}:{attempt}"),
                        kind,
                        vec![(3, 4)],
                        None,
                        &c,
                        &o,
                    ));
                }
                let (_, slot) = tracker.take(&c, &o).unwrap();
                assert_eq!(slot, kind as usize * 4 + phase as usize);
                assert!(
                    tracker.take(&c, &o).is_none(),
                    "early events must not spend later phase slots"
                );
            }
        }
        assert_eq!(tracker.counts, [4; 4]);
        assert!(tracker.claimed.into_iter().all(|x| x));
    }
    #[test]
    fn deferring_undoes_the_original_claim_even_across_phase_boundary() {
        let (mut state, c) = fixture(179);
        let mut tracker = Tracker::default();
        let o = Observation::from_state(&state, 0);
        tracker.pending.push_back(Event::capture(
            "defer".into(),
            EventKind::Review,
            vec![(3, 4)],
            None,
            &c,
            &o,
        ));
        let (event, slot) = tracker.take(&c, &o).unwrap();
        assert_eq!(slot, 12);
        state.step = 180;
        tracker.defer(event);
        assert!(!tracker.claimed[12]);
        assert_eq!(tracker.counts, [0; 4]);
        let (_, slot) = tracker
            .take(&c, &Observation::from_state(&state, 0))
            .unwrap();
        assert_eq!(slot, 13);
        assert!(tracker.claimed[13]);
    }
    #[test]
    fn followup_requires_a_later_event_on_exact_batch_without_spending_normal_scope() {
        let (mut state, mut c) = fixture(72);
        let o = Observation::from_state(&state, 0);
        let batch = c
            .revise_batch(
                &o,
                &[(3, 4)],
                Some(Production::Crop("CARROT".into())),
                1,
                24,
                180.,
            )
            .unwrap();
        let other = c
            .revise_batch(
                &o,
                &[(2, 4)],
                Some(Production::Crop("CARROT".into())),
                1,
                24,
                180.,
            )
            .unwrap();
        let mut tracker = Tracker::default();
        tracker.pending.push_back(Event::capture(
            "immediate".into(),
            EventKind::Review,
            vec![(3, 4)],
            Some(batch),
            &c,
            &o,
        ));
        assert!(tracker.take_for_batch(&c, &o, batch, 72).is_none());
        state.step = 73;
        let o = Observation::from_state(&state, 0);
        tracker.pending.push_back(Event::capture(
            "other".into(),
            EventKind::Material,
            vec![(2, 4)],
            Some(other),
            &c,
            &o,
        ));
        assert!(tracker.take_for_batch(&c, &o, batch, 72).is_none());
        tracker.pending.push_back(Event::capture(
            "later".into(),
            EventKind::Funding,
            vec![(3, 4)],
            Some(batch),
            &c,
            &o,
        ));
        tracker.claimed[8] = true;
        tracker.counts[2] = 1;
        let event = tracker.take_for_batch(&c, &o, batch, 72).unwrap();
        assert_eq!(event.key, "later");
        assert_eq!(event.observed_step, 73);
        assert_eq!(tracker.counts, [0, 0, 1, 0]);
        assert_eq!(tracker.pending.len(), 2);
        tracker.defer(event);
        assert!(
            tracker.claimed[8],
            "deferring followup cannot release a normal claim"
        );
        assert_eq!(tracker.counts, [0, 0, 1, 0]);
        assert!(tracker.take_for_batch(&c, &o, batch, 72).is_some());
    }
    #[test]
    fn followup_cannot_attach_to_superseded_batch_or_crop_generation() {
        let (mut state, mut c) = fixture(72);
        let o = Observation::from_state(&state, 0);
        let batch = c
            .revise_batch(
                &o,
                &[(3, 4)],
                Some(Production::Crop("CARROT".into())),
                1,
                24,
                180.,
            )
            .unwrap();
        state.step = 73;
        let o = Observation::from_state(&state, 0);
        let mut tracker = Tracker::default();
        tracker.pending.push_back(Event::capture(
            "stale-batch".into(),
            EventKind::Material,
            vec![(3, 4)],
            Some(batch),
            &c,
            &o,
        ));
        let replacement = c
            .revise_batch(
                &o,
                &[(3, 4)],
                Some(Production::Crop("TOMATO".into())),
                1,
                24,
                180.,
            )
            .unwrap();
        assert!(tracker.take_for_batch(&c, &o, batch, 72).is_none());
        tracker.pending.push_back(Event::capture(
            "stale-crop".into(),
            EventKind::Harvest,
            vec![(3, 4)],
            Some(replacement),
            &c,
            &o,
        ));
        if let Cell::Plant { planted_day, .. } = &mut state.farms[0].tiles[4][3] {
            *planted_day += 1;
        }
        assert!(tracker
            .take_for_batch(&c, &Observation::from_state(&state, 0), replacement, 72)
            .is_none());
        assert!(tracker.pending.is_empty());
        assert_eq!(tracker.counts, [0; 4]);
    }
    #[test]
    fn inherited_pending_links_receive_harvest_and_resource_events() {
        let (mut state, mut c) = fixture(72);
        state.farms[0].money = 0.;
        let o = Observation::from_state(&state, 0);
        c.execute_choice(
            &o,
            super::super::plan_chain::Choice {
                base: None,
                links: vec![super::super::plan_chain::Link {
                    site: (3, 4),
                    first: Production::Crop("WHEAT".into()),
                    next: Production::Crop("CARROT".into()),
                    cycles: 1,
                }],
                features: vec![],
            },
        );
        assert!(c.batches.is_empty());
        assert!(c.pending_at((3, 4)));
        let mut tracker = Tracker::default();
        tracker.observe(&c, &o);
        let harvest = tracker
            .pending
            .iter()
            .find(|e| e.kind == EventKind::Harvest && e.sites.contains(&(3, 4)))
            .expect("legacy commitment must be revisable");
        assert!(harvest
            .identities
            .iter()
            .any(|i| i.link == c.pending_id((3, 4))));
        tracker.pending.clear();
        state.step += 1;
        state.farms[0].money = 1000.;
        tracker.observe(&c, &Observation::from_state(&state, 0));
        assert!(tracker
            .pending
            .iter()
            .any(|e| e.kind == EventKind::Funding && e.sites == vec![(3, 4)]));
        assert!(
            c.batches.is_empty(),
            "tracking must not alter inherited execution"
        );
        assert!(!c.event_mode());
    }
    #[test]
    fn partial_edit_does_not_reissue_harvest_for_the_unchanged_crop_generation() {
        let (mut state, mut c) = fixture(72);
        let o = Observation::from_state(&state, 0);
        let mut tracker = Tracker::default();
        tracker.observe(&c, &o);
        let (event, _) = tracker.take(&c, &o).unwrap();
        assert_eq!(event.sites.len(), 2);
        c.revise_batch(
            &o,
            &[(3, 4)],
            Some(Production::Crop("CARROT".into())),
            1,
            24,
            180.,
        )
        .unwrap();
        tracker.refresh_after_edit(&c, &o);
        assert!(!tracker.pending.iter().any(|e| e.kind == EventKind::Harvest));
        state.step += 1;
        tracker.observe(&c, &Observation::from_state(&state, 0));
        assert!(!tracker.pending.iter().any(|e| e.kind == EventKind::Harvest));
    }
    #[test]
    fn stale_events_cannot_attach_to_new_crop_or_revised_commitment() {
        let (mut state, mut c) = fixture(72);
        let o = Observation::from_state(&state, 0);
        let crop_event = Event::capture(
            "crop".into(),
            EventKind::Harvest,
            vec![(3, 4)],
            None,
            &c,
            &o,
        );
        if let Cell::Plant { planted_day, .. } = &mut state.farms[0].tiles[4][3] {
            *planted_day += 1;
        }
        let changed = Observation::from_state(&state, 0);
        assert!(crop_event.live_sites(&c, &changed).is_empty());
        let id = c
            .revise_batch(
                &changed,
                &[(2, 4), (3, 4)],
                Some(Production::Crop("CARROT".into())),
                1,
                24,
                180.,
            )
            .unwrap();
        let batch_event = Event::capture(
            "batch".into(),
            EventKind::Material,
            vec![(2, 4), (3, 4)],
            Some(id),
            &c,
            &changed,
        );
        c.revise_batch(
            &changed,
            &[(3, 4)],
            Some(Production::Crop("TOMATO".into())),
            1,
            24,
            180.,
        )
        .unwrap();
        assert!(batch_event.live_sites(&c, &changed).is_empty());
        let mut tracker = Tracker::default();
        tracker.pending.push_back(crop_event);
        tracker.pending.push_back(batch_event);
        assert!(tracker.take(&c, &changed).is_none());
        assert_eq!(tracker.counts, [0; 4]);
    }
    #[test]
    fn funding_events_respect_other_due_commitments() {
        let (mut state, mut c) = fixture(48);
        state.farms[0].money = 0.;
        let o = Observation::from_state(&state, 0);
        for site in [(3, 4), (2, 4)] {
            c.revise_batch(
                &o,
                &[site],
                Some(Production::Crop("CARROT".into())),
                1,
                48,
                180.,
            )
            .unwrap();
        }
        let mut tracker = Tracker::default();
        tracker.observe(&c, &o);
        tracker.pending.clear();
        state.step += 1;
        state.farms[0].money = 180. + rules::crop("CARROT").unwrap().seed_cost as f64;
        tracker.observe(&c, &Observation::from_state(&state, 0));
        assert_eq!(
            tracker
                .pending
                .iter()
                .filter(|e| e.kind == EventKind::Funding)
                .count(),
            1,
            "two batches must not both claim the same cash"
        );
    }
    #[test]
    fn delayed_commitment_triggers_review_without_moving_the_due_date() {
        let (mut state, mut c) = fixture(48);
        state.private[0].seeds.add("CARROT", 1);
        let o = Observation::from_state(&state, 0);
        c.revise_batch(
            &o,
            &[(3, 4)],
            Some(Production::Crop("CARROT".into())),
            1,
            24,
            180.,
        )
        .unwrap();
        let due = Schedule::build(&c, &o).needs[0].service_due;
        let mut tracker = Tracker::default();
        tracker.observe(&c, &o);
        tracker.pending.clear();
        state.step = due + 1;
        tracker.observe(&c, &Observation::from_state(&state, 0));
        assert!(tracker.pending.iter().any(|e| e.kind == EventKind::Review));
        tracker.pending.clear();
        state.step += 1;
        tracker.observe(&c, &Observation::from_state(&state, 0));
        assert!(
            !tracker.pending.iter().any(|e| e.kind == EventKind::Review),
            "the same delay must not produce a review every step"
        );
    }
    #[test]
    fn candidates_and_executor_agree_on_current_batch_and_season_deadline() {
        let (mut state, mut c) = fixture(72);
        let o = Observation::from_state(&state, 0);
        let event = Event::capture(
            "timing".into(),
            EventKind::Harvest,
            vec![(2, 4), (3, 4)],
            None,
            &c,
            &o,
        );
        for choice in choices(&c, &o, &event) {
            choice
                .apply(&mut c.clone(), &o)
                .expect("generated edit must pass the same timing/resource guard");
        }
        state.step = 600;
        for x in [2, 3] {
            if let Cell::Plant { planted_day, .. } = &mut state.farms[0].tiles[4][x] {
                *planted_day = 24;
            }
        }
        let late = Observation::from_state(&state, 0);
        let event = Event::capture(
            "late".into(),
            EventKind::Harvest,
            vec![(3, 4)],
            None,
            &c,
            &late,
        );
        assert_eq!(choices(&c, &late, &event).len(), 1);
        assert!(c
            .revise_batch(
                &late,
                &[(3, 4)],
                Some(Production::Crop("CARROT".into())),
                1,
                24,
                180.
            )
            .is_err());
        assert!(c.batches.is_empty());
        assert!(
            c.progress.is_empty(),
            "infeasible replacement must not partially mutate commitments"
        );
    }
    #[test]
    fn observations_do_not_retrigger_consumed_events_and_clones_preserve_scope() {
        let (mut s, c) = fixture(72);
        let o = Observation::from_state(&s, 0);
        let mut t = Tracker::default();
        t.observe(&c, &o);
        let (_, slot) = t.take(&c, &o).unwrap();
        assert_eq!(slot, 0);
        let mut branch = t.clone();
        t.observe(&c, &o);
        assert!(t.take(&c, &o).is_none());
        s.step += 1;
        let o = Observation::from_state(&s, 0);
        branch.observe(&c, &o);
        assert!(branch.take(&c, &o).is_none());
    }
    #[test]
    fn funding_and_material_events_follow_actual_threshold_crossings() {
        let (mut s, mut c) = fixture(72);
        s.farms[0].money = 0.;
        let o = Observation::from_state(&s, 0);
        let id = c
            .revise_batch(
                &o,
                &[(3, 4)],
                Some(Production::Crop("CARROT".into())),
                1,
                24,
                180.,
            )
            .unwrap();
        let mut t = Tracker::default();
        t.observe(&c, &o);
        t.pending.clear();
        s.step += 1;
        s.farms[0].money = 1000.;
        let o = Observation::from_state(&s, 0);
        t.observe(&c, &o);
        let (e, slot) = t.take(&c, &o).unwrap();
        assert_eq!(e.kind, EventKind::Funding);
        assert_eq!(e.batch, Some(id));
        assert_eq!(slot, 8);
        s.step += 1;
        s.private[0].seeds.add("CARROT", 1);
        let o = Observation::from_state(&s, 0);
        t.observe(&c, &o);
        let (e, slot) = t.take(&c, &o).unwrap();
        assert_eq!(e.kind, EventKind::Material);
        assert_eq!(slot, 4);
        s.step += 1;
        s.farms[0].money += 10.;
        let o = Observation::from_state(&s, 0);
        t.observe(&c, &o);
        assert!(t.take(&c, &o).is_none());
    }
}

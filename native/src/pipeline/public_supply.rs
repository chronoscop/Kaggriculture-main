//! Descriptive supply signals from observed public tiles; no predicted sales or private inventory.
use kagg_engine::{
    rules,
    state::{Cell, Farm, ANIMAL_NAMES, CROP_NAMES, TURNS_PER_DAY},
};
use std::collections::VecDeque;

pub const KINDS: usize = 8; // CROP_NAMES followed by ANIMAL_NAMES.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Maturity {
    pub immature: i64,
    pub earliest_steps: i64,
    pub within_24: i64,
    pub within_72: i64,
}
impl Default for Maturity {
    fn default() -> Self {
        Self {
            immature: 0,
            earliest_steps: 720,
            within_24: 0,
            within_72: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Snapshot {
    pub step: i64,
    pub counts: [i64; KINDS],
    pub visible_units: [i64; KINDS],
    pub ready_units: [i64; KINDS],
    pub maturity: [Maturity; 5],
    // Cumulative observed cohort transitions, not inferred opponent actions.
    pub additions: [i64; KINDS],
    pub removals: [i64; KINDS],
}
impl Snapshot {
    pub fn changed(&self, old: &Self) -> bool {
        self.counts != old.counts
            || self.visible_units != old.visible_units
            || self.ready_units != old.ready_units
            || self.additions != old.additions
            || self.removals != old.removals
            || self.maturity.iter().zip(old.maturity).any(|(a, b)| {
                a.immature != b.immature || a.within_24 != b.within_24 || a.within_72 != b.within_72
            })
        // A countdown changing by one alone does not create an event every turn.
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Site {
    position: (usize, usize),
    kind: usize,
    since_day: i64,
}

#[derive(Clone, Default)]
pub struct History {
    pub points: VecDeque<Snapshot>,
    sites: Vec<Site>,
}
impl History {
    pub fn observe(&mut self, farm: &Farm, step: i64) {
        if self.points.back().is_some_and(|p| p.step == step) {
            return;
        }
        let mut point = Snapshot {
            step,
            counts: [0; KINDS],
            visible_units: [0; KINDS],
            ready_units: [0; KINDS],
            maturity: [Maturity::default(); 5],
            additions: [0; KINDS],
            removals: [0; KINDS],
        };
        let mut sites = vec![];
        for (y, row) in farm.tiles.iter().enumerate() {
            for (x, tile) in row.iter().enumerate() {
                match tile {
                    Cell::Plant {
                        crop,
                        planted_day,
                        yield_units,
                        ..
                    } => {
                        let Some(kind) = CROP_NAMES.iter().position(|name| *name == crop) else {
                            continue;
                        };
                        let first = (*planted_day + rules::crop(crop).unwrap().first_yield_day)
                            * TURNS_PER_DAY;
                        let remaining = (first - step).max(0);
                        point.counts[kind] += 1;
                        point.visible_units[kind] += *yield_units;
                        if remaining == 0 {
                            point.ready_units[kind] += *yield_units;
                        } else {
                            let m = &mut point.maturity[kind];
                            m.immature += 1;
                            m.earliest_steps = m.earliest_steps.min(remaining);
                            // A crop maturing after the last playable action cannot be sold this season.
                            m.within_24 += i64::from(remaining <= 24 && first <= 718);
                            m.within_72 += i64::from(remaining <= 72 && first <= 718);
                        }
                        sites.push(Site {
                            position: (y, x),
                            kind,
                            since_day: *planted_day,
                        });
                    }
                    Cell::Structure {
                        animal: Some(a), ..
                    } => {
                        let Some(kind) = ANIMAL_NAMES
                            .iter()
                            .position(|name| *name == a.animal)
                            .map(|j| j + 5)
                        else {
                            continue;
                        };
                        point.counts[kind] += 1;
                        point.visible_units[kind] += a.yield_units;
                        point.ready_units[kind] += a.yield_units;
                        sites.push(Site {
                            position: (y, x),
                            kind,
                            since_day: a.placed_day,
                        });
                    }
                    _ => {}
                }
            }
        }
        if let Some(previous) = self.points.back() {
            point.additions = previous.additions;
            point.removals = previous.removals;
            // Both vectors are in tile order. A changed crop/animal or planting day
            // identifies a visible replacement even if total farm counts did not change.
            let (mut i, mut j) = (0, 0);
            while i < self.sites.len() || j < sites.len() {
                if j == sites.len()
                    || (i < self.sites.len() && self.sites[i].position < sites[j].position)
                {
                    point.removals[self.sites[i].kind] += 1;
                    i += 1;
                } else if i == self.sites.len() || sites[j].position < self.sites[i].position {
                    point.additions[sites[j].kind] += 1;
                    j += 1;
                } else {
                    if self.sites[i] != sites[j] {
                        point.removals[self.sites[i].kind] += 1;
                        point.additions[sites[j].kind] += 1;
                    }
                    i += 1;
                    j += 1;
                }
            }
        }
        self.sites = sites;
        self.points.push_back(point);
        while self.points.front().is_some_and(|p| p.step < step - 24) {
            self.points.pop_front();
        }
    }

    /// 20 maturity values, 32 trend values, 16 cohort changes and observed history age.
    pub fn features(&self) -> Vec<f32> {
        let Some(now) = self.points.back() else {
            return vec![0.; 69];
        };
        let mut out = vec![];
        for m in now.maturity {
            out.extend([
                m.immature as f32 / 25.,
                m.earliest_steps as f32 / 720.,
                m.within_24 as f32 / 25.,
                m.within_72 as f32 / 25.,
            ]);
        }
        for horizon in [4, 24] {
            let past = self
                .points
                .iter()
                .rev()
                .find(|p| p.step <= now.step - horizon)
                .unwrap_or_else(|| self.points.front().unwrap());
            for j in 0..KINDS {
                out.extend([
                    (now.counts[j] - past.counts[j]) as f32 / 25.,
                    (now.visible_units[j] - past.visible_units[j]) as f32 / 100.,
                ]);
            }
        }
        let oldest = self.points.front().unwrap();
        for j in 0..KINDS {
            out.extend([
                (now.additions[j] - oldest.additions[j]) as f32 / 25.,
                (now.removals[j] - oldest.removals[j]) as f32 / 25.,
            ]);
        }
        out.push((now.step - oldest.step) as f32 / 24.);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{
        encoding,
        executor::{unit, Executor, Observation},
        planner::{Choice, Problem},
        trading::{self, MarketMode, Trading},
    };
    use kagg_engine::{
        engine,
        state::{AnimalTile, State},
    };

    fn plant(name: &str, day: i64, amount: i64) -> Cell {
        Cell::Plant {
            crop: name.into(),
            planted_day: day,
            watered_today: false,
            consecutive_unwatered: 0,
            yield_units: amount,
            max_lifespan_step: 900,
            fertilized_until_day: -1,
        }
    }
    fn cow(day: i64, amount: i64) -> Cell {
        Cell::Structure {
            kind: "PASTURE".into(),
            animal: Some(AnimalTile {
                animal: "COW".into(),
                placed_day: day,
                yield_units: amount,
                consecutive_unfed: 0,
                fed_today: false,
                cared_today: false,
                fertilizer_available: false,
                pending_care_bonus: 0,
            }),
        }
    }

    #[test]
    fn maturity_uses_actual_turn_boundary_and_agrees_with_harvest_legality() {
        let mut s = State::new(190);
        s.farms[1].tiles[3][3] = plant("WHEAT", 0, 4);
        s.farms[1].farmer = (3, 3);
        let mut h = History::default();
        for (step, remaining, soon, ready) in [
            (23, 25, 0, 0),
            (24, 24, 1, 0),
            (47, 1, 1, 0),
            (48, 720, 0, 4),
        ] {
            h.observe(&s.farms[1], step);
            let p = h.points.back().unwrap();
            assert_eq!(p.maturity[0].earliest_steps, remaining);
            assert_eq!(p.maturity[0].within_24, soon);
            assert_eq!(p.ready_units[0], ready);
            let mut farm = s.farms[1].clone();
            let mut private = s.private[1].clone();
            engine::apply_unit_action(
                &mut farm,
                &mut private,
                0,
                &unit("HARVEST", "", 0),
                step / 24,
            );
            assert_eq!(private.inventories[0].get("WHEAT"), ready);
        }
        // Reaching maturity at terminal step 720 does not create a within-season cohort.
        s.farms[1].tiles[3][3] = plant("WHEAT", 28, 2);
        h.observe(&s.farms[1], 700);
        let p = h.points.back().unwrap();
        assert_eq!(p.maturity[0].earliest_steps, 20);
        assert_eq!(p.maturity[0].within_24, 0);
        assert_eq!(p.maturity[0].within_72, 0);
    }

    #[test]
    fn observed_replant_and_expansion_survive_unchanged_total_counts() {
        let mut s = State::new(191);
        s.farms[1].tiles[3][3] = plant("WHEAT", 3, 1);
        let mut h = History::default();
        h.observe(&s.farms[1], 100);
        assert_eq!(h.points.back().unwrap().additions, [0; KINDS]);
        s.farms[1].tiles[3][3] = plant("WHEAT", 4, 1);
        s.farms[1].tiles[3][4] = cow(4, 2);
        h.observe(&s.farms[1], 101);
        h.observe(&s.farms[1], 101);
        assert_eq!(h.points.len(), 2);
        let p = h.points.back().unwrap();
        assert_eq!(p.counts[0], 1);
        assert_eq!(p.additions[0], 1);
        assert_eq!(p.removals[0], 1);
        let cow_index = ANIMAL_NAMES.iter().position(|x| *x == "COW").unwrap() + 5;
        assert_eq!(p.additions[cow_index], 1);
        // A visible yield decrease is recorded as such, without claiming it was sold.
        s.farms[1].tiles[3][4] = cow(4, 0);
        h.observe(&s.farms[1], 102);
        let f = h.features();
        assert_eq!(f.len(), 69);
        assert_eq!(f[52], 1. / 25.); // observed WHEAT additions over available history
        assert_eq!(f[53], 1. / 25.); // observed WHEAT removals, despite zero net count
        assert_eq!(h.points.back().unwrap().visible_units[cow_index], 0);
        assert!(h.points.back().unwrap().changed(&h.points[1]));
    }

    #[test]
    fn supply_history_is_bounded_and_trends_only_use_observed_window() {
        let mut s = State::new(192);
        let mut h = History::default();
        for step in 0..=40 {
            if step == 4 {
                s.farms[1].tiles[3][3] = plant("WHEAT", 0, 1);
            }
            if step == 25 {
                s.farms[1].tiles[3][4] = plant("WHEAT", 1, 1);
            }
            h.observe(&s.farms[1], step);
        }
        assert_eq!(h.points.len(), 25);
        assert_eq!(h.points.front().unwrap().step, 16);
        let f = h.features();
        assert_eq!(f[20], 0.); // count change in last four steps
        assert_eq!(f[36], 1. / 25.); // count change in last 24 steps
        assert_eq!(f[52], 1. / 25.); // excludes the addition before the window
        assert_eq!(f[68], 1.);
    }

    #[test]
    fn rival_changes_trigger_before_prices_move_without_retriggering_a_countdown() {
        let mut s = State::new(193);
        s.step = 1;
        let mut e = Executor::new();
        let o = Observation::from_state(&s, 0);
        e.observe(&o);
        let mut t = Trading::begin_if_due(&o, o.clone(), &mut vec![], &mut e).unwrap();
        t.apply(vec![]); // end the phase
        s.step = 2;
        s.farms[1].tiles[3][3] = plant("WHEAT", 0, 2);
        let o = Observation::from_state(&s, 0);
        e.observe(&o);
        let flags = trading::events(&o, &o, &e, false);
        assert!(flags[8]);
        assert_eq!(flags.iter().filter(|x| **x).count(), 1);
        let mut t = Trading::begin_if_due(&o, o.clone(), &mut vec![], &mut e).unwrap();
        t.apply(vec![]);
        assert_eq!(e.stats.trade_events[8], 1);
        s.step = 3;
        let o = Observation::from_state(&s, 0);
        e.observe(&o);
        assert!(!trading::events(&o, &o, &e, false).iter().any(|x| *x));
        assert!(Trading::begin_if_due(&o, o.clone(), &mut vec![], &mut e).is_none());
        // Crossing the near-maturity window changes supply information even with unchanged prices.
        s.step = 24;
        let o = Observation::from_state(&s, 0);
        e.observe(&o);
        assert!(trading::events(&o, &o, &e, false)[8]);
    }

    #[test]
    fn supply_inputs_are_shared_by_both_modes_and_do_not_use_rival_private_state() {
        let mut s = State::new(194);
        let mut e = Executor::new();
        e.observe(&Observation::from_state(&s, 0));
        s.step = 1;
        s.farms[1].tiles[3][3] = plant("MELON", 0, 3);
        let o = Observation::from_state(&s, 0);
        e.observe(&o);
        let p = Problem {
            actor: None,
            choices: vec![Choice::Continue],
        };
        let original = encoding::encode(&o, &e, &p);
        let mut rule = e.clone();
        rule.market_mode = MarketMode::Rule;
        assert_eq!(original, encoding::encode(&o, &rule, &p));
        s.private[1].shed.add("MILK", 80);
        s.private[1].seeds.add("WHEAT", 50);
        s.private[1].inventories[0].add("FERTILIZER", 20);
        let hidden = Observation::from_state(&s, 0);
        e.observe(&hidden);
        assert_eq!(original, encoding::encode(&hidden, &e, &p));
        // A newly observed public change is available to the next decision.
        s.step = 2;
        s.farms[1].tiles[3][3] = plant("CARROT", 0, 1);
        e.observe(&Observation::from_state(&s, 0));
        assert_ne!(
            original.0,
            encoding::encode(&Observation::from_state(&s, 0), &e, &p).0
        );
    }
}

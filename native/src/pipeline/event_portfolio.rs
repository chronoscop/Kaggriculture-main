//! Event-scoped learning layered over an immutable accepted v3 foundation.
use super::{
    executor::Observation,
    plan_chain::Controller,
    plan_events::{self, Choice, Tracker},
    plan_portfolio as legacy,
    plan_prototype::{Agent, Config},
};
use crate::learning::policy::{Policy, Rng, Sample};
use kagg_engine::{engine::PlayerAction, json::Json};
pub use legacy::SLOTS;
use std::collections::BTreeMap;
pub const SCHEMA: &str = "event-plan-improvement-v4";
pub const CONTRACT: &str = plan_events::CONTRACT;
pub const LEGACY_CONTRACT: &str = "event-batch-context320-actions32-v2";
/// A bounded replacement consists of the initial choice and, optionally, one
/// conditional revision of the batch it creates. Both models are frozen and
/// accepted together; accepting one never grants control over other batches.
#[derive(Clone, Debug, PartialEq)]
pub struct Slot {
    pub iteration: u64,
    pub weights: Json,
    pub followup: Option<Json>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Portfolio {
    pub revision: u64,
    pub slots: Vec<Option<Slot>>,
    pub foundation: legacy::Portfolio,
}
impl Portfolio {
    pub fn empty() -> Self {
        Self {
            revision: 0,
            slots: vec![None; SLOTS],
            foundation: legacy::Portfolio::empty(),
        }
    }
    pub fn from_foundation(foundation: legacy::Portfolio) -> Self {
        Self {
            foundation,
            ..Self::empty()
        }
    }
    fn patches(&self) -> legacy::Portfolio {
        legacy::Portfolio {
            revision: self.revision,
            slots: self
                .slots
                .iter()
                .map(|s| {
                    s.as_ref().map(|s| legacy::Slot {
                        iteration: s.iteration,
                        weights: s.weights.clone(),
                    })
                })
                .collect(),
        }
    }
    fn followups(&self) -> legacy::Portfolio {
        legacy::Portfolio {
            revision: self.revision,
            slots: self
                .slots
                .iter()
                .map(|s| {
                    s.as_ref().and_then(|s| {
                        s.followup.as_ref().map(|weights| legacy::Slot {
                            iteration: s.iteration,
                            weights: weights.clone(),
                        })
                    })
                })
                .collect(),
        }
    }
    pub fn propose(&self, slot: usize, iteration: u64, weights: Json) -> Result<Self, String> {
        self.replacement(slot, iteration, weights, None)
    }
    pub fn propose_paired(
        &self,
        slot: usize,
        iteration: u64,
        weights: Json,
        followup_weights: Json,
    ) -> Result<Self, String> {
        self.replacement(slot, iteration, weights, Some(followup_weights))
    }
    fn replacement(
        &self,
        slot: usize,
        iteration: u64,
        weights: Json,
        followup: Option<Json>,
    ) -> Result<Self, String> {
        if slot >= SLOTS
            || self.slots.len() != SLOTS
            || !weights.is_obj()
            || followup.as_ref().is_some_and(|w| !w.is_obj())
        {
            return Err("invalid event policy replacement".into());
        }
        let mut next = self.clone();
        next.revision = self
            .revision
            .checked_add(1)
            .ok_or("event portfolio revision exhausted")?;
        next.slots[slot] = Some(Slot {
            iteration,
            weights,
            followup,
        });
        Ok(next)
    }
    pub fn json(&self) -> Json {
        let mut patches = self.patches().json();
        patches.set_path(
            "slots",
            Json::Arr(
                self.slots
                    .iter()
                    .map(|s| match s {
                        None => Json::Null,
                        Some(s) => Json::Obj(vec![
                            ("iteration".into(), Json::Str(s.iteration.to_string())),
                            ("weights".into(), s.weights.clone()),
                            ("followup".into(), s.followup.clone().unwrap_or(Json::Null)),
                        ]),
                    })
                    .collect(),
            ),
        );
        Json::Obj(vec![
            ("contract".into(), Json::Str(CONTRACT.into())),
            ("foundation".into(), self.foundation.json()),
            ("event_patches".into(), patches),
        ])
    }
    pub fn parse(j: &Json) -> Result<Self, String> {
        let legacy_contract = matches!(
            j.get("contract").str(),
            LEGACY_CONTRACT | "event-batch-context320-actions32-season-pair-v3"
        );
        if j.get("contract").str() != CONTRACT && !legacy_contract {
            return Err("wrong event policy contract".into());
        }
        let p = legacy::Portfolio::parse(j.get("event_patches"))?;
        if legacy_contract && (p.revision != 0 || p.slots.iter().any(Option::is_some)) {
            return Err("cannot reinterpret accepted policies under a changed event execution contract; preserve the original checkpoint and deployment".into());
        }
        let slots = p
            .slots
            .into_iter()
            .enumerate()
            .map(|(i, s)| {
                s.map(|s| Slot {
                    iteration: s.iteration,
                    weights: s.weights,
                    followup: match j.get("event_patches").get("slots").arr()[i].get("followup") {
                        Json::Null => None,
                        weights => Some(weights.clone()),
                    },
                })
            })
            .collect();
        let out = Self {
            revision: p.revision,
            slots,
            foundation: legacy::Portfolio::parse(j.get("foundation"))?,
        };
        // Apply the same complete parameter validation to both halves.
        legacy::Portfolio::parse(&out.followups().json())?;
        Ok(out)
    }
}
pub struct Runtime {
    pub conditional_plans: bool,
    pub menu_reference: Option<Box<Runtime>>,
    pub batch_lifetime: bool,
    pub foundation: legacy::Runtime,
    patches: legacy::Runtime,
    followups: legacy::Runtime,
    followup_portfolio: legacy::Portfolio,
    device: i32,
    unified: Option<(Vec<usize>, Option<Policy>)>,
}
impl Runtime {
    fn event_choices(
        &self,
        c: &Controller,
        o: &Observation,
        e: &plan_events::Event,
        slot: usize,
        followup: bool,
    ) -> Result<(Vec<Choice>, Sample), String> {
        let raw = plan_events::choices(c, o, e);
        let mut row = plan_events::sample(c, o, e, &raw);
        row.context[295] = f32::from(followup);
        if let Some(anchor) = &self.menu_reference {
            if self
                .unified
                .as_ref()
                .is_some_and(|(scope, _)| scope.contains(&slot))
            {
                let index =
                    if self.conditional_plans && e.batch.is_some_and(|b| c.conditional_batch(b)) {
                        0
                    } else if followup {
                        anchor.select_followup(slot, &row)?
                    } else {
                        anchor.select(Some(slot), &row)?
                    };
                let choices = if self.conditional_plans {
                    super::plan_menu::build_conditional(c, o, e, &raw, index)?
                } else {
                    super::plan_menu::build(c, o, e, &raw, index)?
                };
                row.features = choices.iter().map(|v| v.features.clone()).collect();
                super::plan_menu::annotate_context(&mut row, o);
                return Ok((choices, row));
            }
        }
        Ok((raw, row))
    }
    pub fn select(&self, slot: Option<usize>, row: &Sample) -> Result<usize, String> {
        if let Some((scope, Some(model))) = &self.unified {
            if slot.is_some_and(|s| scope.contains(&s)) {
                return Ok(model.infer(&[row.clone()], true, &mut Rng(0))?[0].action);
            }
        }
        self.patches.select(slot, row)
    }
    fn select_followup(&self, slot: usize, row: &Sample) -> Result<usize, String> {
        if let Some((scope, model)) = &self.unified {
            if scope.contains(&slot) {
                return match model {
                    Some(p) => Ok(p.infer(&[row.clone()], true, &mut Rng(0))?[0].action),
                    None => Ok(0),
                };
            }
        }
        self.followups.select(Some(slot), row)
    }
    /// ONE network for both arrangements and revisions. Unset weights preserve
    /// the foundation; controlled exploration may still reach a same-batch event.
    pub fn unified(
        p: &Portfolio,
        scope: &[usize],
        weights: Option<&Json>,
        device: i32,
    ) -> Result<Self, String> {
        let mut runtime = Self::load(p, device)?;
        let model = if let Some(w) = weights {
            let mut model = Policy::plans(device, 0, 0.0003)?;
            model.load_weights(w)?;
            Some(model)
        } else {
            None
        };
        runtime.unified = Some((scope.to_vec(), model));
        Ok(runtime)
    }
    pub fn load(p: &Portfolio, device: i32) -> Result<Self, String> {
        let followup_portfolio = p.followups();
        Ok(Self {
            conditional_plans: false,
            menu_reference: None,
            batch_lifetime: false,
            foundation: legacy::Runtime::load(&p.foundation, device)?,
            patches: legacy::Runtime::load(&p.patches(), device)?,
            followups: legacy::Runtime::load(&followup_portfolio, device)?,
            followup_portfolio,
            device,
            unified: None,
        })
    }
    pub fn has_followup(&self, slot: usize) -> bool {
        if self
            .unified
            .as_ref()
            .is_some_and(|(scope, _)| scope.contains(&slot))
        {
            return true;
        }
        self.followup_portfolio
            .slots
            .get(slot)
            .is_some_and(Option::is_some)
    }
    /// Counterfactual collection may replace only the second choice. All
    /// ordinary scopes continue using the immutable accepted portfolio.
    pub fn override_followup(&mut self, slot: usize, weights: &Json) -> Result<(), String> {
        let candidate = self.followup_portfolio.propose(slot, 0, weights.clone())?;
        let runtime = legacy::Runtime::load(&candidate, self.device)?;
        self.followup_portfolio = candidate;
        self.followups = runtime;
        Ok(())
    }
}
#[derive(Clone, Debug)]
struct FollowupToken {
    batch: usize,
    lifetime: bool,
    revision: u64,
    not_before: i64,
}
#[derive(Clone)]
pub struct Deployed {
    pub controller: Controller,
    pub tracker: Tracker,
    foundation_cursor: legacy::Cursor,
    pub last_event: Option<plan_events::Event>,
    followups: BTreeMap<(usize, usize), FollowupToken>,
    last_slot: Option<usize>,
    last_followup: bool,
}
#[derive(Clone)]
pub struct Decision {
    pub slot: Option<usize>,
    pub followup: bool,
    pub choices: Vec<Choice>,
    pub row: Sample,
    pub selected: usize,
}
impl Deployed {
    pub fn new(c: Config) -> Self {
        Self {
            controller: Controller::new(c),
            tracker: Tracker::default(),
            foundation_cursor: legacy::Cursor::default(),
            last_event: None,
            followups: BTreeMap::new(),
            last_slot: None,
            last_followup: false,
        }
    }
    pub fn arm_followup(&mut self, slot: usize, batch: usize, step: i64) {
        self.arm_owned_followup(slot, batch, step, false);
    }
    fn arm_owned_followup(&mut self, slot: usize, batch: usize, step: i64, lifetime: bool) {
        if slot < SLOTS {
            if let Some(b) = self.controller.batches.get(batch).filter(|b| !b.cancelled) {
                self.followups.insert(
                    (slot, batch),
                    FollowupToken {
                        batch,
                        lifetime,
                        revision: b.revision,
                        not_before: step,
                    },
                );
            }
        }
    }
    pub fn cancel_followup(&mut self, slot: usize) {
        self.followups.retain(|(owner, _), _| *owner != slot);
    }
    pub fn prepare(&mut self, o: &Observation, p: &Runtime) -> Result<Option<Decision>, String> {
        self.last_slot = None;
        self.last_followup = false;
        self.controller.hold_transitions = true;
        self.controller.observe(o);
        // Daily workers expire before this observation. Run the existing hiring
        // action first and judge revisions against the next real observation,
        // rather than spending the single follow-up on a transient Keep/Cancel
        // mask. No workers or cash are assumed before the engine confirms them.
        if o.step > 0 && o.step % 24 == 0 {
            return Ok(None);
        }
        self.tracker.observe(&self.controller, o);
        self.followups.retain(|_, token| {
            self.controller.batches.get(token.batch).is_some_and(|b| {
                !b.cancelled
                    && b.revision == token.revision
                    && (!(p.batch_lifetime || token.lifetime)
                        || b.stage.links.iter().any(|id| {
                            let progress = &self.controller.progress[*id];
                            self.controller.is_active(*id)
                                && !progress.failed
                                && !progress.successor_started
                        }))
            })
        });
        // A later real event for the same batch is handled before ordinary
        // scopes, so collection and deployment consume the identical event.
        for (&(slot, batch), token) in &self.followups {
            if !p.has_followup(slot)
                || (!p.conditional_plans && self.controller.conditional_batch(batch))
            {
                continue;
            }
            let Some(event) =
                self.tracker
                    .take_for_batch(&self.controller, o, token.batch, token.not_before)
            else {
                continue;
            };
            let (choices, row) = p.event_choices(&self.controller, o, &event, slot, true)?;
            if choices.len() < 2 {
                self.tracker.defer(event);
                continue;
            }
            let selected = p.select_followup(slot, &row)?;
            self.last_event = Some(event);
            self.last_slot = Some(slot);
            self.last_followup = true;
            self.followups.remove(&(slot, batch));
            return Ok(Some(Decision {
                slot: Some(slot),
                followup: true,
                choices,
                row,
                selected,
            }));
        }
        let protected_batches: Vec<_> = self
            .followups
            .iter()
            .filter(|((slot, _), _)| p.has_followup(*slot))
            .map(|(_, token)| token.batch)
            .collect();
        for _ in 0..8 {
            let Some((event, slot)) =
                self.tracker
                    .take_excluding_batches(&self.controller, o, &protected_batches)
            else {
                break;
            };
            let (choices, row) = p.event_choices(&self.controller, o, &event, slot, false)?;
            if choices.len() < 2 {
                self.tracker.defer(event);
                continue;
            }
            let selected = p.select(Some(slot), &row)?;
            self.last_event = Some(event);
            self.last_slot = Some(slot);
            return Ok(Some(Decision {
                slot: Some(slot),
                followup: false,
                choices,
                row,
                selected,
            }));
        }
        Ok(None)
    }
    pub fn execute_choice(
        &mut self,
        o: &Observation,
        choice: Choice,
        p: &Runtime,
    ) -> Result<PlayerAction, String> {
        let previous_batches = self.controller.batches.len();
        choice.apply(&mut self.controller, o)?;
        if !choice.keep {
            self.tracker.refresh_after_edit(&self.controller, o);
        }
        if let Some(slot) = self.last_slot.take() {
            let conditional = choice.conditional
                || self
                    .last_event
                    .as_ref()
                    .and_then(|e| e.batch)
                    .is_some_and(|b| self.controller.conditional_batch(b));
            let lifetime = p.batch_lifetime || conditional;
            if self.last_followup && lifetime && p.has_followup(slot) {
                // A partial edit splits ownership: retain the untouched suffix
                // and its replacement, never acquiring an unrelated batch.
                if let Some(batch) = self.last_event.as_ref().and_then(|e| e.batch) {
                    self.arm_owned_followup(slot, batch, o.step, lifetime);
                }
                if !choice.keep
                    && choice.next.is_some()
                    && self.controller.batches.len() > previous_batches
                {
                    self.arm_owned_followup(slot, previous_batches, o.step, lifetime);
                }
            } else if !self.last_followup
                && p.has_followup(slot)
                && choice.next.is_some()
                && !choice.keep
                && self.controller.batches.len() > previous_batches
            {
                self.arm_owned_followup(slot, previous_batches, o.step, lifetime);
            }
        }
        self.continue_action(o, p)
    }
    pub fn continue_action(
        &mut self,
        o: &Observation,
        p: &Runtime,
    ) -> Result<PlayerAction, String> {
        self.controller.hold_transitions = false;
        self.controller.observe(o);
        if Agent::planning_due(o) {
            let mut cs = self.controller.proposals(o);
            let row = self.controller.sample(o, &cs);
            let slot = self.foundation_cursor.claim(o.step, cs.len() > 1);
            let selected = p.foundation.select(slot, &row)?;
            Ok(self.controller.execute_choice(o, cs.swap_remove(selected)))
        } else {
            Ok(self.controller.continue_action(o))
        }
    }
    pub fn action(&mut self, o: &Observation, p: &Runtime) -> Result<PlayerAction, String> {
        match self.prepare(o, p)? {
            Some(mut d) => self.execute_choice(o, d.choices.swap_remove(d.selected), p),
            None => self.continue_action(o, p),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning::tensor;
    use kagg_engine::{engine, state::State};
    #[test]
    fn unified_runtime_uses_same_weights_for_both_roles() {
        tensor::worker_threads();
        let p = Policy::mixed_routes(-1, 73, 0.0003).unwrap();
        let mut p = p;
        p.plan_residual = true;
        let runtime = Runtime::unified(
            &Portfolio::empty(),
            &[0, 1, 2, 3],
            Some(&p.weights_json().unwrap()),
            -1,
        )
        .unwrap();
        for role in [0., 1.] {
            let mut row = Sample {
                context: vec![0.; 320],
                features: vec![vec![0.; 32]; 3],
                ..Default::default()
            };
            row.context[295] = role;
            row.features[1][4] = 1.;
            row.features[2][7] = 1.;
            let expected = p.infer(&[row.clone()], true, &mut Rng(0)).unwrap()[0].action;
            assert_eq!(runtime.select(Some(0), &row).unwrap(), expected);
            assert_eq!(runtime.select_followup(0, &row).unwrap(), expected);
            assert_eq!(runtime.select(Some(7), &row).unwrap(), 0);
        }
    }
    #[test]
    fn empty_event_layer_preserves_foundation_actions_in_real_engine() {
        tensor::worker_threads();
        let foundation = legacy::Portfolio::empty();
        let old = legacy::Runtime::load(&foundation, -1).unwrap();
        let runtime = Runtime::load(&Portfolio::from_foundation(foundation), -1).unwrap();
        let mut a = legacy::Deployed::new(Config::default());
        let mut b = Deployed::new(Config::default());
        let mut s = State::new(29);
        for _ in 0..220 {
            let o = Observation::from_state(&s, 0);
            let before = a.action(&o, &old).unwrap();
            let after = b.action(&o, &runtime).unwrap();
            assert_eq!(
                super::super::executor::action_json(&before),
                super::super::executor::action_json(&after),
                "step {}",
                s.step
            );
            engine::step(&mut s, &[before, Default::default()]);
        }
    }
    #[test]
    fn empty_event_layer_preserves_nonempty_frozen_foundation() {
        tensor::worker_threads();
        let mut network = crate::learning::policy::Policy::mixed_routes(-1, 73, 0.0003).unwrap();
        network.plan_residual = true;
        let weights = network.weights_json().unwrap();
        let foundation = legacy::Portfolio::empty()
            .propose(0, 1, weights.clone())
            .unwrap()
            .propose(2, 2, weights.clone())
            .unwrap()
            .propose(5, 3, weights)
            .unwrap();
        let old = legacy::Runtime::load(&foundation, -1).unwrap();
        let runtime = Runtime::load(&Portfolio::from_foundation(foundation), -1).unwrap();
        let mut a = legacy::Deployed::new(Config::default());
        let mut b = Deployed::new(Config::default());
        let mut s = State::new(37);
        for _ in 0..300 {
            let o = Observation::from_state(&s, 0);
            let before = a.action(&o, &old).unwrap();
            let after = b.action(&o, &runtime).unwrap();
            assert_eq!(
                super::super::executor::action_json(&before),
                super::super::executor::action_json(&after),
                "step {}",
                s.step
            );
            engine::step(&mut s, &[before, Default::default()]);
        }
        assert!(b.controller.batches.is_empty());
        assert!(!b.controller.event_mode());
    }
    #[test]
    fn fork_and_deployment_consume_same_real_business_event() {
        tensor::worker_threads();
        let runtime = Runtime::load(&Portfolio::empty(), -1).unwrap();
        let (mut s, c) = super::super::plan_resources::tests::fixture(73);
        let mut a = Deployed::new(Config::default());
        a.controller = c;
        let o = Observation::from_state(&s, 0);
        let d = a.prepare(&o, &runtime).unwrap().unwrap();
        let slot = d.slot.unwrap();
        let mut b = a.clone();
        let alt = d
            .choices
            .into_iter()
            .find(|p| {
                !p.keep
                    && p.sites.len() == 1
                    && p.next.as_ref().is_some_and(|k| k.name() == "CARROT")
            })
            .unwrap();
        let action = a.execute_choice(&o, alt.clone(), &runtime).unwrap();
        let other = b.execute_choice(&o, alt, &runtime).unwrap();
        assert_eq!(
            super::super::executor::action_json(&action),
            super::super::executor::action_json(&other)
        );
        engine::step(&mut s, &[action, Default::default()]);
        let o = Observation::from_state(&s, 0);
        let na = a.prepare(&o, &runtime).unwrap();
        let nb = b.prepare(&o, &runtime).unwrap();
        assert_eq!(na.as_ref().map(|d| d.slot), nb.as_ref().map(|d| d.slot));
        assert_ne!(na.as_ref().and_then(|d| d.slot), Some(slot));
    }
    fn weights(seed: u64) -> Json {
        crate::learning::policy::Policy::plans(-1, seed, 0.0003)
            .unwrap()
            .weights_json()
            .unwrap()
    }
    #[test]
    fn paired_models_roundtrip_and_replace_atomically() {
        tensor::worker_threads();
        let original = Portfolio::empty().propose(1, 2, weights(31)).unwrap();
        let before = original.clone();
        let candidate = original
            .propose_paired(5, 3, weights(32), weights(33))
            .unwrap();
        assert_eq!(
            original, before,
            "a proposal must not mutate accepted behavior"
        );
        assert_eq!(candidate.slots[1], original.slots[1]);
        assert_eq!(candidate.foundation, original.foundation);
        assert_eq!(candidate.revision, original.revision + 1);
        let parsed = Portfolio::parse(&candidate.json()).unwrap();
        assert_eq!(parsed, candidate);
        assert!(Runtime::load(&parsed, -1).unwrap().has_followup(5));
        let mut broken = candidate.clone();
        broken.slots[5].as_mut().unwrap().followup = Some(Json::Num(1.));
        assert!(Portfolio::parse(&broken.json()).is_err());
    }
    #[test]
    fn scope_migration_preserves_empty_layer_and_rejects_accepted_old_scope() {
        tensor::worker_threads();
        let foundation = legacy::Portfolio::empty()
            .propose(2, 55, weights(4))
            .unwrap();
        let empty = Portfolio::from_foundation(foundation);
        let mut j = empty.json();
        j.set_path("contract", Json::Str(LEGACY_CONTRACT.into()));
        assert_eq!(Portfolio::parse(&j).unwrap(), empty);
        let accepted = empty.propose(1, 60, weights(5)).unwrap();
        let mut j = accepted.json();
        j.set_path("contract", Json::Str(LEGACY_CONTRACT.into()));
        assert!(Portfolio::parse(&j)
            .unwrap_err()
            .contains("cannot reinterpret"));
    }
    #[test]
    fn follower_override_keeps_primary_and_other_frozen_followers() {
        tensor::worker_threads();
        let primary = weights(17);
        let follower = weights(18);
        let p = Portfolio::empty()
            .propose_paired(0, 2, primary, follower.clone())
            .unwrap();
        let mut runtime = Runtime::load(&p, -1).unwrap();
        let before = runtime.followup_portfolio.slots[0].clone();
        runtime.override_followup(2, &weights(19)).unwrap();
        assert_eq!(runtime.followup_portfolio.slots[0], before);
        assert!(runtime.has_followup(0) && runtime.has_followup(2));
        let row = Sample {
            context: vec![0.; 320],
            features: vec![vec![0.; 32], vec![0.; 32]],
            ..Default::default()
        };
        assert_eq!(runtime.select(Some(2), &row).unwrap(), 0);
        assert!(runtime.override_followup(SLOTS, &follower).is_err());
    }
    #[test]
    fn conditional_batch_revisits_real_edges_but_frozen_legacy_preserves_it() {
        use super::super::{event_policy::Version, executor::Production};
        tensor::worker_threads();
        let base = Version::initial(Portfolio::empty(), vec![0, 1, 2, 3]).unwrap();
        let learner = Policy::event_plans(-1, 17, 0.0003).unwrap();
        let mut v = base.propose(1, learner.weights_json().unwrap()).unwrap();
        v.conditional_plans = true;
        v.menu_anchor = Some(Box::new(base.clone()));
        let runtime = v.runtime(-1).unwrap();
        let legacy = base.runtime(-1).unwrap();
        let (mut state, mut c) = super::super::plan_resources::tests::fixture(73);
        state.farms[0].money = 0.;
        let obs = Observation::from_state(&state, 0);
        let batch = c
            .revise_batch_mode(
                &obs,
                &[(2, 4), (3, 4)],
                Some(Production::Animal("SHEEP".into())),
                1,
                24,
                180.,
                true,
            )
            .unwrap();
        let mut a = Deployed::new(Config::default());
        a.controller = c;
        a.tracker.observe(&a.controller, &obs);
        a.arm_owned_followup(0, batch, 73, true);
        state.step = 74;
        state.farms[0].money = 2000.;
        let obs = Observation::from_state(&state, 0);
        let mut frozen = a.clone();
        assert!(frozen
            .prepare(&obs, &legacy)
            .unwrap()
            .is_none_or(|d| !d.followup));
        assert!(frozen.followups.contains_key(&(0, batch)));
        let decision = a.prepare(&obs, &runtime).unwrap().expect("funding edge");
        assert!(decision.followup && decision.choices[0].keep);
        a.execute_choice(&obs, decision.choices[0].clone(), &runtime)
            .unwrap();
        assert!(
            a.followups.contains_key(&(0, batch)),
            "Keep must preserve conditional ownership"
        );
        assert!(a
            .prepare(&obs, &runtime)
            .unwrap()
            .is_none_or(|d| !d.followup));
        a.controller.agent.executor.routes.clear();
        state.step = 75;
        state.private[0].shed.add("SHEEP", 2);
        state.private[0].shed.add("WHEAT", 4);
        let next = Observation::from_state(&state, 0);
        let d = a
            .prepare(&next, &runtime)
            .unwrap()
            .expect("confirmed material edge");
        assert!(d.followup);
        assert_eq!(a.last_event.as_ref().unwrap().batch, Some(batch));
    }
    #[test]
    fn batch_responsibility_rearms_keep_and_tracks_partial_suffixes() {
        use super::super::executor::Production;
        tensor::worker_threads();
        let mut runtime = Runtime::unified(&Portfolio::empty(), &[0, 1, 2, 3], None, -1).unwrap();
        runtime.batch_lifetime = true;
        let (mut s, mut c) = super::super::plan_resources::tests::fixture(73);
        s.farms[0].money = 0.;
        let obs = Observation::from_state(&s, 0);
        let batch = c
            .revise_batch(
                &obs,
                &[(2, 4), (3, 4)],
                Some(Production::Crop("CARROT".into())),
                1,
                24,
                180.,
            )
            .unwrap();
        let mut a = Deployed::new(Config::default());
        a.controller = c;
        a.tracker.observe(&a.controller, &obs);
        a.arm_followup(0, batch, obs.step);
        s.step = 74;
        s.farms[0].money = 2000.;
        let obs = Observation::from_state(&s, 0);
        let first = a.prepare(&obs, &runtime).unwrap().unwrap();
        assert!(first.followup);
        let mut kept = a.clone();
        kept.execute_choice(&obs, first.choices[0].clone(), &runtime)
            .unwrap();
        assert_eq!(kept.followups.len(), 1);
        assert_eq!(kept.followups[&(0, batch)].not_before, 74);
        assert!(kept
            .prepare(&obs, &runtime)
            .unwrap()
            .is_none_or(|d| !d.followup));
        let choice = first
            .choices
            .iter()
            .find(|p| p.sites.len() == 1 && p.next.as_ref().is_some_and(|p| p.name() == "WHEAT"))
            .unwrap()
            .clone();
        a.execute_choice(&obs, choice, &runtime).unwrap();
        assert_eq!(a.followups.len(), 2);
        assert!(a.followups.contains_key(&(0, batch)));
        assert!(a.followups.contains_key(&(0, batch + 1)));
        assert_eq!(a.controller.batches[batch].sites.len(), 1);
        assert_eq!(a.controller.batches[batch + 1].sites.len(), 1);
        a.controller.agent.executor.routes.clear();
        s.step = 75;
        s.market.prices.add("CARROT", 100);
        let next = Observation::from_state(&s, 0);
        let d = a.prepare(&next, &runtime).unwrap().unwrap();
        assert!(d.followup);
        assert_eq!(d.slot, Some(0));
        assert_eq!(a.last_event.as_ref().unwrap().batch, Some(batch));
        let cancel = d
            .choices
            .iter()
            .find(|p| !p.keep && p.next.is_none())
            .unwrap()
            .clone();
        a.execute_choice(&next, cancel, &runtime).unwrap();
        assert!(!a.followups.contains_key(&(0, batch)));
        assert!(a.followups.contains_key(&(0, batch + 1)));
        for id in a.controller.batches[batch + 1].stage.links.clone() {
            a.controller.progress[id].successor_started = true;
        }
        a.controller.agent.executor.routes.clear();
        s.step = 76;
        a.prepare(&Observation::from_state(&s, 0), &runtime)
            .unwrap();
        assert!(a.followups.is_empty());
    }
    #[test]
    fn primary_edit_arms_one_batch_and_keep_does_not() {
        tensor::worker_threads();
        let p = Portfolio::empty()
            .propose_paired(0, 1, weights(20), weights(21))
            .unwrap();
        let runtime = Runtime::load(&p, -1).unwrap();
        let (s, c) = super::super::plan_resources::tests::fixture(73);
        let o = Observation::from_state(&s, 0);
        let mut a = Deployed::new(Config::default());
        a.controller = c;
        let d = a.prepare(&o, &runtime).unwrap().unwrap();
        assert_eq!(d.slot, Some(0));
        assert!(!d.followup);
        let mut keep = a.clone();
        keep.execute_choice(&o, d.choices[0].clone(), &runtime)
            .unwrap();
        assert!(keep.followups.is_empty());
        let edit = d
            .choices
            .into_iter()
            .find(|v| !v.keep && v.next.is_some())
            .unwrap();
        a.execute_choice(&o, edit, &runtime).unwrap();
        assert_eq!(a.followups.len(), 1);
        assert_eq!(a.followups[&(0, 0)].batch, 0);
        assert_eq!(a.followups[&(0, 0)].not_before, o.step);
        a.cancel_followup(0);
        assert!(a.followups.is_empty());
    }
    #[test]
    fn route_delayed_followup_survives_ordinary_scope_exhaustion() {
        use super::super::executor::{unit, Production, Route, Scheduled};
        tensor::worker_threads();
        let p = Portfolio::empty()
            .propose_paired(0, 1, weights(40), weights(41))
            .unwrap();
        for (ordinary_claimed, lifetime) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let mut runtime = Runtime::load(&p, -1).unwrap();
            runtime.batch_lifetime = lifetime;
            let (mut s, mut c) = super::super::plan_resources::tests::fixture(72);
            s.farms[0].money = 0.;
            let o = Observation::from_state(&s, 0);
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
            let mut a = Deployed::new(Config::default());
            a.controller = c;
            a.tracker.observe(&a.controller, &o);
            a.arm_followup(0, batch, o.step);
            a.tracker.claimed[8] = ordinary_claimed;
            a.tracker.counts[2] = usize::from(ordinary_claimed);
            let mut route = Route::default();
            route.sites.insert((3, 4));
            route.steps.push_back(Scheduled {
                at: 90,
                position: (3, 4),
                action: unit("WATER", "", 0),
            });
            a.controller.agent.executor.assign(0, route);
            s.step = 73;
            s.farms[0].money = 2000.;
            let blocked = Observation::from_state(&s, 0);
            assert!(!a.controller.editable(&blocked, (3, 4)));
            assert!(a
                .prepare(&blocked, &runtime)
                .unwrap()
                .is_none_or(|d| !d.followup));
            assert_eq!(a.followups.len(), 1);
            // No second funding edge occurs. Releasing the real route reservation
            // must deliver the original deferred event through shared deployment.
            a.controller.agent.executor.routes.clear();
            s.step = 74;
            let released = Observation::from_state(&s, 0);
            let d = a
                .prepare(&released, &runtime)
                .unwrap()
                .expect("ordinary event budget must preserve a delayed paired decision");
            assert!(d.followup);
            assert_eq!(d.slot, Some(0));
            let event = a.last_event.as_ref().unwrap();
            assert_eq!(event.kind, plan_events::EventKind::Funding);
            assert_eq!(event.batch, Some(batch));
            assert_eq!(event.observed_step, 73);
            assert!(a.followups.is_empty());
            assert_eq!(a.tracker.claimed[8], ordinary_claimed);
            assert!(a
                .prepare(&released, &runtime)
                .unwrap()
                .is_none_or(|d| !d.followup));
        }
    }
    #[test]
    fn later_same_batch_funding_event_is_shared_bounded_and_nonrecursive() {
        tensor::worker_threads();
        let p = Portfolio::empty()
            .propose_paired(0, 1, weights(40), weights(41))
            .unwrap();
        let runtime = Runtime::load(&p, -1).unwrap();
        let (mut s, c) = super::super::plan_resources::tests::fixture(73);
        s.farms[0].money = 0.;
        let o = Observation::from_state(&s, 0);
        let mut a = Deployed::new(Config::default());
        a.controller = c;
        let first = a.prepare(&o, &runtime).unwrap().unwrap();
        let edit = first
            .choices
            .into_iter()
            .find(|v| !v.keep && v.next.as_ref().is_some_and(|p| p.name() == "CARROT"))
            .unwrap();
        // Isolate event dispatch from ordinary route occupancy: use the exact
        // shared edit operation, then let an actual cash edge produce funding.
        edit.apply(&mut a.controller, &o).unwrap();
        a.tracker.refresh_after_edit(&a.controller, &o);
        a.arm_followup(0, 0, o.step);
        assert!(
            a.prepare(&o, &runtime).unwrap().is_none_or(|d| !d.followup),
            "the initial edit is not its own later business event"
        );
        s.step += 1;
        s.farms[0].money = 2000.;
        let o = Observation::from_state(&s, 0);
        let d = a
            .prepare(&o, &runtime)
            .unwrap()
            .expect("new funding must reach the pending paired decision");
        assert!(d.followup);
        assert_eq!(d.slot, Some(0));
        assert_eq!(d.row.context[295], 1.);
        assert_eq!(a.last_event.as_ref().unwrap().batch, Some(0));
        assert!(
            a.followups.is_empty(),
            "consume exactly once even if choice stays unchanged"
        );
        let mut b = a.clone();
        let edit = d
            .choices
            .into_iter()
            .find(|v| !v.keep && v.next.as_ref().is_some_and(|p| p.name() == "TOMATO"))
            .unwrap();
        let x = a.execute_choice(&o, edit.clone(), &runtime).unwrap();
        let y = b.execute_choice(&o, edit, &runtime).unwrap();
        assert_eq!(
            super::super::executor::action_json(&x),
            super::super::executor::action_json(&y)
        );
        assert!(
            a.followups.is_empty(),
            "a follow-up cannot recursively arm another follow-up"
        );
    }
    #[test]
    fn daily_rollover_waits_for_real_hires_before_spending_followup() {
        use super::super::executor::Production;
        use super::super::plan_resources::Schedule;
        use kagg_engine::state::Cell;
        tensor::worker_threads();
        let package = Portfolio::empty()
            .propose_paired(0, 1, weights(51), weights(52))
            .unwrap();
        for (funded, lifetime) in [(true, false), (false, false), (true, true), (false, true)] {
            let mut runtime = Runtime::load(&package, -1).unwrap();
            runtime.batch_lifetime = lifetime;
            let (mut state, mut controller) = super::super::plan_resources::tests::fixture(70);
            let before = Observation::from_state(&state, 0);
            let batch = controller
                .revise_batch(
                    &before,
                    &[(3, 4)],
                    Some(Production::Crop("CARROT".into())),
                    1,
                    24,
                    180.,
                )
                .unwrap();
            // A real planted farm temporarily has only the farmer at day rollover.
            // Twenty-four crop obligations consume the 48-step service estimate.
            let plant = Cell::Plant {
                crop: "WHEAT".into(),
                planted_day: 0,
                watered_today: true,
                consecutive_unwatered: 0,
                yield_units: 0,
                max_lifespan_step: 144,
                fertilized_until_day: -1,
            };
            state.farms[0].unlocked_quadrants = ["NW", "NE", "SW", "SE"]
                .into_iter()
                .map(str::to_owned)
                .collect();
            for cell in state.farms[0].tiles.iter_mut().flatten() {
                if matches!(cell, Cell::Locked) {
                    *cell = Cell::Empty;
                }
            }
            let mut added = 0;
            for (y, row) in state.farms[0].tiles.iter_mut().enumerate() {
                for (x, cell) in row.iter_mut().enumerate() {
                    if (x, y) == (4, 4) {
                        continue;
                    }
                    if (x, y) == (3, 4) || (matches!(cell, Cell::Empty) && added < 24) {
                        *cell = plant.clone();
                        added += 1;
                    }
                }
            }
            assert!(added >= 24);
            state.step = 71;
            state.farms[0].money = 0.;
            let mut deployed = Deployed::new(Config::default());
            deployed.controller = controller;
            let before = Observation::from_state(&state, 0);
            deployed.controller.observe(&before);
            deployed.tracker.observe(&deployed.controller, &before);
            deployed.arm_followup(0, batch, before.step);
            state.step = 72;
            state.farms[0].money = if funded { 20_000. } else { 0. };
            state.market.prices.add("CARROT", 100);
            let midnight = Observation::from_state(&state, 0);
            assert_eq!(
                Schedule::build(&deployed.controller, &midnight).free_work,
                0.
            );
            let claims = deployed.tracker.claimed;
            assert!(deployed.prepare(&midnight, &runtime).unwrap().is_none());
            assert_eq!(deployed.followups.len(), 1);
            assert_eq!(deployed.tracker.claimed, claims);
            let action = deployed.continue_action(&midnight, &runtime).unwrap();
            assert_eq!(action.market.iter().any(|a| a[0] == "HIRE"), funded);
            engine::step(&mut state, &[action, Default::default()]);
            let after = Observation::from_state(&state, 0);
            assert_eq!(after.step, 73);
            assert_eq!(!after.farm.hands.is_empty(), funded);
            let decision = deployed
                .prepare(&after, &runtime)
                .unwrap()
                .expect("a real later funding/material/price event remains available");
            assert!(decision.followup);
            assert_eq!(decision.choices.iter().any(|c| c.next.is_some()), funded);
            assert!(
                decision.choices.iter().any(|c| !c.keep && c.next.is_none()),
                "persistent real shortage can still cancel the pending suffix"
            );
            assert!(deployed.followups.is_empty());
        }
    }
}

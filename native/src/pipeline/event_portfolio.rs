//! Event-scoped learning layered over an immutable accepted v3 foundation.
use super::{
    executor::Observation,
    plan_chain::Controller,
    plan_events::{self, Choice, Tracker},
    plan_portfolio as legacy,
    plan_prototype::{Agent, Config},
};
use crate::learning::policy::Sample;
use kagg_engine::{engine::PlayerAction, json::Json};
pub use legacy::{Slot, SLOTS};
pub const SCHEMA: &str = "event-plan-improvement-v4";
pub const CONTRACT: &str = plan_events::CONTRACT;
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
            slots: self.slots.clone(),
        }
    }
    pub fn propose(&self, slot: usize, iteration: u64, weights: Json) -> Result<Self, String> {
        let p = self.patches().propose(slot, iteration, weights)?;
        Ok(Self {
            revision: p.revision,
            slots: p.slots,
            foundation: self.foundation.clone(),
        })
    }
    pub fn json(&self) -> Json {
        Json::Obj(vec![
            ("contract".into(), Json::Str(CONTRACT.into())),
            ("foundation".into(), self.foundation.json()),
            ("event_patches".into(), self.patches().json()),
        ])
    }
    pub fn parse(j: &Json) -> Result<Self, String> {
        if j.get("contract").str() != CONTRACT {
            return Err("wrong event policy contract".into());
        }
        let p = legacy::Portfolio::parse(j.get("event_patches"))?;
        Ok(Self {
            revision: p.revision,
            slots: p.slots,
            foundation: legacy::Portfolio::parse(j.get("foundation"))?,
        })
    }
}
pub struct Runtime {
    pub foundation: legacy::Runtime,
    patches: legacy::Runtime,
}
impl Runtime {
    pub fn select(&self, slot: Option<usize>, row: &Sample) -> Result<usize, String> {
        self.patches.select(slot, row)
    }
    pub fn load(p: &Portfolio, device: i32) -> Result<Self, String> {
        Ok(Self {
            foundation: legacy::Runtime::load(&p.foundation, device)?,
            patches: legacy::Runtime::load(&p.patches(), device)?,
        })
    }
}
#[derive(Clone)]
pub struct Deployed {
    pub controller: Controller,
    pub tracker: Tracker,
    foundation_cursor: legacy::Cursor,
    pub last_event: Option<plan_events::Event>,
}
pub struct Decision {
    pub slot: Option<usize>,
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
        }
    }
    pub fn prepare(&mut self, o: &Observation, p: &Runtime) -> Result<Option<Decision>, String> {
        self.controller.hold_transitions = true;
        self.controller.observe(o);
        self.tracker.observe(&self.controller, o);
        for _ in 0..8 {
            let Some((event, slot)) = self.tracker.take(&self.controller, o) else {
                break;
            };
            let choices = plan_events::choices(&self.controller, o, &event);
            if choices.len() < 2 {
                self.tracker.defer(event);
                continue;
            }
            let row = plan_events::sample(&self.controller, o, &event, &choices);
            let selected = p.patches.select(Some(slot), &row)?;
            self.last_event = Some(event);
            return Ok(Some(Decision {
                slot: Some(slot),
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
        choice.apply(&mut self.controller, o)?;
        if !choice.keep {
            self.tracker.refresh_after_edit(&self.controller, o);
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
        let (mut s, c) = super::super::plan_resources::tests::fixture(72);
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
}

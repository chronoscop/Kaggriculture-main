//! A bounded collection of accepted decision replacements.
//!
//! A slot is one opportunity in a time window, not permission to change every
//! decision in that phase. Replacing a slot preserves all other accepted models
//! directly; there is no recursive call through earlier policy generations.
use super::{
    executor::Observation,
    plan_chain::{Choice, Controller},
    plan_prototype::{Agent, Config},
};
use crate::learning::policy::{Policy, Rng, Sample};
use kagg_engine::{engine::PlayerAction, json::Json};

pub const WINDOWS: [i64; 17] = [
    0, 6, 12, 24, 48, 96, 168, 240, 288, 360, 432, 504, 552, 576, 600, 624, 719,
];
pub const SLOTS: usize = WINDOWS.len() - 1;
pub const CONTRACT: &str = "accepted-plan-slots-v1";

/// Identify the decision window. Terminal steps are never claimed by Cursor.
pub fn slot_at(step: i64) -> usize {
    WINDOWS[1..]
        .iter()
        .position(|end| step < *end)
        .unwrap_or(SLOTS - 1)
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cursor {
    consumed: [bool; SLOTS],
}
impl Cursor {
    /// Claim immediately before selection, including when selection stays at
    /// the default. Clone AFTER claiming when forking so both continuations
    /// consume exactly the same intervention opportunity.
    pub fn claim(&mut self, step: i64, eligible: bool) -> Option<usize> {
        if !eligible || !(WINDOWS[0]..WINDOWS[SLOTS]).contains(&step) {
            return None;
        }
        let slot = slot_at(step);
        if self.consumed[slot] {
            None
        } else {
            self.consumed[slot] = true;
            Some(slot)
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Slot {
    pub iteration: u64,
    pub weights: Json,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Portfolio {
    pub revision: u64,
    pub slots: Vec<Option<Slot>>,
}
impl Default for Portfolio {
    fn default() -> Self {
        Self::empty()
    }
}
impl Portfolio {
    pub fn empty() -> Self {
        Self {
            revision: 0,
            slots: vec![None; SLOTS],
        }
    }
    /// Construct a full candidate policy that differs at exactly one slot.
    /// The caller commits this value only after complete-game acceptance.
    pub fn propose(&self, slot: usize, iteration: u64, weights: Json) -> Result<Self, String> {
        if slot >= SLOTS || self.slots.len() != SLOTS || !weights.is_obj() {
            return Err("invalid plan portfolio replacement".into());
        }
        let mut next = self.clone();
        next.revision = self
            .revision
            .checked_add(1)
            .ok_or("plan portfolio revision exhausted")?;
        next.slots[slot] = Some(Slot { iteration, weights });
        Ok(next)
    }
    pub fn json(&self) -> Json {
        Json::Obj(vec![
            ("contract".into(), Json::Str(CONTRACT.into())),
            ("revision".into(), Json::Str(self.revision.to_string())),
            (
                "windows".into(),
                Json::Arr(WINDOWS.iter().map(|v| Json::Num(*v as f64)).collect()),
            ),
            (
                "slots".into(),
                Json::Arr(
                    self.slots
                        .iter()
                        .map(|s| match s {
                            None => Json::Null,
                            Some(s) => Json::Obj(vec![
                                ("iteration".into(), Json::Str(s.iteration.to_string())),
                                ("weights".into(), s.weights.clone()),
                            ]),
                        })
                        .collect(),
                ),
            ),
        ])
    }
    pub fn parse(j: &Json) -> Result<Self, String> {
        let windows = Json::Arr(WINDOWS.iter().map(|v| Json::Num(*v as f64)).collect());
        if j.get("contract").str() != CONTRACT
            || *j.get("windows") != windows
            || !j.get("slots").is_arr()
            || j.get("slots").arr().len() != SLOTS
        {
            return Err("invalid plan portfolio contract/windows/slot count".into());
        }
        let revision = j
            .get("revision")
            .str()
            .parse::<u64>()
            .map_err(|_| "invalid plan portfolio revision")?;
        let mut slots = Vec::with_capacity(SLOTS);
        // Use the policy's actual parameter definitions for shape validation,
        // so checkpoint validation cannot silently diverge from the network.
        let validator = if j.get("slots").arr().iter().any(|s| *s != Json::Null) {
            Some(Policy::plans(-1, 0, 0.0003)?)
        } else {
            None
        };
        for s in j.get("slots").arr() {
            if *s == Json::Null {
                slots.push(None);
                continue;
            }
            let iteration = s
                .get("iteration")
                .str()
                .parse::<u64>()
                .map_err(|_| "invalid accepted slot iteration")?;
            let weights = s.get("weights");
            let policy = validator.as_ref().unwrap();
            validate_weights(policy, weights)?;
            slots.push(Some(Slot {
                iteration,
                weights: weights.clone(),
            }));
        }
        if revision == 0 && slots.iter().any(Option::is_some) {
            return Err("revision zero cannot contain accepted replacements".into());
        }
        Ok(Self { revision, slots })
    }
}

fn validate_weights(p: &Policy, weights: &Json) -> Result<(), String> {
    let Json::Obj(entries) = weights else {
        return Err("accepted slot requires model weights".into());
    };
    if entries.len() != p.parameters.len() {
        return Err("wrong accepted model parameter count".into());
    }
    for parameter in &p.parameters {
        let row = weights.get(parameter.name);
        let shape = row.get("shape");
        let data = row.get("data");
        if !shape.is_arr()
            || shape.arr().len() != parameter.shape.len()
            || shape
                .arr()
                .iter()
                .zip(&parameter.shape)
                .any(|(v, expected)| !v.is_num() || v.f64() != *expected as f64)
            || !data.is_arr()
            || data.arr().len() != parameter.shape.iter().product::<i64>() as usize
            || data
                .arr()
                .iter()
                .any(|v| !v.is_num() || !(v.f64() as f32).is_finite())
        {
            return Err(format!(
                "invalid accepted model parameter {}",
                parameter.name
            ));
        }
    }
    Ok(())
}

/// Inference uses exactly the accepted table also used by training rollouts.
/// No candidate model is consulted at an unclaimed opportunity.
pub struct Runtime {
    models: Vec<Option<Policy>>,
}
impl Runtime {
    pub fn load(portfolio: &Portfolio, device: i32) -> Result<Self, String> {
        if portfolio.slots.len() != SLOTS {
            return Err("invalid plan portfolio slot count".into());
        }
        let mut models = Vec::with_capacity(SLOTS);
        for slot in &portfolio.slots {
            let model = if let Some(slot) = slot {
                let mut p = Policy::plans(device, 0, 0.0003)?;
                validate_weights(&p, &slot.weights)?;
                p.load_weights(&slot.weights)?;
                Some(p)
            } else {
                None
            };
            models.push(model);
        }
        Ok(Self { models })
    }
    pub fn select(&self, slot: Option<usize>, row: &Sample) -> Result<usize, String> {
        if row.features.is_empty() {
            return Err("cannot select an empty plan candidate set".into());
        }
        let Some(slot) = slot else {
            return Ok(0);
        };
        let model = self.models.get(slot).ok_or("invalid plan slot index")?;
        if row.features.len() == 1 {
            return Ok(0);
        }
        match model {
            None => Ok(0),
            Some(p) => Ok(p.infer(std::slice::from_ref(row), true, &mut Rng(0))?[0].action),
        }
    }
}

/// The complete deployed controller state. Collection, counterfactual branches,
/// evaluation and the submission runner all use this same decision dispatch.
#[derive(Clone)]
pub struct Deployed {
    pub controller: Controller,
    pub cursor: Cursor,
}
#[derive(Clone)]
pub struct Decision {
    pub slot: Option<usize>,
    pub choices: Vec<Choice>,
    pub row: Sample,
    pub selected: usize,
}
impl Deployed {
    pub fn new(config: Config) -> Self {
        Self {
            controller: Controller::new(config),
            cursor: Cursor::default(),
        }
    }
    /// Reconcile exactly once and consume the opportunity before a caller forks.
    /// The caller must either execute this decision or continue the existing work.
    pub fn prepare(
        &mut self,
        o: &Observation,
        policy: &Runtime,
    ) -> Result<Option<Decision>, String> {
        self.controller.observe(o);
        if !Agent::planning_due(o) {
            return Ok(None);
        }
        let choices = self.controller.proposals(o);
        let row = self.controller.sample(o, &choices);
        let slot = self.cursor.claim(o.step, choices.len() > 1);
        let selected = policy.select(slot, &row)?;
        Ok(Some(Decision {
            slot,
            choices,
            row,
            selected,
        }))
    }
    pub fn action(&mut self, o: &Observation, policy: &Runtime) -> Result<PlayerAction, String> {
        match self.prepare(o, policy)? {
            Some(mut decision) => Ok(self
                .controller
                .execute_choice(o, decision.choices.swap_remove(decision.selected))),
            None => Ok(self.controller.continue_action(o)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning::{policy::CONTEXT, tensor};

    fn row() -> Sample {
        let mut features = vec![vec![0.; 32]; 2];
        features[0][31] = 1.;
        features[1][31] = 1.;
        features[1][30] = 2.; // Makes a loaded model choose 1; fallback must still choose 0.
        Sample {
            context: vec![0.; CONTEXT],
            features,
            ..Default::default()
        }
    }
    #[test]
    fn replacement_is_one_slot_and_does_not_nest_history() {
        tensor::worker_threads();
        let weights = Policy::plans(-1, 8, 0.0003)
            .unwrap()
            .weights_json()
            .unwrap();
        let a = Portfolio::empty().propose(2, 4, weights.clone()).unwrap();
        let b = a.propose(7, 9, weights.clone()).unwrap();
        let c = b.propose(2, 12, weights).unwrap();
        assert_eq!(a.revision, 1);
        assert_eq!(c.revision, 3);
        assert_eq!(c.slots.len(), SLOTS);
        assert_eq!(c.slots.iter().filter(|s| s.is_some()).count(), 2);
        assert_eq!(c.slots[2].as_ref().unwrap().iteration, 12);
        assert_eq!(a.slots[2].as_ref().unwrap().iteration, 4);
        for slot in 0..SLOTS {
            if slot != 2 {
                assert_eq!(b.slots[slot], c.slots[slot]);
            }
        }
    }
    #[test]
    fn checkpoint_validates_windows_and_actual_model_shapes() {
        tensor::worker_threads();
        let weights = Policy::plans(-1, 8, 0.0003)
            .unwrap()
            .weights_json()
            .unwrap();
        let a = Portfolio::empty().propose(0, 4, weights.clone()).unwrap();
        assert_eq!(a, Portfolio::parse(&a.json()).unwrap());
        let mut wrong = a.json();
        wrong.set_path("windows", Json::Arr(vec![Json::Num(0.)]));
        assert!(Portfolio::parse(&wrong).is_err());
        let mut malformed = weights;
        let Json::Obj(ref mut parameters) = malformed else {
            unreachable!()
        };
        parameters[0]
            .1
            .set_path("shape", Json::Arr(vec![Json::Num(1.)]));
        let wrong = Portfolio::empty().propose(0, 4, malformed).unwrap();
        assert!(Portfolio::parse(&wrong.json()).is_err());
        assert!(Runtime::load(&wrong, -1).is_err());
    }
    #[test]
    fn cursor_consumes_once_and_clone_preserves_the_consumed_fork() {
        let mut c = Cursor::default();
        assert_eq!(c.claim(0, false), None);
        assert_eq!(c.claim(2, true), Some(0));
        let mut branch = c.clone();
        assert_eq!(branch.claim(3, true), None);
        assert_eq!(c.claim(5, true), None);
        assert_eq!(c.claim(6, true), Some(1));
        assert_eq!(branch.claim(6, true), Some(1));
        assert_eq!(c.claim(624, true), Some(15));
        assert_eq!(c.claim(718, true), None);
        assert_eq!(c.claim(719, true), None);
        assert_eq!(c.claim(-1, true), None);
    }
    #[test]
    fn runtime_only_uses_the_claimed_accepted_slot() {
        tensor::worker_threads();
        let row = row();
        let empty = Runtime::load(&Portfolio::empty(), -1).unwrap();
        assert_eq!(empty.select(Some(0), &row).unwrap(), 0);
        let weights = Policy::plans(-1, 1, 0.0003)
            .unwrap()
            .weights_json()
            .unwrap();
        let portfolio = Portfolio::empty().propose(3, 4, weights).unwrap();
        let runtime = Runtime::load(&portfolio, -1).unwrap();
        assert_eq!(runtime.select(None, &row).unwrap(), 0);
        assert_eq!(runtime.select(Some(2), &row).unwrap(), 0);
        assert_eq!(runtime.select(Some(3), &row).unwrap(), 1);
        assert!(runtime.select(Some(SLOTS), &row).is_err());
    }
}

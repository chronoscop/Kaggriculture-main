//! Stable, complete event policy versions. The original accepted foundation is
//! immutable; one shared network replaces the declared event scope as a whole.
use super::event_portfolio::{Portfolio, Runtime, SLOTS};
use kagg_engine::json::Json;
pub const SCHEMA: &str = "event-policy-iteration-v9";
pub const PREFIX_SCHEMA: &str = "event-policy-iteration-v8";
pub const MENU_CONTRACT: &str = "event-complete-menu-single-v1";
pub const MENU_BATCH_CONTRACT: &str = "event-complete-menu-batch-v1";
pub const NORMALIZED_SCHEMA: &str = "event-policy-iteration-v7";
pub const EVIDENCE_SCHEMA: &str = "event-policy-iteration-v6";
pub const LEGACY_SCHEMA: &str = "event-policy-iteration-v5";
pub const CONTRACT: &str = "event-shared-policy-season-scope-v5";
pub const BATCH_CONTRACT: &str = "event-shared-policy-batch-lineage-v8";
pub const COLLECTION_CONTRACT: &str = "complete-executable-menu-v1";
#[derive(Clone, Debug, PartialEq)]
pub struct Version {
    pub revision: u64,
    pub iteration: u64,
    pub foundation: Portfolio,
    pub scope: Vec<usize>,
    pub weights: Option<Json>,
    pub batch_lifetime: bool,
    pub menu_anchor: Option<Box<Version>>,
}
impl Version {
    pub fn initial(foundation: Portfolio, scope: Vec<usize>) -> Result<Self, String> {
        validate_scope(&scope)?;
        // An old nonempty event layer has distinct accepted execution semantics.
        // Do not silently remap it into a shared-network scope.
        if foundation.revision != 0 || foundation.slots.iter().any(Option::is_some) {
            return Err("cannot import accepted paired event patches into a shared policy; keep using the original deployment".into());
        }
        Ok(Self {
            revision: 0,
            iteration: 0,
            foundation,
            scope,
            weights: None,
            batch_lifetime: false,
            menu_anchor: None,
        })
    }
    pub fn propose(&self, iteration: u64, weights: Json) -> Result<Self, String> {
        if !weights.is_obj() {
            return Err("missing shared policy weights".into());
        }
        let mut v = self.clone();
        v.revision = self.revision.checked_add(1).ok_or("revision exhausted")?;
        v.iteration = iteration;
        v.weights = Some(weights);
        Ok(v)
    }
    pub fn validate_successor(&self, next: &Self) -> Result<(), String> {
        if next.revision != self.revision + 1
            || next.foundation != self.foundation
            || next.scope != self.scope
            || next.weights.is_none()
            || (self.batch_lifetime && !next.batch_lifetime)
            || (self.menu_anchor.is_some() && self.menu_anchor != next.menu_anchor)
            || (self.menu_anchor.is_none()
                && next.menu_anchor.as_deref().is_some_and(|a| a != self))
        {
            return Err(
                "promotion must replace the complete shared policy inside the SAME declared scope"
                    .into(),
            );
        }
        Ok(())
    }
    pub fn runtime(&self, device: i32) -> Result<Runtime, String> {
        let mut runtime =
            Runtime::unified(&self.foundation, &self.scope, self.weights.as_ref(), device)?;
        runtime.batch_lifetime = self.batch_lifetime;
        runtime.menu_reference = self
            .menu_anchor
            .as_ref()
            .map(|v| v.runtime(device).map(Box::new))
            .transpose()?;
        Ok(runtime)
    }
    pub fn contract(&self) -> &'static str {
        if self.menu_anchor.is_some() {
            if self.batch_lifetime {
                MENU_BATCH_CONTRACT
            } else {
                MENU_CONTRACT
            }
        } else if self.batch_lifetime {
            BATCH_CONTRACT
        } else {
            CONTRACT
        }
    }
    pub fn json(&self) -> Json {
        Json::Obj(vec![
            ("contract".into(), Json::Str(self.contract().into())),
            ("revision".into(), Json::Str(self.revision.to_string())),
            ("iteration".into(), Json::Str(self.iteration.to_string())),
            ("foundation".into(), self.foundation.json()),
            (
                "scope".into(),
                Json::Arr(self.scope.iter().map(|s| Json::Num(*s as f64)).collect()),
            ),
            ("weights".into(), self.weights.clone().unwrap_or(Json::Null)),
            (
                "menu_anchor".into(),
                self.menu_anchor
                    .as_ref()
                    .map(|v| v.json())
                    .unwrap_or(Json::Null),
            ),
        ])
    }
    pub fn parse(j: &Json) -> Result<Self, String> {
        if !matches!(
            j.get("contract").str(),
            CONTRACT | BATCH_CONTRACT | MENU_CONTRACT | MENU_BATCH_CONTRACT
        ) {
            return Err("wrong stable event contract".into());
        }
        let scope: Vec<_> = j
            .get("scope")
            .arr()
            .iter()
            .map(|x| x.i64() as usize)
            .collect();
        validate_scope(&scope)?;
        let foundation = Portfolio::parse(j.get("foundation"))?;
        Self::initial(foundation.clone(), scope.clone())?;
        let menu_contract = matches!(j.get("contract").str(), MENU_CONTRACT | MENU_BATCH_CONTRACT);
        let menu_anchor = if menu_contract {
            let a = j.get("menu_anchor");
            if !matches!(a.get("contract").str(), CONTRACT | BATCH_CONTRACT) {
                return Err("menu anchor must be frozen legacy policy".into());
            }
            let a = Box::new(Self::parse(a)?);
            if a.foundation != foundation || a.scope != scope {
                return Err("menu anchor scope mismatch".into());
            }
            Some(a)
        } else {
            if !matches!(j.get("menu_anchor"), Json::Null) {
                return Err("legacy version cannot enable menu".into());
            }
            None
        };
        let v = Self {
            menu_anchor,
            revision: j
                .get("revision")
                .str()
                .parse()
                .map_err(|_| "bad revision")?,
            iteration: j
                .get("iteration")
                .str()
                .parse()
                .map_err(|_| "bad iteration")?,
            scope,
            foundation,
            batch_lifetime: matches!(
                j.get("contract").str(),
                BATCH_CONTRACT | MENU_BATCH_CONTRACT
            ),
            weights: match j.get("weights") {
                Json::Null => None,
                w => Some(w.clone()),
            },
        };
        if (v.revision == 0) != v.weights.is_none()
            || ((v.batch_lifetime || menu_contract) && v.weights.is_none())
        {
            return Err("accepted revision/weights mismatch".into());
        }
        v.runtime(-1)?; // Validate all tensor shapes/values on restore.
        Ok(v)
    }
}
pub fn validate_scope(scope: &[usize]) -> Result<(), String> {
    if scope.is_empty()
        || scope.iter().any(|s| *s >= SLOTS)
        || scope.windows(2).any(|w| w[0] >= w[1])
    {
        return Err("scope must be sorted, distinct event ids in 0..16".into());
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        learning::{
            policy::{Policy, Sample},
            tensor,
        },
        pipeline::{
            event_portfolio::Deployed,
            executor::{action_json, Observation},
            plan_prototype::Config,
        },
    };
    #[test]
    fn complete_menu_zero_head_preserves_legacy_actions_and_frozen_anchor() {
        tensor::worker_threads();
        let base = Version::initial(Portfolio::empty(), vec![0, 1, 2, 3]).unwrap();
        let p = Policy::event_plans(-1, 17, 0.0003).unwrap();
        let mut menu = base.propose(1, p.weights_json().unwrap()).unwrap();
        menu.menu_anchor = Some(Box::new(base.clone()));
        base.validate_successor(&menu).unwrap();
        assert_eq!(Version::parse(&menu.json()).unwrap(), menu);
        let old = base.runtime(-1).unwrap();
        let new = menu.runtime(-1).unwrap();
        let mut a = Deployed::new(Config::default());
        let mut b = a.clone();
        let mut game = kagg_engine::state::State::new(37);
        let mut events = 0;
        while game.step < 719 {
            let obs = Observation::from_state(&game, 0);
            let x = a.action(&obs, &old).unwrap();
            let y = if let Some(mut d) = b.prepare(&obs, &new).unwrap() {
                if d.slot.is_some_and(|s| menu.scope.contains(&s)) {
                    events += 1;
                    assert!(d.choices.len() <= 4);
                    assert_eq!(d.selected, 0);
                    let event = b.last_event.as_ref().unwrap();
                    let raw = crate::pipeline::plan_events::choices(&b.controller, &obs, event);
                    let old_row =
                        crate::pipeline::plan_events::sample(&b.controller, &obs, event, &raw);
                    assert_eq!(&d.row.context[283..294], &old_row.context[283..294]);
                    assert_eq!(d.row.context[294], obs.seat as f32);
                    assert!(d.row.features.iter().flatten().all(|x| x.is_finite()));
                }
                b.execute_choice(&obs, d.choices.swap_remove(d.selected), &new)
                    .unwrap()
            } else {
                b.continue_action(&obs, &new).unwrap()
            };
            assert_eq!(action_json(&x), action_json(&y), "step {}", game.step);
            kagg_engine::engine::step(&mut game, &[x, Default::default()]);
        }
        assert!(events > 0);
        let mut bad = menu.propose(2, p.weights_json().unwrap()).unwrap();
        bad.menu_anchor = None;
        assert!(menu.validate_successor(&bad).is_err());
    }
    #[test]
    fn initial_version_preserves_accepted_foundation_actions() {
        tensor::worker_threads();
        let base = Portfolio::empty();
        let old = Runtime::load(&base, -1).unwrap();
        let v = Version::initial(base, vec![0, 1, 2, 3]).unwrap();
        let new = v.runtime(-1).unwrap();
        let mut a = Deployed::new(Config::default());
        let mut b = a.clone();
        let mut game = kagg_engine::state::State::new(37);
        for _ in 0..300 {
            let obs = Observation::from_state(&game, 0);
            let x = a.action(&obs, &old).unwrap();
            let y = b.action(&obs, &new).unwrap();
            assert_eq!(action_json(&x), action_json(&y), "step {}", game.step);
            kagg_engine::engine::step(&mut game, &[x, Default::default()]);
        }
    }
    #[test]
    fn complete_model_replacement_is_bounded_and_roundtrips() {
        tensor::worker_threads();
        let p = Policy::plans(-1, 3, 0.0003).unwrap();
        let base = Version::initial(Portfolio::empty(), vec![0, 1, 2, 3]).unwrap();
        let candidate = base.propose(7, p.weights_json().unwrap()).unwrap();
        base.validate_successor(&candidate).unwrap();
        assert_eq!(Version::parse(&candidate.json()).unwrap(), candidate);
        let runtime = candidate.runtime(-1).unwrap();
        let mut row = Sample {
            context: vec![0.; 320],
            features: vec![vec![0.; 32]; 2],
            ..Default::default()
        };
        row.features[1][30] = 2.;
        row.features[0][31] = 1.;
        row.features[1][31] = 1.;
        assert_eq!(runtime.select(Some(0), &row).unwrap(), 1);
        assert_eq!(runtime.select(Some(5), &row).unwrap(), 0);
        assert!(runtime.has_followup(0));
        assert!(!runtime.has_followup(5));
        let mut invalid = candidate.clone();
        invalid.scope.push(5);
        assert!(base.validate_successor(&invalid).is_err());
        assert!(base.weights.is_none());
    }
}

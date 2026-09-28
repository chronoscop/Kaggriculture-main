//! Plan-specific supervised warm start, protected replay, and full-update checks.
use super::{
    policy::{Batch, Policy, Rng, Sample},
    tensor::{NoGrad, Tensor},
};
use kagg_engine::json::Json;

#[derive(Clone)]
pub struct Anchor {
    pub row: Sample,
    pub probabilities: Vec<f32>,
    pub score: f64,
    pub tier: u8,
    pub opponent: usize,
}
impl Anchor {
    pub fn json(&self) -> Json {
        Json::Obj(vec![
            ("row".into(), self.row.json()),
            (
                "probabilities".into(),
                Json::Arr(
                    self.probabilities
                        .iter()
                        .map(|x| Json::Num(*x as f64))
                        .collect(),
                ),
            ),
            ("score".into(), Json::Num(self.score)),
            ("tier".into(), Json::Num(self.tier as f64)),
            ("opponent".into(), Json::Num(self.opponent as f64)),
        ])
    }
    pub fn parse(j: &Json) -> Result<Self, String> {
        let row = Sample::parse(j.get("row"))?;
        let probabilities: Vec<f32> = j
            .get("probabilities")
            .arr()
            .iter()
            .map(|x| x.f64() as f32)
            .collect();
        if probabilities.len() != row.features.len()
            || probabilities.iter().any(|x| !x.is_finite() || *x < 0.)
            || (probabilities.iter().sum::<f32>() - 1.).abs() > 1e-3
        {
            return Err("invalid protected distribution".into());
        }
        Ok(Self {
            row,
            probabilities,
            score: j.get("score").f64(),
            tier: j.get("tier").i64() as u8,
            opponent: j.get("opponent").i64() as usize,
        })
    }
}
impl Policy {
    pub fn distributions(&self, rows: &[Sample]) -> Result<Vec<Vec<f32>>, String> {
        if rows.is_empty() {
            return Ok(vec![]);
        }
        let _g = NoGrad::new();
        let b = Batch::new(rows, self.device)?;
        let data = self.forward(&b)?.0.unary(16)?.data()?;
        Ok(rows
            .iter()
            .enumerate()
            .map(|(i, r)| data[i * b.width..i * b.width + r.features.len()].to_vec())
            .collect())
    }
    pub fn distill_plans(
        &mut self,
        rows: &[Sample],
        targets: &[Vec<f32>],
        weight: f64,
    ) -> Result<f64, String> {
        if rows.is_empty() || weight == 0. {
            return Ok(0.);
        }
        if rows.len() != targets.len() {
            return Err("distillation row mismatch".into());
        }
        let b = Batch::new(rows, self.device)?;
        let mut flat = vec![0.; rows.len() * b.width];
        for (i, (row, t)) in rows.iter().zip(targets).enumerate() {
            if t.len() != row.features.len()
                || t.iter().any(|x| !x.is_finite() || *x < 0.)
                || (t.iter().sum::<f32>() - 1.).abs() > 1e-3
            {
                return Err("invalid distillation target".into());
            }
            flat[i * b.width..i * b.width + t.len()].copy_from_slice(t);
        }
        let target = Tensor::floats(
            &flat,
            &[rows.len() as i64, b.width as i64],
            self.device,
            false,
        )?;
        let lp = self.forward(&b)?.0;
        let loss = lp
            .binary(14, &target)?
            .dim(17, -1)?
            .unary(18)?
            .scalar(31, -weight)?;
        let scalar = loss.value()?;
        if !scalar.is_finite() {
            return Err("nonfinite plan distillation".into());
        }
        for p in &mut self.parameters {
            p.value.zero_grad();
        }
        loss.backward()?;
        self.adam()?;
        Ok(scalar)
    }
}
pub fn divergence(before: &[Vec<f32>], after: &[Vec<f32>]) -> (f64, f64) {
    if before.is_empty() {
        return (0., 0.);
    }
    let mut kl = 0.;
    let mut changes = 0.;
    let argmax = |v: &Vec<f32>| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .unwrap()
            .0
    };
    for (a, b) in before.iter().zip(after) {
        kl += a
            .iter()
            .zip(b)
            .filter(|(p, _)| **p > 0.)
            .map(|(p, q)| *p as f64 * ((*p as f64).ln() - (*q as f64).max(1e-30).ln()))
            .sum::<f64>();
        changes += f64::from(argmax(a) != argmax(b));
    }
    (kl / before.len() as f64, changes / before.len() as f64)
}

pub struct ResultMetrics {
    pub updates: usize,
    pub loss: f64,
    pub kl: f64,
    pub anchor_kl: f64,
    pub anchor_changes: f64,
    pub reverted: bool,
}
/// Replay gets its own actor-only update after EVERY PPO minibatch. Final checks
/// include both types of updates. A rejection restores Adam moments and weights.
pub fn update(
    policy: &mut Policy,
    rows: &[Sample],
    bank: &[Anchor],
    epochs: usize,
    batch_size: usize,
    rng: &mut Rng,
    anchor_weight: f64,
    max_kl: f64,
) -> Result<ResultMetrics, String> {
    let checkpoint = policy.checkpoint(0, rng)?;
    let old = policy.distributions(rows)?;
    let anchors: Vec<_> = bank.iter().map(|a| a.row.clone()).collect();
    let targets: Vec<_> = bank.iter().map(|a| a.probabilities.clone()).collect();
    let before_anchor = policy.distributions(&anchors)?;
    let old_anchor_kl = divergence(&targets, &before_anchor).0;
    let mut ids: Vec<_> = (0..rows.len()).collect();
    let mut updates = 0;
    let mut loss = 0.;
    'epochs: for _ in 0..epochs {
        rng.shuffle(&mut ids);
        for chunk in ids.chunks(batch_size) {
            let samples: Vec<_> = chunk.iter().map(|i| rows[*i].clone()).collect();
            let m = policy.update(&samples, 1, batch_size, rng)?;
            updates += m.updates;
            loss += m.loss * m.updates as f64;
            if !bank.is_empty() {
                let start = (rng.next() as usize) % bank.len();
                let ids: Vec<_> = (0..bank.len().min(64))
                    .map(|i| (start + i) % bank.len())
                    .collect();
                policy.distill_plans(
                    &ids.iter().map(|i| anchors[*i].clone()).collect::<Vec<_>>(),
                    &ids.iter().map(|i| targets[*i].clone()).collect::<Vec<_>>(),
                    anchor_weight,
                )?;
            }
            if m.kl_stopped {
                break 'epochs;
            }
        }
    }
    let (kl, _) = divergence(&old, &policy.distributions(rows)?);
    let after_anchor = policy.distributions(&anchors)?;
    let (anchor_kl, _) = divergence(&targets, &after_anchor);
    let (_, anchor_changes) = divergence(&before_anchor, &after_anchor);
    let reverted = !kl.is_finite()
        || !anchor_kl.is_finite()
        || kl > max_kl
        || anchor_kl > old_anchor_kl + max_kl
        || anchor_changes > 0.20;
    if reverted {
        policy.restore(&checkpoint)?;
    }
    Ok(ResultMetrics {
        updates,
        loss: loss / updates.max(1) as f64,
        kl,
        anchor_kl,
        anchor_changes,
        reverted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn residual_prior_has_exact_likelihood_and_can_be_overridden_by_learning() {
        super::super::tensor::threads(1);
        let mut p = Policy::plans(-1, 43, 0.01).unwrap();
        let mut features = vec![vec![0.; 32]; 2];
        features[0][30] = 3f32.ln();
        features[0][31] = 1.;
        features[1][31] = 1.;
        features[1][1] = 1.;
        let row = Sample {
            context: vec![0.; super::super::policy::CONTEXT],
            features,
            ..Default::default()
        };
        let probs = p.distributions(&[row.clone()]).unwrap();
        assert!((probs[0][0] - 0.75).abs() < 1e-6);
        let d = p
            .infer(&[row.clone()], false, &mut Rng(0))
            .unwrap()
            .remove(0);
        assert!((d.logp.exp() - probs[0][d.action]).abs() < 1e-6);
        for _ in 0..30 {
            p.distill_plans(&[row.clone()], &[vec![0., 1.]], 1.)
                .unwrap();
        }
        assert_eq!(p.infer(&[row], true, &mut Rng(0)).unwrap()[0].action, 1);
        let checkpoint = p.checkpoint(1, &Rng(0)).unwrap();
        let mut restored = Policy::plans(-1, 1, 1e-4).unwrap();
        restored.restore(&checkpoint).unwrap();
        assert!(restored.plan_residual);
    }
    #[test]
    fn supervised_updates_actor_and_guard_restores_complete_optimizer() {
        super::super::tensor::threads(1);
        let mut p = Policy::mixed_routes(-1, 81, 1e-3).unwrap();
        let mut f = vec![vec![0.; 32]; 2];
        f[0][1] = 1.;
        f[1][2] = 1.;
        f[0][31] = 1.;
        f[1][31] = 1.;
        let row = Sample {
            context: vec![0.; super::super::policy::CONTEXT],
            features: f,
            ..Default::default()
        };
        let before = p.distributions(&[row.clone()]).unwrap()[0][1];
        for _ in 0..5 {
            p.distill_plans(&[row.clone()], &[vec![0., 1.]], 1.)
                .unwrap();
        }
        assert!(p.distributions(&[row.clone()]).unwrap()[0][1] > before);
        let mut rng = Rng(9);
        let ck = p.checkpoint(0, &rng).unwrap();
        let d = p.infer(&[row.clone()], false, &mut rng).unwrap().remove(0);
        let row = Sample {
            action: d.action,
            logp: d.logp,
            value: d.value,
            reward: 1.,
            advantage: 1.,
            ..row
        };
        let m = update(&mut p, &[row], &[], 2, 1, &mut rng, 0., 0.).unwrap();
        assert!(m.reverted);
        let restored = p.checkpoint(0, &Rng(9)).unwrap();
        assert_eq!(ck, restored);
    }
}

//! Same-state, same-continuation pairwise policy improvement. No whole-game PPO labels.
use super::{
    plan_update::divergence,
    policy::{Batch, Policy, Rng, Sample},
    tensor::{NoGrad, Tensor},
};
use kagg_engine::json::Json;
#[derive(Clone)]
pub struct Pair {
    /// Exactly two TESTED choices; untested choices receive no negative labels.
    pub row: Sample,
    pub target: Vec<f32>,
    pub gain: f64,
    pub iteration: u64,
    pub seed: i64,
    pub seat: usize,
    pub opponent: usize,
    pub bucket: usize,
    pub evidence: Json,
}
impl Pair {
    pub fn incumbent_revision(&self) -> Result<u64, String> {
        evidence_integer(
            self.evidence.get("incumbent_revision"),
            "incumbent_revision",
        )
    }
    pub fn matches_revision(&self, revision: u64) -> Result<bool, String> {
        Ok(self.incumbent_revision()? == revision)
    }
    pub fn improvement_target(&self) -> Result<f32, String> {
        improvement_target(
            evidence_cash(&self.evidence, "reference_cash")?,
            evidence_cash(&self.evidence, "alternative_cash")?,
        )
    }
    pub fn json(&self) -> Json {
        Json::Obj(vec![
            ("row".into(), self.row.json()),
            (
                "target".into(),
                Json::Arr(self.target.iter().map(|v| Json::Num(*v as f64)).collect()),
            ),
            ("gain".into(), Json::Num(self.gain)),
            ("iteration".into(), Json::Str(self.iteration.to_string())),
            ("seed".into(), Json::Num(self.seed as f64)),
            ("seat".into(), Json::Num(self.seat as f64)),
            ("opponent".into(), Json::Num(self.opponent as f64)),
            ("bucket".into(), Json::Num(self.bucket as f64)),
            ("evidence".into(), self.evidence.clone()),
        ])
    }
    pub fn parse(j: &Json) -> Result<Self, String> {
        let p = Self {
            row: Sample::parse(j.get("row"))?,
            target: j
                .get("target")
                .arr()
                .iter()
                .map(|x| x.f64() as f32)
                .collect(),
            gain: j.get("gain").f64(),
            iteration: j
                .get("iteration")
                .str()
                .parse()
                .map_err(|_| "invalid pair iteration")?,
            seed: j.get("seed").i64(),
            seat: j.get("seat").i64() as usize,
            opponent: j.get("opponent").i64() as usize,
            bucket: j.get("bucket").i64() as usize,
            evidence: j.get("evidence").clone(),
        };
        if p.row.features.len() != 2
            || p.target.len() != 2
            || p.target.iter().any(|v| !v.is_finite() || *v < 0.)
            || (p.target.iter().sum::<f32>() - 1.).abs() > 1e-5
            || !p.gain.is_finite()
            || p.gain < 0.
            || p.bucket >= 81
            || p.seat > 1
            || p.opponent > 2
        {
            return Err("invalid comparison sample".into());
        }
        Ok(p)
    }
}
/// Win/draw/loss is primary. For the same outcome, use actual terminal relative
/// cash margin; differences below 0.2% are inconclusive, not success labels.
pub fn preference(a: [f64; 2], b: [f64; 2]) -> Option<(Vec<f32>, f64)> {
    let score = |v: [f64; 2]| {
        if v[0] > v[1] {
            1.
        } else if v[0] < v[1] {
            0.
        } else {
            0.5
        }
    };
    let margin = |v: [f64; 2]| (v[0] - v[1]) / (v[0].abs() + v[1].abs()).max(1.);
    let delta_score: f64 = score(b) - score(a);
    let delta = margin(b) - margin(a);
    if delta_score == 0. && delta.abs() < 0.002 {
        return None;
    }
    let better = if delta_score != 0. {
        delta_score > 0.
    } else {
        delta > 0.
    };
    Some((
        if better {
            vec![0.1, 0.9]
        } else {
            vec![0.9, 0.1]
        },
        delta_score.abs() + delta.abs().min(1.),
    ))
}
#[derive(Clone, Default)]
pub struct Bank {
    pub elite: Vec<Pair>,
    pub recent: Vec<Pair>,
}
impl Bank {
    /// A comparison is conditional on the continuation policy. Old revisions
    /// remain audit evidence, but are not labels for the next policy revision.
    pub fn for_revision(&self, revision: u64) -> Result<Self, String> {
        let filter = |pairs: &[Pair]| -> Result<Vec<Pair>, String> {
            let mut out = Vec::new();
            for pair in pairs {
                if pair.matches_revision(revision)? {
                    out.push(pair.clone());
                }
            }
            Ok(out)
        };
        Ok(Self {
            elite: filter(&self.elite)?,
            recent: filter(&self.recent)?,
        })
    }
    pub fn admit(&mut self, pairs: &[Pair]) {
        for p in pairs {
            // Fixed, finite phase/opponent/successor buckets have protected quotas.
            // A novel business label cannot evict a different bucket's history.
            let ids: Vec<_> = self
                .elite
                .iter()
                .enumerate()
                .filter(|(_, v)| v.bucket == p.bucket)
                .map(|(i, _)| i)
                .collect();
            let duplicate = ids.iter().copied().find(|i| {
                self.elite[*i].seed == p.seed
                    && self.elite[*i].seat == p.seat
                    && self.elite[*i].row.step == p.row.step
            });
            if let Some(i) = duplicate {
                if p.gain > self.elite[i].gain {
                    self.elite[i] = p.clone();
                }
            } else if ids.len() < 4 {
                self.elite.push(p.clone());
            } else {
                let worst = *ids
                    .iter()
                    .min_by(|a, b| self.elite[**a].gain.total_cmp(&self.elite[**b].gain))
                    .unwrap();
                if p.gain > self.elite[worst].gain {
                    self.elite[worst] = p.clone();
                }
            }
            self.recent.push(p.clone());
        }
        if self.recent.len() > 512 {
            self.recent.drain(..self.recent.len() - 512);
        }
    }
    pub fn json(&self) -> Json {
        Json::Obj(vec![
            (
                "elite".into(),
                Json::Arr(self.elite.iter().map(Pair::json).collect()),
            ),
            (
                "recent".into(),
                Json::Arr(self.recent.iter().map(Pair::json).collect()),
            ),
        ])
    }
    pub fn parse(j: &Json) -> Result<Self, String> {
        if !j.get("elite").is_arr() || !j.get("recent").is_arr() {
            return Err("missing comparison bank".into());
        }
        let b = Self {
            elite: j
                .get("elite")
                .arr()
                .iter()
                .map(Pair::parse)
                .collect::<Result<_, _>>()?,
            recent: j
                .get("recent")
                .arr()
                .iter()
                .map(Pair::parse)
                .collect::<Result<_, _>>()?,
        };
        if b.recent.len() > 512
            || (0..81).any(|slot| b.elite.iter().filter(|p| p.bucket == slot).count() > 4)
        {
            return Err("comparison bank quota exceeded".into());
        }
        Ok(b)
    }
}
fn evidence_integer(j: &Json, name: &str) -> Result<u64, String> {
    match j {
        Json::Str(v) => v.parse().map_err(|_| format!("invalid comparison {name}")),
        Json::Num(v) if v.is_finite() && *v >= 0. && *v < u64::MAX as f64 && v.fract() == 0. => {
            Ok(*v as u64)
        }
        _ => Err(format!("missing or invalid comparison {name}")),
    }
}
fn evidence_cash(evidence: &Json, name: &str) -> Result<[f64; 2], String> {
    match evidence.get(name) {
        Json::Arr(v) if v.len() == 2 => {
            let mut cash = [0.; 2];
            for (i, value) in v.iter().enumerate() {
                match value {
                    Json::Num(n) if n.is_finite() => cash[i] = *n,
                    _ => return Err(format!("non-numeric or nonfinite {name}[{i}]")),
                }
            }
            Ok(cash)
        }
        _ => Err(format!("missing or invalid comparison {name}")),
    }
}
/// Terminal utility difference, not a label based on production counts. Cash is
/// learner-first in BOTH branches. The bounded margin term cannot reverse a
/// win/draw/loss improvement, even with negative cash or very large numbers.
pub fn improvement_target(reference: [f64; 2], alternative: [f64; 2]) -> Result<f32, String> {
    if reference
        .iter()
        .chain(alternative.iter())
        .any(|v| !v.is_finite())
    {
        return Err("nonfinite terminal cash in improvement comparison".into());
    }
    let score = |v: [f64; 2]| {
        if v[0] > v[1] {
            1.
        } else if v[0] < v[1] {
            0.
        } else {
            0.5
        }
    };
    let margin = |v: [f64; 2]| {
        // Algebraically (a-b)/max(abs(a)+abs(b),1), without overflowing.
        let scale = v[0].abs().max(v[1].abs()).max(1.);
        let a = v[0] / scale;
        let b = v[1] / scale;
        (a - b) / (a.abs() + b.abs()).max(1. / scale)
    };
    Ok(
        (score(alternative) - score(reference) + 0.1 * (margin(alternative) - margin(reference)))
            as f32,
    )
}
fn validate_improvement(pair: &Pair, revision: u64) -> Result<f32, String> {
    if !pair.matches_revision(revision)? {
        return Err("fresh comparison uses a different incumbent revision".into());
    }
    if pair.row.features.len() != 2 {
        return Err("improvement regression requires exactly two tested choices".into());
    }
    // Validate in-memory rows too: collectors do not necessarily parse JSON.
    Sample::parse(&pair.row.json())?;
    pair.improvement_target()
}
fn regression_loss(p: &Policy, pairs: &[Pair]) -> Result<Tensor, String> {
    let rows: Vec<_> = pairs.iter().map(|r| r.row.clone()).collect();
    let targets: Vec<_> = pairs
        .iter()
        .map(Pair::improvement_target)
        .collect::<Result<_, _>>()?;
    let b = Batch::new(&rows, p.device)?;
    let lp = p.forward(&b)?.0;
    // Normalization cancels: log p(alternative)-log p(reference) is exactly
    // their network score difference. No untested option gets a label.
    let reference = Tensor::operation(27, &[&lp], &[1, 0, 1], &[])?.dim(4, 1)?;
    let alternative = Tensor::operation(27, &[&lp], &[1, 1, 1], &[])?.dim(4, 1)?;
    let target = Tensor::floats(&targets, &[pairs.len() as i64], p.device, false)?;
    alternative
        .binary(13, &reference)?
        .binary(13, &target)?
        .unary(24)?
        .unary(18)
}
fn improvement_metrics(p: &Policy, pairs: &[Pair]) -> Result<(f64, f64), String> {
    if pairs.is_empty() {
        return Ok((0., 0.));
    }
    let _guard = NoGrad::new();
    let mse = regression_loss(p, pairs)?.value()?;
    let rows: Vec<_> = pairs.iter().map(|r| r.row.clone()).collect();
    let distributions = p.distributions(&rows)?;
    let mut correct = 0.;
    for (pair, probabilities) in pairs.iter().zip(distributions) {
        if (probabilities[1] > probabilities[0]) == (pair.improvement_target()? > 0.) {
            correct += 1.;
        }
    }
    Ok((mse, correct / pairs.len() as f64))
}
/// Fits actual local improvement under ONE accepted continuation revision.
/// This network is only a proposal: whole-game acceptance controls deployment.
/// Fresh and replay evidence share each loss and each Adam update.
pub fn update_improvement(
    p: &mut Policy,
    fresh: &[Pair],
    bank: &Bank,
    revision: u64,
    epochs: usize,
    batch: usize,
    rng: &mut Rng,
) -> Result<Json, String> {
    if !p.plan_residual {
        return Err("improvement scorer requires a flat plan policy".into());
    }
    if epochs == 0 || batch == 0 {
        return Err("improvement epochs and batch must be positive".into());
    }
    for pair in fresh {
        validate_improvement(pair, revision)?;
    }
    let active = bank.for_revision(revision)?;
    for pair in active.elite.iter().chain(&active.recent) {
        validate_improvement(pair, revision)?;
    }
    if fresh.is_empty() {
        return Ok(Json::Obj(vec![
            ("updates".into(), Json::Num(0.)),
            (
                "reason".into(),
                Json::Str("no fresh comparisons for current revision".into()),
            ),
        ]));
    }
    let checkpoint = p.checkpoint(0, rng)?;
    let (before_loss, before_accuracy) = improvement_metrics(p, fresh)?;
    let (retained_before_loss, retained_before_accuracy) = improvement_metrics(p, &active.elite)?;
    let mut ids: Vec<_> = (0..fresh.len()).collect();
    let mut count = 0;
    let mut loss = 0.;
    let attempt = (|| -> Result<(), String> {
        for _ in 0..epochs {
            rng.shuffle(&mut ids);
            for chunk in ids.chunks((batch / 2).max(1)) {
                let mut examples: Vec<_> = chunk.iter().map(|i| fresh[*i].clone()).collect();
                for n in 0..chunk.len().min(batch - chunk.len()) {
                    let pool = if n % 2 == 0 {
                        &active.elite
                    } else {
                        &active.recent
                    };
                    if !pool.is_empty() {
                        examples.push(pool[(rng.next() as usize) % pool.len()].clone());
                    }
                }
                let objective = regression_loss(p, &examples)?;
                let scalar = objective.value()?;
                if !scalar.is_finite() {
                    return Err("nonfinite improvement regression loss".into());
                }
                for parameter in &mut p.parameters {
                    parameter.value.zero_grad();
                }
                objective.backward()?;
                p.adam()?;
                loss += scalar;
                count += 1;
            }
        }
        Ok(())
    })();
    if let Err(error) = attempt {
        p.restore(&checkpoint)?;
        return Err(error);
    }
    let metrics = (|| -> Result<((f64, f64), (f64, f64)), String> {
        let fresh_metrics = improvement_metrics(p, fresh)?;
        let retained = improvement_metrics(p, &active.elite)?;
        if !fresh_metrics.0.is_finite() || !retained.0.is_finite() {
            return Err("nonfinite updated improvement scorer".into());
        }
        Ok((fresh_metrics, retained))
    })();
    let ((after_loss, after_accuracy), (retained_after_loss, retained_after_accuracy)) =
        match metrics {
            Ok(metrics) => metrics,
            Err(error) => {
                p.restore(&checkpoint)?;
                return Err(error);
            }
        };
    Ok(Json::Obj(vec![
        (
            "objective".into(),
            Json::Str("terminal_utility_difference_regression".into()),
        ),
        ("incumbent_revision".into(), Json::Str(revision.to_string())),
        ("updates".into(), Json::Num(count as f64)),
        ("pair_loss".into(), Json::Num(loss / count.max(1) as f64)),
        ("pair_mse_before".into(), Json::Num(before_loss)),
        ("pair_mse_after".into(), Json::Num(after_loss)),
        ("pair_accuracy_before".into(), Json::Num(before_accuracy)),
        ("pair_accuracy_after".into(), Json::Num(after_accuracy)),
        (
            "retained_mse_before".into(),
            Json::Num(retained_before_loss),
        ),
        ("retained_mse_after".into(), Json::Num(retained_after_loss)),
        (
            "retained_accuracy_before".into(),
            Json::Num(retained_before_accuracy),
        ),
        (
            "retained_accuracy_after".into(),
            Json::Num(retained_after_accuracy),
        ),
        (
            "replay_pairs".into(),
            Json::Num((active.elite.len() + active.recent.len()) as f64),
        ),
        (
            "stale_replay_filtered".into(),
            Json::Num(
                (bank.elite.len() + bank.recent.len() - active.elite.len() - active.recent.len())
                    as f64,
            ),
        ),
        ("reverted".into(), Json::Bool(false)),
    ]))
}

/// Evidence for a bounded deployment slot, accounting for ALL currently legal
/// alternatives. Pair accuracy alone does not prove the deployed argmax was tested.
pub fn support_report(p: &Policy, pairs: &[Pair], revision: u64) -> Result<Json, String> {
    use std::collections::{BTreeMap, BTreeSet};
    #[derive(Default)]
    struct Counts {
        tested: usize,
        supported: usize,
        contradicted: usize,
        untested: usize,
        unchanged: usize,
        seeds: BTreeSet<i64>,
        supported_gain: f64,
        observed_delta: f64,
    }
    let mut unique = BTreeSet::new();
    let mut states: BTreeMap<(usize, i64, usize, i64), (Sample, usize, BTreeMap<usize, f32>)> =
        BTreeMap::new();
    for pair in pairs {
        if !pair.matches_revision(revision)? {
            continue;
        }
        let delta = validate_improvement(pair, revision)?;
        let slot = evidence_integer(pair.evidence.get("slot_id"), "slot_id")? as usize;
        if slot >= 16 {
            return Err("comparison slot_id must be below 16".into());
        }
        let reference =
            evidence_integer(pair.evidence.get("reference_index"), "reference_index")? as usize;
        let alternative =
            evidence_integer(pair.evidence.get("alternative_index"), "alternative_index")? as usize;
        let full = Sample::parse(pair.evidence.get("full_row"))?;
        if reference == alternative
            || reference >= full.features.len()
            || alternative >= full.features.len()
            || full.context != pair.row.context
            || full.step != pair.row.step
            || full.features[reference] != pair.row.features[0]
            || full.features[alternative] != pair.row.features[1]
        {
            return Err("comparison full_row does not match its tested pair".into());
        }
        let key = (slot, pair.seed, pair.seat, pair.row.step);
        if !unique.insert((pair.seed, pair.seat, pair.row.step, alternative)) {
            continue;
        }
        let state = states
            .entry(key)
            .or_insert_with(|| (full.clone(), reference, BTreeMap::new()));
        if state.1 != reference
            || state.0.context != full.context
            || state.0.features != full.features
        {
            return Err("conflicting comparison candidate sets for the same state".into());
        }
        state.2.insert(alternative, delta);
    }
    let mut slots: BTreeMap<usize, Counts> = BTreeMap::new();
    let rows: Vec<_> = states.values().map(|state| state.0.clone()).collect();
    let probabilities = p.distributions(&rows)?;
    for (((slot, seed, _, _), (_, reference, tested)), scores) in states.iter().zip(probabilities) {
        let selected = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .unwrap()
            .0;
        let counts = slots.entry(*slot).or_default();
        counts.tested += tested.len();
        if selected == *reference {
            counts.unchanged += 1;
        } else if let Some(delta) = tested.get(&selected) {
            counts.observed_delta += *delta as f64;
            if *delta > 0. {
                counts.supported += 1;
                counts.seeds.insert(*seed);
                counts.supported_gain += *delta as f64;
            } else {
                counts.contradicted += 1;
            }
        } else {
            counts.untested += 1;
        }
    }
    Ok(Json::Obj(vec![
        ("incumbent_revision".into(), Json::Str(revision.to_string())),
        ("states".into(), Json::Num(states.len() as f64)),
        (
            "slots".into(),
            Json::Arr(
                slots
                    .into_iter()
                    .map(|(slot, c)| {
                        Json::Obj(vec![
                            ("slot_id".into(), Json::Num(slot as f64)),
                            ("tested_changes".into(), Json::Num(c.tested as f64)),
                            ("supported_changes".into(), Json::Num(c.supported as f64)),
                            (
                                "contradicted_changes".into(),
                                Json::Num(c.contradicted as f64),
                            ),
                            ("untested_choices".into(), Json::Num(c.untested as f64)),
                            ("unchanged_choices".into(), Json::Num(c.unchanged as f64)),
                            ("supported_seeds".into(), Json::Num(c.seeds.len() as f64)),
                            (
                                "mean_supported_gain".into(),
                                Json::Num(c.supported_gain / c.supported.max(1) as f64),
                            ),
                            ("total_observed_delta".into(), Json::Num(c.observed_delta)),
                        ])
                    })
                    .collect(),
            ),
        ),
    ]))
}

pub fn accuracy(p: &Policy, rows: &[Pair]) -> Result<f64, String> {
    if rows.is_empty() {
        return Ok(0.);
    }
    let ps = p.distributions(&rows.iter().map(|r| r.row.clone()).collect::<Vec<_>>())?;
    Ok(ps
        .iter()
        .zip(rows)
        // Match deployed deterministic selection: exact ties select candidate 0.
        // Giving ties half credit can report learning without ANY action changing.
        .map(|(v, r)| f64::from((v[1] > v[0]) == (r.target[1] > r.target[0])))
        .sum::<f64>()
        / rows.len() as f64)
}
pub fn update(
    p: &mut Policy,
    fresh: &[Pair],
    bank: &Bank,
    epochs: usize,
    batch: usize,
    rng: &mut Rng,
) -> Result<Json, String> {
    if fresh.is_empty() {
        return Ok(Json::Obj(vec![
            ("updates".into(), Json::Num(0.)),
            ("reason".into(), Json::Str("no decisive comparisons".into())),
        ]));
    }
    let checkpoint = p.checkpoint(0, rng)?;
    let before = accuracy(p, fresh)?;
    let retained_before = accuracy(p, &bank.elite)?;
    let rows: Vec<_> = fresh.iter().map(|r| r.row.clone()).collect();
    let probs_before = p.distributions(&rows)?;
    let mut count = 0;
    let mut loss = 0.;
    let mut ids: Vec<_> = (0..fresh.len()).collect();
    for _ in 0..epochs {
        rng.shuffle(&mut ids);
        // Fresh and protected comparisons enter the SAME loss/Adam update.
        for chunk in ids.chunks((batch / 2).max(1)) {
            let mut examples: Vec<_> = chunk.iter().map(|i| fresh[*i].clone()).collect();
            for n in 0..chunk.len() {
                let pool = if n % 2 == 0 {
                    &bank.elite
                } else {
                    &bank.recent
                };
                if !pool.is_empty() {
                    examples.push(pool[(rng.next() as usize) % pool.len()].clone());
                }
            }
            loss += p.distill_plans(
                &examples.iter().map(|x| x.row.clone()).collect::<Vec<_>>(),
                &examples
                    .iter()
                    .map(|x| x.target.clone())
                    .collect::<Vec<_>>(),
                1.,
            )?;
            count += 1;
        }
    }
    let after = accuracy(p, fresh)?;
    let retained_after = accuracy(p, &bank.elite)?;
    let (kl, changed) = divergence(&probs_before, &p.distributions(&rows)?);
    // Protect demonstrated choices, not arbitrary initial distributions or all argmax changes.
    let reverted = !loss.is_finite()
        || after + 0.05 < before
        || (!bank.elite.is_empty() && retained_after + 0.05 < retained_before);
    if reverted {
        p.restore(&checkpoint)?;
    }
    Ok(Json::Obj(vec![
        ("updates".into(), Json::Num(count as f64)),
        ("pair_loss".into(), Json::Num(loss / count.max(1) as f64)),
        ("pair_accuracy_before".into(), Json::Num(before)),
        (
            "pair_accuracy_after".into(),
            Json::Num(if reverted { before } else { after }),
        ),
        ("attempted_pair_accuracy".into(), Json::Num(after)),
        (
            "retained_accuracy_before".into(),
            Json::Num(retained_before),
        ),
        (
            "retained_accuracy_after".into(),
            Json::Num(if reverted {
                retained_before
            } else {
                retained_after
            }),
        ),
        ("pair_kl".into(), Json::Num(kl)),
        ("pair_argmax_changes".into(), Json::Num(changed)),
        ("reverted".into(), Json::Bool(reverted)),
    ]))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn regression_pair(
        feature: f32,
        reference: [f64; 2],
        alternative: [f64; 2],
        revision: u64,
    ) -> Pair {
        let mut features = vec![vec![0.; 32]; 2];
        features[0][31] = 1.;
        features[1][31] = 1.;
        features[1][14] = feature;
        let row = Sample {
            context: vec![0.; 320],
            features,
            ..Default::default()
        };
        Pair {
            row: row.clone(),
            target: vec![0.5, 0.5],
            gain: 0.3,
            iteration: 1,
            seed: 1,
            seat: 0,
            opponent: 0,
            bucket: 0,
            evidence: Json::Obj(vec![
                ("incumbent_revision".into(), Json::Str(revision.to_string())),
                ("slot_id".into(), Json::Num(0.)),
                ("reference_index".into(), Json::Num(0.)),
                ("alternative_index".into(), Json::Num(1.)),
                ("full_row".into(), row.json()),
                (
                    "reference_cash".into(),
                    Json::Arr(reference.map(Json::Num).to_vec()),
                ),
                (
                    "alternative_cash".into(),
                    Json::Arr(alternative.map(Json::Num).to_vec()),
                ),
            ]),
        }
    }
    #[test]
    fn improvement_target_uses_actual_magnitude_and_outcomes_before_cash() {
        assert!(improvement_target([50_000., 40_000.], [80_000., 90_000.]).unwrap() < 0.);
        assert!(improvement_target([1., 2.], [-5., -6.]).unwrap() > 0.);
        assert!(improvement_target([100., 100.], [101., 100.]).unwrap() > 0.5);
        let small = improvement_target([100., 90.], [110., 90.]).unwrap();
        let large = improvement_target([100., 90.], [150., 90.]).unwrap();
        assert!(small > 0. && large > small);
        assert_eq!(improvement_target([100., 90.], [100., 90.]).unwrap(), 0.);
        assert!(
            improvement_target([f64::MAX, -f64::MAX], [-f64::MAX, f64::MAX])
                .unwrap()
                .is_finite()
        );
        assert!(improvement_target([f64::NAN, 0.], [1., 0.]).is_err());
    }
    #[test]
    fn improvement_learning_fits_both_positive_and_negative_choices() {
        crate::learning::tensor::threads(1);
        let mut p = Policy::plans(-1, 55, 0.002).unwrap();
        let positive = regression_pair(1., [100., 100.], [150., 100.], 3);
        let mut negative = regression_pair(-1., [100., 100.], [50., 100.], 3);
        negative.seed = 2;
        let rows = vec![positive.clone(), negative];
        let metrics =
            update_improvement(&mut p, &rows, &Bank::default(), 3, 60, 8, &mut Rng(7)).unwrap();
        assert_eq!(metrics.get("pair_accuracy_before").f64(), 0.5);
        assert_eq!(metrics.get("pair_accuracy_after").f64(), 1.);
        assert!(metrics.get("pair_mse_after").f64() < metrics.get("pair_mse_before").f64() * 0.1);
        let choices = p
            .infer(
                &rows.iter().map(|p| p.row.clone()).collect::<Vec<_>>(),
                true,
                &mut Rng(0),
            )
            .unwrap();
        assert_eq!(choices[0].action, 1);
        assert_eq!(choices[1].action, 0);
        // A duplicated elite/recent record must count as ONE supporting state.
        let support = support_report(&p, &[positive.clone(), positive], 3).unwrap();
        let slot = &support.get("slots").arr()[0];
        assert_eq!(slot.get("supported_changes").i64(), 1);
        assert_eq!(slot.get("supported_seeds").i64(), 1);
    }
    #[test]
    fn missing_evidence_is_rejected_before_any_update() {
        crate::learning::tensor::threads(1);
        let mut p = Policy::plans(-1, 61, 0.001).unwrap();
        let mut pair = regression_pair(1., [100., 100.], [150., 100.], 3);
        let before = p.weights_json().unwrap();
        if let Json::Obj(fields) = &mut pair.evidence {
            fields.retain(|(k, _)| k != "reference_cash");
        }
        assert!(pair.improvement_target().is_err());
        assert!(
            update_improvement(&mut p, &[pair], &Bank::default(), 3, 1, 8, &mut Rng(0)).is_err()
        );
        assert_eq!(before, p.weights_json().unwrap());
        let mut invalid = regression_pair(1., [100., 100.], [150., 100.], 3);
        if let Json::Obj(fields) = &mut invalid.evidence {
            fields
                .iter_mut()
                .find(|(k, _)| k == "alternative_cash")
                .unwrap()
                .1 = Json::Arr(vec![Json::Num(f64::INFINITY), Json::Num(100.)]);
        }
        assert!(invalid.improvement_target().is_err());
    }
    #[test]
    fn continuation_revisions_cannot_mix_in_training_or_support() {
        crate::learning::tensor::threads(1);
        let mut p = Policy::plans(-1, 73, 0.001).unwrap();
        let fresh = regression_pair(1., [100., 100.], [150., 100.], 4);
        let old = regression_pair(1., [100., 100.], [50., 100.], 3);
        let bank = Bank {
            elite: vec![old.clone()],
            recent: vec![fresh.clone(), old.clone()],
        };
        let filtered = bank.for_revision(4).unwrap();
        assert!(filtered.elite.is_empty());
        assert_eq!(filtered.recent.len(), 1);
        assert!(update_improvement(&mut p, &[old.clone()], &bank, 4, 1, 8, &mut Rng(0)).is_err());
        let metrics = update_improvement(&mut p, &[fresh], &bank, 4, 2, 8, &mut Rng(0)).unwrap();
        assert_eq!(metrics.get("stale_replay_filtered").i64(), 2);
        assert_eq!(metrics.get("replay_pairs").i64(), 1);
        assert_eq!(
            support_report(&p, &[old], 4).unwrap().get("states").i64(),
            0
        );
    }
    #[test]
    fn support_distinguishes_untested_argmax_from_tested_improvement() {
        crate::learning::tensor::threads(1);
        let mut p = Policy::plans(-1, 55, 0.002).unwrap();
        let mut pair = regression_pair(1., [100., 100.], [150., 100.], 0);
        update_improvement(
            &mut p,
            &[pair.clone()],
            &Bank::default(),
            0,
            30,
            8,
            &mut Rng(7),
        )
        .unwrap();
        let mut full = pair.row.clone();
        let mut untested = full.features[1].clone();
        untested[30] = 20.;
        full.features.push(untested);
        if let Json::Obj(fields) = &mut pair.evidence {
            fields.iter_mut().find(|(k, _)| k == "full_row").unwrap().1 = full.json();
        }
        let report = support_report(&p, &[pair.clone()], 0).unwrap();
        let slot = &report.get("slots").arr()[0];
        assert_eq!(slot.get("untested_choices").i64(), 1);
        assert_eq!(slot.get("supported_changes").i64(), 0);
        if let Json::Obj(fields) = &mut pair.evidence {
            fields.retain(|(k, _)| k != "full_row");
        }
        assert!(support_report(&p, &[pair], 0).is_err());
    }
    #[test]
    fn results_rank_wins_before_cash_and_ignore_noise() {
        assert!(preference([50000., 40000.], [80000., 90000.]).unwrap().0[0] > 0.5);
        assert!(preference([50000., 40000.], [51000., 40000.]).unwrap().0[1] > 0.5);
        assert!(preference([50000., 40000.], [50001., 40000.]).is_none());
    }
    #[test]
    fn accuracy_uses_the_executed_choice_including_initial_ties() {
        crate::learning::tensor::threads(1);
        let p = Policy::plans(-1, 55, 0.001).unwrap();
        let mut features = vec![vec![0.; 32]; 2];
        features[0][31] = 1.;
        features[1][31] = 1.;
        features[1][14] = 1.;
        let mut pair = Pair {
            row: Sample {
                context: vec![0.; 320],
                features,
                ..Default::default()
            },
            target: vec![0.9, 0.1],
            gain: 0.3,
            iteration: 1,
            seed: 1,
            seat: 0,
            opponent: 0,
            bucket: 0,
            evidence: Json::Null,
        };
        assert_eq!(
            p.infer(&[pair.row.clone()], true, &mut Rng(0)).unwrap()[0].action,
            0
        );
        assert_eq!(accuracy(&p, &[pair.clone()]).unwrap(), 1.);
        pair.target = vec![0.1, 0.9];
        assert_eq!(accuracy(&p, &[pair]).unwrap(), 0.);
    }
    #[test]
    fn ranking_updates_deterministic_choice_and_bank_roundtrips() {
        crate::learning::tensor::threads(1);
        let mut p = Policy::plans(-1, 55, 0.001).unwrap();
        let mut f = vec![vec![0.; 32]; 2];
        f[0][31] = 1.;
        f[1][31] = 1.;
        f[1][14] = 1.;
        let pair = Pair {
            row: Sample {
                context: vec![0.; 320],
                features: f,
                ..Default::default()
            },
            target: vec![0.1, 0.9],
            gain: 0.3,
            iteration: 1,
            seed: 1,
            seat: 0,
            opponent: 0,
            bucket: 0,
            evidence: Json::Null,
        };
        let m = update(
            &mut p,
            &[pair.clone()],
            &Bank::default(),
            20,
            8,
            &mut Rng(7),
        )
        .unwrap();
        assert!(!matches!(m.get("reverted"), Json::Bool(true)));
        assert_eq!(
            p.infer(&[pair.row.clone()], true, &mut Rng(0)).unwrap()[0].action,
            1
        );
        let mut b = Bank::default();
        b.admit(&[pair.clone()]);
        for n in 0..600 {
            let mut recent = pair.clone();
            recent.iteration = n + 2;
            recent.gain = 0.01;
            b.admit(&[recent]);
        }
        assert!(b.elite.iter().any(|r| r.iteration == 1));
        assert_eq!(b.recent.len(), 512);
        assert_eq!(b.json(), Bank::parse(&b.json()).unwrap().json());
    }
}

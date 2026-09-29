//! All executable choices share one supervised row. Replay and mini-batches
//! sample complete sets; untested/padded actions never receive invented labels.
use super::{
    plan_compare::{Bank, Pair, MATCH_SCORE_OBJECTIVE},
    policy::{Batch, Policy, Rng, Sample},
    tensor::{NoGrad, Tensor},
};
use kagg_engine::json::Json;
use std::collections::BTreeSet;
fn unique(pairs: &[Pair]) -> Vec<Pair> {
    let mut seen = BTreeSet::new();
    pairs
        .iter()
        .filter(|p| seen.insert(p.evidence.get("candidate_set_id").str().to_string()))
        .cloned()
        .collect()
}
pub fn validate(p: &Pair) -> Result<(), String> {
    let e = &p.evidence;
    let row = Sample::parse(e.get("full_row"))?;
    let cash = e.get("all_terminal_cash").arr();
    if e.get("candidate_set_id").str().is_empty()
        || e.get("objective").str() != MATCH_SCORE_OBJECTIVE
        || row.features.len() < 2
        || row.features.len() > crate::pipeline::plan_menu::MAX_CONTEXTUAL_CHOICES
        || cash.len() != row.features.len()
        || e.get("reference_index").i64() != 0
        || e.get("terminal_step").i64() != 719
    {
        return Err("incomplete candidate set".into());
    }
    for v in cash {
        if v.arr().len() != 2 || v.arr().iter().any(|x| !x.is_num() || !x.f64().is_finite()) {
            return Err("invalid terminal set outcome".into());
        }
    }
    let alt = e.get("alternative_index").i64() as usize;
    if p.row.context != row.context
        || p.row.step != row.step
        || alt == 0
        || alt >= cash.len()
        || e.get("reference_cash") != &cash[0]
        || e.get("alternative_cash") != &cash[alt]
        || p.row.features != vec![row.features[0].clone(), row.features[alt].clone()]
    {
        return Err("set/pair evidence mismatch".into());
    }
    Ok(())
}
fn score(v: &Json) -> f32 {
    let a = v.arr();
    if a[0].f64() > a[1].f64() {
        1.
    } else if a[0].f64() == a[1].f64() {
        0.5
    } else {
        0.
    }
}
fn loss(p: &Policy, sets: &[Pair]) -> Result<Tensor, String> {
    let rows = sets
        .iter()
        .map(|r| Sample::parse(r.evidence.get("full_row")))
        .collect::<Result<Vec<_>, _>>()?;
    let b = Batch::new(&rows, p.device)?;
    let lp = p.forward(&b)?.0;
    let reference = Tensor::operation(27, &[&lp], &[1, 0, 1], &[])?;
    let mut targets = vec![0.; sets.len() * b.width];
    let mut mask = targets.clone();
    // Each set contributes equally even when it has fewer legal candidates.
    for (i, r) in sets.iter().enumerate() {
        let cash = r.evidence.get("all_terminal_cash").arr();
        for j in 1..cash.len() {
            targets[i * b.width + j] = score(&cash[j]) - score(&cash[0]);
            mask[i * b.width + j] = b.width as f32 / (cash.len() - 1) as f32;
        }
    }
    let shape = [sets.len() as i64, b.width as i64];
    let target = Tensor::floats(&targets, &shape, p.device, false)?;
    let mask = Tensor::floats(&mask, &shape, p.device, false)?;
    // Mask BEFORE squaring: padded log probabilities may contain large sentinels.
    lp.binary(13, &reference)?
        .binary(13, &target)?
        .binary(14, &mask.unary(28)?)?
        .unary(24)?
        .unary(18)
}
pub fn metrics(p: &Policy, pairs: &[Pair]) -> Result<Json, String> {
    for pair in pairs {
        validate(pair)?;
    }
    let sets = unique(pairs);
    let mut all = Json::Obj(vec![]);
    let _guard = NoGrad::new();
    for stage in ["all", "arrangement", "revision"] {
        let rs = sets
            .iter()
            .filter(|r| stage == "all" || r.evidence.get("stage").str() == stage)
            .collect::<Vec<_>>();
        let mut correct = 0;
        let mut informative = 0;
        let mut gain = 0.;
        let mut regret = 0.;
        for chunk in rs.chunks(64) {
            let rows = chunk
                .iter()
                .map(|r| Sample::parse(r.evidence.get("full_row")))
                .collect::<Result<Vec<_>, _>>()?;
            let ds = p.infer(&rows, true, &mut Rng(0))?;
            for (r, d) in chunk.iter().zip(ds) {
                let scores = r
                    .evidence
                    .get("all_terminal_cash")
                    .arr()
                    .iter()
                    .map(score)
                    .collect::<Vec<_>>();
                let best = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let worst = scores.iter().copied().fold(f32::INFINITY, f32::min);
                if best > worst {
                    informative += 1;
                    correct += usize::from(scores[d.action] == best);
                }
                gain += scores[d.action] - scores[0];
                regret += best - scores[d.action];
            }
        }
        all.set_path(
            stage,
            Json::Obj(vec![
                ("sets".into(), Json::Num(rs.len() as f64)),
                ("informative_sets".into(), Json::Num(informative as f64)),
                (
                    "best_choice_accuracy".into(),
                    if informative > 0 {
                        Json::Num(correct as f64 / informative as f64)
                    } else {
                        Json::Null
                    },
                ),
                (
                    "mean_recorded_score_gain".into(),
                    Json::Num(gain as f64 / rs.len().max(1) as f64),
                ),
                (
                    "mean_regret".into(),
                    Json::Num(regret as f64 / rs.len().max(1) as f64),
                ),
                ("unsupported_argmax".into(), Json::Num(0.)),
            ]),
        );
    }
    Ok(all)
}
pub fn update(
    p: &mut Policy,
    fresh: &[Pair],
    bank: &Bank,
    revision: u64,
    epochs: usize,
    batch: usize,
    rng: &mut Rng,
) -> Result<Json, String> {
    for r in fresh.iter().chain(&bank.elite).chain(&bank.recent) {
        validate(r)?;
        if !r.matches_revision(revision)? {
            return Err("stale complete set".into());
        }
    }
    let fresh = unique(fresh);
    let history = unique(&bank.elite);
    let recent = unique(&bank.recent);
    let before = metrics(p, &fresh)?;
    if fresh.is_empty() {
        return Ok(Json::Obj(vec![
            ("updates".into(), Json::Num(0.)),
            (
                "reason".into(),
                Json::Str("no complete sets within budget".into()),
            ),
        ]));
    }
    let checkpoint = p.checkpoint(0, rng)?;
    let mut total = 0.;
    let mut updates = 0;
    let mut ids = (0..fresh.len()).collect::<Vec<_>>();
    let attempt = (|| -> Result<(), String> {
        for _ in 0..epochs {
            rng.shuffle(&mut ids);
            for ix in ids.chunks((batch / 2).max(1)) {
                let mut examples = ix.iter().map(|i| fresh[*i].clone()).collect::<Vec<_>>();
                for i in 0..ix.len().min(batch - ix.len()) {
                    let pool = if i % 2 == 0 { &history } else { &recent };
                    if !pool.is_empty() {
                        examples.push(pool[(rng.next() as usize) % pool.len()].clone());
                    }
                }
                let l = loss(p, &examples)?;
                let scalar = l.value()?;
                if !scalar.is_finite() {
                    return Err("nonfinite full-set loss".into());
                }
                for x in &mut p.parameters {
                    x.value.zero_grad();
                }
                l.backward()?;
                p.adam()?;
                total += scalar;
                updates += 1;
            }
        }
        Ok(())
    })();
    if let Err(e) = attempt {
        p.restore(&checkpoint)?;
        return Err(e);
    }
    Ok(Json::Obj(vec![
        ("objective".into(), Json::Str(MATCH_SCORE_OBJECTIVE.into())),
        (
            "learning_unit".into(),
            Json::Str("complete_candidate_set".into()),
        ),
        ("updates".into(), Json::Num(updates as f64)),
        ("set_loss".into(), Json::Num(total / updates.max(1) as f64)),
        ("before".into(), before),
        ("after".into(), metrics(p, &fresh)?),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn example(id: &str, context: f32, win: usize, n: usize) -> Pair {
        let mut row = Sample {
            context: vec![0.; 320],
            features: vec![vec![0.; 32]; n],
            ..Default::default()
        };
        row.context[0] = context;
        for i in 0..n {
            row.features[i][i + 2] = 1.;
            row.features[i][31] = 1.;
        }
        let cash = (0..n)
            .map(|i| {
                Json::Arr(vec![
                    Json::Num(if i == win { 2. } else { 0. }),
                    Json::Num(1.),
                ])
            })
            .collect::<Vec<_>>();
        Pair {
            row: Sample {
                features: vec![row.features[0].clone(), row.features[1].clone()],
                ..row.clone()
            },
            target: vec![0.5, 0.5],
            gain: 0.,
            iteration: 1,
            seed: 1,
            seat: 0,
            opponent: 0,
            bucket: 0,
            evidence: Json::Obj(vec![
                ("candidate_set_id".into(), Json::Str(id.into())),
                ("objective".into(), Json::Str(MATCH_SCORE_OBJECTIVE.into())),
                ("full_row".into(), row.json()),
                ("all_terminal_cash".into(), Json::Arr(cash.clone())),
                ("reference_index".into(), Json::Num(0.)),
                ("alternative_index".into(), Json::Num(1.)),
                ("terminal_step".into(), Json::Num(719.)),
                ("reference_cash".into(), cash[0].clone()),
                ("alternative_cash".into(), cash[1].clone()),
                ("incumbent_revision".into(), Json::Str("0".into())),
                ("stage".into(), Json::Str("arrangement".into())),
            ]),
        }
    }
    #[test]
    fn complete_loss_learns_opposite_conditional_choices_and_masks_padding() {
        crate::learning::tensor::worker_threads();
        let mut p = Policy::event_plans(-1, 27, 0.003).unwrap();
        let examples = vec![example("a", -1., 1, 4), example("b", 1., 2, 3)];
        let report = update(
            &mut p,
            &examples,
            &Bank {
                elite: vec![],
                recent: vec![],
            },
            0,
            150,
            8,
            &mut Rng(4),
        )
        .unwrap();
        assert!(report.get("set_loss").f64().is_finite());
        assert_eq!(
            report
                .get("after")
                .get("all")
                .get("best_choice_accuracy")
                .f64(),
            1.
        );
        let mut invalid = examples[0].clone();
        invalid
            .evidence
            .set_path("all_terminal_cash", Json::Arr(vec![]));
        assert!(validate(&invalid).is_err());
        assert!(metrics(&p, &[invalid]).is_err());
    }
}

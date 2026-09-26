//! Hierarchical policy, explicit exploration behavior likelihood and native PPO.
use super::tensor::{NoGrad, Tensor};
use kagg_engine::json::Json;
pub const CONTEXT: usize = 96;
pub const CANDIDATE: usize = 32;
pub const GROUPS: usize = 16;
const SPECS: [(&str, &[i64]); 16] = [
    ("context.0.weight", &[64, 96]),
    ("context.0.bias", &[64]),
    ("context.2.weight", &[64, 64]),
    ("context.2.bias", &[64]),
    ("candidate.0.weight", &[64, 32]),
    ("candidate.0.bias", &[64]),
    ("score.0.weight", &[64, 128]),
    ("score.0.bias", &[64]),
    ("score.2.weight", &[1, 64]),
    ("score.2.bias", &[1]),
    ("value.0.weight", &[64, 64]),
    ("value.0.bias", &[64]),
    ("value.2.weight", &[1, 64]),
    ("value.2.bias", &[1]),
    ("group.weight", &[16, 64]),
    ("group.bias", &[16]),
];
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
    pub fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn shuffle(&mut self, indices: &mut [usize]) {
        for i in (1..indices.len()).rev() {
            let range = i as u64 + 1;
            let threshold = range.wrapping_neg() % range;
            let x = loop {
                let x = self.next();
                if x >= threshold {
                    break x;
                }
            };
            indices.swap(i, (x % range) as usize);
        }
    }
}
pub struct Parameter {
    pub name: &'static str,
    pub shape: Vec<i64>,
    pub value: Tensor,
    pub m: Tensor,
    pub v: Tensor,
    pub step: u64,
}
pub struct Policy {
    pub parameters: Vec<Parameter>,
    pub device: i32,
    pub lr: f64,
}
impl Policy {
    pub fn mixed_routes(device: i32, seed: u64, lr: f64) -> Result<Self, String> {
        if !lr.is_finite() || lr <= 0. {
            return Err("invalid policy initialization".into());
        }
        let mut rng = Rng(seed);
        let mut parameters = Vec::new();
        for (index, (name, shape)) in SPECS.iter().enumerate() {
            let n = shape.iter().product::<i64>() as usize;
            let fan = if shape.len() == 2 {
                shape[1]
            } else {
                SPECS[index - 1].1[1]
            };
            let bound = 1. / (fan as f64).sqrt();
            let data: Vec<f32> = (0..n)
                .map(|_| ((2. * rng.uniform() - 1.) * bound) as f32)
                .collect();
            let value = Tensor::floats(&data, shape, device, true)?;
            let m = value.unary(26)?;
            let v = value.unary(26)?;
            parameters.push(Parameter {
                name,
                shape: shape.to_vec(),
                value,
                m,
                v,
                step: 0,
            });
        }
        Ok(Self {
            parameters,
            device,
            lr,
        })
    }
    fn linear(&self, x: &Tensor, index: usize) -> Result<Tensor, String> {
        Tensor::operation(
            0,
            &[
                x,
                &self.parameters[index].value,
                &self.parameters[index + 1].value,
            ],
            &[],
            &[],
        )
    }
    pub fn forward(&self, b: &Batch) -> Result<(Tensor, Tensor), String> {
        let z = self
            .linear(&self.linear(&b.context, 0)?.unary(1)?, 2)?
            .unary(1)?;
        let v = self.linear(&b.candidates, 4)?.unary(1)?;
        let expanded = Tensor::operation(5, &[&z.dim(3, 1)?], &[-1, b.width as i64, -1], &[])?;
        let joined = Tensor::operation(2, &[&v, &expanded], &[-1], &[])?;
        let scores = self
            .linear(&self.linear(&joined, 6)?.unary(1)?, 8)?
            .dim(4, -1)?;
        let head = self.linear(&z, 14)?;
        let lp = Tensor::operation(44, &[&scores, &head, &b.groups, &b.mask], &[], &[])?;
        let value = self
            .linear(&self.linear(&z, 10)?.unary(1)?, 12)?
            .dim(4, -1)?;
        Ok((lp, value))
    }
    pub fn infer(
        &self,
        rows: &[Sample],
        deterministic: bool,
        rng: &mut Rng,
    ) -> Result<Vec<Decision>, String> {
        if rows.is_empty() {
            return Ok(vec![]);
        }
        let _guard = NoGrad::new();
        let batch = Batch::new(rows, self.device)?;
        let (lp, value) = self.forward(&batch)?;
        let base = lp.data()?;
        let behavior = self.behavior(&lp, &batch)?.data()?;
        let lp = if deterministic { &base } else { &behavior };
        let values = value.data()?;
        let mut decisions = Vec::with_capacity(rows.len());
        for (i, row) in rows.iter().enumerate() {
            let probs = &lp[i * batch.width..i * batch.width + row.features.len()];
            let action = if deterministic {
                hierarchical_argmax(probs, &row.features)
            } else {
                let total = probs.iter().map(|x| f64::from(*x).exp()).sum::<f64>();
                let pick = rng.uniform() * total;
                let mut sum = 0.;
                let mut index = probs.len() - 1;
                for (k, p) in probs.iter().enumerate() {
                    sum += f64::from(*p).exp();
                    if pick < sum {
                        index = k;
                        break;
                    }
                }
                index
            };
            decisions.push(Decision {
                action,
                logp: probs[action],
                value: values[i],
                wait_probability: row
                    .features
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| f[31] == 0.)
                    .map(|(k, _)| base[i * batch.width + k].exp())
                    .sum(),
            });
        }
        Ok(decisions)
    }
    pub fn behavior(&self, lp: &Tensor, b: &Batch) -> Result<Tensor, String> {
        Tensor::operation(45, &[lp, &b.proposal, &b.exploration, &b.mask], &[], &[])
    }

    /// Old self-generated experience has its own advantage-weighted imitation loss.
    /// It never enters PPO ratios or the critic's on-policy regression.
    pub fn imitate(&mut self, rows: &[Sample], coefficient: f64) -> Result<(usize, f64), String> {
        if rows.is_empty() || coefficient == 0. {
            return Ok((0, 0.));
        }
        let b = Batch::new(rows, self.device)?;
        let (lp, value) = self.forward(&b)?;
        let targets = Tensor::floats(
            &rows.iter().map(|r| r.mc_return).collect::<Vec<_>>(),
            &[rows.len() as i64],
            self.device,
            false,
        )?;
        let weight = Tensor::operation(
            19,
            &[&targets.binary(13, &value)?.unary(25)?],
            &[],
            &[0., 1.],
        )?;
        if weight.unary(18)?.value()? <= 0. {
            return Ok((0, 0.));
        }
        let actions = Tensor::integers(
            &rows.iter().map(|r| r.action as i64).collect::<Vec<_>>(),
            &[rows.len() as i64, 1],
            self.device,
            false,
        )?;
        let selected = Tensor::operation(21, &[&lp, &actions], &[1], &[])?.dim(4, 1)?;
        let loss = selected
            .binary(14, &weight)?
            .unary(18)?
            .scalar(31, -coefficient)?;
        let scalar = loss.value()?;
        if !scalar.is_finite() {
            return Err("non-finite imitation loss".into());
        }
        for p in &mut self.parameters {
            p.value.zero_grad();
        }
        loss.backward()?;
        self.adam()?;
        Ok((rows.len(), scalar))
    }
    pub fn load_weights(&mut self, weights: &Json) -> Result<(), String> {
        for p in &mut self.parameters {
            let row = weights.get(p.name);
            if !row.is_obj() {
                return Err(format!("missing parameter {}", p.name));
            }
            let shape: Vec<_> = row.get("shape").arr().iter().map(Json::i64).collect();
            if shape != p.shape {
                return Err(format!("wrong shape for {}", p.name));
            }
            let data = parse_floats(row.get("data"))?;
            let value = Tensor::floats(&data, &shape, self.device, false)?;
            p.value.copy_from(&value)?;
        }
        Ok(())
    }
    pub fn weights_json(&self) -> Result<Json, String> {
        Ok(Json::Obj(
            self.parameters
                .iter()
                .map(|p| {
                    Ok((
                        p.name.into(),
                        Json::Obj(vec![
                            (
                                "shape".into(),
                                Json::Arr(p.shape.iter().map(|&d| Json::Num(d as f64)).collect()),
                            ),
                            (
                                "data".into(),
                                Json::Arr(
                                    p.value
                                        .data()?
                                        .into_iter()
                                        .map(|v| Json::Num(v as f64))
                                        .collect(),
                                ),
                            ),
                        ]),
                    ))
                })
                .collect::<Result<_, String>>()?,
        ))
    }
    /// Model plus Adam moments. Native checkpoints are explicit JSON, not Python pickle.
    pub fn checkpoint(&self, iteration: u64, rng: &Rng) -> Result<Json, String> {
        let array =
            |values: Vec<f32>| Json::Arr(values.into_iter().map(|v| Json::Num(v as f64)).collect());
        let optimizer = self
            .parameters
            .iter()
            .map(|p| {
                Ok((
                    p.name.to_owned(),
                    Json::Obj(vec![
                        ("step".into(), Json::Str(p.step.to_string())),
                        ("m".into(), array(p.m.data()?)),
                        ("v".into(), array(p.v.data()?)),
                    ]),
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Json::Obj(vec![
            (
                "schema".into(),
                Json::Str("mixed-production-v7-ppo-v1".into()),
            ),
            (
                "policy_contract".into(),
                Json::Str(crate::pipeline::ENCODING.into()),
            ),
            ("iteration".into(), Json::Str(iteration.to_string())),
            ("rng".into(), Json::Str(rng.0.to_string())),
            ("rng_algorithm".into(), Json::Str("splitmix64".into())),
            ("learning_rate".into(), Json::Num(self.lr)),
            ("weights".into(), self.weights_json()?),
            ("optimizer".into(), Json::Obj(optimizer)),
        ]))
    }
    pub fn restore(&mut self, checkpoint: &Json) -> Result<(u64, Rng), String> {
        if checkpoint.get("schema").str() != "mixed-production-v7-ppo-v1"
            || checkpoint.get("policy_contract").str() != crate::pipeline::ENCODING
            || checkpoint.get("rng_algorithm").str() != "splitmix64"
        {
            return Err("incompatible native checkpoint".into());
        }
        let uint = |key: &str| {
            checkpoint
                .get(key)
                .str()
                .parse::<u64>()
                .map_err(|_| format!("invalid {key}"))
        };
        let iteration = uint("iteration")?;
        let rng = Rng(uint("rng")?);
        let lr = checkpoint.get("learning_rate").f64();
        if !lr.is_finite() || lr <= 0. {
            return Err("invalid checkpoint learning rate".into());
        }
        self.load_weights(checkpoint.get("weights"))?;
        for p in &mut self.parameters {
            let state = checkpoint.get("optimizer").get(p.name);
            p.step = state
                .get("step")
                .str()
                .parse::<u64>()
                .map_err(|_| "invalid Adam step")?;
            let m = parse_floats(state.get("m"))?;
            let v = parse_floats(state.get("v"))?;
            if v.iter().any(|x| *x < 0.) {
                return Err("negative Adam second moment".into());
            }
            p.m = Tensor::floats(&m, &p.shape, self.device, false)?;
            p.v = Tensor::floats(&v, &p.shape, self.device, false)?;
        }
        self.lr = lr;
        Ok((iteration, rng))
    }
    fn adam(&mut self) -> Result<(), String> {
        let _guard = NoGrad::new();
        let mut grads = Vec::new();
        let mut norms = Vec::new();
        for (i, p) in self.parameters.iter().enumerate() {
            if p.value.has_grad() {
                let g = p.value.unary(33)?;
                norms.push(g.unary(37)?.dim(3, 0)?);
                grads.push((i, g));
            }
        }
        if grads.is_empty() {
            return Ok(());
        }
        let refs: Vec<_> = norms.iter().collect();
        let norm = Tensor::operation(2, &refs, &[0], &[])?.unary(37)?;
        let coef = Tensor::operation(
            19,
            &[&norm.scalar(30, 1e-6)?.unary(38)?.scalar(31, 0.5)?],
            &[],
            &[0., 1.],
        )?;
        for (i, mut g) in grads {
            let p = &mut self.parameters[i];
            let clipped = g.binary(14, &coef)?;
            g.copy_from(&clipped)?;
            p.step += 1;
            p.m = Tensor::operation(40, &[&p.m, &g], &[], &[0.1])?;
            p.v = Tensor::operation(41, &[&p.v.scalar(31, 0.999)?, &g, &g], &[], &[0.001])?;
            let correction1 = 1. - 0.9f64.powf(p.step as f64);
            let correction2 = (1. - 0.999f64.powf(p.step as f64)).sqrt();
            let denom = p.v.unary(28)?.scalar(43, correction2)?.scalar(30, 1e-8)?;
            let updated = Tensor::operation(
                39,
                &[&p.value, &p.m, &denom],
                &[],
                &[-self.lr / correction1],
            )?;
            p.value.copy_from(&updated)?;
        }
        Ok(())
    }
    pub fn update_batch(
        &mut self,
        b: &Batch,
        old: &Tensor,
        actions: &Tensor,
        returns: &Tensor,
        advantage: &Tensor,
    ) -> Result<Update, String> {
        let (lp, value) = self.forward(b)?;
        let behavior = self.behavior(&lp, b)?;
        let new = Tensor::operation(21, &[&behavior, &actions.dim(3, 1)?], &[1], &[])?.dim(4, 1)?;
        let log_ratio = new.binary(13, old)?;
        let ratio = log_ratio.unary(16)?;
        let kl = ratio
            .scalar(30, -1.)?
            .binary(13, &log_ratio)?
            .unary(18)?
            .value()?;
        if !kl.is_finite() {
            return Err("non-finite PPO divergence".into());
        }
        if kl > 0.02 {
            return Ok(Update {
                loss: None,
                kl,
                stopped: true,
            });
        }
        let unclipped = ratio.binary(14, advantage)?;
        let clipped = Tensor::operation(19, &[&ratio], &[], &[0.9, 1.1])?.binary(14, advantage)?;
        let policy = unclipped.binary(20, &clipped)?.unary(18)?.unary(32)?;
        let values = value
            .binary(13, returns)?
            .unary(24)?
            .unary(18)?
            .scalar(31, 0.5)?;
        let normalized = lp.binary(13, &lp.dim(35, -1)?)?;
        let entropy = normalized
            .dim(42, -1)?
            .binary(14, &normalized)?
            .dim(17, -1)?
            .unary(18)?
            .unary(32)?;
        let loss = policy
            .binary(12, &values)?
            .binary(13, &entropy.scalar(31, 0.001)?)?;
        let scalar = loss.value()?;
        if !scalar.is_finite() {
            return Err("non-finite PPO loss".into());
        }
        for p in &mut self.parameters {
            p.value.zero_grad();
        }
        loss.backward()?;
        self.adam()?;
        Ok(Update {
            loss: Some(scalar),
            kl,
            stopped: false,
        })
    }
    pub fn update(
        &mut self,
        rows: &[Sample],
        epochs: usize,
        batch_size: usize,
        rng: &mut Rng,
    ) -> Result<Metrics, String> {
        let mut metrics = Metrics {
            samples: rows.len(),
            updates: 0,
            loss: 0.,
            mean_kl: 0.,
            kl_stopped: false,
        };
        if rows.is_empty() {
            return Ok(metrics);
        }
        if epochs == 0 || batch_size == 0 {
            return Err("positive training epochs and batch size required".into());
        }
        let full = Batch::new(rows, self.device)?;
        let old = Tensor::floats(
            &rows.iter().map(|r| r.logp).collect::<Vec<_>>(),
            &[rows.len() as i64],
            self.device,
            false,
        )?;
        let actions = Tensor::integers(
            &rows.iter().map(|r| r.action as i64).collect::<Vec<_>>(),
            &[rows.len() as i64],
            self.device,
            false,
        )?;
        let returns = Tensor::floats(
            &rows.iter().map(|r| r.reward).collect::<Vec<_>>(),
            &[rows.len() as i64],
            self.device,
            false,
        )?;
        let mut adv = Tensor::floats(
            &rows.iter().map(|r| r.advantage).collect::<Vec<_>>(),
            &[rows.len() as i64],
            self.device,
            false,
        )?;
        if rows.len() > 1 {
            adv = adv.binary(15, &adv.dim(23, 0)?.scalar(29, 1.)?)?;
        }
        let mut ids: Vec<_> = (0..rows.len()).collect();
        'epochs: for _ in 0..epochs {
            rng.shuffle(&mut ids);
            for chunk in ids.chunks(batch_size) {
                let ix = Tensor::integers(
                    &chunk.iter().map(|&i| i as i64).collect::<Vec<_>>(),
                    &[chunk.len() as i64],
                    self.device,
                    false,
                )?;
                let select = |t: &Tensor| Tensor::operation(22, &[t, &ix], &[0], &[]);
                let batch = full.select(&ix)?;
                let result = self.update_batch(
                    &batch,
                    &select(&old)?,
                    &select(&actions)?,
                    &select(&returns)?,
                    &select(&adv)?,
                )?;
                if result.stopped {
                    metrics.kl_stopped = true;
                    break 'epochs;
                }
                metrics.updates += 1;
                metrics.loss += result.loss.unwrap();
                metrics.mean_kl += result.kl;
            }
        }
        if metrics.updates > 0 {
            metrics.loss /= metrics.updates as f64;
            metrics.mean_kl /= metrics.updates as f64;
        }
        Ok(metrics)
    }
}
pub struct Decision {
    pub wait_probability: f32,
    pub action: usize,
    pub logp: f32,
    pub value: f32,
}
pub struct Update {
    pub loss: Option<f64>,
    pub kl: f64,
    pub stopped: bool,
}
pub struct Metrics {
    pub samples: usize,
    pub updates: usize,
    pub loss: f64,
    pub mean_kl: f64,
    pub kl_stopped: bool,
}
#[derive(Clone, Debug, Default)]
pub struct Sample {
    pub exploration: f32,
    pub step: i64,
    pub cash: f32,
    pub elapsed: i64,
    pub cash_delta: f32,
    pub advantage: f32,
    pub mc_return: f32,
    pub context: Vec<f32>,
    pub features: Vec<Vec<f32>>,
    pub action: usize,
    pub logp: f32,
    pub value: f32,
    pub reward: f32,
}
fn parse_floats(j: &Json) -> Result<Vec<f32>, String> {
    if !j.is_arr() {
        return Err("expected float array".into());
    }
    j.arr()
        .iter()
        .map(|v| {
            let x = v.f64() as f32;
            if !v.is_num() || !x.is_finite() {
                Err("non-finite model input".into())
            } else {
                Ok(x)
            }
        })
        .collect()
}
impl Sample {
    pub fn json(&self) -> Json {
        let floats = |v: &[f32]| Json::Arr(v.iter().map(|&x| Json::Num(x as f64)).collect());
        Json::Obj(vec![
            ("context".into(), floats(&self.context)),
            (
                "features".into(),
                Json::Arr(self.features.iter().map(|f| floats(f)).collect()),
            ),
            ("action".into(), Json::Num(self.action as f64)),
            ("logp".into(), Json::Num(self.logp as f64)),
            ("value".into(), Json::Num(self.value as f64)),
            ("return".into(), Json::Num(self.reward as f64)),
            ("exploration".into(), Json::Num(self.exploration as f64)),
            ("step".into(), Json::Num(self.step as f64)),
            ("cash".into(), Json::Num(self.cash as f64)),
            ("elapsed".into(), Json::Num(self.elapsed as f64)),
            ("cash_delta".into(), Json::Num(self.cash_delta as f64)),
            ("advantage".into(), Json::Num(self.advantage as f64)),
            ("mc_return".into(), Json::Num(self.mc_return as f64)),
        ])
    }
    pub fn parse(j: &Json) -> Result<Self, String> {
        let features = j
            .get("features")
            .arr()
            .iter()
            .map(parse_floats)
            .collect::<Result<Vec<_>, _>>()?;
        let context = parse_floats(j.get("context"))?;
        for key in [
            "action",
            "logp",
            "value",
            "return",
            "exploration",
            "step",
            "cash",
            "elapsed",
            "cash_delta",
            "advantage",
            "mc_return",
        ] {
            if !j.get(key).is_num() {
                return Err(format!("missing numeric training field {key}"));
            }
        }
        let action = j.get("action").i64();
        if j.get("action").f64() != action as f64 {
            return Err("non-integer action".into());
        }
        if context.len() != CONTEXT
            || features.is_empty()
            || features.len() > 128
            || features.iter().any(|f| f.len() != CANDIDATE)
            || action < 0
            || action as usize >= features.len()
        {
            return Err("invalid training sample shape/action".into());
        }
        let row = Self {
            context,
            features,
            action: action as usize,
            logp: j.get("logp").f64() as f32,
            value: j.get("value").f64() as f32,
            reward: j.get("return").f64() as f32,
            exploration: j.get("exploration").f64() as f32,
            step: j.get("step").i64(),
            cash: j.get("cash").f64() as f32,
            elapsed: j.get("elapsed").i64(),
            cash_delta: j.get("cash_delta").f64() as f32,
            advantage: j.get("advantage").f64() as f32,
            mc_return: j.get("mc_return").f64() as f32,
        };
        if !(0. ..=1.).contains(&row.exploration) || row.step < 0 || row.elapsed < 0 {
            return Err("invalid sample time or exploration".into());
        }
        if ![
            row.logp,
            row.value,
            row.reward,
            row.exploration,
            row.cash,
            row.cash_delta,
            row.advantage,
            row.mc_return,
        ]
        .iter()
        .all(|v| v.is_finite())
        {
            return Err("non-finite training target".into());
        }
        Ok(row)
    }
}
pub struct Batch {
    pub groups: Tensor,
    pub proposal: Tensor,
    pub exploration: Tensor,
    pub context: Tensor,
    pub candidates: Tensor,
    pub mask: Tensor,
    pub width: usize,
}
impl Batch {
    pub fn new(rows: &[Sample], device: i32) -> Result<Self, String> {
        let width = rows
            .iter()
            .map(|r| r.features.len())
            .max()
            .ok_or("empty tensor batch")?;
        let n = rows.len();
        let mut context = Vec::with_capacity(n * CONTEXT);
        let mut candidates = vec![0.; n * width * CANDIDATE];
        let mut mask = vec![0; n * width];
        let mut groups = vec![0; n * width];
        let mut proposal = vec![0.; n * width];
        let mut exploration = vec![0.; n];
        for (i, row) in rows.iter().enumerate() {
            if row.context.len() != CONTEXT
                || row.features.is_empty()
                || row.features.iter().any(|v| v.len() != CANDIDATE)
            {
                return Err("invalid features".into());
            }
            if !(0. ..=1.).contains(&row.exploration)
                || row.features.iter().any(|f| {
                    f[31] < 0.
                        || f[31] >= GROUPS as f32
                        || f[31].fract() != 0.
                        || !f[31].is_finite()
                })
            {
                return Err("invalid category or exploration probability".into());
            }
            let q = exploration_proposal(&row.features);
            let has_proposal = q.iter().sum::<f32>() > 0.;
            exploration[i] = if has_proposal { row.exploration } else { 0. };
            context.extend_from_slice(&row.context);
            for (j, feat) in row.features.iter().enumerate() {
                let from = (i * width + j) * CANDIDATE;
                candidates[from..from + CANDIDATE].copy_from_slice(feat);
                mask[i * width + j] = 1;
                groups[i * width + j] = feat[31] as i64;
                proposal[i * width + j] = q[j];
            }
        }
        Ok(Self {
            groups: Tensor::integers(&groups, &[n as i64, width as i64], device, false)?,
            proposal: Tensor::floats(&proposal, &[n as i64, width as i64], device, false)?,
            exploration: Tensor::floats(&exploration, &[n as i64, 1], device, false)?,
            context: Tensor::floats(&context, &[n as i64, CONTEXT as i64], device, false)?,
            candidates: Tensor::floats(
                &candidates,
                &[n as i64, width as i64, CANDIDATE as i64],
                device,
                false,
            )?,
            mask: Tensor::integers(&mask, &[n as i64, width as i64], device, true)?,
            width,
        })
    }
    pub fn select(&self, indices: &Tensor) -> Result<Self, String> {
        let select = |t: &Tensor| Tensor::operation(22, &[t, indices], &[0], &[]);
        Ok(Self {
            groups: select(&self.groups)?,
            proposal: select(&self.proposal)?,
            exploration: select(&self.exploration)?,
            context: select(&self.context)?,
            candidates: select(&self.candidates)?,
            mask: select(&self.mask)?,
            width: self.width,
        })
    }
}

/// Uniform over productive categories, then uniform within each category.
pub fn exploration_proposal(features: &[Vec<f32>]) -> Vec<f32> {
    let mut counts = [0usize; GROUPS];
    for f in features {
        let g = f[31] as usize;
        if matches!(g, 1..=4 | 8..=15) {
            counts[g] += 1;
        }
    }
    let present = counts.iter().filter(|&&n| n > 0).count();
    features
        .iter()
        .map(|f| {
            let n = counts[f[31] as usize];
            if n == 0 {
                0.
            } else {
                1. / (present * n) as f32
            }
        })
        .collect()
}
pub fn hierarchical_argmax(logp: &[f32], features: &[Vec<f32>]) -> usize {
    let mut mass = [0.; GROUPS];
    for (lp, f) in logp.iter().zip(features) {
        mass[f[31] as usize] += lp.exp();
    }
    let mut group = 0;
    for g in 1..GROUPS {
        if mass[g] > mass[group] {
            group = g;
        }
    }
    (0..features.len())
        .filter(|&i| features[i][31] as usize == group)
        .max_by(|&a, &b| logp[a].total_cmp(&logp[b]).then_with(|| b.cmp(&a)))
        .unwrap()
}
/// Undiscounted money objective, with a trace decay measured in environment steps.
/// Same-step investment/worker decisions have dt=0 and retain the full trace.
pub fn cash_returns(rows: &mut [Sample], final_step: i64, final_cash: f32, lambda: f32) {
    let (mut next_step, mut next_cash, mut next_value, mut next_adv) =
        (final_step, final_cash, 0., 0.);
    for r in rows.iter_mut().rev() {
        r.elapsed = next_step - r.step;
        r.cash_delta = next_cash - r.cash;
        r.advantage =
            r.cash_delta + next_value - r.value + lambda.powi(r.elapsed as i32) * next_adv;
        r.reward = r.value + r.advantage;
        r.mc_return = final_cash - r.cash;
        next_step = r.step;
        next_cash = r.cash;
        next_value = r.value;
        next_adv = r.advantage;
    }
}

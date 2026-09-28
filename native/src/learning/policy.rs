//! Hierarchical policy, explicit exploration behavior likelihood and native PPO.
use super::tensor::{NoGrad, Tensor};
use kagg_engine::json::Json;
pub const CONTEXT: usize = crate::pipeline::CONTEXT;
pub const CANDIDATE: usize = 32;
pub const EVENT_INPUT_ENCODING: &str = "event-centered-bounded-v1";
pub const GROUPS: usize = crate::pipeline::GROUPS;
const SPECS: [(&str, &[i64]); 20] = [
    ("context.0.weight", &[64, CONTEXT as i64]),
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
    ("group.weight", &[GROUPS as i64, 64]),
    ("group.bias", &[GROUPS as i64]),
    ("critic_context.0.weight", &[64, CONTEXT as i64]),
    ("critic_context.0.bias", &[64]),
    ("critic_context.2.weight", &[64, 64]),
    ("critic_context.2.bias", &[64]),
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
    pub market_mode: crate::pipeline::trading::MarketMode,
    pub plan_residual: bool,
    pub event_input_scaling: bool,
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
            market_mode: Default::default(),
            plan_residual: false,
            event_input_scaling: false,
        })
    }
    /// Start exactly at the executable planner in greedy mode. The fixed prior
    /// is visible in candidate feature 30; the network learns unrestricted logit
    /// corrections. This is still a learned, on-policy categorical distribution.
    pub fn plans(device: i32, seed: u64, lr: f64) -> Result<Self, String> {
        let mut p = Self::mixed_routes(device, seed, lr)?;
        p.plan_residual = true;
        for i in [8, 9] {
            let zero = p.parameters[i].value.unary(26)?;
            p.parameters[i].value.copy_from(&zero)?;
        }
        Ok(p)
    }
    /// New event learners only. Legacy weights retain their original input units.
    pub fn event_plans(device: i32, seed: u64, lr: f64) -> Result<Self, String> {
        let mut p = Self::plans(device, seed, lr)?;
        p.event_input_scaling = true;
        Ok(p)
    }
    fn bounded(x: &Tensor) -> Result<Tensor, String> {
        // Smooth signed compression: retains order and has no clipping boundary.
        let denominator = x.unary(24)?.scalar(30, 1.)?.unary(28)?;
        x.binary(15, &denominator)
    }
    fn actor_inputs(&self, b: &Batch) -> Result<(Tensor, Tensor), String> {
        if !self.event_input_scaling {
            return Ok((b.context.unary(34)?, b.candidates.unary(34)?));
        }
        let mut scale = vec![1.; CONTEXT];
        let mut offset = vec![0.; CONTEXT];
        // Raw event rows store market stock /100. Initial stock is 10000:
        // (raw - 100)/100 == (actual_stock - 10000)/10000.
        for i in 0..9 {
            scale[17 + 3 * i] = 0.01;
            offset[17 + 3 * i] = -1.;
        }
        let scale = Tensor::floats(&scale, &[CONTEXT as i64], self.device, false)?;
        let offset = Tensor::floats(&offset, &[CONTEXT as i64], self.device, false)?;
        let context = Self::bounded(&b.context.binary(14, &scale)?.binary(12, &offset)?)?;
        Ok((context, Self::bounded(&b.candidates)?))
    }
    /// Diagnostics on observed rows, using the SAME transform as train/inference.
    pub fn input_diagnostics(&self, rows: &[Sample]) -> Result<Json, String> {
        if rows.is_empty() {
            return Ok(Json::Null);
        }
        let _guard = NoGrad::new();
        let b = Batch::new(&rows[..rows.len().min(128)], self.device)?;
        let (x, features) = self.actor_inputs(&b)?;
        let z1 = self.linear(&x, 0)?.unary(1)?;
        let z2 = self.linear(&z1, 2)?.unary(1)?;
        let first = z1.data()?;
        let second = z2.data()?;
        let mut span = 0f32;
        for col in 0..64 {
            let vs: Vec<_> = second.iter().skip(col).step_by(64).copied().collect();
            span = span.max(
                vs.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                    - vs.iter().copied().fold(f32::INFINITY, f32::min),
            );
        }
        let max_abs = |xs: Vec<f32>| xs.into_iter().map(f32::abs).fold(0f32, f32::max) as f64;
        Ok(Json::Obj(vec![
            (
                "encoding".into(),
                Json::Str(
                    if self.event_input_scaling {
                        EVENT_INPUT_ENCODING
                    } else {
                        "legacy-raw"
                    }
                    .into(),
                ),
            ),
            ("rows".into(), Json::Num(rows.len().min(128) as f64)),
            ("context_abs_max".into(), Json::Num(max_abs(x.data()?))),
            (
                "candidate_abs_max".into(),
                Json::Num(max_abs(features.data()?)),
            ),
            (
                "context_layer1_saturated_fraction".into(),
                Json::Num(
                    first.iter().filter(|x| x.abs() > 0.99).count() as f64 / first.len() as f64,
                ),
            ),
            ("context_layer2_max_span".into(), Json::Num(span as f64)),
            (
                "context_gradient_norm".into(),
                Json::Num(self.parameters[0].value.unary(33)?.unary(37)?.value()?),
            ),
        ]))
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
        let (context, candidates) = self.actor_inputs(b)?;
        let z = self
            .linear(&self.linear(&context, 0)?.unary(1)?, 2)?
            .unary(1)?;
        let v = self.linear(&candidates, 4)?.unary(1)?;
        let expanded = Tensor::operation(5, &[&z.dim(3, 1)?], &[-1, b.width as i64, -1], &[])?;
        let joined = Tensor::operation(2, &[&v, &expanded], &[-1], &[])?;
        let mut scores = self
            .linear(&self.linear(&joined, 6)?.unary(1)?, 8)?
            .dim(4, -1)?;
        if self.plan_residual {
            // Select candidate feature 30 without making a Python-side tensor.
            let index = Tensor::integers(&[30], &[1], self.device, false)?;
            let prior = Tensor::operation(22, &[&b.candidates, &index], &[2], &[])?.dim(4, 2)?;
            scores = scores.binary(12, &prior)?;
        }
        let lp = if self.plan_residual {
            Tensor::operation(46, &[&scores, &b.mask], &[], &[])?
        } else {
            let head = self.linear(&z, 14)?;
            Tensor::operation(44, &[&scores, &head, &b.groups, &b.mask], &[], &[])?
        };
        // Value regression must not update the actor context representation.
        let critic_z = self
            .linear(&self.linear(&context, 16)?.unary(1)?, 18)?
            .unary(1)?;
        let value = self
            .linear(&self.linear(&critic_z, 10)?.unary(1)?, 12)?
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
        let behavior = if deterministic {
            Vec::new()
        } else {
            self.behavior(&lp, &batch)?.data()?
        };
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
        let scaling = match weights.get("_event_input_encoding") {
            Json::Null => false,
            Json::Str(s) if s == EVENT_INPUT_ENCODING => true,
            _ => return Err("unknown event input encoding".into()),
        };
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
        self.event_input_scaling = scaling;
        Ok(())
    }
    pub fn weights_json(&self) -> Result<Json, String> {
        let mut result = Json::Obj(
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
        );
        if self.event_input_scaling {
            result.set_path(
                "_event_input_encoding",
                Json::Str(EVENT_INPUT_ENCODING.into()),
            );
        }
        Ok(result)
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
                Json::Str("mixed-production-v8-ppo-v1".into()),
            ),
            (
                "policy_contract".into(),
                Json::Str(crate::pipeline::ENCODING.into()),
            ),
            ("iteration".into(), Json::Str(iteration.to_string())),
            ("rng".into(), Json::Str(rng.0.to_string())),
            ("rng_algorithm".into(), Json::Str("splitmix64".into())),
            ("learning_rate".into(), Json::Num(self.lr)),
            (
                "market_mode".into(),
                Json::Str(self.market_mode.name().into()),
            ),
            ("plan_residual".into(), Json::Bool(self.plan_residual)),
            ("weights".into(), self.weights_json()?),
            ("optimizer".into(), Json::Obj(optimizer)),
        ]))
    }
    pub fn restore(&mut self, checkpoint: &Json) -> Result<(u64, Rng), String> {
        if checkpoint.get("schema").str() != "mixed-production-v8-ppo-v1"
            || checkpoint.get("policy_contract").str() != crate::pipeline::ENCODING
            || checkpoint.get("rng_algorithm").str() != "splitmix64"
        {
            return Err("incompatible native checkpoint: market policy with independent actor/critic requires a new run or a matching checkpoint; older observation/action checkpoints are not migrated".into());
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
        let market_mode =
            crate::pipeline::trading::MarketMode::parse(checkpoint.get("market_mode").str())?;
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
        self.market_mode = market_mode;
        self.plan_residual = matches!(checkpoint.get("plan_residual"), Json::Bool(true));
        Ok((iteration, rng))
    }
    fn is_critic(name: &str) -> bool {
        name.starts_with("critic_context.") || name.starts_with("value.")
    }
    pub(crate) fn adam(&mut self) -> Result<(), String> {
        let _guard = NoGrad::new();
        // Clip separately: a large critic gradient must not scale down actor updates.
        for critic in [false, true] {
            let mut grads = Vec::new();
            let mut norms = Vec::new();
            for (i, p) in self.parameters.iter().enumerate() {
                if Self::is_critic(p.name) == critic && p.value.has_grad() {
                    let g = p.value.unary(33)?;
                    norms.push(g.unary(37)?.dim(3, 0)?);
                    grads.push((i, g));
                }
            }
            if grads.is_empty() {
                continue;
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
                policy_loss: 0.,
                value_loss: 0.,
                entropy: 0.,
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
            policy_loss: policy.value()?,
            value_loss: values.value()?,
            entropy: entropy.value()?,
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
            policy_loss: 0.,
            value_loss: 0.,
            entropy: 0.,
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
                metrics.policy_loss += result.policy_loss;
                metrics.value_loss += result.value_loss;
                metrics.entropy += result.entropy;
            }
        }
        if metrics.updates > 0 {
            metrics.loss /= metrics.updates as f64;
            metrics.mean_kl /= metrics.updates as f64;
            metrics.policy_loss /= metrics.updates as f64;
            metrics.value_loss /= metrics.updates as f64;
            metrics.entropy /= metrics.updates as f64;
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
    pub policy_loss: f64,
    pub value_loss: f64,
    pub entropy: f64,
    pub kl: f64,
    pub stopped: bool,
}
pub struct Metrics {
    pub samples: usize,
    pub updates: usize,
    pub loss: f64,
    pub policy_loss: f64,
    pub value_loss: f64,
    pub entropy: f64,
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

/// Uniform over productive/market categories (including market hold), then candidates.
pub fn exploration_proposal(features: &[Vec<f32>]) -> Vec<f32> {
    let mut counts = [0usize; GROUPS];
    for f in features {
        let g = f[31] as usize;
        if matches!(g, 1..=4 | 8..=18) {
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

#[cfg(test)]
mod independent_tests {
    use super::*;

    fn batch(device: i32) -> Batch {
        let rows: Vec<_> = (0..3)
            .map(|i| {
                let mut features = vec![vec![0.; CANDIDATE]; 3];
                features[1][31] = 2.;
                features[1][2] = 1.;
                features[2][31] = 14.;
                features[2][1] = 1.;
                Sample {
                    context: (0..CONTEXT).map(|j| ((i + j) % 7) as f32 / 7.).collect(),
                    features,
                    ..Sample::default()
                }
            })
            .collect();
        Batch::new(&rows, device).unwrap()
    }

    #[test]
    fn actor_and_critic_gradients_and_optimizer_states_are_isolated() {
        super::super::tensor::threads(1);
        let devices = if std::env::var_os("ROUTE_RL_TEST_CUDA").is_some() {
            vec![-1, 0]
        } else {
            vec![-1]
        };
        for device in devices {
            for train_critic in [false, true] {
                let mut p = Policy::mixed_routes(device, 432, 1e-4).unwrap();
                let b = batch(device);
                let before = p.weights_json().unwrap();
                let (lp, v) = p.forward(&b).unwrap();
                let value_before = v.data().unwrap();
                let lp_before = lp.data().unwrap();
                let loss = if train_critic {
                    v.scalar(30, -3.)
                        .unwrap()
                        .unary(24)
                        .unwrap()
                        .unary(18)
                        .unwrap()
                } else {
                    lp.unary(18).unwrap().unary(32).unwrap()
                };
                loss.backward().unwrap();
                for param in &p.parameters {
                    assert_eq!(
                        param.value.has_grad(),
                        Policy::is_critic(param.name) == train_critic,
                        "{}",
                        param.name
                    );
                }
                p.adam().unwrap();
                let after = p.weights_json().unwrap();
                for param in &p.parameters {
                    if Policy::is_critic(param.name) != train_critic {
                        assert_eq!(before.get(param.name), after.get(param.name));
                        assert_eq!(param.step, 0);
                        assert!(param.m.data().unwrap().iter().all(|&x| x == 0.));
                        assert!(param.v.data().unwrap().iter().all(|&x| x == 0.));
                    }
                }
                let (lp, v) = p.forward(&b).unwrap();
                if train_critic {
                    assert_eq!(lp_before, lp.data().unwrap());
                    assert_ne!(value_before, v.data().unwrap());
                } else {
                    assert_eq!(value_before, v.data().unwrap());
                    assert_ne!(lp_before, lp.data().unwrap());
                }
            }
        }
    }

    #[test]
    fn critic_gradient_scale_does_not_change_actor_adam_updates() {
        super::super::tensor::threads(1);
        let mut a = Policy::mixed_routes(-1, 765, 1e-4).unwrap();
        let mut b = Policy::mixed_routes(-1, 765, 1e-4).unwrap();
        let batch = batch(-1);
        for _ in 0..3 {
            for (p, scale) in [(&mut a, 1.), (&mut b, 10000.)] {
                for param in &mut p.parameters {
                    param.value.zero_grad();
                }
                let (lp, v) = p.forward(&batch).unwrap();
                let actor = lp.unary(18).unwrap().unary(32).unwrap();
                let critic = v
                    .scalar(30, -3.)
                    .unwrap()
                    .unary(24)
                    .unwrap()
                    .unary(18)
                    .unwrap()
                    .scalar(31, scale)
                    .unwrap();
                actor.binary(12, &critic).unwrap().backward().unwrap();
                p.adam().unwrap();
            }
            for (a, b) in a.parameters.iter().zip(&b.parameters) {
                if !Policy::is_critic(a.name) {
                    assert_eq!(a.value.data().unwrap(), b.value.data().unwrap());
                    assert_eq!(a.m.data().unwrap(), b.m.data().unwrap());
                    assert_eq!(a.v.data().unwrap(), b.v.data().unwrap());
                    assert_eq!(a.step, b.step);
                }
            }
        }
        let checkpoint = a.checkpoint(3, &Rng(19)).unwrap();
        let mut restored = Policy::mixed_routes(-1, 99, 1e-4).unwrap();
        restored.restore(&checkpoint).unwrap();
        assert_eq!(checkpoint, restored.checkpoint(3, &Rng(19)).unwrap());
        let mut old = checkpoint;
        old.set_path(
            "policy_contract",
            Json::Str("mixed-routes-96x32-hierarchy-cash-v3".into()),
        );
        assert!(restored
            .restore(&old)
            .err()
            .unwrap()
            .contains("independent actor/critic"));
    }
}

#[cfg(test)]
mod event_input_tests {
    use super::*;
    fn rows() -> Vec<Sample> {
        (0..2)
            .map(|i| {
                let mut context = vec![0.; CONTEXT];
                for k in 0..9 {
                    context[17 + 3 * k] = 100.;
                }
                context[1] = if i == 0 { 0.05 } else { 5. };
                let mut f = vec![vec![0.; 32]; 2];
                f[0][0] = 1.;
                f[1][2] = 1.;
                f[0][31] = 1.;
                f[1][31] = 1.;
                Sample {
                    context,
                    features: f,
                    ..Default::default()
                }
            })
            .collect()
    }
    #[test]
    fn event_input_normalization_is_versioned_and_preserves_market_differences() {
        crate::learning::tensor::worker_threads();
        let old = Policy::plans(-1, 220000, 0.0003).unwrap();
        let new = Policy::event_plans(-1, 220000, 0.0003).unwrap();
        let mut r = rows();
        r[0].context[17] = 90.;
        r[1].context[17] = 110.;
        let b = Batch::new(&r, -1).unwrap();
        let (old_x, _) = old.actor_inputs(&b).unwrap();
        assert_eq!(
            old_x.data().unwrap(),
            r.iter().flat_map(|r| r.context.clone()).collect::<Vec<_>>()
        );
        let (new_x, f) = new.actor_inputs(&b).unwrap();
        let xs = new_x.data().unwrap();
        assert!(xs[17] < -0.09 && xs[CONTEXT + 17] > 0.09);
        assert!(xs
            .iter()
            .chain(f.data().unwrap().iter())
            .all(|v| v.abs() <= 1.));
        let health = new.input_diagnostics(&r).unwrap();
        assert!(health.get("context_layer1_saturated_fraction").f64() < 0.1);
        assert!(health.get("context_layer2_max_span").f64() > 0.01);
        let mut restore = Policy::plans(-1, 1, 0.0003).unwrap();
        restore.load_weights(&new.weights_json().unwrap()).unwrap();
        assert!(restore.event_input_scaling);
        assert_eq!(
            new.forward(&b).unwrap().0.data().unwrap(),
            restore.forward(&b).unwrap().0.data().unwrap()
        );
        restore.load_weights(&old.weights_json().unwrap()).unwrap();
        assert!(!restore.event_input_scaling);
        assert_eq!(
            old.forward(&b).unwrap().0.data().unwrap(),
            restore.forward(&b).unwrap().0.data().unwrap()
        );
        let mut bad = new.weights_json().unwrap();
        bad.set_path("_event_input_encoding", Json::Str("unknown".into()));
        assert!(restore.load_weights(&bad).is_err());
    }
    #[test]
    fn event_input_context_can_learn_opposite_choices_with_identical_candidates() {
        crate::learning::tensor::worker_threads();
        let mut p = Policy::event_plans(-1, 220000, 0.003).unwrap();
        let r = rows();
        let b = Batch::new(&r, -1).unwrap();
        let target = Tensor::floats(&[-1., 1.], &[2], -1, false).unwrap();
        let mut gradient_seen = false;
        for _ in 0..200 {
            let lp = p.forward(&b).unwrap().0;
            let a = Tensor::operation(27, &[&lp], &[1, 0, 1], &[])
                .unwrap()
                .dim(4, 1)
                .unwrap();
            let c = Tensor::operation(27, &[&lp], &[1, 1, 1], &[])
                .unwrap()
                .dim(4, 1)
                .unwrap();
            let loss = c
                .binary(13, &a)
                .unwrap()
                .binary(13, &target)
                .unwrap()
                .unary(24)
                .unwrap()
                .unary(18)
                .unwrap();
            for v in &mut p.parameters {
                v.value.zero_grad();
            }
            loss.backward().unwrap();
            gradient_seen |= p.parameters[0]
                .value
                .unary(33)
                .unwrap()
                .unary(37)
                .unwrap()
                .value()
                .unwrap()
                > 1e-6;
            p.adam().unwrap();
        }
        assert!(gradient_seen);
        let d = p.infer(&r, true, &mut Rng(0)).unwrap();
        assert_eq!((d[0].action, d[1].action), (0, 1));
        let mut restored = Policy::plans(-1, 1, 0.0003).unwrap();
        restored
            .restore(&p.checkpoint(200, &Rng(123)).unwrap())
            .unwrap();
        assert!(restored.event_input_scaling);
        assert_eq!(
            p.forward(&b).unwrap().0.data().unwrap(),
            restored.forward(&b).unwrap().0.data().unwrap()
        );
    }
}

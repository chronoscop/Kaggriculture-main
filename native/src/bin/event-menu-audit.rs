//! Read-only/offline investigation of archived complete menus; no deployments.
#[cfg(feature = "train")]
mod app {
    use kagg_engine::json::{self, Json};
    use route_rl_native::learning::{
        event_sets,
        plan_compare::{Bank, Pair},
        policy::{Policy, Rng, Sample},
        tensor,
    };
    use std::{collections::BTreeSet, io::Write, path::Path};
    fn read(p: &str) -> Result<Json, String> {
        json::parse(&std::fs::read_to_string(p).map_err(|e| e.to_string())?)
    }
    fn load(path: &str) -> Result<Vec<Pair>, String> {
        let mut out = vec![];
        let mut seen = BTreeSet::new();
        for line in std::fs::read_to_string(path)
            .map_err(|e| e.to_string())?
            .lines()
        {
            let e = json::parse(line)?;
            if !seen.insert(e.get("candidate_set_id").str().to_string()) {
                continue;
            }
            let mut row = Sample::parse(e.get("full_row"))?;
            let b = e.get("alternative_index").i64() as usize;
            row.features = vec![row.features[0].clone(), row.features[b].clone()];
            let mut p = Pair {
                row,
                target: vec![0.5, 0.5],
                gain: 0.,
                iteration: e.get("iteration").i64() as u64,
                seed: e.get("seed").i64(),
                seat: e.get("seat").i64() as usize,
                opponent: e.get("opponent").i64() as usize,
                bucket: 0,
                evidence: e,
            };
            p.relabel_match_score()?;
            event_sets::validate(&p)?;
            out.push(p);
        }
        Ok(out)
    }
    fn predictions(p: &Policy, rs: &[Pair], path: &Path) -> Result<(), String> {
        let rows = rs
            .iter()
            .map(|r| Sample::parse(r.evidence.get("full_row")))
            .collect::<Result<Vec<_>, _>>()?;
        let mut file = std::fs::File::create(path).map_err(|e| e.to_string())?;
        for start in (0..rows.len()).step_by(64) {
            let end = (start + 64).min(rows.len());
            let batch = &rows[start..end];
            let pr = p.distributions(batch)?;
            let shuffled = batch
                .iter()
                .enumerate()
                .map(|(i, r)| Sample {
                    context: rows[(start + i + rows.len() / 2 + 1) % rows.len()]
                        .context
                        .clone(),
                    ..r.clone()
                })
                .collect::<Vec<_>>();
            let spr = p.distributions(&shuffled)?;
            for (i, (p, s)) in pr.into_iter().zip(spr).enumerate() {
                let j = Json::Obj(vec![
                    (
                        "id".into(),
                        rs[start + i].evidence.get("candidate_set_id").clone(),
                    ),
                    (
                        "probabilities".into(),
                        Json::Arr(p.into_iter().map(|x| Json::Num(x as f64)).collect()),
                    ),
                    (
                        "shuffled_context_probabilities".into(),
                        Json::Arr(s.into_iter().map(|x| Json::Num(x as f64)).collect()),
                    ),
                ]);
                writeln!(file, "{}", j.dump()).map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }
    pub fn run() -> Result<(), String> {
        let a = std::env::args().skip(1).collect::<Vec<_>>();
        if a.len() != 5 {
            return Err("TRAIN HELDOUT CHECKPOINT NEW_OUT EPOCHS".into());
        }
        let train = load(&a[0])?;
        let test = load(&a[1])?;
        let epochs = a[4].parse::<usize>().map_err(|_| "epochs")?;
        if epochs > 1000 {
            return Err("audit capped at 1000 epochs".into());
        }
        if epochs > 0 && a[0] != a[1] && train.iter().any(|r| test.iter().any(|s| r.seed == s.seed))
        {
            return Err("seed overlap".into());
        }
        let dir = Path::new(&a[3]);
        if dir.exists() {
            return Err("new output directory required".into());
        }
        tensor::threads(1);
        tensor::worker_threads();
        let mut p = Policy::event_plans(-1, 20260930, 0.0003)?;
        let c = read(&a[2])?;
        p.load_weights(if c.get("model").is_obj() {
            c.get("model").get("weights")
        } else {
            &c
        })?;
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        predictions(&p, &train, &dir.join("train_before.jsonl"))?;
        predictions(&p, &test, &dir.join("test_before.jsonl"))?;
        let mut log = std::fs::File::create(dir.join("fit.jsonl")).map_err(|e| e.to_string())?;
        let mut rng = Rng(20260930);
        for e in (0..epochs).step_by(25) {
            let report = event_sets::update(
                &mut p,
                &train,
                &Bank::default(),
                train[0].incumbent_revision()?,
                25.min(epochs - e),
                64,
                &mut rng,
            )?;
            writeln!(
                log,
                "{}",
                Json::Obj(vec![
                    ("epoch".into(), Json::Num((e + 25).min(epochs) as f64)),
                    ("report".into(), report)
                ])
                .dump()
            )
            .map_err(|e| e.to_string())?;
        }
        predictions(&p, &train, &dir.join("train_after.jsonl"))?;
        predictions(&p, &test, &dir.join("test_after.jsonl"))?;
        let summary = Json::Obj(vec![
            ("train".into(), event_sets::metrics(&p, &train)?),
            ("heldout".into(), event_sets::metrics(&p, &test)?),
        ]);
        std::fs::write(dir.join("summary.json"), summary.dump()).map_err(|e| e.to_string())?;
        println!("{}", summary.dump());
        Ok(())
    }
}
#[cfg(feature = "train")]
fn main() {
    if let Err(e) = app::run() {
        eprintln!("{e}");
        std::process::exit(1)
    }
}
#[cfg(not(feature = "train"))]
fn main() {
    eprintln!("requires train feature")
}

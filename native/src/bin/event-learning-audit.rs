//! Offline audit only: fit archived paired terminal outcomes, never run the engine
//! or alter accepted deployments. Uses the exact event Policy and training loss.
#[cfg(feature = "train")]
mod app {
    use kagg_engine::json::{self, Json};
    use route_rl_native::learning::{
        plan_compare::{self, Bank, Pair},
        policy::{Policy, Rng, Sample},
        tensor,
    };
    use std::{io::Write, path::Path, time::Instant};
    fn read(path: &str) -> Result<Json, String> {
        json::parse(&std::fs::read_to_string(path).map_err(|e| e.to_string())?)
    }
    fn load(path: &str) -> Result<Vec<Pair>, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        text.lines()
            .map(|line| {
                let e = json::parse(line)?;
                if e.get("objective").str() != plan_compare::MATCH_SCORE_OBJECTIVE {
                    return Err("audit accepts terminal match score labels only".into());
                }
                let mut row = Sample::parse(e.get("full_row"))?;
                let a = e.get("reference_index").i64() as usize;
                let b = e.get("alternative_index").i64() as usize;
                if a == b || a >= row.features.len() || b >= row.features.len() {
                    return Err("invalid recorded pair indices".into());
                }
                row.features = vec![row.features[a].clone(), row.features[b].clone()];
                row.action = 0;
                row.exploration = 0.;
                let mut pair = Pair {
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
                pair.relabel_match_score()?;
                Pair::parse(&pair.json())
            })
            .collect()
    }
    fn write(path: &Path, j: &Json) -> Result<(), String> {
        std::fs::write(path, j.dump()).map_err(|e| e.to_string())
    }
    fn predictions(p: &Policy, pairs: &[Pair], path: &Path) -> Result<(), String> {
        let mut f = std::fs::File::create(path).map_err(|e| e.to_string())?;
        for chunk in pairs.chunks(64) {
            let rows: Vec<_> = chunk.iter().map(|r| r.row.clone()).collect();
            let probs = p.distributions(&rows)?;
            let full = chunk
                .iter()
                .map(|r| Sample::parse(r.evidence.get("full_row")))
                .collect::<Result<Vec<_>, _>>()?;
            let decisions = p.infer(&full, true, &mut Rng(0))?;
            for ((r, pr), d) in chunk.iter().zip(probs).zip(decisions) {
                let mut v = Json::Obj(vec![]);
                for k in [
                    "seed",
                    "seat",
                    "opponent",
                    "step",
                    "iteration",
                    "source",
                    "stage",
                    "prefix_id",
                    "reference_index",
                    "alternative_index",
                    "reference_plan",
                    "alternative_plan",
                    "reference_cash",
                    "alternative_cash",
                ] {
                    v.set_path(k, r.evidence.get(k).clone());
                }
                v.set_path("target", Json::Num(r.improvement_target()? as f64));
                v.set_path("alternative_probability", Json::Num(pr[1] as f64));
                v.set_path("pair_choose_alternative", Json::Bool(pr[1] > pr[0]));
                v.set_path("full_argmax", Json::Num(d.action as f64));
                writeln!(f, "{}", v.dump()).map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }
    pub fn run() -> Result<(), String> {
        let a: Vec<_> = std::env::args().skip(1).collect();
        if a.len() != 6 {
            return Err("usage: event-learning-audit TRAIN.jsonl HELDOUT.jsonl INITIAL_WEIGHTS.json NEW_OUT EPOCHS cpu|cuda".into());
        }
        let epochs = a[4].parse::<usize>().map_err(|_| "bad epochs")?;
        if !(1..=2000).contains(&epochs) {
            return Err("epochs out of range".into());
        }
        let device = match a[5].as_str() {
            "cpu" => -1,
            "cuda" => 0,
            _ => return Err("invalid device".into()),
        };
        let out = Path::new(&a[3]);
        if out.exists() {
            return Err("audit output must be a new directory".into());
        }
        let train = load(&a[0])?;
        let heldout = load(&a[1])?;
        let revision = train
            .first()
            .ok_or("empty train set")?
            .incumbent_revision()?;
        if train
            .iter()
            .chain(&heldout)
            .any(|r| r.incumbent_revision().ok() != Some(revision))
        {
            return Err("mixed continuation versions".into());
        }
        // Memorization deliberately uses the same file for fit and readout.
        if a[0] != a[1]
            && train
                .iter()
                .any(|r| heldout.iter().any(|v| v.seed == r.seed))
        {
            return Err("seed leakage".into());
        }
        tensor::threads(1);
        tensor::worker_threads();
        let mut p = Policy::event_plans(device, 20260929, 0.0003)?;
        p.load_weights(&read(&a[2])?)?;
        if !p.event_input_scaling {
            return Err("normalized initial weights required".into());
        }
        std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
        predictions(&p, &train, &out.join("train_before.jsonl"))?;
        predictions(&p, &heldout, &out.join("heldout_before.jsonl"))?;
        let mut log =
            std::fs::File::create(out.join("training.jsonl")).map_err(|e| e.to_string())?;
        let mut rng = Rng(20260929);
        let start = Instant::now();
        let mut completed = 0;
        while completed < epochs {
            let n = 25.min(epochs - completed);
            let mut report = plan_compare::update_improvement(
                &mut p,
                &train,
                &Bank::default(),
                revision,
                n,
                64,
                &mut rng,
            )?;
            completed += n;
            report.set_path("epochs", Json::Num(completed as f64));
            report.set_path("seconds", Json::Num(start.elapsed().as_secs_f64()));
            writeln!(log, "{}", report.dump()).map_err(|e| e.to_string())?;
            log.flush().map_err(|e| e.to_string())?;
            println!(
                "epochs={completed} train_accuracy={:.4} train_mse={:.6} seconds={:.1}",
                report.get("pair_accuracy_after").f64(),
                report.get("pair_mse_after").f64(),
                start.elapsed().as_secs_f64()
            );
        }
        // Fixed final checkpoint. No heldout-based early stopping or selection.
        write(&out.join("audit_weights.json"), &p.weights_json()?)?;
        predictions(&p, &train, &out.join("train_after.jsonl"))?;
        predictions(&p, &heldout, &out.join("heldout_after.jsonl"))?;
        println!("audit complete; no games, promotion, or source checkpoint writes");
        Ok(())
    }
}
#[cfg(feature = "train")]
fn main() {
    if let Err(e) = app::run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
#[cfg(not(feature = "train"))]
fn main() {
    eprintln!("build with --features train");
    std::process::exit(1);
}

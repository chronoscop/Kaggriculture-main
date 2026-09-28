//! Persistent observation/action JSONL runner for the accepted plan portfolio.
//! Training weights are proposals and are never loaded for deployment.
#[cfg(feature = "train")]
fn load_deployment(
    checkpoint: &kagg_engine::json::Json,
) -> Result<
    (
        route_rl_native::pipeline::plan_prototype::Config,
        route_rl_native::pipeline::event_portfolio::Runtime,
    ),
    String,
> {
    use route_rl_native::pipeline::{
        event_portfolio::{Portfolio, Runtime},
        plan_prototype::Config,
    };
    if checkpoint.get("schema").str() != "event-plan-improvement-v4"
        || checkpoint.get("policy_contract").str()
            != route_rl_native::pipeline::event_portfolio::CONTRACT
    {
        return Err("requires an event-plan-improvement-v4 checkpoint with the current event contract and accepted deployment".into());
    }
    let config = Config::parse(checkpoint.get("config"))?;
    let accepted = Portfolio::parse(checkpoint.get("deployment"))?;
    Ok((config, Runtime::load(&accepted, -1)?))
}

#[cfg(feature = "train")]
fn run() -> Result<(), String> {
    use kagg_engine::{json, obsstate};
    use route_rl_native::{
        learning::tensor,
        pipeline::{
            event_portfolio::Deployed,
            executor::{action_json, Observation},
        },
    };
    use std::io::{self, BufRead, Write};
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 || args[0] != "--checkpoint" {
        return Err("event-agent --checkpoint runs/event_plan_trial/best.json".into());
    }
    tensor::threads(1);
    let checkpoint = json::parse(&std::fs::read_to_string(&args[1]).map_err(|e| e.to_string())?)?;
    let (config, runtime) = load_deployment(&checkpoint)?;
    let mut agents = [Deployed::new(config.clone()), Deployed::new(config.clone())];
    let mut last = [-1, -1];
    let stdout = io::stdout();
    let mut output = stdout.lock();
    for line in io::stdin().lock().lines() {
        let j = json::parse(&line.map_err(|e| e.to_string())?)?;
        let obs = if j.get("observation").is_obj() {
            j.get("observation")
        } else {
            &j
        };
        let (s, seat) = obsstate::from_obs(obs).ok_or("invalid observation")?;
        let o = Observation::from_state(&s, seat);
        if o.step <= last[seat] {
            agents[seat] = Deployed::new(config.clone());
        }
        last[seat] = o.step;
        let action = agents[seat].action(&o, &runtime)?;
        writeln!(output, "{}", action_json(&action).dump()).map_err(|e| e.to_string())?;
        output.flush().map_err(|e| e.to_string())?;
    }
    Ok(())
}
#[cfg(feature = "train")]
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
#[cfg(not(feature = "train"))]
fn main() {
    eprintln!("build with --features train");
    std::process::exit(1);
}

#[cfg(all(test, feature = "train"))]
mod tests {
    use super::load_deployment;
    use kagg_engine::json::Json;
    use route_rl_native::{
        learning::{
            policy::{Policy, Sample},
            tensor,
        },
        pipeline::{event_portfolio::Portfolio, plan_prototype::Config},
    };
    fn checkpoint(deployment: &Portfolio) -> Json {
        Json::Obj(vec![
            (
                "schema".into(),
                Json::Str("event-plan-improvement-v4".into()),
            ),
            (
                "policy_contract".into(),
                Json::Str(route_rl_native::pipeline::event_portfolio::CONTRACT.into()),
            ),
            ("config".into(), Config::default().json()),
            ("deployment".into(), deployment.json()),
            // Deliberately not a model: a deployment runner must not read this.
            (
                "model".into(),
                Json::Str("unaccepted learner is not deployable".into()),
            ),
        ])
    }
    fn row() -> Sample {
        let mut features = vec![vec![0.; 32]; 2];
        features[0][31] = 1.;
        features[1][31] = 1.;
        features[1][30] = 2.;
        Sample {
            context: vec![0.; 320],
            features,
            ..Default::default()
        }
    }
    #[test]
    fn deployment_loads_only_accepted_slot_models() {
        tensor::worker_threads();
        let empty = load_deployment(&checkpoint(&Portfolio::empty())).unwrap().1;
        assert_eq!(empty.select(Some(0), &row()).unwrap(), 0);
        let weights = Policy::plans(-1, 3, 0.0003)
            .unwrap()
            .weights_json()
            .unwrap();
        let portfolio = Portfolio::empty().propose(2, 5, weights).unwrap();
        let runtime = load_deployment(&checkpoint(&portfolio)).unwrap().1;
        assert_eq!(runtime.select(Some(2), &row()).unwrap(), 1);
        assert_eq!(runtime.select(Some(1), &row()).unwrap(), 0);
        assert_eq!(runtime.select(None, &row()).unwrap(), 0);
    }
    #[test]
    fn deployment_rejects_legacy_contract_and_missing_portfolio() {
        let mut legacy = checkpoint(&Portfolio::empty());
        legacy.set_path("schema", Json::Str("plan-comparison-v2".into()));
        assert!(load_deployment(&legacy).is_err());
        let mut wrong_contract = checkpoint(&Portfolio::empty());
        wrong_contract.set_path("policy_contract", Json::Str("plan-chain-320x32-v2".into()));
        assert!(load_deployment(&wrong_contract).is_err());
        let mut absent = checkpoint(&Portfolio::empty());
        absent.set_path("deployment", Json::Null);
        assert!(load_deployment(&absent).is_err());
    }
}

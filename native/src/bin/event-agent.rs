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
    use route_rl_native::pipeline::event_policy::{
        BATCH_CONTRACT, CONDITIONAL_CONTRACT, CONTRACT, EVIDENCE_SCHEMA, LEGACY_SCHEMA,
        MENU_BATCH_CONTRACT, MENU_CONTRACT, MENU_SCHEMA, NORMALIZED_SCHEMA, PREFIX_SCHEMA, SCHEMA,
    };
    use route_rl_native::pipeline::{
        event_portfolio::{Portfolio, Runtime},
        plan_prototype::Config,
    };
    if matches!(
        checkpoint.get("schema").str(),
        SCHEMA | MENU_SCHEMA | PREFIX_SCHEMA | NORMALIZED_SCHEMA | EVIDENCE_SCHEMA | LEGACY_SCHEMA
    ) {
        let contract = CONTRACT;
        if checkpoint.get("policy_contract").str() != contract
            || !matches!(
                checkpoint.get("deployment").get("contract").str(),
                CONTRACT
                    | BATCH_CONTRACT
                    | MENU_CONTRACT
                    | MENU_BATCH_CONTRACT
                    | CONDITIONAL_CONTRACT
            )
        {
            return Err("wrong shared event execution contract".into());
        }
        if !matches!(
            checkpoint.get("schema").str(),
            SCHEMA | MENU_SCHEMA | PREFIX_SCHEMA
        ) && checkpoint.get("deployment").get("contract").str() != CONTRACT
        {
            return Err("legacy wrapper cannot reinterpret batch responsibility".into());
        }
        if !matches!(checkpoint.get("schema").str(), SCHEMA | MENU_SCHEMA)
            && matches!(
                checkpoint.get("deployment").get("contract").str(),
                MENU_CONTRACT | MENU_BATCH_CONTRACT
            )
        {
            return Err("legacy wrapper cannot reinterpret menu policy".into());
        }
        if checkpoint.get("deployment").get("contract").str() == CONDITIONAL_CONTRACT
            && checkpoint.get("schema").str() != SCHEMA
        {
            return Err("conditional policy requires v10 wrapper".into());
        }
        let config = Config::parse(checkpoint.get("config"))?;
        let accepted =
            route_rl_native::pipeline::event_policy::Version::parse(checkpoint.get("deployment"))?;
        return Ok((config, accepted.runtime(-1)?));
    }
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
    fn stable_deployment_ignores_unaccepted_learner() {
        use route_rl_native::pipeline::event_policy::{Version, CONTRACT, SCHEMA};
        tensor::worker_threads();
        let p = Version::initial(Portfolio::empty(), vec![0, 1, 2, 3]).unwrap();
        let j = Json::Obj(vec![
            ("schema".into(), Json::Str(SCHEMA.into())),
            ("policy_contract".into(), Json::Str(CONTRACT.into())),
            ("config".into(), Config::default().json()),
            ("deployment".into(), p.json()),
            ("model".into(), Json::Str("invalid and unaccepted".into())),
        ]);
        assert_eq!(
            load_deployment(&j)
                .unwrap()
                .1
                .select(Some(0), &row())
                .unwrap(),
            0
        );
    }
    #[test]
    fn legacy_v5_v6_and_v7_wrappers_preserve_deployed_decisions() {
        use route_rl_native::pipeline::event_policy::{
            Version, CONTRACT, EVIDENCE_SCHEMA, LEGACY_SCHEMA, NORMALIZED_SCHEMA, SCHEMA,
        };
        tensor::worker_threads();
        let p = Version::initial(Portfolio::empty(), vec![0, 1, 2, 3])
            .unwrap()
            .propose(
                1,
                Policy::plans(-1, 7, 0.0003)
                    .unwrap()
                    .weights_json()
                    .unwrap(),
            )
            .unwrap();
        let mut j = Json::Obj(vec![
            ("schema".into(), Json::Str(SCHEMA.into())),
            ("policy_contract".into(), Json::Str(CONTRACT.into())),
            ("config".into(), Config::default().json()),
            ("deployment".into(), p.json()),
        ]);
        let current = load_deployment(&j)
            .unwrap()
            .1
            .select(Some(0), &row())
            .unwrap();
        j.set_path("schema", Json::Str(NORMALIZED_SCHEMA.into()));
        assert!(!load_deployment(&j).unwrap().1.batch_lifetime);
        assert_eq!(
            load_deployment(&j)
                .unwrap()
                .1
                .select(Some(0), &row())
                .unwrap(),
            current
        );
        j.set_path("schema", Json::Str(EVIDENCE_SCHEMA.into()));
        assert_eq!(
            load_deployment(&j)
                .unwrap()
                .1
                .select(Some(0), &row())
                .unwrap(),
            current
        );
        j.set_path("schema", Json::Str(LEGACY_SCHEMA.into()));
        assert_eq!(
            load_deployment(&j)
                .unwrap()
                .1
                .select(Some(0), &row())
                .unwrap(),
            current
        );
    }
    #[test]
    fn batch_scope_is_loaded_only_from_the_accepted_version() {
        use route_rl_native::pipeline::event_policy::{Version, CONTRACT, SCHEMA};
        tensor::worker_threads();
        let old = Version::initial(Portfolio::empty(), vec![0, 1, 2, 3]).unwrap();
        let mut batch = old
            .propose(
                1,
                Policy::event_plans(-1, 9, 0.0003)
                    .unwrap()
                    .weights_json()
                    .unwrap(),
            )
            .unwrap();
        batch.batch_lifetime = true;
        let mut j = Json::Obj(vec![
            ("schema".into(), Json::Str(SCHEMA.into())),
            ("policy_contract".into(), Json::Str(CONTRACT.into())),
            ("config".into(), Config::default().json()),
            ("target_batch_lifetime".into(), Json::Bool(true)),
            ("deployment".into(), old.json()),
            ("candidate".into(), batch.json()),
        ]);
        assert!(!load_deployment(&j).unwrap().1.batch_lifetime);
        j.set_path("deployment", batch.json());
        assert!(load_deployment(&j).unwrap().1.batch_lifetime);
        assert_eq!(Version::parse(&batch.json()).unwrap(), batch);
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

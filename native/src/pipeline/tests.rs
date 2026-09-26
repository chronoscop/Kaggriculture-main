use super::{encoding, executor::*, planner};
use kagg_engine::{
    engine,
    state::{AnimalTile, Cell, OMap, State},
};
fn crop(name: &str, day: i64, yield_units: i64) -> Cell {
    Cell::Plant {
        crop: name.into(),
        planted_day: day,
        watered_today: false,
        consecutive_unwatered: 0,
        yield_units,
        max_lifespan_step: 9999,
        fertilized_until_day: -1,
    }
}
fn mixed() -> State {
    let mut s = State::new(42);
    s.step = 22 * 24;
    s.farms[0].tiles[4][4] = Cell::Structure {
        kind: "PASTURE".into(),
        animal: Some(AnimalTile {
            animal: "COW".into(),
            placed_day: 10,
            yield_units: 2,
            consecutive_unfed: 0,
            fed_today: false,
            cared_today: false,
            fertilizer_available: true,
            pending_care_bonus: 0,
        }),
    };
    s.farms[0].tiles[4][3] = crop("STRAWBERRY", 10, 2);
    s.farms[0].tiles[3][3] = crop("WHEAT", 18, 5);
    s.private[0].shed.add("WHEAT", 3);
    s.private[0].seeds.add("WHEAT", 1);
    s
}
#[test]
fn mixed_route_transfers_fertilizer_replants_and_sells_actual_products() {
    let mut s = mixed();
    let initial_money = s.farms[0].money;
    let mut e = Executor::new();
    let o = Observation::from_state(&s, 0);
    e.observe(&o);
    let p = planner::route_problem(&o, &e, 0);
    let (index, route) = p
        .choices
        .iter()
        .enumerate()
        .find_map(|(i, c)| match c {
            planner::Choice::Route { route, .. }
                if route.crop_jobs > 0
                    && route.animal_jobs > 0
                    && route.reused_fertilizer > 0
                    && route.replants > 0
                    && route
                        .steps
                        .iter()
                        .any(|s| s.action.op == "FERTILIZE" && s.position == (3, 4)) =>
            {
                Some((i, route.clone()))
            }
            _ => None,
        })
        .expect("planner must discover an executable mixed material chain");
    let collected = route
        .steps
        .iter()
        .position(|s| s.action.op == "COLLECT_FERTILIZER")
        .unwrap();
    let fertilized = route
        .steps
        .iter()
        .position(|s| s.action.op == "FERTILIZE")
        .unwrap();
    assert!(collected < fertilized);
    let harvested = route
        .steps
        .iter()
        .position(|s| s.action.op == "HARVEST" && s.position == (3, 3))
        .unwrap();
    let planted = route
        .steps
        .iter()
        .position(|s| s.action.op == "PLANT")
        .unwrap();
    assert!(harvested < planted);
    p.select(index, &mut e, &o).unwrap();
    let finish = route.steps.back().unwrap().at;
    while s.step <= finish {
        let o = Observation::from_state(&s, 0);
        e.observe(&o);
        let a = e.action(&o, vec![]);
        engine::step(&mut s, &[a, Default::default()]);
    }
    e.observe(&Observation::from_state(&s, 0));
    assert_eq!(e.stats.receipt_failures, 0);
    assert_eq!(e.stats.invalidated, 0);
    assert!(s.farms[0].money > initial_money);
    assert!(
        matches!(s.farms[0].tiles[4][3],Cell::Plant{fertilized_until_day,..} if fertilized_until_day>=22)
    );
    assert!(matches!(
        s.farms[0].tiles[3][3],
        Cell::Plant {
            planted_day: 22,
            watered_today: true,
            ..
        }
    ));
}
#[test]
fn resource_and_plot_reservations_prevent_two_workers_claiming_same_job() {
    let mut s = mixed();
    s.farms[0].hands.push((4, 4));
    s.private[0].inventories.push(OMap::default());
    let o = Observation::from_state(&s, 0);
    let mut e = Executor::new();
    e.observe(&o);
    let p = planner::route_problem(&o, &e, 0);
    p.select(p.heuristic(), &mut e, &o).unwrap();
    let (reserved, _, _) = e.reserved(1);
    assert!(!reserved.is_empty());
    for c in planner::route_problem(&o, &e, 1).choices {
        if let planner::Choice::Route { route, .. } = c {
            assert!(route.sites.is_disjoint(&reserved));
        }
    }
}
#[test]
fn failed_purchase_does_not_invent_inventory_or_leave_permanent_reservation() {
    let mut s = State::new(1);
    let mut e = Executor::new();
    e.projects.insert(
        (3, 4),
        Project {
            production: Production::Crop("WHEAT".into()),
            requested: 0,
            confirmed: false,
            failures: 0,
        },
    );
    let o = Observation::from_state(&s, 0);
    e.observe(&o);
    assert!(planner::route_problem(&o,&e,0).choices.iter().all(|c|!matches!(c,planner::Choice::Route{route,..} if route.steps.iter().any(|s|s.action.op=="PLANT"))));
    s.step = 25;
    e.observe(&Observation::from_state(&s, 0));
    assert!(!e.projects.contains_key(&(3, 4)));
    assert_eq!(e.stats.expired_projects, 1);
}
#[test]
fn stale_route_replans_and_full_shed_does_not_destroy_cargo() {
    let mut s = State::new(1);
    s.private[0].shed.add("MILK", 100);
    s.private[0].inventories[0].add("WHEAT", 6);
    let o = Observation::from_state(&s, 0);
    let mut e = Executor::new();
    e.observe(&o);
    let mut r = Route::default();
    r.steps.push_back(Scheduled {
        at: 0,
        position: (4, 4),
        action: unit("DROP", "", 0),
    });
    e.assign(0, r);
    let a = e.action(&o, vec![]);
    assert_ne!(a.farmer.op, "DROP");
    engine::step(&mut s, &[a, Default::default()]);
    assert_eq!(s.private[0].inventories[0].get("WHEAT"), 6);
    let mut r = Route::default();
    r.steps.push_back(Scheduled {
        at: 2,
        position: (4, 4),
        action: unit("NORTH", "", 0),
    });
    e.assign(0, r);
    let o = Observation::from_state(&s, 0);
    e.observe(&o);
    let a = e.action(&o, vec![]);
    assert_eq!(a.farmer.op, "PASS");
    assert_eq!(e.stats.invalidated, 1);
}
#[test]
fn candidates_ignore_seed_and_opponent_private_state() {
    let s = mixed();
    let mut altered = s.clone();
    altered.seed = 999;
    altered.private[1].shed.add("MILK", 1000);
    altered.private[1].seeds.add("MELON", 99);
    let mut outputs = Vec::new();
    for state in [s, altered] {
        let o = Observation::from_state(&state, 0);
        let mut e = Executor::new();
        e.observe(&o);
        let p = planner::route_problem(&o, &e, 0);
        outputs.push(encoding::encode(&o, &e, &p));
    }
    assert_eq!(outputs[0], outputs[1]);
    assert_eq!(outputs[0].0.len(), 96);
    assert!(outputs[0].1.iter().all(|v| v.len() == 32));
}
#[test]
fn new_projects_are_selectable_without_baseline_or_free_materials() {
    let s = State::new(42);
    let o = Observation::from_state(&s, 0);
    let mut e = Executor::new();
    e.observe(&o);
    let p = planner::investment_problem(&o, &e);
    let i=p.choices.iter().position(|c|matches!(c,planner::Choice::Invest{production:Some(Production::Animal(a)),..} if a=="COW")).unwrap();
    let orders = p.select(i, &mut e, &o).unwrap();
    assert!(orders.iter().any(|o| o[0] == "BUY_ANIMAL"));
    assert!(orders.iter().any(|o| o[0] == "BUY_PRODUCT"));
    assert!(planner::route_problem(&o,&e,0).choices.iter().all(|c|!matches!(c,planner::Choice::Route{route,..} if route.steps.iter().any(|s|s.action.op=="PLACE"&&s.action.item=="COW"))));
    assert!(p.select(usize::MAX, &mut e, &o).is_err());
}
#[cfg(feature = "train")]
#[test]
fn policy_updates_and_resumes_without_old_gate_or_baseline_checkpoint() {
    use crate::learning::{
        policy::{Batch, Policy, Rng, Sample},
        tensor,
    };
    tensor::threads(1);
    let devices = if std::env::var_os("ROUTE_RL_TEST_CUDA").is_some() {
        vec![-1, 0]
    } else {
        vec![-1]
    };
    for device in devices {
        let mut model = Policy::mixed_routes(device, 123, 1e-4).unwrap();
        let mut rows = Vec::new();
        for i in 0..8 {
            let mut features = vec![vec![0.; 32]; 1 + i % 4];
            for (j, f) in features.iter_mut().enumerate() {
                f[j] = 1.;
            }
            rows.push(Sample {
                context: vec![i as f32 / 8.; 96],
                features,
                action: 0,
                logp: 0.,
                value: 0.,
                reward: if i % 2 == 0 { 1. } else { -1. },
                ..Sample::default()
            });
        }
        let mut rng = Rng(99);
        let decisions = model.infer(&rows, false, &mut rng).unwrap();
        for (r, d) in rows.iter_mut().zip(decisions) {
            r.action = d.action;
            r.logp = d.logp;
            r.value = d.value;
            r.advantage = r.reward - r.value;
        }
        let (lp, _) = model.forward(&Batch::new(&rows, device).unwrap()).unwrap();
        let lp = lp.data().unwrap();
        for i in 0..8 {
            let sum: f32 = lp[i * 4..i * 4 + rows[i].features.len()]
                .iter()
                .map(|v| v.exp())
                .sum();
            assert!((sum - 1.).abs() < 1e-5);
        }
        let before = model.weights_json().unwrap().dump();
        let update = model.update(&rows, 1, 8, &mut rng).unwrap();
        assert!(update.loss.is_finite());
        assert_eq!(update.updates, 1);
        assert_ne!(before, model.weights_json().unwrap().dump());
        let ck = model.checkpoint(1, &rng).unwrap();
        let mut resumed = Policy::mixed_routes(device, 0, 1e-4).unwrap();
        let (i, mut saved_rng) = resumed.restore(&ck).unwrap();
        assert_eq!(i, 1);
        model.update(&rows, 1, 8, &mut rng).unwrap();
        resumed.update(&rows, 1, 8, &mut saved_rng).unwrap();
        assert_eq!(
            model.weights_json().unwrap().dump(),
            resumed.weights_json().unwrap().dump()
        );
        let mut old = ck.clone();
        if let kagg_engine::json::Json::Obj(fields) = &mut old {
            fields.iter_mut().find(|(k, _)| k == "schema").unwrap().1 =
                kagg_engine::json::Json::Str("mixed-production-v6-ppo-v1".into());
        }
        assert!(resumed.restore(&old).is_err());
    }
}

#[test]
fn funded_project_dispatches_but_missing_stock_allows_waiting() {
    let mut s = State::new(42);
    let mut e = Executor::new();
    let o = Observation::from_state(&s, 0);
    e.observe(&o);
    let p = planner::investment_problem(&o, &e);
    let index = p
        .choices
        .iter()
        .position(|c| {
            matches!(c,
        planner::Choice::Invest { production: Some(Production::Crop(c)), .. } if c == "WHEAT")
        })
        .unwrap();
    let orders = p.select(index, &mut e, &o).unwrap();
    let unfunded = planner::route_problem(&o, &e, 0);
    assert!(unfunded
        .choices
        .iter()
        .any(|c| matches!(c, planner::Choice::Continue)));
    let action = e.action(&o, orders);
    engine::step(&mut s, &[action, Default::default()]);
    let o = Observation::from_state(&s, 0);
    e.observe(&o);
    let funded = planner::route_problem(&o, &e, 0);
    assert!(!funded.choices.is_empty());
    assert!(funded
        .choices
        .iter()
        .all(|c| matches!(c, planner::Choice::Route { .. })));
    funded.select(0, &mut e, &o).unwrap();
    let finish = e.routes[0].as_ref().unwrap().steps.back().unwrap().at;
    while s.step <= finish {
        let o = Observation::from_state(&s, 0);
        e.observe(&o);
        let action = e.action(&o, vec![]);
        engine::step(&mut s, &[action, Default::default()]);
    }
    assert!(s.farms[0]
        .tiles
        .iter()
        .flatten()
        .any(|t| matches!(t, Cell::Plant { crop, .. } if crop == "WHEAT")));
    assert!(e.stats.work > 0);
}

#[cfg(feature = "train")]
#[test]
fn singleton_return_route_is_executed_without_a_policy_request() {
    use super::rollout::{Game, Opponent};
    let mut game = Game::new(42, 0, Opponent::Heuristic, false);
    game.state.step = 718;
    game.state.private[0].inventories[0].add("WHEAT", 2);
    for seat in 0..2 {
        let o = Observation::from_state(&game.state, seat);
        game.agents[seat].observe(&o);
        game.agents[seat].last_market = 718;
        game.agents[seat].last_cash = o.farm.money;
    }
    let o = Observation::from_state(&game.state, 0);
    assert_eq!(
        planner::route_problem(&o, &game.agents[0], 0).choices.len(),
        1
    );
    assert!(game.prepare().is_none());
    assert_eq!(game.state.step, 719);
    assert_eq!(game.agents[0].stats.routes, 1);
    assert!(game.state.farms[0].money > 3000.);
}

#[cfg(feature = "train")]
#[test]
fn league_mixes_opponents_excludes_historical_rows_and_preserves_frozen_weights() {
    use super::{
        league::League,
        rollout::{self, Opponent},
    };
    use crate::learning::{
        policy::{Policy, Rng},
        tensor,
    };
    tensor::threads(1);
    let device = if std::env::var_os("ROUTE_RL_TEST_CUDA").is_some() {
        0
    } else {
        -1
    };
    let model = Policy::mixed_routes(device, 2026, 1e-4).unwrap();
    let league = League::new(&model).unwrap();
    let restored = League::restore(&league.json()).unwrap();
    let pool = restored.policies(device).unwrap();
    let before = pool[0].weights_json().unwrap().dump();
    let c = rollout::collect_exploring(
        &model,
        &[7301, 7302, 7303, 7304],
        2,
        Opponent::League,
        &pool,
        false,
        &mut Rng(123),
        true,
        0.2,
    )
    .unwrap();
    assert_eq!(c.games.len(), 8);
    assert_eq!(c.games.iter().filter(|g| g.greedy_probe).count(), 2);
    assert_eq!(
        c.games
            .iter()
            .filter(|g| g.opponent == Opponent::Heuristic)
            .count(),
        2
    );
    assert_eq!(
        c.games
            .iter()
            .filter(|g| g.opponent == Opponent::SelfPlay)
            .count(),
        2
    );
    assert_eq!(
        c.games
            .iter()
            .filter(|g| matches!(g.opponent, Opponent::Frozen(_)))
            .count(),
        4
    );
    let expected: usize = c
        .games
        .iter()
        .map(|g| {
            g.trace
                .iter()
                .filter(|t| {
                    t.get("selected").is_str()
                        && !g.greedy_probe
                        && (g.opponent == Opponent::SelfPlay
                            || t.get("seat").i64() as usize == g.learner)
                })
                .count()
        })
        .sum();
    assert_eq!(c.samples.len(), expected);
    assert!(expected > 0);
    assert!(c.samples.iter().all(|r| r.exploration == 0.2));
    assert!(c.games.iter().all(|g| g.state.step == 719));
    assert_eq!(before, pool[0].weights_json().unwrap().dump());
    assert!(rollout::collect_with_pool(
        &model,
        &[1],
        1,
        Opponent::Frozen(1),
        &pool,
        true,
        &mut Rng(0),
        false
    )
    .is_err());
}

#[cfg(feature = "train")]
#[test]
fn league_gate_rejects_idle_regression_and_losing_to_champion() {
    use super::league::{qualifies, Score};
    let champion = Score {
        games: 8,
        cash: 3200.,
        margin: -200.,
        win_rate: 0.25,
        work: 100.,
        harvest: 20.,
        idle_fraction: 0.5,
        inactive_games: 0,
    };
    let mut candidate = champion.clone();
    candidate.cash = 3500.;
    candidate.margin = 100.;
    let mut duel = candidate.clone();
    duel.win_rate = 0.75;
    assert!(qualifies(&candidate, &champion, &duel));
    candidate.inactive_games = 1;
    assert!(!qualifies(&candidate, &champion, &duel));
    candidate.inactive_games = 0;
    candidate.margin = -201.;
    assert!(!qualifies(&candidate, &champion, &duel));
    candidate.margin = 100.;
    duel.margin = -1.;
    assert!(!qualifies(&candidate, &champion, &duel));
}

#[cfg(feature = "train")]
#[test]
fn validation_cannot_promote_an_unchanged_policy() {
    use super::league::League;
    use crate::learning::{policy::Policy, tensor};
    tensor::threads(1);
    let policy = Policy::mixed_routes(-1, 1200, 1e-4).unwrap();
    let mut league = League::new(&policy).unwrap();
    let before = league.json().dump();
    let (promoted, report) = league
        .evaluate_and_promote(&policy, 5, &[1000000000], 2, 77)
        .unwrap();
    assert!(!promoted);
    assert_eq!(before, league.json().dump());
    assert_eq!(
        report.get("greedy_vs_heuristic").dump(),
        report.get("champion_vs_heuristic").dump()
    );
    assert_eq!(report.get("sampled_vs_heuristic").get("games").i64(), 2);
}

#[test]
fn expansion_budget_and_pending_projects_bound_exploration_without_forcing_investment() {
    let s = State::new(42);
    let o = Observation::from_state(&s, 0);
    let mut e = Executor::new();
    e.observe(&o);
    let first = planner::investment_problem(&o, &e);
    assert!(matches!(first.choices[0], planner::Choice::Continue));
    for _ in 0..2 {
        let p = planner::investment_problem(&o, &e);
        let i = p
            .choices
            .iter()
            .position(|c| {
                matches!(c, planner::Choice::Invest {
            production: Some(Production::Crop(c)), .. } if c == "WHEAT")
            })
            .unwrap();
        p.select(i, &mut e, &o).unwrap();
    }
    let p = planner::investment_problem(&o, &e);
    assert!(p.choices.iter().all(|c| !matches!(
        c,
        planner::Choice::Invest {
            production: Some(_),
            ..
        }
    )));
    // Even a direct re-query cannot cancel a project before its startup receipt.
    assert!(matches!(p.choices[0], planner::Choice::Continue));
    let mut e = Executor::new();
    e.observe(&o);
    e.expansion_spent = e.expansion_budget;
    assert!(planner::investment_problem(&o, &e)
        .choices
        .iter()
        .all(|c| !matches!(c,
        planner::Choice::Invest { production: Some(_), cost, .. } if *cost > 0.)));
    let mut s = s;
    s.step = 25;
    e.observe(&Observation::from_state(&s, 0));
    assert_eq!(e.new_projects_today, 0);
    assert_eq!(e.expansion_spent, 0.);
}

#[cfg(feature = "train")]
#[test]
fn hierarchical_probabilities_exploration_and_greedy_share_one_contract() {
    use crate::learning::{
        policy::{exploration_proposal, hierarchical_argmax, Batch, Policy, Rng, Sample},
        tensor,
    };
    tensor::threads(1);
    let devices = if std::env::var_os("ROUTE_RL_TEST_CUDA").is_some() {
        vec![-1, 0]
    } else {
        vec![-1]
    };
    for device in devices {
        let mut model = Policy::mixed_routes(device, 81, 1e-4).unwrap();
        // All scores zero: two legal categories have equal mass regardless of candidate count.
        for p in &mut model.parameters {
            let zero = p.value.unary(26).unwrap();
            p.value.copy_from(&zero).unwrap();
        }
        let mut features = vec![vec![0.; 32]; 21];
        for f in &mut features[1..] {
            f[31] = 2.;
        }
        let row = Sample {
            context: vec![0.; 96],
            features: features.clone(),
            exploration: 0.2,
            ..Sample::default()
        };
        let b = Batch::new(&[row.clone()], device).unwrap();
        let (lp, _) = model.forward(&b).unwrap();
        let base = lp.data().unwrap();
        assert!((base[0].exp() - 0.5).abs() < 1e-5);
        assert!((base[1..].iter().map(|x| x.exp()).sum::<f32>() - 0.5).abs() < 1e-5);
        let mixed = model.behavior(&lp, &b).unwrap().data().unwrap();
        assert!((mixed[0].exp() - 0.4).abs() < 1e-5);
        assert!((mixed[1..].iter().map(|x| x.exp()).sum::<f32>() - 0.6).abs() < 1e-5);
        let q = exploration_proposal(&features);
        assert_eq!(q[0], 0.);
        let custom: Vec<_> = (0..21)
            .map(|i| {
                if i == 0 {
                    0.15f32.ln()
                } else {
                    (0.85f32 / 20.).ln()
                }
            })
            .collect();
        assert_ne!(hierarchical_argmax(&custom, &features), 0);
        let d = model
            .infer(&[row.clone()], false, &mut Rng(5))
            .unwrap()
            .remove(0);
        assert!((d.logp - mixed[d.action]).abs() < 1e-6);
        let mut sample = row;
        sample.action = d.action;
        sample.logp = d.logp;
        sample.value = d.value;
        sample.advantage = 1.;
        sample.reward = 1.;
        let u = model.update(&[sample], 1, 1, &mut Rng(9)).unwrap();
        assert_eq!(u.updates, 1);
        assert!(u.mean_kl.abs() < 1e-5);
    }
}

#[cfg(feature = "train")]
#[test]
fn cash_returns_include_purchase_cost_future_receipts_and_real_elapsed_time() {
    use crate::learning::policy::{cash_returns, Sample};
    let mut rows = vec![
        Sample {
            step: 0,
            cash: 0.3,
            value: 0.1,
            ..Sample::default()
        },
        Sample {
            step: 0,
            cash: 0.3,
            value: 0.2,
            ..Sample::default()
        },
        Sample {
            step: 10,
            cash: 0.2,
            value: 0.15,
            ..Sample::default()
        },
    ];
    cash_returns(&mut rows, 30, 0.5, 1.);
    assert_eq!(
        rows.iter().map(|r| r.elapsed).collect::<Vec<_>>(),
        vec![0, 10, 20]
    );
    assert!((rows[1].cash_delta + 0.1).abs() < 1e-6);
    assert!((rows[2].cash_delta - 0.3).abs() < 1e-6);
    for r in &rows {
        assert!((r.reward - (0.5 - r.cash)).abs() < 1e-6);
    }
    cash_returns(&mut rows, 30, 0.5, 0.9);
    let expected_last = 0.3 - 0.15;
    let expected_middle = -0.1 + 0.15 - 0.2 + 0.9f32.powi(10) * expected_last;
    assert!((rows[1].advantage - expected_middle).abs() < 1e-6);
    assert!((rows[0].advantage - (0.2 - 0.1 + expected_middle)).abs() < 1e-6);
}

#[cfg(feature = "train")]
#[test]
fn profitable_experience_is_bounded_resumable_and_learned_separately() {
    use crate::learning::{
        experience::{Experience, ExperienceBank, CAPACITY},
        policy::{Batch, Policy, Rng, Sample},
        tensor,
    };
    tensor::threads(1);
    let mut features = vec![vec![0.; 32]; 2];
    features[1][31] = 2.;
    features[1][2] = 1.;
    features[1][9] = 1.;
    let row = Sample {
        context: vec![0.; 96],
        features,
        action: 1,
        mc_return: 2.,
        reward: 2.,
        ..Sample::default()
    };
    assert!(Experience::from_episode(1, 0, -1., 3, &[row.clone()]).is_none());
    assert!(Experience::from_episode(1, 0, 100., 0, &[row.clone()]).is_none());
    let mut bank = ExperienceBank::default();
    for seed in 0..20 {
        bank.insert(
            Experience::from_episode(seed, 0, 100. + seed as f64, 3, &[row.clone()]).unwrap(),
        );
    }
    assert_eq!(bank.episodes.len(), 2);
    assert!(bank.episodes.len() <= CAPACITY);
    let j = bank.json();
    let restored = ExperienceBank::restore(&j).unwrap();
    assert_eq!(restored.json().dump(), j.dump());
    assert_eq!(
        bank.sample(4, &mut Rng(12))[0].json().dump(),
        restored.sample(4, &mut Rng(12))[0].json().dump()
    );
    let mut model = Policy::mixed_routes(-1, 77, 1e-3).unwrap();
    let b = Batch::new(&[row.clone()], -1).unwrap();
    let before = model.forward(&b).unwrap().0.data().unwrap()[1];
    let (n, loss) = model.imitate(&[row], 0.05).unwrap();
    assert_eq!(n, 1);
    assert!(loss > 0.);
    let after = model.forward(&b).unwrap().0.data().unwrap()[1];
    assert!(
        after > before,
        "successful action probability must increase"
    );
}

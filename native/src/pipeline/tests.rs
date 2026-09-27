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
    use super::league::{anchor_guard, qualifies_for_pool, Score};
    let champion = Score {
        games: 8,
        cash: 3200.,
        margin: -200.,
        win_rate: 0.25,
        draw_rate: 0.,
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
    assert!(anchor_guard(&candidate, &champion) && qualifies_for_pool(&candidate, &duel));
    candidate.inactive_games = 1;
    assert!(!(anchor_guard(&candidate, &champion) && qualifies_for_pool(&candidate, &duel)));
    candidate.inactive_games = 0;
    candidate.cash = 2000.;
    assert!(!anchor_guard(&candidate, &champion));
    candidate.cash = 3500.;
    duel.margin = -1.;
    assert!(!(anchor_guard(&candidate, &champion) && qualifies_for_pool(&candidate, &duel)));
}

#[cfg(feature = "train")]
#[test]
fn validation_cannot_promote_an_unchanged_policy() {
    use super::league::League;
    use crate::learning::{policy::Policy, tensor};
    tensor::threads(1);
    let policy = Policy::mixed_routes(-1, 1200, 1e-4).unwrap();
    let mut league = League::new(&policy).unwrap();
    let weights = league.snapshots[0].weights.clone();
    let (promoted, report) = league
        .evaluate_and_promote(&policy, 5, &[1000000000], 2, 77)
        .unwrap();
    assert!(!promoted);
    assert_eq!(league.snapshots.len(), 1);
    assert_eq!(weights, league.snapshots[0].weights);
    assert!(!report.get("pool_admitted").bool());
    assert!(report.get("confirmation").is_null());
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
    assert_eq!(bank.episodes.len(), 20);
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

#[cfg(feature = "train")]
#[test]
fn probe_assignment_covers_all_opponent_types_in_each_32_game_batch() {
    use super::rollout::probe_schedule;
    use crate::learning::policy::Rng;
    for seed in 0..64 {
        for offset in 0..4 {
            let mask = probe_schedule(16, &mut Rng(seed));
            let mut counts = [[0; 2]; 3];
            for (i, probe) in mask.iter().enumerate() {
                let kind = match (i + offset) % 4 {
                    0 => 0,
                    1 => 1,
                    _ => 2,
                };
                counts[kind][usize::from(*probe)] += 2;
            }
            assert_eq!(counts, [[6, 2], [6, 2], [12, 4]]);
        }
        for size in 0..33 {
            assert_eq!(
                probe_schedule(size, &mut Rng(seed))
                    .iter()
                    .filter(|v| **v)
                    .count(),
                size / 4
            );
        }
    }
}

#[cfg(feature = "train")]
#[test]
fn competitive_pool_entry_does_not_replace_or_evict_champion() {
    use super::league::{anchor_guard, qualifies_for_pool, League, Score, Snapshot, CAPACITY};
    use kagg_engine::json::Json;
    let champion = Score {
        games: 8,
        cash: 60000.,
        margin: 54000.,
        win_rate: 1.,
        draw_rate: 0.,
        work: 800.,
        harvest: 200.,
        idle_fraction: 0.1,
        inactive_games: 0,
    };
    let mut candidate = champion.clone();
    candidate.margin -= 326.;
    let mut duel = champion.clone();
    duel.margin = 5026.;
    duel.win_rate = 0.75;
    assert!(anchor_guard(&candidate, &champion));
    assert!(qualifies_for_pool(&candidate, &duel));
    duel.inactive_games = 1;
    assert!(!qualifies_for_pool(&candidate, &duel));
    let legacy = Json::Arr(vec![Json::Obj(vec![
        ("iteration".into(), Json::Str("1025".into())),
        ("weights".into(), Json::Obj(vec![])),
    ])]);
    let mut league = League::restore(&legacy).unwrap();
    for iteration in 1026..1040 {
        league.admit(
            Snapshot {
                iteration,
                weights: Json::Obj(vec![]),
                profile: None,
            },
            false,
        );
    }
    assert_eq!(league.snapshots.len(), CAPACITY);
    assert_eq!(league.snapshots[league.champion_index()].iteration, 1025);
    let restored = League::restore(&league.json()).unwrap();
    assert_eq!(restored.champion_iteration, 1025);
    assert_eq!(restored.json(), league.json());
    league.admit(
        Snapshot {
            iteration: 1040,
            weights: Json::Obj(vec![]),
            profile: None,
        },
        true,
    );
    assert_eq!(league.snapshots[league.champion_index()].iteration, 1040);
}

#[cfg(feature = "train")]
#[test]
fn experience_retains_recent_routes_and_balances_opponent_sources() {
    use crate::learning::{
        experience::{Experience, ExperienceBank, PER_SOURCE},
        policy::{Rng, Sample},
    };
    let mut row = Sample {
        context: vec![0.; 96],
        features: vec![vec![0.; 32]],
        mc_return: 2.,
        ..Sample::default()
    };
    row.features[0][31] = 2.;
    let make = |seed: i64, source: &str, style: u32, profit: f64| {
        let mut e = Experience::from_episode(seed, 0, profit, 2, &[row.clone()]).unwrap();
        e.opponent = source.into();
        e.style = style;
        e.collected_iteration = seed as u64;
        e.learner_seat = Some(0);
        e.rows[0].context[0] = match source {
            "heuristic" => 0.,
            "historical" => 1.,
            _ => 2.,
        };
        e
    };
    let mut bank = ExperienceBank::default();
    for seed in 1..20 {
        bank.insert(make(seed, "heuristic", 1, 10000. - seed as f64));
    }
    assert!(bank.episodes.iter().any(|e| e.seed == 1));
    assert!(bank.episodes.iter().any(|e| e.seed == 19));
    assert!(bank.episodes.iter().any(|e| e.seed == 18));
    for style in 2..50 {
        bank.insert(make(100 + style as i64, "heuristic", style, 100000.));
    }
    for seed in 200..208 {
        bank.insert(make(seed, "historical", seed as u32, 100.));
    }
    for seed in 300..308 {
        bank.insert(make(seed, "current", seed as u32, 100.));
    }
    assert!(
        bank.episodes
            .iter()
            .filter(|e| e.opponent == "heuristic")
            .count()
            <= PER_SOURCE
    );
    assert_eq!(
        bank.episodes
            .iter()
            .filter(|e| e.opponent == "historical")
            .count(),
        8
    );
    let mut counts = [0; 3];
    for r in bank.sample(6000, &mut Rng(10)) {
        counts[r.context[0] as usize] += 1;
    }
    assert!(
        counts.iter().all(|n| (1700..2300).contains(n)),
        "{counts:?}"
    );
    let mut a = make(999, "historical", 999, 100.);
    bank.insert(a.clone());
    a.learner_seat = Some(1);
    bank.insert(a);
    assert_eq!(bank.episodes.iter().filter(|e| e.seed == 999).count(), 2);
    let restored = ExperienceBank::restore(&bank.json()).unwrap();
    assert_eq!(bank.json(), restored.json());
    assert_eq!(
        bank.sample(10, &mut Rng(5))
            .iter()
            .map(Sample::json)
            .collect::<Vec<_>>(),
        restored
            .sample(10, &mut Rng(5))
            .iter()
            .map(Sample::json)
            .collect::<Vec<_>>()
    );
    let mut legacy = make(0, "heuristic", 1, 100.).json();
    if let kagg_engine::json::Json::Obj(ref mut fields) = legacy {
        fields.retain(|(k, _)| {
            ![
                "opponent",
                "learner_seat",
                "collected_iteration",
                "retention_bucket",
            ]
            .contains(&k.as_str())
        });
    }
    let old = ExperienceBank::restore(&kagg_engine::json::Json::Arr(vec![legacy])).unwrap();
    assert_eq!(old.episodes[0].opponent, "legacy");
    assert_eq!(old.episodes[0].learner_seat, None);
    bank.insert(old.episodes[0].clone());
    assert!(!bank.episodes.iter().any(|e| e.opponent == "legacy"));
}

#[cfg(feature = "train")]
#[test]
fn protected_elites_survive_hundreds_of_new_styles_and_resume() {
    use crate::learning::{
        experience::{Experience, ExperienceBank, PER_BUCKET, PER_SOURCE},
        policy::{Rng, Sample},
    };
    let mut row = Sample {
        context: vec![0.; 96],
        features: vec![vec![0.; 32]],
        mc_return: 1.,
        ..Default::default()
    };
    row.features[0][31] = 2.;
    let make = |seed: i64, profit: f64| {
        let mut e = Experience::from_episode(seed, 0, profit, 1, &[row.clone()]).unwrap();
        e.opponent = "historical".into();
        e.style = seed as u32;
        e.collected_iteration = seed as u64;
        e.learner_seat = Some(0);
        e.rows[0].context[0] = seed as f32;
        e
    };
    let mut bank = ExperienceBank::default();
    for i in 1..=16 {
        bank.insert(make(i, 10000. + i as f64));
    }
    for i in 17..600 {
        bank.insert(make(i, 100.));
    }
    assert_eq!(bank.episodes.len(), PER_SOURCE);
    for name in ["elite", "recent", "diverse"] {
        assert_eq!(
            bank.episodes
                .iter()
                .filter(|e| e.retention_bucket == name)
                .count(),
            PER_BUCKET
        );
    }
    assert_eq!(
        bank.episodes
            .iter()
            .filter(|e| e.retention_bucket == "elite")
            .map(|e| e.seed)
            .collect::<std::collections::BTreeSet<_>>(),
        (1..=16).collect()
    );
    assert!(bank
        .episodes
        .iter()
        .filter(|e| e.retention_bucket == "recent")
        .all(|e| e.seed >= 584));
    let sampled = bank.sample(6000, &mut Rng(8));
    let elite = sampled.iter().filter(|r| r.context[0] <= 16.).count();
    assert!((1700..2300).contains(&elite));
    let mut restored = ExperienceBank::restore(&bank.json()).unwrap();
    assert_eq!(bank.json(), restored.json());
    for i in 600..900 {
        let e = make(i, 110.);
        bank.insert(e.clone());
        restored.insert(e);
    }
    assert_eq!(bank.json(), restored.json());
    assert_eq!(
        bank.episodes
            .iter()
            .filter(|e| e.retention_bucket == "elite" && e.seed <= 16)
            .count(),
        16
    );
    assert_eq!(
        bank.episodes
            .iter()
            .map(|e| (e.seed, e.seat, e.learner_seat))
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        bank.episodes.len()
    );
}

#[cfg(feature = "train")]
#[test]
fn training_roster_covers_history_and_preserves_seat_pair_probes() {
    use super::{league::Roster, rollout::Opponent};
    use crate::learning::policy::Rng;
    let roster = Roster {
        champion: 2,
        recent: vec![4, 3],
        historical: vec![0, 1, 2],
        probabilities: vec![0.4, 0.3, 0.3],
        phase: 10,
    };
    let schedule = roster.schedule(16, &mut Rng(9));
    let count = |role: &str| schedule.iter().filter(|x| x.1 == role).count() * 2;
    assert_eq!(
        [
            count("recent"),
            count("pfsp"),
            count("coverage"),
            count("current"),
            count("heuristic")
        ],
        [12, 12, 4, 2, 2]
    );
    assert!(schedule.iter().filter(|x| x.1 == "recent").all(|x| [
        Opponent::Frozen(3),
        Opponent::Frozen(4)
    ]
    .contains(&x.0)));
    assert_eq!(schedule.iter().filter(|x| x.2).count(), 4);
    for n in 1..33 {
        assert_eq!(
            roster
                .schedule(n, &mut Rng(9))
                .iter()
                .filter(|x| x.2)
                .count(),
            n / 4
        );
    }
    let mut a = Rng(80);
    let mut b = Rng(80);
    assert_eq!(roster.schedule(32, &mut a), roster.schedule(32, &mut b));
}

#[cfg(feature = "train")]
#[test]
fn actual_market_profile_accounts_for_partial_sales_and_preserves_state() {
    use super::behavior::{observe_market, Profile, TradeStats};
    use kagg_engine::engine::PlayerAction;
    let mut state = State::new(42);
    state.private[0].shed.add("MILK", 3);
    state.private[1].shed.add("MILK", 2);
    let a = |n: &str| PlayerAction {
        market: vec![vec!["SELL".into(), "MILK".into(), n.into()]],
        ..Default::default()
    };
    let actions = [a("99"), a("2")];
    let mut stats: [TradeStats; 2] = Default::default();
    observe_market(&state, &actions, &mut stats);
    assert_eq!(state.private[0].shed.get("MILK"), 3);
    assert_eq!(state.farms[0].money, 3000.);
    let mut actual = state.clone();
    engine::step(&mut actual, &actions);
    let milk = kagg_engine::state::PRODUCTS
        .iter()
        .position(|x| *x == "MILK")
        .unwrap();
    assert_eq!(stats[0].units[milk], 3);
    assert_eq!(stats[1].units[milk], 2);
    for p in 0..2 {
        assert_eq!(
            stats[p].revenue[milk],
            actual.farms[p].money - state.farms[p].money
        );
    }
    let p = Profile {
        values: vec![0.; 11],
    };
    assert_eq!(p.distance(&p), 0.);
    assert_eq!(Profile::parse(&p.json()).unwrap().unwrap().values, p.values);
}

#[cfg(feature = "train")]
#[test]
fn hard_history_and_recent_versions_survive_without_behavior_novelty() {
    use super::league::{League, Snapshot};
    use kagg_engine::json::Json;
    let mut league = League::restore(&Json::Arr(vec![Json::Obj(vec![
        ("iteration".into(), Json::Str("0".into())),
        ("weights".into(), Json::Obj(vec![])),
    ])]))
    .unwrap();
    for i in 1..30 {
        league.admit(
            Snapshot {
                iteration: i,
                weights: Json::Obj(vec![]),
                profile: None,
            },
            false,
        );
        if i == 2 {
            for _ in 0..32 {
                league
                    .outcomes
                    .entry(2)
                    .or_default()
                    .greedy
                    .observe(i, 40000., 60000.);
            }
        }
    }
    for i in [0, 1, 2, 28, 29] {
        assert!(league.snapshots.iter().any(|s| s.iteration == i));
    }
    let roster = league.roster(30);
    assert_eq!(
        roster
            .recent
            .iter()
            .map(|&i| league.snapshots[i].iteration)
            .collect::<Vec<_>>(),
        vec![29, 28]
    );
    let hard = roster
        .historical
        .iter()
        .position(|&i| league.snapshots[i].iteration == 2)
        .unwrap();
    assert!(roster.probabilities[hard] > 1. / roster.historical.len() as f64);
    assert_eq!(
        League::restore(&league.json()).unwrap().json(),
        league.json()
    );
}

#[cfg(feature = "train")]
#[test]
fn actual_market_profile_matches_drop_sale_buy_and_order_limit() {
    use super::behavior::{observe_market, TradeStats};
    use kagg_engine::{
        engine::PlayerAction,
        state::{MAX_MARKET_ORDERS, PRODUCTS},
    };
    let mut s = State::new(99);
    s.step = 30 * 2;
    s.private[0].inventories[0].add("MILK", 3);
    let mut a = PlayerAction {
        farmer: unit("DROP", "", 0),
        market: vec![
            vec!["SELL".into(), "MILK".into(), "9".into()],
            vec!["BUY_PRODUCT".into(), "WHEAT".into(), "3".into()],
        ],
        ..Default::default()
    };
    for _ in 2..MAX_MARKET_ORDERS + 2 {
        a.market
            .push(vec!["BUY_SEED".into(), "WHEAT".into(), "1".into()]);
    }
    let actions = [a, PlayerAction::default()];
    let mut stats: [TradeStats; 2] = Default::default();
    observe_market(&s, &actions, &mut stats);
    let before = s.farms[0].money;
    engine::step(&mut s, &actions);
    assert_eq!(
        stats[0].units[PRODUCTS.iter().position(|x| *x == "MILK").unwrap()],
        3
    );
    assert_eq!(
        before + stats[0].revenue.iter().sum::<f64>() - stats[0].spending,
        s.farms[0].money
    );
    assert_eq!(stats[0].early_spending, stats[0].spending);
}

#[cfg(feature = "train")]
#[test]
fn training_outcomes_use_version_seat_mode_and_completed_games() {
    use super::{
        league::{League, Snapshot},
        rollout::{Collection, Game, Opponent},
    };
    use kagg_engine::json::Json;
    let mut league = League::restore(&Json::Arr(vec![Json::Obj(vec![
        ("iteration".into(), Json::Str("100".into())),
        ("weights".into(), Json::Obj(vec![])),
    ])]))
    .unwrap();
    let make = |seat, probe, step| {
        let mut g = Game::new(42, seat, Opponent::Frozen(0), true);
        g.greedy_probe = probe;
        g.state.step = step;
        g.state.farms[0].money = 60000.;
        g.state.farms[1].money = 40000.;
        g
    };
    let c = Collection {
        games: vec![
            make(0, false, 719),
            make(1, false, 719),
            make(1, true, 719),
            make(0, false, 100),
        ],
        samples: vec![],
        experiences: vec![],
        seconds: 0.,
        inference_seconds: 0.,
        inference_calls: 0,
        mean_batch: 0.,
    };
    league.observe_training(&c, 105);
    let r = &league.outcomes[&100];
    assert_eq!(r.sampled.total_games, 2);
    assert_eq!(r.sampled.points, 1.);
    assert_eq!(r.sampled.cash / r.sampled.games, 50000.);
    assert_eq!(r.greedy.total_games, 1);
    assert_eq!(r.greedy.points, 0.);
    let saved = r.json();
    // Insertion changes the slot index, but historical statistics stay on version 100.
    league.admit(
        Snapshot {
            iteration: 50,
            weights: Json::Obj(vec![]),
            profile: None,
        },
        false,
    );
    assert_eq!(league.champion_index(), 1);
    assert_eq!(league.outcomes[&100].json(), saved);
    let restored = League::restore(&league.json()).unwrap();
    assert_eq!(restored.json(), league.json());
    assert_eq!(
        restored
            .roster(106)
            .schedule(16, &mut crate::learning::policy::Rng(123)),
        league
            .roster(106)
            .schedule(16, &mut crate::learning::policy::Rng(123))
    );
}

#[cfg(feature = "train")]
#[test]
fn periodic_snapshots_advance_without_champion_promotion() {
    use super::league::League;
    use crate::learning::{policy::Policy, tensor};
    tensor::threads(1);
    let p = Policy::mixed_routes(-1, 89, 1e-4).unwrap();
    let mut league = League::new(&p).unwrap();
    assert!(!league.freeze_recent(&p, 9, false).unwrap());
    assert!(league.freeze_recent(&p, 10, false).unwrap());
    assert!(!league.freeze_recent(&p, 10, false).unwrap());
    assert!(league.freeze_recent(&p, 20, false).unwrap());
    assert_eq!(league.champion_iteration, 0);
    assert_eq!(
        league
            .roster(21)
            .recent
            .iter()
            .map(|&i| league.snapshots[i].iteration)
            .collect::<Vec<_>>(),
        vec![20, 10]
    );
    assert_eq!(
        league.snapshots.last().unwrap().weights,
        p.weights_json().unwrap()
    );
}

#[cfg(feature = "train")]
#[test]
fn league_exploration_preserves_random_policy_and_broad_episode_pairs() {
    use super::rollout::league_exploration_plan;
    use crate::learning::policy::Rng;
    let probes: Vec<_> = (0..16).map(|i| i % 4 == 0).collect();
    let plan = league_exploration_plan(&probes, 0.2, &mut Rng(72));
    assert_eq!(plan.iter().filter(|x| x.0 == "greedy").count(), 4);
    assert_eq!(plan.iter().filter(|x| x.0 == "focused").count(), 9);
    assert_eq!(plan.iter().filter(|x| x.0 == "broad").count(), 3);
    for (i, p) in plan.iter().enumerate() {
        assert_eq!(p.0 == "greedy", probes[i]);
        assert_eq!(p.1, if p.0 == "broad" { 0.2 } else { 0. });
    }
    assert_eq!(plan, league_exploration_plan(&probes, 0.2, &mut Rng(72)));
}

#[cfg(feature = "train")]
#[test]
fn pfsp_uses_greedy_evidence_when_all_exploratory_games_lose() {
    use super::league::League;
    use kagg_engine::json::Json;
    let entries = (0..6)
        .map(|i| {
            Json::Obj(vec![
                ("iteration".into(), Json::Str(i.to_string())),
                ("weights".into(), Json::Obj(vec![])),
            ])
        })
        .collect();
    let mut league = League::restore(&Json::Arr(entries)).unwrap();
    for i in 0..3 {
        for _ in 0..32 {
            league
                .outcomes
                .entry(i)
                .or_default()
                .sampled
                .observe(10, 100., 200.);
        }
    }
    for _ in 0..32 {
        league
            .outcomes
            .entry(0)
            .or_default()
            .greedy
            .observe(10, 200., 100.);
        league
            .outcomes
            .entry(1)
            .or_default()
            .greedy
            .observe(10, 100., 200.);
    }
    let r = league.roster(10);
    let probability = |version| {
        r.probabilities[r
            .historical
            .iter()
            .position(|&i| league.snapshots[i].iteration == version)
            .unwrap()]
    };
    assert!(probability(1) > probability(2) && probability(2) > probability(0));
    assert_eq!(league.outcomes[&2].greedy.score(10), 0.5);
}

#[cfg(feature = "train")]
#[test]
fn mixed_exploration_rows_keep_exact_behavior_likelihood_for_ppo() {
    use crate::learning::{
        policy::{Batch, Policy, Rng, Sample},
        tensor,
    };
    tensor::threads(1);
    let mut p = Policy::mixed_routes(-1, 76, 1e-4).unwrap();
    let features = vec![vec![0.; 32], {
        let mut f = vec![0.; 32];
        f[31] = 2.;
        f
    }];
    let mut rows: Vec<_> = [0., 0.2]
        .iter()
        .map(|&exploration| Sample {
            context: vec![0.; 96],
            features: features.clone(),
            exploration,
            ..Sample::default()
        })
        .collect();
    let b = Batch::new(&rows, -1).unwrap();
    let (lp, _) = p.forward(&b).unwrap();
    let actual = p.behavior(&lp, &b).unwrap().data().unwrap();
    let decisions = p.infer(&rows, false, &mut Rng(33)).unwrap();
    for (i, (row, d)) in rows.iter_mut().zip(decisions).enumerate() {
        assert!((d.logp - actual[i * b.width + d.action]).abs() < 1e-6);
        row.action = d.action;
        row.logp = d.logp;
        row.value = d.value;
        row.advantage = 1.;
        row.reward = 1.;
    }
    let u = p.update(&rows, 1, 2, &mut Rng(44)).unwrap();
    assert_eq!(u.updates, 1);
    assert!(u.mean_kl.abs() < 1e-5);
}

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
            });
        }
        let mut rng = Rng(99);
        let decisions = model.infer(&rows, false, &mut rng).unwrap();
        for (r, d) in rows.iter_mut().zip(decisions) {
            r.action = d.action;
            r.logp = d.logp;
            r.value = d.value;
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
                kagg_engine::json::Json::Str("baseline-local-v4".into());
        }
        assert!(resumed.restore(&old).is_err());
    }
}

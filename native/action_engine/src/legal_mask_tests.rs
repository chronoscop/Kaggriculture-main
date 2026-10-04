use super::*;
use crate::catalog::{
    MARKET_ACTION_COUNT, MAX_OWN_UNITS, UNIT_ACTION_COUNT, decode_player_action_ids,
};
use crate::legal_masks::{MARKET_MASK_SIZE, UNIT_MASK_SIZE, write_action_masks_into};

fn masks(engine: &Engine, player: usize) -> (Vec<bool>, Vec<bool>) {
    let mut unit = vec![true; UNIT_MASK_SIZE];
    let mut market = vec![true; MARKET_MASK_SIZE];
    write_action_masks_into(engine, player, &mut unit, &mut market).unwrap();
    (unit, market)
}

fn own_state(engine: &Engine, player: usize) -> String {
    format!("{:?}{:?}", engine.farms[player], engine.privates[player])
}

fn assert_unit_resolver(engine: &Engine, player: usize) {
    let (mask, _) = masks(engine, player);
    let before = own_state(engine, player);
    for unit in 0..(1 + engine.farms[player].hands.len()).min(MAX_OWN_UNITS) {
        for id in 0..UNIT_ACTION_COUNT {
            let mut ids = [4; MAX_OWN_UNITS];
            ids[unit] = id;
            let action = decode_player_action_ids(engine, player, &ids, &[0; 10]).unwrap();
            let selected = if unit == 0 {
                &action.farmer
            } else {
                &action.hands[unit - 1]
            };
            let mut trial = engine.clone();
            apply_unit_action(
                &mut trial.farms[player],
                &mut trial.privates[player],
                unit,
                selected,
                &[false; N_CROPS],
                engine.cfg.board_size,
                engine.step_no / engine.cfg.turns_per_day,
                engine.cfg.turns_per_day,
                engine.cfg.shed_capacity,
            );
            let expected = id == 4 || own_state(&trial, player) != before;
            assert_eq!(
                mask[unit * UNIT_ACTION_COUNT as usize + id as usize],
                expected,
                "unit={unit} id={id} action={selected:?} state={before}"
            );
        }
    }
}

fn assert_market_resolver(engine: &Engine, player: usize) {
    let (_, mask) = masks(engine, player);
    let before = own_state(engine, player);
    for id in 0..MARKET_ACTION_COUNT {
        let mut ids = [0; 10];
        ids[0] = id;
        let mut actions = [PlayerAction::empty(), PlayerAction::empty()];
        actions[player] =
            decode_player_action_ids(engine, player, &[4; MAX_OWN_UNITS], &ids).unwrap();
        let mut trial = engine.clone();
        trial.process_market(&actions);
        assert_eq!(
            mask[id as usize],
            id == 0 || own_state(&trial, player) != before,
            "market id={id} state={before}"
        );
    }
}

#[test]
fn all_unit_catalog_entries_match_resolver_at_tile_and_inventory_boundaries() {
    let mut cases = vec![
        Tile::Empty,
        Tile::Locked,
        Tile::Weed,
        Tile::Structure(Structure::Coop),
        Tile::Structure(Structure::Pasture),
    ];
    for (crop, data) in CROPS.iter().enumerate() {
        for age in [data.first_yield_day - 1, data.first_yield_day] {
            for watered in [false, true] {
                let mut plant = Plant::new(crop, 10 - age, 24);
                plant.watered_today = watered;
                plant.fertilized_until_day = 20;
                plant.yield_units = 1;
                cases.push(Tile::Plant(plant));
            }
        }
    }
    for animal in 0..N_ANIMALS {
        for flags in 0..16 {
            let mut tile = AnimalTile::new(animal, 0);
            tile.yield_units = flags & 1;
            tile.fed_today = flags & 2 != 0;
            tile.cared_today = flags & 4 != 0;
            tile.fertilizer_available = flags & 8 != 0;
            cases.push(Tile::Animal(tile));
        }
    }
    for (case, tile) in cases.into_iter().enumerate() {
        for stocked in [false, true] {
            let mut engine = Engine::new(42, Config::default());
            let player = case % 2;
            engine.step_no = 240;
            engine.cfg.shed_capacity = if case % 2 == 0 { 1 } else { 100 };
            engine.farms[player].money = 0.0;
            engine.farms[player].tiles[4][4] = tile.clone();
            engine.farms[player].tiles[5][5] = tile.clone();
            engine.farms[player].tiles[0][0] = tile.clone();
            engine.farms[player].hands = vec![(5, 5), (0, 0)];
            if stocked {
                engine.privates[player].shed.fill(1);
                engine.privates[player].seeds.fill(1);
            }
            let inventory = if stocked {
                Inventory((0..N_ITEMS).map(|item| (item, 1)).collect())
            } else {
                Inventory::default()
            };
            engine.privates[player].inventories = vec![inventory; 3];
            assert_unit_resolver(&engine, player);
        }
    }
}

#[test]
fn all_market_catalog_entries_match_resolver_at_money_capacity_and_land_boundaries() {
    let base = Engine::new(42, Config::default());
    let product_price = market_price(WHEAT, base.market.inventory[WHEAT] - 1, &base.market.params);
    for (case, money) in [
        0.0,
        0.999,
        1.0,
        9.999,
        10.0,
        19.999,
        20.0,
        50.0,
        80.0,
        100.0,
        299.999,
        300.0,
        400.0,
        500.0,
        999.999,
        1000.0,
        2000.0,
        4000.0,
        product_price as f64 - 0.001,
        product_price as f64,
    ]
    .into_iter()
    .enumerate()
    {
        for full in [false, true] {
            let mut engine = base.clone();
            let player = case % 2;
            engine.farms[player].money = money;
            engine.farms[player].hires_today = case as i64 % 15;
            engine.farms[player].unlocked_quadrants =
                vec!["NW", "NE", "SW", "SE"][..1 + case % 4].to_vec();
            if full {
                engine.privates[player].shed.fill(1);
            }
            engine.cfg.shed_capacity = 12;
            assert_market_resolver(&engine, player);
        }
    }
    let mut negative_stock = base;
    negative_stock.market.inventory[WHEAT] = -100;
    assert_market_resolver(&negative_stock, 0);
}

#[test]
fn initial_terminal_padded_units_and_invalid_inputs() {
    let mut engine = Engine::new(42, Config::default());
    let (mut units, mut market) = masks(&engine, 0);
    let allowed: Vec<_> = units[..500]
        .iter()
        .enumerate()
        .filter_map(|(id, &v)| v.then_some(id))
        .collect();
    assert_eq!(allowed, [0, 1, 2, 3, 4, 494, 495]);
    assert_eq!(market.iter().filter(|&&v| v).count(), 1003);
    assert!(
        units[500..]
            .chunks_exact(500)
            .all(|row| row[4] && row.iter().filter(|&&v| v).count() == 1)
    );
    let before = (units.clone(), market.clone());
    assert!(write_action_masks_into(&engine, 2, &mut units, &mut market).is_err());
    assert!(write_action_masks_into(&engine, 0, &mut units[..5], &mut market).is_err());
    assert_eq!((units.clone(), market.clone()), before);
    engine.done = true;
    write_action_masks_into(&engine, 0, &mut units, &mut market).unwrap();
    assert_eq!(units.iter().filter(|&&v| v).count(), 20);
    assert_eq!(market.iter().filter(|&&v| v).count(), 1);
}

#[test]
fn hidden_state_independence_and_no_state_mutation() {
    let engine = Engine::new(42, Config::default());
    let before = format!(
        "{:?}{:?}{:?}{:?}",
        engine.farms, engine.privates, engine.market, engine.town
    );
    let expected = masks(&engine, 0);
    assert_eq!(
        before,
        format!(
            "{:?}{:?}{:?}{:?}",
            engine.farms, engine.privates, engine.market, engine.town
        )
    );
    let mut different = engine.clone();
    different.privates[1].shed.fill(999);
    different.privates[1].seeds.fill(999);
    different.privates[1].inventories[0].add(WHEAT, 999);
    different.farms[1].money = 123456.0;
    assert_eq!(expected, masks(&different, 0));
}

#[test]
fn independent_masks_do_not_claim_joint_seed_feasibility() {
    let mut engine = Engine::new(42, Config::default());
    engine.privates[0].seeds[WHEAT] = 1;
    engine.farms[0].hands.push((3, 4));
    engine.privates[0].inventories.push(Inventory::default());
    let (units, _) = masks(&engine, 0);
    assert!(units[486] && units[500 + 486]);
    let mut ids = [4; MAX_OWN_UNITS];
    ids[..2].fill(486);
    let action = decode_player_action_ids(&engine, 0, &ids, &[0; 10]).unwrap();
    engine.apply_player_actions(0, &action, 0);
    assert_eq!(engine.privates[0].seeds[WHEAT], 1);
    assert!(matches!(engine.farms[0].tiles[4][4], Tile::Empty));
}

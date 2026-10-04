//! Independent, pre-action effect masks for the frozen 500/1075 action catalogs.
//!
//! Other Unit actions and all other market orders are assumed to be no-ops. These
//! masks do not solve joint resource conflicts or predict an opponent's orders.
//! Only the observer's farm/private state and the public market are read.

use crate::catalog::{
    DROP_ID, MARKET_ACTION_COUNT, MAX_OWN_UNITS, PICKUP_START, PLACE_START, PLANT_START,
    SELL_START, UNIT_ACTION_COUNT,
};
use crate::engine::{Engine, fib, is_shed_adjacent, market_price};
use crate::state::*;

pub const UNIT_MASK_SIZE: usize = MAX_OWN_UNITS * UNIT_ACTION_COUNT as usize;
pub const MARKET_MASK_SIZE: usize = MARKET_ACTION_COUNT as usize;
pub const PASS_ID: usize = 4;
pub const NOOP_ID: usize = 0;
const UNIT_QUANTITIES: usize = 20;
const BUY_QUANTITIES: usize = 100;
const SELL_QUANTITIES: usize = 8;
const WATER_ID: usize = PLANT_START as usize + N_CROPS;
const HARVEST_ID: usize = WATER_ID + 1;
const FERTILIZE_ID: usize = WATER_ID + 2;
const BUILD_COOP_ID: usize = WATER_ID + 3;
const BUILD_PASTURE_ID: usize = WATER_ID + 4;
const DIG_ID: usize = WATER_ID + 5;
const FEED_ID: usize = WATER_ID + 6;
const COLLECT_ID: usize = WATER_ID + 7;
const CARE_ID: usize = WATER_ID + 8;
const HIRE_ID: usize = SELL_START as usize + N_PRODUCTS * SELL_QUANTITIES;
const BUY_LAND_ID: usize = HIRE_ID + 1;

/// Reuses caller-owned dense buffers; no heap allocation, engine clone or token decoding.
pub fn write_action_masks_into(
    engine: &Engine,
    player: usize,
    units: &mut [bool],
    market: &mut [bool],
) -> Result<(), &'static str> {
    if player >= 2 || units.len() != UNIT_MASK_SIZE || market.len() != MARKET_MASK_SIZE {
        return Err("masks require player 0/1, units [20,500] and market [1075]");
    }
    units.fill(false);
    market.fill(false);
    for row in units.chunks_exact_mut(UNIT_ACTION_COUNT as usize) {
        row[PASS_ID] = true;
    }
    market[NOOP_ID] = true;
    if engine.done {
        return Ok(());
    }

    let farm = &engine.farms[player];
    let private = &engine.privates[player];
    let room = private.shed.iter().sum::<i64>() < engine.cfg.shed_capacity;
    let day = engine.step_no / engine.cfg.turns_per_day;
    let empty_inventory = Inventory::default();
    let positions = std::iter::once(farm.farmer).chain(farm.hands.iter().copied());
    for (unit, (mask, (x, y))) in units
        .chunks_exact_mut(UNIT_ACTION_COUNT as usize)
        .zip(positions)
        .enumerate()
    {
        let board_size = engine.cfg.board_size;
        for (action, (dx, dy)) in [(0, -1), (0, 1), (1, 0), (-1, 0)].into_iter().enumerate() {
            // The official engine allows movement onto LOCKED tiles.
            mask[action] =
                (0..board_size).contains(&(x + dx)) && (0..board_size).contains(&(y + dy));
        }
        let tile = &farm.tiles[y as usize][x as usize];
        let inventory = private.inventories.get(unit).unwrap_or(&empty_inventory);
        let adjacent = is_shed_adjacent(x, y, board_size);
        mask[DROP_ID as usize] = adjacent && !inventory.0.is_empty();
        for item in 0..N_ITEMS {
            let pickup = adjacent && private.shed[item] > 0;
            let matching_structure = item >= FIRST_ANIMAL
                && matches!(tile, Tile::Structure(s) if *s == ANIMALS[item - FIRST_ANIMAL].structure);
            let place = inventory.get(item) > 0 && (matching_structure || (adjacent && room));
            // Every positive quantity may be partially fulfilled by the resolver.
            let pickup_start = PICKUP_START as usize + item * UNIT_QUANTITIES;
            mask[pickup_start..pickup_start + UNIT_QUANTITIES].fill(pickup);
            let place_start = PLACE_START as usize + item * UNIT_QUANTITIES;
            mask[place_start..place_start + UNIT_QUANTITIES].fill(place);
        }
        match tile {
            Tile::Empty => {
                for crop in 0..N_CROPS {
                    mask[PLANT_START as usize + crop] = private.seeds[crop] > 0;
                }
                // Building is free, including when the player has no money.
                mask[BUILD_COOP_ID] = true;
                mask[BUILD_PASTURE_ID] = true;
            }
            Tile::Plant(plant) => {
                mask[WATER_ID] = !plant.watered_today;
                mask[HARVEST_ID] = plant.yield_units > 0
                    && day - plant.planted_day >= CROPS[plant.crop].first_yield_day;
                mask[FERTILIZE_ID] = inventory.get(FERTILIZER) > 0;
                mask[DIG_ID] = true;
            }
            Tile::Animal(animal) => {
                mask[HARVEST_ID] = animal.yield_units > 0;
                mask[FEED_ID] = !animal.fed_today && inventory.get(WHEAT) > 0;
                mask[COLLECT_ID] = animal.fertilizer_available;
                mask[CARE_ID] = !animal.cared_today;
            }
            Tile::Weed | Tile::Structure(_) => mask[DIG_ID] = true,
            Tile::Locked => {}
        }
    }

    for (crop, data) in CROPS.iter().enumerate() {
        let start = 1 + crop * BUY_QUANTITIES;
        market[start..start + BUY_QUANTITIES].fill(farm.money >= data.seed as f64);
    }
    for (slot, item) in [WHEAT, FERTILIZER].into_iter().enumerate() {
        let price = market_price(
            item,
            engine.market.inventory[item] - 1,
            &engine.market.params,
        );
        let start = 1 + (N_CROPS + slot) * BUY_QUANTITIES;
        // Public market inventory may be negative; it is a price signal, not stock.
        market[start..start + BUY_QUANTITIES].fill(room && farm.money >= price as f64);
    }
    for (animal, data) in ANIMALS.iter().enumerate() {
        let start = 1 + (N_CROPS + 2 + animal) * BUY_QUANTITIES;
        market[start..start + BUY_QUANTITIES].fill(room && farm.money >= data.cost as f64);
    }
    for item in 0..N_PRODUCTS {
        let start = SELL_START as usize + item * SELL_QUANTITIES;
        market[start..start + SELL_QUANTITIES].fill(private.shed[item] > 0);
    }
    market[HIRE_ID] = farm.money >= (engine.cfg.hire_mult * fib(farm.hires_today)) as f64;
    let extra_land = farm.unlocked_quadrants.len() - 1;
    market[BUY_LAND_ID] =
        extra_land < LAND_PRICES.len() && farm.money >= LAND_PRICES[extra_land] as f64;
    Ok(())
}

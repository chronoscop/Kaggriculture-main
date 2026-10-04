//! Fixed-shape features for the legacy 55548303 tape policy evaluator.

use std::f64::consts::PI;

use crate::engine::Engine;
use crate::state::*;

pub const GLOBAL_DIM: usize = 43;
pub const RESOURCE_TOKENS: usize = 9;
pub const RESOURCE_DIM: usize = 16;
pub const SHOP_TOKENS: usize = 8;
pub const SHOP_DIM: usize = 19;
pub const TILE_TOKENS: usize = 200;
pub const TILE_DIM: usize = 28;
pub const UNIT_TOKENS: usize = 20;
pub const UNIT_DIM: usize = 48;
pub const MARKET_LABEL_SLOTS: usize = 19;
const UNIT_TYPE_COUNT: usize = 18;
const UNIT_INDEX_OFFSET: usize = 35;

pub struct TapeHistory<'a> {
    pub unit_types: &'a [i32],
    pub unit_count: i32,
    pub market_qty: &'a [i32],
    pub hire: i32,
    pub buy_land: i32,
    pub available: bool,
}

pub struct TapeFeatures {
    pub global: Vec<f32>,
    pub resource: Vec<f32>,
    pub shop: Vec<f32>,
    pub tile: Vec<f32>,
    pub unit: Vec<f32>,
    pub unit_pad_mask: Vec<bool>,
}

pub struct TapeFeatureOutput<'a> {
    pub global: &'a mut [f32],
    pub resource: &'a mut [f32],
    pub shop: &'a mut [f32],
    pub tile: &'a mut [f32],
    pub unit: &'a mut [f32],
    pub unit_pad_mask: &'a mut [bool],
}

fn fill_global(engine: &Engine, player: usize, history: &TapeHistory<'_>, values: &mut [f32]) {
    let farm = &engine.farms[player];
    let private = &engine.privates[player];
    let step = engine.step_no;
    let day = step / 24;
    let hour = step % 24;
    values[0] = step as f32 / 720.0;
    values[1] = day as f32 / 30.0;
    values[2] = hour as f32 / 24.0;
    values[3] = (2.0 * PI * hour as f64 / 24.0).sin() as f32;
    values[4] = (2.0 * PI * hour as f64 / 24.0).cos() as f32;
    values[5] = farm.money.max(0.0).ln_1p() as f32;
    values[7] = farm.hires_today as f32 / 10.0;
    values[9] = farm.hands.len() as f32 / 10.0;
    for (index, quadrant) in ["NW", "NE", "SW", "SE"].iter().enumerate() {
        values[11 + index] = farm.unlocked_quadrants.contains(quadrant) as u8 as f32;
    }
    values[19] = private.shed.iter().sum::<i64>() as f32 / 100.0;
    values[20] = 1.0;
    if history.available {
        for (slot, &quantity) in history.market_qty.iter().enumerate() {
            values[21 + slot] = quantity as f32 / 68.0;
        }
        values[40] = history.hire as f32 / 10.0;
        values[41] = history.buy_land as f32;
        values[42] = 1.0;
    }
}

fn fill_resources(engine: &Engine, player: usize, values: &mut [f32]) {
    let private = &engine.privates[player];
    for item in 0..N_PRODUCTS {
        let row = &mut values[item * RESOURCE_DIM..(item + 1) * RESOURCE_DIM];
        row[item] = 1.0;
        row[12] = engine.market.params[item].base.f().ln_1p() as f32;
        row[13] = (private.shed[item] as f64).ln_1p() as f32;
        if item < N_CROPS {
            row[14] = (private.seeds[item] as f64).ln_1p() as f32;
        }
        row[15] = matches!(item, WHEAT | FERTILIZER) as u8 as f32;
    }
}

fn fill_shops(engine: &Engine, values: &mut [f32]) {
    for shop in 0..N_SHOPS {
        let row = &mut values[shop * SHOP_DIM..(shop + 1) * SHOP_DIM];
        row[shop] = 1.0;
        row[8] = engine.town.unlocked_shops.contains(&shop) as u8 as f32;
        for &item in SHOP_PRODUCTS[shop] {
            row[9 + item] = 1.0;
        }
        row[18] = (SHOP_PRODUCTS[shop].len() == 1) as u8 as f32;
    }
}

fn shed_access(x: usize, y: usize) -> bool {
    matches!((x, y), (4, 4) | (5, 4) | (4, 5) | (5, 5))
}

fn fill_tile(tile: &Tile, day: i64, x: usize, y: usize, row: &mut [f32]) {
    row[0] = 1.0;
    row[1] = x as f32 / 9.0;
    row[2] = y as f32 / 9.0;
    row[27] = shed_access(x, y) as u8 as f32;
    match tile {
        Tile::Locked => row[3] = 1.0,
        Tile::Empty => row[4] = 1.0,
        Tile::Weed => row[5] = 1.0,
        Tile::Plant(plant) => {
            row[6] = 1.0;
            row[7 + plant.crop] = 1.0;
            row[12] = (day - plant.planted_day) as f32 / 30.0;
            row[13] = plant.yield_units.max(0) as f32 / 6.0;
            row[14] = plant.watered_today as u8 as f32;
            row[15] = plant.consecutive_unwatered as f32 / 2.0;
            row[16] = (plant.fertilized_until_day - day + 1).max(0) as f32 / 3.0;
        }
        Tile::Structure(structure) => {
            row[match structure {
                Structure::Coop => 17,
                Structure::Pasture => 18,
            }] = 1.0;
        }
        Tile::Animal(animal) => {
            row[match ANIMALS[animal.animal].structure {
                Structure::Coop => 17,
                Structure::Pasture => 18,
            }] = 1.0;
            row[19 + animal.animal] = 1.0;
            row[13] = animal.yield_units.max(0) as f32 / 6.0;
            row[22] = animal.fed_today as u8 as f32;
            row[23] = animal.cared_today as u8 as f32;
            row[24] = animal.fertilizer_available as u8 as f32;
            row[25] = animal.pending_care_bonus.min(3) as f32 / 3.0;
            row[26] = (day - animal.placed_day) as f32 / 30.0;
        }
    }
}

fn fill_tiles(engine: &Engine, player: usize, values: &mut [f32]) {
    let day = engine.step_no / 24;
    for y in 0..10 {
        for x in 0..10 {
            let token = y * 10 + x;
            fill_tile(
                &engine.farms[player].tiles[y][x],
                day,
                x,
                y,
                &mut values[token * TILE_DIM..(token + 1) * TILE_DIM],
            );
        }
    }
}

fn fill_units(
    engine: &Engine,
    player: usize,
    history: &TapeHistory<'_>,
    values: &mut [f32],
    pad_mask: &mut [bool],
) -> Result<(), &'static str> {
    let farm = &engine.farms[player];
    let private = &engine.privates[player];
    let hour = engine.step_no % 24;
    let positions = std::iter::once(farm.farmer).chain(farm.hands.iter().copied());
    for (index, (x, y)) in positions.enumerate() {
        if index >= UNIT_TOKENS {
            return Err("fixed tape features support at most 20 own units");
        }
        let row = &mut values[index * UNIT_DIM..(index + 1) * UNIT_DIM];
        row[0] = (index == 0) as u8 as f32;
        row[1] = 1.0;
        row[2] = x as f32 / 9.0;
        row[3] = y as f32 / 9.0;
        row[UNIT_INDEX_OFFSET + index.min(12)] = 1.0;
        if let Some(inventory) = private.inventories.get(index) {
            for &(item, count) in &inventory.0 {
                row[4 + item] = (count as f64).ln_1p() as f32;
            }
        }
        let aligns = index == 0 || (hour != 0 && index < history.unit_count.max(0) as usize);
        if history.available && aligns {
            let action = history.unit_types[index];
            if !(0..UNIT_TYPE_COUNT as i32).contains(&action) {
                return Err("previous unit type is outside the fixed tape vocabulary");
            }
            row[16 + action as usize] = 1.0;
            row[34] = 1.0;
        }
        pad_mask[index] = false;
    }
    Ok(())
}

pub fn encode_fixed_tape(
    engine: &Engine,
    player: usize,
    history: &TapeHistory<'_>,
) -> Result<TapeFeatures, &'static str> {
    if player > 1
        || engine.cfg.board_size != 10
        || history.unit_types.len() != UNIT_TOKENS
        || history.market_qty.len() != MARKET_LABEL_SLOTS
    {
        return Err("invalid fixed tape feature input shape");
    }
    let mut result = TapeFeatures {
        global: vec![0.0; GLOBAL_DIM],
        resource: vec![0.0; RESOURCE_TOKENS * RESOURCE_DIM],
        shop: vec![0.0; SHOP_TOKENS * SHOP_DIM],
        tile: vec![0.0; TILE_TOKENS * TILE_DIM],
        unit: vec![0.0; UNIT_TOKENS * UNIT_DIM],
        unit_pad_mask: vec![true; UNIT_TOKENS],
    };
    encode_fixed_tape_into(
        engine,
        player,
        history,
        TapeFeatureOutput {
            global: &mut result.global,
            resource: &mut result.resource,
            shop: &mut result.shop,
            tile: &mut result.tile,
            unit: &mut result.unit,
            unit_pad_mask: &mut result.unit_pad_mask,
        },
    )?;
    Ok(result)
}

pub fn encode_fixed_tape_into(
    engine: &Engine,
    player: usize,
    history: &TapeHistory<'_>,
    output: TapeFeatureOutput<'_>,
) -> Result<(), &'static str> {
    if player > 1
        || engine.cfg.board_size != 10
        || history.unit_types.len() != UNIT_TOKENS
        || history.market_qty.len() != MARKET_LABEL_SLOTS
        || output.global.len() != GLOBAL_DIM
        || output.resource.len() != RESOURCE_TOKENS * RESOURCE_DIM
        || output.shop.len() != SHOP_TOKENS * SHOP_DIM
        || output.tile.len() != TILE_TOKENS * TILE_DIM
        || output.unit.len() != UNIT_TOKENS * UNIT_DIM
        || output.unit_pad_mask.len() != UNIT_TOKENS
    {
        return Err("invalid fixed tape feature output shape");
    }
    output.global.fill(0.0);
    output.resource.fill(0.0);
    output.shop.fill(0.0);
    output.tile.fill(0.0);
    output.unit.fill(0.0);
    output.unit_pad_mask.fill(true);
    fill_global(engine, player, history, output.global);
    fill_resources(engine, player, output.resource);
    fill_shops(engine, output.shop);
    fill_tiles(engine, player, output.tile);
    fill_units(engine, player, history, output.unit, output.unit_pad_mask)
}

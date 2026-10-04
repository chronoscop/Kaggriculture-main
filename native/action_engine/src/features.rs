//! Fixed-shape exp43 feature encoding directly from the authoritative Rust state.

use crate::engine::Engine;
use crate::state::*;

pub const FEATURE_DIM: usize = 124;
pub const MEMORY_DIM: usize = 50;
pub const FIXED_TOKENS: usize = 264;
pub const MAX_OWN_UNITS: usize = 20;
pub const MAX_TOTAL_UNITS: usize = 40;
pub const MARKET_SLOTS: usize = 10;

const TOKEN_GLOBAL: usize = 0;
const TOKEN_CELL: usize = 1;
const TOKEN_UNIT: usize = 2;
const TOKEN_PRODUCT: usize = 3;
const TOKEN_MEMORY: usize = 4;
const TOKEN_MARKET_SLOT: usize = 5;
const FARM_NONE: usize = 6;
const FARM_SELF: usize = 7;
const FARM_OPPONENT: usize = 8;
const TILE_EMPTY: usize = 9;
const MEMORY_PACK_START: usize = TILE_EMPTY;
const TILE_LOCKED: usize = 10;
const TILE_WEED: usize = 11;
const TILE_PLANT: usize = 12;
const TILE_COOP: usize = 13;
const TILE_PASTURE: usize = 14;
const CROP_START: usize = 15;
const ANIMAL_START: usize = 20;
const ITEM_START: usize = 23;
const UNIT_FARMER: usize = 35;
const UNIT_HAND: usize = 36;
const MARKET_SLOT_START: usize = 37;
const PLANT_AGE: usize = 47;
const YIELD_UNITS: usize = 48;
const WATERED: usize = 49;
const CONSECUTIVE_UNWATERED: usize = 50;
const FERTILIZER_REMAINING: usize = 51;
const HAS_DECAY_STEP: usize = 52;
const TURNS_UNTIL_DECAY: usize = 53;
const ANIMAL_AGE: usize = 54;
const FED: usize = 55;
const CONSECUTIVE_UNFED: usize = 56;
const CARED: usize = 57;
const FERTILIZER_AVAILABLE: usize = 58;
const PENDING_CARE_BONUS: usize = 59;
const SHED_ACCESS: usize = 60;
const UNIT_SHED_ACCESS: usize = 61;
const FARMER_OCCUPANCY: usize = 62;
const HAND_OCCUPANCY: usize = 63;
const CELL_X: usize = 64;
const CELL_Y: usize = 65;
const UNIT_INDEX: usize = 66;
const UNIT_X: usize = 67;
const UNIT_Y: usize = 68;
const INVENTORY_VISIBLE: usize = 69;
const INVENTORY_START: usize = 70;
const SEASON_PROGRESS: usize = 82;
const DAY_PROGRESS: usize = 83;
const DAY_REMAINING: usize = 84;
const SEASON_REMAINING: usize = 85;
const SHOP_TICK_PROGRESS: usize = 86;
const TOWN_TICK_PROGRESS: usize = 87;
const SELF_MONEY: usize = 88;
const OPPONENT_MONEY: usize = 89;
const SELF_HIRES_TODAY: usize = 90;
const OPPONENT_HIRES_TODAY: usize = 91;
const SELF_UNLOCKED_COUNT: usize = 92;
const OPPONENT_UNLOCKED_COUNT: usize = 93;
const SELF_QUADRANT_START: usize = 94;
const OPPONENT_QUADRANT_START: usize = 98;
const SHOP_COUNT_START: usize = 102;
const PRODUCT_SHED_COUNT: usize = 110;
const PRODUCT_SEED_COUNT: usize = 111;
const PRODUCT_SEED_APPLICABLE: usize = 112;
const PRODUCT_MARKET_INVENTORY: usize = 113;
const PRODUCT_MARKET_PRICE: usize = 114;
const PRODUCT_MARKET_VISIBLE: usize = 115;
const PRODUCT_TOWN_DEMAND: usize = 116;
const SHED_TOTAL: usize = 117;
const SHED_ROOM: usize = 118;
const UNITS_INVENTORY_TOTAL: usize = 119;
const SHED_OVERFLOW_PENDING: usize = 120;
const INVENTORY_TOTAL: usize = 121;
const SELF_HAND_COUNT: usize = 122;
const OPPONENT_HAND_COUNT: usize = 123;

const BOARD_SIZE: usize = 10;
const EPISODE_STEPS: f64 = 720.0;
const SHED_CAPACITY: f64 = 100.0;

pub struct FixedFeatures {
    pub features: Vec<f32>,
    pub memory_features: Vec<f32>,
    pub coordinates: Vec<f32>,
    pub spatial_mask: Vec<bool>,
    pub rope_groups: Vec<i32>,
    pub token_mask: Vec<bool>,
    pub unit_indices: Vec<i32>,
    pub market_indices: Vec<i32>,
    pub actual_own_units: usize,
    pub encoded_own_units: usize,
}

#[derive(Clone, Copy)]
pub struct FeatureCounts {
    pub actual_own_units: usize,
    pub encoded_own_units: usize,
}

fn clipped(value: f64, low: f64, high: f64) -> f32 {
    value.max(low).min(high) as f32
}

fn log_scaled_exact(value: f64, reference: f64) -> f32 {
    (value.max(0.0).ln_1p() / reference.ln_1p()) as f32
}

fn shed_access(x: i64, y: i64) -> bool {
    matches!((x, y), (4, 4) | (5, 4) | (4, 5) | (5, 5))
}

fn feature_row(features: &mut [f32], token: usize) -> &mut [f32] {
    &mut features[token * FEATURE_DIM..(token + 1) * FEATURE_DIM]
}

fn set_base(row: &mut [f32], token_type: usize, farm_role: usize) {
    row[token_type] = 1.0;
    row[farm_role] = 1.0;
}

fn fill_global(engine: &Engine, player: usize, row: &mut [f32]) {
    let opponent = 1 - player;
    let step = engine.step_no;
    let hour = step % 24;
    let self_farm = &engine.farms[player];
    let opponent_farm = &engine.farms[opponent];
    set_base(row, TOKEN_GLOBAL, FARM_NONE);
    row[SEASON_PROGRESS] = clipped(step as f64 / 719.0, 0.0, 1.0);
    row[DAY_PROGRESS] = clipped(hour as f64 / 23.0, 0.0, 1.0);
    row[DAY_REMAINING] = clipped((23 - hour) as f64 / 24.0, 0.0, 1.0);
    row[SEASON_REMAINING] = clipped((719 - step) as f64 / EPISODE_STEPS, 0.0, 1.0);
    row[SHOP_TICK_PROGRESS] = (step % 4) as f32 / 3.0;
    row[TOWN_TICK_PROGRESS] = (step % 24) as f32 / 23.0;
    row[SELF_MONEY] = log_scaled_exact(self_farm.money, 1_000_000.0);
    row[OPPONENT_MONEY] = log_scaled_exact(opponent_farm.money, 1_000_000.0);
    row[SELF_HIRES_TODAY] = log_scaled_exact(self_farm.hires_today as f64, 32.0);
    row[OPPONENT_HIRES_TODAY] = log_scaled_exact(opponent_farm.hires_today as f64, 32.0);
    row[SELF_UNLOCKED_COUNT] = self_farm.unlocked_quadrants.len() as f32 / 4.0;
    row[OPPONENT_UNLOCKED_COUNT] = opponent_farm.unlocked_quadrants.len() as f32 / 4.0;
    for (index, quadrant) in ["NW", "NE", "SW", "SE"].iter().enumerate() {
        row[SELF_QUADRANT_START + index] =
            self_farm.unlocked_quadrants.contains(quadrant) as u8 as f32;
        row[OPPONENT_QUADRANT_START + index] =
            opponent_farm.unlocked_quadrants.contains(quadrant) as u8 as f32;
    }
    for &shop in &engine.town.unlocked_shops {
        row[SHOP_COUNT_START + shop] += 1.0 / 8.0;
    }
    let private = &engine.privates[player];
    let shed_total: i64 = private.shed.iter().map(|&count| count.max(0)).sum();
    let shed_room = (SHED_CAPACITY as i64 - shed_total).max(0);
    let carried: i64 = private
        .inventories
        .iter()
        .flat_map(|inventory| inventory.0.iter().map(|&(_, count)| count.max(0)))
        .sum();
    row[SHED_TOTAL] = clipped(shed_total as f64 / SHED_CAPACITY, 0.0, 1.0);
    row[SHED_ROOM] = clipped(shed_room as f64 / SHED_CAPACITY, 0.0, 1.0);
    // Linear like the Python encoder so carried items compare with shed room.
    row[UNITS_INVENTORY_TOTAL] = (carried as f64 / SHED_CAPACITY) as f32;
    row[SHED_OVERFLOW_PENDING] = clipped((carried - shed_room).max(0) as f64 / SHED_CAPACITY, 0.0, 1.0);
    row[SELF_HAND_COUNT] = self_farm.hands.len() as f32 / 32.0;
    row[OPPONENT_HAND_COUNT] = opponent_farm.hands.len() as f32 / 32.0;
}

fn fill_cell(
    tile: &Tile,
    row: &mut [f32],
    time: (i64, i64),
    position: (usize, usize),
    occupancy: (usize, usize),
) {
    let (day, step) = time;
    let (x, y) = position;
    let (farmer_count, hand_count) = occupancy;
    row[SHED_ACCESS] = shed_access(x as i64, y as i64) as u8 as f32;
    row[FARMER_OCCUPANCY] = clipped(farmer_count as f64, 0.0, 1.0);
    row[HAND_OCCUPANCY] = log_scaled_exact(hand_count as f64, 32.0);
    row[CELL_X] = clipped(x as f64 / 9.0, 0.0, 1.0);
    row[CELL_Y] = clipped(y as f64 / 9.0, 0.0, 1.0);
    match tile {
        Tile::Empty => row[TILE_EMPTY] = 1.0,
        Tile::Locked => row[TILE_LOCKED] = 1.0,
        Tile::Weed => row[TILE_WEED] = 1.0,
        Tile::Structure(structure) => {
            row[match structure {
                Structure::Coop => TILE_COOP,
                Structure::Pasture => TILE_PASTURE,
            }] = 1.0;
        }
        Tile::Plant(plant) => {
            row[TILE_PLANT] = 1.0;
            row[CROP_START + plant.crop] = 1.0;
            row[PLANT_AGE] = clipped((day - plant.planted_day) as f64 / 30.0, 0.0, 1.0);
            row[YIELD_UNITS] = log_scaled_exact(plant.yield_units as f64, 8.0);
            row[WATERED] = plant.watered_today as u8 as f32;
            row[CONSECUTIVE_UNWATERED] =
                clipped(plant.consecutive_unwatered as f64 / 2.0, 0.0, 1.0);
            row[FERTILIZER_REMAINING] = clipped(
                (plant.fertilized_until_day - day + 1) as f64 / 3.0,
                0.0,
                1.0,
            );
            if plant.max_lifespan_step >= 0 {
                row[HAS_DECAY_STEP] = 1.0;
                row[TURNS_UNTIL_DECAY] = clipped(
                    (plant.max_lifespan_step - step) as f64 / EPISODE_STEPS,
                    -1.0,
                    1.0,
                );
            }
        }
        Tile::Animal(animal) => {
            row[match ANIMALS[animal.animal].structure {
                Structure::Coop => TILE_COOP,
                Structure::Pasture => TILE_PASTURE,
            }] = 1.0;
            row[ANIMAL_START + animal.animal] = 1.0;
            row[ANIMAL_AGE] = clipped((day - animal.placed_day) as f64 / 30.0, 0.0, 1.0);
            row[YIELD_UNITS] = log_scaled_exact(animal.yield_units as f64, 8.0);
            row[FED] = animal.fed_today as u8 as f32;
            row[CONSECUTIVE_UNFED] = clipped(animal.consecutive_unfed as f64 / 2.0, 0.0, 1.0);
            row[CARED] = animal.cared_today as u8 as f32;
            row[FERTILIZER_AVAILABLE] = animal.fertilizer_available as u8 as f32;
            row[PENDING_CARE_BONUS] = log_scaled_exact(animal.pending_care_bonus as f64, 16.0);
        }
    }
}

fn fill_unit(
    row: &mut [f32],
    role: usize,
    unit_index: usize,
    position: (i64, i64),
    inventory: Option<&Inventory>,
) {
    set_base(row, TOKEN_UNIT, role);
    row[if unit_index == 0 {
        UNIT_FARMER
    } else {
        UNIT_HAND
    }] = 1.0;
    row[UNIT_INDEX] = clipped(unit_index as f64 / 31.0, 0.0, 1.0);
    row[UNIT_X] = clipped(position.0 as f64 / 9.0, 0.0, 1.0);
    row[UNIT_Y] = clipped(position.1 as f64 / 9.0, 0.0, 1.0);
    row[UNIT_SHED_ACCESS] = shed_access(position.0, position.1) as u8 as f32;
    if let Some(inventory) = inventory {
        row[INVENTORY_VISIBLE] = 1.0;
        for item in 0..N_ITEMS {
            row[INVENTORY_START + item] =
                log_scaled_exact(inventory.get(item) as f64, SHED_CAPACITY);
        }
        let total: i64 = (0..N_ITEMS).map(|item| inventory.get(item).max(0)).sum();
        row[INVENTORY_TOTAL] = (total as f64 / SHED_CAPACITY) as f32;
    }
}

fn town_demand(engine: &Engine) -> [i64; N_PRODUCTS] {
    let mut demand = [0; N_PRODUCTS];
    demand[..FERTILIZER].fill(1);
    for &shop in &engine.town.unlocked_shops {
        let products = SHOP_PRODUCTS[shop];
        let multiplier = if products.len() == 1 { 2 } else { 1 };
        for &item in products {
            demand[item] += multiplier;
        }
    }
    demand
}

fn fill_product(
    engine: &Engine,
    player: usize,
    item: usize,
    demand: &[i64; N_PRODUCTS],
    row: &mut [f32],
) {
    set_base(row, TOKEN_PRODUCT, FARM_NONE);
    row[ITEM_START + item] = 1.0;
    row[PRODUCT_SHED_COUNT] =
        log_scaled_exact(engine.privates[player].shed[item] as f64, SHED_CAPACITY);
    if item < N_CROPS {
        row[PRODUCT_SEED_APPLICABLE] = 1.0;
        row[PRODUCT_SEED_COUNT] =
            log_scaled_exact(engine.privates[player].seeds[item] as f64, 128.0);
    }
    if item < N_PRODUCTS {
        row[PRODUCT_MARKET_VISIBLE] = 1.0;
        row[PRODUCT_MARKET_INVENTORY] =
            (((engine.market.inventory[item] - MARKET_I0) as f64 / 100.0).asinh() / 8.0) as f32;
        row[PRODUCT_MARKET_PRICE] = log_scaled_exact(engine.market.prices[item].f(), 10_000.0);
        row[PRODUCT_TOWN_DEMAND] = demand[item] as f32 / 20.0;
    }
}

pub fn encode_partitioned_into(
    engine: &Engine,
    player: usize,
    memory: &[f32],
) -> Result<FixedFeatures, &'static str> {
    let mut features = vec![0.0; FIXED_TOKENS * FEATURE_DIM];
    encode_partitioned_features_into(engine, player, memory, &mut features)?;
    encode_fixed_from_partitioned(engine, player, memory, features)
}

pub fn encode_partitioned_features_into(
    engine: &Engine,
    player: usize,
    memory: &[f32],
    features: &mut [f32],
) -> Result<FeatureCounts, &'static str> {
    if player > 1
        || engine.cfg.board_size != BOARD_SIZE as i64
        || memory.len() != MEMORY_DIM
        || features.len() != FIXED_TOKENS * FEATURE_DIM
    {
        return Err(
            "fixed feature encoder requires player 0/1, a 10x10 board, 50 memory values, and the fixed output shape",
        );
    }
    features.fill(0.0);
    let opponent = 1 - player;
    let actual_own_units = 1 + engine.farms[player].hands.len();
    let actual_opponent_units = 1 + engine.farms[opponent].hands.len();
    let encoded_own_units = actual_own_units.min(MAX_OWN_UNITS);
    let encoded_opponent_units = actual_opponent_units.min(MAX_TOTAL_UNITS - encoded_own_units);
    let selected_tokens = 1
        + 2 * BOARD_SIZE * BOARD_SIZE
        + encoded_own_units
        + encoded_opponent_units
        + N_ITEMS
        + 1
        + MARKET_SLOTS;
    fill_global(engine, player, feature_row(features, 0));

    let (day, _) = engine.day_hour();
    let mut cursor = 1;
    for (farm_player, role) in [(player, FARM_SELF), (opponent, FARM_OPPONENT)] {
        let farm = &engine.farms[farm_player];
        let mut farmer_counts = [0usize; BOARD_SIZE * BOARD_SIZE];
        let mut hand_counts = [0usize; BOARD_SIZE * BOARD_SIZE];
        farmer_counts[farm.farmer.1 as usize * BOARD_SIZE + farm.farmer.0 as usize] += 1;
        for &(x, y) in &farm.hands {
            hand_counts[y as usize * BOARD_SIZE + x as usize] += 1;
        }
        for y in 0..BOARD_SIZE {
            for x in 0..BOARD_SIZE {
                let token = cursor + y * BOARD_SIZE + x;
                set_base(feature_row(features, token), TOKEN_CELL, role);
                fill_cell(
                    &farm.tiles[y][x],
                    feature_row(features, token),
                    (day, engine.step_no),
                    (x, y),
                    (
                        farmer_counts[y * BOARD_SIZE + x],
                        hand_counts[y * BOARD_SIZE + x],
                    ),
                );
            }
        }
        cursor += BOARD_SIZE * BOARD_SIZE;
    }

    let own_positions = std::iter::once(engine.farms[player].farmer)
        .chain(engine.farms[player].hands.iter().copied());
    for (index, position) in own_positions.take(encoded_own_units).enumerate() {
        fill_unit(
            feature_row(features, cursor),
            FARM_SELF,
            index,
            position,
            engine.privates[player].inventories.get(index),
        );
        cursor += 1;
    }
    let opponent_positions = std::iter::once(engine.farms[opponent].farmer)
        .chain(engine.farms[opponent].hands.iter().copied());
    for (index, position) in opponent_positions.take(encoded_opponent_units).enumerate() {
        fill_unit(
            feature_row(features, cursor),
            FARM_OPPONENT,
            index,
            position,
            None,
        );
        cursor += 1;
    }

    let demand = town_demand(engine);
    for item in 0..N_ITEMS {
        fill_product(engine, player, item, &demand, feature_row(features, cursor));
        cursor += 1;
    }
    set_base(feature_row(features, cursor), TOKEN_MEMORY, FARM_NONE);
    feature_row(features, cursor)[MEMORY_PACK_START..MEMORY_PACK_START + MEMORY_DIM]
        .copy_from_slice(memory);
    cursor += 1;
    for slot in 0..MARKET_SLOTS {
        let row = feature_row(features, cursor);
        set_base(row, TOKEN_MARKET_SLOT, FARM_NONE);
        row[MARKET_SLOT_START + slot] = 1.0;
        cursor += 1;
    }
    debug_assert_eq!(cursor, selected_tokens);
    Ok(FeatureCounts {
        actual_own_units,
        encoded_own_units,
    })
}

fn encode_fixed_from_partitioned(
    engine: &Engine,
    player: usize,
    memory: &[f32],
    features: Vec<f32>,
) -> Result<FixedFeatures, &'static str> {
    let opponent = 1 - player;
    let actual_own_units = 1 + engine.farms[player].hands.len();
    let actual_opponent_units = 1 + engine.farms[opponent].hands.len();
    let encoded_own_units = actual_own_units.min(MAX_OWN_UNITS);
    let encoded_opponent_units = actual_opponent_units.min(MAX_TOTAL_UNITS - encoded_own_units);
    let selected_tokens = 1
        + 2 * BOARD_SIZE * BOARD_SIZE
        + encoded_own_units
        + encoded_opponent_units
        + N_ITEMS
        + 1
        + MARKET_SLOTS;
    let mut coordinates = vec![0.0; FIXED_TOKENS * 2];
    let mut spatial_mask = vec![false; FIXED_TOKENS];
    let mut rope_groups = vec![0; FIXED_TOKENS];
    let mut token_mask = vec![false; FIXED_TOKENS];
    let mut unit_indices = vec![0; MAX_OWN_UNITS];
    let mut market_indices = vec![0; MARKET_SLOTS];
    token_mask[..selected_tokens].fill(true);

    let mut unit_cursor = 0;
    for token in 0..selected_tokens {
        let row = &features[token * FEATURE_DIM..(token + 1) * FEATURE_DIM];
        let spatial = row[TOKEN_CELL] > 0.5 || row[TOKEN_UNIT] > 0.5;
        if spatial {
            coordinates[token * 2] = ((row[CELL_X] + row[UNIT_X]) * 9.0).round();
            coordinates[token * 2 + 1] = ((row[CELL_Y] + row[UNIT_Y]) * 9.0).round();
            spatial_mask[token] = true;
            rope_groups[token] = if row[FARM_SELF] > 0.5 { 1 } else { 2 };
        }
        if row[TOKEN_UNIT] > 0.5 && row[FARM_SELF] > 0.5 && unit_cursor < encoded_own_units {
            unit_indices[unit_cursor] = token as i32;
            unit_cursor += 1;
        }
        if row[TOKEN_MARKET_SLOT] > 0.5 {
            let slot = row[MARKET_SLOT_START..MARKET_SLOT_START + MARKET_SLOTS]
                .iter()
                .position(|&value| value > 0.5)
                .ok_or("market slot token is missing its slot id")?;
            market_indices[slot] = token as i32;
        }
    }

    Ok(FixedFeatures {
        features,
        memory_features: memory.to_vec(),
        coordinates,
        spatial_mask,
        rope_groups,
        token_mask,
        unit_indices,
        market_indices,
        actual_own_units,
        encoded_own_units,
    })
}

pub fn encode_fixed(
    engine: &Engine,
    player: usize,
    memory: &[f32],
) -> Result<FixedFeatures, &'static str> {
    encode_partitioned_into(engine, player, memory)
}

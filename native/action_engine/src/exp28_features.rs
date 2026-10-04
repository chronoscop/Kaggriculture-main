//! Exact fixed-shape feature encoder for the experiment 28 summary policy.

use crate::engine::Engine;
use crate::state::*;

pub const FEATURE_DIM: usize = 168;
pub const MEMORY_DIM: usize = 50;
pub const FIXED_TOKENS: usize = 154;
pub const MAX_OWN_UNITS: usize = 20;
pub const MARKET_SLOTS: usize = 10;

const TOKEN_GLOBAL: usize = 0;
const TOKEN_CELL: usize = 1;
const TOKEN_UNIT: usize = 2;
const TOKEN_OPPONENT_CAPACITY: usize = 3;
const TOKEN_OPPONENT_PRODUCT: usize = 4;
const TOKEN_PRODUCT: usize = 5;
const TOKEN_MEMORY: usize = 6;
const TOKEN_MARKET_SLOT: usize = 7;
const FARM_NONE: usize = 8;
const FARM_SELF: usize = 9;
const FARM_OPPONENT: usize = 10;
const TILE_EMPTY: usize = 11;
const TILE_LOCKED: usize = 12;
const TILE_WEED: usize = 13;
const TILE_PLANT: usize = 14;
const TILE_COOP: usize = 15;
const TILE_PASTURE: usize = 16;
const CROP_START: usize = 17;
const ANIMAL_START: usize = 22;
const ITEM_START: usize = 25;
const UNIT_FARMER: usize = 37;
const UNIT_HAND: usize = 38;
const MARKET_SLOT_START: usize = 39;
const PLANT_AGE: usize = 49;
const YIELD_UNITS: usize = 50;
const WATERED: usize = 51;
const CONSECUTIVE_UNWATERED: usize = 52;
const FERTILIZER_REMAINING: usize = 53;
const HAS_DECAY_STEP: usize = 54;
const TURNS_UNTIL_DECAY: usize = 55;
const ANIMAL_AGE: usize = 56;
const FED: usize = 57;
const CONSECUTIVE_UNFED: usize = 58;
const CARED: usize = 59;
const FERTILIZER_AVAILABLE: usize = 60;
const PENDING_CARE_BONUS: usize = 61;
const SHED_ACCESS: usize = 62;
const UNIT_SHED_ACCESS: usize = 63;
const FARMER_OCCUPANCY: usize = 64;
const HAND_OCCUPANCY: usize = 65;
const CELL_X: usize = 66;
const CELL_Y: usize = 67;
const UNIT_INDEX: usize = 68;
const UNIT_X: usize = 69;
const UNIT_Y: usize = 70;
const INVENTORY_VISIBLE: usize = 71;
const INVENTORY_START: usize = 72;
const SEASON_PROGRESS: usize = 84;
const DAY_PROGRESS: usize = 85;
const DAY_REMAINING: usize = 86;
const SEASON_REMAINING: usize = 87;
const SHOP_TICK_PROGRESS: usize = 88;
const TOWN_TICK_PROGRESS: usize = 89;
const SELF_MONEY: usize = 90;
const OPPONENT_MONEY: usize = 91;
const SELF_HIRES_TODAY: usize = 92;
const OPPONENT_HIRES_TODAY: usize = 93;
const SELF_UNLOCKED_COUNT: usize = 94;
const OPPONENT_UNLOCKED_COUNT: usize = 95;
const SELF_QUADRANT_START: usize = 96;
const OPPONENT_QUADRANT_START: usize = 100;
const SHOP_COUNT_START: usize = 104;
const PRODUCT_SHED_COUNT: usize = 112;
const PRODUCT_SEED_COUNT: usize = 113;
const PRODUCT_SEED_APPLICABLE: usize = 114;
const PRODUCT_MARKET_INVENTORY: usize = 115;
const PRODUCT_MARKET_PRICE: usize = 116;
const PRODUCT_MARKET_VISIBLE: usize = 117;
const PRODUCT_TOWN_DEMAND: usize = 118;
const OPPONENT_CROP_COUNT_START: usize = 119;
const OPPONENT_ANIMAL_COUNT_START: usize = 124;
const OPPONENT_WEED_COUNT: usize = 127;
const OPPONENT_EMPTY_CROP_COUNT: usize = 128;
const OPPONENT_EMPTY_COOP_COUNT: usize = 129;
const OPPONENT_EMPTY_PASTURE_COUNT: usize = 130;
const OPPONENT_CURRENT_UNIT_COUNT: usize = 131;
const OPPONENT_REMAINING_ACTION_SLOTS: usize = 132;
const OPPONENT_CRITICAL_COUNT: usize = 133;
const OPPONENT_REACHABLE_CRITICAL_COUNT: usize = 134;
const READY_UNITS_START: usize = 135;
const READY_CELLS_START: usize = 141;
const READY_UNITS_DAY_END: usize = 147;
const READY_CELLS_DAY_END: usize = 148;
const NOMINAL_CAPACITY_START: usize = 149;
const BOOSTED_CAPACITY_START: usize = 153;
const NOMINAL_ACTIONS_START: usize = 157;
const BOOSTED_EXTRA_ACTIONS_START: usize = 161;
const PRODUCTIVE_CELLS_START: usize = 165;

const BOARD_SIZE: usize = 10;
const EPISODE_STEPS: f64 = 720.0;
const SHED_CAPACITY: f64 = 100.0;
const CAPACITY_HORIZONS: [usize; 4] = [1, 3, 7, 14];

#[derive(Clone, Copy)]
pub struct FeatureCounts {
    pub actual_own_units: usize,
    pub encoded_own_units: usize,
}

#[derive(Clone, Copy, Default)]
struct ProductSummary {
    ready_units: [f64; 6],
    ready_cells: [f64; 6],
    ready_units_day_end: f64,
    ready_cells_day_end: f64,
    nominal_capacity: [f64; 4],
    boosted_capacity: [f64; 4],
    nominal_actions: [f64; 4],
    boosted_extra_actions: [f64; 4],
    productive_cells: [f64; 3],
}

fn clipped(value: f64, low: f64, high: f64) -> f32 {
    value.max(low).min(high) as f32
}

fn log_scaled(value: f64, reference: f64) -> f32 {
    (value.max(0.0).ln_1p() / reference.ln_1p()) as f32
}

fn row(features: &mut [f32], token: usize) -> &mut [f32] {
    &mut features[token * FEATURE_DIM..(token + 1) * FEATURE_DIM]
}

fn set_base(values: &mut [f32], token: usize, farm: usize) {
    values[token] = 1.0;
    values[farm] = 1.0;
}

fn shed_access(x: i64, y: i64) -> bool {
    matches!((x, y), (4, 4) | (5, 4) | (4, 5) | (5, 5))
}

fn fill_global(engine: &Engine, player: usize, values: &mut [f32]) {
    let opponent = 1 - player;
    let step = engine.step_no;
    let hour = step % 24;
    let own = &engine.farms[player];
    let other = &engine.farms[opponent];
    set_base(values, TOKEN_GLOBAL, FARM_NONE);
    values[SEASON_PROGRESS] = clipped(step as f64 / 719.0, 0.0, 1.0);
    values[DAY_PROGRESS] = clipped(hour as f64 / 23.0, 0.0, 1.0);
    values[DAY_REMAINING] = clipped((23 - hour) as f64 / 24.0, 0.0, 1.0);
    values[SEASON_REMAINING] = clipped((719 - step) as f64 / EPISODE_STEPS, 0.0, 1.0);
    values[SHOP_TICK_PROGRESS] = (step % 4) as f32 / 3.0;
    values[TOWN_TICK_PROGRESS] = (step % 24) as f32 / 23.0;
    values[SELF_MONEY] = log_scaled(own.money, 1_000_000.0);
    values[OPPONENT_MONEY] = log_scaled(other.money, 1_000_000.0);
    values[SELF_HIRES_TODAY] = log_scaled(own.hires_today as f64, 32.0);
    values[OPPONENT_HIRES_TODAY] = log_scaled(other.hires_today as f64, 32.0);
    values[SELF_UNLOCKED_COUNT] = own.unlocked_quadrants.len() as f32 / 4.0;
    values[OPPONENT_UNLOCKED_COUNT] = other.unlocked_quadrants.len() as f32 / 4.0;
    for (index, quadrant) in ["NW", "NE", "SW", "SE"].iter().enumerate() {
        values[SELF_QUADRANT_START + index] =
            own.unlocked_quadrants.contains(quadrant) as u8 as f32;
        values[OPPONENT_QUADRANT_START + index] =
            other.unlocked_quadrants.contains(quadrant) as u8 as f32;
    }
    for &shop in &engine.town.unlocked_shops {
        values[SHOP_COUNT_START + shop] += 1.0 / 8.0;
    }
}

fn fill_cell(
    tile: &Tile,
    time: (i64, i64),
    position: (usize, usize),
    occupancy: (usize, usize),
    values: &mut [f32],
) {
    let (day, step) = time;
    let (x, y) = position;
    let (farmer_count, hand_count) = occupancy;
    set_base(values, TOKEN_CELL, FARM_SELF);
    values[SHED_ACCESS] = shed_access(x as i64, y as i64) as u8 as f32;
    values[FARMER_OCCUPANCY] = clipped(farmer_count as f64, 0.0, 1.0);
    values[HAND_OCCUPANCY] = log_scaled(hand_count as f64, 32.0);
    values[CELL_X] = clipped(x as f64 / 9.0, 0.0, 1.0);
    values[CELL_Y] = clipped(y as f64 / 9.0, 0.0, 1.0);
    match tile {
        Tile::Empty => values[TILE_EMPTY] = 1.0,
        Tile::Locked => values[TILE_LOCKED] = 1.0,
        Tile::Weed => values[TILE_WEED] = 1.0,
        Tile::Structure(structure) => {
            values[match structure {
                Structure::Coop => TILE_COOP,
                Structure::Pasture => TILE_PASTURE,
            }] = 1.0;
        }
        Tile::Plant(plant) => {
            values[TILE_PLANT] = 1.0;
            values[CROP_START + plant.crop] = 1.0;
            values[PLANT_AGE] = clipped((day - plant.planted_day) as f64 / 30.0, 0.0, 1.0);
            values[YIELD_UNITS] = log_scaled(plant.yield_units as f64, 8.0);
            values[WATERED] = plant.watered_today as u8 as f32;
            values[CONSECUTIVE_UNWATERED] =
                clipped(plant.consecutive_unwatered as f64 / 2.0, 0.0, 1.0);
            values[FERTILIZER_REMAINING] = clipped(
                (plant.fertilized_until_day - day + 1) as f64 / 3.0,
                0.0,
                1.0,
            );
            if plant.max_lifespan_step >= 0 {
                values[HAS_DECAY_STEP] = 1.0;
                values[TURNS_UNTIL_DECAY] = clipped(
                    (plant.max_lifespan_step - step) as f64 / EPISODE_STEPS,
                    -1.0,
                    1.0,
                );
            }
        }
        Tile::Animal(animal) => {
            values[match ANIMALS[animal.animal].structure {
                Structure::Coop => TILE_COOP,
                Structure::Pasture => TILE_PASTURE,
            }] = 1.0;
            values[ANIMAL_START + animal.animal] = 1.0;
            values[ANIMAL_AGE] = clipped((day - animal.placed_day) as f64 / 30.0, 0.0, 1.0);
            values[YIELD_UNITS] = log_scaled(animal.yield_units as f64, 8.0);
            values[FED] = animal.fed_today as u8 as f32;
            values[CONSECUTIVE_UNFED] = clipped(animal.consecutive_unfed as f64 / 2.0, 0.0, 1.0);
            values[CARED] = animal.cared_today as u8 as f32;
            values[FERTILIZER_AVAILABLE] = animal.fertilizer_available as u8 as f32;
            values[PENDING_CARE_BONUS] = log_scaled(animal.pending_care_bonus as f64, 16.0);
        }
    }
}

fn fill_unit(
    unit_index: usize,
    position: (i64, i64),
    inventory: Option<&Inventory>,
    values: &mut [f32],
) {
    set_base(values, TOKEN_UNIT, FARM_SELF);
    values[if unit_index == 0 {
        UNIT_FARMER
    } else {
        UNIT_HAND
    }] = 1.0;
    values[UNIT_INDEX] = clipped(unit_index as f64 / 31.0, 0.0, 1.0);
    values[UNIT_X] = clipped(position.0 as f64 / 9.0, 0.0, 1.0);
    values[UNIT_Y] = clipped(position.1 as f64 / 9.0, 0.0, 1.0);
    values[UNIT_SHED_ACCESS] = shed_access(position.0, position.1) as u8 as f32;
    if let Some(inventory) = inventory {
        values[INVENTORY_VISIBLE] = 1.0;
        for item in 0..N_ITEMS {
            values[INVENTORY_START + item] = log_scaled(inventory.get(item) as f64, SHED_CAPACITY);
        }
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
    values: &mut [f32],
) {
    set_base(values, TOKEN_PRODUCT, FARM_NONE);
    values[ITEM_START + item] = 1.0;
    values[PRODUCT_SHED_COUNT] =
        log_scaled(engine.privates[player].shed[item] as f64, SHED_CAPACITY);
    if item < N_CROPS {
        values[PRODUCT_SEED_APPLICABLE] = 1.0;
        values[PRODUCT_SEED_COUNT] = log_scaled(engine.privates[player].seeds[item] as f64, 128.0);
    }
    if item < N_PRODUCTS {
        values[PRODUCT_MARKET_VISIBLE] = 1.0;
        values[PRODUCT_MARKET_INVENTORY] =
            (((engine.market.inventory[item] - MARKET_I0) as f64 / 100.0).asinh() / 8.0) as f32;
        values[PRODUCT_MARKET_PRICE] = log_scaled(engine.market.prices[item].f(), 10_000.0);
        values[PRODUCT_TOWN_DEMAND] = demand[item] as f32 / 20.0;
    }
}

fn survival_schedule(
    days: usize,
    consecutive_missed: i64,
    already_done_today: bool,
    forced_days: &[usize],
) -> Vec<bool> {
    let mut schedule = Vec::with_capacity(days);
    let mut missed = consecutive_missed;
    for offset in 0..days {
        if offset == 0 && already_done_today {
            schedule.push(false);
            missed = 0;
            continue;
        }
        let perform = forced_days.contains(&offset) || missed >= 1;
        schedule.push(perform);
        missed = if perform { 0 } else { missed + 1 };
    }
    schedule
}

fn crop_production_offsets(crop: usize, age: i64, horizon: usize) -> Vec<usize> {
    let rule = &CROPS[crop];
    if !rule.ongoing {
        return Vec::new();
    }
    (0..horizon)
        .filter(|&offset| {
            let since_first = age + offset as i64 + 1 - rule.first_yield_day;
            since_first >= 0
                && since_first % rule.interval == 0
                && since_first / rule.interval < rule.max_yield
        })
        .collect()
}

fn plant_capacity(
    crop: usize,
    age: i64,
    current_yield: i64,
    watered_today: bool,
    consecutive_unwatered: i64,
    fertilizer_remaining: i64,
) -> [[f64; 4]; 4] {
    let rule = &CROPS[crop];
    std::array::from_fn(|index| {
        let horizon = CAPACITY_HORIZONS[index];
        if rule.ongoing {
            let offsets = crop_production_offsets(crop, age, horizon);
            if offsets.is_empty() {
                let harvest = (current_yield > 0) as u8 as f64;
                return [current_yield as f64, current_yield as f64, harvest, 0.0];
            }
            let active_fertilizer: Vec<_> = offsets
                .iter()
                .copied()
                .filter(|&offset| offset < fertilizer_remaining.max(0) as usize)
                .collect();
            let days = offsets.last().copied().unwrap_or(0) + 1;
            let nominal_water = survival_schedule(
                days,
                consecutive_unwatered,
                watered_today,
                &active_fertilizer,
            );
            let boosted_water =
                survival_schedule(days, consecutive_unwatered, watered_today, &offsets);
            let mut nominal = current_yield as f64;
            let mut boosted = current_yield as f64;
            for &offset in &offsets {
                nominal += if offset < fertilizer_remaining.max(0) as usize && nominal_water[offset]
                {
                    2.0
                } else {
                    1.0
                };
                boosted += 2.0;
            }
            let nominal_harvests = if nominal > 0.0 {
                (nominal / rule.max_yield as f64).ceil()
            } else {
                0.0
            };
            let boosted_harvests = if boosted > 0.0 {
                (boosted / rule.max_yield as f64).ceil()
            } else {
                0.0
            };
            let mut fertilizer_actions = 0.0;
            let mut covered_until = fertilizer_remaining - 1;
            for &offset in &offsets {
                if offset as i64 > covered_until {
                    fertilizer_actions += 1.0;
                    covered_until = offset as i64 + 2;
                }
            }
            return [
                nominal,
                boosted,
                nominal_water.iter().filter(|&&value| value).count() as f64 + nominal_harvests,
                boosted_water.iter().filter(|&&value| value).count() as f64
                    - nominal_water.iter().filter(|&&value| value).count() as f64
                    + fertilizer_actions
                    + boosted_harvests
                    - nominal_harvests,
            ];
        }

        let latest = horizon as i64 - 1;
        if age + latest < rule.first_yield_day {
            return [0.0; 4];
        }
        let harvest_offset = 0.max(latest.min(rule.max_yield_day - age)) as usize;
        let bonus_start = (rule.max_yield_day + 1) / 2;
        let bonus_days: Vec<_> = (0..=harvest_offset)
            .filter(|&offset| {
                let crop_age = age + offset as i64;
                crop_age >= bonus_start
                    && crop_age <= rule.max_yield_day
                    && !(offset == 0 && watered_today)
            })
            .collect();
        if harvest_offset == 0 && bonus_days.is_empty() {
            return [current_yield as f64, current_yield as f64, 1.0, 0.0];
        }
        let water = survival_schedule(
            harvest_offset + 1,
            consecutive_unwatered,
            watered_today,
            &bonus_days,
        );
        let mut nominal = current_yield as f64;
        let mut boosted = current_yield as f64;
        let mut fertilizer_actions = 0.0;
        let mut covered_until = fertilizer_remaining - 1;
        for &offset in &bonus_days {
            nominal += if offset < fertilizer_remaining.max(0) as usize {
                2.0
            } else {
                1.0
            };
            boosted += 2.0;
            if offset as i64 > covered_until {
                fertilizer_actions += 1.0;
                covered_until = offset as i64 + 2;
            }
        }
        [
            nominal.min(rule.max_yield as f64),
            boosted.min(rule.max_yield as f64),
            water.iter().filter(|&&value| value).count() as f64 + 1.0,
            fertilizer_actions,
        ]
    })
}

fn animal_production_offsets(animal: usize, age: i64, horizon: usize) -> Vec<usize> {
    let rule = &ANIMALS[animal];
    (0..horizon)
        .filter(|&offset| {
            let production_age = age + offset as i64 + 1;
            production_age >= rule.first_yield_day
                && (production_age - rule.first_yield_day) % rule.interval == 0
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn animal_capacity(
    animal: usize,
    age: i64,
    current_yield: i64,
    fed_today: bool,
    consecutive_unfed: i64,
    cared_today: bool,
    pending_care_bonus: i64,
    fertilizer_available: bool,
) -> [[f64; 8]; 4] {
    let rule = &ANIMALS[animal];
    std::array::from_fn(|index| {
        let horizon = CAPACITY_HORIZONS[index];
        let offsets = animal_production_offsets(animal, age, horizon);
        let (nominal, boosted, nominal_actions, boosted_extra_actions) = if offsets.is_empty() {
            let harvest = (current_yield > 0) as u8 as f64;
            (current_yield as f64, current_yield as f64, harvest, 0.0)
        } else {
            let last_day = *offsets.last().unwrap();
            let forced = if pending_care_bonus != 0 {
                vec![offsets[0]]
            } else {
                Vec::new()
            };
            let nominal_feed =
                survival_schedule(last_day + 1, consecutive_unfed, fed_today, &forced);
            let mut nominal = current_yield as f64;
            let mut pending = pending_care_bonus;
            for (offset, &nominally_fed) in nominal_feed.iter().enumerate().take(last_day + 1) {
                if offsets.contains(&offset) {
                    nominal += 1.0 + if nominally_fed { pending as f64 } else { 0.0 };
                    pending = 0;
                }
                if offset == 0 && cared_today && (fed_today || nominally_fed) {
                    pending += 1;
                }
            }
            let mut boosted = current_yield as f64;
            let mut boosted_pending = pending_care_bonus;
            for offset in 0..horizon {
                if offsets.contains(&offset) {
                    boosted += 1.0 + boosted_pending as f64;
                    boosted_pending = 0;
                }
                boosted_pending += 1;
            }
            let nominal_harvests = if nominal > 0.0 {
                (nominal / rule.max_held as f64).ceil()
            } else {
                0.0
            };
            let boosted_harvests = if boosted > 0.0 {
                (boosted / rule.max_held as f64).ceil()
            } else {
                0.0
            };
            let nominal_feed_actions = nominal_feed.iter().filter(|&&value| value).count() as f64;
            let care_actions = horizon as f64 - cared_today as u8 as f64;
            let feed_actions =
                (horizon as f64 - fed_today as u8 as f64 - nominal_feed_actions).max(0.0);
            (
                nominal,
                boosted,
                nominal_feed_actions + nominal_harvests,
                care_actions + feed_actions + boosted_harvests - nominal_harvests,
            )
        };
        let fertilizer_feed = survival_schedule(horizon, consecutive_unfed, fed_today, &[]);
        let fertilizer_capacity = fertilizer_available as u8 as f64 + horizon as f64;
        let fertilizer_actions = fertilizer_feed.iter().filter(|&&value| value).count() as f64
            + fertilizer_available as u8 as f64
            + horizon.saturating_sub(1) as f64;
        [
            nominal,
            boosted,
            nominal_actions,
            boosted_extra_actions,
            fertilizer_capacity,
            fertilizer_capacity,
            fertilizer_actions,
            0.0,
        ]
    })
}

fn nearest_action_steps(positions: &[(i64, i64)], x: usize, y: usize) -> usize {
    1 + positions
        .iter()
        .map(|&(unit_x, unit_y)| {
            (unit_x - x as i64).unsigned_abs() as usize
                + (unit_y - y as i64).unsigned_abs() as usize
        })
        .min()
        .unwrap_or(0)
}

fn ready_bucket(value: usize) -> usize {
    match value {
        1 => 0,
        2 => 1,
        3..=4 => 2,
        5..=8 => 3,
        9..=16 => 4,
        17..=19 => 5,
        _ => unreachable!("board distance must fit the ready-action buckets"),
    }
}

fn shed_bucket(x: usize, y: usize) -> usize {
    let distance = [(4, 4), (5, 4), (4, 5), (5, 5)]
        .into_iter()
        .map(|(shed_x, shed_y)| x.abs_diff(shed_x) + y.abs_diff(shed_y))
        .min()
        .unwrap();
    match distance {
        0..=2 => 0,
        3..=5 => 1,
        6..=8 => 2,
        _ => unreachable!("board distance must fit the shed buckets"),
    }
}

fn add_ready(
    summary: &mut ProductSummary,
    quantity: f64,
    x: usize,
    y: usize,
    positions: &[(i64, i64)],
    turns_remaining: usize,
) {
    let steps = nearest_action_steps(positions, x, y);
    let bucket = ready_bucket(steps);
    summary.ready_units[bucket] += quantity;
    summary.ready_cells[bucket] += 1.0;
    if steps <= turns_remaining {
        summary.ready_units_day_end += quantity;
        summary.ready_cells_day_end += 1.0;
    }
}

fn add_capacity(summary: &mut ProductSummary, capacity: &[[f64; 4]; 4], x: usize, y: usize) {
    for (index, values) in capacity.iter().enumerate() {
        summary.nominal_capacity[index] += values[0];
        summary.boosted_capacity[index] += values[1];
        summary.nominal_actions[index] += values[2];
        summary.boosted_extra_actions[index] += values[3];
    }
    if capacity.iter().any(|values| values[1] > 0.0) {
        summary.productive_cells[shed_bucket(x, y)] += 1.0;
    }
}

fn fill_opponent_summary(engine: &Engine, player: usize, features: &mut [f32], cursor: &mut usize) {
    let opponent = 1 - player;
    let farm = &engine.farms[opponent];
    let day = engine.step_no / 24;
    let hour = engine.step_no % 24;
    let turns_remaining = (24 - hour) as usize;
    let positions: Vec<_> = std::iter::once(farm.farmer)
        .chain(farm.hands.iter().copied())
        .collect();
    let mut summaries = [ProductSummary::default(); N_PRODUCTS];
    let mut crop_counts = [0usize; N_CROPS];
    let mut animal_counts = [0usize; N_ANIMALS];
    let mut weed_count = 0usize;
    let mut empty_crop_cells = 0usize;
    let mut empty_coops = 0usize;
    let mut empty_pastures = 0usize;
    let mut critical_count = 0usize;
    let mut reachable_critical_count = 0usize;

    for y in 0..BOARD_SIZE {
        for x in 0..BOARD_SIZE {
            match &farm.tiles[y][x] {
                Tile::Empty => empty_crop_cells += 1,
                Tile::Locked => {}
                Tile::Weed => weed_count += 1,
                Tile::Structure(Structure::Coop) => empty_coops += 1,
                Tile::Structure(Structure::Pasture) => empty_pastures += 1,
                Tile::Plant(plant) => {
                    crop_counts[plant.crop] += 1;
                    let age = day - plant.planted_day;
                    let current_yield = plant.yield_units.max(0);
                    if age >= CROPS[plant.crop].first_yield_day && current_yield > 0 {
                        add_ready(
                            &mut summaries[plant.crop],
                            current_yield as f64,
                            x,
                            y,
                            &positions,
                            turns_remaining,
                        );
                    }
                    if !plant.watered_today && plant.consecutive_unwatered.max(0) >= 1 {
                        critical_count += 1;
                        if nearest_action_steps(&positions, x, y) <= turns_remaining {
                            reachable_critical_count += 1;
                        }
                    }
                    let fertilizer_remaining = (plant.fertilized_until_day - day + 1).max(0);
                    add_capacity(
                        &mut summaries[plant.crop],
                        &plant_capacity(
                            plant.crop,
                            age,
                            current_yield,
                            plant.watered_today,
                            plant.consecutive_unwatered.max(0),
                            fertilizer_remaining,
                        ),
                        x,
                        y,
                    );
                }
                Tile::Animal(animal) => {
                    animal_counts[animal.animal] += 1;
                    let product = ANIMALS[animal.animal].product;
                    let current_yield = animal.yield_units.max(0);
                    if current_yield > 0 {
                        add_ready(
                            &mut summaries[product],
                            current_yield as f64,
                            x,
                            y,
                            &positions,
                            turns_remaining,
                        );
                    }
                    if animal.fertilizer_available {
                        add_ready(
                            &mut summaries[FERTILIZER],
                            1.0,
                            x,
                            y,
                            &positions,
                            turns_remaining,
                        );
                    }
                    if !animal.fed_today && animal.consecutive_unfed.max(0) >= 1 {
                        critical_count += 1;
                        if nearest_action_steps(&positions, x, y) <= turns_remaining {
                            reachable_critical_count += 1;
                        }
                    }
                    let envelopes = animal_capacity(
                        animal.animal,
                        day - animal.placed_day,
                        current_yield,
                        animal.fed_today,
                        animal.consecutive_unfed.max(0),
                        animal.cared_today,
                        animal.pending_care_bonus.max(0),
                        animal.fertilizer_available,
                    );
                    let product_capacity =
                        envelopes.map(|values| [values[0], values[1], values[2], values[3]]);
                    let fertilizer_capacity =
                        envelopes.map(|values| [values[4], values[5], values[6], values[7]]);
                    add_capacity(&mut summaries[product], &product_capacity, x, y);
                    add_capacity(&mut summaries[FERTILIZER], &fertilizer_capacity, x, y);
                }
            }
        }
    }

    let capacity = row(features, *cursor);
    set_base(capacity, TOKEN_OPPONENT_CAPACITY, FARM_OPPONENT);
    for (crop, &count) in crop_counts.iter().enumerate() {
        capacity[OPPONENT_CROP_COUNT_START + crop] = count as f32 / 100.0;
    }
    for (animal, &count) in animal_counts.iter().enumerate() {
        capacity[OPPONENT_ANIMAL_COUNT_START + animal] = count as f32 / 100.0;
    }
    capacity[OPPONENT_WEED_COUNT] = weed_count as f32 / 100.0;
    capacity[OPPONENT_EMPTY_CROP_COUNT] = empty_crop_cells as f32 / 100.0;
    capacity[OPPONENT_EMPTY_COOP_COUNT] = empty_coops as f32 / 100.0;
    capacity[OPPONENT_EMPTY_PASTURE_COUNT] = empty_pastures as f32 / 100.0;
    capacity[OPPONENT_CURRENT_UNIT_COUNT] = log_scaled(positions.len() as f64, 32.0);
    capacity[OPPONENT_REMAINING_ACTION_SLOTS] = clipped(
        positions.len() as f64 * turns_remaining as f64 / (32.0 * 24.0),
        0.0,
        1.0,
    );
    capacity[OPPONENT_CRITICAL_COUNT] = critical_count as f32 / 100.0;
    capacity[OPPONENT_REACHABLE_CRITICAL_COUNT] = reachable_critical_count as f32 / 100.0;
    *cursor += 1;

    for (item, summary) in summaries.iter().enumerate() {
        let product = row(features, *cursor);
        set_base(product, TOKEN_OPPONENT_PRODUCT, FARM_OPPONENT);
        product[ITEM_START + item] = 1.0;
        for bucket in 0..6 {
            product[READY_UNITS_START + bucket] = log_scaled(summary.ready_units[bucket], 600.0);
            product[READY_CELLS_START + bucket] = summary.ready_cells[bucket] as f32 / 100.0;
        }
        product[READY_UNITS_DAY_END] = log_scaled(summary.ready_units_day_end, 600.0);
        product[READY_CELLS_DAY_END] = summary.ready_cells_day_end as f32 / 100.0;
        for horizon in 0..4 {
            product[NOMINAL_CAPACITY_START + horizon] =
                log_scaled(summary.nominal_capacity[horizon], 2_800.0);
            product[BOOSTED_CAPACITY_START + horizon] =
                log_scaled(summary.boosted_capacity[horizon], 2_800.0);
            product[NOMINAL_ACTIONS_START + horizon] =
                log_scaled(summary.nominal_actions[horizon], 4_200.0);
            product[BOOSTED_EXTRA_ACTIONS_START + horizon] =
                log_scaled(summary.boosted_extra_actions[horizon], 4_200.0);
        }
        for bucket in 0..3 {
            product[PRODUCTIVE_CELLS_START + bucket] =
                summary.productive_cells[bucket] as f32 / 100.0;
        }
        *cursor += 1;
    }
}

pub fn encode_features_into(
    engine: &Engine,
    player: usize,
    features: &mut [f32],
) -> Result<FeatureCounts, &'static str> {
    if player > 1
        || engine.cfg.board_size != BOARD_SIZE as i64
        || features.len() != FIXED_TOKENS * FEATURE_DIM
    {
        return Err("exp28 encoder requires player 0/1, a 10x10 board, and the fixed output shape");
    }
    features.fill(0.0);
    let actual_own_units = 1 + engine.farms[player].hands.len();
    let encoded_own_units = actual_own_units.min(MAX_OWN_UNITS);
    fill_global(engine, player, row(features, 0));

    let farm = &engine.farms[player];
    let day = engine.step_no / 24;
    let mut farmer_counts = [0usize; BOARD_SIZE * BOARD_SIZE];
    let mut hand_counts = [0usize; BOARD_SIZE * BOARD_SIZE];
    farmer_counts[farm.farmer.1 as usize * BOARD_SIZE + farm.farmer.0 as usize] += 1;
    for &(x, y) in &farm.hands {
        hand_counts[y as usize * BOARD_SIZE + x as usize] += 1;
    }
    for y in 0..BOARD_SIZE {
        for x in 0..BOARD_SIZE {
            let token = 1 + y * BOARD_SIZE + x;
            fill_cell(
                &farm.tiles[y][x],
                (day, engine.step_no),
                (x, y),
                (
                    farmer_counts[y * BOARD_SIZE + x],
                    hand_counts[y * BOARD_SIZE + x],
                ),
                row(features, token),
            );
        }
    }

    let mut cursor = 1 + BOARD_SIZE * BOARD_SIZE;
    let positions = std::iter::once(farm.farmer).chain(farm.hands.iter().copied());
    for (index, position) in positions.take(encoded_own_units).enumerate() {
        fill_unit(
            index,
            position,
            engine.privates[player].inventories.get(index),
            row(features, cursor),
        );
        cursor += 1;
    }
    fill_opponent_summary(engine, player, features, &mut cursor);
    let demand = town_demand(engine);
    for item in 0..N_ITEMS {
        fill_product(engine, player, item, &demand, row(features, cursor));
        cursor += 1;
    }
    set_base(row(features, cursor), TOKEN_MEMORY, FARM_NONE);
    cursor += 1;
    for slot in 0..MARKET_SLOTS {
        let values = row(features, cursor);
        set_base(values, TOKEN_MARKET_SLOT, FARM_NONE);
        values[MARKET_SLOT_START + slot] = 1.0;
        cursor += 1;
    }
    debug_assert_eq!(cursor, 134 + encoded_own_units);
    Ok(FeatureCounts {
        actual_own_units,
        encoded_own_units,
    })
}

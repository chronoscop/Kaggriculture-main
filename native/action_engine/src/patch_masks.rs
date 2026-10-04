//! Decode-rule support only: market feasibility is deliberately not constrained.

use crate::catalog::MAX_OWN_UNITS;
use crate::engine::Engine;
use crate::state::*;

pub const CONTEXT_SIZE: usize = MAX_OWN_UNITS * 2 + N_CROPS;
pub const STEMS: usize = 44;
pub const MASK_SIZE: usize = MAX_OWN_UNITS * STEMS + 22;

pub fn fertilize_gains(plant: &Plant, day: i64, last_day: i64) -> bool {
    let crop = &CROPS[plant.crop];
    (day..day + 3).any(|covered_day| {
        if covered_day <= plant.fertilized_until_day || covered_day > last_day {
            return false;
        }
        if !crop.ongoing {
            let age = covered_day - plant.planted_day;
            return age >= (crop.max_yield_day + 1) / 2
                && age <= crop.max_yield_day
                && !(covered_day == day && plant.watered_today)
                && plant.yield_units < crop.max_yield - 1;
        }
        let since = covered_day + 1 - plant.planted_day - crop.first_yield_day;
        since >= 0
            && since % crop.interval == 0
            && since / crop.interval + 1 <= crop.max_yield
            && covered_day < last_day
    })
}

pub fn care_gains(tile: &Tile, day: i64, last_day: i64) -> bool {
    match tile {
        Tile::Structure(_) => true,
        Tile::Animal(animal) if !animal.cared_today => {
            let data = &ANIMALS[animal.animal];
            (day + 1..last_day).any(|night| {
                let since = night + 1 - animal.placed_day - data.first_yield_day;
                since >= 0 && since % data.interval == 0
            })
        }
        _ => false,
    }
}

/// Catalog order is frozen; quantity variants share their operation/item stem.
pub fn action_stem(action: usize) -> usize {
    match action {
        0..=4 => action,
        5..=244 => 5 + (action - 5) / 20,
        245 => 17,
        246..=485 => 18 + (action - 246) / 20,
        486..=499 => 30 + action - 486,
        _ => panic!("unit action out of catalog"),
    }
}

pub fn write_patch_inputs(
    engine: &Engine,
    player: usize,
    masks: &mut [bool],
    context: &mut [i32],
) -> Result<(), &'static str> {
    if player >= 2 || masks.len() != MASK_SIZE || context.len() != CONTEXT_SIZE {
        return Err("patch inputs require player0/1, mask[902], context[45]");
    }
    masks.fill(true);
    context.fill(0);
    context[..MAX_OWN_UNITS].fill(-1);
    let farm = &engine.farms[player];
    let day = engine.step_no / engine.cfg.turns_per_day;
    let last_day = (engine.cfg.episode_steps - 1) / engine.cfg.turns_per_day;
    let empty = Inventory::default();
    for (unit, (x, y)) in std::iter::once(farm.farmer)
        .chain(farm.hands.iter().copied())
        .take(MAX_OWN_UNITS)
        .enumerate()
    {
        let tile = &farm.tiles[y as usize][x as usize];
        context[unit] = (y * engine.cfg.board_size + x) as i32;
        context[MAX_OWN_UNITS + unit] = match tile {
            Tile::Structure(Structure::Coop) => 1,
            Tile::Structure(Structure::Pasture) => 2,
            _ => 0,
        };
        let row = &mut masks[unit * STEMS..(unit + 1) * STEMS];
        for (action, (dx, dy)) in [(0, -1), (0, 1), (1, 0), (-1, 0)].into_iter().enumerate() {
            row[action] = (0..engine.cfg.board_size).contains(&(x + dx))
                && (0..engine.cfg.board_size).contains(&(y + dy));
        }
        let held = engine.privates[player]
            .inventories
            .get(unit)
            .unwrap_or(&empty)
            .get(FERTILIZER);
        row[action_stem(493)] =
            held > 0 && matches!(tile, Tile::Plant(p) if fertilize_gains(p, day, last_day));
        row[action_stem(499)] = care_gains(tile, day, last_day);
    }
    for crop in 0..N_CROPS {
        context[2 * MAX_OWN_UNITS + crop] =
            engine.privates[player].seeds[crop].clamp(0, i32::MAX as i64) as i32;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::UNIT_ACTION_COUNT;

    #[test]
    fn catalog_is_losslessly_partitioned() {
        assert_eq!(action_stem(UNIT_ACTION_COUNT as usize - 1), STEMS - 1);
        for action in 1..UNIT_ACTION_COUNT as usize {
            assert!(action_stem(action) >= action_stem(action - 1));
        }
    }

    #[test]
    fn fertilizer_caps_and_covered_days() {
        let mut plant = Plant::new(0, 0, 24);
        assert!(fertilize_gains(&plant, 2, 29));
        plant.yield_units = 5;
        assert!(!fertilize_gains(&plant, 2, 29));
        plant.yield_units = 1;
        plant.fertilized_until_day = 4;
        assert!(!fertilize_gains(&plant, 2, 29));
    }
}

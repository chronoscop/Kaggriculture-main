//! Shed rules applied to decoded absolute-quantity actions; no automatic seed buying.

use crate::catalog::{ABSOLUTE_SELL_START, DROP_ID, MARKET_SLOTS, MAX_OWN_UNITS};
use crate::engine::{Engine, is_shed_adjacent, market_price};
use crate::state::*;

fn project(engine: &Engine, player: usize, units: &[i32]) -> ([i64; N_ITEMS], [i64; N_ITEMS]) {
    let farm = &engine.farms[player];
    let private = &engine.privates[player];
    let mut shed = private.shed;
    let mut pockets = [0; N_ITEMS];
    for (index, (x, y)) in std::iter::once(farm.farmer)
        .chain(farm.hands.iter().copied())
        .take(MAX_OWN_UNITS)
        .enumerate()
    {
        let mut held = [0; N_ITEMS];
        if let Some(inventory) = private.inventories.get(index) {
            for &(item, quantity) in &inventory.0 {
                held[item] = quantity;
            }
        }
        let action = units[index];
        let tile = &farm.tiles[y as usize][x as usize];
        let adjacent = is_shed_adjacent(x, y, engine.cfg.board_size);
        if action == DROP_ID && adjacent {
            // The official inventory retains insertion order; the patch projects in that order.
            if let Some(inventory) = private.inventories.get(index) {
                for &(item, quantity) in &inventory.0 {
                    let moved =
                        quantity.min((engine.cfg.shed_capacity - shed.iter().sum::<i64>()).max(0));
                    shed[item] += moved;
                }
            }
            held.fill(0);
        } else if (5..245).contains(&action) && adjacent {
            let item = ((action - 5) / 20) as usize;
            let wanted = ((action - 5) % 20 + 1) as i64;
            let moved = wanted.min(shed[item]);
            shed[item] -= moved;
            held[item] += moved;
        } else if (246..486).contains(&action) {
            let item = ((action - 246) / 20) as usize;
            let wanted = ((action - 246) % 20 + 1) as i64;
            let pen = item >= FIRST_ANIMAL
                && matches!(tile, Tile::Structure(s) if *s == ANIMALS[item-FIRST_ANIMAL].structure);
            if pen {
                held[item] = (held[item] - 1).max(0);
            } else if adjacent {
                let moved = wanted
                    .min(held[item])
                    .min((engine.cfg.shed_capacity - shed.iter().sum::<i64>()).max(0));
                shed[item] += moved;
                held[item] -= moved;
            }
        } else {
            match (action, tile) {
                (492, Tile::Plant(p)) => held[p.crop] += p.yield_units.max(0),
                (492, Tile::Animal(a)) => held[ANIMALS[a.animal].product] += a.yield_units.max(0),
                (498, Tile::Animal(a)) if a.fertilizer_available => held[FERTILIZER] += 1,
                (497, Tile::Animal(_)) => held[WHEAT] = (held[WHEAT] - 1).max(0),
                (493, Tile::Plant(_)) => held[FERTILIZER] = (held[FERTILIZER] - 1).max(0),
                _ => (),
            }
        }
        for item in 0..N_ITEMS {
            pockets[item] += held[item].max(0);
        }
    }
    (shed, pockets)
}

fn sell(action: i32) -> Option<(usize, i64)> {
    let value = action - ABSOLUTE_SELL_START;
    (0..N_PRODUCTS as i32 * 100)
        .contains(&value)
        .then_some(((value / 100) as usize, (value % 100 + 1) as i64))
}

pub fn patch_action_ids(
    engine: &Engine,
    player: usize,
    units: &mut [i32],
    market: &mut [i32],
    unit_valid: &mut [bool],
    market_valid: &mut [bool],
) -> Result<(), &'static str> {
    if player >= 2
        || units.len() != MAX_OWN_UNITS
        || market.len() != MARKET_SLOTS
        || unit_valid.len() != MAX_OWN_UNITS
        || market_valid.len() != MARKET_SLOTS
    {
        return Err("invalid patch action buffer shape");
    }
    unit_valid.fill(true);
    market_valid.fill(true);
    let step = engine.step_no;
    let final_step = engine.cfg.episode_steps - 2;
    let final_sale = step == final_step || step == final_step - 1;
    let night =
        step % engine.cfg.turns_per_day == engine.cfg.turns_per_day - 1 && step < final_step - 1;
    if !night && !final_sale {
        return Ok(());
    }
    // Python decodes/clamps sales before applying the shed patch. Reproduce that
    // ordering before a final DROP changes the stock available to those orders.
    let mut before_slots = [0; MARKET_SLOTS * N_PRODUCTS];
    crate::catalog::sell_stocks_before_slots(engine, player, units, market, &mut before_slots)?;
    for (slot, action) in market.iter_mut().enumerate() {
        if let Some((item, quantity)) = sell(*action) {
            let actual = quantity.min(before_slots[slot * N_PRODUCTS + item] as i64);
            *action = if actual > 0 {
                ABSOLUTE_SELL_START + item as i32 * 100 + actual as i32 - 1
            } else {
                0
            };
        }
    }
    let farm = &engine.farms[player];
    if step == final_step {
        for (unit, (x, y)) in std::iter::once(farm.farmer)
            .chain(farm.hands.iter().copied())
            .take(MAX_OWN_UNITS)
            .enumerate()
        {
            let carrying = engine.privates[player]
                .inventories
                .get(unit)
                .is_some_and(|v| v.0.iter().map(|v| v.1).sum::<i64>() > 0);
            if carrying && is_shed_adjacent(x, y, engine.cfg.board_size) {
                units[unit] = DROP_ID;
                unit_valid[unit] = false;
            }
        }
    }
    if final_sale {
        for (slot, action) in market.iter_mut().enumerate() {
            if *action != 0 && sell(*action).is_none() {
                *action = 0;
                market_valid[slot] = false;
            }
        }
    }
    let (shed, pockets) = project(engine, player, units);
    let mut sold = [0; N_PRODUCTS];
    for &action in market.iter() {
        if let Some((item, quantity)) = sell(action) {
            sold[item] += quantity;
        }
    }
    let mut stock = [0; N_PRODUCTS];
    for item in 0..N_PRODUCTS {
        stock[item] = (shed[item] - sold[item]).max(0);
    }
    let mut extra = [0; N_PRODUCTS];
    if night {
        let excess = shed.iter().sum::<i64>() - sold.iter().sum::<i64>()
            + pockets.iter().sum::<i64>()
            - engine.cfg.shed_capacity;
        let animals = farm
            .tiles
            .iter()
            .flatten()
            .filter(|t| matches!(t, Tile::Animal(_)))
            .count() as i64;
        stock[WHEAT] = (stock[WHEAT] - (animals - pockets[WHEAT]).max(0)).max(0);
        for _ in 0..excess.max(0) {
            let candidate = (0..N_PRODUCTS)
                .filter(|&i| stock[i] > extra[i])
                .max_by(|&a, &b| {
                    let pa = market_price(
                        a,
                        engine.market.inventory[a] + extra[a],
                        &engine.market.params,
                    );
                    let pb = market_price(
                        b,
                        engine.market.inventory[b] + extra[b],
                        &engine.market.params,
                    );
                    let base = [25, 35, 60, 120, 250, 50, 160, 200, 100];
                    (pa * base[b])
                        .cmp(&(pb * base[a]))
                        .then(pa.cmp(&pb))
                        .then(ITEM_NAMES[b].cmp(ITEM_NAMES[a]))
                });
            if let Some(item) = candidate {
                extra[item] += 1;
            } else {
                break;
            }
        }
    } else {
        extra = stock;
    }
    let mut priority: Vec<usize> = (0..N_PRODUCTS).filter(|&i| extra[i] > 0).collect();
    priority.sort_by_key(|&i| {
        -market_price(i, engine.market.inventory[i], &engine.market.params) * extra[i]
    });
    let mut added = [false; MARKET_SLOTS];
    for item in priority {
        let existing = market
            .iter()
            .position(|&a| sell(a).is_some_and(|(i, _)| i == item));
        if let Some(slot) = existing.or_else(|| market.iter().position(|&a| a == 0)) {
            added[slot] = existing.is_none();
            let quantity = (existing.and_then(|s| sell(market[s])).map_or(0, |(_, q)| q)
                + extra[item])
                .min(100);
            market[slot] = ABSOLUTE_SELL_START + item as i32 * 100 + quantity as i32 - 1;
            market_valid[slot] = false;
        }
    }
    // Preserve append order: filling an earlier NOOP can change buying power.
    let mut ordered = [0; MARKET_SLOTS];
    let mut cursor = 0;
    for appended in [false, true] {
        for slot in 0..MARKET_SLOTS {
            if market[slot] != 0 && added[slot] == appended {
                ordered[cursor] = market[slot];
                cursor += 1;
            }
        }
    }
    market.copy_from_slice(&ordered);
    Ok(())
}

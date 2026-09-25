//! The step function -- a line-by-line port of the interpreter's mutation
//! path: `_apply_unit_action`, `_process_market`, `_town_consume`,
//! `_decay_plants`, `_end_of_day` and everything they call.
//!
//! Fidelity is proven, not assumed: the differential suite compares a full
//! state digest against the official interpreter after every step.
//!
//! Porting discipline: statement order is preserved even where it looks
//! reorderable, because several behaviours hide in the order --
//!   * player 0's WHOLE unit-action list applies before player 1's;
//!   * the market runs one ORDER INDEX at a time, and within an index one
//!     UNIT at a time, both players quoted at the same pre-commit inventory;
//!   * prices refresh after each order index, not after each unit;
//!   * at end of day, farm 0 is fully refreshed (plants, animals, weeds,
//!     shed sweep) before farm 1 touches the shared RNG stream.

use crate::market;
use crate::mt19937::MT;
use crate::rules;
use crate::state::{
    default_spawn, AnimalTile, Cell, Farm, OMap, Private, State, BOARD, EPISODE_STEPS,
    MAX_MARKET_ORDERS, PRODUCTS, SHED_CAP, TOWN_CENTER_SELL_INTERVAL, TOWN_SHOP_SELL_INTERVAL,
    TOWN_SHOP_UNLOCK_INTERVAL, TURNS_PER_DAY, WEED_CHANCE,
};

/// `sorted(SHOPS)` -- `rng.choice` indexes this, so order AND length are part
/// of the RNG contract (an earlier 6-entry copy validated against itself).
pub const SHOPS_SORTED: [&str; 8] = [
    "BAKERY",
    "BRUNCH_SPOT",
    "FARMERS_MARKET",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "PIZZA_SHOP",
    "SMOOTHIE_SHOP",
    "YARN_STORE",
];

/// One shop's product list, keyed by SHOPS_SORTED order.
pub fn shop_products(shop: &str) -> &'static [&'static str] {
    match shop {
        "BAKERY" => &["EGG", "WHEAT"],
        "PIZZA_SHOP" => &["MILK", "TOMATO", "WHEAT"],
        "BRUNCH_SPOT" => &["EGG", "WHEAT", "STRAWBERRY"],
        "YARN_STORE" => &["WOOL"],
        "ICE_CREAM_SHOP" => &["STRAWBERRY", "MILK", "WHEAT"],
        "PET_CAFE" => &["CARROT"],
        "SMOOTHIE_SHOP" => &["STRAWBERRY", "MILK"],
        "FARMERS_MARKET" => &["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
        _ => &[],
    }
}

/// A unit action: op token plus optional item / count, e.g. `PLANT WHEAT`,
/// `PICKUP WHEAT 3`, `NORTH`.
#[derive(Clone, Debug, Default)]
pub struct UnitAction {
    pub op: String,
    pub item: String,
    pub n: i64,
    pub has_n: bool,
}

impl UnitAction {
    pub fn parse(tokens: &[&str]) -> Self {
        UnitAction {
            op: tokens.first().unwrap_or(&"PASS").to_string(),
            item: tokens.get(1).unwrap_or(&"").to_string(),
            n: tokens.get(2).and_then(|t| t.parse().ok()).unwrap_or(1),
            has_n: tokens.len() >= 3 && tokens[2].parse::<i64>().is_ok(),
        }
    }
}

/// Split a `;`-joined hands/market field POSITIONALLY. An empty segment is an
/// empty action (`[]`) and must keep its slot: the official interpreter
/// enumerates `hands` by position (hand i gets hands[i]) and settles `market`
/// per ORDER INDEX in lockstep, so dropping an interior `[]` hands a later
/// hand's action to the wrong worker and changes which orders settle at the
/// same index. A wholly empty field is zero entries.
pub fn positional(field: &str) -> Vec<&str> {
    if field.is_empty() {
        Vec::new()
    } else {
        field.split(';').collect()
    }
}

/// One player's turn: farmer + hands + market queue.
#[derive(Clone, Debug, Default)]
pub struct PlayerAction {
    pub farmer: UnitAction,
    pub hands: Vec<UnitAction>,
    pub market: Vec<Vec<String>>,
}

fn move_delta(op: &str) -> Option<(i64, i64)> {
    match op {
        "NORTH" => Some((0, -1)),
        "SOUTH" => Some((0, 1)),
        "EAST" => Some((1, 0)),
        "WEST" => Some((-1, 0)),
        _ => None,
    }
}

fn farmer_position(farm: &Farm, idx: usize) -> Option<(i64, i64)> {
    if idx == 0 {
        Some(farm.farmer)
    } else {
        farm.hands.get(idx - 1).copied()
    }
}

fn set_farmer_position(farm: &mut Farm, idx: usize, pos: (i64, i64)) {
    if idx == 0 {
        farm.farmer = pos;
    } else {
        farm.hands[idx - 1] = pos;
    }
}

/// `_farmer_inventory` grows the list if idx is past the end.
fn farmer_inventory(private: &mut Private, idx: usize) -> &mut OMap {
    while private.inventories.len() <= idx {
        private.inventories.push(OMap::default());
    }
    &mut private.inventories[idx]
}

/// `_apply_unit_action`. Invalid / illegal actions are silent no-ops -- the
/// single most important property to preserve, because on the Python side
/// bugs in agents look exactly like laziness.
#[allow(clippy::too_many_lines)]
pub fn apply_unit_action(
    farm: &mut Farm,
    private: &mut Private,
    idx: usize,
    action: &UnitAction,
    day: i64,
) {
    let op = action.op.as_str();
    if op.is_empty() {
        return;
    }
    let Some((fx, fy)) = farmer_position(farm, idx) else {
        return;
    };

    if let Some((dx, dy)) = move_delta(op) {
        let (nx, ny) = (fx + dx, fy + dy);
        if !(0..BOARD).contains(&nx) || !(0..BOARD).contains(&ny) {
            return;
        }
        // Movement onto LOCKED tiles is allowed (a hand can spawn on one).
        set_farmer_position(farm, idx, (nx, ny));
        return;
    }

    if op == "PASS" {
        return;
    }

    // Shed operations resolve before the LOCKED guard: they use the tile only
    // as a standing position, and three of the four shed-access tiles start
    // LOCKED.
    if op == "DROP" {
        if !rules::is_shed_adjacent((fx, fy), BOARD) {
            return;
        }
        let inv = farmer_inventory(private, idx);
        let items: Vec<(String, i64)> = inv.0.clone();
        for (item, n) in items {
            if n <= 0 {
                private.inventories[idx].remove(&item);
                continue;
            }
            let room = (SHED_CAP - private.shed.sum()).max(0);
            let take = n.min(room);
            if take > 0 {
                private.shed.add(&item, take);
            }
            private.inventories[idx].remove(&item);
        }
        return;
    }

    if op == "PICKUP" {
        if !rules::is_shed_adjacent((fx, fy), BOARD) {
            return;
        }
        if action.item.is_empty() {
            return;
        }
        let n = if action.has_n { action.n } else { 1 };
        if n <= 0 {
            return;
        }
        let available = private.shed.get(&action.item);
        let n = n.min(available);
        if n <= 0 {
            return;
        }
        private.shed.sub(&action.item, n);
        farmer_inventory(private, idx).add(&action.item, n);
        return;
    }

    if op == "PLACE" {
        if action.item.is_empty() {
            return;
        }
        let item = action.item.clone();
        // Animal placement: standing on a matching unoccupied structure.
        if let Some(a) = rules::animal(&item) {
            if let Cell::Structure { kind, animal: None } = &farm.tiles[fy as usize][fx as usize] {
                if kind == a.structure {
                    if farmer_inventory(private, idx).take(&item, 1) {
                        farm.tiles[fy as usize][fx as usize] = Cell::Structure {
                            kind: a.structure.to_string(),
                            animal: Some(new_animal(&item, day)),
                        };
                    }
                    return;
                }
            }
        }
        // Shed drop path.
        if rules::is_shed_adjacent((fx, fy), BOARD) {
            let n = if action.has_n { action.n } else { 1 };
            if n <= 0 {
                return;
            }
            let inv_n = farmer_inventory(private, idx).get(&item);
            let n = n.min(inv_n);
            if n <= 0 {
                return;
            }
            let room = (SHED_CAP - private.shed.sum()).max(0);
            let n = n.min(room);
            if n <= 0 {
                return;
            }
            let inv = farmer_inventory(private, idx);
            // `inv[item] -= n; if 0: del` == take()
            inv.take(&item, n);
            private.shed.add(&item, n);
        }
        return;
    }

    // Everything below mutates the tile the unit stands on: owned tiles only.
    if farm.tiles[fy as usize][fx as usize] == Cell::Locked {
        return;
    }

    match op {
        "PLANT" => {
            let Some(_c) = rules::crop(&action.item) else {
                return;
            };
            if farm.tiles[fy as usize][fx as usize] != Cell::Empty {
                return;
            }
            if private.seeds.get(&action.item) <= 0 {
                return;
            }
            private.seeds.sub(&action.item, 1);
            farm.tiles[fy as usize][fx as usize] = new_plant(&action.item, day);
        }
        "WATER" => {
            let tile = &mut farm.tiles[fy as usize][fx as usize];
            let Cell::Plant {
                crop,
                planted_day,
                watered_today,
                yield_units,
                fertilized_until_day,
                ..
            } = tile
            else {
                return;
            };
            if *watered_today {
                return;
            }
            *watered_today = true;
            let cd = rules::crop(crop).unwrap();
            if !cd.ongoing {
                let age_days = day - *planted_day;
                let window_start = (cd.max_yield_day + 1) / 2;
                if window_start <= age_days && age_days <= cd.max_yield_day {
                    let bonus = if *fertilized_until_day >= day { 2 } else { 1 };
                    *yield_units = cd.max_yield.min(*yield_units + bonus);
                }
            }
        }
        "HARVEST" => {
            let tile = &mut farm.tiles[fy as usize][fx as usize];
            match tile {
                Cell::Plant {
                    crop,
                    planted_day,
                    yield_units,
                    ..
                } => {
                    if *yield_units <= 0 {
                        return;
                    }
                    let cd = rules::crop(crop).unwrap();
                    if day - *planted_day < cd.first_yield_day {
                        return;
                    }
                    let units = *yield_units;
                    *yield_units = 0;
                    let crop_name = crop.clone();
                    let ongoing = cd.ongoing;
                    farmer_inventory(private, idx).add(&crop_name, units);
                    if !ongoing {
                        farm.tiles[fy as usize][fx as usize] = Cell::Empty;
                    }
                }
                Cell::Structure {
                    animal: Some(a), ..
                } => {
                    if a.yield_units <= 0 {
                        return;
                    }
                    let units = a.yield_units;
                    a.yield_units = 0;
                    let product = rules::animal(&a.animal).unwrap().product.to_string();
                    farmer_inventory(private, idx).add(&product, units);
                }
                _ => {}
            }
        }
        "FERTILIZE" => {
            {
                let tile = &farm.tiles[fy as usize][fx as usize];
                if !matches!(tile, Cell::Plant { .. }) {
                    return;
                }
            }
            if !farmer_inventory(private, idx).take("FERTILIZER", 1) {
                return;
            }
            if let Cell::Plant {
                fertilized_until_day,
                ..
            } = &mut farm.tiles[fy as usize][fx as usize]
            {
                *fertilized_until_day = (*fertilized_until_day).max(day + 2);
            }
        }
        "DIG" => {
            let tile = &farm.tiles[fy as usize][fx as usize];
            if *tile == Cell::Empty {
                return;
            }
            if matches!(
                tile,
                Cell::Structure {
                    animal: Some(_),
                    ..
                }
            ) {
                return;
            }
            farm.tiles[fy as usize][fx as usize] = Cell::Empty;
        }
        "BUILD_COOP" => {
            if farm.tiles[fy as usize][fx as usize] != Cell::Empty {
                return;
            }
            farm.tiles[fy as usize][fx as usize] = Cell::Structure {
                kind: "COOP".to_string(),
                animal: None,
            };
        }
        "BUILD_PASTURE" => {
            if farm.tiles[fy as usize][fx as usize] != Cell::Empty {
                return;
            }
            farm.tiles[fy as usize][fx as usize] = Cell::Structure {
                kind: "PASTURE".to_string(),
                animal: None,
            };
        }
        "FEED" => {
            {
                let tile = &farm.tiles[fy as usize][fx as usize];
                let Cell::Structure {
                    animal: Some(a), ..
                } = tile
                else {
                    return;
                };
                if a.fed_today {
                    return;
                }
            }
            if !farmer_inventory(private, idx).take("WHEAT", 1) {
                return;
            }
            if let Cell::Structure {
                animal: Some(a), ..
            } = &mut farm.tiles[fy as usize][fx as usize]
            {
                a.fed_today = true;
            }
        }
        "COLLECT_FERTILIZER" => {
            {
                let tile = &mut farm.tiles[fy as usize][fx as usize];
                let Cell::Structure {
                    animal: Some(a), ..
                } = tile
                else {
                    return;
                };
                if !a.fertilizer_available {
                    return;
                }
                a.fertilizer_available = false;
            }
            farmer_inventory(private, idx).add("FERTILIZER", 1);
        }
        "CARE" => {
            let tile = &mut farm.tiles[fy as usize][fx as usize];
            let Cell::Structure {
                animal: Some(a), ..
            } = tile
            else {
                return;
            };
            if a.cared_today {
                return;
            }
            a.cared_today = true;
        }
        _ => {}
    }
}

fn new_plant(crop: &str, day: i64) -> Cell {
    let c = rules::crop(crop).unwrap();
    Cell::Plant {
        crop: crop.to_string(),
        planted_day: day,
        watered_today: false,
        consecutive_unwatered: 1, // planting day counts as unwatered
        yield_units: rules::initial_yield(c),
        max_lifespan_step: rules::max_lifespan_step(c, day, TURNS_PER_DAY),
        fertilized_until_day: -1,
    }
}

fn new_animal(animal: &str, day: i64) -> AnimalTile {
    AnimalTile {
        animal: animal.to_string(),
        placed_day: day,
        yield_units: 0,
        consecutive_unfed: 0,
        fed_today: false,
        cared_today: false,
        fertilizer_available: false,
        pending_care_bonus: 0,
    }
}

// ------------------------------------------------------------------ market --

#[derive(Clone, Debug)]
struct OrderState {
    typ: String,
    item: String,
    remaining: i64,
}

/// `_parse_order`.
fn parse_order(order: &[String]) -> Option<OrderState> {
    let op = order.first()?.as_str();
    if op == "HIRE" || op == "BUY_LAND" {
        return Some(OrderState {
            typ: op.to_string(),
            item: String::new(),
            remaining: 0,
        });
    }
    if matches!(op, "BUY_SEED" | "BUY_PRODUCT" | "BUY_ANIMAL" | "SELL") {
        if order.len() < 3 {
            return None;
        }
        let n: i64 = order[2].parse().ok()?;
        if n <= 0 {
            return None;
        }
        return Some(OrderState {
            typ: op.to_string(),
            item: order[1].clone(),
            remaining: n,
        });
    }
    None
}

/// `_commit_unit`.
fn commit_unit(
    op: &str,
    item: &str,
    price: i64,
    farm: &mut Farm,
    private: &mut Private,
    mkt: &mut crate::state::Market,
) -> bool {
    match op {
        "SELL" => {
            if private.shed.get(item) <= 0 {
                return false;
            }
            private.shed.sub(item, 1);
            farm.money += price as f64;
            // Sales at $1 do not increase market supply.
            if price > 1 {
                mkt.inventory.add(item, 1);
            }
            true
        }
        "BUY_PRODUCT" => {
            if farm.money < price as f64 {
                return false;
            }
            if private.shed.sum() >= SHED_CAP {
                return false;
            }
            farm.money -= price as f64;
            private.shed.add(item, 1);
            mkt.inventory.sub(item, 1);
            true
        }
        "BUY_SEED" => {
            if farm.money < price as f64 {
                return false;
            }
            farm.money -= price as f64;
            private.seeds.add(item, 1);
            true
        }
        "BUY_ANIMAL" => {
            if farm.money < price as f64 {
                return false;
            }
            if private.shed.sum() >= SHED_CAP {
                return false;
            }
            farm.money -= price as f64;
            private.shed.add(item, 1);
            true
        }
        _ => false,
    }
}

/// `_spawn_hand`: first free shed-access tile, ties by min occupancy then
/// NWSE order.
fn spawn_hand(farm: &Farm) -> (i64, i64) {
    let tiles = rules::shed_access_tiles(BOARD);
    let mut occ = [0i64; 4];
    let mut all = vec![farm.farmer];
    all.extend(farm.hands.iter().copied());
    for pos in all {
        if let Some(i) = tiles.iter().position(|t| *t == pos) {
            occ[i] += 1;
        }
    }
    let mut order: Vec<usize> = (0..4).collect();
    order.sort_by_key(|&i| (occ[i], i));
    tiles[order[0]]
}

fn do_hire(farm: &mut Farm, private: &mut Private) {
    let cost = rules::hire_cost(farm.hires_today as u32, rules::FARM_HAND_COST_MULT);
    if farm.money < cost as f64 {
        return;
    }
    farm.money -= cost as f64;
    farm.hires_today += 1;
    let pos = spawn_hand(farm);
    farm.hands.push(pos);
    private.inventories.push(OMap::default());
}

fn do_buy_land(farm: &mut Farm) {
    let n_extra = farm.unlocked_quadrants.len() - 1;
    let Some((quadrant, cost)) = rules::next_land(n_extra) else {
        return;
    };
    if farm.money < cost as f64 {
        return;
    }
    farm.money -= cost as f64;
    farm.unlocked_quadrants.push(quadrant.to_string());
    for y in 0..BOARD {
        for x in 0..BOARD {
            if rules::quadrant_of(x, y, BOARD) == quadrant
                && farm.tiles[y as usize][x as usize] == Cell::Locked
            {
                farm.tiles[y as usize][x as usize] = Cell::Empty;
            }
        }
    }
}

/// `_process_market`: per-unit lockstep, both players quoted at the same
/// pre-commit inventory; prices refresh after every ORDER INDEX.
pub fn process_market(st: &mut State, actions: &[PlayerAction; 2]) {
    let queues: Vec<Vec<Vec<String>>> = actions
        .iter()
        .map(|a| a.market.iter().take(MAX_MARKET_ORDERS).cloned().collect())
        .collect();
    let max_len = queues.iter().map(Vec::len).max().unwrap_or(0);

    for i in 0..max_len {
        let mut order_states: Vec<Option<OrderState>> = queues
            .iter()
            .map(|q| q.get(i).and_then(|o| parse_order(o)))
            .collect();

        // Atomic orders (HIRE, BUY_LAND): once, in player order.
        for pid in 0..2 {
            if let Some(os) = &order_states[pid] {
                if os.typ == "HIRE" {
                    let (f, p) = (&mut st.farms[pid], &mut st.private[pid]);
                    do_hire(f, p);
                    order_states[pid] = None;
                } else if os.typ == "BUY_LAND" {
                    do_buy_land(&mut st.farms[pid]);
                    order_states[pid] = None;
                }
            }
        }

        // Per-unit lockstep for SELL / BUY_*.
        let mut idx_esc = 0;
        loop {
            idx_esc += 1;
            if idx_esc >= 100_000 {
                break;
            }
            let mut quoted: [Option<(String, String, i64)>; 2] = [None, None];
            for pid in 0..2 {
                let Some(os) = &order_states[pid] else {
                    continue;
                };
                if os.remaining <= 0 {
                    continue;
                }
                let (typ, item) = (os.typ.clone(), os.item.clone());
                let q = match typ.as_str() {
                    "SELL" if PRODUCTS.contains(&item.as_str()) => {
                        let p = market::param(&item).unwrap();
                        Some(market::price(p, st.market.inventory.get(&item) as f64))
                    }
                    "BUY_PRODUCT" if item == "WHEAT" || item == "FERTILIZER" => {
                        // Quoted at post-buy inventory.
                        let p = market::param(&item).unwrap();
                        Some(market::price(
                            p,
                            (st.market.inventory.get(&item) - 1) as f64,
                        ))
                    }
                    "BUY_SEED" => rules::crop(&item).map(|c| c.seed_cost),
                    "BUY_ANIMAL" => rules::animal(&item).map(|a| a.cost),
                    _ => None,
                };
                match q {
                    Some(price) => quoted[pid] = Some((typ, item, price)),
                    None => order_states[pid] = None, // malformed; abort
                }
            }

            if quoted.iter().all(Option::is_none) {
                break;
            }

            let mut committed_any = false;
            for pid in 0..2 {
                let Some((typ, item, price)) = quoted[pid].take() else {
                    continue;
                };
                let ok = {
                    let (farms, privates, mkt) = (&mut st.farms, &mut st.private, &mut st.market);
                    commit_unit(&typ, &item, price, &mut farms[pid], &mut privates[pid], mkt)
                };
                if ok {
                    if let Some(os) = &mut order_states[pid] {
                        os.remaining -= 1;
                    }
                    committed_any = true;
                } else {
                    order_states[pid] = None;
                }
            }
            if !committed_any {
                break;
            }
        }

        refresh_prices(&mut st.market);
    }
}

pub fn refresh_prices(mkt: &mut crate::state::Market) {
    for p in market::PARAMS.iter() {
        let inv = mkt.inventory.get(p.item);
        let price = market::price(p, inv as f64);
        if let Some(e) = mkt.prices.0.iter_mut().find(|(n, _)| n == p.item) {
            e.1 = price;
        }
    }
}

// -------------------------------------------------------------------- town --

/// `_town_consume`.
pub fn town_consume(st: &mut State, step: i64) {
    if step % TOWN_SHOP_SELL_INTERVAL == 0 {
        let shops = st.town.unlocked_shops.clone();
        for shop in shops {
            let products = shop_products(&shop);
            let multiplier = if products.len() == 1 { 2 } else { 1 };
            for item in products {
                st.market.inventory.sub(item, multiplier);
            }
        }
    }
    if step % TOWN_CENTER_SELL_INTERVAL == 0 {
        for item in PRODUCTS.iter().filter(|i| **i != "FERTILIZER") {
            st.market.inventory.sub(item, 1);
        }
    }
    refresh_prices(&mut st.market);
}

// ------------------------------------------------------------------- daily --

/// `_decay_plants`.
pub fn decay_plants(farm: &mut Farm, step: i64) {
    for y in 0..BOARD as usize {
        for x in 0..BOARD as usize {
            let Cell::Plant {
                yield_units,
                max_lifespan_step,
                ..
            } = &mut farm.tiles[y][x]
            else {
                continue;
            };
            let mls = *max_lifespan_step;
            if mls < 0 || step < mls {
                continue;
            }
            if (step - mls) % 2 != 0 {
                continue;
            }
            *yield_units -= 1;
            if *yield_units <= 0 {
                farm.tiles[y][x] = Cell::Weed;
            }
        }
    }
}

/// `_daily_refresh_plants`.
fn daily_refresh_plants(farm: &mut Farm, current_day: i64) {
    let next_day = current_day + 1;
    for y in 0..BOARD as usize {
        for x in 0..BOARD as usize {
            let Cell::Plant {
                crop,
                planted_day,
                watered_today,
                consecutive_unwatered,
                yield_units,
                max_lifespan_step,
                fertilized_until_day,
            } = &mut farm.tiles[y][x]
            else {
                continue;
            };
            let was_watered = *watered_today;
            if was_watered {
                *consecutive_unwatered = 0;
            } else {
                *consecutive_unwatered += 1;
            }
            *watered_today = false;
            if *consecutive_unwatered >= 2 {
                farm.tiles[y][x] = Cell::Weed;
                continue;
            }
            let cd = rules::crop(crop).unwrap();
            if !cd.ongoing {
                continue;
            }
            let days_since_first = next_day - *planted_day - cd.first_yield_day;
            if days_since_first < 0 {
                continue;
            }
            if days_since_first % cd.interval != 0 {
                continue;
            }
            let production_count = days_since_first / cd.interval + 1;
            if production_count > cd.max_yield {
                continue;
            }
            // Fertilizer bonus only applies on watered days.
            let fertilized = was_watered && *fertilized_until_day >= current_day;
            *yield_units = cd
                .max_yield
                .min(*yield_units + if fertilized { 2 } else { 1 });
            if production_count == cd.max_yield {
                *max_lifespan_step = (next_day + 1) * TURNS_PER_DAY;
            }
        }
    }
}

/// `_daily_refresh_animals`.
fn daily_refresh_animals(farm: &mut Farm, day: i64) {
    let next_day = day + 1;
    for y in 0..BOARD as usize {
        for x in 0..BOARD as usize {
            let Cell::Structure {
                animal: Some(a), ..
            } = &mut farm.tiles[y][x]
            else {
                continue;
            };
            if a.fed_today {
                a.consecutive_unfed = 0;
            } else {
                a.consecutive_unfed += 1;
            }
            if a.consecutive_unfed >= 2 {
                // Animal escapes; structure remains.
                let structure = rules::animal(&a.animal).unwrap().structure.to_string();
                farm.tiles[y][x] = Cell::Structure {
                    kind: structure,
                    animal: None,
                };
                continue;
            }
            let ad = rules::animal(&a.animal).unwrap();
            let days_since_first = next_day - a.placed_day - ad.first_yield_day;
            if days_since_first >= 0 && days_since_first % ad.interval == 0 {
                // Care bonus only consumed on a fed production day; Python
                // pops the key, then re-sets it to 0.
                let bonus = if a.fed_today {
                    let b = a.pending_care_bonus;
                    a.pending_care_bonus = 0;
                    b
                } else {
                    0
                };
                a.yield_units = ad.max_held.min(a.yield_units + 1 + bonus);
                a.pending_care_bonus = 0;
            }
            if a.cared_today && a.fed_today {
                a.pending_care_bonus += 1;
            }
            a.fertilizer_available = true;
            a.fed_today = false;
            a.cared_today = false;
        }
    }
}

/// `_spawn_weeds` -- RNG consumer #1: one draw per EMPTY tile, y-then-x.
fn spawn_weeds(farm: &mut Farm, rng: &mut MT) {
    for y in 0..BOARD as usize {
        for x in 0..BOARD as usize {
            if farm.tiles[y][x] == Cell::Empty && rng.random() < WEED_CHANCE {
                farm.tiles[y][x] = Cell::Weed;
            }
        }
    }
}

/// `_drop_inventories_to_shed`: overflow is DISCARDED, and which item
/// overflows depends on inventory insertion order (see state.rs).
fn drop_inventories_to_shed(private: &mut Private) {
    for i in 0..private.inventories.len() {
        let items: Vec<(String, i64)> = private.inventories[i].0.clone();
        for (item, n) in items {
            if n <= 0 {
                private.inventories[i].remove(&item);
                continue;
            }
            let room = (SHED_CAP - private.shed.sum()).max(0);
            let take = n.min(room);
            if take > 0 {
                private.shed.add(&item, take);
            }
            private.inventories[i].remove(&item);
        }
    }
}

/// `_end_of_day`. Farm 0 is fully processed before farm 1 touches the shared
/// per-day RNG stream; the shop draw comes after both farms.
pub fn end_of_day(st: &mut State, day: i64) {
    let mut rng = MT::for_day(st.seed, day);
    for pid in 0..2 {
        daily_refresh_plants(&mut st.farms[pid], day);
        daily_refresh_animals(&mut st.farms[pid], day);
        spawn_weeds(&mut st.farms[pid], &mut rng);
        drop_inventories_to_shed(&mut st.private[pid]);
        st.farms[pid].farmer = default_spawn();
        st.farms[pid].hands.clear();
        st.farms[pid].hires_today = 0;
        st.private[pid].inventories = vec![OMap::default()];
    }
    let next_day = day + 1;
    if next_day > 0
        && next_day % TOWN_SHOP_UNLOCK_INTERVAL == 0
        && st.town.unlocked_shops.len() < rules::MAX_SHOP_INSTANCES
    {
        let shop = *rng.choice(&SHOPS_SORTED);
        st.town.unlocked_shops.push(shop.to_string());
    }
}

// -------------------------------------------------------------------- step --

/// One full interpreter step: unit actions (player 0's whole list first),
/// market, town, decay, end-of-day. Returns true while the episode runs.
pub fn step(st: &mut State, actions: &[PlayerAction; 2]) -> bool {
    let step = st.step;
    let day = step / TURNS_PER_DAY;

    for pid in 0..2 {
        let pa = &actions[pid];
        // Atomic PLANT validation: if total PLANT requests for a crop exceed
        // available seeds, drop ALL PLANT requests for that crop this turn.
        let mut unit_actions: Vec<UnitAction> = vec![pa.farmer.clone()];
        unit_actions.extend(pa.hands.iter().cloned());
        let mut demand: Vec<(String, i64)> = Vec::new();
        for a in &unit_actions {
            if a.op == "PLANT" && !a.item.is_empty() {
                if let Some(e) = demand.iter_mut().find(|(c, _)| *c == a.item) {
                    e.1 += 1;
                } else {
                    demand.push((a.item.clone(), 1));
                }
            }
        }
        let blocked: Vec<String> = demand
            .iter()
            .filter(|(c, n)| *n > st.private[pid].seeds.get(c))
            .map(|(c, _)| c.clone())
            .collect();
        for (idx, a) in unit_actions.iter().enumerate() {
            let a = if a.op == "PLANT" && blocked.contains(&a.item) {
                UnitAction {
                    op: "PASS".to_string(),
                    ..Default::default()
                }
            } else {
                a.clone()
            };
            let (farm, private) = (&mut st.farms[pid], &mut st.private[pid]);
            apply_unit_action(farm, private, idx, &a, day);
        }
    }

    process_market(st, actions);
    town_consume(st, step);
    for pid in 0..2 {
        decay_plants(&mut st.farms[pid], step);
    }
    if (step + 1) % TURNS_PER_DAY == 0 {
        end_of_day(st, day);
    }
    st.step = step + 1;
    // Python fires DONE when the pre-step counter reaches episodeSteps - 2.
    step < EPISODE_STEPS - 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positional_keeps_empty_slots() {
        assert!(positional("").is_empty());
        assert_eq!(positional("A"), vec!["A"]);
        assert_eq!(positional(";A"), vec!["", "A"]);
        assert_eq!(positional("A;;B"), vec!["A", "", "B"]);
        assert_eq!(positional("A;"), vec!["A", ""]);
    }

    #[test]
    fn step_reports_done_at_the_official_boundary() {
        let mut st = State::new(1);
        let idle = [PlayerAction::default(), PlayerAction::default()];
        let mut running = true;
        let mut n = 0;
        while running {
            running = step(&mut st, &idle);
            n += 1;
        }
        // DONE fires on the step whose pre-step counter is 718.
        assert_eq!(n, 719);
        assert_eq!(st.step, crate::state::FINAL_STEP);
    }
}

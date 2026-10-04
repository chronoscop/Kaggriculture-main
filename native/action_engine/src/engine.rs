//! Turn processing: a 1:1 port of `../child_exp00/fast_env.py` (validated
//! against the reference engine). Processing order per step: unit actions
//! (player 0 then 1) -> market lockstep -> town consumption -> plant decay ->
//! end-of-day refresh (weeds/shop unlock). Any deviation here is a parity bug.

use crate::py_random::PyRandom;
use crate::state::*;

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/// Quadrant id: 0=NW 1=NE 2=SW 3=SE.
fn quadrant_of(x: i64, y: i64, board_size: i64) -> usize {
    let half = board_size / 2;
    let south = if y < half { 0 } else { 2 };
    let east = if x < half { 0 } else { 1 };
    south + east
}

pub const QUADRANT_NAMES: [&str; 4] = ["NW", "NE", "SW", "SE"];

fn quadrant_id(name: &str) -> usize {
    QUADRANT_NAMES
        .iter()
        .position(|q| *q == name)
        .expect("valid quadrant")
}

/// Four inner-corner tiles around the shed, in NWSE order.
fn shed_access_tiles(board_size: i64) -> [(i64, i64); 4] {
    let half = board_size / 2;
    [
        (half - 1, half - 1),
        (half, half - 1),
        (half - 1, half),
        (half, half),
    ]
}

pub(crate) fn is_shed_adjacent(x: i64, y: i64, board_size: i64) -> bool {
    shed_access_tiles(board_size).contains(&(x, y))
}

/// First shed-access tile inside NW; with an even board that is always the
/// first (NW-most) tile, matching the reference loop's first hit.
fn default_spawn(board_size: i64) -> (i64, i64) {
    for tile in shed_access_tiles(board_size) {
        if quadrant_of(tile.0, tile.1, board_size) == 0 {
            return tile;
        }
    }
    (0, 0)
}

fn new_farm(board_size: i64, starting_money: i64) -> Farm {
    let b = board_size as usize;
    let tiles = (0..b)
        .map(|y| {
            (0..b)
                .map(|x| {
                    if quadrant_of(x as i64, y as i64, board_size) == 0 {
                        Tile::Empty
                    } else {
                        Tile::Locked
                    }
                })
                .collect()
        })
        .collect();
    Farm {
        money: starting_money as f64,
        tiles,
        farmer: default_spawn(board_size),
        hands: Vec::new(),
        unlocked_quadrants: vec!["NW"],
        hires_today: 0,
    }
}

// ---------------------------------------------------------------------------
// Market pricing
// ---------------------------------------------------------------------------

/// CPython `round()` on a float: nearest integer, ties to even.
fn py_round(x: f64) -> i64 {
    x.round_ties_even() as i64
}

pub fn market_price(item: usize, inventory: i64, params: &[MarketParam; N_PRODUCTS]) -> i64 {
    let p = &params[item];
    let base = p.base.f();
    let i0 = p.i0.f();
    let t = p.t.f();
    let inv = inventory as f64;
    let price = if inv < i0 {
        let amp = p.below_target.f() * base / p.below_func.shape(t, t);
        base + amp * p.below_func.shape(i0 - inv, t)
    } else {
        let amp = p.above_target.f() * base / p.above_func.shape(t, t);
        base - amp * p.above_func.shape(inv - i0, t)
    };
    py_round(price).max(PRICE_FLOOR)
}

fn refresh_prices(market: &mut Market) {
    for item in 0..N_PRODUCTS {
        market.prices[item] = Num::Int(market_price(item, market.inventory[item], &market.params));
    }
}

// ---------------------------------------------------------------------------
// Farmer/hand bookkeeping
// ---------------------------------------------------------------------------

/// idx 0 = main farmer, 1+ = hand index; None when the hand does not exist.
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

fn shed_total(shed: &[i64; N_ITEMS]) -> i64 {
    shed.iter().sum()
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

// ---------------------------------------------------------------------------
// Unit actions
// ---------------------------------------------------------------------------

/// Process one farmer/hand action. Invalid / illegal actions are silent no-ops.
/// `blocked` implements the atomic-PLANT rule: PLANT requests for a crop whose
/// total demand this turn exceeds the seed stock are dropped for all units.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_unit_action(
    farm: &mut Farm,
    private: &mut Private,
    idx: usize,
    action: &TokenList,
    blocked: &[bool; N_CROPS],
    board_size: i64,
    day: i64,
    turns_per_day: i64,
    shed_capacity: i64,
) {
    let Some(tokens) = action else { return };
    if tokens.is_empty() {
        return;
    }
    // A non-string op matches nothing in the reference either.
    let Token::Str(op) = &tokens[0] else { return };
    let Some((fx, fy)) = farmer_position(farm, idx) else {
        return;
    };
    // Mirrors the reference's defensive inventory growth (a no-op in practice
    // because hands and inventories are appended/reset together).
    while private.inventories.len() <= idx {
        private.inventories.push(Inventory::default());
    }

    if let Some((dx, dy)) = move_delta(op) {
        let (nx, ny) = (fx + dx, fy + dy);
        if nx < 0 || nx >= board_size || ny < 0 || ny >= board_size {
            return;
        }
        // Movement onto LOCKED tiles is allowed (see reference for rationale).
        set_farmer_position(farm, idx, (nx, ny));
        return;
    }
    if op == "PASS" {
        return;
    }

    let (tx, ty) = (fx as usize, fy as usize);

    // Shed operations resolve before the LOCKED guard (reference ordering).
    match op.as_str() {
        "DROP" => {
            if !is_shed_adjacent(fx, fy, board_size) {
                return;
            }
            // Every entry leaves the inventory; overflow past capacity is discarded.
            let entries = std::mem::take(&mut private.inventories[idx].0);
            for (item, n) in entries {
                if n <= 0 {
                    continue;
                }
                let room = (shed_capacity - shed_total(&private.shed)).max(0);
                let take = n.min(room);
                if take > 0 {
                    private.shed[item] += take;
                }
            }
            return;
        }
        "PICKUP" => {
            if !is_shed_adjacent(fx, fy, board_size) {
                return;
            }
            if tokens.len() < 2 {
                return;
            }
            let n0 = match tokens.get(2) {
                None => 1,
                Some(t) => match token_int(t) {
                    Some(v) => v,
                    None => return,
                },
            };
            if n0 <= 0 {
                return;
            }
            // Unknown item names read as shed count 0 in the reference.
            let Some(item) = token_str(&tokens[1]).and_then(item_id) else {
                return;
            };
            let n = n0.min(private.shed[item]);
            if n <= 0 {
                return;
            }
            private.shed[item] -= n;
            private.inventories[idx].add(item, n);
            return;
        }
        "PLACE" => {
            if tokens.len() < 2 {
                return;
            }
            let item_opt = token_str(&tokens[1]).and_then(item_id);
            // Animal placement: standing on a matching unoccupied structure.
            if let Some(item) = item_opt
                && item >= FIRST_ANIMAL
            {
                let a = &ANIMALS[item - FIRST_ANIMAL];
                if matches!(&farm.tiles[ty][tx], Tile::Structure(s) if *s == a.structure) {
                    if private.inventories[idx].take(item, 1) {
                        farm.tiles[ty][tx] =
                            Tile::Animal(AnimalTile::new(item - FIRST_ANIMAL, day));
                    }
                    return;
                }
            }
            // Shed drop: orthogonally adjacent to the shed; obeys shedCapacity.
            if is_shed_adjacent(fx, fy, board_size) {
                let n0 = match tokens.get(2) {
                    None => 1,
                    Some(t) => match token_int(t) {
                        Some(v) => v,
                        None => return,
                    },
                };
                if n0 <= 0 {
                    return;
                }
                let Some(item) = item_opt else { return };
                let n1 = n0.min(private.inventories[idx].get(item));
                if n1 <= 0 {
                    return;
                }
                let room = (shed_capacity - shed_total(&private.shed)).max(0);
                let n = n1.min(room);
                if n <= 0 {
                    return;
                }
                private.inventories[idx].take(item, n);
                private.shed[item] += n;
            }
            return;
        }
        _ => {}
    }

    // Everything below mutates the tile the unit stands on -> requires ownership.
    if matches!(farm.tiles[ty][tx], Tile::Locked) {
        return;
    }

    match op.as_str() {
        "PLANT" => {
            if tokens.len() < 2 {
                return;
            }
            // Non-crop names are always "blocked" upstream too (demand > 0 seeds).
            let Some(crop) = token_str(&tokens[1]).and_then(crop_id) else {
                return;
            };
            if blocked[crop] {
                return;
            }
            if !matches!(farm.tiles[ty][tx], Tile::Empty) {
                return;
            }
            if private.seeds[crop] <= 0 {
                return;
            }
            private.seeds[crop] -= 1;
            farm.tiles[ty][tx] = Tile::Plant(Plant::new(crop, day, turns_per_day));
        }
        "WATER" => {
            let Tile::Plant(plant) = &mut farm.tiles[ty][tx] else {
                return;
            };
            if plant.watered_today {
                return;
            }
            plant.watered_today = true;
            let cd = &CROPS[plant.crop];
            if !cd.ongoing {
                let age_days = day - plant.planted_day;
                let window_start = (cd.max_yield_day + 1) / 2;
                if window_start <= age_days && age_days <= cd.max_yield_day {
                    let bonus = if plant.fertilized_until_day >= day {
                        2
                    } else {
                        1
                    };
                    plant.yield_units = (plant.yield_units + bonus).min(cd.max_yield);
                }
            }
        }
        "HARVEST" => match &mut farm.tiles[ty][tx] {
            Tile::Plant(plant) => {
                if plant.yield_units <= 0 {
                    return;
                }
                let cd = &CROPS[plant.crop];
                if day - plant.planted_day < cd.first_yield_day {
                    return;
                }
                let units = plant.yield_units;
                plant.yield_units = 0;
                let crop = plant.crop;
                let ongoing = cd.ongoing;
                private.inventories[idx].add(crop, units);
                if !ongoing {
                    farm.tiles[ty][tx] = Tile::Empty;
                }
            }
            Tile::Animal(animal) => {
                if animal.yield_units <= 0 {
                    return;
                }
                let units = animal.yield_units;
                animal.yield_units = 0;
                let product = ANIMALS[animal.animal].product;
                private.inventories[idx].add(product, units);
            }
            _ => {}
        },
        "FERTILIZE" => {
            if !matches!(farm.tiles[ty][tx], Tile::Plant(_)) {
                return;
            }
            if !private.inventories[idx].take(FERTILIZER, 1) {
                return;
            }
            if let Tile::Plant(plant) = &mut farm.tiles[ty][tx] {
                // Active for day, day+1, day+2 (3 days inclusive).
                plant.fertilized_until_day = plant.fertilized_until_day.max(day + 2);
            }
        }
        "DIG" => {
            // Removes plants, weeds, empty coop/pasture. Not empty tiles or animals.
            match farm.tiles[ty][tx] {
                Tile::Empty | Tile::Animal(_) => {}
                _ => farm.tiles[ty][tx] = Tile::Empty,
            }
        }
        "BUILD_COOP" => {
            if matches!(farm.tiles[ty][tx], Tile::Empty) {
                farm.tiles[ty][tx] = Tile::Structure(Structure::Coop);
            }
        }
        "BUILD_PASTURE" => {
            if matches!(farm.tiles[ty][tx], Tile::Empty) {
                farm.tiles[ty][tx] = Tile::Structure(Structure::Pasture);
            }
        }
        "FEED" => {
            let Tile::Animal(animal) = &farm.tiles[ty][tx] else {
                return;
            };
            if animal.fed_today {
                return;
            }
            if !private.inventories[idx].take(WHEAT, 1) {
                return;
            }
            if let Tile::Animal(animal) = &mut farm.tiles[ty][tx] {
                animal.fed_today = true;
            }
        }
        "COLLECT_FERTILIZER" => {
            let Tile::Animal(animal) = &mut farm.tiles[ty][tx] else {
                return;
            };
            if !animal.fertilizer_available {
                return;
            }
            animal.fertilizer_available = false;
            private.inventories[idx].add(FERTILIZER, 1);
        }
        "CARE" => {
            if let Tile::Animal(animal) = &mut farm.tiles[ty][tx]
                && !animal.cared_today
            {
                animal.cared_today = true;
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Market orders
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TradeOp {
    Sell,
    BuyProduct,
    BuySeed,
    BuyAnimal,
}

#[derive(Clone, Debug)]
pub enum Order {
    Hire,
    BuyLand,
    Trade {
        op: TradeOp,
        /// item id when the name is canonical; None reads as malformed at quote
        /// time (the whole order is silently dropped, like the reference).
        item: Option<usize>,
        remaining: i64,
    },
}

pub fn parse_order(tokens: &TokenList) -> Option<Order> {
    let toks = tokens.as_ref()?;
    if toks.is_empty() {
        return None;
    }
    let Token::Str(op) = &toks[0] else {
        return None;
    };
    match op.as_str() {
        "HIRE" => Some(Order::Hire),
        "BUY_LAND" => Some(Order::BuyLand),
        "BUY_SEED" | "BUY_PRODUCT" | "BUY_ANIMAL" | "SELL" => {
            if toks.len() < 3 {
                return None;
            }
            let n = token_int(&toks[2])?;
            if n <= 0 {
                return None;
            }
            let trade_op = match op.as_str() {
                "SELL" => TradeOp::Sell,
                "BUY_SEED" => TradeOp::BuySeed,
                "BUY_PRODUCT" => TradeOp::BuyProduct,
                _ => TradeOp::BuyAnimal,
            };
            Some(Order::Trade {
                op: trade_op,
                item: token_str(&toks[1]).and_then(item_id),
                remaining: n,
            })
        }
        _ => None,
    }
}

fn commit_unit(
    op: TradeOp,
    item: usize,
    price: i64,
    farm: &mut Farm,
    private: &mut Private,
    market: &mut Market,
    shed_capacity: i64,
) -> bool {
    match op {
        TradeOp::Sell => {
            if private.shed[item] <= 0 {
                return false;
            }
            private.shed[item] -= 1;
            farm.money += price as f64;
            // Sales at $1 do not increase market supply.
            if price > 1 {
                market.inventory[item] += 1;
            }
            true
        }
        TradeOp::BuyProduct => {
            if farm.money < price as f64 {
                return false;
            }
            // Bought goods land in the shed, which obeys shedCapacity.
            if shed_total(&private.shed) >= shed_capacity {
                return false;
            }
            farm.money -= price as f64;
            private.shed[item] += 1;
            market.inventory[item] -= 1;
            true
        }
        TradeOp::BuySeed => {
            if farm.money < price as f64 {
                return false;
            }
            farm.money -= price as f64;
            private.seeds[item] += 1;
            true
        }
        TradeOp::BuyAnimal => {
            if farm.money < price as f64 {
                return false;
            }
            if shed_total(&private.shed) >= shed_capacity {
                return false;
            }
            farm.money -= price as f64;
            private.shed[item] += 1;
            true
        }
    }
}

/// Indexed so fib(0)=1, fib(1)=1, fib(2)=2, fib(3)=3, fib(4)=5...
pub(crate) fn fib(n: i64) -> i64 {
    let (mut a, mut b) = (1i64, 1i64);
    for _ in 0..n {
        (a, b) = (b, a + b);
    }
    a
}

fn do_hire(farm: &mut Farm, private: &mut Private, board_size: i64, mult: i64) {
    let cost = mult * fib(farm.hires_today);
    if farm.money < cost as f64 {
        return;
    }
    farm.money -= cost as f64;
    farm.hires_today += 1;
    let spawn = spawn_hand(farm, board_size);
    farm.hands.push(spawn);
    private.inventories.push(Inventory::default());
}

/// First free shed-access tile (NWSE order); ties broken by min occupancy.
fn spawn_hand(farm: &Farm, board_size: i64) -> (i64, i64) {
    let tiles = shed_access_tiles(board_size);
    let mut counts = [0i64; 4];
    for pos in std::iter::once(farm.farmer).chain(farm.hands.iter().copied()) {
        if let Some(k) = tiles.iter().position(|t| *t == pos) {
            counts[k] += 1;
        }
    }
    // Stable min by (count, NWSE index) == Python's sorted(...)[0].
    let mut best = 0;
    for k in 1..4 {
        if counts[k] < counts[best] {
            best = k;
        }
    }
    tiles[best]
}

fn do_buy_land(farm: &mut Farm, board_size: i64) {
    let n_unlocked_extra = farm.unlocked_quadrants.len() - 1; // NW is always there
    if n_unlocked_extra >= LAND_ORDER.len() {
        return;
    }
    let cost = LAND_PRICES[n_unlocked_extra];
    if farm.money < cost as f64 {
        return;
    }
    farm.money -= cost as f64;
    let quadrant = LAND_ORDER[n_unlocked_extra];
    farm.unlocked_quadrants.push(quadrant);
    let q = quadrant_id(quadrant);
    for (y, row) in farm.tiles.iter_mut().enumerate() {
        for (x, tile) in row.iter_mut().enumerate() {
            if quadrant_of(x as i64, y as i64, board_size) == q && matches!(tile, Tile::Locked) {
                *tile = Tile::Empty;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Daily refresh
// ---------------------------------------------------------------------------

fn decay_plants(farm: &mut Farm, step: i64) {
    for row in farm.tiles.iter_mut() {
        for tile in row.iter_mut() {
            let Tile::Plant(plant) = tile else { continue };
            let mls = plant.max_lifespan_step;
            if mls < 0 || step < mls {
                continue;
            }
            if (step - mls) % 2 != 0 {
                continue;
            }
            plant.yield_units -= 1;
            if plant.yield_units <= 0 {
                *tile = Tile::Weed;
            }
        }
    }
}

fn daily_refresh_plants(farm: &mut Farm, current_day: i64, turns_per_day: i64) {
    let next_day = current_day + 1;
    for row in farm.tiles.iter_mut() {
        for tile in row.iter_mut() {
            let Tile::Plant(plant) = tile else { continue };
            let was_watered = plant.watered_today;
            if was_watered {
                plant.consecutive_unwatered = 0;
            } else {
                plant.consecutive_unwatered += 1;
            }
            plant.watered_today = false;
            if plant.consecutive_unwatered >= 2 {
                *tile = Tile::Weed;
                continue;
            }
            let cd = &CROPS[plant.crop];
            if !cd.ongoing {
                continue;
            }
            let days_since_first = next_day - plant.planted_day - cd.first_yield_day;
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
            // Fertilizer bonus only applies on watered days (basic needs first).
            let fertilized = was_watered && plant.fertilized_until_day >= current_day;
            plant.yield_units =
                (plant.yield_units + if fertilized { 2 } else { 1 }).min(cd.max_yield);
            if production_count == cd.max_yield {
                plant.max_lifespan_step = (next_day + 1) * turns_per_day;
            }
        }
    }
}

fn daily_refresh_animals(farm: &mut Farm, day: i64) {
    let next_day = day + 1;
    for row in farm.tiles.iter_mut() {
        for tile in row.iter_mut() {
            let Tile::Animal(animal) = tile else { continue };
            if animal.fed_today {
                animal.consecutive_unfed = 0;
            } else {
                animal.consecutive_unfed += 1;
            }
            if animal.consecutive_unfed >= 2 {
                // Animal escapes; structure remains.
                *tile = Tile::Structure(ANIMALS[animal.animal].structure);
                continue;
            }
            let a = &ANIMALS[animal.animal];
            let days_since_first = next_day - animal.placed_day - a.first_yield_day;
            if days_since_first >= 0 && days_since_first % a.interval == 0 {
                // Care bonus only consumed on a fed production day.
                let bonus = if animal.fed_today {
                    animal.pending_care_bonus
                } else {
                    0
                };
                animal.yield_units = (animal.yield_units + 1 + bonus).min(a.max_held);
                animal.pending_care_bonus = 0;
            }
            if animal.cared_today && animal.fed_today {
                animal.pending_care_bonus += 1;
            }
            animal.fertilizer_available = true;
            animal.fed_today = false;
            animal.cared_today = false;
        }
    }
}

fn spawn_weeds(farm: &mut Farm, weed_chance: f64, rng: &mut PyRandom) {
    for row in farm.tiles.iter_mut() {
        for tile in row.iter_mut() {
            // Short-circuit order matters: rng only advances on empty tiles.
            if matches!(tile, Tile::Empty) && rng.random() < weed_chance {
                *tile = Tile::Weed;
            }
        }
    }
}

/// Drop every per-farmer inventory into the shed up to capacity; overflow is discarded.
fn drop_inventories_to_shed(private: &mut Private, capacity: i64) {
    let inventories = std::mem::take(&mut private.inventories);
    for inv in inventories {
        for (item, n) in inv.0 {
            if n <= 0 {
                continue;
            }
            let room = (capacity - shed_total(&private.shed)).max(0);
            let take = n.min(room);
            if take > 0 {
                private.shed[item] += take;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Engine {
    pub cfg: Config,
    pub seed: i64,
    pub step_no: i64,
    pub done: bool,
    pub farms: [Farm; 2],
    pub privates: [Private; 2],
    pub market: Market,
    pub town: Town,
}

impl Engine {
    pub fn new(seed: i64, mut cfg: Config) -> Engine {
        // Same clamps as FastEnv.__init__.
        cfg.max_orders = cfg.max_orders.max(1);
        cfg.turns_per_day = cfg.turns_per_day.max(1);
        cfg.shop_unlock_interval = cfg.shop_unlock_interval.max(1);
        cfg.shop_sell_interval = cfg.shop_sell_interval.max(1);
        cfg.center_sell_interval = cfg.center_sell_interval.max(1);
        let mut engine = Engine {
            seed,
            step_no: 0,
            done: false,
            farms: [
                new_farm(cfg.board_size, cfg.starting_money),
                new_farm(cfg.board_size, cfg.starting_money),
            ],
            privates: [Private::new(), Private::new()],
            market: Market::new(cfg.market_overrides.clone()),
            town: Town::default(),
            cfg,
        };
        engine.reset(seed);
        engine
    }

    pub fn reset(&mut self, seed: i64) {
        self.seed = seed;
        self.step_no = 0;
        self.done = false;
        self.farms = [
            new_farm(self.cfg.board_size, self.cfg.starting_money),
            new_farm(self.cfg.board_size, self.cfg.starting_money),
        ];
        self.privates = [Private::new(), Private::new()];
        self.market = Market::new(self.cfg.market_overrides.clone());
        self.town = Town::default();
    }

    pub fn rewards(&self) -> [f64; 2] {
        [self.farms[0].money, self.farms[1].money]
    }

    /// Observation day/hour for the current (post-step) step counter.
    pub fn day_hour(&self) -> (i64, i64) {
        (
            self.step_no / self.cfg.turns_per_day,
            self.step_no % self.cfg.turns_per_day,
        )
    }

    pub fn step(&mut self, actions: &[PlayerAction; 2]) -> Result<bool, &'static str> {
        if self.done {
            return Err("Environment done, reset required.");
        }
        // Mirrors the reference interpreter's pre-increment counter.
        let step = self.step_no;
        let day = step / self.cfg.turns_per_day;

        for (p, action) in actions.iter().enumerate() {
            self.apply_player_actions(p, action, day);
        }
        self.process_market(actions);
        self.town_consume(step);
        for farm in &mut self.farms {
            decay_plants(farm, step);
        }
        if (step + 1) % self.cfg.turns_per_day == 0 {
            self.end_of_day(day);
        }

        self.step_no = step + 1;
        if step >= self.cfg.episode_steps - 2 {
            self.done = true;
        }
        Ok(self.done)
    }

    fn apply_player_actions(&mut self, player: usize, action: &PlayerAction, day: i64) {
        // Atomic PLANT validation: if total PLANT requests for a crop this turn
        // exceed available seeds, drop ALL PLANT requests for that crop.
        let mut demand = [0i64; N_CROPS];
        for a in std::iter::once(&action.farmer)
            .chain(action.hands.iter())
            .flatten()
        {
            if a.len() >= 2
                && matches!(&a[0], Token::Str(s) if s == "PLANT")
                && let Some(crop) = token_str(&a[1]).and_then(crop_id)
            {
                demand[crop] += 1;
            }
        }
        let seeds = &self.privates[player].seeds;
        let blocked: [bool; N_CROPS] = std::array::from_fn(|c| demand[c] > seeds[c]);

        let (board_size, turns_per_day, shed_capacity) = (
            self.cfg.board_size,
            self.cfg.turns_per_day,
            self.cfg.shed_capacity,
        );
        let farm = &mut self.farms[player];
        let private = &mut self.privates[player];
        apply_unit_action(
            farm,
            private,
            0,
            &action.farmer,
            &blocked,
            board_size,
            day,
            turns_per_day,
            shed_capacity,
        );
        for (h_idx, hand_action) in action.hands.iter().enumerate() {
            apply_unit_action(
                farm,
                private,
                h_idx + 1,
                hand_action,
                &blocked,
                board_size,
                day,
                turns_per_day,
                shed_capacity,
            );
        }
    }

    /// Per-unit lockstep: at each iteration, quote both players' current-unit
    /// prices against the same pre-commit inventory, then commit both.
    pub(crate) fn process_market(&mut self, actions: &[PlayerAction; 2]) {
        let (board_size, max_orders, hire_mult, shed_capacity) = (
            self.cfg.board_size,
            self.cfg.max_orders,
            self.cfg.hire_mult,
            self.cfg.shed_capacity,
        );
        let farms = &mut self.farms;
        let privates = &mut self.privates;
        let market = &mut self.market;

        let mut queues: [Vec<Option<Order>>; 2] = [
            actions[0]
                .market
                .iter()
                .take(max_orders)
                .map(parse_order)
                .collect(),
            actions[1]
                .market
                .iter()
                .take(max_orders)
                .map(parse_order)
                .collect(),
        ];

        let max_len = queues[0].len().max(queues[1].len());
        for i in 0..max_len {
            let mut order_states: [Option<Order>; 2] = [
                queues[0].get_mut(i).and_then(Option::take),
                queues[1].get_mut(i).and_then(Option::take),
            ];

            // Atomic orders (HIRE, BUY_LAND): handle once, in player order.
            for p in 0..2 {
                match order_states[p] {
                    Some(Order::Hire) => {
                        do_hire(&mut farms[p], &mut privates[p], board_size, hire_mult);
                        order_states[p] = None;
                    }
                    Some(Order::BuyLand) => {
                        do_buy_land(&mut farms[p], board_size);
                        order_states[p] = None;
                    }
                    _ => {}
                }
            }

            // Per-unit lockstep loop for SELL / BUY_*.
            let mut idx_esc = 0u32;
            loop {
                idx_esc += 1;
                if idx_esc >= 100_000 {
                    eprintln!(
                        "WARNING: kaggriculture market loop exceeded 100k iterations; aborting"
                    );
                    break;
                }
                let mut quoted: [Option<(TradeOp, usize, i64)>; 2] = [None, None];
                for p in 0..2 {
                    let Some(Order::Trade {
                        op,
                        item,
                        remaining,
                    }) = &order_states[p]
                    else {
                        continue;
                    };
                    if *remaining <= 0 {
                        continue;
                    }
                    let resolved = match op {
                        TradeOp::Sell => item.filter(|&it| it < N_PRODUCTS),
                        TradeOp::BuyProduct => item.filter(|&it| it == WHEAT || it == FERTILIZER),
                        TradeOp::BuySeed => item.filter(|&it| it < N_CROPS),
                        TradeOp::BuyAnimal => item.filter(|&it| it >= FIRST_ANIMAL),
                    };
                    match resolved {
                        // Malformed sub-op; abort this order.
                        None => order_states[p] = None,
                        Some(it) => {
                            let price = match op {
                                TradeOp::Sell => {
                                    market_price(it, market.inventory[it], &market.params)
                                }
                                // Quote at post-buy inventory so a buy/sell round-trip
                                // against an unchanged market nets zero.
                                TradeOp::BuyProduct => {
                                    market_price(it, market.inventory[it] - 1, &market.params)
                                }
                                TradeOp::BuySeed => CROPS[it].seed,
                                TradeOp::BuyAnimal => ANIMALS[it - FIRST_ANIMAL].cost,
                            };
                            quoted[p] = Some((*op, it, price));
                        }
                    }
                }

                if quoted.iter().all(Option::is_none) {
                    break;
                }

                let mut committed_any = false;
                for p in 0..2 {
                    let Some((op, item, price)) = quoted[p] else {
                        continue;
                    };
                    let ok = commit_unit(
                        op,
                        item,
                        price,
                        &mut farms[p],
                        &mut privates[p],
                        market,
                        shed_capacity,
                    );
                    if ok {
                        if let Some(Order::Trade { remaining, .. }) = &mut order_states[p] {
                            *remaining -= 1;
                        }
                        committed_any = true;
                    } else {
                        // Can't continue this order.
                        order_states[p] = None;
                    }
                }

                if !committed_any {
                    break;
                }
            }

            refresh_prices(market);
        }
    }

    fn town_consume(&mut self, step: i64) {
        if step % self.cfg.shop_sell_interval == 0 {
            // unlocked_shops may list the same shop more than once (shops are drawn
            // with replacement); each instance consumes independently.
            for &shop in &self.town.unlocked_shops {
                let products = SHOP_PRODUCTS[shop];
                let multiplier = if products.len() == 1 { 2 } else { 1 };
                for &item in products {
                    self.market.inventory[item] -= multiplier;
                }
            }
        }

        if step % self.cfg.center_sell_interval == 0 {
            // Town center consumes 1 of every product except FERTILIZER (flat rate).
            for item in 0..N_PRODUCTS {
                if item != FERTILIZER {
                    self.market.inventory[item] -= 1;
                }
            }
        }

        refresh_prices(&mut self.market);
    }

    fn end_of_day(&mut self, day: i64) {
        // Stable RNG keyed off seed + day so replays reproduce (reference formula).
        // Weed draws are shared farm0 -> farm1 -> shop choice, in that order.
        let mut rng = PyRandom::new((self.seed.wrapping_mul(1_000_003) ^ day) as u64);

        for p in 0..2 {
            daily_refresh_plants(&mut self.farms[p], day, self.cfg.turns_per_day);
            daily_refresh_animals(&mut self.farms[p], day);
            spawn_weeds(&mut self.farms[p], self.cfg.weed_chance, &mut rng);
            drop_inventories_to_shed(&mut self.privates[p], self.cfg.shed_capacity);
            self.farms[p].farmer = default_spawn(self.cfg.board_size);
            self.farms[p].hands.clear();
            self.farms[p].hires_today = 0;
            self.privates[p].inventories = vec![Inventory::default()];
        }

        let next_day = day + 1;
        if next_day > 0 && next_day % self.cfg.shop_unlock_interval == 0 {
            // Drawn with replacement: the same shop can unlock repeatedly, and each
            // copy consumes independently. Only the total instance count is capped.
            if self.town.unlocked_shops.len() < MAX_SHOP_INSTANCES {
                let choice = rng.choice_index(SHOPS_SORTED.len());
                self.town.unlocked_shops.push(SHOPS_SORTED[choice]);
            }
        }
    }
}

#[cfg(test)]
#[path = "legal_mask_tests.rs"]
mod legal_mask_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fib_is_one_indexed_from_one() {
        assert_eq!([fib(0), fib(1), fib(2), fib(3), fib(4)], [1, 1, 2, 3, 5]);
    }

    #[test]
    fn py_round_uses_bankers_rounding() {
        assert_eq!(py_round(2.5), 2);
        assert_eq!(py_round(3.5), 4);
        assert_eq!(py_round(-2.5), -2);
        assert_eq!(py_round(2.4999999999), 2);
    }

    #[test]
    fn market_price_at_i0_is_base() {
        let params = default_market_params();
        for (item, p) in params.iter().enumerate() {
            assert_eq!(market_price(item, MARKET_I0, &params), p.base.f() as i64);
        }
    }

    #[test]
    fn default_spawn_is_nw_shed_corner() {
        assert_eq!(default_spawn(10), (4, 4));
        assert_eq!(shed_access_tiles(10), [(4, 4), (5, 4), (4, 5), (5, 5)]);
    }

    #[test]
    fn pass_episode_runs_to_done_at_719_steps() {
        let mut engine = Engine::new(42, Config::default());
        let actions = [PlayerAction::empty(), PlayerAction::empty()];
        let mut n = 0;
        loop {
            let done = engine.step(&actions).unwrap();
            n += 1;
            if done {
                break;
            }
        }
        assert_eq!(n, 719);
        assert_eq!(engine.rewards(), [3000.0, 3000.0]);
        assert!(engine.step(&actions).is_err());
    }

    #[test]
    fn buy_seed_and_plant_flow() {
        let mut engine = Engine::new(0, Config::default());
        let buy = PlayerAction {
            farmer: Some(vec![Token::Str("PASS".into())].into()),
            hands: vec![],
            market: vec![Some(
                vec![
                    Token::Str("BUY_SEED".into()),
                    Token::Str("WHEAT".into()),
                    Token::Int(2),
                ]
                .into(),
            )],
        };
        engine.step(&[buy, PlayerAction::empty()]).unwrap();
        assert_eq!(engine.privates[0].seeds[WHEAT], 2);
        assert_eq!(engine.farms[0].money, 3000.0 - 2.0 * 10.0);

        let plant = PlayerAction {
            farmer: Some(vec![Token::Str("PLANT".into()), Token::Str("WHEAT".into())].into()),
            hands: vec![],
            market: vec![],
        };
        engine.step(&[plant, PlayerAction::empty()]).unwrap();
        assert_eq!(engine.privates[0].seeds[WHEAT], 1);
        let (fx, fy) = engine.farms[0].farmer;
        assert!(matches!(
            engine.farms[0].tiles[fy as usize][fx as usize],
            Tile::Plant(_)
        ));
    }
}

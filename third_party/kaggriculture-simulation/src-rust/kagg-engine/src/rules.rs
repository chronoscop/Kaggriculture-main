//! Pure rule functions, transcribed from the interpreter.
//!
//! Everything here is a total function of its arguments -- no state, no RNG --
//! which makes each one exhaustively differential-testable rather than
//! spot-checked. They are ported before the stateful step function precisely
//! because "exhaustively verified" is available here and is not available there.

pub const LAND_ORDER: [&str; 3] = ["NE", "SW", "SE"];
pub const LAND_PRICES: [i64; 3] = [1000, 2000, 4000];
pub const FARM_HAND_COST_MULT: i64 = 1;
pub const MAX_SHOP_INSTANCES: usize = 8;

/// `_fib`, indexed so fib(0)=1, fib(1)=1, fib(2)=2, fib(3)=3, fib(4)=5.
///
/// Note the offset: this is NOT the textbook indexing, and the n-th hire of a
/// day costs `mult * fib(n)`, so an off-by-one here silently mis-prices every
/// hire after the first.
pub fn fib(n: u32) -> i64 {
    let (mut a, mut b) = (1i64, 1i64);
    for _ in 0..n {
        let next = a + b;
        a = b;
        b = next;
    }
    a
}

pub fn hire_cost(n_already_today: u32, mult: i64) -> i64 {
    mult * fib(n_already_today)
}

/// `_quadrant_of`: "N"/"S" by y, "W"/"E" by x. y grows DOWNWARD.
pub fn quadrant_of(x: i64, y: i64, board_size: i64) -> String {
    let half = board_size / 2;
    let ns = if y < half { "N" } else { "S" };
    let we = if x < half { "W" } else { "E" };
    format!("{ns}{we}")
}

/// `_shed_access_tiles`: the four inner corners, in NWSE order.
pub fn shed_access_tiles(board_size: i64) -> [(i64, i64); 4] {
    let half = board_size / 2;
    [
        (half - 1, half - 1),
        (half, half - 1),
        (half - 1, half),
        (half, half),
    ]
}

pub fn is_shed_adjacent(pos: (i64, i64), board_size: i64) -> bool {
    shed_access_tiles(board_size).contains(&pos)
}

/// Cost of the next land unlock, or None when all quadrants are owned.
pub fn next_land(n_unlocked_beyond_nw: usize) -> Option<(&'static str, i64)> {
    if n_unlocked_beyond_nw >= LAND_ORDER.len() {
        None
    } else {
        Some((
            LAND_ORDER[n_unlocked_beyond_nw],
            LAND_PRICES[n_unlocked_beyond_nw],
        ))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Crop {
    pub name: &'static str,
    pub seed_cost: i64,
    pub first_yield_day: i64,
    pub max_yield_day: i64,
    pub interval: i64,
    pub max_yield: i64,
    pub ongoing: bool,
}

pub const CROPS: [Crop; 5] = [
    Crop {
        name: "WHEAT",
        seed_cost: 10,
        first_yield_day: 2,
        max_yield_day: 4,
        interval: 0,
        max_yield: 6,
        ongoing: false,
    },
    Crop {
        name: "CARROT",
        seed_cost: 20,
        first_yield_day: 2,
        max_yield_day: 3,
        interval: 0,
        max_yield: 4,
        ongoing: false,
    },
    Crop {
        name: "TOMATO",
        seed_cost: 50,
        first_yield_day: 8,
        max_yield_day: 8,
        interval: 1,
        max_yield: 4,
        ongoing: true,
    },
    Crop {
        name: "STRAWBERRY",
        seed_cost: 100,
        first_yield_day: 10,
        max_yield_day: 10,
        interval: 2,
        max_yield: 4,
        ongoing: true,
    },
    Crop {
        name: "MELON",
        seed_cost: 80,
        first_yield_day: 10,
        max_yield_day: 12,
        interval: 0,
        max_yield: 6,
        ongoing: false,
    },
];

#[derive(Clone, Copy, Debug)]
pub struct Animal {
    pub name: &'static str,
    pub cost: i64,
    pub structure: &'static str,
    pub first_yield_day: i64,
    pub interval: i64,
    pub max_held: i64,
    pub product: &'static str,
}

pub const ANIMALS: [Animal; 3] = [
    Animal {
        name: "GOOSE",
        cost: 300,
        structure: "COOP",
        first_yield_day: 4,
        interval: 1,
        max_held: 4,
        product: "EGG",
    },
    Animal {
        name: "COW",
        cost: 400,
        structure: "PASTURE",
        first_yield_day: 8,
        interval: 2,
        max_held: 6,
        product: "MILK",
    },
    Animal {
        name: "SHEEP",
        cost: 500,
        structure: "PASTURE",
        first_yield_day: 6,
        interval: 3,
        max_held: 6,
        product: "WOOL",
    },
];

pub fn crop(name: &str) -> Option<&'static Crop> {
    CROPS.iter().find(|c| c.name == name)
}

pub fn animal(name: &str) -> Option<&'static Animal> {
    ANIMALS.iter().find(|a| a.name == name)
}

/// `_new_plant`'s lifespan rule: ongoing crops never expire (-1), others die
/// at `(day + max_yield_day + 1) * turns_per_day`.
pub fn max_lifespan_step(c: &Crop, day: i64, turns_per_day: i64) -> i64 {
    if c.ongoing {
        -1
    } else {
        (day + c.max_yield_day + 1) * turns_per_day
    }
}

/// `_new_plant`'s initial yield: ongoing crops start at 0, others at 1.
pub fn initial_yield(c: &Crop) -> i64 {
    if c.ongoing {
        0
    } else {
        1
    }
}

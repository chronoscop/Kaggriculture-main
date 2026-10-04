//! Game state types and constant tables, a 1:1 mirror of
//! `../child_exp00/fast_env.py` (itself validated against the reference engine).
//!
//! Items share one id space so sheds/inventories can be flat arrays:
//! 0..9 = products (market goods), 9..12 = animals (held in shed before placement).

use std::sync::Arc;

pub const N_ITEMS: usize = 12;
pub const N_PRODUCTS: usize = 9;
pub const N_CROPS: usize = 5;
pub const N_ANIMALS: usize = 3;
pub const N_SHOPS: usize = 8;

pub const ITEM_NAMES: [&str; N_ITEMS] = [
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
    "GOOSE",
    "COW",
    "SHEEP",
];

pub const WHEAT: usize = 0;
pub const FERTILIZER: usize = 8;
pub const FIRST_ANIMAL: usize = 9;

pub fn item_id(name: &str) -> Option<usize> {
    ITEM_NAMES.iter().position(|n| *n == name)
}

pub fn crop_id(name: &str) -> Option<usize> {
    item_id(name).filter(|&i| i < N_CROPS)
}

pub struct CropData {
    pub seed: i64,
    pub first_yield_day: i64,
    pub max_yield_day: i64,
    pub interval: i64,
    pub max_yield: i64,
    pub ongoing: bool,
}

/// Indexed by crop id (= item id 0..5).
pub const CROPS: [CropData; N_CROPS] = [
    CropData {
        seed: 10,
        first_yield_day: 2,
        max_yield_day: 4,
        interval: 0,
        max_yield: 6,
        ongoing: false,
    }, // WHEAT
    CropData {
        seed: 20,
        first_yield_day: 2,
        max_yield_day: 3,
        interval: 0,
        max_yield: 4,
        ongoing: false,
    }, // CARROT
    CropData {
        seed: 50,
        first_yield_day: 8,
        max_yield_day: 8,
        interval: 1,
        max_yield: 4,
        ongoing: true,
    }, // TOMATO
    CropData {
        seed: 100,
        first_yield_day: 10,
        max_yield_day: 10,
        interval: 2,
        max_yield: 4,
        ongoing: true,
    }, // STRAWBERRY
    CropData {
        seed: 80,
        first_yield_day: 10,
        max_yield_day: 12,
        interval: 0,
        max_yield: 6,
        ongoing: false,
    }, // MELON
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Structure {
    Coop,
    Pasture,
}

impl Structure {
    pub fn name(self) -> &'static str {
        match self {
            Structure::Coop => "COOP",
            Structure::Pasture => "PASTURE",
        }
    }
}

pub struct AnimalData {
    pub cost: i64,
    pub structure: Structure,
    pub first_yield_day: i64,
    pub interval: i64,
    pub max_held: i64,
    pub product: usize,
}

/// Indexed by animal id (= item id - FIRST_ANIMAL).
pub const ANIMALS: [AnimalData; N_ANIMALS] = [
    AnimalData {
        cost: 300,
        structure: Structure::Coop,
        first_yield_day: 4,
        interval: 1,
        max_held: 4,
        product: 5,
    }, // GOOSE -> EGG
    AnimalData {
        cost: 400,
        structure: Structure::Pasture,
        first_yield_day: 8,
        interval: 2,
        max_held: 6,
        product: 6,
    }, // COW -> MILK
    AnimalData {
        cost: 500,
        structure: Structure::Pasture,
        first_yield_day: 6,
        interval: 3,
        max_held: 6,
        product: 7,
    }, // SHEEP -> WOOL
];

pub const PRICE_FLOOR: i64 = 1;
pub const MARKET_I0: i64 = 10_000;

pub const LAND_ORDER: [&str; 3] = ["NE", "SW", "SE"];
pub const LAND_PRICES: [i64; 3] = [1000, 2000, 4000];

/// Shop names in the reference SHOPS dict order (unlock bookkeeping uses this id).
pub const SHOP_NAMES: [&str; N_SHOPS] = [
    "BAKERY",
    "PIZZA_SHOP",
    "BRUNCH_SPOT",
    "YARN_STORE",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "SMOOTHIE_SHOP",
    "FARMERS_MARKET",
];

/// Product ids consumed by each shop, same order as the reference lists.
pub const SHOP_PRODUCTS: [&[usize]; N_SHOPS] = [
    &[5, 0],       // BAKERY: EGG, WHEAT
    &[6, 2, 0],    // PIZZA_SHOP: MILK, TOMATO, WHEAT
    &[5, 0, 3],    // BRUNCH_SPOT: EGG, WHEAT, STRAWBERRY
    &[7],          // YARN_STORE: WOOL
    &[3, 6, 0],    // ICE_CREAM_SHOP: STRAWBERRY, MILK, WHEAT
    &[1],          // PET_CAFE: CARROT
    &[3, 6],       // SMOOTHIE_SHOP: STRAWBERRY, MILK
    &[0, 1, 2, 3], // FARMERS_MARKET: WHEAT, CARROT, TOMATO, STRAWBERRY
];

/// Shop ids in alphabetical name order — the reference draws the daily unlock
/// from `sorted(SHOPS)`.
pub const SHOPS_SORTED: [usize; N_SHOPS] = [0, 2, 7, 4, 5, 1, 6, 3];

/// Maximum number of shop instances the town will ever unlock. Shops are drawn
/// with replacement, so this caps total count, not variety.
pub const MAX_SHOP_INSTANCES: usize = 8;

pub const REMAINING_OVERAGE_TIME: i64 = 60;

/// Python value that keeps its int-vs-float type: the parity harness compares
/// obs scalars by type, so overridden market params must round-trip exactly.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Num {
    Int(i64),
    Float(f64),
}

impl Num {
    pub fn f(self) -> f64 {
        match self {
            Num::Int(i) => i as f64,
            Num::Float(f) => f,
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum Func {
    Linear,
    Sq,
    Sqrt,
    Log,
    Log10,
    /// 1.32.7: linear in x/T below the knee, quadratic above it, with f(T) == 1.
    Hinge,
    /// Unknown func string: the reference `_shape` falls back to identity.
    Other(String),
}

/// Quadratic gain applied past the hinge knee (reference `HINGE_GAIN`).
pub const HINGE_GAIN: f64 = 8.0;

impl Func {
    pub fn parse(s: &str) -> Func {
        match s {
            "linear" => Func::Linear,
            "sq" => Func::Sq,
            "sqrt" => Func::Sqrt,
            "log" => Func::Log,
            "log10" => Func::Log10,
            "hinge" => Func::Hinge,
            _ => Func::Other(s.to_string()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Func::Linear => "linear",
            Func::Sq => "sq",
            Func::Sqrt => "sqrt",
            Func::Log => "log",
            Func::Log10 => "log10",
            Func::Hinge => "hinge",
            Func::Other(s) => s,
        }
    }

    pub fn shape(&self, x: f64, t: f64) -> f64 {
        let x = x.max(0.0);
        match self {
            Func::Linear | Func::Other(_) => x,
            Func::Sq => x * x,
            Func::Sqrt => x.sqrt(),
            Func::Log => (1.0 + x).ln(),
            Func::Log10 => (1.0 + x).log10(),
            Func::Hinge => {
                // Degenerates to linear if T is missing or non-positive.
                if t <= 0.0 {
                    return x;
                }
                let u = x / t;
                u + HINGE_GAIN * (u - 1.0).max(0.0).powi(2)
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MarketParam {
    pub base: Num,
    pub i0: Num,
    pub t: Num,
    pub below_func: Func,
    pub below_target: Num,
    pub above_func: Func,
    pub above_target: Num,
}

pub fn default_market_params() -> [MarketParam; N_PRODUCTS] {
    let p = |base: i64, t: i64, bf: Func, bt: f64, af: Func, at: f64| MarketParam {
        base: Num::Int(base),
        i0: Num::Int(MARKET_I0),
        t: Num::Int(t),
        below_func: bf,
        below_target: Num::Float(bt),
        above_func: af,
        above_target: Num::Float(at),
    };
    [
        p(25, 400, Func::Sqrt, 0.80, Func::Log, 0.20), // WHEAT
        p(35, 450, Func::Hinge, 1.00, Func::Sqrt, 0.70), // CARROT
        p(60, 200, Func::Hinge, 0.40, Func::Sqrt, 0.60), // TOMATO
        p(120, 100, Func::Sqrt, 0.70, Func::Linear, 1.60), // STRAWBERRY
        p(250, 300, Func::Log, 0.20, Func::Sq, 3.60),  // MELON
        p(50, 332, Func::Hinge, 0.40, Func::Log, 0.20), // EGG
        p(160, 122, Func::Sqrt, 0.60, Func::Linear, 1.60), // MILK
        p(200, 105, Func::Log, 0.20, Func::Sq, 3.20),  // WOOL
        p(100, 200, Func::Linear, 0.40, Func::Linear, 0.40), // FERTILIZER
    ]
}

#[derive(Clone, Debug, PartialEq)]
pub struct Plant {
    pub crop: usize,
    pub planted_day: i64,
    pub watered_today: bool,
    pub consecutive_unwatered: i64,
    pub yield_units: i64,
    pub max_lifespan_step: i64,
    pub fertilized_until_day: i64,
}

impl Plant {
    pub fn new(crop: usize, day: i64, turns_per_day: i64) -> Plant {
        let cd = &CROPS[crop];
        Plant {
            crop,
            planted_day: day,
            watered_today: false,
            // Planting day counts as unwatered.
            consecutive_unwatered: 1,
            yield_units: if cd.ongoing { 0 } else { 1 },
            max_lifespan_step: if cd.ongoing {
                -1
            } else {
                (day + cd.max_yield_day + 1) * turns_per_day
            },
            fertilized_until_day: -1,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AnimalTile {
    pub animal: usize, // index into ANIMALS
    pub placed_day: i64,
    pub yield_units: i64,
    pub consecutive_unfed: i64,
    pub fed_today: bool,
    pub cared_today: bool,
    pub fertilizer_available: bool,
    pub pending_care_bonus: i64,
}

impl AnimalTile {
    pub fn new(animal: usize, day: i64) -> AnimalTile {
        AnimalTile {
            animal,
            placed_day: day,
            yield_units: 0,
            consecutive_unfed: 0,
            fed_today: false,
            cared_today: false,
            fertilizer_available: false,
            pending_care_bonus: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Tile {
    Empty,
    Locked,
    Weed,
    Structure(Structure),
    Plant(Plant),
    Animal(AnimalTile),
}

/// Per-farmer inventory. Python dicts preserve insertion order, and the shed
/// drop-off order (which decides what overflows when capacity runs out) depends
/// on it, so this is an insertion-ordered assoc list, not a hash map.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Inventory(pub Vec<(usize, i64)>);

impl Inventory {
    pub fn get(&self, item: usize) -> i64 {
        self.0
            .iter()
            .find(|(i, _)| *i == item)
            .map_or(0, |(_, n)| *n)
    }

    pub fn add(&mut self, item: usize, n: i64) {
        match self.0.iter_mut().find(|(i, _)| *i == item) {
            Some((_, have)) => *have += n,
            None => self.0.push((item, n)),
        }
    }

    /// Mirrors `_inv_take`: remove the entry when the count hits zero
    /// (a later re-add appends at the end, like a Python dict re-insert).
    pub fn take(&mut self, item: usize, n: i64) -> bool {
        let Some(pos) = self.0.iter().position(|(i, _)| *i == item) else {
            return false;
        };
        if self.0[pos].1 < n {
            return false;
        }
        self.0[pos].1 -= n;
        if self.0[pos].1 == 0 {
            self.0.remove(pos);
        }
        true
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Farm {
    pub money: f64,
    pub tiles: Vec<Vec<Tile>>,
    pub farmer: (i64, i64),
    pub hands: Vec<(i64, i64)>,
    pub unlocked_quadrants: Vec<&'static str>,
    pub hires_today: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Private {
    pub shed: [i64; N_ITEMS],
    pub seeds: [i64; N_CROPS],
    pub inventories: Vec<Inventory>,
}

impl Private {
    pub fn new() -> Private {
        Private {
            shed: [0; N_ITEMS],
            seeds: [0; N_CROPS],
            inventories: vec![Inventory::default()],
        }
    }
}

impl Default for Private {
    fn default() -> Private {
        Private::new()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Market {
    pub inventory: [i64; N_PRODUCTS],
    pub prices: [Num; N_PRODUCTS],
    pub params: [MarketParam; N_PRODUCTS],
    /// Whether obs carry a "params" key (the reference adds it only when
    /// configuration.marketParams is a non-empty dict).
    pub params_overridden: bool,
}

impl Market {
    pub fn new(overrides: Option<[MarketParam; N_PRODUCTS]>) -> Market {
        let params_overridden = overrides.is_some();
        let params = overrides.unwrap_or_else(default_market_params);
        let inventory = std::array::from_fn(|i| match params[i].i0 {
            Num::Int(v) => v,
            Num::Float(f) => f as i64,
        });
        let prices = std::array::from_fn(|i| params[i].base);
        Market {
            inventory,
            prices,
            params,
            params_overridden,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Town {
    /// Shop ids (SHOP_NAMES index) in unlock order.
    pub unlocked_shops: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub episode_steps: i64,
    pub board_size: i64,
    pub starting_money: i64,
    pub max_orders: usize,
    pub turns_per_day: i64,
    pub shed_capacity: i64,
    pub weed_chance: f64,
    pub shop_unlock_interval: i64,
    pub shop_sell_interval: i64,
    pub center_sell_interval: i64,
    pub hire_mult: i64,
    /// Fully resolved per-product params when configuration.marketParams was a
    /// non-empty dict; None means defaults (and no "params" key in obs).
    pub market_overrides: Option<[MarketParam; N_PRODUCTS]>,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            episode_steps: 720,
            board_size: 10,
            starting_money: 3000,
            max_orders: 10,
            turns_per_day: 24,
            shed_capacity: 100,
            weed_chance: 0.005,
            shop_unlock_interval: 3,
            shop_sell_interval: 4,
            center_sell_interval: 24,
            hire_mult: 1,
            market_overrides: None,
        }
    }
}

/// One element of an agent-supplied action list, mirroring the Python value
/// space the interpreter actually distinguishes.
#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    Str(String),
    Int(i64),
    Float(f64),
    /// Any other Python object: never matches an op/item and fails int().
    Other,
}

/// `None` = the value was not a Python list (silent no-op, like the reference).
pub type TokenList = Option<Arc<[Token]>>;

#[derive(Clone, Debug, PartialEq)]
pub struct PlayerAction {
    pub farmer: TokenList,
    pub hands: Vec<TokenList>,
    pub market: Vec<TokenList>,
}

impl PlayerAction {
    /// Equivalent of a non-dict / empty action: farmer defaults to ["PASS"].
    pub fn empty() -> PlayerAction {
        PlayerAction {
            farmer: Some(vec![Token::Str("PASS".to_string())].into()),
            hands: Vec::new(),
            market: Vec::new(),
        }
    }
}

pub fn token_str(t: &Token) -> Option<&str> {
    match t {
        Token::Str(s) => Some(s),
        _ => None,
    }
}

/// Python `int(x)` for the values agents can send; None where int() would fail.
pub fn token_int(t: &Token) -> Option<i64> {
    match t {
        Token::Int(i) => Some(*i),
        Token::Float(f) => Some(f.trunc() as i64),
        Token::Str(s) => s.trim().parse::<i64>().ok(),
        Token::Other => None,
    }
}

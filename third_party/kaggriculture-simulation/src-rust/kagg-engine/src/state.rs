//! Engine state, mirroring the Python interpreter's observation EXACTLY.
//!
//! Layout mirrors the observation the Python interpreter emits, because the
//! differential harness compares STATE, not just final rewards: an engine that
//! agrees on the bank while disagreeing on a tile has a bug that will surface
//! on a different seed. Field names match the Python keys deliberately, so a
//! mismatch is greppable across both implementations.
//!
//! Shapes that are easy to get wrong, encoded explicitly:
//!   * `tiles` is a 10x10 NESTED grid, `tiles[y][x]`, and a cell is one of
//!     None / "LOCKED" / weed / plant / structure(+animal) -- a flat walk
//!     silently reports zero crops and zero animals.
//!   * `private` is PER SEAT. The opponent's farm is public; their shed,
//!     seeds and hand inventories are not.
//!   * Farmer inventories are Python dicts, and Python dicts iterate in
//!     INSERTION order. That order is load-bearing: `DROP` and the end-of-day
//!     shed sweep deposit items in it, and when shed capacity binds, WHICH
//!     item overflows depends on it. So `OMap` preserves insertion order
//!     rather than sorting -- a BTreeMap here would be a silent behaviour
//!     divergence, not a style choice.

pub const BOARD: i64 = 10;
pub const TURNS_PER_DAY: i64 = 24;
pub const EPISODE_STEPS: i64 = 720;
/// The terminal step. The official runner solicits actions for steps
/// `0..=FINAL_STEP-1` (719 actions) and scores the step-`FINAL_STEP` state:
/// the interpreter marks DONE once the pre-step counter reaches
/// `episodeSteps - 2`. A driver that applies a 720th action (and the day-30
/// end-of-day it triggers) banks differently for last-turn actors.
pub const FINAL_STEP: i64 = EPISODE_STEPS - 1;
pub const SHED_CAP: i64 = 100;
pub const MAX_MARKET_ORDERS: usize = 10;
pub const STARTING_MONEY: f64 = 3000.0;
pub const WEED_CHANCE: f64 = 0.005;
pub const TOWN_SHOP_SELL_INTERVAL: i64 = 4;
pub const TOWN_CENTER_SELL_INTERVAL: i64 = 24;
pub const TOWN_SHOP_UNLOCK_INTERVAL: i64 = 3;

/// PRODUCTS in the interpreter's declaration order (also the shed/seeds/market
/// dict insertion order, which the digest reproduces).
pub const PRODUCTS: [&str; 9] = [
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
];
pub const ANIMAL_NAMES: [&str; 3] = ["GOOSE", "COW", "SHEEP"];
pub const CROP_NAMES: [&str; 5] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"];

/// An insertion-ordered String->i64 map, replicating a Python dict.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OMap(pub Vec<(String, i64)>);

impl OMap {
    pub fn get(&self, k: &str) -> i64 {
        self.0
            .iter()
            .find(|(n, _)| n == k)
            .map(|(_, v)| *v)
            .unwrap_or(0)
    }
    /// `d[k] = d.get(k, 0) + n` -- appends the key on first touch.
    pub fn add(&mut self, k: &str, n: i64) {
        if let Some(e) = self.0.iter_mut().find(|(nm, _)| nm == k) {
            e.1 += n;
        } else {
            self.0.push((k.to_string(), n));
        }
    }
    /// `d[k] -= n` WITHOUT delete-on-zero (the shed/seeds behaviour).
    pub fn sub(&mut self, k: &str, n: i64) {
        self.add(k, -n);
    }
    /// `_inv_take`: return false if short; DELETE the key when it hits zero
    /// (the farmer-inventory behaviour -- the deletion changes iteration
    /// order for later deposits, so it must be replicated, not approximated).
    pub fn take(&mut self, k: &str, n: i64) -> bool {
        let Some(i) = self.0.iter().position(|(nm, _)| nm == k) else {
            return false;
        };
        if self.0[i].1 < n {
            return false;
        }
        self.0[i].1 -= n;
        if self.0[i].1 == 0 {
            self.0.remove(i);
        }
        true
    }
    pub fn remove(&mut self, k: &str) {
        self.0.retain(|(nm, _)| nm != k);
    }
    pub fn sum(&self) -> i64 {
        self.0.iter().map(|(_, v)| v).sum()
    }
    pub fn seeded(keys: &[&str]) -> Self {
        OMap(keys.iter().map(|k| (k.to_string(), 0)).collect())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Cell {
    Empty,
    Locked,
    Weed,
    Plant {
        crop: String,
        planted_day: i64,
        watered_today: bool,
        consecutive_unwatered: i64,
        yield_units: i64,
        max_lifespan_step: i64,
        fertilized_until_day: i64,
    },
    /// A COOP or PASTURE, empty or holding an animal. `kind` is the structure
    /// string; the animal block matches `_new_animal`'s fields.
    Structure {
        kind: String,
        animal: Option<AnimalTile>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct AnimalTile {
    pub animal: String,
    pub placed_day: i64,
    pub yield_units: i64,
    pub consecutive_unfed: i64,
    pub fed_today: bool,
    pub cared_today: bool,
    pub fertilizer_available: bool,
    pub pending_care_bonus: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Farm {
    pub money: f64,
    /// (x, y) -- the interpreter stores [x, y] and indexes tiles[y][x].
    pub farmer: (i64, i64),
    /// (x, y) per hand -- POSITIONALLY aligned with the action's `hands`.
    pub hands: Vec<(i64, i64)>,
    pub hires_today: i64,
    pub unlocked_quadrants: Vec<String>,
    pub tiles: Vec<Vec<Cell>>,
}

impl Farm {
    pub fn new() -> Self {
        let tiles = (0..BOARD)
            .map(|y| {
                (0..BOARD)
                    .map(|x| {
                        if crate::rules::quadrant_of(x, y, BOARD) == "NW" {
                            Cell::Empty
                        } else {
                            Cell::Locked
                        }
                    })
                    .collect()
            })
            .collect();
        Farm {
            money: STARTING_MONEY,
            farmer: default_spawn(),
            hands: Vec::new(),
            hires_today: 0,
            unlocked_quadrants: vec!["NW".to_string()],
            tiles,
        }
    }
}

impl Default for Farm {
    fn default() -> Self {
        Self::new()
    }
}

/// `_default_spawn`: first shed-access tile inside NW.
pub fn default_spawn() -> (i64, i64) {
    for t in crate::rules::shed_access_tiles(BOARD) {
        if crate::rules::quadrant_of(t.0, t.1, BOARD) == "NW" {
            return t;
        }
    }
    (0, 0)
}

/// Per-seat hidden state. Never expose the opponent's copy to a policy.
#[derive(Clone, Debug, PartialEq)]
pub struct Private {
    pub shed: OMap,
    pub seeds: OMap,
    /// inventories[0] = main farmer; hands appended each hire, reset daily.
    pub inventories: Vec<OMap>,
}

impl Private {
    pub fn new() -> Self {
        let mut shed_keys: Vec<&str> = PRODUCTS.to_vec();
        shed_keys.extend_from_slice(&ANIMAL_NAMES);
        Private {
            shed: OMap::seeded(&shed_keys),
            seeds: OMap::seeded(&CROP_NAMES),
            inventories: vec![OMap::default()],
        }
    }
}

impl Default for Private {
    fn default() -> Self {
        Self::new()
    }
}

/// The single shared market. Price is a function of standing inventory, so both
/// farms' order flow moves the same curve -- which is why one player's selling
/// changes the other's economics.
#[derive(Clone, Debug, PartialEq)]
pub struct Market {
    pub inventory: OMap,
    pub prices: OMap,
}

impl Market {
    pub fn new() -> Self {
        let mut inv = OMap::default();
        let mut prices = OMap::default();
        for p in crate::market::PARAMS.iter() {
            inv.add(p.item, p.i0 as i64);
            prices.add(p.item, p.base as i64);
        }
        Market {
            inventory: inv,
            prices,
        }
    }
}

impl Default for Market {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Town {
    pub unlocked_shops: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct State {
    pub step: i64,
    pub seed: i64,
    pub farms: [Farm; 2],
    pub private: [Private; 2],
    pub market: Market,
    pub town: Town,
}

impl State {
    pub fn new(seed: i64) -> Self {
        State {
            step: 0,
            seed,
            farms: [Farm::new(), Farm::new()],
            private: [Private::new(), Private::new()],
            market: Market::new(),
            town: Town::default(),
        }
    }

    pub fn day(&self) -> i64 {
        self.step / TURNS_PER_DAY
    }

    /// Canonical full-state digest, compared against the Python side at EVERY
    /// step by the differential suite (src-python/kaggsim/fidelity.py). Money is emitted as its IEEE-754 bit
    /// pattern -- a decimal rendering can round-trip differently on the two
    /// sides and fake a divergence. Maps are emitted SORTED so
    /// that Python's insertion order (behaviour-relevant, replicated in OMap)
    /// never leaks into the comparison format itself.
    pub fn digest(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("t{}", self.step));
        for (i, f) in self.farms.iter().enumerate() {
            s.push_str(&format!(
                "|f{i}:m{};p{},{};h{};r{};q{}",
                f.money.to_bits(),
                f.farmer.0,
                f.farmer.1,
                f.hands
                    .iter()
                    .map(|(x, y)| format!("{x},{y}"))
                    .collect::<Vec<_>>()
                    .join(" "),
                f.hires_today,
                f.unlocked_quadrants.join(",")
            ));
            s.push(';');
            for row in &f.tiles {
                for c in row {
                    match c {
                        Cell::Empty => s.push('.'),
                        Cell::Locked => s.push('L'),
                        Cell::Weed => s.push('W'),
                        Cell::Plant {
                            crop,
                            planted_day,
                            watered_today,
                            consecutive_unwatered,
                            yield_units,
                            max_lifespan_step,
                            fertilized_until_day,
                        } => s.push_str(&format!(
                            "[P:{},{},{},{},{},{},{}]",
                            crop,
                            planted_day,
                            *watered_today as i64,
                            consecutive_unwatered,
                            yield_units,
                            max_lifespan_step,
                            fertilized_until_day
                        )),
                        Cell::Structure { kind, animal: None } => {
                            s.push_str(&format!("[S:{kind}]"))
                        }
                        Cell::Structure {
                            kind,
                            animal: Some(a),
                        } => s.push_str(&format!(
                            "[A:{},{},{},{},{},{},{},{},{}]",
                            kind,
                            a.animal,
                            a.placed_day,
                            a.yield_units,
                            a.consecutive_unfed,
                            a.fed_today as i64,
                            a.cared_today as i64,
                            a.fertilizer_available as i64,
                            a.pending_care_bonus
                        )),
                    }
                }
            }
        }
        for (i, p) in self.private.iter().enumerate() {
            s.push_str(&format!(
                "|s{i}:{};{};{}",
                fmt_sorted(&p.shed),
                fmt_sorted(&p.seeds),
                p.inventories
                    .iter()
                    .map(fmt_sorted)
                    .collect::<Vec<_>>()
                    .join("/")
            ));
        }
        s.push_str(&format!(
            "|mk:{};{}",
            fmt_sorted(&self.market.inventory),
            fmt_sorted(&self.market.prices)
        ));
        s.push_str(&format!("|tw:{}", self.town.unlocked_shops.join(",")));
        s
    }
}

/// `k=v` pairs sorted by key; zero entries included only if the key exists.
fn fmt_sorted(m: &OMap) -> String {
    let mut rows: Vec<&(String, i64)> = m.0.iter().collect();
    rows.sort();
    rows.iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(",")
}

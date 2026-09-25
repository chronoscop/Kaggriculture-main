//! Built-in synthetic policies (test fixtures and load generators).
//!
//! These are NOT agents: they exist to exercise the engine and to generate
//! cheap games for tooling tests and throughput runs. Deterministic given
//! their seed and the sequence of states they see.

use crate::engine::PlayerAction;
use crate::state::{State, CROP_NAMES, PRODUCTS};
use crate::tape::parse_action_line;

/// xorshift64* -- a tiny deterministic RNG for fixtures.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }

    pub fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len() as u64) as usize]
    }

    /// Uniform float in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

pub const KINDS: [&str; 3] = ["idle", "chaos", "random"];

const MOVES: [&str; 4] = ["NORTH", "SOUTH", "EAST", "WEST"];
const SIMPLE: [&str; 11] = [
    "WATER",
    "HARVEST",
    "DROP",
    "FEED",
    "CARE",
    "COLLECT_FERTILIZER",
    "FERTILIZE",
    "DIG",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "PASS",
];

pub struct Policy {
    kind: String,
    rng: Rng,
}

impl Policy {
    pub fn new(kind: &str, seed: u64) -> Result<Self, String> {
        if !KINDS.contains(&kind) {
            return Err(format!(
                "unknown builtin policy {kind:?}; choose from {KINDS:?}"
            ));
        }
        Ok(Policy {
            kind: kind.to_string(),
            rng: Rng::new(seed),
        })
    }

    /// The action as a tape line (the same encoding Python agents produce).
    pub fn act_line(&mut self, st: &State, seat: usize) -> String {
        match self.kind.as_str() {
            "idle" => "PASS\t\t".to_string(),
            "chaos" => self.chaos(),
            _ => self.random(st, seat),
        }
    }

    pub fn act(&mut self, st: &State, seat: usize) -> PlayerAction {
        parse_action_line(&self.act_line(st, seat))
    }

    fn unit(&mut self) -> String {
        let r = self.rng.below(100);
        if r < 30 {
            self.rng.pick(&MOVES).to_string()
        } else if r < 45 {
            format!("PLANT {}", self.rng.pick(&CROP_NAMES))
        } else {
            self.rng.pick(&SIMPLE).to_string()
        }
    }

    fn chaos(&mut self) -> String {
        let farmer = self.unit();
        let nh = self.rng.below(4);
        let hands: Vec<String> = (0..nh)
            .map(|_| {
                if self.rng.below(10) == 0 {
                    String::new()
                } else {
                    self.unit()
                }
            })
            .collect();
        let no = self.rng.below(5);
        let orders: Vec<String> = (0..no)
            .map(|_| match self.rng.below(8) {
                0 => format!(
                    "BUY_SEED {} {}",
                    self.rng.pick(&CROP_NAMES),
                    1 + self.rng.below(5)
                ),
                1 | 2 => format!(
                    "SELL {} {}",
                    self.rng.pick(&PRODUCTS),
                    1 + self.rng.below(20)
                ),
                3 => format!("BUY_PRODUCT WHEAT {}", 1 + self.rng.below(3)),
                4 => "HIRE".to_string(),
                5 => "BUY_LAND".to_string(),
                6 => String::new(),
                _ => "NONSENSE X 1".to_string(),
            })
            .collect();
        format!("{farmer}\t{}\t{}", hands.join(";"), orders.join(";"))
    }

    fn random(&mut self, st: &State, seat: usize) -> String {
        let me = seat.min(1);
        let farmer = self.unit();
        let hands: Vec<String> = (0..st.farms[me].hands.len()).map(|_| self.unit()).collect();
        let mut orders: Vec<String> = Vec::new();
        if self.rng.below(2) == 0 {
            let stocked: Vec<&str> = PRODUCTS
                .iter()
                .copied()
                .filter(|p| st.private[me].shed.get(p) > 0)
                .collect();
            if let Some(p) = stocked.first() {
                orders.push(format!("SELL {p} {}", st.private[me].shed.get(p)));
            }
            if st.farms[me].money > 400.0 {
                orders.push(format!(
                    "BUY_SEED {} {}",
                    self.rng.pick(&CROP_NAMES),
                    1 + self.rng.below(3)
                ));
            }
        }
        format!("{farmer}\t{}\t{}", hands.join(";"), orders.join(";"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine;

    #[test]
    fn deterministic_and_playable() {
        for kind in KINDS {
            let run = || {
                let mut st = State::new(5);
                let mut a = Policy::new(kind, 1).unwrap();
                let mut b = Policy::new(kind, 2).unwrap();
                while st.step < crate::state::FINAL_STEP {
                    let x = a.act(&st, 0);
                    let y = b.act(&st, 1);
                    engine::step(&mut st, &[x, y]);
                }
                st.digest()
            };
            assert_eq!(run(), run(), "{kind} must be deterministic");
        }
        assert!(Policy::new("bogus", 0).is_err());
    }

    #[test]
    fn rng_unit_interval() {
        let mut r = Rng::new(9);
        for _ in 0..1000 {
            let u = r.unit();
            assert!((0.0..1.0).contains(&u));
        }
    }
}

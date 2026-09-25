//! The price model, transcribed from the interpreter.
//!
//! ```text
//! price(inv) = base + sign * amp * f(|inv - I0|)
//!   sign = +1 below I0 (scarcity), -1 above I0 (glut)
//!   amp  = target * base / f(T)
//!   f in {linear, sq, sqrt, log}   -- log is ln(1+x) so f(0) == 0
//! floored at PRICE_FLOOR
//! ```
//!
//! `hinge` (1.32.7) is linear in x/T up to T, then adds a quadratic runaway:
//! `u + 8 * max(0, u - 1)^2` with `u = x / T`.
//!
//! This is the one place a numeric hazard lives: `sqrt`/`ln` are libm calls, and Rust's
//! libm need not agree with CPython's in the last bit. The engine takes
//! `int(round(price))`, which absorbs almost every such difference, but
//! "almost" over 720 steps x 30 days is exactly where drift hides. The
//! differential test sweeps the full inventory range per product and compares
//! the rounded integers, so any residual disagreement is caught rather than
//! assumed away.

pub const MARKET_I0: i64 = 10_000;
pub const PRICE_FLOOR: i64 = 1;

/// The engine version this table reproduces. CARROT/TOMATO/EGG use the
/// "hinge" scarcity shape introduced in kaggle-environments 1.32.7.
pub const ENGINE_VERSION: &str = "1.32.7";

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    Linear,
    Sq,
    Sqrt,
    Log,
    Log10,
    Hinge,
}

/// f(hinge, T, T) == 1 by construction, so `target` keeps its meaning.
pub const HINGE_GAIN: f64 = 8.0;

pub fn shape(f: Shape, x: f64, t: f64) -> f64 {
    let x = if x < 0.0 { 0.0 } else { x };
    match f {
        Shape::Linear => x,
        Shape::Sq => x * x,
        Shape::Sqrt => x.sqrt(),
        Shape::Log => (1.0 + x).ln(),
        Shape::Log10 => (1.0 + x).log10(),
        Shape::Hinge => {
            // Degenerates to linear if T is non-positive (interpreter rule).
            if t <= 0.0 {
                return x;
            }
            let u = x / t;
            let over = if u - 1.0 > 0.0 { u - 1.0 } else { 0.0 };
            // Python evaluates max(...)**2 FIRST, then *8: keep that exact
            // operation order -- f64 multiplication is not associative.
            u + HINGE_GAIN * (over * over)
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MarketParam {
    pub item: &'static str,
    pub base: f64,
    pub i0: f64,
    pub t: f64,
    pub below_func: Shape,
    pub below_target: f64,
    pub above_func: Shape,
    pub above_target: f64,
}

/// MARKET_PARAMS, in the interpreter's declaration order (1.32.7).
pub const PARAMS: [MarketParam; 9] = [
    MarketParam {
        item: "WHEAT",
        base: 25.0,
        i0: 10_000.0,
        t: 400.0,
        below_func: Shape::Sqrt,
        below_target: 0.80,
        above_func: Shape::Log,
        above_target: 0.20,
    },
    MarketParam {
        item: "CARROT",
        base: 35.0,
        i0: 10_000.0,
        t: 450.0,
        below_func: Shape::Hinge,
        below_target: 1.00,
        above_func: Shape::Sqrt,
        above_target: 0.70,
    },
    MarketParam {
        item: "TOMATO",
        base: 60.0,
        i0: 10_000.0,
        t: 200.0,
        below_func: Shape::Hinge,
        below_target: 0.40,
        above_func: Shape::Sqrt,
        above_target: 0.60,
    },
    MarketParam {
        item: "STRAWBERRY",
        base: 120.0,
        i0: 10_000.0,
        t: 100.0,
        below_func: Shape::Sqrt,
        below_target: 0.70,
        above_func: Shape::Linear,
        above_target: 1.60,
    },
    MarketParam {
        item: "MELON",
        base: 250.0,
        i0: 10_000.0,
        t: 300.0,
        below_func: Shape::Log,
        below_target: 0.20,
        above_func: Shape::Sq,
        above_target: 3.60,
    },
    MarketParam {
        item: "EGG",
        base: 50.0,
        i0: 10_000.0,
        t: 332.0,
        below_func: Shape::Hinge,
        below_target: 0.40,
        above_func: Shape::Log,
        above_target: 0.20,
    },
    MarketParam {
        item: "MILK",
        base: 160.0,
        i0: 10_000.0,
        t: 122.0,
        below_func: Shape::Sqrt,
        below_target: 0.60,
        above_func: Shape::Linear,
        above_target: 1.60,
    },
    MarketParam {
        item: "WOOL",
        base: 200.0,
        i0: 10_000.0,
        t: 105.0,
        below_func: Shape::Log,
        below_target: 0.20,
        above_func: Shape::Sq,
        above_target: 3.20,
    },
    MarketParam {
        item: "FERTILIZER",
        base: 100.0,
        i0: 10_000.0,
        t: 200.0,
        below_func: Shape::Linear,
        below_target: 0.40,
        above_func: Shape::Linear,
        above_target: 0.40,
    },
];

pub fn param(item: &str) -> Option<&'static MarketParam> {
    PARAMS.iter().find(|p| p.item == item)
}

/// Raw, unrounded price -- kept separate so a rounding change cannot silently
/// alter the curve, and so the differential test can inspect both.
pub fn price_raw(p: &MarketParam, inventory: f64) -> f64 {
    if inventory < p.i0 {
        let amp = p.below_target * p.base / shape(p.below_func, p.t, p.t);
        p.base + amp * shape(p.below_func, p.i0 - inventory, p.t)
    } else {
        let amp = p.above_target * p.base / shape(p.above_func, p.t, p.t);
        p.base - amp * shape(p.above_func, inventory - p.i0, p.t)
    }
}

/// What the engine actually quotes: `max(PRICE_FLOOR, int(round(price)))`.
///
/// CPython's `round()` is banker's rounding (half to EVEN); Rust's
/// `f64::round()` is half-away-from-zero. They differ on exact .5 values, which
/// a price curve does produce, so this replicates Python's rule rather than
/// using the built-in.
pub fn price(p: &MarketParam, inventory: f64) -> i64 {
    let raw = price_raw(p, inventory);
    let r = round_half_even(raw);
    if r < PRICE_FLOOR as f64 {
        PRICE_FLOOR
    } else {
        r as i64
    }
}

/// Python's `round()` for floats: half to even.
pub fn round_half_even(x: f64) -> f64 {
    let f = x.floor();
    let diff = x - f;
    if diff > 0.5 {
        f + 1.0
    } else if diff < 0.5 {
        f
    } else if (f as i64) % 2 == 0 {
        f
    } else {
        f + 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hinge_is_linear_below_capacity_and_quadratic_above() {
        assert_eq!(shape(Shape::Hinge, 0.0, 450.0), 0.0);
        assert_eq!(shape(Shape::Hinge, 450.0, 450.0), 1.0);
        // u = 2: 2 + 8 * 1^2 = 10
        assert_eq!(shape(Shape::Hinge, 900.0, 450.0), 10.0);
        // T <= 0 degenerates to linear
        assert_eq!(shape(Shape::Hinge, 5.0, 0.0), 5.0);
    }

    #[test]
    fn carrot_uses_the_1_32_7_hinge() {
        let c = param("CARROT").unwrap();
        assert_eq!(c.below_func, Shape::Hinge);
        assert_eq!(c.below_target, 1.0);
        // At I0 the price is the base.
        assert_eq!(price(c, 10_000.0), 35);
        // One capacity below I0: base + target * base = 70.
        assert_eq!(price(c, 10_000.0 - 450.0), 70);
        // Two capacities below I0: base + base * 10 = 385.
        assert_eq!(price(c, 10_000.0 - 900.0), 385);
        for item in ["TOMATO", "EGG"] {
            assert_eq!(param(item).unwrap().below_func, Shape::Hinge);
        }
    }

    #[test]
    fn python_bankers_rounding() {
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(3.5), 4.0);
        assert_eq!(round_half_even(-0.4), -0.0);
        assert_eq!(round_half_even(7.51), 8.0);
    }
}

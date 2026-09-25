//! Read-only engine diagnostics used by the differential tests: RNG probe,
//! weed draws, quoted prices, raw price sweeps and the pure rule tables.

use kagg_engine::mt19937::MT;
use kagg_engine::{engine, market, rules};

const SHOPS: [&str; 8] = engine::SHOPS_SORTED;

pub fn rng_probe(seed: i64, day: i64) {
    let mut a = MT::for_day(seed, day);
    // Transmit the IEEE-754 bit pattern, not a decimal rendering, so the
    // comparison against CPython is exact.
    let randoms: Vec<String> = (0..8).map(|_| a.random().to_bits().to_string()).collect();
    let mut b = MT::for_day(seed, day);
    let bits: Vec<String> = (0..4).map(|_| b.getrandbits(32).to_string()).collect();
    let mut c = MT::for_day(seed, day);
    let choices: Vec<String> = (0..4)
        .map(|_| format!("\"{}\"", c.choice(&SHOPS)))
        .collect();
    println!(
        "{{\"day\": {}, \"first8_random\": [{}], \"getrandbits32_first4\": [{}], \"choice_first4\": [{}]}}",
        day,
        randoms.join(", "),
        bits.join(", "),
        choices.join(", ")
    );
}

/// The weed draw for one day: one `random()` per empty tile, per player, in the
/// interpreter's iteration order. Emitting the raw sequence lets the harness
/// compare against Python without needing the board state.
pub fn weeds(seed: i64, day: i64, n: usize) {
    let mut r = MT::for_day(seed, day);
    let vals: Vec<String> = (0..n).map(|_| r.random().to_bits().to_string()).collect();
    println!("[{}]", vals.join(", "));
}

/// Every product's quoted price at one inventory level.
pub fn prices(inv: f64) {
    let parts: Vec<String> = market::PARAMS
        .iter()
        .map(|p| format!("\"{}\": {}", p.item, market::price(p, inv)))
        .collect();
    println!("{{{}}}", parts.join(", "));
}

/// Raw and quoted price across a range, for the differential sweep. Raw is
/// emitted as a bit pattern so float comparison is exact.
pub fn price_sweep(item: &str, lo: f64, hi: f64, step: f64) {
    let p = match market::param(item) {
        Some(p) => p,
        None => {
            eprintln!("unknown item {item}");
            std::process::exit(2);
        }
    };
    let mut rows: Vec<String> = Vec::new();
    let mut inv = lo;
    while inv <= hi {
        rows.push(format!(
            "[{}, {}, {}]",
            inv,
            market::price_raw(p, inv).to_bits(),
            market::price(p, inv)
        ));
        inv += step;
    }
    println!("[{}]", rows.join(", "));
}

/// Dump every pure rule function/table so the differential test can compare
/// them against the interpreter without re-implementing anything in Python.
pub fn rules_dump() {
    let fibs: Vec<String> = (0..25).map(|n| rules::fib(n).to_string()).collect();
    let hires: Vec<String> = (0..15)
        .map(|n| rules::hire_cost(n, rules::FARM_HAND_COST_MULT).to_string())
        .collect();
    let mut quads: Vec<String> = Vec::new();
    for y in 0..10i64 {
        for x in 0..10i64 {
            quads.push(format!("\"{}\"", rules::quadrant_of(x, y, 10)));
        }
    }
    let shed: Vec<String> = rules::shed_access_tiles(10)
        .iter()
        .map(|(x, y)| format!("[{x}, {y}]"))
        .collect();
    let crops: Vec<String> = rules::CROPS
        .iter()
        .map(|c| format!(
            "\"{}\": {{\"seed\": {}, \"first_yield_day\": {}, \"max_yield_day\": {}, \"interval\": {}, \"max_yield\": {}, \"ongoing\": {}}}",
            c.name, c.seed_cost, c.first_yield_day, c.max_yield_day,
            c.interval, c.max_yield, c.ongoing))
        .collect();
    let animals: Vec<String> = rules::ANIMALS
        .iter()
        .map(|a| format!(
            "\"{}\": {{\"cost\": {}, \"structure\": \"{}\", \"first_yield_day\": {}, \"interval\": {}, \"max_held\": {}, \"product\": \"{}\"}}",
            a.name, a.cost, a.structure, a.first_yield_day, a.interval,
            a.max_held, a.product))
        .collect();
    println!(
        "{{\"fib\": [{}], \"hire_cost\": [{}], \"quadrants\": [{}], \"shed_access\": [{}], \"land_order\": [\"{}\"], \"land_prices\": [{}], \"max_shop_instances\": {}, \"crops\": {{{}}}, \"animals\": {{{}}}}}",
        fibs.join(", "), hires.join(", "), quads.join(", "), shed.join(", "),
        rules::LAND_ORDER.join("\", \""),
        rules::LAND_PRICES.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "),
        rules::MAX_SHOP_INSTANCES,
        crops.join(", "), animals.join(", "));
}

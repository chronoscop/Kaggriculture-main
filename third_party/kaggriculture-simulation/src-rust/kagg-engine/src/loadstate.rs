//! STATE INJECTION: parse the [`crate::obsjson::json_state`] format back
//! into a `State`, so a caller can continue an episode from a mid-game state
//! rather than from a seed (agents never see the episode seed:
//! `configuration.seed` is None). This is the exact inverse of `json_state`:
//!
//! ```text
//! state_from_json(parse(&json_state(&st, false))).digest() == st.digest()
//! ```
//!
//! is the round-trip law the unit tests enforce. The one field the JSON does
//! not carry is the RNG seed: an optional `"seed"` key supplies it (default
//! 0). With a wrong seed, a rollout sees a RESAMPLED random future (weed
//! spawns, shop unlocks); the market math stays deterministic in the actions.

use crate::json::Json;
use crate::state::{AnimalTile, Cell, Farm, Market, OMap, Private, State, Town, BOARD, FINAL_STEP};

fn omap_from(j: &Json) -> OMap {
    let mut v: Vec<(String, i64)> = Vec::new();
    if j.is_obj() {
        for (k, val) in j.obj() {
            v.push((k.clone(), val.i64()));
        }
    }
    OMap(v)
}

fn pair_from(j: &Json) -> (i64, i64) {
    (j.idx(0).i64(), j.idx(1).i64())
}

fn cell_from(j: &Json) -> Result<Cell, String> {
    if j.is_null() {
        return Ok(Cell::Empty);
    }
    if !j.is_obj() {
        // the only bare-string tile is "LOCKED"
        return match j.str() {
            "LOCKED" => Ok(Cell::Locked),
            other => Err(format!("unknown tile literal {other:?}")),
        };
    }
    let kind = j.get("kind").str().to_string();
    match kind.as_str() {
        "WEED" => Ok(Cell::Weed),
        "PLANT" => Ok(Cell::Plant {
            crop: j.get("crop").str().to_string(),
            planted_day: j.get("planted_day").i64(),
            watered_today: j.get("watered_today").bool(),
            consecutive_unwatered: j.get("consecutive_unwatered").i64(),
            yield_units: j.get("yield_units").i64(),
            max_lifespan_step: j.get("max_lifespan_step").i64(),
            fertilized_until_day: j.get("fertilized_until_day").i64(),
        }),
        _ => {
            let animal = if j.get("animal").is_null() {
                None
            } else {
                Some(AnimalTile {
                    animal: j.get("animal").str().to_string(),
                    placed_day: j.get("placed_day").i64(),
                    yield_units: j.get("yield_units").i64(),
                    consecutive_unfed: j.get("consecutive_unfed").i64(),
                    fed_today: j.get("fed_today").bool(),
                    cared_today: j.get("cared_today").bool(),
                    fertilizer_available: j.get("fertilizer_available").bool(),
                    pending_care_bonus: j.get("pending_care_bonus").i64(),
                })
            };
            Ok(Cell::Structure { kind, animal })
        }
    }
}

fn in_board(p: (i64, i64)) -> bool {
    (0..BOARD).contains(&p.0) && (0..BOARD).contains(&p.1)
}

/// Rebuild one farm; rejects shapes the step function could not index.
fn farm_from(j: &Json) -> Result<Farm, String> {
    if !j.is_obj() {
        return Err("farm must be an object".into());
    }
    let rows = j.get("tiles").arr();
    if rows.len() != BOARD as usize {
        return Err(format!("tiles must have {BOARD} rows"));
    }
    let mut tiles: Vec<Vec<Cell>> = Vec::new();
    for row in rows {
        if row.arr().len() != BOARD as usize {
            return Err(format!("every tile row must have {BOARD} cells"));
        }
        let mut r: Vec<Cell> = Vec::new();
        for c in row.arr() {
            r.push(cell_from(c)?);
        }
        tiles.push(r);
    }
    let farmer = pair_from(j.get("farmer"));
    let hands: Vec<(i64, i64)> = j.get("hands").arr().iter().map(pair_from).collect();
    if !in_board(farmer) || !hands.iter().all(|h| in_board(*h)) {
        return Err("farmer or hand position outside the board".into());
    }
    Ok(Farm {
        money: j.get("money").f64(),
        farmer,
        hands,
        hires_today: j.get("hires_today").i64(),
        unlocked_quadrants: j
            .get("unlocked_quadrants")
            .arr()
            .iter()
            .map(|q| q.str().to_string())
            .collect(),
        tiles,
    })
}

fn private_from(j: &Json) -> Private {
    let mut inventories: Vec<OMap> = j.get("inventories").arr().iter().map(omap_from).collect();
    if inventories.is_empty() {
        inventories.push(OMap::default());
    }
    Private {
        shed: omap_from(j.get("shed")),
        seeds: omap_from(j.get("seeds")),
        inventories,
    }
}

/// Parse a full-state JSON; returns `Err` (never panics later in the step
/// function) for shapes the engine cannot play: wrong farm count, a board
/// that is not 10x10, positions off the board, or a step outside
/// `0..=FINAL_STEP`.
pub fn state_from_json(j: &Json) -> Result<State, String> {
    let farms_j = j.get("farms");
    let priv_j = j.get("private");
    if farms_j.arr().len() != 2 || priv_j.arr().len() != 2 {
        return Err("state needs exactly 2 farms and 2 private blocks".into());
    }
    let step = j.get("step").i64();
    if !(0..=FINAL_STEP).contains(&step) {
        return Err(format!("step must be in 0..={FINAL_STEP}"));
    }
    let seed = if j.get("seed").is_null() {
        0
    } else {
        j.get("seed").i64()
    };
    Ok(State {
        step,
        seed,
        farms: [farm_from(farms_j.idx(0))?, farm_from(farms_j.idx(1))?],
        private: [private_from(priv_j.idx(0)), private_from(priv_j.idx(1))],
        market: Market {
            inventory: omap_from(j.get("market").get("inventory")),
            prices: omap_from(j.get("market").get("prices")),
        },
        town: Town {
            unlocked_shops: j
                .get("town")
                .get("unlocked_shops")
                .arr()
                .iter()
                .map(|s| s.str().to_string())
                .collect(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine;
    use crate::tape::parse_action_line;

    #[test]
    fn json_round_trip_preserves_the_digest() {
        let mut st = State::new(424242);
        let script = [
            "PASS\t\tBUY_SEED WHEAT 4;HIRE;BUY_ANIMAL GOOSE 1",
            "SOUTH\tNORTH\tBUY_PRODUCT FERTILIZER 2",
            "PLANT WHEAT\tWEST\t",
            "WATER\t\tSELL WHEAT 1",
            "BUILD_COOP\t\t",
        ];
        for k in 0..60 {
            let a = parse_action_line(script[k % script.len()]);
            let b = parse_action_line(script[(k + 2) % script.len()]);
            engine::step(&mut st, &[a, b]);
        }
        let js = crate::obsjson::json_state(&st, false);
        let mut back = state_from_json(&crate::json::parse(&js).unwrap()).unwrap();
        back.seed = st.seed;
        assert_eq!(back.digest(), st.digest());
    }

    #[test]
    fn malformed_states_are_errors_not_panics() {
        let st = State::new(3);
        let good = crate::obsjson::json_state(&st, false);
        assert!(state_from_json(&crate::json::parse(&good).unwrap()).is_ok());
        let bad = [
            "{}".to_string(),
            good.replace("\"step\": 0", "\"step\": 9999"),
            good.replace("\"farmer\": [4, 4]", "\"farmer\": [40, 4]"),
            good.replacen("[null, ", "[", 1),
        ];
        for b in bad {
            let j = crate::json::parse(&b).unwrap();
            assert!(state_from_json(&j).is_err(), "{}", &b[..60.min(b.len())]);
        }
    }
}

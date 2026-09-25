//! Per-seat observation JSON -> engine `State`.
//!
//! An agent receives the OFFICIAL interpreter's per-seat observation. This
//! rebuilds as much engine state from it as the observation reveals:
//!
//!   1. `private[1 - me]` is UNKNOWN. The interpreter hides the opponent's
//!      shed, seeds and carried inventories, so the block is left EMPTY;
//!      callers that need an estimate must supply their own.
//!   2. `seed` is a PLACEHOLDER (0). The episode seed is not in the
//!      observation, so weed spawns and the shop draw of any rollout from
//!      this state are a resampled future, not the real one.
//!   3. Everything else is EXACT -- both boards, both moneys, the shared
//!      market inventory and prices, and the unlocked shop list.
//!
//! `OMap` insertion order is behaviour-relevant on the engine side (DROP and
//! the dusk shed sweep deposit in it, and which item overflows a full shed
//! depends on it), so maps are rebuilt in the JSON's own key order rather than
//! sorted. `json.rs` preserves object order for exactly this reason.

use crate::json::Json;
use crate::state::{AnimalTile, Cell, Farm, Market, OMap, Private, State, Town, BOARD};

fn omap(v: &Json) -> OMap {
    let mut m = OMap::default();
    for (k, val) in v.obj() {
        m.0.push((k.clone(), val.i64()));
    }
    m
}

fn xy(v: &Json) -> (i64, i64) {
    (v.idx(0).i64(), v.idx(1).i64())
}

fn cell(tl: &Json) -> Cell {
    match tl {
        Json::Null => Cell::Empty,
        Json::Str(s) if s == "LOCKED" => Cell::Locked,
        Json::Obj(_) => {
            let kind = tl.get("kind").str().to_string();
            if !tl.get("animal").is_null() {
                return Cell::Structure {
                    kind,
                    animal: Some(AnimalTile {
                        animal: tl.get("animal").str().to_string(),
                        placed_day: tl.get("placed_day").i64(),
                        yield_units: tl.get("yield_units").i64(),
                        consecutive_unfed: tl.get("consecutive_unfed").i64(),
                        fed_today: tl.get("fed_today").bool(),
                        cared_today: tl.get("cared_today").bool(),
                        fertilizer_available: tl.get("fertilizer_available").bool(),
                        pending_care_bonus: tl.get("pending_care_bonus").i64(),
                    }),
                };
            }
            match kind.as_str() {
                "WEED" => Cell::Weed,
                "PLANT" => Cell::Plant {
                    crop: tl.get("crop").str().to_string(),
                    planted_day: tl.get("planted_day").i64(),
                    watered_today: tl.get("watered_today").bool(),
                    consecutive_unwatered: tl.get("consecutive_unwatered").i64(),
                    yield_units: tl.get("yield_units").i64(),
                    max_lifespan_step: tl.get("max_lifespan_step").i64(),
                    fertilized_until_day: tl.get("fertilized_until_day").i64(),
                },
                "" => Cell::Empty,
                _ => Cell::Structure { kind, animal: None },
            }
        }
        // Anything else is a schema we do not understand. An unknown tile
        // treated as EMPTY would invite the planner to plant on it; LOCKED is
        // the conservative reading (the planner simply ignores the tile).
        _ => Cell::Locked,
    }
}

fn farm(v: &Json) -> Farm {
    let mut f = Farm::new();
    f.money = v.get("money").f64();
    f.farmer = xy(v.get("farmer"));
    f.hands = v.get("hands").arr().iter().map(xy).collect();
    f.hires_today = v.get("hires_today").i64();
    let q: Vec<String> = v
        .get("unlocked_quadrants")
        .arr()
        .iter()
        .map(|s| s.str().to_string())
        .collect();
    if !q.is_empty() {
        f.unlocked_quadrants = q;
    }
    let rows = v.get("tiles").arr();
    if rows.len() == BOARD as usize && rows.iter().all(|r| r.arr().len() == BOARD as usize) {
        f.tiles = rows
            .iter()
            .map(|row| row.arr().iter().map(cell).collect())
            .collect();
    }
    f
}

/// Rebuild the engine state from one per-seat observation.
///
/// Returns `None` when the observation is not the expected shape, so a
/// caller never simulates a state it invented.
pub fn from_obs(obs: &Json) -> Option<(State, usize)> {
    let farms = obs.get("farms").arr();
    if farms.len() != 2 {
        return None;
    }
    let me = obs.get("player").i64().clamp(0, 1) as usize;
    let mut st = State::new(0);
    st.step = obs.get("step").i64().max(0);
    st.farms = [farm(&farms[0]), farm(&farms[1])];
    let on_board = |p: &(i64, i64)| (0..BOARD).contains(&p.0) && (0..BOARD).contains(&p.1);
    if !st
        .farms
        .iter()
        .all(|f| on_board(&f.farmer) && f.hands.iter().all(on_board))
    {
        return None;
    }

    let mkt = obs.get("market");
    let inv = omap(mkt.get("inventory"));
    let prices = omap(mkt.get("prices"));
    if inv.0.is_empty() || prices.0.is_empty() {
        return None;
    }
    st.market = Market {
        inventory: inv,
        prices,
    };

    st.town = Town {
        unlocked_shops: obs
            .get("town")
            .get("unlocked_shops")
            .arr()
            .iter()
            .map(|s| s.str().to_string())
            .collect(),
    };

    // This seat's private block is observed; the opponent's is NOT (left
    // empty on purpose).
    let p = obs.get("private");
    let mut mine = Private::new();
    if p.is_obj() {
        mine.shed = omap(p.get("shed"));
        mine.seeds = omap(p.get("seeds"));
        let invs: Vec<OMap> = p.get("inventories").arr().iter().map(omap).collect();
        if !invs.is_empty() {
            mine.inventories = invs;
        }
    }
    st.private[me] = mine;
    st.private[1 - me] = Private::new();
    // `Private::new()` seeds every product key at 0; zero the shed explicitly
    // so a future default cannot leak a non-empty opponent shed.
    for e in st.private[1 - me].shed.0.iter_mut() {
        e.1 = 0;
    }
    Some((st, me))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine;
    use crate::json;
    use crate::obsjson::seat_obs_json;
    use crate::tape::parse_action_line;

    #[test]
    fn rebuilds_public_state_and_own_private_only() {
        let mut st = State::new(8);
        let a = parse_action_line("PLANT WHEAT\t\tBUY_SEED WHEAT 3;HIRE");
        let b = parse_action_line("SOUTH\t\tBUY_PRODUCT WHEAT 2");
        for _ in 0..30 {
            engine::step(&mut st, &[a.clone(), b.clone()]);
        }
        let obs = json::parse(&seat_obs_json(&st, 1)).unwrap();
        let (got, me) = from_obs(&obs).unwrap();
        assert_eq!(me, 1);
        assert_eq!(got.step, st.step);
        assert_eq!(got.farms, st.farms);
        assert_eq!(got.market, st.market);
        assert_eq!(got.town, st.town);
        assert_eq!(got.private[1], st.private[1]);
        assert_eq!(got.private[0].shed.sum(), 0);
        assert!(from_obs(&json::parse("{}").unwrap()).is_none());
    }
}

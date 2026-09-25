//! Full-state JSON in the official observation schema.
//!
//! Field names and nesting match the interpreter's observation, so the same
//! agent code parses both. Both seats' `private` blocks are included; a
//! driver must hand each seat only its own (see `docs/serve-protocol.md`).

use crate::state::{Cell, Farm, OMap, Private, State};

/// The body of a JSON string literal (escaped, without the quotes).
pub fn json_escape(s: &str) -> String {
    let q = crate::json::quote(s);
    q[1..q.len() - 1].to_string()
}

pub fn json_omap(m: &OMap) -> String {
    let rows: Vec<String> =
        m.0.iter()
            .map(|(k, v)| format!("\"{}\": {}", json_escape(k), v))
            .collect();
    format!("{{{}}}", rows.join(", "))
}

pub fn json_tile(c: &Cell) -> String {
    match c {
        Cell::Empty => "null".to_string(),
        Cell::Locked => "\"LOCKED\"".to_string(),
        Cell::Weed => "{\"kind\": \"WEED\"}".to_string(),
        Cell::Plant {
            crop,
            planted_day,
            watered_today,
            consecutive_unwatered,
            yield_units,
            max_lifespan_step,
            fertilized_until_day,
        } => format!(
            "{{\"kind\": \"PLANT\", \"crop\": \"{}\", \"planted_day\": {}, \
             \"watered_today\": {}, \"consecutive_unwatered\": {}, \
             \"yield_units\": {}, \"max_lifespan_step\": {}, \
             \"fertilized_until_day\": {}}}",
            crop,
            planted_day,
            watered_today,
            consecutive_unwatered,
            yield_units,
            max_lifespan_step,
            fertilized_until_day
        ),
        Cell::Structure { kind, animal: None } => format!("{{\"kind\": \"{kind}\"}}"),
        Cell::Structure {
            kind,
            animal: Some(a),
        } => format!(
            "{{\"kind\": \"{}\", \"animal\": \"{}\", \"placed_day\": {}, \
             \"yield_units\": {}, \"consecutive_unfed\": {}, \
             \"fed_today\": {}, \"cared_today\": {}, \
             \"fertilizer_available\": {}, \"pending_care_bonus\": {}}}",
            kind,
            a.animal,
            a.placed_day,
            a.yield_units,
            a.consecutive_unfed,
            a.fed_today,
            a.cared_today,
            a.fertilizer_available,
            a.pending_care_bonus
        ),
    }
}

pub fn json_farm(f: &Farm) -> String {
    let hands: Vec<String> = f.hands.iter().map(|(x, y)| format!("[{x}, {y}]")).collect();
    let tiles: Vec<String> = f
        .tiles
        .iter()
        .map(|row| {
            let cells: Vec<String> = row.iter().map(json_tile).collect();
            format!("[{}]", cells.join(", "))
        })
        .collect();
    let quads: Vec<String> = f
        .unlocked_quadrants
        .iter()
        .map(|q| format!("\"{q}\""))
        .collect();
    format!(
        "{{\"money\": {}, \"farmer\": [{}, {}], \"hands\": [{}], \
         \"hires_today\": {}, \"unlocked_quadrants\": [{}], \"tiles\": [{}]}}",
        f.money,
        f.farmer.0,
        f.farmer.1,
        hands.join(", "),
        f.hires_today,
        quads.join(", "),
        tiles.join(", ")
    )
}

pub fn json_private(p: &Private) -> String {
    let invs: Vec<String> = p.inventories.iter().map(json_omap).collect();
    format!(
        "{{\"shed\": {}, \"seeds\": {}, \"inventories\": [{}]}}",
        json_omap(&p.shed),
        json_omap(&p.seeds),
        invs.join(", ")
    )
}

/// Full-state observation in the official schema, plus both seats' private
/// blocks and a `done` flag. [`crate::loadstate::state_from_json`] is its
/// exact inverse (except the RNG seed, which is not serialised).
pub fn json_state(st: &State, done: bool) -> String {
    let shops: Vec<String> = st
        .town
        .unlocked_shops
        .iter()
        .map(|s| format!("\"{s}\""))
        .collect();
    format!(
        "{{\"step\": {}, \"day\": {}, \"hour\": {}, \"done\": {}, \
         \"farms\": [{}, {}], \
         \"market\": {{\"inventory\": {}, \"prices\": {}}}, \
         \"town\": {{\"unlocked_shops\": [{}]}}, \
         \"private\": [{}, {}]}}",
        st.step,
        st.day(),
        st.step % crate::state::TURNS_PER_DAY,
        done,
        json_farm(&st.farms[0]),
        json_farm(&st.farms[1]),
        json_omap(&st.market.inventory),
        json_omap(&st.market.prices),
        shops.join(", "),
        json_private(&st.private[0]),
        json_private(&st.private[1])
    )
}

/// `remainingOverageTime` as the official runner reports it to an agent
/// that has not used any overage.
pub const REMAINING_OVERAGE_TIME: i64 = 60;

/// The per-seat observation exactly as the official runner hands it to the
/// agent in `seat`: shared public fields plus ONLY that seat's `private`.
pub fn seat_obs_json(st: &State, seat: usize) -> String {
    let shops: Vec<String> = st
        .town
        .unlocked_shops
        .iter()
        .map(|s| format!("\"{}\"", json_escape(s)))
        .collect();
    format!(
        "{{\"remainingOverageTime\": {}, \"step\": {}, \"day\": {}, \
         \"hour\": {}, \"player\": {}, \"farms\": [{}, {}], \
         \"market\": {{\"inventory\": {}, \"prices\": {}}}, \
         \"town\": {{\"unlocked_shops\": [{}]}}, \"private\": {}}}",
        REMAINING_OVERAGE_TIME,
        st.step,
        st.day(),
        st.step % crate::state::TURNS_PER_DAY,
        seat,
        json_farm(&st.farms[0]),
        json_farm(&st.farms[1]),
        json_omap(&st.market.inventory),
        json_omap(&st.market.prices),
        shops.join(", "),
        json_private(&st.private[seat.min(1)])
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json;

    #[test]
    fn seat_obs_carries_only_its_own_private() {
        let mut st = State::new(1);
        st.private[0].shed.add("WHEAT", 7);
        st.private[1].shed.add("WHEAT", 9);
        let o0 = json::parse(&seat_obs_json(&st, 0)).unwrap();
        let o1 = json::parse(&seat_obs_json(&st, 1)).unwrap();
        assert_eq!(o0.get("player").i64(), 0);
        assert_eq!(o1.get("player").i64(), 1);
        assert_eq!(o0.get("private").get("shed").get("WHEAT").i64(), 7);
        assert_eq!(o1.get("private").get("shed").get("WHEAT").i64(), 9);
        assert_eq!(o0.get("remainingOverageTime").i64(), 60);
        assert_eq!(o0.get("farms").arr().len(), 2);
    }

    #[test]
    fn full_state_has_both_privates_and_done() {
        let st = State::new(1);
        let j = json::parse(&json_state(&st, true)).unwrap();
        assert!(j.get("done").bool());
        assert_eq!(j.get("private").arr().len(), 2);
    }
}

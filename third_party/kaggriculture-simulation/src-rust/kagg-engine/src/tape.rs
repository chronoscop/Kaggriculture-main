//! The tape action format and whole-episode rollouts.
//!
//! One seat's action for one step is a single line:
//!
//! ```text
//! <farmer tokens> TAB <hand;hand;...> TAB <order;order;...>
//! ```
//!
//! Tokens inside an action are space separated. The hands and market fields
//! are POSITIONAL (see [`crate::engine::positional`]): an empty segment is an
//! empty action `[]` and keeps its index. A single-seat tape file is
//! `SEED <int>` followed by one such line per step; the two-seat episode
//! format (`kagg episode`) interleaves seat 0 and seat 1 lines per step.
//! See `docs/tape-format.md`.

use crate::engine::{self, PlayerAction, UnitAction};
use crate::state::{State, FINAL_STEP};

/// A single-seat tape: the seed header plus one action per step.
#[derive(Clone, Debug, Default)]
pub struct Tape {
    pub seed: i64,
    pub steps: Vec<PlayerAction>,
}

fn tokens(s: &str) -> Vec<&str> {
    s.split(' ').filter(|t| !t.is_empty()).collect()
}

/// Parse one tape line into a [`PlayerAction`]. Missing fields are empty.
pub fn parse_action_line(line: &str) -> PlayerAction {
    let line = line.trim_end_matches(['\r', '\n']);
    let mut parts = line.split('\t');
    let farmer = parts.next().unwrap_or("PASS");
    let hands = parts.next().unwrap_or("");
    let market = parts.next().unwrap_or("");
    PlayerAction {
        farmer: UnitAction::parse(&tokens(farmer)),
        hands: engine::positional(hands)
            .into_iter()
            .map(|h| UnitAction::parse(&tokens(h)))
            .collect(),
        market: engine::positional(market)
            .into_iter()
            .map(|o| tokens(o).into_iter().map(str::to_string).collect())
            .collect(),
    }
}

/// Parse the `SEED <n>` header line.
pub fn parse_seed_line(line: &str) -> Result<i64, String> {
    line.trim()
        .strip_prefix("SEED ")
        .ok_or_else(|| "tape must start with 'SEED <n>'".to_string())?
        .trim()
        .parse()
        .map_err(|_| "bad seed".to_string())
}

/// Parse a single-seat tape from its text.
pub fn parse_tape(raw: &str) -> Result<Tape, String> {
    let mut lines = raw.lines();
    let seed = parse_seed_line(lines.next().ok_or("empty tape")?)?;
    let steps = lines.map(parse_action_line).collect();
    Ok(Tape { seed, steps })
}

/// Load a single-seat tape file.
pub fn load_tape(path: &str) -> Result<Tape, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    parse_tape(&raw)
}

/// True once the state is terminal under the official runner's rule.
pub fn is_terminal(st: &State) -> bool {
    st.step >= FINAL_STEP
}

/// Play one full episode from two single-seat tapes and return final banks.
///
/// Applies actions for steps `0..FINAL_STEP` (719 actions, exactly as the
/// official runner) -- a tape line for step 719 is ignored. Steps past the
/// end of a tape are empty actions.
pub fn play_pair(seed: i64, a: &Tape, b: &Tape) -> (f64, f64) {
    let mut st = State::new(seed);
    let empty = PlayerAction::default();
    while !is_terminal(&st) {
        let i = st.step as usize;
        let a0 = a.steps.get(i).unwrap_or(&empty).clone();
        let a1 = b.steps.get(i).unwrap_or(&empty).clone();
        engine::step(&mut st, &[a0, a1]);
    }
    (st.farms[0].money, st.farms[1].money)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interior_empty_market_order_keeps_its_index() {
        let a = parse_action_line("PASS\t\tSELL WHEAT 1;;BUY_SEED WHEAT 2");
        assert_eq!(a.market.len(), 3);
        assert!(a.market[1].is_empty());
        assert_eq!(a.market[2][0], "BUY_SEED");
    }

    #[test]
    fn leading_and_trailing_empty_slots_are_kept() {
        let a = parse_action_line("PASS\t;WATER;\t;HIRE");
        assert_eq!(a.hands.len(), 3);
        assert_eq!(a.hands[0].op, "PASS");
        assert_eq!(a.hands[1].op, "WATER");
        assert_eq!(a.hands[2].op, "PASS");
        assert_eq!(a.market.len(), 2);
        assert!(a.market[0].is_empty());
        assert_eq!(a.market[1], vec!["HIRE".to_string()]);
    }

    #[test]
    fn wholly_empty_field_is_zero_entries() {
        let a = parse_action_line("NORTH\t\t");
        assert_eq!(a.farmer.op, "NORTH");
        assert!(a.hands.is_empty());
        assert!(a.market.is_empty());
        let b = parse_action_line("");
        assert_eq!(b.farmer.op, "PASS");
    }

    #[test]
    fn play_pair_applies_exactly_719_actions() {
        // Idle until step 718, then buy one WHEAT seed on steps 718 AND 719.
        // The official runner applies the step-718 action and never solicits
        // step 719, so exactly one purchase may land.
        let mut raw = String::from(
            "SEED 7
",
        );
        for i in 0..720 {
            if i >= 718 {
                raw.push_str(
                    "PASS		BUY_SEED WHEAT 1
",
                );
            } else {
                raw.push_str(
                    "PASS		
",
                );
            }
        }
        let buyer = parse_tape(&raw).unwrap();
        let idle = parse_tape(
            "SEED 7
",
        )
        .unwrap();
        let (b0, b1) = play_pair(7, &buyer, &idle);
        assert_eq!(b1, 3000.0);
        let price = crate::rules::CROPS
            .iter()
            .find(|c| c.name == "WHEAT")
            .unwrap()
            .seed_cost as f64;
        assert_eq!(b0, 3000.0 - price);
    }

    #[test]
    fn seed_header() {
        assert_eq!(parse_seed_line("SEED 42").unwrap(), 42);
        assert!(parse_seed_line("SEEDX").is_err());
    }
}

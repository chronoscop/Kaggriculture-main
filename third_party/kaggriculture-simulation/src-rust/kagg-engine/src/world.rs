//! Realized worlds: the town shops an episode actually unlocks.
//!
//! The per-day RNG draws one `random()` per EMPTY tile for weeds before the
//! shop `choice()`, so the shops depend on the seed AND both action streams.
//! The first shop is drawn while step 71 runs (end of day 2), the second
//! while step 143 runs (end of day 5): actions at steps >= 72 / >= 144 can
//! no longer change them.

use crate::engine::{self, PlayerAction};
use crate::state::{State, TOWN_SHOP_UNLOCK_INTERVAL, TURNS_PER_DAY};

/// First step at which the `k`-th shop (1-based) is already drawn.
pub fn split_step(k: usize) -> i64 {
    k as i64 * TOWN_SHOP_UNLOCK_INTERVAL * TURNS_PER_DAY
}

/// `"SHOP1|SHOP2"` for the first `k` shops, or None when fewer exist.
pub fn world_key(shops: &[String], k: usize) -> Option<String> {
    if k == 0 || shops.len() < k {
        None
    } else {
        Some(shops[..k].join("|"))
    }
}

/// The world realized by two action streams (index = step). Plays only as
/// far as needed to draw `k` shops.
pub fn realized_world(
    seed: i64,
    a: &[PlayerAction],
    b: &[PlayerAction],
    k: usize,
) -> Option<String> {
    let mut st = State::new(seed);
    let stop = split_step(k);
    let empty = PlayerAction::default();
    while st.step < stop {
        let i = st.step as usize;
        let x = a.get(i).unwrap_or(&empty).clone();
        let y = b.get(i).unwrap_or(&empty).clone();
        engine::step(&mut st, &[x, y]);
    }
    world_key(&st.town.unlocked_shops, k)
}

/// The world under an idle (empty-action) drive -- a cheap baseline label.
pub fn idle_world(seed: i64, k: usize) -> Option<String> {
    realized_world(seed, &[], &[], k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::parse_action_line;

    #[test]
    fn split_steps() {
        assert_eq!(split_step(1), 72);
        assert_eq!(split_step(2), 144);
    }

    #[test]
    fn keys() {
        let s = vec!["A".to_string(), "B".to_string()];
        assert_eq!(world_key(&s, 1).as_deref(), Some("A"));
        assert_eq!(world_key(&s, 2).as_deref(), Some("A|B"));
        assert_eq!(world_key(&s, 3), None);
        assert_eq!(world_key(&s, 0), None);
    }

    #[test]
    fn idle_world_matches_a_full_idle_game() {
        for seed in 0..5 {
            let mut st = State::new(seed);
            while st.step < 200 {
                engine::step(&mut st, &[PlayerAction::default(), PlayerAction::default()]);
            }
            assert_eq!(idle_world(seed, 2), world_key(&st.town.unlocked_shops, 2));
        }
    }

    #[test]
    fn actions_after_the_split_cannot_move_the_key() {
        let plant = parse_action_line("PLANT WHEAT\t\tBUY_SEED WHEAT 3");
        let late: Vec<PlayerAction> = (0..300)
            .map(|i| {
                if i >= 72 {
                    plant.clone()
                } else {
                    PlayerAction::default()
                }
            })
            .collect();
        for seed in 0..10 {
            assert_eq!(realized_world(seed, &late, &[], 1), idle_world(seed, 1));
        }
    }
}

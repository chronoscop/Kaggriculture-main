//! `kagg serve`: a line-oriented step environment over stdin/stdout.
//!
//! Every request is one line; every response is one JSON line. Observations
//! use the official interpreter's schema (plus both seats' `private` blocks
//! and a `done` flag). Full specification: `docs/serve-protocol.md`.
//!
//! ```text
//! RESET <seed>                          new episode, both seats caller-driven
//! RESET <seed> OPP <seat> <tapepath>    seat <seat> replays a single-seat tape
//! STEP <line>                           action for the free seat (other seat:
//!                                       tape if OPP given, else empty action)
//! STEP2 <line0>\x1e<line1>              both seats
//! LOADSTATE <json>                      continue from a full-state JSON
//! ROLLOUT <H> <json>\x1e<lines0>\x1e<lines1>   step H times from a state,
//!                                       lines joined by \x1f; returns final obs
//! GENGAME <seed>\x1e<lines0>\x1e<lines1>       whole game in one call
//! QUIT
//! ```
//!
//! Episode end follows the official runner: the state is terminal at step
//! 719 (`FINAL_STEP`), after 719 actions. `done` is true from then on and
//! further STEP/STEP2 requests are no-ops that return the terminal state.

use kagg_engine::engine::{self, PlayerAction};
use kagg_engine::json;
use kagg_engine::loadstate::state_from_json;
use kagg_engine::obsjson::{json_escape, json_omap, json_state};
use kagg_engine::state::{State, FINAL_STEP, TURNS_PER_DAY};
use kagg_engine::tape::{is_terminal, load_tape, parse_action_line, Tape};
use std::io::{BufRead, BufWriter, Write};

fn err(msg: &str) -> String {
    format!("{{\"error\": \"{}\"}}", json_escape(msg))
}

fn obs(st: &State) -> String {
    json_state(st, is_terminal(st))
}

fn split_lines(s: &str) -> Vec<&str> {
    s.split('\x1f').collect()
}

fn line_at<'a>(lines: &[&'a str], i: usize) -> &'a str {
    lines.get(i).copied().unwrap_or("")
}

/// Step from `st` until terminal or `h` steps, seat lines indexed from 0.
fn rollout(st: &mut State, h: usize, l0: &[&str], l1: &[&str]) {
    for i in 0..h {
        if is_terminal(st) {
            break;
        }
        let a0 = parse_action_line(line_at(l0, i));
        let a1 = parse_action_line(line_at(l1, i));
        engine::step(st, &[a0, a1]);
    }
}

/// Compact per-day record for GENGAME.
fn day_record(st: &State) -> String {
    let shops: Vec<String> = st
        .town
        .unlocked_shops
        .iter()
        .map(|s| format!("\"{}\"", json_escape(s)))
        .collect();
    format!(
        "{{\"step\": {}, \"day\": {}, \"money\": [{}, {}], \
         \"unlocked_shops\": [{}], \"inventory\": {}}}",
        st.step,
        st.day(),
        st.farms[0].money,
        st.farms[1].money,
        shops.join(", "),
        json_omap(&st.market.inventory)
    )
}

/// Handle one request line; `None` means QUIT.
pub fn handle(
    line: &str,
    st: &mut Option<State>,
    opp: &mut Option<(usize, Tape)>,
) -> Option<String> {
    let line = line.trim_end_matches(['\r', '\n']);
    if line == "QUIT" {
        return None;
    }
    if let Some(rest) = line.strip_prefix("LOADSTATE ") {
        *opp = None;
        return Some(match json::parse(rest).and_then(|j| state_from_json(&j)) {
            Ok(s) => {
                let r = obs(&s);
                *st = Some(s);
                r
            }
            Err(e) => err(&e),
        });
    }
    if let Some(rest) = line.strip_prefix("ROLLOUT ") {
        let mut head = rest.splitn(2, ' ');
        let h: usize = match head.next().unwrap_or("").parse() {
            Ok(h) => h,
            Err(_) => return Some(err("bad horizon")),
        };
        let mut parts = head.next().unwrap_or("").split('\x1e');
        let snap = parts.next().unwrap_or("");
        let l0 = split_lines(parts.next().unwrap_or(""));
        let l1 = split_lines(parts.next().unwrap_or(""));
        return Some(match json::parse(snap).and_then(|j| state_from_json(&j)) {
            Ok(mut s) => {
                rollout(&mut s, h, &l0, &l1);
                obs(&s)
            }
            Err(e) => err(&e),
        });
    }
    if let Some(rest) = line.strip_prefix("GENGAME ") {
        let mut parts = rest.split('\x1e');
        let seed: i64 = match parts.next().unwrap_or("").trim().parse() {
            Ok(s) => s,
            Err(_) => return Some(err("bad seed")),
        };
        let l0 = split_lines(parts.next().unwrap_or(""));
        let l1 = split_lines(parts.next().unwrap_or(""));
        let mut s = State::new(seed);
        let mut days: Vec<String> = Vec::new();
        while !is_terminal(&s) {
            let i = s.step as usize;
            let a0 = parse_action_line(line_at(&l0, i));
            let a1 = parse_action_line(line_at(&l1, i));
            engine::step(&mut s, &[a0, a1]);
            if s.step % TURNS_PER_DAY == 0 {
                days.push(day_record(&s));
            }
        }
        return Some(format!(
            "{{\"days\": [{}], \"final\": {}}}",
            days.join(", "),
            obs(&s)
        ));
    }
    if let Some(rest) = line.strip_prefix("RESET ") {
        let mut parts = rest.splitn(4, ' ');
        let seed: i64 = match parts.next().unwrap_or("").trim().parse() {
            Ok(s) => s,
            Err(_) => return Some(err("bad seed")),
        };
        *opp = None;
        if parts.next() == Some("OPP") {
            let seat: usize = parts.next().unwrap_or("1").parse().unwrap_or(1);
            match load_tape(parts.next().unwrap_or("")) {
                Ok(t) => *opp = Some((seat.min(1), t)),
                Err(e) => return Some(err(&e)),
            }
        }
        let s = State::new(seed);
        let r = obs(&s);
        *st = Some(s);
        return Some(r);
    }
    let (a0, a1) = if let Some(rest) = line.strip_prefix("STEP2 ") {
        let mut halves = rest.split('\x1e');
        (
            parse_action_line(halves.next().unwrap_or("")),
            parse_action_line(halves.next().unwrap_or("")),
        )
    } else if let Some(rest) = line.strip_prefix("STEP ") {
        let ours = parse_action_line(rest);
        match (st.as_ref(), opp.as_ref()) {
            (Some(s), Some((seat, tape))) => {
                let scripted = tape.steps.get(s.step as usize).cloned().unwrap_or_default();
                if *seat == 0 {
                    (scripted, ours)
                } else {
                    (ours, scripted)
                }
            }
            _ => (ours, PlayerAction::default()),
        }
    } else {
        return Some(err("unknown command"));
    };
    let Some(s) = st.as_mut() else {
        return Some(err("no episode; RESET first"));
    };
    if s.step < FINAL_STEP {
        engine::step(s, &[a0, a1]);
    }
    Some(obs(s))
}

pub fn serve() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let mut st: Option<State> = None;
    let mut opp: Option<(usize, Tape)> = None;
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        // A malformed request must answer with an error, never kill the
        // process: contain any panic from the engine and reset the episode.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            handle(&line, &mut st, &mut opp)
        }));
        let resp = match result {
            Ok(Some(resp)) => resp,
            Ok(None) => break,
            Err(_) => {
                st = None;
                opp = None;
                err("internal error while handling the request; episode reset")
            }
        };
        // The client closed the pipe: stop quietly.
        if writeln!(out, "{resp}").is_err() || out.flush().is_err() {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(cmds: &[&str]) -> Vec<String> {
        let mut st = None;
        let mut opp = None;
        cmds.iter()
            .filter_map(|c| handle(c, &mut st, &mut opp))
            .collect()
    }

    #[test]
    fn step_is_a_noop_after_the_terminal_step() {
        let mut cmds = vec!["RESET 5".to_string()];
        for _ in 0..721 {
            cmds.push("STEP2 PASS\t\tBUY_SEED WHEAT 1\x1ePASS\t\t".to_string());
        }
        let refs: Vec<&str> = cmds.iter().map(String::as_str).collect();
        let out = run(&refs);
        let last = json::parse(out.last().unwrap()).unwrap();
        assert_eq!(last.get("step").i64(), FINAL_STEP);
        assert!(last.get("done").bool());
        assert_eq!(out[out.len() - 1], out[out.len() - 2]);
        assert_eq!(out[719], out[720]);
    }

    #[test]
    fn gengame_matches_stepwise_play() {
        let line0 = "PASS\t\tBUY_SEED WHEAT 1";
        let lines: Vec<&str> = std::iter::repeat_n(line0, 719).collect();
        let joined = lines.join("\x1f");
        let g = run(&[&format!("GENGAME 9\x1e{joined}\x1e")]);
        let gj = json::parse(&g[0]).unwrap();
        let mut cmds = vec!["RESET 9".to_string()];
        for _ in 0..719 {
            cmds.push(format!("STEP2 {line0}\x1e"));
        }
        let refs: Vec<&str> = cmds.iter().map(String::as_str).collect();
        let stepped = run(&refs);
        let sj = json::parse(stepped.last().unwrap()).unwrap();
        assert_eq!(
            gj.get("final").get("farms").idx(0).get("money").f64(),
            sj.get("farms").idx(0).get("money").f64()
        );
        // 29 day boundaries (steps 24..=696) inside a 719-step game.
        assert_eq!(gj.get("days").arr().len(), 29);
    }

    #[test]
    fn loadstate_round_trips_and_rollout_continues() {
        let r = run(&["RESET 3", "STEP2 PASS\t\tHIRE\x1ePASS\t\t"]);
        let snap = &r[1];
        let loaded = run(&[&format!("LOADSTATE {snap}")]);
        assert_eq!(&loaded[0], snap);
        let rolled = run(&[&format!("ROLLOUT 5 {snap}\x1e\x1e")]);
        let rj = json::parse(&rolled[0]).unwrap();
        assert_eq!(rj.get("step").i64(), 6);
    }

    #[test]
    fn malformed_states_are_rejected_before_stepping() {
        let r = run(&["LOADSTATE {}", "STEP PASS\t\t", "ROLLOUT 1 {}\x1e\x1e"]);
        assert!(r.iter().all(|l| l.starts_with("{\"error\"")), "{r:?}");
    }

    #[test]
    fn errors_are_json() {
        let r = run(&["STEP PASS\t\t", "BOGUS", "RESET x"]);
        assert!(r.iter().all(|l| l.starts_with("{\"error\"")));
    }
}

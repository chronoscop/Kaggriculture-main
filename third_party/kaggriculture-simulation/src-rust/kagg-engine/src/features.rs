//! Generic numeric features of one seat's view (no strategy).
//!
//! Only information in the seat's own observation is used (public board,
//! market, town, and the seat's OWN private block). Feature names match
//! `kaggsim.processor.basic_features` so consumers can mix sources. Groups:
//!
//! `time` `money` `market` `shed` `seeds` `carried` `hands` `quadrants`
//! `tiles` `shops`

use crate::engine::SHOPS_SORTED;
use crate::state::{Cell, State, CROP_NAMES, PRODUCTS, TURNS_PER_DAY};

pub const GROUPS: [&str; 10] = [
    "time",
    "money",
    "market",
    "shed",
    "seeds",
    "carried",
    "hands",
    "quadrants",
    "tiles",
    "shops",
];

/// Validate group names.
pub fn check_groups(groups: &[String]) -> Result<(), String> {
    for g in groups {
        if !GROUPS.contains(&g.as_str()) {
            return Err(format!(
                "unknown feature group {g:?}; choose from {GROUPS:?}"
            ));
        }
    }
    Ok(())
}

/// Ordered (name, value) features for `seat`, restricted to `groups`
/// (all groups when empty). Unknown group names are an error.
pub fn extract(st: &State, seat: usize, groups: &[String]) -> Result<Vec<(String, f64)>, String> {
    check_groups(groups)?;
    let on = |g: &str| groups.is_empty() || groups.iter().any(|x| x == g);
    let me = seat.min(1);
    let opp = 1 - me;
    let mut f: Vec<(String, f64)> = Vec::with_capacity(96);
    if on("time") {
        f.push(("step".into(), st.step as f64));
        f.push(("day".into(), st.day() as f64));
        f.push(("hour".into(), (st.step % TURNS_PER_DAY) as f64));
    }
    if on("money") {
        f.push(("money_me".into(), st.farms[me].money));
        f.push(("money_opp".into(), st.farms[opp].money));
    }
    if on("hands") {
        f.push(("hands_me".into(), st.farms[me].hands.len() as f64));
        f.push(("hands_opp".into(), st.farms[opp].hands.len() as f64));
    }
    if on("quadrants") {
        f.push((
            "quadrants_me".into(),
            st.farms[me].unlocked_quadrants.len() as f64,
        ));
        f.push((
            "quadrants_opp".into(),
            st.farms[opp].unlocked_quadrants.len() as f64,
        ));
    }
    for p in PRODUCTS {
        if on("market") {
            f.push((format!("price_{p}"), st.market.prices.get(p) as f64));
            f.push((format!("minv_{p}"), st.market.inventory.get(p) as f64));
        }
        if on("shed") {
            f.push((format!("shed_{p}"), st.private[me].shed.get(p) as f64));
        }
    }
    if on("seeds") {
        for c in CROP_NAMES {
            f.push((format!("seeds_{c}"), st.private[me].seeds.get(c) as f64));
        }
    }
    if on("carried") {
        let total: i64 = st.private[me].inventories.iter().map(|m| m.sum()).sum();
        f.push(("carried_total".into(), total as f64));
    }
    if on("tiles") {
        let names = [
            "empty",
            "locked",
            "weed",
            "plant",
            "structure",
            "animal",
            "ripe",
        ];
        for (who, i) in [("me", me), ("opp", opp)] {
            let mut c = [0i64; 7];
            for row in &st.farms[i].tiles {
                for t in row {
                    match t {
                        Cell::Empty => c[0] += 1,
                        Cell::Locked => c[1] += 1,
                        Cell::Weed => c[2] += 1,
                        Cell::Plant { yield_units, .. } => {
                            c[3] += 1;
                            if *yield_units > 0 {
                                c[6] += 1;
                            }
                        }
                        Cell::Structure { animal: None, .. } => c[4] += 1,
                        Cell::Structure {
                            animal: Some(_), ..
                        } => c[5] += 1,
                    }
                }
            }
            for (n, v) in names.iter().zip(c) {
                f.push((format!("tiles_{n}_{who}"), v as f64));
            }
        }
    }
    if on("shops") {
        for s in SHOPS_SORTED {
            let n = st.town.unlocked_shops.iter().filter(|x| *x == s).count();
            f.push((format!("shop_{s}"), n as f64));
        }
    }
    Ok(f)
}

/// Features as a JSON object.
pub fn to_json(f: &[(String, f64)]) -> String {
    let parts: Vec<String> = f.iter().map(|(k, v)| format!("\"{k}\": {v}")).collect();
    format!("{{{}}}", parts.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn groups_select_and_validate() {
        let st = State::new(3);
        let all = extract(&st, 0, &[]).unwrap();
        assert!(all.len() > 60);
        let t = extract(&st, 1, &["time".into(), "money".into()]).unwrap();
        let names: Vec<&str> = t.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(names, vec!["step", "day", "hour", "money_me", "money_opp"]);
        assert!(extract(&st, 0, &["bogus".into()]).is_err());
    }

    #[test]
    fn features_are_seat_relative() {
        let mut st = State::new(3);
        st.farms[0].money = 10.0;
        st.farms[1].money = 20.0;
        st.private[1].shed.add("EGG", 4);
        let m = |seat| -> HashMap<String, f64> {
            extract(&st, seat, &["money".into(), "shed".into()])
                .unwrap()
                .into_iter()
                .collect()
        };
        assert_eq!(m(0)["money_me"], 10.0);
        assert_eq!(m(1)["money_me"], 20.0);
        assert_eq!(m(0)["shed_EGG"], 0.0);
        assert_eq!(m(1)["shed_EGG"], 4.0);
        let j = to_json(&extract(&st, 0, &["money".into()]).unwrap());
        assert_eq!(j, "{\"money_me\": 10, \"money_opp\": 20}");
    }
}

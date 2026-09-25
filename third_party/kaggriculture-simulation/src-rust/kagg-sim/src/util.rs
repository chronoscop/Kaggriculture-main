//! Small shared helpers.

use kagg_engine::json::{self, Json};

/// 64-bit FNV-1a.
pub fn fnv64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in data {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Parse `"0-9,15,20-22"` into seeds.
pub fn parse_seeds(spec: &str) -> Result<Vec<i64>, String> {
    let mut out = Vec::new();
    for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        if let Some((lo, hi)) = part.split_once('-') {
            let lo: i64 = lo
                .trim()
                .parse()
                .map_err(|_| format!("bad seed range {part:?}"))?;
            let hi: i64 = hi
                .trim()
                .parse()
                .map_err(|_| format!("bad seed range {part:?}"))?;
            if hi < lo {
                return Err(format!("empty seed range {part:?}"));
            }
            out.extend(lo..=hi);
        } else {
            out.push(part.parse().map_err(|_| format!("bad seed {part:?}"))?);
        }
    }
    Ok(out)
}

/// Read and parse a JSON file.
pub fn load_json(path: &str) -> Result<Json, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    json::parse(&raw).map_err(|e| format!("{path}: {e}"))
}

/// Apply `key.path=value` overrides (value parsed as JSON, else a string).
pub fn apply_overrides(cfg: &mut Json, overrides: &[String]) -> Result<(), String> {
    for o in overrides {
        let (k, v) = o
            .split_once('=')
            .ok_or_else(|| format!("override {o:?} must be key=value"))?;
        let value = json::parse(v).unwrap_or_else(|_| Json::Str(v.to_string()));
        cfg.set_path(k.trim(), value);
    }
    Ok(())
}

/// Deep-merge `over` onto `base` (objects merge, everything else replaces).
pub fn merge(base: &Json, over: &Json) -> Json {
    match (base, over) {
        (Json::Obj(b), Json::Obj(o)) => {
            let mut out = b.clone();
            for (k, v) in o {
                match out.iter_mut().find(|(kk, _)| kk == k) {
                    Some(slot) => slot.1 = merge(&slot.1, v),
                    None => out.push((k.clone(), v.clone())),
                }
            }
            Json::Obj(out)
        }
        (_, o) => o.clone(),
    }
}

pub fn str_or<'a>(j: &'a Json, key: &str, default: &'a str) -> &'a str {
    let v = j.get(key);
    if v.is_str() {
        v.str()
    } else {
        default
    }
}

pub fn i64_or(j: &Json, key: &str, default: i64) -> i64 {
    let v = j.get(key);
    if v.is_num() {
        v.i64()
    } else {
        default
    }
}

pub fn f64_or(j: &Json, key: &str, default: f64) -> f64 {
    let v = j.get(key);
    if v.is_num() {
        v.f64()
    } else {
        default
    }
}

pub fn bool_or(j: &Json, key: &str, default: bool) -> bool {
    match j.get(key) {
        Json::Bool(b) => *b,
        _ => default,
    }
}

pub fn strings(j: &Json) -> Vec<String> {
    j.arr().iter().map(|v| v.str().to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeds() {
        assert_eq!(parse_seeds("0-3, 7").unwrap(), vec![0, 1, 2, 3, 7]);
        assert!(parse_seeds("5-1").is_err());
        assert!(parse_seeds("x").is_err());
    }

    #[test]
    fn overrides_and_merge() {
        let mut j = json::parse(r#"{"a": {"b": 1}, "w": 2}"#).unwrap();
        apply_overrides(
            &mut j,
            &["a.b=5".into(), "name=hello".into(), "w=[1,2]".into()],
        )
        .unwrap();
        assert_eq!(j.get("a").get("b").i64(), 5);
        assert_eq!(j.get("name").str(), "hello");
        assert_eq!(j.get("w").arr().len(), 2);
        assert!(apply_overrides(&mut j, &["nokey".into()]).is_err());
        let base = json::parse(r#"{"x": {"y": 1, "z": 2}, "k": 1}"#).unwrap();
        let over = json::parse(r#"{"x": {"y": 9}}"#).unwrap();
        let m = merge(&base, &over);
        assert_eq!(m.get("x").get("y").i64(), 9);
        assert_eq!(m.get("x").get("z").i64(), 2);
        assert_eq!(m.get("k").i64(), 1);
    }

    #[test]
    fn accessors_and_hash() {
        let j = json::parse(r#"{"s": "v", "n": 3, "b": true, "l": ["a", "b"]}"#).unwrap();
        assert_eq!(str_or(&j, "s", "d"), "v");
        assert_eq!(str_or(&j, "n", "d"), "d");
        assert_eq!(i64_or(&j, "n", 0), 3);
        assert_eq!(f64_or(&j, "missing", 1.5), 1.5);
        assert!(bool_or(&j, "b", false));
        assert_eq!(strings(j.get("l")), vec!["a", "b"]);
        assert_ne!(fnv64(b"a"), fnv64(b"b"));
    }
}

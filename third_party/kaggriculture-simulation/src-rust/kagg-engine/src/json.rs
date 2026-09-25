//! A minimal, dependency-free JSON reader.
//!
//! Why hand-rolled rather than serde_json: the engine crate has ZERO
//! dependencies, so it builds offline and reproducibly anywhere.
//!
//! Scope is exactly the observation the interpreter emits: objects, arrays,
//! strings (ASCII tokens, with the standard escapes handled anyway), numbers,
//! `true`/`false`/`null`. Objects preserve insertion order (a Vec, like
//! `state::OMap`) because Python dict order is behaviour-relevant elsewhere in
//! the engine and a silent reorder here would be hard to see.
//!
//! Errors are values, not panics: a malformed line must produce an error
//! response, never abort the process.

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// Object member lookup. Returns `&Json::Null` for a miss so callers can
    /// chain without unwrapping -- a missing key and an explicit null behave
    /// identically, which matches the Python `_get(o, k, d)` helper the
    /// reference planner uses.
    pub fn get(&self, key: &str) -> &Json {
        static NULL: Json = Json::Null;
        match self {
            Json::Obj(m) => m
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v)
                .unwrap_or(&NULL),
            _ => &NULL,
        }
    }

    pub fn idx(&self, i: usize) -> &Json {
        static NULL: Json = Json::Null;
        match self {
            Json::Arr(a) => a.get(i).unwrap_or(&NULL),
            _ => &NULL,
        }
    }

    pub fn arr(&self) -> &[Json] {
        match self {
            Json::Arr(a) => a.as_slice(),
            _ => &[],
        }
    }

    pub fn obj(&self) -> &[(String, Json)] {
        match self {
            Json::Obj(m) => m.as_slice(),
            _ => &[],
        }
    }

    pub fn f64(&self) -> f64 {
        match self {
            Json::Num(n) => *n,
            Json::Bool(true) => 1.0,
            Json::Bool(false) => 0.0,
            _ => 0.0,
        }
    }

    pub fn i64(&self) -> i64 {
        self.f64() as i64
    }

    pub fn bool(&self) -> bool {
        match self {
            Json::Bool(b) => *b,
            Json::Num(n) => *n != 0.0,
            _ => false,
        }
    }

    pub fn str(&self) -> &str {
        match self {
            Json::Str(s) => s.as_str(),
            _ => "",
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }

    pub fn is_obj(&self) -> bool {
        matches!(self, Json::Obj(_))
    }

    pub fn is_arr(&self) -> bool {
        matches!(self, Json::Arr(_))
    }

    pub fn is_str(&self) -> bool {
        matches!(self, Json::Str(_))
    }

    pub fn is_num(&self) -> bool {
        matches!(self, Json::Num(_))
    }

    /// Set `a.b.c` to `value`, creating objects on the way.
    pub fn set_path(&mut self, path: &str, value: Json) {
        let mut node = self;
        let parts: Vec<&str> = path.split('.').collect();
        for (i, part) in parts.iter().enumerate() {
            if !node.is_obj() {
                *node = Json::Obj(Vec::new());
            }
            let Json::Obj(m) = node else { unreachable!() };
            let pos = match m.iter().position(|(k, _)| k == part) {
                Some(p) => p,
                None => {
                    m.push((part.to_string(), Json::Null));
                    m.len() - 1
                }
            };
            if i + 1 == parts.len() {
                m[pos].1 = value;
                return;
            }
            node = &mut m[pos].1;
        }
    }

    /// Serialise (compact, key order preserved).
    pub fn dump(&self) -> String {
        let mut out = String::new();
        self.dump_into(&mut out);
        out
    }

    fn dump_into(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(n) => out.push_str(&num(*n)),
            Json::Str(s) => out.push_str(&quote(s)),
            Json::Arr(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    v.dump_into(out);
                }
                out.push(']');
            }
            Json::Obj(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(&quote(k));
                    out.push_str(": ");
                    v.dump_into(out);
                }
                out.push('}');
            }
        }
    }
}

/// A JSON number: integral values without a fraction, NaN/inf as null.
pub fn num(n: f64) -> String {
    if !n.is_finite() {
        "null".to_string()
    } else if n.fract() == 0.0 && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}

/// A JSON string literal with the standard escapes.
pub fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

struct P<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> P<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn lit(&mut self, s: &str) -> bool {
        if self.b[self.i..].starts_with(s.as_bytes()) {
            self.i += s.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self, depth: u32) -> Result<Json, String> {
        if depth > 64 {
            return Err("json too deep".into());
        }
        self.ws();
        if self.i >= self.b.len() {
            return Err("unexpected end of json".into());
        }
        match self.b[self.i] {
            b'{' => {
                self.i += 1;
                let mut out = Vec::new();
                self.ws();
                if self.i < self.b.len() && self.b[self.i] == b'}' {
                    self.i += 1;
                    return Ok(Json::Obj(out));
                }
                loop {
                    self.ws();
                    let k = match self.value(depth + 1)? {
                        Json::Str(s) => s,
                        _ => return Err("object key must be a string".into()),
                    };
                    self.ws();
                    if self.i >= self.b.len() || self.b[self.i] != b':' {
                        return Err("expected ':'".into());
                    }
                    self.i += 1;
                    let v = self.value(depth + 1)?;
                    out.push((k, v));
                    self.ws();
                    if self.i >= self.b.len() {
                        return Err("unterminated object".into());
                    }
                    match self.b[self.i] {
                        b',' => self.i += 1,
                        b'}' => {
                            self.i += 1;
                            return Ok(Json::Obj(out));
                        }
                        _ => return Err("expected ',' or '}'".into()),
                    }
                }
            }
            b'[' => {
                self.i += 1;
                let mut out = Vec::new();
                self.ws();
                if self.i < self.b.len() && self.b[self.i] == b']' {
                    self.i += 1;
                    return Ok(Json::Arr(out));
                }
                loop {
                    out.push(self.value(depth + 1)?);
                    self.ws();
                    if self.i >= self.b.len() {
                        return Err("unterminated array".into());
                    }
                    match self.b[self.i] {
                        b',' => self.i += 1,
                        b']' => {
                            self.i += 1;
                            return Ok(Json::Arr(out));
                        }
                        _ => return Err("expected ',' or ']'".into()),
                    }
                }
            }
            b'"' => {
                self.i += 1;
                let mut s = String::new();
                loop {
                    if self.i >= self.b.len() {
                        return Err("unterminated string".into());
                    }
                    let c = self.b[self.i];
                    self.i += 1;
                    match c {
                        b'"' => return Ok(Json::Str(s)),
                        b'\\' => {
                            if self.i >= self.b.len() {
                                return Err("bad escape".into());
                            }
                            let e = self.b[self.i];
                            self.i += 1;
                            match e {
                                b'"' => s.push('"'),
                                b'\\' => s.push('\\'),
                                b'/' => s.push('/'),
                                b'b' => s.push('\u{8}'),
                                b'f' => s.push('\u{c}'),
                                b'n' => s.push('\n'),
                                b'r' => s.push('\r'),
                                b't' => s.push('\t'),
                                b'u' => {
                                    if self.i + 4 > self.b.len() {
                                        return Err("bad \\u".into());
                                    }
                                    let hex = std::str::from_utf8(&self.b[self.i..self.i + 4])
                                        .map_err(|_| "bad \\u".to_string())?;
                                    let mut cp = u32::from_str_radix(hex, 16)
                                        .map_err(|_| "bad \\u".to_string())?;
                                    self.i += 4;
                                    // A high surrogate followed by a low one is a
                                    // single non-BMP character (Python ensure_ascii).
                                    if (0xD800..0xDC00).contains(&cp)
                                        && self.b.get(self.i) == Some(&b'\\')
                                        && self.b.get(self.i + 1) == Some(&b'u')
                                        && self.i + 6 <= self.b.len()
                                    {
                                        let lo =
                                            std::str::from_utf8(&self.b[self.i + 2..self.i + 6])
                                                .ok()
                                                .and_then(|h| u32::from_str_radix(h, 16).ok());
                                        if let Some(lo) =
                                            lo.filter(|l| (0xDC00..0xE000).contains(l))
                                        {
                                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                            self.i += 6;
                                        }
                                    }
                                    s.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                                }
                                _ => return Err("bad escape".into()),
                            }
                        }
                        _ => {
                            // Copy the raw byte; multi-byte UTF-8 sequences pass
                            // through unchanged because we rebuild from bytes.
                            let start = self.i - 1;
                            let mut end = self.i;
                            while end < self.b.len() && self.b[end] != b'"' && self.b[end] != b'\\'
                            {
                                end += 1;
                            }
                            s.push_str(
                                std::str::from_utf8(&self.b[start..end])
                                    .map_err(|_| "bad utf8 in string".to_string())?,
                            );
                            self.i = end;
                        }
                    }
                }
            }
            b't' => {
                if self.lit("true") {
                    Ok(Json::Bool(true))
                } else {
                    Err("bad literal".into())
                }
            }
            b'f' => {
                if self.lit("false") {
                    Ok(Json::Bool(false))
                } else {
                    Err("bad literal".into())
                }
            }
            b'n' => {
                if self.lit("null") {
                    Ok(Json::Null)
                } else {
                    Err("bad literal".into())
                }
            }
            b'N' => {
                // Python's json.dumps emits bare NaN/Infinity by default. The
                // observation should never contain them, but a total parser is
                // cheaper than a process that dies on one.
                if self.lit("NaN") {
                    Ok(Json::Num(f64::NAN))
                } else {
                    Err("bad literal".into())
                }
            }
            b'I' => {
                if self.lit("Infinity") {
                    Ok(Json::Num(f64::INFINITY))
                } else {
                    Err("bad literal".into())
                }
            }
            _ => {
                let start = self.i;
                if self.b[self.i] == b'-' {
                    self.i += 1;
                    if self.lit("Infinity") {
                        return Ok(Json::Num(f64::NEG_INFINITY));
                    }
                }
                while self.i < self.b.len()
                    && matches!(
                        self.b[self.i],
                        b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-'
                    )
                {
                    self.i += 1;
                }
                if start == self.i {
                    return Err(format!("unexpected byte {:?}", self.b[start] as char));
                }
                std::str::from_utf8(&self.b[start..self.i])
                    .ok()
                    .and_then(|s| s.parse::<f64>().ok())
                    .map(Json::Num)
                    .ok_or_else(|| "bad number".to_string())
            }
        }
    }
}

pub fn parse(src: &str) -> Result<Json, String> {
    let mut p = P {
        b: src.as_bytes(),
        i: 0,
    };
    let v = p.value(0)?;
    p.ws();
    if p.i != p.b.len() {
        return Err("trailing bytes after json value".into());
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dump_round_trips() {
        let src = r#"{"a": [1, 2.5, "x\ty", null, true], "b": {"c": -3}}"#;
        let j = parse(src).unwrap();
        assert_eq!(parse(&j.dump()).unwrap(), j);
        assert_eq!(j.dump(), src);
    }

    #[test]
    fn surrogate_pairs_decode_to_one_character() {
        // what Python's json.dumps writes for a non-BMP character
        let j = parse(r#""😀""#).unwrap();
        assert_eq!(j.str(), "\u{1F600}");
        let lone = parse(r#""\ud83d x""#).unwrap();
        assert_eq!(lone.str(), "\u{FFFD} x");
    }

    #[test]
    fn set_path_creates_nodes() {
        let mut j = parse(r#"{"a": {"b": 1}}"#).unwrap();
        j.set_path("a.b", Json::Num(2.0));
        j.set_path("x.y.z", Json::Str("q".into()));
        assert_eq!(j.get("a").get("b").i64(), 2);
        assert_eq!(j.get("x").get("y").get("z").str(), "q");
    }

    #[test]
    fn numbers_and_quotes() {
        assert_eq!(num(3.0), "3");
        assert_eq!(num(0.25), "0.25");
        assert_eq!(num(f64::NAN), "null");
        assert_eq!(quote("a\"b\n"), r#""a\"b\n""#);
        assert_eq!(
            parse(&quote("\u{1}\\")).unwrap(),
            Json::Str("\u{1}\\".into())
        );
    }
}

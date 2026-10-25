//! An order-preserving JSON value and Python-compatible serialization.
//!
//! Chat templates are written against Python: `tojson` is `json.dumps(x,
//! ensure_ascii=False)` (", " and ": " separators, keys in insertion order,
//! floats as `repr`), and `| string` is `str(x)` (`True`, `None`, `1e-05`).
//! Request bodies are parsed into [`Json`] so object keys keep the order the
//! client sent, as `json.loads` would give them to transformers.

use std::fmt::Write as _;

use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Str(String),
    Arr(Vec<Json>),
    /// Keys in insertion order; a repeated key keeps its first position and
    /// its last value, like a Python dict.
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn parse(text: &str) -> serde_json::Result<Json> {
        serde_json::from_str(text)
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut Json> {
        match self {
            Json::Obj(pairs) => pairs.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Inserts or replaces `key` (a replaced key keeps its position).
    pub fn insert(&mut self, key: &str, value: Json) {
        if let Json::Obj(pairs) = self {
            obj_insert(pairs, key.to_string(), value);
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    /// `json.dumps(self, ensure_ascii=False)`.
    pub fn dumps(&self) -> String {
        let mut out = String::new();
        self.dump_into(&mut out);
        out
    }

    fn dump_into(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Int(i) => {
                let _ = write!(out, "{i}");
            }
            Json::UInt(u) => {
                let _ = write!(out, "{u}");
            }
            Json::Float(f) => out.push_str(&py_float(*f, true)),
            Json::Str(s) => dump_str(s, out),
            Json::Arr(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    item.dump_into(out);
                }
                out.push(']');
            }
            Json::Obj(pairs) => {
                out.push('{');
                for (i, (k, v)) in pairs.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    dump_str(k, out);
                    out.push_str(": ");
                    v.dump_into(out);
                }
                out.push('}');
            }
        }
    }

    pub fn to_value(&self) -> minijinja::Value {
        use minijinja::Value;
        match self {
            Json::Null => Value::from(()),
            Json::Bool(b) => Value::from(*b),
            Json::Int(i) => Value::from(*i),
            Json::UInt(u) => Value::from(*u),
            Json::Float(f) => Value::from(*f),
            Json::Str(s) => Value::from(s.as_str()),
            Json::Arr(items) => Value::from(items.iter().map(Json::to_value).collect::<Vec<_>>()),
            Json::Obj(pairs) => pairs
                .iter()
                .map(|(k, v)| (Value::from(k.as_str()), v.to_value()))
                .collect::<Value>(),
        }
    }

    /// The template value as Python would hold it, or `None` for values
    /// `json.dumps` rejects (undefined, functions, other objects).
    pub fn from_value(v: &minijinja::Value) -> Option<Json> {
        use minijinja::value::ValueKind;
        Some(match v.kind() {
            ValueKind::None => Json::Null,
            ValueKind::Bool => Json::Bool(v.is_true()),
            ValueKind::Number => {
                if v.is_integer() {
                    if let Some(i) = v.as_i64() {
                        Json::Int(i)
                    } else {
                        Json::UInt(u64::try_from(v.clone()).ok()?)
                    }
                } else {
                    Json::Float(f64::try_from(v.clone()).ok()?)
                }
            }
            ValueKind::String => Json::Str(v.as_str()?.to_string()),
            ValueKind::Seq | ValueKind::Iterable => {
                Json::Arr(v.try_iter().ok()?.map(|x| Json::from_value(&x)).collect::<Option<_>>()?)
            }
            ValueKind::Map => {
                let mut pairs = Vec::new();
                for key in v.try_iter().ok()? {
                    let item = v.get_item(&key).ok()?;
                    // json.dumps turns None/bool/number keys into strings.
                    let k = match Json::from_value(&key)? {
                        Json::Str(s) => s,
                        Json::Null => "null".into(),
                        Json::Bool(b) => if b { "true" } else { "false" }.into(),
                        Json::Int(i) => i.to_string(),
                        Json::UInt(u) => u.to_string(),
                        Json::Float(f) => py_float(f, true),
                        _ => return None,
                    };
                    obj_insert(&mut pairs, k, Json::from_value(&item)?);
                }
                Json::Obj(pairs)
            }
            _ => return None,
        })
    }

    pub fn to_serde(&self) -> serde_json::Value {
        use serde_json::Value;
        match self {
            Json::Null => Value::Null,
            Json::Bool(b) => Value::Bool(*b),
            Json::Int(i) => Value::from(*i),
            Json::UInt(u) => Value::from(*u),
            Json::Float(f) => serde_json::Number::from_f64(*f).map_or(Value::Null, Value::Number),
            Json::Str(s) => Value::String(s.clone()),
            Json::Arr(items) => Value::Array(items.iter().map(Json::to_serde).collect()),
            Json::Obj(pairs) => {
                Value::Object(pairs.iter().map(|(k, v)| (k.clone(), v.to_serde())).collect())
            }
        }
    }
}

fn obj_insert(pairs: &mut Vec<(String, Json)>, key: String, value: Json) {
    if let Some(slot) = pairs.iter_mut().find(|(k, _)| *k == key) {
        slot.1 = value;
    } else {
        pairs.push((key, value));
    }
}

/// A JSON string literal as `json.dumps(..., ensure_ascii=False)` writes it.
pub fn dump_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr(float)` (shortest round-trip digits; exponent form below
/// 1e-4 and from 1e16). `json` selects `json.dumps` spellings of the
/// non-finite values (`NaN`, `Infinity`) over `str`'s (`nan`, `inf`).
pub fn py_float(x: f64, json: bool) -> String {
    if x.is_nan() {
        return if json { "NaN" } else { "nan" }.into();
    }
    if x.is_infinite() {
        let s = if json { "Infinity" } else { "inf" };
        return if x < 0.0 { format!("-{s}") } else { s.into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    // `{:e}` gives the shortest round-trip digits: "-1.2345e-7".
    let e_form = format!("{x:e}");
    let (mantissa, exp) = e_form.split_once('e').expect("exponent form");
    let exp: i32 = exp.parse().expect("exponent");
    let negative = mantissa.starts_with('-');
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let mut out = String::new();
    if negative {
        out.push('-');
    }
    if (-4..16).contains(&exp) {
        if exp < 0 {
            out.push_str("0.");
            for _ in 0..(-exp - 1) {
                out.push('0');
            }
            out.push_str(&digits);
        } else {
            let int_len = exp as usize + 1;
            if digits.len() <= int_len {
                out.push_str(&digits);
                for _ in digits.len()..int_len {
                    out.push('0');
                }
                out.push_str(".0");
            } else {
                out.push_str(&digits[..int_len]);
                out.push('.');
                out.push_str(&digits[int_len..]);
            }
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(out, "e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    out
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Json, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Json;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a JSON value")
            }
            fn visit_unit<E>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }
            fn visit_none<E>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }
            fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Json, D::Error> {
                Json::deserialize(d)
            }
            fn visit_bool<E>(self, b: bool) -> Result<Json, E> {
                Ok(Json::Bool(b))
            }
            fn visit_i64<E>(self, i: i64) -> Result<Json, E> {
                Ok(Json::Int(i))
            }
            fn visit_u64<E>(self, u: u64) -> Result<Json, E> {
                Ok(i64::try_from(u).map_or(Json::UInt(u), Json::Int))
            }
            fn visit_f64<E>(self, f: f64) -> Result<Json, E> {
                Ok(Json::Float(f))
            }
            fn visit_str<E: de::Error>(self, s: &str) -> Result<Json, E> {
                Ok(Json::Str(s.to_string()))
            }
            fn visit_string<E>(self, s: String) -> Result<Json, E> {
                Ok(Json::Str(s))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element::<Json>()? {
                    items.push(item);
                }
                Ok(Json::Arr(items))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
                let mut pairs = Vec::new();
                while let Some((k, v)) = map.next_entry::<String, Json>()? {
                    obj_insert(&mut pairs, k, v);
                }
                Ok(Json::Obj(pairs))
            }
        }
        deserializer.deserialize_any(V)
    }
}

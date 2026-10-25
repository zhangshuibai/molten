//! The checkpoint's own chat template (`chat_template.jinja`), rendered with
//! minijinja the way transformers' `apply_chat_template(..., tokenize=False)`
//! renders it with Jinja2.
//!
//! transformers compiles the template in an `ImmutableSandboxedEnvironment`
//! with `trim_blocks` and `lstrip_blocks`, the `loopcontrols` extension and
//! its `{% generation %}` tag (a pass-through block that only marks assistant
//! spans), replaces `tojson` with `json.dumps(x, ensure_ascii=False)` and adds
//! `raise_exception`. It passes `messages`, `tools`, `documents`,
//! `add_generation_prompt`, the tokenizer's special tokens and any extra
//! keyword arguments. This module reproduces each of those; Python string and
//! dict methods (`.get`, `.items`, `.split`, `.endswith`, ...) come from
//! minijinja-contrib's pycompat layer, and `string` and `trim` follow
//! Python's `str()` and `str.strip()`.

use std::path::Path;

use anyhow::{Context as _, Result, bail};
use minijinja::value::{Value, ValueKind};
use minijinja::{Environment, Error, ErrorKind};

use crate::json::{Json, py_float};

pub struct ChatTemplate {
    env: Environment<'static>,
    /// `bos_token`, `eos_token`, `pad_token`, ... from tokenizer_config.json.
    special_tokens: Vec<(String, String)>,
}

impl ChatTemplate {
    /// Loads `chat_template.jinja` (or `tokenizer_config.json`'s
    /// `chat_template`) and the special-token strings from a model directory.
    pub fn load(dir: &Path) -> Result<Self> {
        let config: serde_json::Value = match std::fs::read(dir.join("tokenizer_config.json")) {
            Ok(bytes) => serde_json::from_slice(&bytes).context("parsing tokenizer_config.json")?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => serde_json::Value::Null,
            Err(e) => return Err(e).context("reading tokenizer_config.json"),
        };
        let path = dir.join("chat_template.jinja");
        let source = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                match config.get("chat_template").and_then(|v| v.as_str()) {
                    Some(text) => text.to_string(),
                    None => bail!("no chat template in {}", dir.display()),
                }
            }
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        let mut special_tokens = Vec::new();
        for key in ["bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token"] {
            let text = match config.get(key) {
                Some(serde_json::Value::String(s)) => Some(s.clone()),
                Some(serde_json::Value::Object(o)) => {
                    o.get("content").and_then(|v| v.as_str()).map(str::to_string)
                }
                _ => None,
            };
            if let Some(text) = text {
                special_tokens.push((key.to_string(), text));
            }
        }
        Self::from_source(&source, special_tokens)
    }

    pub fn from_source(source: &str, special_tokens: Vec<(String, String)>) -> Result<Self> {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson);
        env.add_filter("string", py_string);
        env.add_filter("trim", py_trim);
        // Jinja2's `iterable` is `iter(value)` succeeding: None, booleans and
        // numbers are not iterable (minijinja iterates none as empty).
        env.add_test("iterable", |v: &Value| {
            !matches!(v.kind(), ValueKind::None | ValueKind::Bool | ValueKind::Number)
                && v.try_iter().is_ok()
        });
        env.add_function("raise_exception", |message: String| -> Result<Value, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, message))
        });
        env.add_template_owned("chat", rewrite_generation_tags(source))
            .context("compiling the chat template")?;
        Ok(Self { env, special_tokens })
    }

    /// `apply_chat_template(messages, tools=tools, add_generation_prompt=...,
    /// tokenize=False, **kwargs)`.
    pub fn render(
        &self,
        messages: &[Json],
        tools: Option<&[Json]>,
        add_generation_prompt: bool,
        kwargs: &[(String, Json)],
    ) -> Result<String> {
        let mut ctx: Vec<(String, Value)> = Vec::new();
        let mut set = |k: &str, v: Value| {
            if let Some(slot) = ctx.iter_mut().find(|(key, _)| key == k) {
                slot.1 = v;
            } else {
                ctx.push((k.to_string(), v));
            }
        };
        for (k, v) in &self.special_tokens {
            set(k, Value::from(v.as_str()));
        }
        for (k, v) in kwargs {
            if matches!(k.as_str(), "messages" | "tools" | "documents" | "add_generation_prompt") {
                bail!("chat_template_kwargs may not set {k:?}");
            }
            set(k, v.to_value());
        }
        set("messages", Value::from(messages.iter().map(Json::to_value).collect::<Vec<_>>()));
        set(
            "tools",
            match tools {
                Some(t) => Value::from(t.iter().map(Json::to_value).collect::<Vec<_>>()),
                None => Value::from(()),
            },
        );
        set("documents", Value::from(()));
        set("add_generation_prompt", Value::from(add_generation_prompt));
        let context: Value = ctx.into_iter().collect();
        let template = self.env.get_template("chat")?;
        template.render(context).map_err(|e| anyhow::anyhow!("chat template: {e:#}"))
    }
}

/// transformers' `{% generation %}...{% endgeneration %}` only records where
/// assistant text lies; rendering passes the body through. An always-true
/// `if` with the same whitespace control renders identically.
fn rewrite_generation_tags(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(start) = rest.find("{%") {
        out.push_str(&rest[..start]);
        let tail = &rest[start..];
        let Some(end) = tail.find("%}") else {
            out.push_str(tail);
            return out;
        };
        let tag = &tail[..end + 2];
        let inner = &tag[2..tag.len() - 2];
        let (lead, inner) = match inner.strip_prefix('-') {
            Some(s) => ("-", s),
            None => match inner.strip_prefix('+') {
                Some(s) => ("+", s),
                None => ("", inner),
            },
        };
        let (inner, trail) = match inner.strip_suffix('-') {
            Some(s) => (s, "-"),
            None => match inner.strip_suffix('+') {
                Some(s) => (s, "+"),
                None => (inner, ""),
            },
        };
        match inner.trim() {
            "generation" => out.push_str(&format!("{{%{lead} if true {trail}%}}")),
            "endgeneration" => out.push_str(&format!("{{%{lead} endif {trail}%}}")),
            _ => out.push_str(tag),
        }
        rest = &tail[end + 2..];
    }
    out.push_str(rest);
    out
}

/// `json.dumps(x, ensure_ascii=False)`, transformers' `tojson`. `indent`,
/// `separators` and `sort_keys` are not supported (the template must not
/// pass them).
fn tojson(value: &Value, kwargs: minijinja::value::Kwargs) -> Result<Value, Error> {
    for key in ["indent", "separators", "sort_keys", "ensure_ascii"] {
        if let Ok(Some(v)) = kwargs.get::<Option<Value>>(key)
            && !v.is_none()
            && !(key == "ensure_ascii" && !v.is_true())
        {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("tojson: {key} is not supported"),
            ));
        }
    }
    kwargs.assert_all_used()?;
    let json = Json::from_value(value).ok_or_else(|| {
        Error::new(ErrorKind::InvalidOperation, "tojson: value is not JSON serializable")
    })?;
    Ok(Value::from(json.dumps()))
}

/// Python `str(x)`: `True`/`False`, `None`, float `repr`, `[1, 'a']`.
fn py_string(value: &Value) -> String {
    let mut out = String::new();
    py_str_into(value, false, &mut out);
    out
}

fn py_str_into(value: &Value, quoted: bool, out: &mut String) {
    match value.kind() {
        ValueKind::Undefined => {}
        ValueKind::None => out.push_str("None"),
        ValueKind::Bool => out.push_str(if value.is_true() { "True" } else { "False" }),
        ValueKind::Number if !value.is_integer() => {
            out.push_str(&py_float(f64::try_from(value.clone()).unwrap_or(f64::NAN), false))
        }
        ValueKind::String if quoted => py_repr_str(value.as_str().unwrap_or_default(), out),
        ValueKind::Seq | ValueKind::Iterable => {
            out.push('[');
            if let Ok(iter) = value.try_iter() {
                for (i, item) in iter.enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    py_str_into(&item, true, out);
                }
            }
            out.push(']');
        }
        ValueKind::Map => {
            out.push('{');
            if let Ok(iter) = value.try_iter() {
                for (i, key) in iter.enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    py_str_into(&key, true, out);
                    out.push_str(": ");
                    py_str_into(&value.get_item(&key).unwrap_or_default(), true, out);
                }
            }
            out.push('}');
        }
        _ => out.push_str(&value.to_string()),
    }
}

/// Python `repr(str)` for printable text: single quotes unless the text has a
/// single quote and no double quote.
fn py_repr_str(s: &str, out: &mut String) {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
}

/// Python `str.isspace()` characters (`str.strip()` with no argument).
pub fn is_py_space(c: char) -> bool {
    matches!(c, '\t'..='\r' | '\u{1c}'..='\u{20}' | '\u{85}' | '\u{a0}' | '\u{1680}'
        | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}')
}

/// Jinja2's `trim(value, chars=None)`: `str(value).strip(chars)`.
fn py_trim(value: &Value, chars: Option<String>) -> String {
    let s = py_string(value);
    match chars {
        Some(chars) => s.trim_matches(|c| chars.contains(c)).to_string(),
        None => s.trim_matches(is_py_space).to_string(),
    }
}

//! Output parsing for the chat API: the reasoning split (`<think>...</think>`)
//! and LFM2's Pythonic tool calls (`<|tool_call_start|>[f(a="x", b=1)]
//! <|tool_call_end|>`), both incremental so streamed text is classified as it
//! arrives.
//!
//! The reasoning split follows vLLM's reasoning parsers: generation starts
//! inside the reasoning block when the prompt ends with `<think>` (this
//! model's generation prompt always does); a `<think>` at the very start of
//! the output is also accepted. Everything up to `</think>` is
//! `reasoning_content`, everything after it `content`; without a closing tag
//! the whole output is reasoning.

use crate::json::Json;

pub const THINK_OPEN: &str = "<think>";
pub const THINK_CLOSE: &str = "</think>";
pub const TOOL_OPEN: &str = "<|tool_call_start|>";
pub const TOOL_CLOSE: &str = "<|tool_call_end|>";

#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub name: String,
    /// The arguments as a JSON object string (`json.dumps` style).
    pub arguments: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Piece {
    Reasoning(String),
    Content(String),
    Tools(Vec<ToolCall>),
}

pub struct OutputParser {
    parse_tools: bool,
    in_reasoning: bool,
    /// Nothing classified yet (a leading `<think>` may still arrive).
    at_start: bool,
    in_tool: bool,
    buf: String,
}

impl OutputParser {
    /// `prompt_opens_reasoning`: the prompt ends inside an open `<think>`.
    pub fn new(split_reasoning: bool, parse_tools: bool, prompt_opens_reasoning: bool) -> Self {
        Self {
            parse_tools,
            in_reasoning: split_reasoning && prompt_opens_reasoning,
            at_start: split_reasoning && !prompt_opens_reasoning,
            in_tool: false,
            buf: String::new(),
        }
    }

    pub fn push(&mut self, delta: &str) -> Vec<Piece> {
        self.buf.push_str(delta);
        let mut out = Vec::new();
        self.drain(&mut out, false);
        out
    }

    pub fn finish(&mut self) -> Vec<Piece> {
        let mut out = Vec::new();
        self.drain(&mut out, true);
        out
    }

    fn drain(&mut self, out: &mut Vec<Piece>, eof: bool) {
        loop {
            if self.at_start {
                if self.buf.starts_with(THINK_OPEN) {
                    self.buf.drain(..THINK_OPEN.len());
                    self.at_start = false;
                    self.in_reasoning = true;
                    continue;
                }
                if !eof && THINK_OPEN.starts_with(self.buf.as_str()) {
                    return; // could still become "<think>"
                }
                self.at_start = false;
            }
            if self.in_reasoning {
                if let Some(i) = self.buf.find(THINK_CLOSE) {
                    push_piece(out, Piece::Reasoning(self.buf[..i].to_string()));
                    self.buf.drain(..i + THINK_CLOSE.len());
                    self.in_reasoning = false;
                    continue;
                }
                let cut = if eof { self.buf.len() } else { safe_len(&self.buf, THINK_CLOSE) };
                push_piece(out, Piece::Reasoning(self.buf[..cut].to_string()));
                self.buf.drain(..cut);
                return;
            }
            if !self.parse_tools {
                push_piece(out, Piece::Content(std::mem::take(&mut self.buf)));
                return;
            }
            if self.in_tool {
                let (block, rest) = match self.buf.find(TOOL_CLOSE) {
                    Some(i) => (self.buf[..i].to_string(), Some(i + TOOL_CLOSE.len())),
                    None if eof => (self.buf.clone(), None),
                    None => return,
                };
                match parse_tool_calls(&block) {
                    Some(calls) => push_piece(out, Piece::Tools(calls)),
                    None => {
                        let close = if rest.is_some() { TOOL_CLOSE } else { "" };
                        push_piece(out, Piece::Content(format!("{TOOL_OPEN}{block}{close}")));
                    }
                }
                match rest {
                    Some(r) => drop(self.buf.drain(..r)),
                    None => self.buf.clear(),
                }
                self.in_tool = false;
                continue;
            }
            if let Some(i) = self.buf.find(TOOL_OPEN) {
                push_piece(out, Piece::Content(self.buf[..i].to_string()));
                self.buf.drain(..i + TOOL_OPEN.len());
                self.in_tool = true;
                continue;
            }
            let cut = if eof { self.buf.len() } else { safe_len(&self.buf, TOOL_OPEN) };
            push_piece(out, Piece::Content(self.buf[..cut].to_string()));
            self.buf.drain(..cut);
            return;
        }
    }
}

fn push_piece(out: &mut Vec<Piece>, piece: Piece) {
    match &piece {
        Piece::Reasoning(s) | Piece::Content(s) if s.is_empty() => {}
        _ => out.push(piece),
    }
}

/// Length of the prefix of `buf` that cannot be part of `tag`: everything
/// except the longest suffix that is a proper prefix of `tag`.
fn safe_len(buf: &str, tag: &str) -> usize {
    for (i, _) in buf.char_indices() {
        let suffix = &buf[i..];
        if suffix.len() < tag.len() && tag.starts_with(suffix) {
            return i;
        }
    }
    buf.len()
}

/// The non-streaming result of a whole output.
#[derive(Debug, Default)]
pub struct ParsedOutput {
    pub reasoning: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

impl ParsedOutput {
    pub fn add(&mut self, piece: Piece) {
        match piece {
            Piece::Reasoning(s) => self.reasoning.get_or_insert_with(String::new).push_str(&s),
            Piece::Content(s) => self.content.get_or_insert_with(String::new).push_str(&s),
            Piece::Tools(calls) => self.tool_calls.extend(calls),
        }
    }
}

// ---------------------------------------------------------------- tool calls

/// Parses a Pythonic tool-call list, `[f(a='x', b=1), g()]` (a single call
/// without brackets is accepted too). Arguments must be keyword arguments
/// with literal values: strings, numbers, True/False/None (or
/// true/false/null), lists, tuples and dicts. `None` if the text is anything
/// else.
pub fn parse_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let mut p = Py { s: text.as_bytes(), i: 0, src: text };
    p.ws();
    let mut calls = Vec::new();
    if p.eat(b'[') {
        p.ws();
        if !p.eat(b']') {
            loop {
                calls.push(p.call()?);
                p.ws();
                if p.eat(b',') {
                    p.ws();
                    if p.eat(b']') {
                        break;
                    }
                    continue;
                }
                if p.eat(b']') {
                    break;
                }
                return None;
            }
        }
    } else {
        calls.push(p.call()?);
    }
    p.ws();
    if p.i != p.s.len() || calls.is_empty() {
        return None;
    }
    Some(calls)
}

struct Py<'a> {
    s: &'a [u8],
    i: usize,
    src: &'a str,
}

impl Py<'_> {
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn eat(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn ident(&mut self, extra: &[u8]) -> Option<String> {
        let start = self.i;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == b'_' || extra.contains(&c) || c >= 0x80 {
                self.i += 1;
            } else {
                break;
            }
        }
        if self.i == start || self.s[start].is_ascii_digit() {
            return None;
        }
        Some(self.src[start..self.i].to_string())
    }

    fn call(&mut self) -> Option<ToolCall> {
        let name = self.ident(b".-")?;
        self.ws();
        if !self.eat(b'(') {
            return None;
        }
        let mut args: Vec<(String, Json)> = Vec::new();
        self.ws();
        if !self.eat(b')') {
            loop {
                let key = self.ident(b"")?;
                self.ws();
                if !self.eat(b'=') {
                    return None;
                }
                self.ws();
                let value = self.value()?;
                if let Some(slot) = args.iter_mut().find(|(k, _)| *k == key) {
                    slot.1 = value;
                } else {
                    args.push((key, value));
                }
                self.ws();
                if self.eat(b',') {
                    self.ws();
                    if self.eat(b')') {
                        break;
                    }
                    continue;
                }
                if self.eat(b')') {
                    break;
                }
                return None;
            }
        }
        Some(ToolCall { name, arguments: Json::Obj(args).dumps() })
    }

    fn value(&mut self) -> Option<Json> {
        match self.peek()? {
            b'\'' | b'"' => self.string().map(Json::Str),
            b'[' => self.items(b']').map(Json::Arr),
            b'(' => self.items(b')').map(Json::Arr),
            b'{' => self.dict(),
            b'-' | b'+' | b'0'..=b'9' | b'.' => self.number(),
            _ => {
                let word = self.ident(b"")?;
                match word.as_str() {
                    "True" | "true" => Some(Json::Bool(true)),
                    "False" | "false" => Some(Json::Bool(false)),
                    "None" | "null" => Some(Json::Null),
                    // String prefixes: r'..', u'..'.
                    "r" | "R" | "u" | "U" if matches!(self.peek(), Some(b'\'' | b'"')) => {
                        let raw = word.eq_ignore_ascii_case("r");
                        if raw { self.raw_string() } else { self.string() }.map(Json::Str)
                    }
                    _ => None,
                }
            }
        }
    }

    fn items(&mut self, close: u8) -> Option<Vec<Json>> {
        self.i += 1;
        let mut items = Vec::new();
        self.ws();
        if self.eat(close) {
            return Some(items);
        }
        loop {
            items.push(self.value()?);
            self.ws();
            if self.eat(b',') {
                self.ws();
                if self.eat(close) {
                    return Some(items);
                }
                continue;
            }
            if self.eat(close) {
                return Some(items);
            }
            return None;
        }
    }

    fn dict(&mut self) -> Option<Json> {
        self.i += 1;
        let mut pairs: Vec<(String, Json)> = Vec::new();
        self.ws();
        if self.eat(b'}') {
            return Some(Json::Obj(pairs));
        }
        loop {
            let key = match self.value()? {
                Json::Str(s) => s,
                Json::Int(i) => i.to_string(),
                Json::UInt(u) => u.to_string(),
                Json::Float(f) => crate::json::py_float(f, true),
                Json::Bool(b) => if b { "true" } else { "false" }.to_string(),
                Json::Null => "null".to_string(),
                _ => return None,
            };
            self.ws();
            if !self.eat(b':') {
                return None;
            }
            self.ws();
            let value = self.value()?;
            if let Some(slot) = pairs.iter_mut().find(|(k, _)| *k == key) {
                slot.1 = value;
            } else {
                pairs.push((key, value));
            }
            self.ws();
            if self.eat(b',') {
                self.ws();
                if self.eat(b'}') {
                    return Some(Json::Obj(pairs));
                }
                continue;
            }
            if self.eat(b'}') {
                return Some(Json::Obj(pairs));
            }
            return None;
        }
    }

    fn number(&mut self) -> Option<Json> {
        let start = self.i;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || matches!(c, b'.' | b'-' | b'+' | b'_') {
                // A sign is only part of the number at its start or after an
                // exponent marker.
                if matches!(c, b'-' | b'+')
                    && self.i > start
                    && !matches!(self.s[self.i - 1], b'e' | b'E')
                {
                    break;
                }
                self.i += 1;
            } else {
                break;
            }
        }
        let text: String = self.src[start..self.i].chars().filter(|c| *c != '_').collect();
        let body = text.trim_start_matches(['-', '+']);
        let negative = text.starts_with('-');
        if let Some(hex) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
            let v = i64::from_str_radix(hex, 16).ok()?;
            return Some(Json::Int(if negative { -v } else { v }));
        }
        if body.chars().all(|c| c.is_ascii_digit()) && !body.is_empty() {
            if let Ok(v) = text.parse::<i64>() {
                return Some(Json::Int(v));
            }
            if let Ok(v) = text.parse::<u64>() {
                return Some(Json::UInt(v));
            }
        }
        let lower = body.to_ascii_lowercase();
        if lower == "inf" || lower == "nan" || body.is_empty() {
            return None;
        }
        text.parse::<f64>().ok().filter(|f| f.is_finite()).map(Json::Float)
    }

    /// A Python string literal (single, double or triple quoted) with
    /// Python's escapes; unknown escapes keep their backslash.
    fn string(&mut self) -> Option<String> {
        let quote = self.peek()?;
        let triple = self.s[self.i..].starts_with(&[quote; 3]);
        self.i += if triple { 3 } else { 1 };
        let mut out = String::new();
        loop {
            if triple {
                if self.s[self.i..].starts_with(&[quote; 3]) {
                    self.i += 3;
                    return Some(out);
                }
            } else if self.peek()? == quote {
                self.i += 1;
                return Some(out);
            }
            let c = self.src[self.i..].chars().next()?;
            if c == '\n' && !triple {
                return None;
            }
            self.i += c.len_utf8();
            if c != '\\' {
                out.push(c);
                continue;
            }
            let e = self.src[self.i..].chars().next()?;
            self.i += e.len_utf8();
            match e {
                '\n' => {}
                '\\' => out.push('\\'),
                '\'' => out.push('\''),
                '"' => out.push('"'),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                'b' => out.push('\u{8}'),
                'f' => out.push('\u{c}'),
                'v' => out.push('\u{b}'),
                'a' => out.push('\u{7}'),
                'x' => out.push(self.hex_escape(2)?),
                'u' => out.push(self.hex_escape(4)?),
                'U' => out.push(self.hex_escape(8)?),
                '0'..='7' => {
                    let mut v = e.to_digit(8)?;
                    for _ in 0..2 {
                        match self.peek() {
                            Some(d @ b'0'..=b'7') => {
                                v = v * 8 + u32::from(d - b'0');
                                self.i += 1;
                            }
                            _ => break,
                        }
                    }
                    out.push(char::from_u32(v)?);
                }
                other => {
                    out.push('\\');
                    out.push(other);
                }
            }
        }
    }

    fn raw_string(&mut self) -> Option<String> {
        let quote = self.peek()?;
        self.i += 1;
        let start = self.i;
        loop {
            let c = self.peek()?;
            if c == quote {
                let s = self.src[start..self.i].to_string();
                self.i += 1;
                return Some(s);
            }
            self.i += if c == b'\\' { 2 } else { 1 };
        }
    }

    fn hex_escape(&mut self, n: usize) -> Option<char> {
        let digits = self.src.get(self.i..self.i + n)?;
        let v = u32::from_str_radix(digits, 16).ok()?;
        self.i += n;
        char::from_u32(v)
    }
}

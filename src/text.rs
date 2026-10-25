//! Text side of the engine: the checkpoint's tokenizer (`tokenizer.json`
//! through the `tokenizers` crate, pure-Rust regex backend), incremental
//! detokenization for streaming, stop strings, the chat template and the
//! generation defaults from `generation_config.json`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context as _, Result, anyhow};

use crate::chat_template::ChatTemplate;
use crate::json::Json;

/// Sampling defaults of `generation_config.json`.
#[derive(Clone, Copy, Debug)]
pub struct SamplingDefaults {
    pub temperature: f64,
    pub top_k: u64,
    pub top_p: f64,
    pub repetition_penalty: f64,
}

pub struct Text {
    pub tokenizer: Tokenizer,
    pub template: ChatTemplate,
    /// `eos_token_id` of `generation_config.json` (the tokenizer's eos token
    /// when absent).
    pub eos_ids: Vec<u32>,
    pub defaults: SamplingDefaults,
}

impl Text {
    pub fn load(dir: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::load(&dir.join("tokenizer.json"))?;
        let template = ChatTemplate::load(dir)?;
        let config: serde_json::Value = match std::fs::read(dir.join("generation_config.json")) {
            Ok(bytes) => serde_json::from_slice(&bytes).context("parsing generation_config.json")?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => serde_json::Value::Null,
            Err(e) => return Err(e).context("reading generation_config.json"),
        };
        let mut eos_ids: Vec<u32> = match config.get("eos_token_id") {
            Some(serde_json::Value::Array(a)) => {
                a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect()
            }
            Some(v) => v.as_u64().map(|v| vec![v as u32]).unwrap_or_default(),
            None => Vec::new(),
        };
        if eos_ids.is_empty() {
            eos_ids.extend(tokenizer.token_to_id("<|im_end|>"));
        }
        let do_sample = config.get("do_sample").and_then(|v| v.as_bool()).unwrap_or(false);
        let num = |k: &str, d: f64| config.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
        let defaults = SamplingDefaults {
            temperature: if do_sample { num("temperature", 1.0) } else { 0.0 },
            top_k: config.get("top_k").and_then(|v| v.as_u64()).unwrap_or(0),
            top_p: num("top_p", 1.0),
            repetition_penalty: num("repetition_penalty", 1.0),
        };
        Ok(Self { tokenizer, template, eos_ids, defaults })
    }

    /// Renders a conversation with the chat template and encodes it.
    pub fn chat_prompt(
        &self,
        messages: &[Json],
        tools: Option<&[Json]>,
        kwargs: &[(String, Json)],
    ) -> Result<(String, Vec<u32>)> {
        let text = self.template.render(messages, tools, true, kwargs)?;
        let ids = self.tokenizer.encode(&text)?;
        Ok((text, ids))
    }
}

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    /// Per token id: the bytes the ByteLevel decoder produces for it and
    /// whether it is a special added token.
    pieces: Vec<Option<(Vec<u8>, bool)>>,
}

impl Tokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow!("{e}"))
            .with_context(|| format!("loading {}", path.display()))?;
        let byte_level = inner
            .get_decoder()
            .map(|d| serde_json::to_value(d).ok().and_then(|v| v.get("type").cloned()))
            .is_some_and(|t| t == Some(serde_json::Value::String("ByteLevel".into())));
        anyhow::ensure!(byte_level, "the tokenizer's decoder must be ByteLevel");
        let special: HashMap<u32, bool> = inner
            .get_added_tokens_decoder()
            .into_iter()
            .map(|(id, tok)| (id, tok.special))
            .collect();
        let chars = char_bytes();
        let size = inner.get_vocab_size(true);
        let pieces = (0..size as u32)
            .map(|id| {
                inner.id_to_token(id).map(|token| {
                    // tokenizers' ByteLevel::decode_chain: map every char
                    // back to its byte, or keep the token's UTF-8 when any
                    // char is outside the byte alphabet.
                    let mapped: Option<Vec<u8>> = token.chars().map(|c| chars.get(&c).copied()).collect();
                    let bytes = mapped.unwrap_or_else(|| token.as_bytes().to_vec());
                    (bytes, special.get(&id).copied().unwrap_or(false))
                })
            })
            .collect();
        Ok(Self { inner, pieces })
    }

    /// Token ids of `text` (no special tokens added; the chat template writes
    /// every marker itself).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.inner.encode_fast(text, false).map_err(|e| anyhow!("encoding: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner.decode(ids, skip_special_tokens).map_err(|e| anyhow!("decoding: {e}"))
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn detokenizer(&self, skip_special_tokens: bool) -> Detokenizer<'_> {
        Detokenizer { tok: self, skip_special_tokens, pending: Vec::new() }
    }
}

/// GPT-2's byte-to-unicode alphabet, inverted.
fn char_bytes() -> HashMap<char, u8> {
    let mut printable: Vec<u32> = (b'!' as u32..=b'~' as u32).collect();
    printable.extend(0xa1..=0xac);
    printable.extend(0xae..=0xff);
    let mut map = HashMap::new();
    let mut extra = 0u32;
    for b in 0u32..256 {
        let c = if printable.contains(&b) {
            b
        } else {
            extra += 1;
            255 + extra
        };
        map.insert(char::from_u32(c).expect("byte alphabet"), b as u8);
    }
    map
}

/// Streaming detokenizer. The ByteLevel decoder concatenates the tokens'
/// bytes and decodes them as lossy UTF-8, so text can be released as soon as
/// it is complete: a character split across tokens is held until its last
/// byte arrives. The concatenation of everything `push` and `finish` return
/// equals `decode(ids)` exactly.
pub struct Detokenizer<'a> {
    tok: &'a Tokenizer,
    skip_special_tokens: bool,
    pending: Vec<u8>,
}

impl Detokenizer<'_> {
    pub fn push(&mut self, id: u32) -> String {
        if let Some(Some((bytes, special))) = self.tok.pieces.get(id as usize)
            && !(*special && self.skip_special_tokens)
        {
            self.pending.extend_from_slice(bytes);
        }
        let mut out = String::new();
        let mut start = 0;
        loop {
            match std::str::from_utf8(&self.pending[start..]) {
                Ok(s) => {
                    out.push_str(s);
                    start = self.pending.len();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    out.push_str(
                        std::str::from_utf8(&self.pending[start..start + valid]).expect("valid prefix"),
                    );
                    start += valid;
                    match e.error_len() {
                        Some(len) => {
                            out.push('\u{FFFD}');
                            start += len;
                        }
                        // An incomplete character at the end: wait for more.
                        None => break,
                    }
                }
            }
        }
        self.pending.drain(..start);
        out
    }

    /// The text of an incomplete trailing character, as lossy decoding
    /// renders it.
    pub fn finish(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

/// Stop-string matching over streamed text. Text that could be the start of
/// a stop string is held back until it is decided.
pub struct StopMatcher {
    stops: Vec<String>,
    held: String,
}

impl StopMatcher {
    pub fn new(stops: &[String]) -> Self {
        Self { stops: stops.iter().filter(|s| !s.is_empty()).cloned().collect(), held: String::new() }
    }

    /// Adds text; returns the text that is safe to release and whether a stop
    /// string was found (the released text then ends just before it).
    pub fn push(&mut self, delta: &str) -> (String, bool) {
        self.held.push_str(delta);
        if self.stops.is_empty() {
            return (std::mem::take(&mut self.held), false);
        }
        if let Some(pos) = self.stops.iter().filter_map(|s| self.held.find(s.as_str())).min() {
            let out = self.held[..pos].to_string();
            self.held.clear();
            return (out, true);
        }
        // Hold the longest suffix that is a proper prefix of a stop string.
        let mut keep = 0;
        for s in &self.stops {
            for (i, _) in self.held.char_indices() {
                let suffix = &self.held[i..];
                if suffix.len() < s.len() && s.starts_with(suffix) {
                    keep = keep.max(suffix.len());
                    break;
                }
            }
        }
        let cut = self.held.len() - keep;
        let out = self.held[..cut].to_string();
        self.held.drain(..cut);
        (out, false)
    }

    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

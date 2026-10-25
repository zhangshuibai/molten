//! JSONL stdio protocol v2 / v2.1 for the LFM2.5 engine.

use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

use anyhow::{Context as _, Result, anyhow, ensure};
use serde_json::{Map, Value, json};

use crate::lfm::sample::Sampling;
use crate::lfm::{
    HEAD_DIM, HIDDEN, LfmModel, LfmState, Mode, N_KV_HEADS, Scratch, Snapshot, VOCAB,
};
use crate::metal::MetalContext;

const CAPABILITIES: [&str; 8] = [
    "reset",
    "forward",
    "generate",
    "checkpoint",
    "restore",
    "tensor_manifest",
    "tensor_chunk",
    "tensor_end",
];

pub struct Contract {
    pub context_tokens: usize,
    pub max_logits_rows: usize,
}

impl Contract {
    pub fn parse(text: &str) -> Result<Self> {
        let raw: Value = serde_json::from_str(text).context("parsing --contract-json")?;
        let context_tokens = raw
            .get("context_tokens")
            .and_then(Value::as_u64)
            .context("contract.context_tokens")? as usize;
        Ok(Self {
            context_tokens,
            max_logits_rows: raw
                .get("max_logits_rows")
                .and_then(Value::as_u64)
                .unwrap_or(32) as usize,
        })
    }
}

pub struct Engine {
    ctx: MetalContext,
    model: LfmModel,
    state: LfmState,
    scratch: Scratch,
    snapshots: BTreeMap<String, Snapshot>,
    next_snapshot: u64,
    contract: Contract,
    eof: Arc<AtomicBool>,
    transfer: Option<Transfer>,
}

struct Transfer {
    tensors: Vec<SelectedTensor>,
}

struct SelectedTensor {
    name: String,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl Engine {
    pub fn new(model_dir: &str, contract: Contract, eof: Arc<AtomicBool>) -> Result<Self> {
        let ctx = MetalContext::new()?;
        let model = LfmModel::load(&ctx, model_dir)?;
        let max_seq = contract
            .context_tokens
            .min(model.cfg.max_position_embeddings)
            .max(1);
        let state = LfmState::new(&ctx, max_seq)?;
        let scratch = model.scratch(&ctx, crate::lfm::PREFILL_CAP)?;
        Ok(Self {
            ctx,
            model,
            state,
            scratch,
            snapshots: BTreeMap::new(),
            next_snapshot: 0,
            contract,
            eof,
            transfer: None,
        })
    }

    fn envelope(&self, req: &Map<String, Value>) -> Map<String, Value> {
        let mut m = Map::new();
        for key in ["version", "id", "operation", "generation", "model_contract_sha256"] {
            if let Some(v) = req.get(key) {
                m.insert(key.to_string(), v.clone());
            }
        }
        m
    }

    fn handle(&mut self, req: Map<String, Value>) -> Result<Option<Map<String, Value>>> {
        let op = req
            .get("operation")
            .and_then(Value::as_str)
            .context("request.operation")?
            .to_string();
        match op.as_str() {
            "hello" => {
                let mut out = self.envelope(&req);
                out.insert("ok".into(), json!(true));
                out.insert("ready".into(), json!(true));
                out.insert("capabilities".into(), json!(CAPABILITIES));
                Ok(Some(out))
            }
            "reset" => {
                let mode = req
                    .get("mode")
                    .and_then(Value::as_str)
                    .context("reset.mode")?;
                let mode = match mode {
                    "teacher_forced" => Mode::TeacherForced,
                    "greedy" => Mode::Greedy,
                    other => anyhow::bail!("unknown reset mode {other}"),
                };
                self.state.reset()?;
                self.state.mode = mode;
                let mut out = self.envelope(&req);
                out.insert("ok".into(), json!(true));
                out.insert("state".into(), self.state.state_json());
                Ok(Some(out))
            }
            "forward" => {
                let tokens = int_array(&req, "tokens")?;
                let positions = size_array(&req, "logits_positions")?;
                ensure!(
                    positions.len() <= self.contract.max_logits_rows,
                    "too many logits positions"
                );
                self.model.forward(
                    &self.ctx,
                    &mut self.state,
                    &mut self.scratch,
                    &tokens,
                    &positions,
                    &self.eof,
                )?;
                let mut out = self.envelope(&req);
                out.insert("ok".into(), json!(true));
                out.insert("state".into(), self.state.state_json());
                out.insert("logits_positions".into(), json!(positions));
                out.insert("device_completed".into(), json!(true));
                Ok(Some(out))
            }
            "generate" => {
                let prompt = int_array(&req, "prompt")?;
                let count = req
                    .get("count")
                    .and_then(Value::as_u64)
                    .context("generate.count")? as usize;
                let sampling = Sampling::from_request(req.get("sampling"))?;
                let tokens = self.model.generate(
                    &self.ctx,
                    &mut self.state,
                    &mut self.scratch,
                    &prompt,
                    count,
                    sampling.as_ref(),
                    &self.eof,
                )?;
                let mut out = self.envelope(&req);
                out.insert("ok".into(), json!(true));
                out.insert("tokens".into(), json!(tokens));
                out.insert("state".into(), self.state.state_json());
                out.insert("device_completed".into(), json!(true));
                Ok(Some(out))
            }
            "checkpoint" => {
                let snap = self.model.checkpoint(&self.state)?;
                let handle = format!("lfmckpt{:08x}", self.next_snapshot);
                self.next_snapshot += 1;
                self.snapshots.insert(handle.clone(), snap);
                let mut out = self.envelope(&req);
                out.insert("ok".into(), json!(true));
                out.insert("handle".into(), json!(handle));
                out.insert("state".into(), self.state.state_json());
                Ok(Some(out))
            }
            "restore" => {
                let handle = req
                    .get("handle")
                    .and_then(Value::as_str)
                    .context("restore.handle")?
                    .to_string();
                let snap = self
                    .snapshots
                    .get(&handle)
                    .ok_or_else(|| anyhow!("unknown checkpoint handle"))?;
                self.model.restore(&mut self.state, snap)?;
                let mut out = self.envelope(&req);
                out.insert("ok".into(), json!(true));
                out.insert("restored".into(), json!(true));
                out.insert("state".into(), self.state.state_json());
                Ok(Some(out))
            }
            "tensor_manifest" => {
                let tensors = self.build_manifest(&req)?;
                let mut out = transfer_envelope(&req);
                out.insert(
                    "tensors".into(),
                    Value::Array(
                        tensors
                            .iter()
                            .map(|t| {
                                json!({
                                    "name": t.name,
                                    "shape": t.shape,
                                    "dtype": "float32-le",
                                    "nbytes": t.data.len(),
                                })
                            })
                            .collect(),
                    ),
                );
                self.transfer = Some(Transfer { tensors });
                Ok(Some(out))
            }
            "tensor_chunk" => {
                let name = req
                    .get("name")
                    .and_then(Value::as_str)
                    .context("tensor_chunk.name")?
                    .to_string();
                let offset = req
                    .get("offset")
                    .and_then(Value::as_u64)
                    .context("tensor_chunk.offset")? as usize;
                let nbytes = req
                    .get("nbytes")
                    .and_then(Value::as_u64)
                    .context("tensor_chunk.nbytes")? as usize;
                let transfer = self
                    .transfer
                    .as_ref()
                    .ok_or_else(|| anyhow!("tensor_chunk without a manifest"))?;
                let tensor = transfer
                    .tensors
                    .iter()
                    .find(|t| t.name == name)
                    .ok_or_else(|| anyhow!("tensor {name} not in manifest"))?;
                ensure!(
                    offset.checked_add(nbytes).is_some_and(|e| e <= tensor.data.len()),
                    "tensor chunk out of range"
                );
                let data = base64(&tensor.data[offset..offset + nbytes]);
                let mut out = transfer_envelope(&req);
                out.insert("name".into(), json!(name));
                out.insert("offset".into(), json!(offset));
                out.insert("nbytes".into(), json!(nbytes));
                out.insert("data".into(), json!(data));
                Ok(Some(out))
            }
            "tensor_end" => {
                let transfer = self
                    .transfer
                    .as_ref()
                    .ok_or_else(|| anyhow!("tensor_end without a manifest"))?;
                let total: usize = transfer.tensors.iter().map(|t| t.data.len()).sum();
                let count = transfer.tensors.len();
                let mut out = transfer_envelope(&req);
                out.insert("complete".into(), json!(true));
                out.insert("tensor_count".into(), json!(count));
                out.insert("total_bytes".into(), json!(total));
                self.transfer = None;
                Ok(Some(out))
            }
            other => anyhow::bail!("unknown operation {other}"),
        }
    }

    fn build_manifest(&self, req: &Map<String, Value>) -> Result<Vec<SelectedTensor>> {
        let selected_rows: Option<Vec<usize>> = match req.get("select") {
            Some(sel) => match sel.get("rows") {
                Some(Value::Array(rows)) => Some(
                    rows.iter()
                        .map(|v| {
                            v.as_u64()
                                .map(|x| x as usize)
                                .ok_or_else(|| anyhow!("select.rows must be integers"))
                        })
                        .collect::<Result<Vec<_>>>()?,
                ),
                Some(Value::Null) | None => None,
                _ => anyhow::bail!("select.rows must be an array"),
            },
            None => None,
        };
        let mut names: Vec<String> = Vec::new();
        if let Some(list) = req.get("tensors").and_then(Value::as_array) {
            for item in list {
                match item {
                    Value::String(s) => names.push(s.clone()),
                    Value::Object(o) => {
                        if let Some(s) = o.get("name").and_then(Value::as_str) {
                            names.push(s.to_string());
                        }
                    }
                    _ => anyhow::bail!("tensors entries must be names"),
                }
            }
        }
        if names.is_empty() {
            for p in self.state.logits.keys() {
                names.push(format!("logits.{p}"));
            }
            for (l, s) in self.state.conv.iter().enumerate() {
                if s.is_some() {
                    names.push(format!("conv.{l}.state"));
                }
            }
            for (l, s) in self.state.kv.iter().enumerate() {
                if s.is_some() {
                    names.push(format!("attn.{l}.k"));
                    names.push(format!("attn.{l}.v"));
                }
            }
        }
        let mut out = Vec::with_capacity(names.len());
        for name in names {
            out.push(self.build_tensor(&name, selected_rows.as_deref())?);
        }
        Ok(out)
    }

    fn build_tensor(&self, name: &str, rows: Option<&[usize]>) -> Result<SelectedTensor> {
        let parts: Vec<&str> = name.split('.').collect();
        if parts.len() == 2 && parts[0] == "logits" {
            let pos: usize = parts[1].parse().context("logits position")?;
            let row = self
                .state
                .logits
                .get(&pos)
                .ok_or_else(|| anyhow!("no retained logits row at {pos}"))?;
            ensure!(row.len() == VOCAB, "logits row width");
            let mut data = Vec::with_capacity(row.len() * 4);
            for v in row {
                data.extend_from_slice(&v.to_le_bytes());
            }
            return Ok(SelectedTensor {
                name: name.to_string(),
                shape: vec![VOCAB],
                data,
            });
        }
        if parts.len() == 3 && parts[0] == "conv" && parts[2] == "state" {
            let l: usize = parts[1].parse().context("conv layer")?;
            let bufs = self
                .state
                .conv
                .get(l)
                .and_then(Option::as_ref)
                .ok_or_else(|| anyhow!("layer {l} is not a conv layer"))?;
            let cur = &bufs[self.state.conv_slot];
            let data = bf16_to_f32_le(cur.raw_bytes());
            return Ok(SelectedTensor {
                name: name.to_string(),
                shape: vec![1, 2, HIDDEN],
                data,
            });
        }
        if parts.len() == 3 && parts[0] == "attn" && (parts[2] == "k" || parts[2] == "v") {
            let l: usize = parts[1].parse().context("attention layer")?;
            let cache = self
                .state
                .kv
                .get(l)
                .and_then(Option::as_ref)
                .ok_or_else(|| anyhow!("layer {l} is not an attention layer"))?;
            let t = if parts[2] == "k" { &cache.0 } else { &cache.1 };
            let pos = self.state.pos;
            let selected: Vec<usize> = match rows {
                Some(r) => r.to_vec(),
                None => (0..pos).collect(),
            };
            for &r in &selected {
                ensure!(r < pos, "selected row {r} beyond consumed {pos}");
            }
            let max_seq = t.shape()[1];
            let raw = t.raw_bytes();
            let mut data = Vec::with_capacity(N_KV_HEADS * selected.len() * HEAD_DIM * 4);
            for h in 0..N_KV_HEADS {
                for &r in &selected {
                    let start = (h * max_seq + r) * HEAD_DIM * 2;
                    data.extend_from_slice(&bf16_to_f32_le(
                        &raw[start..start + HEAD_DIM * 2],
                    ));
                }
            }
            return Ok(SelectedTensor {
                name: name.to_string(),
                shape: vec![1, N_KV_HEADS, selected.len(), HEAD_DIM],
                data,
            });
        }
        anyhow::bail!("unknown tensor name {name}")
    }
}

fn bf16_to_f32_le(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() * 2);
    for pair in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([pair[0], pair[1]]);
        let v = f32::from_bits((bits as u32) << 16);
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn int_array(req: &Map<String, Value>, key: &str) -> Result<Vec<u32>> {
    let arr = req
        .get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("request.{key}"))?;
    arr.iter()
        .map(|v| {
            v.as_u64()
                .filter(|x| *x <= u32::MAX as u64)
                .map(|x| x as u32)
                .ok_or_else(|| anyhow!("{key} entries must be u32 token ids"))
        })
        .collect()
}

fn size_array(req: &Map<String, Value>, key: &str) -> Result<Vec<usize>> {
    let arr = req
        .get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("request.{key}"))?;
    arr.iter()
        .map(|v| {
            v.as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| anyhow!("{key} entries must be non-negative integers"))
        })
        .collect()
}

fn transfer_envelope(req: &Map<String, Value>) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("ok".to_string(), Value::Bool(true));
    for key in [
        "version",
        "id",
        "operation",
        "generation",
        "transfer_id",
        "model_contract_sha256",
    ] {
        if let Some(v) = req.get(key) {
            m.insert(key.to_string(), v.clone());
        }
    }
    m
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64(data: &[u8]) -> String {
    let mut out = Vec::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let a = chunk[0] as u32;
        let b = *chunk.get(1).unwrap_or(&0) as u32;
        let c = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (a << 16) | (b << 8) | c;
        out.push(B64[((n >> 18) & 0x3f) as usize]);
        out.push(B64[((n >> 12) & 0x3f) as usize]);
        out.push(if chunk.len() > 1 { B64[((n >> 6) & 0x3f) as usize] } else { b'=' });
        out.push(if chunk.len() > 2 { B64[(n & 0x3f) as usize] } else { b'=' });
    }
    String::from_utf8(out).expect("base64 alphabet is ASCII")
}

pub fn run(model_dir: &str, contract_text: &str) -> Result<()> {
    let contract = Contract::parse(contract_text)?;
    let eof = Arc::new(AtomicBool::new(false));
    // Load before reading requests; `hello` reports readiness.
    let mut engine = Engine::new(model_dir, contract, eof.clone())?;

    let (tx, rx) = mpsc::channel::<String>();
    let reader_eof = eof.clone();
    std::thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(line) => {
                    if tx.send(line).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        reader_eof.store(true, Ordering::SeqCst);
    });

    let stdout = io::stdout();
    let mut out = io::BufWriter::with_capacity(1 << 20, stdout.lock());
    for line in rx {
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("molten: malformed request: {e}");
                std::process::exit(1);
            }
        };
        let obj = match value {
            Value::Object(o) => o,
            _ => {
                eprintln!("molten: request is not a JSON object");
                std::process::exit(1);
            }
        };
        match engine.handle(obj) {
            Ok(Some(reply)) => {
                let text = serde_json::to_string(&Value::Object(reply))?;
                writeln!(out, "{text}")?;
                out.flush()?;
            }
            Ok(None) => {}
            Err(e) => {
                if eof.load(Ordering::SeqCst) {
                    return Ok(());
                }
                eprintln!("molten: {e:#}");
                std::process::exit(1);
            }
        }
    }
    Ok(())
}

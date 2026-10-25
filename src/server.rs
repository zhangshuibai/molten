//! OpenAI-compatible HTTP server (`molten serve`): hand-written HTTP/1.1 on
//! `std::net`, one thread per connection, and the GPU session on the main
//! thread. Requests are served one at a time in arrival order (the engine is
//! single-sequence); the others wait in the job queue.
//!
//! Endpoints: `POST /v1/chat/completions`, `POST /v1/completions`,
//! `GET /v1/models`, `GET /v1/models/{id}`, `GET /health`. See docs/SERVER.md.

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result};
use serde_json::{Map, Value, json};

use crate::json::Json;
use crate::lfm::sample::Sampling;
use crate::parse::{OutputParser, ParsedOutput, Piece, THINK_OPEN, ToolCall};
use crate::session::{Finish, GenOutput, GenParams, Session};
use crate::text::Text;

const MAX_HEADER_BYTES: usize = 64 << 10;
const MAX_BODY_BYTES: usize = 64 << 20;
const IDLE_TIMEOUT: Duration = Duration::from_secs(300);
const WRITE_TIMEOUT: Duration = Duration::from_secs(60);

pub struct ServeConfig {
    pub host: String,
    pub port: u16,
    pub model_name: String,
    pub reasoning_split: bool,
    pub api_key: Option<String>,
}

struct Shared {
    cfg: ServeConfig,
    text: Arc<Text>,
    context: usize,
    created: u64,
    jobs: Mutex<mpsc::Sender<Job>>,
    next_id: AtomicU64,
}

struct Job {
    prompt: Vec<u32>,
    params: GenParams,
    events: mpsc::Sender<Event>,
    cancel: Arc<AtomicBool>,
    label: String,
    queued: Instant,
}

enum Event {
    Text(String),
    Done(Box<GenOutput>),
    Error(String),
}

/// Binds, then serves until the process is stopped. The listening address is
/// printed on stderr once the model is ready.
pub fn serve(mut session: Session, cfg: ServeConfig) -> Result<()> {
    let listener = TcpListener::bind((cfg.host.as_str(), cfg.port))
        .with_context(|| format!("binding {}:{}", cfg.host, cfg.port))?;
    let addr = listener.local_addr()?;
    let (tx, rx) = mpsc::channel::<Job>();
    let shared = Arc::new(Shared {
        text: session.text.clone(),
        context: session.context(),
        created: unix_now(),
        jobs: Mutex::new(tx),
        next_id: AtomicU64::new(1),
        cfg,
    });
    eprintln!(
        "molten: serving {} on http://{addr}/v1 (context {}, one request at a time)",
        shared.cfg.model_name, shared.context
    );
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let shared = shared.clone();
            std::thread::spawn(move || handle_connection(stream, &shared));
        }
    });
    for job in rx {
        if job.cancel.load(Ordering::SeqCst) {
            continue;
        }
        let queued_ms = job.queued.elapsed().as_secs_f64() * 1e3;
        let events = job.events.clone();
        let cancel = job.cancel.clone();
        let result = session.generate(&job.prompt, &job.params, &mut |delta| {
            if !delta.is_empty() && events.send(Event::Text(delta.to_string())).is_err() {
                return false;
            }
            !cancel.load(Ordering::SeqCst)
        });
        match result {
            Ok(out) => {
                eprintln!(
                    "molten: {} prompt={} cached={} completion={} finish={:?} queue_ms={:.0} ttft_ms={:.1} decode_tok_s={}",
                    job.label,
                    out.prompt_tokens,
                    out.cached_tokens,
                    out.tokens.len(),
                    out.finish,
                    queued_ms,
                    out.ttft.as_secs_f64() * 1e3,
                    out.decode_tok_s().map_or("-".into(), |r| format!("{r:.1}")),
                );
                let _ = job.events.send(Event::Done(Box::new(out)));
            }
            Err(e) => {
                eprintln!("molten: {} failed: {e:#}", job.label);
                let _ = job.events.send(Event::Error(format!("{e:#}")));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------- HTTP

struct Request {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    keep_alive: bool,
}

impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }
}

struct ApiError {
    status: u16,
    kind: &'static str,
    message: String,
    param: Option<&'static str>,
}

impl ApiError {
    fn bad(message: impl Into<String>) -> Self {
        Self { status: 400, kind: "invalid_request_error", message: message.into(), param: None }
    }

    fn param(param: &'static str, message: impl Into<String>) -> Self {
        Self { param: Some(param), ..Self::bad(message) }
    }

    fn body(&self) -> Value {
        json!({"error": {"message": self.message, "type": self.kind, "param": self.param, "code": Value::Null}})
    }
}

fn handle_connection(stream: TcpStream, shared: &Shared) {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(IDLE_TIMEOUT));
    let _ = stream.set_write_timeout(Some(WRITE_TIMEOUT));
    let Ok(read_half) = stream.try_clone() else { return };
    let mut reader = BufReader::new(read_half);
    let mut writer = stream;
    loop {
        let req = match read_request(&mut reader, &mut writer) {
            Ok(Some(req)) => req,
            Ok(None) => return,
            Err(e) => {
                let _ = write_json(&mut writer, e.status, &e.body(), false);
                return;
            }
        };
        let keep_alive = req.keep_alive;
        match route(&req, shared, &mut writer) {
            Ok(()) => {}
            Err(e) => {
                if write_json(&mut writer, e.status, &e.body(), keep_alive).is_err() {
                    return;
                }
            }
        }
        if !keep_alive {
            return;
        }
    }
}

fn read_line_limited(reader: &mut BufReader<TcpStream>, budget: &mut usize) -> Result<Option<String>, ApiError> {
    let mut line = Vec::new();
    let n = reader
        .by_ref()
        .take(*budget as u64 + 1)
        .read_until(b'\n', &mut line)
        .map_err(|_| ApiError::bad("connection error"))?;
    if n == 0 {
        return Ok(None);
    }
    if n > *budget || !line.ends_with(b"\n") {
        return Err(ApiError { status: 431, ..ApiError::bad("request header too large") });
    }
    *budget -= n;
    while line.last().is_some_and(|b| *b == b'\n' || *b == b'\r') {
        line.pop();
    }
    String::from_utf8(line).map(Some).map_err(|_| ApiError::bad("request header is not UTF-8"))
}

fn read_request(reader: &mut BufReader<TcpStream>, writer: &mut TcpStream) -> Result<Option<Request>, ApiError> {
    let mut budget = MAX_HEADER_BYTES;
    let line = loop {
        match read_line_limited(reader, &mut budget)? {
            None => return Ok(None),
            Some(l) if l.is_empty() => continue, // tolerate blank lines between requests
            Some(l) => break l,
        }
    };
    let mut parts = line.split_whitespace();
    let (Some(method), Some(target), Some(version)) = (parts.next(), parts.next(), parts.next()) else {
        return Err(ApiError::bad("malformed request line"));
    };
    if !version.starts_with("HTTP/1.") {
        return Err(ApiError { status: 505, ..ApiError::bad("HTTP version not supported") });
    }
    let mut headers = Vec::new();
    loop {
        let Some(h) = read_line_limited(reader, &mut budget)? else {
            return Err(ApiError::bad("truncated request header"));
        };
        if h.is_empty() {
            break;
        }
        let Some((k, v)) = h.split_once(':') else {
            return Err(ApiError::bad("malformed header line"));
        };
        headers.push((k.trim().to_ascii_lowercase(), v.trim().to_string()));
    }
    let get = |name: &str| headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.to_ascii_lowercase());
    let connection = get("connection").unwrap_or_default();
    let keep_alive = if version == "HTTP/1.0" {
        connection.contains("keep-alive")
    } else {
        !connection.contains("close")
    };
    let chunked = get("transfer-encoding").is_some_and(|v| v.contains("chunked"));
    let length = match get("content-length") {
        Some(v) => Some(v.parse::<usize>().map_err(|_| ApiError::bad("invalid Content-Length"))?),
        None => None,
    };
    if length.is_some_and(|n| n > MAX_BODY_BYTES) {
        return Err(ApiError { status: 413, ..ApiError::bad("request body too large") });
    }
    if (chunked || length.unwrap_or(0) > 0) && get("expect").is_some_and(|v| v == "100-continue") {
        writer
            .write_all(b"HTTP/1.1 100 Continue\r\n\r\n")
            .map_err(|_| ApiError::bad("connection error"))?;
    }
    let mut body = Vec::new();
    if chunked {
        loop {
            let size_line = read_line_limited(reader, &mut budget)?.ok_or_else(|| ApiError::bad("truncated chunked body"))?;
            let size_hex = size_line.split(';').next().unwrap_or("").trim();
            let size = usize::from_str_radix(size_hex, 16).map_err(|_| ApiError::bad("invalid chunk size"))?;
            if size == 0 {
                // Trailers, then the final blank line.
                while let Some(t) = read_line_limited(reader, &mut budget)? {
                    if t.is_empty() {
                        break;
                    }
                }
                break;
            }
            if body.len() + size > MAX_BODY_BYTES {
                return Err(ApiError { status: 413, ..ApiError::bad("request body too large") });
            }
            let start = body.len();
            body.resize(start + size, 0);
            reader.read_exact(&mut body[start..]).map_err(|_| ApiError::bad("truncated chunked body"))?;
            let mut crlf = [0u8; 2];
            reader.read_exact(&mut crlf).map_err(|_| ApiError::bad("truncated chunked body"))?;
        }
    } else if let Some(n) = length {
        body.resize(n, 0);
        reader.read_exact(&mut body).map_err(|_| ApiError::bad("truncated request body"))?;
    }
    let path = target.split('?').next().unwrap_or("/");
    let path = if path.len() > 1 { path.trim_end_matches('/') } else { path };
    Ok(Some(Request {
        method: method.to_string(),
        path: path.to_string(),
        headers,
        body,
        keep_alive,
    }))
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        505 => "HTTP Version Not Supported",
        _ => "Error",
    }
}

fn write_json(w: &mut TcpStream, status: u16, body: &Value, keep_alive: bool) -> io::Result<()> {
    let body = serde_json::to_vec(body).expect("JSON body");
    let head = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: {}\r\n\r\n",
        reason(status),
        body.len(),
        if keep_alive { "keep-alive" } else { "close" }
    );
    w.write_all(head.as_bytes())?;
    w.write_all(&body)?;
    w.flush()
}

/// A `text/event-stream` response with chunked transfer encoding.
struct Sse<'a> {
    w: &'a mut TcpStream,
}

impl<'a> Sse<'a> {
    fn start(w: &'a mut TcpStream) -> io::Result<Self> {
        w.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\
              Transfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n",
        )?;
        Ok(Self { w })
    }

    fn data(&mut self, payload: &str) -> io::Result<()> {
        let event = format!("data: {payload}\n\n");
        let chunk = format!("{:x}\r\n{event}\r\n", event.len());
        self.w.write_all(chunk.as_bytes())?;
        self.w.flush()
    }

    fn event(&mut self, value: &Value) -> io::Result<()> {
        self.data(&serde_json::to_string(value).expect("JSON event"))
    }

    fn done(self) -> io::Result<()> {
        let event = "data: [DONE]\n\n";
        let chunk = format!("{:x}\r\n{event}\r\n0\r\n\r\n", event.len());
        self.w.write_all(chunk.as_bytes())?;
        self.w.flush()
    }
}

// ------------------------------------------------------------------- routing

fn route(req: &Request, shared: &Shared, w: &mut TcpStream) -> Result<(), ApiError> {
    if req.path.starts_with("/v1/")
        && let Some(key) = &shared.cfg.api_key
        && req.header("authorization") != Some(&format!("Bearer {key}"))
    {
        return Err(ApiError {
            status: 401,
            kind: "authentication_error",
            message: "invalid or missing API key".into(),
            param: None,
        });
    }
    let model = json!({
        "id": shared.cfg.model_name,
        "object": "model",
        "created": shared.created,
        "owned_by": "molten",
        "max_model_len": shared.context,
    });
    let known = ["/health", "/v1/models", "/v1/chat/completions", "/v1/completions"];
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => ok(w, &json!({"status": "ok"}), req),
        ("GET", "/v1/models") => ok(w, &json!({"object": "list", "data": [model]}), req),
        ("GET", p) if p.strip_prefix("/v1/models/") == Some(shared.cfg.model_name.as_str()) => {
            ok(w, &model, req)
        }
        ("POST", "/v1/chat/completions") => completion(req, shared, w, Kind::Chat),
        ("POST", "/v1/completions") => completion(req, shared, w, Kind::Text),
        (_, p) if known.contains(&p) => Err(ApiError {
            status: 405,
            ..ApiError::bad(format!("method {} not allowed on {p}", req.method))
        }),
        (_, p) => Err(ApiError { status: 404, ..ApiError::bad(format!("not found: {p}")) }),
    }
}

fn ok(w: &mut TcpStream, body: &Value, req: &Request) -> Result<(), ApiError> {
    write_json(w, 200, body, req.keep_alive).map_err(|_| ApiError::bad("connection error"))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Chat,
    Text,
}

struct Prepared {
    prompt: Vec<u32>,
    params: GenParams,
    stream: bool,
    include_usage: bool,
    parse_tools: bool,
    opens_reasoning: bool,
}

fn completion(req: &Request, shared: &Shared, w: &mut TcpStream, kind: Kind) -> Result<(), ApiError> {
    let text = std::str::from_utf8(&req.body).map_err(|_| ApiError::bad("request body is not UTF-8"))?;
    let raw = Json::parse(text).map_err(|e| ApiError::bad(format!("invalid JSON body: {e}")))?;
    if !matches!(raw, Json::Obj(_)) {
        return Err(ApiError::bad("the request body must be a JSON object"));
    }
    let body = match raw.to_serde() {
        Value::Object(m) => m,
        _ => unreachable!(),
    };
    let prepared = match kind {
        Kind::Chat => prepare_chat(&raw, &body, shared)?,
        Kind::Text => prepare_text(&body, shared)?,
    };
    let n = shared.next_id.fetch_add(1, Ordering::Relaxed);
    let id = format!("{}-{}", if kind == Kind::Chat { "chatcmpl" } else { "cmpl" }, random_hex(n));
    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let job = Job {
        prompt: prepared.prompt.clone(),
        params: prepared.params.clone(),
        events: tx,
        cancel: cancel.clone(),
        label: format!("{} {}", req.path, id),
        queued: Instant::now(),
    };
    shared
        .jobs
        .lock()
        .expect("job queue")
        .send(job)
        .map_err(|_| ApiError { status: 500, kind: "server_error", ..ApiError::bad("the engine stopped") })?;
    let created = unix_now();
    let model = shared.cfg.model_name.as_str();
    let mut parser = OutputParser::new(
        kind == Kind::Chat && shared.cfg.reasoning_split,
        prepared.parse_tools,
        prepared.opens_reasoning,
    );
    let object = if kind == Kind::Chat { "chat.completion" } else { "text_completion" };
    if !prepared.stream {
        let mut parsed = ParsedOutput::default();
        let mut raw_text = String::new();
        let out = loop {
            match rx.recv() {
                Ok(Event::Text(t)) => {
                    raw_text.push_str(&t);
                    parser.push(&t).into_iter().for_each(|p| parsed.add(p));
                }
                Ok(Event::Done(out)) => break out,
                Ok(Event::Error(e)) => {
                    return Err(ApiError { status: 500, kind: "server_error", ..ApiError::bad(e) });
                }
                Err(_) => return Err(ApiError { status: 500, kind: "server_error", ..ApiError::bad("the engine stopped") }),
            }
        };
        parser.finish().into_iter().for_each(|p| parsed.add(p));
        let finish = finish_reason(out.finish, !parsed.tool_calls.is_empty());
        let choice = match kind {
            Kind::Chat => json!({
                "index": 0,
                "message": chat_message(&parsed, shared.cfg.reasoning_split, &id),
                "logprobs": Value::Null,
                "finish_reason": finish,
            }),
            Kind::Text => json!({"index": 0, "text": raw_text, "logprobs": Value::Null, "finish_reason": finish}),
        };
        let body = json!({
            "id": id, "object": object, "created": created, "model": model,
            "choices": [choice], "usage": usage(&out),
        });
        return ok(w, &body, req);
    }

    // Streaming.
    let chunk_object = if kind == Kind::Chat { "chat.completion.chunk" } else { "text_completion" };
    let chunk = |choice: Value| json!({"id": id, "object": chunk_object, "created": created, "model": model, "choices": [choice]});
    let Ok(mut sse) = Sse::start(w) else {
        cancel.store(true, Ordering::SeqCst);
        return Ok(());
    };
    let mut tool_index = 0usize;
    let mut any_tools = false;
    let mut send_pieces = |sse: &mut Sse, pieces: Vec<Piece>| -> io::Result<()> {
        for piece in pieces {
            let delta = match piece {
                Piece::Reasoning(s) => json!({"reasoning_content": s}),
                Piece::Content(s) if kind == Kind::Text => {
                    sse.event(&chunk(json!({"index": 0, "text": s, "logprobs": Value::Null, "finish_reason": Value::Null})))?;
                    continue;
                }
                Piece::Content(s) => json!({"content": s}),
                Piece::Tools(calls) => {
                    any_tools = true;
                    let calls: Vec<Value> = calls
                        .iter()
                        .map(|c| {
                            let v = tool_call_json(c, &id, tool_index, true);
                            tool_index += 1;
                            v
                        })
                        .collect();
                    json!({"tool_calls": calls})
                }
            };
            sse.event(&chunk(json!({"index": 0, "delta": delta, "logprobs": Value::Null, "finish_reason": Value::Null})))?;
        }
        Ok(())
    };
    let mut io_ok = kind != Kind::Chat
        || sse
            .event(&chunk(json!({"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": Value::Null, "finish_reason": Value::Null})))
            .is_ok();
    let result = loop {
        match rx.recv() {
            Ok(Event::Text(t)) => {
                if io_ok && send_pieces(&mut sse, parser.push(&t)).is_err() {
                    io_ok = false;
                    cancel.store(true, Ordering::SeqCst);
                }
            }
            Ok(Event::Done(out)) => break Ok(out),
            Ok(Event::Error(e)) => break Err(e),
            Err(_) => break Err("the engine stopped".to_string()),
        }
    };
    if !io_ok {
        return Ok(());
    }
    let out = match result {
        Ok(out) => out,
        Err(e) => {
            let err = ApiError { status: 500, kind: "server_error", ..ApiError::bad(e) };
            let _ = sse.event(&err.body());
            let _ = sse.done();
            return Ok(());
        }
    };
    if send_pieces(&mut sse, parser.finish()).is_err() {
        return Ok(());
    }
    let finish = finish_reason(out.finish, any_tools);
    let last = match kind {
        Kind::Chat => json!({"index": 0, "delta": {}, "logprobs": Value::Null, "finish_reason": finish}),
        Kind::Text => json!({"index": 0, "text": "", "logprobs": Value::Null, "finish_reason": finish}),
    };
    let _ = sse.event(&chunk(last));
    if prepared.include_usage {
        let _ = sse.event(&json!({
            "id": id, "object": chunk_object, "created": created, "model": model,
            "choices": [], "usage": usage(&out),
        }));
    }
    let _ = sse.done();
    Ok(())
}

fn finish_reason(finish: Finish, tool_calls: bool) -> &'static str {
    match finish {
        Finish::Stop if tool_calls => "tool_calls",
        f => f.as_str(),
    }
}

fn usage(out: &GenOutput) -> Value {
    json!({
        "prompt_tokens": out.prompt_tokens,
        "completion_tokens": out.tokens.len(),
        "total_tokens": out.prompt_tokens + out.tokens.len(),
        "prompt_tokens_details": {"cached_tokens": out.cached_tokens},
    })
}

fn tool_call_json(call: &ToolCall, id: &str, index: usize, with_index: bool) -> Value {
    let mut v = json!({
        "id": format!("call_{}_{index}", &id[id.len().saturating_sub(16)..]),
        "type": "function",
        "function": {"name": call.name, "arguments": call.arguments},
    });
    if with_index {
        v["index"] = json!(index);
    }
    v
}

fn chat_message(parsed: &ParsedOutput, split: bool, id: &str) -> Value {
    let mut m = Map::new();
    m.insert("role".into(), json!("assistant"));
    let content = match &parsed.content {
        Some(c) if !parsed.tool_calls.is_empty() && c.trim().is_empty() => None,
        Some(c) if !c.is_empty() => Some(c.clone()),
        _ if !split => Some(String::new()),
        _ => None,
    };
    m.insert("content".into(), json!(content));
    if split {
        m.insert("reasoning_content".into(), json!(parsed.reasoning));
    }
    if !parsed.tool_calls.is_empty() {
        let calls: Vec<Value> =
            parsed.tool_calls.iter().enumerate().map(|(i, c)| tool_call_json(c, id, i, false)).collect();
        m.insert("tool_calls".into(), Value::Array(calls));
    }
    Value::Object(m)
}

// ------------------------------------------------------------ request fields

fn opt<'a>(body: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    body.get(key).filter(|v| !v.is_null())
}

fn opt_bool(body: &Map<String, Value>, key: &'static str) -> Result<Option<bool>, ApiError> {
    match opt(body, key) {
        None => Ok(None),
        Some(v) => v.as_bool().map(Some).ok_or_else(|| ApiError::param(key, format!("{key} must be a boolean"))),
    }
}

fn opt_f64(body: &Map<String, Value>, key: &'static str) -> Result<Option<f64>, ApiError> {
    match opt(body, key) {
        None => Ok(None),
        Some(v) => v.as_f64().map(Some).ok_or_else(|| ApiError::param(key, format!("{key} must be a number"))),
    }
}

fn opt_int(body: &Map<String, Value>, key: &'static str) -> Result<Option<i64>, ApiError> {
    match opt(body, key) {
        None => Ok(None),
        Some(v) => v
            .as_i64()
            .or_else(|| v.as_f64().filter(|f| f.fract() == 0.0 && f.abs() < 9e15).map(|f| f as i64))
            .map(Some)
            .ok_or_else(|| ApiError::param(key, format!("{key} must be an integer"))),
    }
}

/// The fields shared by both endpoints: sampling, limits, stop conditions.
fn common_params(
    body: &Map<String, Value>,
    shared: &Shared,
    prompt_len: usize,
    default_max: Option<usize>,
) -> Result<(GenParams, bool, bool), ApiError> {
    if let Some(n) = opt_int(body, "n")?
        && n != 1
    {
        return Err(ApiError::param("n", "only n = 1 is supported"));
    }
    if let Some(n) = opt_int(body, "best_of")?
        && n != 1
    {
        return Err(ApiError::param("best_of", "only best_of = 1 is supported"));
    }
    for key in ["presence_penalty", "frequency_penalty"] {
        if opt_f64(body, key)?.is_some_and(|v| v != 0.0) {
            return Err(ApiError::param(key, format!("{key} is not supported")));
        }
    }
    match opt(body, "logprobs") {
        None | Some(Value::Bool(false)) => {}
        Some(Value::Number(n)) if n.as_u64() == Some(0) => {}
        Some(_) => return Err(ApiError::param("logprobs", "logprobs are not supported")),
    }
    if opt_int(body, "top_logprobs")?.is_some_and(|v| v > 0) {
        return Err(ApiError::param("top_logprobs", "logprobs are not supported"));
    }
    if opt(body, "logit_bias").is_some_and(|v| v.as_object().is_none_or(|m| !m.is_empty())) {
        return Err(ApiError::param("logit_bias", "logit_bias is not supported"));
    }
    if let Some(rf) = opt(body, "response_format")
        && rf.get("type").and_then(Value::as_str) != Some("text")
    {
        return Err(ApiError::param("response_format", "only response_format {\"type\": \"text\"} is supported"));
    }
    let stream = opt_bool(body, "stream")?.unwrap_or(false);
    let include_usage = match opt(body, "stream_options") {
        None => false,
        Some(v) => v.get("include_usage").and_then(Value::as_bool).unwrap_or(false),
    };

    let context = shared.context;
    if prompt_len >= context {
        return Err(ApiError::bad(format!(
            "This model's maximum context length is {context} tokens. However, your prompt has {prompt_len} tokens."
        )));
    }
    let room = context - prompt_len;
    let requested = match opt_int(body, "max_completion_tokens")? {
        Some(v) => Some(v),
        None => opt_int(body, "max_tokens")?,
    };
    let max_tokens = match requested {
        Some(v) if v < 1 => return Err(ApiError::param("max_tokens", "max_tokens must be at least 1")),
        Some(v) if v as u64 > room as u64 => {
            return Err(ApiError::param(
                "max_tokens",
                format!(
                    "This model's maximum context length is {context} tokens. However, you requested {} tokens ({prompt_len} in the prompt, {v} for the completion).",
                    prompt_len as u64 + v as u64
                ),
            ));
        }
        Some(v) => v as usize,
        None => default_max.unwrap_or(room).min(room),
    };

    let d = shared.text.defaults;
    let temperature = opt_f64(body, "temperature")?.unwrap_or(d.temperature);
    if !(temperature >= 0.0 && temperature.is_finite()) {
        return Err(ApiError::param("temperature", "temperature must be >= 0"));
    }
    let top_p = opt_f64(body, "top_p")?.unwrap_or(d.top_p);
    if !(top_p > 0.0 && top_p <= 1.0) {
        return Err(ApiError::param("top_p", "top_p must be in (0, 1]"));
    }
    let top_k = match opt_int(body, "top_k")? {
        None => d.top_k as i64,
        Some(-1) => 0,
        Some(k) if k >= 0 => k,
        Some(_) => return Err(ApiError::param("top_k", "top_k must be -1, 0 or positive")),
    };
    let repetition_penalty = opt_f64(body, "repetition_penalty")?.unwrap_or(d.repetition_penalty);
    if !(1e-6..=1e6).contains(&repetition_penalty) {
        return Err(ApiError::param("repetition_penalty", "repetition_penalty must be in [1e-6, 1e6]"));
    }
    let seed = match opt(body, "seed") {
        None => random_seed(),
        Some(v) => match (v.as_u64(), v.as_i64()) {
            (Some(u), _) => u,
            (None, Some(i)) => i as u64,
            _ => return Err(ApiError::param("seed", "seed must be an integer")),
        },
    };
    // Like vLLM, a temperature below 1e-5 is greedy decoding.
    let sampling = if temperature < 1e-5 {
        None
    } else {
        Sampling::from_request(Some(&json!({
            "temperature": temperature, "top_k": top_k, "top_p": top_p,
            "repetition_penalty": repetition_penalty, "seed": seed,
        })))
        .map_err(|e| ApiError::bad(format!("{e:#}")))?
    };

    let stop = match opt(body, "stop") {
        None => Vec::new(),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .map(|v| v.as_str().map(str::to_string))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| ApiError::param("stop", "stop must be a string or a list of strings"))?,
        Some(_) => return Err(ApiError::param("stop", "stop must be a string or a list of strings")),
    };
    let stop_token_ids = match opt(body, "stop_token_ids") {
        None => Vec::new(),
        Some(Value::Array(a)) => a
            .iter()
            .map(|v| v.as_u64().filter(|x| *x <= u32::MAX as u64).map(|x| x as u32))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| ApiError::param("stop_token_ids", "stop_token_ids must be a list of token ids"))?,
        Some(_) => return Err(ApiError::param("stop_token_ids", "stop_token_ids must be a list of token ids")),
    };
    let params = GenParams {
        max_tokens,
        sampling,
        stop,
        stop_token_ids,
        ignore_eos: opt_bool(body, "ignore_eos")?.unwrap_or(false),
        skip_special_tokens: opt_bool(body, "skip_special_tokens")?.unwrap_or(true),
    };
    Ok((params, stream, include_usage))
}

fn prepare_text(body: &Map<String, Value>, shared: &Shared) -> Result<Prepared, ApiError> {
    for key in ["suffix"] {
        if opt(body, key).is_some() {
            return Err(ApiError::param("suffix", format!("{key} is not supported")));
        }
    }
    if opt_bool(body, "echo")?.unwrap_or(false) {
        return Err(ApiError::param("echo", "echo is not supported"));
    }
    let bad = || ApiError::param("prompt", "prompt must be a string, a list of one string or a list of token ids");
    let ids_of = |a: &[Value]| -> Option<Vec<u32>> {
        a.iter().map(|v| v.as_u64().filter(|x| *x <= u32::MAX as u64).map(|x| x as u32)).collect()
    };
    let prompt = match opt(body, "prompt").ok_or_else(bad)? {
        Value::String(s) => encode(shared, s)?,
        Value::Array(a) if a.len() == 1 && a[0].is_string() => encode(shared, a[0].as_str().unwrap_or(""))?,
        Value::Array(a) if a.len() == 1 && a[0].is_array() => ids_of(a[0].as_array().unwrap_or(&Vec::new())).ok_or_else(bad)?,
        Value::Array(a) if !a.is_empty() && a.iter().all(Value::is_number) => ids_of(a).ok_or_else(bad)?,
        Value::Array(_) => return Err(ApiError::param("prompt", "only one prompt per request is supported")),
        _ => return Err(bad()),
    };
    if prompt.is_empty() {
        return Err(ApiError::param("prompt", "the prompt is empty"));
    }
    let (params, stream, include_usage) = common_params(body, shared, prompt.len(), Some(16))?;
    Ok(Prepared { prompt, params, stream, include_usage, parse_tools: false, opens_reasoning: false })
}

fn encode(shared: &Shared, s: &str) -> Result<Vec<u32>, ApiError> {
    shared.text.tokenizer.encode(s).map_err(|e| ApiError::bad(format!("{e:#}")))
}

fn prepare_chat(raw: &Json, body: &Map<String, Value>, shared: &Shared) -> Result<Prepared, ApiError> {
    let messages = chat_messages(raw.get("messages")).map_err(|m| ApiError::param("messages", m))?;
    let tools = match raw.get("tools") {
        None | Some(Json::Null) => None,
        Some(Json::Arr(tools)) => {
            for t in tools {
                let ok = t.get("type").and_then(Json::as_str) == Some("function")
                    && t.get("function").and_then(|f| f.get("name")).and_then(Json::as_str).is_some();
                if !ok {
                    return Err(ApiError::param("tools", "each tool must be {\"type\": \"function\", \"function\": {\"name\": ...}}"));
                }
            }
            Some(tools.clone())
        }
        Some(_) => return Err(ApiError::param("tools", "tools must be a list")),
    };
    let parse_tools = match opt(body, "tool_choice") {
        None => tools.as_ref().is_some_and(|t| !t.is_empty()),
        Some(Value::String(s)) if s == "auto" => tools.as_ref().is_some_and(|t| !t.is_empty()),
        Some(Value::String(s)) if s == "none" => false,
        Some(_) => return Err(ApiError::param("tool_choice", "tool_choice must be \"auto\" or \"none\"")),
    };
    let kwargs: Vec<(String, Json)> = match raw.get("chat_template_kwargs") {
        None | Some(Json::Null) => Vec::new(),
        Some(Json::Obj(pairs)) => pairs.clone(),
        Some(_) => return Err(ApiError::param("chat_template_kwargs", "chat_template_kwargs must be an object")),
    };
    let rendered = shared
        .text
        .template
        .render(&messages, tools.as_deref(), true, &kwargs)
        .map_err(|e| ApiError::param("messages", format!("{e:#}")))?;
    let prompt = encode(shared, &rendered)?;
    let (params, stream, include_usage) = common_params(body, shared, prompt.len(), None)?;
    Ok(Prepared {
        prompt,
        params,
        stream,
        include_usage,
        parse_tools,
        opens_reasoning: rendered.ends_with(THINK_OPEN),
    })
}

/// Validates the messages and turns them into what the template expects:
/// `developer` becomes `system`, and tool-call `arguments` sent as JSON
/// strings (the OpenAI wire format) are parsed into objects.
pub fn chat_messages(value: Option<&Json>) -> Result<Vec<Json>, String> {
    let Some(Json::Arr(list)) = value else {
        return Err("messages must be a non-empty list".into());
    };
    if list.is_empty() {
        return Err("messages must be a non-empty list".into());
    }
    let mut out = Vec::with_capacity(list.len());
    for (i, m) in list.iter().enumerate() {
        let mut m = m.clone();
        let role = match m.get("role").and_then(Json::as_str) {
            Some(r @ ("system" | "user" | "assistant" | "tool")) => r.to_string(),
            Some("developer") => "system".to_string(),
            _ => {
                return Err(format!("messages[{i}].role must be system, developer, user, assistant or tool"));
            }
        };
        m.insert("role", Json::Str(role.clone()));
        match m.get("content") {
            None | Some(Json::Null) | Some(Json::Str(_)) => {}
            Some(Json::Arr(parts)) => {
                for p in parts {
                    let text_part = p.get("type").and_then(Json::as_str) == Some("text")
                        && p.get("text").and_then(Json::as_str).is_some();
                    if !text_part {
                        return Err(format!("messages[{i}].content: only text parts are supported"));
                    }
                }
            }
            Some(_) => {
                return Err(format!("messages[{i}].content must be a string or a list of parts"));
            }
        }
        if let Some(Json::Arr(calls)) = m.get_mut("tool_calls") {
            for call in calls.iter_mut() {
                let target = if call.get("function").is_some() {
                    call.get_mut("function")
                } else {
                    Some(call)
                };
                if let Some(func) = target
                    && let Some(Json::Str(args)) = func.get("arguments").cloned()
                {
                    let parsed = if args.trim().is_empty() {
                        Json::Obj(Vec::new())
                    } else {
                        Json::parse(&args)
                            .ok()
                            .filter(|v| matches!(v, Json::Obj(_)))
                            .ok_or_else(|| format!("messages[{i}].tool_calls: arguments must be a JSON object"))?
                    };
                    func.insert("arguments", parsed);
                }
            }
        }
        out.push(m);
    }
    Ok(out)
}

// -------------------------------------------------------------------- misc

fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// A random 64-bit value from the standard library's per-process hash keys,
/// the clock and a counter.
pub fn random_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
    h.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
    h.finish()
}

fn random_hex(n: u64) -> String {
    format!("{:016x}{:08x}", random_seed(), n as u32)
}

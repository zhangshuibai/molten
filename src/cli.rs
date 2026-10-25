//! The `molten` command line: `serve`, `run`, `chat` and `text`, plus the
//! JSON-lines engine protocol when no subcommand is given (`molten --model
//! DIR --contract-json JSON`, see docs/PROTOCOL.md).

use std::io::{BufRead, IsTerminal, Write};
use std::path::PathBuf;

use anyhow::{Context as _, Result, bail};
use serde_json::{Value, json};

use crate::json::Json;
use crate::lfm::sample::Sampling;
use crate::parse::{OutputParser, ParsedOutput, Piece, THINK_OPEN};
use crate::session::{DEFAULT_CONTEXT, GenOutput, GenParams, Session};
use crate::text::Text;

pub const USAGE: &str = "\
molten: native Rust + Metal inference engine for LFM2.5-24B-A2B (MLX 4-bit)

usage:
  molten run   --model DIR --prompt TEXT [options]   one prompt, streamed answer
  molten chat  --model DIR [options]                 interactive multi-turn chat
  molten serve --model DIR [--host 127.0.0.1] [--port 8080] [options]
                                                     OpenAI-compatible HTTP server
  molten text  --model DIR                           tokenizer/template tool (JSON lines on stdin)
  molten --model DIR --contract-json JSON            JSON-lines engine protocol (docs/PROTOCOL.md)

generation options (run, chat; for serve, each request sets its own):
  --max-tokens N          tokens to generate (default: up to the context)
  --temperature T         default from generation_config.json (0.1)
  --top-k K               default from generation_config.json (50); 0 = off
  --top-p P               default from generation_config.json (1.0 = off)
  --repetition-penalty R  default from generation_config.json (1.05)
  --seed S                sampling seed (default: random, printed on stderr)
  --greedy                temperature 0 (argmax decoding; no repetition penalty)
  --system TEXT           system message
  --context N             context length in tokens (default 131072; also for serve)

run:
  --raw                   no chat template: encode TEXT as is and print the raw continuation
  --ignore-eos            do not stop at <|im_end|>
  --show-reasoning        print the reasoning (dimmed on a terminal; default when stdout is a terminal)
  --hide-reasoning        print only the answer
  --json                  print one JSON object (ids, text, timings) instead of streaming text
chat:
  --hide-reasoning        do not print the reasoning
  --no-prefix-cache       prefill every turn from scratch
  commands: /reset clears the conversation, /exit (or Ctrl-D) quits
serve:
  --host H, --port P      listen address (default 127.0.0.1:8080)
  --served-model-name N   model id reported by /v1/models (default: the model directory's name)
  --api-key KEY           require `Authorization: Bearer KEY` on /v1/*
  --no-reasoning-split    return the reasoning inside `content` instead of `reasoning_content`
  --no-prefix-cache       do not reuse the previous request's prompt prefix
";

struct Args {
    items: Vec<String>,
}

impl Args {
    fn take_flag(&mut self, name: &str) -> bool {
        match self.items.iter().position(|a| a == name) {
            Some(i) => {
                self.items.remove(i);
                true
            }
            None => false,
        }
    }

    fn take_value(&mut self, name: &str) -> Result<Option<String>> {
        let Some(i) = self.items.iter().position(|a| a == name) else {
            return Ok(None);
        };
        if i + 1 >= self.items.len() {
            bail!("{name} needs a value");
        }
        let v = self.items.remove(i + 1);
        self.items.remove(i);
        Ok(Some(v))
    }

    fn take_parsed<T: std::str::FromStr>(&mut self, name: &str) -> Result<Option<T>> {
        match self.take_value(name)? {
            None => Ok(None),
            Some(v) => v.parse().map(Some).map_err(|_| anyhow::anyhow!("{name}: invalid value {v:?}")),
        }
    }

    fn finish(&self) -> Result<()> {
        if let Some(a) = self.items.first() {
            bail!("unexpected argument {a:?} (see molten --help)");
        }
        Ok(())
    }
}

/// Sampling flags shared by `run` and `chat`.
struct SamplingFlags {
    sampling: Option<Sampling>,
    seed: u64,
    seed_given: bool,
}

fn sampling_flags(args: &mut Args, text: &Text) -> Result<SamplingFlags> {
    let d = text.defaults;
    let greedy = args.take_flag("--greedy");
    let temperature: f64 = args.take_parsed("--temperature")?.unwrap_or(d.temperature);
    let top_k: u64 = args.take_parsed("--top-k")?.unwrap_or(d.top_k);
    let top_p: f64 = args.take_parsed("--top-p")?.unwrap_or(d.top_p);
    let repetition_penalty: f64 = args.take_parsed("--repetition-penalty")?.unwrap_or(d.repetition_penalty);
    let given: Option<u64> = args.take_parsed("--seed")?;
    let seed = given.unwrap_or_else(crate::server::random_seed);
    let sampling = if greedy || temperature == 0.0 {
        None
    } else {
        Sampling::from_request(Some(&json!({
            "temperature": temperature, "top_k": top_k, "top_p": top_p,
            "repetition_penalty": repetition_penalty, "seed": seed,
        })))?
    };
    Ok(SamplingFlags { sampling, seed, seed_given: given.is_some() })
}

fn model_dir(args: &mut Args) -> Result<PathBuf> {
    let dir = args.take_value("--model")?.context("--model DIR is required")?;
    Ok(PathBuf::from(dir))
}

/// Runs a subcommand; `argv` starts with its name.
pub fn main(mut argv: Vec<String>) -> Result<()> {
    let sub = argv.remove(0);
    let args = Args { items: argv };
    match sub.as_str() {
        "run" => run(args),
        "chat" => chat(args),
        "serve" => serve(args),
        "text" => text_tool(args),
        other => bail!("unknown subcommand {other:?} (see molten --help)"),
    }
}

fn serve(mut args: Args) -> Result<()> {
    let dir = model_dir(&mut args)?;
    let host = args.take_value("--host")?.unwrap_or_else(|| "127.0.0.1".into());
    let port: u16 = args.take_parsed("--port")?.unwrap_or(8080);
    let context: usize = args.take_parsed("--context")?.unwrap_or(DEFAULT_CONTEXT);
    let name = match args.take_value("--served-model-name")? {
        Some(n) => n,
        None => dir
            .canonicalize()
            .unwrap_or_else(|_| dir.clone())
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "molten".into()),
    };
    let api_key = args.take_value("--api-key")?;
    let reasoning_split = !args.take_flag("--no-reasoning-split");
    let prefix_cache = !args.take_flag("--no-prefix-cache");
    args.finish()?;
    eprintln!("molten: loading {}", dir.display());
    let mut session = Session::load(&dir, context, prefix_cache)?;
    session.warm_up()?;
    crate::server::serve(
        session,
        crate::server::ServeConfig { host, port, model_name: name, reasoning_split, api_key },
    )
}

const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

/// Prints streamed output: reasoning dimmed (or between `<think>` tags when
/// not on a terminal), or hidden.
struct Printer {
    parser: OutputParser,
    show_reasoning: bool,
    tty: bool,
    in_reasoning: bool,
    parsed: ParsedOutput,
}

impl Printer {
    fn new(raw: bool, show_reasoning: bool, opens_reasoning: bool) -> Self {
        Self {
            parser: OutputParser::new(!raw, false, opens_reasoning),
            show_reasoning,
            tty: std::io::stdout().is_terminal(),
            in_reasoning: false,
            parsed: ParsedOutput::default(),
        }
    }

    fn pieces(&mut self, pieces: Vec<Piece>) {
        let mut out = std::io::stdout().lock();
        for p in pieces {
            match &p {
                Piece::Reasoning(s) if self.show_reasoning => {
                    if !self.in_reasoning {
                        let _ = write!(out, "{}", if self.tty { DIM } else { "<think>" });
                        self.in_reasoning = true;
                    }
                    let _ = write!(out, "{s}");
                }
                Piece::Reasoning(_) => {}
                Piece::Content(_) | Piece::Tools(_) if self.in_reasoning => {
                    let _ = write!(out, "{}", if self.tty { RESET } else { "</think>" });
                    self.in_reasoning = false;
                    if let Piece::Content(s) = &p {
                        let _ = write!(out, "{s}");
                    }
                }
                Piece::Content(s) => {
                    let _ = write!(out, "{s}");
                }
                Piece::Tools(_) => {}
            }
            self.parsed.add(p);
        }
        let _ = out.flush();
    }

    fn push(&mut self, delta: &str) {
        let pieces = self.parser.push(delta);
        self.pieces(pieces);
    }

    fn finish(&mut self) {
        let pieces = self.parser.finish();
        self.pieces(pieces);
        if self.in_reasoning {
            print!("{}", if self.tty { RESET } else { "</think>" });
            self.in_reasoning = false;
        }
        println!();
    }
}

fn stats_line(out: &GenOutput) -> String {
    format!(
        "[{} prompt tokens ({} cached), {} generated, TTFT {:.0} ms, {} tok/s, finish {}]",
        out.prompt_tokens,
        out.cached_tokens,
        out.tokens.len(),
        out.ttft.as_secs_f64() * 1e3,
        out.decode_tok_s().map_or("-".into(), |r| format!("{r:.1}")),
        out.finish.as_str(),
    )
}

fn print_stats(out: &GenOutput) {
    if std::io::stderr().is_terminal() {
        eprintln!("{DIM}{}{RESET}", stats_line(out));
    } else {
        eprintln!("{}", stats_line(out));
    }
}

fn messages(system: Option<&str>, history: &[(String, String)], user: &str) -> Vec<Json> {
    let msg = |role: &str, content: &str| {
        Json::Obj(vec![("role".into(), Json::Str(role.into())), ("content".into(), Json::Str(content.into()))])
    };
    let mut m = Vec::new();
    if let Some(s) = system {
        m.push(msg("system", s));
    }
    for (u, a) in history {
        m.push(msg("user", u));
        m.push(msg("assistant", a));
    }
    m.push(msg("user", user));
    m
}

fn run(mut args: Args) -> Result<()> {
    let dir = model_dir(&mut args)?;
    let mut prompt = args.take_value("--prompt")?.context("--prompt TEXT is required")?;
    if prompt == "-" {
        prompt.clear();
        std::io::stdin().lock().read_to_string_lossy(&mut prompt)?;
    }
    let system = args.take_value("--system")?;
    let max_tokens: Option<usize> = args.take_parsed("--max-tokens")?;
    let context: usize = args.take_parsed("--context")?.unwrap_or(DEFAULT_CONTEXT);
    let raw = args.take_flag("--raw");
    let ignore_eos = args.take_flag("--ignore-eos");
    let json_out = args.take_flag("--json");
    let show = args.take_flag("--show-reasoning");
    let hide = args.take_flag("--hide-reasoning");
    let text = Text::load(&dir)?;
    let flags = sampling_flags(&mut args, &text)?;
    args.finish()?;
    let (rendered, ids) = if raw {
        (prompt.clone(), text.tokenizer.encode(&prompt)?)
    } else {
        text.chat_prompt(&messages(system.as_deref(), &[], &prompt), None, &[])?
    };
    drop(text);
    let mut session = Session::load(&dir, context, false)?;
    if flags.sampling.is_some() && !flags.seed_given && !json_out {
        eprintln!("seed: {}", flags.seed);
    }
    let params = GenParams {
        max_tokens: max_tokens.unwrap_or(usize::MAX).min(session.context().saturating_sub(ids.len())).max(1),
        sampling: flags.sampling,
        stop: Vec::new(),
        stop_token_ids: Vec::new(),
        ignore_eos,
        skip_special_tokens: !raw,
    };
    let show_reasoning = show || (!hide && std::io::stdout().is_terminal());
    let mut printer = Printer::new(raw, show_reasoning, rendered.ends_with(THINK_OPEN));
    let out = if json_out {
        session.generate(&ids, &params, &mut |_| true)?
    } else {
        let out = session.generate(&ids, &params, &mut |d| {
            printer.push(d);
            true
        })?;
        printer.finish();
        out
    };
    if json_out {
        let mut parser = OutputParser::new(!raw, false, rendered.ends_with(THINK_OPEN));
        let mut parsed = ParsedOutput::default();
        parser.push(&out.text).into_iter().chain(parser.finish()).for_each(|p| parsed.add(p));
        let v = json!({
            "prompt_ids": ids,
            "tokens": out.tokens,
            "text": out.text,
            "reasoning_content": parsed.reasoning,
            "content": parsed.content,
            "finish_reason": out.finish.as_str(),
            "seed": flags.sampling.map(|s| s.seed),
            "prompt_tokens": out.prompt_tokens,
            "ttft_ms": out.ttft.as_secs_f64() * 1e3,
            "decode_tok_s": out.decode_tok_s(),
        });
        println!("{v}");
    } else {
        print_stats(&out);
    }
    Ok(())
}

trait ReadLossy {
    fn read_to_string_lossy(&mut self, out: &mut String) -> Result<()>;
}

impl<R: std::io::Read> ReadLossy for R {
    fn read_to_string_lossy(&mut self, out: &mut String) -> Result<()> {
        let mut bytes = Vec::new();
        self.read_to_end(&mut bytes)?;
        out.push_str(&String::from_utf8_lossy(&bytes));
        Ok(())
    }
}

fn chat(mut args: Args) -> Result<()> {
    let dir = model_dir(&mut args)?;
    let system = args.take_value("--system")?;
    let max_tokens: Option<usize> = args.take_parsed("--max-tokens")?;
    let context: usize = args.take_parsed("--context")?.unwrap_or(DEFAULT_CONTEXT);
    let hide = args.take_flag("--hide-reasoning");
    let prefix_cache = !args.take_flag("--no-prefix-cache");
    let text = Text::load(&dir)?;
    let mut flags = sampling_flags(&mut args, &text)?;
    args.finish()?;
    drop(text);
    eprintln!("molten: loading {}", dir.display());
    let mut session = Session::load(&dir, context, prefix_cache)?;
    session.warm_up()?;
    let text = session.text.clone();
    eprintln!("molten chat: /reset clears the conversation, /exit quits");
    let mut history: Vec<(String, String)> = Vec::new();
    let stdin = std::io::stdin();
    loop {
        eprint!("> ");
        let _ = std::io::stderr().flush();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            eprintln!();
            return Ok(());
        }
        let line = line.trim_end_matches(['\n', '\r']);
        match line.trim() {
            "" => continue,
            "/exit" | "/quit" => return Ok(()),
            "/reset" => {
                history.clear();
                eprintln!("(conversation cleared)");
                continue;
            }
            _ => {}
        }
        let (rendered, ids) = text.chat_prompt(&messages(system.as_deref(), &history, line), None, &[])?;
        if ids.len() >= session.context() {
            eprintln!("(the conversation no longer fits the context; /reset to start over)");
            continue;
        }
        let params = GenParams {
            max_tokens: max_tokens.unwrap_or(usize::MAX).min(session.context() - ids.len()).max(1),
            sampling: flags.sampling,
            stop: Vec::new(),
            stop_token_ids: Vec::new(),
            ignore_eos: false,
            skip_special_tokens: true,
        };
        // A fresh seed per turn when none was given.
        if let Some(s) = flags.sampling.as_mut()
            && !flags.seed_given
        {
            s.seed = crate::server::random_seed();
        }
        let mut printer = Printer::new(false, !hide, rendered.ends_with(THINK_OPEN));
        let out = session.generate(&ids, &params, &mut |d| {
            printer.push(d);
            true
        })?;
        printer.finish();
        print_stats(&out);
        let answer = printer.parsed.content.clone().unwrap_or_default();
        history.push((line.to_string(), answer));
    }
}

/// `molten text`: JSON lines on stdin, one reply per line on stdout. No GPU.
///   {"op": "encode", "text": ...}                          -> {"ids": [...]}
///   {"op": "decode", "ids": [...], "skip_special_tokens": b} -> {"text": ...}
///   {"op": "stream_decode", "ids": [...], "skip_special_tokens": b} -> {"chunks": [...]}
///   {"op": "render", "messages": [...], "tools": [...], "add_generation_prompt": b, "kwargs": {...}}
///                                                           -> {"text": ...} or {"error": ...}
///   {"op": "parse", "text": ..., "tools": b, "opens_reasoning": b} -> {"pieces": [...]}
fn text_tool(mut args: Args) -> Result<()> {
    let dir = model_dir(&mut args)?;
    args.finish()?;
    let text = Text::load(&dir)?;
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let req = Json::parse(&line).context("request")?;
        let op = req.get("op").and_then(Json::as_str).unwrap_or("");
        let flag = |k: &str, d: bool| match req.get(k) {
            Some(Json::Bool(b)) => *b,
            _ => d,
        };
        let ids = || -> Vec<u32> {
            match req.get("ids") {
                Some(Json::Arr(a)) => a
                    .iter()
                    .filter_map(|v| match v {
                        Json::Int(i) => Some(*i as u32),
                        _ => None,
                    })
                    .collect(),
                _ => Vec::new(),
            }
        };
        let reply: Value = match op {
            "encode" => match text.tokenizer.encode(req.get("text").and_then(Json::as_str).unwrap_or("")) {
                Ok(ids) => json!({"ids": ids}),
                Err(e) => json!({"error": format!("{e:#}")}),
            },
            "decode" => match text.tokenizer.decode(&ids(), flag("skip_special_tokens", false)) {
                Ok(t) => json!({"text": t}),
                Err(e) => json!({"error": format!("{e:#}")}),
            },
            "stream_decode" => {
                let mut d = text.tokenizer.detokenizer(flag("skip_special_tokens", false));
                let mut chunks: Vec<String> = ids().into_iter().map(|id| d.push(id)).collect();
                chunks.push(d.finish());
                json!({"chunks": chunks})
            }
            "render" => {
                let messages = crate::server::chat_messages(req.get("messages"));
                let tools = match req.get("tools") {
                    Some(Json::Arr(t)) => Some(t.clone()),
                    _ => None,
                };
                let kwargs = match req.get("kwargs") {
                    Some(Json::Obj(p)) => p.clone(),
                    _ => Vec::new(),
                };
                match messages {
                    Ok(m) => match text.template.render(&m, tools.as_deref(), flag("add_generation_prompt", true), &kwargs) {
                        Ok(t) => json!({"text": t}),
                        Err(e) => json!({"error": format!("{e:#}")}),
                    },
                    Err(e) => json!({"error": e}),
                }
            }
            "parse" => {
                let mut p = OutputParser::new(true, flag("tools", true), flag("opens_reasoning", true));
                let s = req.get("text").and_then(Json::as_str).unwrap_or("");
                // Feed one character at a time to exercise the streaming path.
                let mut pieces = Vec::new();
                for (i, c) in s.char_indices() {
                    pieces.extend(p.push(&s[i..i + c.len_utf8()]));
                }
                pieces.extend(p.finish());
                let mut parsed = ParsedOutput::default();
                pieces.into_iter().for_each(|x| parsed.add(x));
                json!({
                    "reasoning_content": parsed.reasoning,
                    "content": parsed.content,
                    "tool_calls": parsed.tool_calls.iter().map(|c| json!({"name": c.name, "arguments": c.arguments})).collect::<Vec<_>>(),
                })
            }
            other => json!({"error": format!("unknown op {other:?}")}),
        };
        writeln!(out, "{reply}")?;
        out.flush()?;
    }
    Ok(())
}

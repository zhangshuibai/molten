//! Development check of the streaming decode loop and the prefix snapshots
//! (`cargo run --release --example stream_check -- MODEL_DIR [--speed ROUNDS]`).
//!
//! For greedy and seeded sampling, on prompts of several lengths:
//! - `generate_stream` over 600 tokens (the 256-slot token ring wraps) returns
//!   exactly the tokens of chained protocol `generate` calls;
//! - stopping the stream at token k (mid-batch, with committed steps beyond it
//!   discarded) and continuing with protocol `generate([], ..)` gives the same
//!   600 tokens: the state ends exactly at the last kept token;
//! - a prefix snapshot restored and prefilled with the rest of the prompt
//!   gives the tokens of a fresh generation cut at the same position, also
//!   after another sequence reused the same snapshot.
//!
//! `--speed N` adds N interleaved rounds of protocol decode vs streamed decode
//! (255 tokens after a 256-token prompt) and prints the rates.
//! Prints one JSON object per check and a summary; exits 1 on any mismatch.

use std::sync::atomic::AtomicBool;
use std::time::Instant;

use anyhow::Result;
use molten::lfm::sample::Sampling;
use molten::lfm::{Flow, LfmModel, LfmState, Mode, PREFILL_CAP, Scratch};
use molten::metal::MetalContext;

const TOTAL: usize = 600;

struct Engine {
    ctx: MetalContext,
    model: LfmModel,
    state: LfmState,
    scratch: Scratch,
    never: AtomicBool,
}

impl Engine {
    fn reset(&mut self) -> Result<()> {
        self.state.reset()?;
        self.state.mode = Mode::Greedy;
        Ok(())
    }

    /// Protocol-path generation of `n` tokens in calls of at most 256.
    fn protocol(&mut self, prompt: &[u32], n: usize, s: Option<&Sampling>) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        let mut first = true;
        while out.len() < n {
            let count = (n - out.len()).min(256);
            let p: &[u32] = if first { prompt } else { &[] };
            first = false;
            out.extend(self.model.generate(&self.ctx, &mut self.state, &mut self.scratch, p, count, s, &self.never)?);
        }
        Ok(out)
    }

    fn stream(
        &mut self,
        prompt: &[u32],
        n: usize,
        s: Option<&Sampling>,
        split: Option<usize>,
        stop_at: Option<usize>,
    ) -> Result<molten::lfm::StreamResult> {
        let mut seen = 0usize;
        let mut cb = |toks: &[u32]| -> Flow {
            let start = seen;
            seen += toks.len();
            match stop_at {
                Some(k) if k < seen => Flow::Stop(k + 1 - start),
                _ => Flow::Continue,
            }
        };
        self.model.generate_stream(&self.ctx, &mut self.state, &mut self.scratch, prompt, n, s, split, &self.never, &mut cb)
    }
}

fn nonce(len: usize, offset: usize) -> Vec<u32> {
    (offset..offset + len).map(|i| ((i as u64 * 2654435761) & 0xFFFF_FFFF) as u32 % 128_000).collect()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).expect("usage: stream_check MODEL_DIR [--speed ROUNDS]");
    let speed_rounds: usize = args
        .iter()
        .position(|a| a == "--speed")
        .and_then(|i| args.get(i + 1))
        .map_or(0, |v| v.parse().expect("--speed ROUNDS"));
    let ctx = MetalContext::new()?;
    let model = LfmModel::load(&ctx, dir)?;
    let state = LfmState::new(&ctx, 16_384)?;
    let scratch = model.scratch(&ctx, PREFILL_CAP)?;
    let mut e = Engine { ctx, model, state, scratch, never: AtomicBool::new(false) };

    // A short chat prompt and nonce prompts across the prefill chunk
    // boundaries.
    let text = molten::text::Text::load(std::path::Path::new(dir))?;
    let user = molten::json::Json::parse(r#"{"role": "user", "content": "What is 17*23? Think step by step."}"#)?;
    let (_, chat) = text.chat_prompt(&[user], None, &[])?;
    let prompts = [("chat17", chat), ("nonce1000", nonce(1000, 777)), ("nonce3000", nonce(3000, 4242))];
    let sampled = Sampling::from_request(Some(&serde_json::json!({
        "temperature": 0.7, "top_k": 50, "top_p": 0.9, "repetition_penalty": 1.05, "seed": 7
    })))?;
    let modes: [(&str, Option<Sampling>); 2] = [("greedy", None), ("sampled", sampled)];
    let mut bad = Vec::new();
    let mut checks = 0usize;
    for (pname, prompt) in &prompts {
        for (mname, s) in &modes {
            let s = s.as_ref();
            e.reset()?;
            let reference = e.protocol(prompt, TOTAL, s)?;
            // A: one stream of TOTAL tokens.
            e.reset()?;
            let got = e.stream(prompt, TOTAL, s, None, None)?.tokens;
            checks += 1;
            let ok = got == reference;
            println!("{}", serde_json::json!({"check": "stream_all", "prompt": pname, "mode": mname, "ok": ok}));
            if !ok {
                bad.push(format!("stream_all {pname} {mname}"));
            }
            // B: stop at k, then continue on the protocol path.
            for k in [0usize, 1, 2, 3, 5, 6, 7, 16, 17, 100, 254, 255, 256, 300] {
                e.reset()?;
                let r = e.stream(prompt, TOTAL, s, None, Some(k))?;
                let pos_ok = e.state.pos == prompt.len() + k && e.state.pending == Some(reference[k]);
                let mut all = r.tokens.clone();
                all.extend(e.protocol(&[], TOTAL - all.len(), s)?);
                checks += 1;
                let ok = all == reference && r.tokens.len() == k + 1 && pos_ok;
                println!("{}", serde_json::json!({"check": "stop_continue", "prompt": pname, "mode": mname, "k": k, "kept": r.tokens.len(), "pos_ok": pos_ok, "ok": ok}));
                if !ok {
                    bad.push(format!("stop_continue {pname} {mname} k={k}"));
                }
            }
            // C: prefix snapshot at q, restored twice (another sequence in between).
            let q = prompt.len() - 4;
            let n = 64;
            e.reset()?;
            let fresh = e.stream(prompt, n, s, Some(q), None)?;
            let mut other: Vec<u32> = prompt[..q].to_vec();
            other.extend([1000u32, 2000, 3000, 4000, 5000, 6000]);
            e.reset()?;
            let fresh_other = e.stream(&other, n, s, Some(q), None)?.tokens;
            // Rebuild the KV rows below q with the snapshot's own prefill, then reuse it.
            e.reset()?;
            let again = e.stream(prompt, n, s, Some(q), None)?;
            let snap2 = again.snapshot.expect("snapshot");
            e.model.restore_prefix(&mut e.state, &snap2)?;
            let restored = e.stream(&prompt[q..], n, s, None, None)?.tokens;
            e.model.restore_prefix(&mut e.state, &snap2)?;
            let restored_other = e.stream(&other[q..], n, s, None, None)?.tokens;
            e.model.restore_prefix(&mut e.state, &snap2)?;
            let restored_again = e.stream(&prompt[q..], n, s, None, None)?.tokens;
            for (name, got, want) in [
                ("prefix_restore", &restored, &fresh.tokens),
                ("prefix_restore_other", &restored_other, &fresh_other),
                ("prefix_restore_again", &restored_again, &fresh.tokens),
            ] {
                checks += 1;
                let ok = got == want;
                println!("{}", serde_json::json!({"check": name, "prompt": pname, "mode": mname, "q": q, "ok": ok}));
                if !ok {
                    bad.push(format!("{name} {pname} {mname}"));
                }
            }
        }
    }

    if speed_rounds > 0 {
        // Interleaved: protocol generate(P, 1) + generate((), 255) vs a stream
        // of 256; the decode time runs from the first token to the last.
        let prompt = nonce(256, 99);
        let mut proto = Vec::new();
        let mut streamed = Vec::new();
        for round in 0..speed_rounds {
            for which in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
                e.reset()?;
                if which == 0 {
                    e.model.generate(&e.ctx, &mut e.state, &mut e.scratch, &prompt, 1, None, &e.never)?;
                    let t = Instant::now();
                    e.model.generate(&e.ctx, &mut e.state, &mut e.scratch, &[], 255, None, &e.never)?;
                    proto.push(255.0 / t.elapsed().as_secs_f64());
                } else {
                    let mut first: Option<Instant> = None;
                    let mut cb = |_: &[u32]| {
                        first.get_or_insert_with(Instant::now);
                        Flow::Continue
                    };
                    e.model.generate_stream(&e.ctx, &mut e.state, &mut e.scratch, &prompt, 256, None, None, &e.never, &mut cb)?;
                    streamed.push(255.0 / first.expect("first token").elapsed().as_secs_f64());
                }
            }
        }
        let median = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).expect("rate"));
            v[v.len() / 2]
        };
        println!(
            "{}",
            serde_json::json!({"check": "speed", "rounds": speed_rounds, "protocol_tok_s": proto.clone(), "stream_tok_s": streamed.clone(),
                "protocol_median": median(&mut proto), "stream_median": median(&mut streamed)})
        );
    }
    println!("{}", serde_json::json!({"summary": true, "checks": checks, "bad": bad}));
    if !bad.is_empty() {
        std::process::exit(1);
    }
    Ok(())
}

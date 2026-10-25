//! The `molten` executable: the subcommands of `molten::cli` (`run`, `chat`,
//! `serve`, `text`), or the JSON-lines engine protocol when no subcommand is
//! given: `molten --model <dir> --contract-json <json>` (docs/PROTOCOL.md).

use std::process::ExitCode;

fn usage() -> ExitCode {
    eprintln!("molten: usage: molten --model <dir> --contract-json <json> (or molten --help)");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match argv.first().map(String::as_str) {
        Some("run" | "chat" | "serve" | "text") => {
            return match molten::cli::main(argv) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("molten: {e:#}");
                    ExitCode::from(1)
                }
            };
        }
        Some("--help" | "-h" | "help") => {
            print!("{}", molten::cli::USAGE);
            return ExitCode::SUCCESS;
        }
        Some("--version" | "-V") => {
            println!("molten {}", env!("CARGO_PKG_VERSION"));
            return ExitCode::SUCCESS;
        }
        _ => {}
    }
    let mut args = argv.into_iter();
    let mut model: Option<String> = None;
    let mut contract: Option<String> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = args.next(),
            "--contract-json" => contract = args.next(),
            _ => return usage(),
        }
    }
    let (Some(model), Some(contract)) = (model, contract) else {
        return usage();
    };
    match molten::protocol::run(&model, &contract) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("molten: {e:#}");
            ExitCode::from(1)
        }
    }
}

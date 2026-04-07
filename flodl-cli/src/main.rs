//! flodl-cli: command-line tool for the floDl deep learning framework.
//!
//! Provides hardware diagnostics, libtorch management, and project scaffolding.
//! This binary is compiled inside Docker and invoked via the `fdl` shell script.

mod diagnose;
mod init;

use std::env;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");

    match cmd {
        "diagnose" => {
            let json = args.iter().any(|a| a == "--json");
            diagnose::run(json);
            ExitCode::SUCCESS
        }
        "init" => {
            let name = args.get(2).map(String::as_str);
            let docker = args.iter().any(|a| a == "--docker");
            match init::run(name, docker) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("error: {}", e);
                    ExitCode::FAILURE
                }
            }
        }
        "help" | "--help" | "-h" => {
            print_usage();
            ExitCode::SUCCESS
        }
        "version" | "--version" | "-V" => {
            println!("flodl-cli {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown command: {}", other);
            eprintln!();
            print_usage();
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    println!("flodl-cli {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("USAGE:");
    println!("    fdl <command> [options]");
    println!();
    println!("COMMANDS:");
    println!("    setup              Detect hardware, download libtorch, build Docker image");
    println!("    init <name>        Scaffold a new floDl project");
    println!("        --docker       Generate Docker-based scaffold (libtorch baked in)");
    println!("    diagnose           System and GPU diagnostics");
    println!("        --json         Output as JSON");
    println!("    help               Show this help");
    println!("    version            Show version");
    println!();
    println!("EXAMPLES:");
    println!("    fdl setup                  # first-time setup");
    println!("    fdl init my-model          # scaffold with mounted libtorch");
    println!("    fdl init my-model --docker # scaffold with Docker (standalone)");
    println!("    fdl diagnose               # hardware + compatibility report");
    println!("    fdl diagnose --json        # machine-readable output");
}

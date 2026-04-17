//! Argv parser and `FdlArgs` trait — the library side of the
//! `#[derive(FdlArgs)]` machinery.
//!
//! The derive macro in `flodl-cli-macros` emits an `impl FdlArgsTrait for
//! Cli` that delegates to the parser exposed here. Binary authors do not
//! import this module directly — they use `#[derive(FdlArgs)]` and
//! `parse_or_schema::<Cli>()` from the top-level `flodl_cli` crate.

pub mod parser;

use crate::config::Schema;

/// Trait implemented by `#[derive(FdlArgs)]`. Carries the metadata needed
/// to parse argv into a concrete type and to emit the `--fdl-schema` JSON.
///
/// The name is `FdlArgsTrait` to avoid colliding with the re-exported
/// derive macro `FdlArgs` (which lives in the derive-macro namespace).
/// Users never refer to this trait directly — the derive implements it.
pub trait FdlArgsTrait: Sized {
    /// Parse argv into `Self`. Uses `std::env::args()` by default.
    fn parse() -> Self {
        let args: Vec<String> = std::env::args().collect();
        match Self::try_parse_from(&args) {
            Ok(t) => t,
            Err(msg) => {
                eprintln!("{msg}");
                std::process::exit(2);
            }
        }
    }

    /// Parse from an explicit argv slice. First element is the program
    /// name (ignored), following elements are flags/values/positionals.
    fn try_parse_from(args: &[String]) -> Result<Self, String>;

    /// Return the JSON schema for this CLI shape.
    fn schema() -> Schema;

    /// Render `--help` to a string.
    fn render_help() -> String;
}

/// Intercept `--fdl-schema` and `--help`, otherwise parse argv.
///
/// - `--fdl-schema` anywhere in argv: print the JSON schema to stdout, exit 0.
/// - `--help` / `-h` anywhere in argv: print help to stdout, exit 0.
/// - Otherwise: parse via `T::try_parse_from`.
pub fn parse_or_schema<T: FdlArgsTrait>() -> T {
    let argv: Vec<String> = std::env::args().collect();
    parse_or_schema_from::<T>(&argv)
}

/// Slice-based variant of [`parse_or_schema`]. The first element is the
/// program name (displayed in help text), the rest are arguments.
///
/// Used by the `fdl` driver itself when dispatching to sub-commands: each
/// sub-command parses its own `args[2..]` tail without re-reading `env::args`.
pub fn parse_or_schema_from<T: FdlArgsTrait>(argv: &[String]) -> T {
    if argv.iter().any(|a| a == "--fdl-schema") {
        let schema = T::schema();
        let json = serde_json::to_string_pretty(&schema)
            .expect("Schema serializes cleanly by construction");
        println!("{json}");
        std::process::exit(0);
    }
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", T::render_help());
        std::process::exit(0);
    }
    match T::try_parse_from(argv) {
        Ok(t) => t,
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(2);
        }
    }
}

//! Comparison tables, baseline validation, and post-hoc report generation.

use std::collections::HashMap;
use std::fmt::Write;

use crate::analyze::RunAnalysis;
use crate::harness::RunResult;

// ---------------------------------------------------------------------------
// Baselines
// ---------------------------------------------------------------------------

/// A baseline entry: expected loss for a (model, mode) pair.
#[derive(Debug, Clone)]
pub struct Baseline {
    pub model: String,
    pub mode: String,
    pub loss: f64,
    pub epochs: usize,
    pub batches: usize,
    pub batch_size: usize,
}

/// Load baselines from a JSON file.
///
/// Format: `[{"model":"linear","mode":"solo-0","loss":1.23,"epochs":5,"batches":1000,"batch_size":64}, ...]`
pub fn load_baselines(path: &str) -> Result<Vec<Baseline>, String> {
    let data = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {path}: {e}"))?;
    parse_baselines(&data)
}

/// Save baselines to a JSON file.
pub fn save_baselines(path: &str, baselines: &[Baseline]) -> Result<(), String> {
    let mut out = String::from("[\n");
    for (i, b) in baselines.iter().enumerate() {
        if i > 0 {
            out.push_str(",\n");
        }
        out.push_str(&format!(
            "  {{\"model\":\"{}\",\"mode\":\"{}\",\"loss\":{:.6},\"epochs\":{},\"batches\":{},\"batch_size\":{}}}",
            b.model, b.mode, b.loss, b.epochs, b.batches, b.batch_size,
        ));
    }
    out.push_str("\n]\n");
    std::fs::write(path, &out)
        .map_err(|e| format!("cannot write {path}: {e}"))
}

/// Build baselines from run results.
pub fn results_to_baselines(
    results: &[RunResult],
    epochs: usize,
    batches: usize,
    batch_size: usize,
) -> Vec<Baseline> {
    results
        .iter()
        .map(|r| Baseline {
            model: r.model_name.clone(),
            mode: r.mode.clone(),
            loss: r.final_loss,
            epochs,
            batches,
            batch_size,
        })
        .collect()
}

/// Validate run results against baselines. Returns (pass_count, fail_count, messages).
///
/// Matches by model name only (ignoring mode) so any DDP mode can be validated
/// against the solo-0 reference. If multiple baselines exist for a model, uses
/// the first one found.
///
/// A result passes if its final loss is within `tolerance` (relative) of the baseline.
/// Missing baselines are reported but not counted as failures.
pub fn validate_results(
    results: &[RunResult],
    baselines: &[Baseline],
    tolerance: f64,
) -> (usize, usize, Vec<String>) {
    let lookup: HashMap<&str, &Baseline> = baselines
        .iter()
        .map(|b| (b.model.as_str(), b))
        .collect();

    let mut pass = 0;
    let mut fail = 0;
    let mut msgs = Vec::new();

    for r in results {
        if let Some(b) = lookup.get(r.model_name.as_str()) {
            let rel_diff = if b.loss.abs() > 1e-10 {
                (r.final_loss - b.loss).abs() / b.loss.abs()
            } else {
                (r.final_loss - b.loss).abs()
            };

            if rel_diff <= tolerance {
                pass += 1;
                msgs.push(format!(
                    "  PASS  {:<16} {:<20} loss={:.6} (baseline={:.6}, diff={:.1}%)",
                    r.model_name, r.mode, r.final_loss, b.loss, rel_diff * 100.0,
                ));
            } else {
                fail += 1;
                msgs.push(format!(
                    "  FAIL  {:<16} {:<20} loss={:.6} (baseline={:.6}, diff={:.1}%)",
                    r.model_name, r.mode, r.final_loss, b.loss, rel_diff * 100.0,
                ));
            }
        } else {
            msgs.push(format!(
                "  SKIP  {:<16} {:<20} loss={:.6} (no baseline)",
                r.model_name, r.mode, r.final_loss,
            ));
        }
    }

    (pass, fail, msgs)
}

fn parse_baselines(json: &str) -> Result<Vec<Baseline>, String> {
    let val: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| format!("invalid JSON: {e}"))?;
    let arr = val.as_array().ok_or("expected JSON array")?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let model = item["model"].as_str().ok_or("missing model")?.to_string();
        let mode = item["mode"].as_str().ok_or("missing mode")?.to_string();
        let loss = item["loss"].as_f64().ok_or("missing loss")?;
        let epochs = item["epochs"].as_u64().unwrap_or(0) as usize;
        let batches = item["batches"].as_u64().unwrap_or(0) as usize;
        let batch_size = item["batch_size"].as_u64().unwrap_or(0) as usize;
        out.push(Baseline { model, mode, loss, epochs, batches, batch_size });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Comparison tables
// ---------------------------------------------------------------------------

/// Print a comparison table for a set of results (typically one model, all modes).
pub fn print_comparison(results: &[RunResult]) {
    if results.is_empty() {
        return;
    }

    let model = &results[0].model_name;
    let n_epochs = results[0].epoch_times_ms.len();

    eprintln!("\n=== {} ({} epochs) ===\n", model, n_epochs);
    eprintln!(
        "  {:<20} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "mode", "loss", "epoch_ms", "total_s", "syncs", "avg_sync", "idle%_0", "idle%_1",
    );
    eprintln!("  {}", "-".repeat(96));

    for r in results {
        let median_epoch = if r.epoch_times_ms.is_empty() {
            0.0
        } else {
            let mut sorted = r.epoch_times_ms.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };

        let s = &r.timeline_summary;
        let idle0 = s.gpu_idle_pct.first().copied().unwrap_or(0.0);
        let idle1 = s.gpu_idle_pct.get(1).copied().unwrap_or(0.0);

        eprintln!(
            "  {:<20} {:>10.6} {:>9.0}ms {:>9.1}s {:>10} {:>9.1}ms {:>7.1}% {:>7.1}%",
            r.mode,
            r.final_loss,
            median_epoch,
            r.total_ms / 1000.0,
            s.sync_count,
            s.avg_sync_ms,
            idle0,
            idle1,
        );
    }
    eprintln!();
}

/// Print a summary across all models.
pub fn print_matrix(all_results: &[Vec<RunResult>]) {
    eprintln!("\n=== Summary Matrix ===\n");
    for group in all_results {
        if group.is_empty() {
            continue;
        }
        print_comparison(group);
    }
}

// ---------------------------------------------------------------------------
// Post-hoc report from timeline analysis
// ---------------------------------------------------------------------------

/// Generate a markdown report from analyzed runs.
/// `analyses` is grouped by model: Vec<(model, Vec<RunAnalysis>)>.
pub fn generate_report(groups: &[(String, Vec<RunAnalysis>)]) -> String {
    let mut md = String::with_capacity(16_000);

    md.push_str("# DDP Benchmark Report\n\n");

    // Setup
    let n_models = groups.len();
    let n_modes: usize = groups.iter().map(|(_, v)| v.len()).max().unwrap_or(0);
    let _ = writeln!(md, "- **Models**: {n_models}");
    let _ = writeln!(md, "- **Modes**: {n_modes}");

    // Speed ratio from solo runs
    if let Some(ratio) = speed_ratio(groups) {
        let _ = writeln!(md, "- **GPU speed ratio**: {ratio:.2}x (solo-0 / solo-1)");
    }
    md.push('\n');

    // Per-model comparison
    md.push_str("## Per-Model Results\n\n");
    for (model, runs) in groups {
        write_model_table(&mut md, model, runs);
    }

    // Speedup vs sync
    if groups.iter().any(|(_, runs)| runs.len() > 1) {
        md.push_str("## Speedup vs Sync\n\n");
        write_speedup_table(&mut md, groups);
    }

    // Idle analysis (the main event)
    md.push_str("## GPU Idle Analysis\n\n");
    md.push_str("Idle gaps >= 500ms, classified by nearest event.\n\n");
    for (model, runs) in groups {
        write_idle_analysis(&mut md, model, runs);
    }

    // Idle summary by cause
    md.push_str("## Idle Breakdown by Cause\n\n");
    write_idle_breakdown(&mut md, groups);

    // ElChe details (anchor + throttle)
    if groups.iter().any(|(_, runs)| runs.iter().any(|r| r.anchor_changes > 0)) {
        md.push_str("## ElChe Calibration\n\n");
        write_elche_details(&mut md, groups);
    }

    md
}

fn speed_ratio(groups: &[(String, Vec<RunAnalysis>)]) -> Option<f64> {
    let mut ratios = Vec::new();
    for (_, runs) in groups {
        let s0 = runs.iter().find(|r| r.mode == "solo-0");
        let s1 = runs.iter().find(|r| r.mode == "solo-1");
        if let (Some(a), Some(b)) = (s0, s1) {
            if a.total_ms > 0 {
                ratios.push(b.total_ms as f64 / a.total_ms as f64);
            }
        }
    }
    if ratios.is_empty() {
        return None;
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(ratios[ratios.len() / 2])
}

fn write_model_table(md: &mut String, model: &str, runs: &[RunAnalysis]) {
    if runs.is_empty() {
        return;
    }
    let _ = writeln!(md, "### {model}\n");
    md.push_str("| Mode | Loss | Total (s) | Syncs | Avg Sync (ms) | GPU0 | GPU1 | Idle (s) |\n");
    md.push_str("|------|------|-----------|-------|--------------|------|------|----------|\n");

    for r in runs {
        let a0 = r.gpu_active_pct.first().copied().unwrap_or(0.0);
        let a1 = r.gpu_active_pct.get(1).copied().unwrap_or(0.0);
        let total_idle_s: f64 = r.idle_by_cause.iter()
            .map(|c| c.total_ms)
            .sum::<f64>() / 1000.0;

        let _ = writeln!(
            md,
            "| {} | {:.6} | {:.1} | {} | {:.1} | {:.0}% | {:.0}% | {:.1} |",
            r.mode,
            r.final_loss,
            r.total_ms as f64 / 1000.0,
            r.sync_count,
            r.avg_sync_ms,
            a0,
            a1,
            total_idle_s,
        );
    }
    md.push('\n');
}

fn write_speedup_table(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
    // Collect mode names from first group
    let modes: Vec<&str> = if let Some((_, runs)) = groups.first() {
        runs.iter().map(|r| r.mode.as_str()).collect()
    } else {
        return;
    };

    md.push_str("| Model |");
    for m in &modes {
        if *m == "sync" { continue; }
        let _ = write!(md, " {m} |");
    }
    md.push('\n');

    md.push_str("|-------|");
    for m in &modes {
        if *m == "sync" { continue; }
        let _ = write!(md, "{}|", "-".repeat(m.len() + 2));
    }
    md.push('\n');

    for (model, runs) in groups {
        let sync_ms = runs.iter()
            .find(|r| r.mode == "sync")
            .map(|r| r.total_ms as f64)
            .unwrap_or(0.0);

        let _ = write!(md, "| {model} |");
        for m in &modes {
            if *m == "sync" { continue; }
            if let Some(r) = runs.iter().find(|r| r.mode == *m) {
                if sync_ms > 0.0 && r.total_ms > 0 {
                    let _ = write!(md, " {:.1}x |", sync_ms / r.total_ms as f64);
                } else {
                    md.push_str(" - |");
                }
            } else {
                md.push_str(" - |");
            }
        }
        md.push('\n');
    }
    md.push('\n');
}

fn write_idle_analysis(md: &mut String, model: &str, runs: &[RunAnalysis]) {
    // Only show runs with idle gaps
    let runs_with_gaps: Vec<&RunAnalysis> = runs.iter()
        .filter(|r| !r.idle_gaps.is_empty())
        .collect();

    if runs_with_gaps.is_empty() {
        return;
    }

    let _ = writeln!(md, "### {model}\n");
    md.push_str("| Mode | GPU | Start (s) | Duration (s) | Cause |\n");
    md.push_str("|------|-----|-----------|-------------|-------|\n");

    for r in &runs_with_gaps {
        // Skip startup gaps, sort by duration descending
        let mut gaps: Vec<&crate::analyze::IdleGap> = r.idle_gaps.iter()
            .filter(|g| !matches!(g.cause, crate::analyze::IdleCause::Startup))
            .collect();
        gaps.sort_by(|a, b| b.duration_ms.cmp(&a.duration_ms));

        // Show top 10 longest gaps per run
        for g in gaps.iter().take(10) {
            let _ = writeln!(
                md,
                "| {} | gpu{} | {:.1} | {:.1} | {} |",
                r.mode,
                g.device,
                g.start_ms as f64 / 1000.0,
                g.duration_ms as f64 / 1000.0,
                g.cause,
            );
        }
    }
    md.push('\n');
}

fn write_idle_breakdown(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
    md.push_str("| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |\n");
    md.push_str("|-------|------|-----|---------------|------|---------|-------------|------------|\n");

    for (model, runs) in groups {
        for r in runs {
            // Skip solo modes and runs with no idle
            if r.mode.starts_with("solo") {
                continue;
            }
            for c in &r.idle_by_cause {
                if c.total_ms < 500.0 {
                    continue; // skip negligible
                }
                let _ = writeln!(
                    md,
                    "| {} | {} | gpu{} | {:.1}s | {:.1}s | {:.1}s | {:.1}s | {:.1}s |",
                    model,
                    r.mode,
                    c.device,
                    c.epoch_boundary_ms / 1000.0,
                    c.sync_ms / 1000.0,
                    c.cpu_avg_ms / 1000.0,
                    c.unexplained_ms / 1000.0,
                    c.total_ms / 1000.0,
                );
            }
        }
    }
    md.push('\n');
}

fn write_elche_details(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
    md.push_str("| Model | Mode | Anchor Changes | Throttles | Syncs | CPU Avgs | Avg CPU Avg (ms) |\n");
    md.push_str("|-------|------|---------------|-----------|-------|---------|----------------|\n");

    for (model, runs) in groups {
        for r in runs {
            if r.anchor_changes == 0 && r.throttle_count == 0 {
                continue;
            }
            let _ = writeln!(
                md,
                "| {} | {} | {} | {} | {} | {} | {:.1} |",
                model,
                r.mode,
                r.anchor_changes,
                r.throttle_count,
                r.sync_count,
                r.cpu_avg_count,
                r.avg_cpu_avg_ms,
            );
        }
    }
    md.push('\n');
}

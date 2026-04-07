//! `fdl diagnose` -- system and GPU diagnostics.

use std::fmt::Write;
use std::fs;
use std::path::Path;
use std::process::Command;

pub fn run(json: bool) {
    if json {
        print_json();
    } else {
        print_report();
    }
}

// ---------------------------------------------------------------------------
// Human-readable report
// ---------------------------------------------------------------------------

fn print_report() {
    println!("floDl Diagnostics");
    println!("=================");
    println!();

    // System
    println!("System");
    let cpu = cpu_model().unwrap_or_else(|| "Unknown".into());
    let threads = cpu_threads();
    let ram_gb = ram_total_gb();
    println!("  CPU:         {} ({} threads, {}GB RAM)", cpu, threads, ram_gb);
    if let Some(os) = os_version() {
        println!("  OS:          {}", os);
    }
    if is_inside_docker() {
        println!("  Docker:      yes (running inside container)");
    } else {
        match docker_version() {
            Some(v) => println!("  Docker:      {}", v),
            None => println!("  Docker:      not found"),
        }
    }
    println!();

    // CUDA / GPU
    println!("CUDA");
    if flodl::cuda_available() {
        let n = flodl::cuda_device_count();
        println!("  Devices:     {}", n);
        let devices = flodl::cuda_devices();
        for d in &devices {
            let vram_gb = d.total_memory / (1024 * 1024 * 1024);
            println!(
                "  [{}] {} -- {}, {}GB VRAM",
                d.index, d.name, d.sm_version(), vram_gb
            );
            match flodl::probe_device(flodl::Device::CUDA(d.index)) {
                Ok(()) => println!("      Probe: OK"),
                Err(e) => println!("      Probe: FAILED -- {}", e),
            }
        }
    } else {
        println!("  No CUDA devices available");
    }
    println!();

    // libtorch
    println!("libtorch");
    match read_active_libtorch() {
        Some(info) => {
            println!("  Active:      {}", info.path);
            if let Some(v) = &info.torch_version {
                println!("  Version:     {}", v);
            }
            if let Some(c) = &info.cuda_version {
                println!("  CUDA:        {}", c);
            }
            if let Some(a) = &info.archs {
                println!("  Archs:       {}", a);
            }
            if let Some(s) = &info.source {
                println!("  Source:      {}", s);
            }
        }
        None => {
            println!("  No active variant (run `fdl setup`)");
        }
    }

    // List available variants
    let variants = list_libtorch_variants();
    if !variants.is_empty() {
        println!("  Variants:    {}", variants.join(", "));
    }
    println!();

    // Compatibility
    if flodl::cuda_available() {
        println!("Compatibility");
        if let Some(info) = read_active_libtorch() {
            let devices = flodl::cuda_devices();
            let archs = info.archs.as_deref().unwrap_or("");
            let mut all_ok = true;
            for d in &devices {
                let arch_str = format!("{}.{}", d.sm_major, d.sm_minor);
                let covered = archs.contains(&arch_str)
                    || archs.contains(&format!("{}", d.sm_major));
                let probe_ok = flodl::probe_device(flodl::Device::CUDA(d.index)).is_ok();
                if probe_ok {
                    println!(
                        "  GPU {} ({}, {}):  OK",
                        d.index,
                        short_gpu_name(&d.name),
                        d.sm_version()
                    );
                } else if !covered {
                    all_ok = false;
                    println!(
                        "  GPU {} ({}, {}):  MISSING -- arch {} not in [{}]",
                        d.index,
                        short_gpu_name(&d.name),
                        d.sm_version(),
                        arch_str,
                        archs
                    );
                } else {
                    all_ok = false;
                    println!(
                        "  GPU {} ({}, {}):  FAILED",
                        d.index,
                        short_gpu_name(&d.name),
                        d.sm_version()
                    );
                }
            }
            if all_ok {
                println!();
                println!("  All GPUs compatible with active libtorch.");
            }
        } else {
            println!("  Cannot check -- no active libtorch variant.");
        }
        println!();
    }
}

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------

fn print_json() {
    let mut b = String::with_capacity(2048);
    b.push('{');

    // System
    let cpu = cpu_model().unwrap_or_else(|| "Unknown".into());
    let _ = write!(
        b,
        "\"system\":{{\"cpu\":\"{}\",\"threads\":{},\"ram_gb\":{}",
        escape_json(&cpu),
        cpu_threads(),
        ram_total_gb()
    );
    if let Some(os) = os_version() {
        let _ = write!(b, ",\"os\":\"{}\"", escape_json(&os));
    }
    if is_inside_docker() {
        b.push_str(",\"docker\":\"container\"");
    } else if let Some(docker) = docker_version() {
        let _ = write!(b, ",\"docker\":\"{}\"", escape_json(&docker));
    }
    b.push('}');

    // GPUs
    b.push_str(",\"gpus\":[");
    if flodl::cuda_available() {
        let devices = flodl::cuda_devices();
        for (i, d) in devices.iter().enumerate() {
            if i > 0 { b.push(','); }
            let probe_ok = flodl::probe_device(flodl::Device::CUDA(d.index)).is_ok();
            let _ = write!(
                b,
                "{{\"index\":{},\"name\":\"{}\",\"sm\":\"{}\",\"vram_bytes\":{},\"probe\":{}}}",
                d.index,
                escape_json(&d.name),
                d.sm_version(),
                d.total_memory,
                probe_ok
            );
        }
    }
    b.push(']');

    // libtorch
    b.push_str(",\"libtorch\":");
    match read_active_libtorch() {
        Some(info) => {
            let _ = write!(b, "{{\"path\":\"{}\"", escape_json(&info.path));
            if let Some(v) = &info.torch_version {
                let _ = write!(b, ",\"version\":\"{}\"", escape_json(v));
            }
            if let Some(c) = &info.cuda_version {
                let _ = write!(b, ",\"cuda\":\"{}\"", escape_json(c));
            }
            if let Some(a) = &info.archs {
                let _ = write!(b, ",\"archs\":\"{}\"", escape_json(a));
            }
            if let Some(s) = &info.source {
                let _ = write!(b, ",\"source\":\"{}\"", escape_json(s));
            }
            b.push('}');
        }
        None => b.push_str("null"),
    }

    b.push('}');
    println!("{}", b);
}

// ---------------------------------------------------------------------------
// System info helpers
// ---------------------------------------------------------------------------

fn cpu_model() -> Option<String> {
    let info = fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in info.lines() {
        if let Some(rest) = line.strip_prefix("model name") {
            if let Some(val) = rest.split(':').nth(1) {
                return Some(val.trim().to_string());
            }
        }
    }
    None
}

fn cpu_threads() -> usize {
    fs::read_to_string("/proc/cpuinfo")
        .ok()
        .map(|s| s.lines().filter(|l| l.starts_with("processor")).count())
        .unwrap_or(1)
}

fn ram_total_gb() -> u64 {
    fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
                    return Some(kb / (1024 * 1024));
                }
            }
            None
        })
        .unwrap_or(0)
}

fn os_version() -> Option<String> {
    let uname = Command::new("uname").arg("-r").output().ok()?;
    let kernel = String::from_utf8_lossy(&uname.stdout).trim().to_string();
    let wsl = if kernel.contains("WSL") || kernel.contains("microsoft") {
        " (WSL2)"
    } else {
        ""
    };
    Some(format!("Linux {}{}", kernel, wsl))
}

fn is_inside_docker() -> bool {
    Path::new("/.dockerenv").exists()
}

fn docker_version() -> Option<String> {
    let out = Command::new("docker").arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    // "Docker version 26.1.3, build ..." -> "26.1.3"
    s.split("version ")
        .nth(1)
        .and_then(|v| v.split(',').next())
        .map(|v| v.trim().to_string())
}

// ---------------------------------------------------------------------------
// libtorch info
// ---------------------------------------------------------------------------

struct LibtorchInfo {
    path: String,
    torch_version: Option<String>,
    cuda_version: Option<String>,
    archs: Option<String>,
    source: Option<String>,
}

fn read_active_libtorch() -> Option<LibtorchInfo> {
    let active = fs::read_to_string("libtorch/.active").ok()?;
    let path = active.trim().to_string();
    if path.is_empty() {
        return None;
    }

    let arch_path = format!("libtorch/{}/.arch", path);
    let mut info = LibtorchInfo {
        path,
        torch_version: None,
        cuda_version: None,
        archs: None,
        source: None,
    };

    if let Ok(arch_content) = fs::read_to_string(&arch_path) {
        for line in arch_content.lines() {
            if let Some(val) = line.strip_prefix("torch=") {
                info.torch_version = Some(val.to_string());
            } else if let Some(val) = line.strip_prefix("cuda=") {
                info.cuda_version = Some(val.to_string());
            } else if let Some(val) = line.strip_prefix("archs=") {
                info.archs = Some(val.to_string());
            } else if let Some(val) = line.strip_prefix("source=") {
                info.source = Some(val.to_string());
            }
        }
    }

    Some(info)
}

fn list_libtorch_variants() -> Vec<String> {
    let mut variants = Vec::new();

    // Precompiled variants
    if let Ok(entries) = fs::read_dir("libtorch/precompiled") {
        for entry in entries.flatten() {
            if entry.path().join("lib").is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    variants.push(format!("precompiled/{}", name));
                }
            }
        }
    }

    // Source builds
    if let Ok(entries) = fs::read_dir("libtorch/builds") {
        for entry in entries.flatten() {
            if entry.path().join("lib").is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    variants.push(format!("builds/{}", name));
                }
            }
        }
    }

    variants.sort();
    variants
}

fn short_gpu_name(name: &str) -> String {
    name.replace("NVIDIA ", "").replace("GeForce ", "")
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

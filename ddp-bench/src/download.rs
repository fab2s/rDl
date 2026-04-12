//! Dataset download and caching for ddp-bench.
//!
//! Downloads standard datasets on first use and caches raw files to disk.
//! Parsing is handled by the flodl dataset parsers in `flodl::data::datasets`.

use std::fs;
use std::io::{Read, Write};
use std::path::Path;

use flodl::data::datasets::{Cifar10, Mnist, Shakespeare};
use flodl::tensor::{Result, TensorError};

// ---------------------------------------------------------------------------
// MNIST
// ---------------------------------------------------------------------------

const MNIST_BASE: &str = "https://storage.googleapis.com/cvdf-datasets/mnist";
const MNIST_TRAIN_IMAGES: &str = "train-images-idx3-ubyte.gz";
const MNIST_TRAIN_LABELS: &str = "train-labels-idx1-ubyte.gz";

/// Download MNIST training data (if not cached) and parse it.
///
/// Files are cached in `{data_dir}/mnist/`.
/// Returns 60,000 images `[N, 1, 28, 28]` and labels `[N]`.
pub fn ensure_mnist(data_dir: &Path) -> Result<Mnist> {
    let dir = data_dir.join("mnist");
    ensure_dir(&dir)?;

    let images_path = dir.join(MNIST_TRAIN_IMAGES);
    let labels_path = dir.join(MNIST_TRAIN_LABELS);

    if !images_path.exists() {
        let url = format!("{MNIST_BASE}/{MNIST_TRAIN_IMAGES}");
        download_to_file(&url, &images_path)?;
    }
    if !labels_path.exists() {
        let url = format!("{MNIST_BASE}/{MNIST_TRAIN_LABELS}");
        download_to_file(&url, &labels_path)?;
    }

    eprintln!("  parsing MNIST...");
    let images_gz = read_file(&images_path)?;
    let labels_gz = read_file(&labels_path)?;
    Mnist::parse(&images_gz, &labels_gz)
}

// ---------------------------------------------------------------------------
// CIFAR-10
// ---------------------------------------------------------------------------

const CIFAR10_URL: &str = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
const CIFAR10_TRAIN_BATCHES: [&str; 5] = [
    "data_batch_1.bin",
    "data_batch_2.bin",
    "data_batch_3.bin",
    "data_batch_4.bin",
    "data_batch_5.bin",
];

/// Download CIFAR-10 training data (if not cached) and parse it.
///
/// Files are cached in `{data_dir}/cifar10/`.
/// Returns 50,000 images `[N, 3, 32, 32]` and labels `[N]`.
pub fn ensure_cifar10(data_dir: &Path) -> Result<Cifar10> {
    let dir = data_dir.join("cifar10");
    ensure_dir(&dir)?;

    let all_present = CIFAR10_TRAIN_BATCHES
        .iter()
        .all(|name| dir.join(name).exists());

    if !all_present {
        // CIFAR-10 tar.gz is ~170MB, exceeds ureq default body limit.
        // Stream to a temp file, then extract.
        let tar_path = dir.join("cifar-10-binary.tar.gz");
        download_large_to_file(CIFAR10_URL, &tar_path)?;
        let tar_gz = read_file(&tar_path)?;
        extract_cifar10(&tar_gz, &dir)?;
        let _ = fs::remove_file(&tar_path); // clean up
    }

    eprintln!("  parsing CIFAR-10...");
    let mut batch_data: Vec<Vec<u8>> = Vec::with_capacity(5);
    for name in &CIFAR10_TRAIN_BATCHES {
        batch_data.push(read_file(&dir.join(name))?);
    }
    let refs: Vec<&[u8]> = batch_data.iter().map(|v| v.as_slice()).collect();
    Cifar10::parse(&refs)
}

/// Extract CIFAR-10 batch files from the tar.gz archive.
fn extract_cifar10(tar_gz: &[u8], out_dir: &Path) -> Result<()> {
    let gz = flate2::read::GzDecoder::new(tar_gz);
    let mut archive = tar::Archive::new(gz);

    for entry in archive
        .entries()
        .map_err(|e| TensorError::new(&format!("tar entries: {e}")))?
    {
        let mut entry =
            entry.map_err(|e| TensorError::new(&format!("tar entry: {e}")))?;

        let path = entry
            .path()
            .map_err(|e| TensorError::new(&format!("tar path: {e}")))?
            .to_path_buf();

        // Extract only .bin files (data_batch_*.bin and test_batch.bin)
        if let Some(name) = path.file_name().and_then(std::ffi::OsStr::to_str)
            && name.ends_with(".bin")
        {
            let dest = out_dir.join(name);
            let mut buf = Vec::new();
            entry
                .read_to_end(&mut buf)
                .map_err(|e| TensorError::new(&format!("read {name}: {e}")))?;
            fs::write(&dest, &buf).map_err(|e| {
                TensorError::new(&format!("write {}: {e}", dest.display()))
            })?;
            eprintln!("    extracted {name} ({} bytes)", buf.len());
        }
    }

    // Verify all training batches were extracted
    for name in &CIFAR10_TRAIN_BATCHES {
        if !out_dir.join(name).exists() {
            return Err(TensorError::new(&format!(
                "CIFAR-10 archive missing {name}"
            )));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Shakespeare
// ---------------------------------------------------------------------------

const SHAKESPEARE_URL: &str =
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

/// Download Shakespeare text (if not cached) and parse into sequences.
///
/// The file is cached in `{data_dir}/shakespeare/input.txt`.
pub fn ensure_shakespeare(data_dir: &Path, seq_len: usize) -> Result<Shakespeare> {
    let dir = data_dir.join("shakespeare");
    ensure_dir(&dir)?;

    let path = dir.join("input.txt");
    if !path.exists() {
        download_to_file(SHAKESPEARE_URL, &path)?;
    }

    let text = fs::read_to_string(&path)
        .map_err(|e| TensorError::new(&format!("read {}: {e}", path.display())))?;

    eprintln!(
        "  parsing Shakespeare ({} chars, seq_len={seq_len})...",
        text.len()
    );
    Shakespeare::parse(&text, seq_len)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn ensure_dir(dir: &Path) -> Result<()> {
    fs::create_dir_all(dir)
        .map_err(|e| TensorError::new(&format!("mkdir {}: {e}", dir.display())))
}

fn read_file(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|e| TensorError::new(&format!("read {}: {e}", path.display())))
}

fn download_to_file(url: &str, dest: &Path) -> Result<()> {
    let bytes = download_bytes(url)?;
    fs::write(dest, &bytes)
        .map_err(|e| TensorError::new(&format!("write {}: {e}", dest.display())))
}

fn download_bytes(url: &str) -> Result<Vec<u8>> {
    eprintln!("    downloading {url}...");
    let buf = ureq::get(url)
        .call()
        .map_err(|e| TensorError::new(&format!("GET {url}: {e}")))?
        .into_body()
        .read_to_vec()
        .map_err(|e| TensorError::new(&format!("read {url}: {e}")))?;
    eprintln!("    {} bytes", buf.len());
    Ok(buf)
}

/// Stream a large download directly to a file (bypasses ureq body size limit).
fn download_large_to_file(url: &str, dest: &Path) -> Result<()> {
    eprintln!("    downloading {url}...");
    let resp = ureq::get(url)
        .call()
        .map_err(|e| TensorError::new(&format!("GET {url}: {e}")))?;
    let mut reader = resp.into_body().into_reader();
    let mut file = fs::File::create(dest)
        .map_err(|e| TensorError::new(&format!("create {}: {e}", dest.display())))?;
    let mut buf = [0u8; 65536];
    let mut total = 0usize;
    loop {
        let n = reader.read(&mut buf)
            .map_err(|e| TensorError::new(&format!("read {url}: {e}")))?;
        if n == 0 { break; }
        file.write_all(&buf[..n])
            .map_err(|e| TensorError::new(&format!("write {}: {e}", dest.display())))?;
        total += n;
    }
    eprintln!("    {total} bytes");
    Ok(())
}

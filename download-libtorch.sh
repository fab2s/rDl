#!/bin/sh
# Download and install libtorch for native (non-Docker) floDl development.
#
# Usage:
#   sh download-libtorch.sh                    # auto-detect CPU or CUDA
#   sh download-libtorch.sh --cpu              # force CPU-only
#   sh download-libtorch.sh --cuda 12.8        # specific CUDA version
#   sh download-libtorch.sh --path ~/libtorch  # custom install directory
#
# Or via curl:
#   curl -sL https://raw.githubusercontent.com/fab2s/floDl/main/download-libtorch.sh | sh
#   curl -sL https://raw.githubusercontent.com/fab2s/floDl/main/download-libtorch.sh | sh -s -- --cuda 12.8
#
# After install, add the printed exports to your shell profile (~/.bashrc, etc.).

set -e

LIBTORCH_VERSION="2.10.0"
INSTALL_PATH=""
VARIANT=""

# --- Parse arguments ---

while [ $# -gt 0 ]; do
    case "$1" in
        --cpu)
            VARIANT="cpu"
            shift
            ;;
        --cuda)
            VARIANT="${2:?--cuda requires a version (12.6 or 12.8)}"
            shift 2
            ;;
        --path)
            INSTALL_PATH="${2:?--path requires a directory}"
            shift 2
            ;;
        --help|-h)
            head -14 "$0" | tail -12
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: sh download-libtorch.sh [--cpu | --cuda 12.6|12.8] [--path DIR]" >&2
            exit 1
            ;;
    esac
done

# --- Detect platform ---

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux)
        if [ "$ARCH" != "x86_64" ]; then
            echo "error: Linux libtorch is only available for x86_64, got $ARCH" >&2
            exit 1
        fi
        ;;
    Darwin)
        if [ "$ARCH" = "arm64" ]; then
            if [ "$VARIANT" != "" ] && [ "$VARIANT" != "cpu" ]; then
                echo "error: macOS only supports CPU libtorch" >&2
                exit 1
            fi
            VARIANT="cpu"
        else
            echo "error: macOS libtorch requires Apple Silicon (arm64), got $ARCH" >&2
            echo "       macOS x86_64 was dropped after PyTorch 2.2" >&2
            exit 1
        fi
        ;;
    *)
        echo "error: unsupported OS '$OS' — use Linux or macOS" >&2
        echo "       Windows: download from https://pytorch.org/get-started/locally/" >&2
        exit 1
        ;;
esac

# --- Auto-detect CUDA ---

if [ -z "$VARIANT" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        # Parse CUDA version from nvidia-smi
        CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        if [ -n "$CUDA_VER" ]; then
            # Check nvcc for toolkit version (more reliable for matching libtorch)
            if command -v nvcc >/dev/null 2>&1; then
                NVCC_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
                case "$NVCC_VER" in
                    12.8*|12.9*|13.*)  VARIANT="12.8" ;;
                    12.*)              VARIANT="12.6" ;;
                    *)                 VARIANT="12.8" ;;
                esac
                echo "Detected CUDA toolkit $NVCC_VER -> using cu${VARIANT//./}" >&2
            else
                # No nvcc, default to latest
                VARIANT="12.8"
                echo "Detected NVIDIA GPU (no nvcc found) -> using cu128" >&2
            fi
        else
            VARIANT="cpu"
            echo "nvidia-smi found but no GPU detected -> CPU" >&2
        fi
    else
        VARIANT="cpu"
        echo "No NVIDIA GPU detected -> CPU" >&2
    fi
fi

# --- Build download URL ---

case "$VARIANT" in
    cpu)
        if [ "$OS" = "Darwin" ]; then
            FILENAME="libtorch-macos-arm64-${LIBTORCH_VERSION}.zip"
            URL="https://download.pytorch.org/libtorch/cpu/${FILENAME}"
        else
            FILENAME="libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip"
            URL="https://download.pytorch.org/libtorch/cpu/${FILENAME}"
        fi
        LABEL="CPU"
        ;;
    12.6)
        FILENAME="libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcu126.zip"
        URL="https://download.pytorch.org/libtorch/cu126/${FILENAME}"
        LABEL="CUDA 12.6"
        ;;
    12.8)
        FILENAME="libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcu128.zip"
        URL="https://download.pytorch.org/libtorch/cu128/${FILENAME}"
        LABEL="CUDA 12.8"
        ;;
    *)
        echo "error: unsupported CUDA version '$VARIANT'" >&2
        echo "       available: 12.6, 12.8 (or --cpu)" >&2
        exit 1
        ;;
esac

# --- Install path ---

if [ -z "$INSTALL_PATH" ]; then
    INSTALL_PATH="$HOME/.local/lib/libtorch"
fi

PARENT_DIR="$(dirname "$INSTALL_PATH")"

# --- Check for existing installation ---

if [ -d "$INSTALL_PATH" ]; then
    EXISTING_VER=""
    if [ -f "$INSTALL_PATH/build-version" ]; then
        EXISTING_VER=$(cat "$INSTALL_PATH/build-version" 2>/dev/null | head -1)
    fi
    if [ "$EXISTING_VER" = "$LIBTORCH_VERSION" ]; then
        echo "libtorch $LIBTORCH_VERSION already installed at $INSTALL_PATH"
        echo ""
        echo "To reinstall, remove it first:"
        echo "  rm -rf $INSTALL_PATH"
        exit 0
    fi
    echo "Existing libtorch found at $INSTALL_PATH (version: ${EXISTING_VER:-unknown})"
    echo "Removing before installing $LIBTORCH_VERSION..."
    rm -rf "$INSTALL_PATH"
fi

# --- Check dependencies ---

if ! command -v unzip >/dev/null 2>&1; then
    echo "error: unzip is required but not installed" >&2
    echo "  Ubuntu/Debian:  sudo apt install unzip" >&2
    echo "  Fedora/RHEL:    sudo dnf install unzip" >&2
    echo "  macOS:          available by default" >&2
    exit 1
fi

# Prefer wget, fall back to curl
if command -v wget >/dev/null 2>&1; then
    DOWNLOAD="wget -q --show-progress -O"
elif command -v curl >/dev/null 2>&1; then
    DOWNLOAD="curl -L --progress-bar -o"
else
    echo "error: wget or curl is required" >&2
    exit 1
fi

# --- Download ---

mkdir -p "$PARENT_DIR"
TMPFILE=$(mktemp "${TMPDIR:-/tmp}/libtorch-XXXXXX.zip")
trap 'rm -f "$TMPFILE"' EXIT

echo ""
echo "Downloading libtorch $LIBTORCH_VERSION ($LABEL)..."
echo "  $URL"
echo ""
$DOWNLOAD "$TMPFILE" "$URL"

# --- Extract ---

echo ""
echo "Extracting to $INSTALL_PATH ..."

# libtorch zips contain a top-level "libtorch/" directory
unzip -q "$TMPFILE" -d "$PARENT_DIR"

# Rename to target path if needed (zip extracts to $PARENT_DIR/libtorch)
EXTRACTED="$PARENT_DIR/libtorch"
if [ "$EXTRACTED" != "$INSTALL_PATH" ]; then
    mv "$EXTRACTED" "$INSTALL_PATH"
fi

rm -f "$TMPFILE"
trap - EXIT

echo "  done."

# --- Verify ---

if [ ! -f "$INSTALL_PATH/lib/libtorch.so" ] && [ ! -f "$INSTALL_PATH/lib/libtorch.dylib" ]; then
    echo ""
    echo "WARNING: libtorch library not found at expected path." >&2
    echo "         The archive structure may have changed." >&2
    echo "         Check: ls $INSTALL_PATH/lib/" >&2
    exit 1
fi

# --- Print setup instructions ---

echo ""
echo "=================================================="
echo "  libtorch $LIBTORCH_VERSION ($LABEL) installed"
echo "  $INSTALL_PATH"
echo "=================================================="
echo ""
echo "Add these to your shell profile (~/.bashrc, ~/.zshrc, etc.):"
echo ""
echo "  export LIBTORCH_PATH=\"$INSTALL_PATH\""
echo "  export LD_LIBRARY_PATH=\"$INSTALL_PATH/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\""
echo ""
echo "Then reload your shell:"
echo ""
echo "  source ~/.bashrc    # or: source ~/.zshrc"
echo ""
echo "Verify:"
echo ""
echo "  cargo build   # in your floDl project"
echo ""

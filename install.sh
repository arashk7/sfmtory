#!/usr/bin/env bash
#
# Install or uninstall the `sfmtory` binary system-wide.
#
#   ./install.sh              # build release and install
#   ./install.sh --uninstall  # remove an installed copy
#   ./install.sh --prefix ~/.local
#
# Installs to $PREFIX/bin (default /usr/local, or ~/.local when not root),
# re-invoking through sudo only when the chosen prefix needs it.

set -euo pipefail

BIN_NAME="sfmtory"
PREFIX=""
ACTION="install"

usage() {
    sed -n '3,12p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --uninstall) ACTION="uninstall"; shift ;;
        --prefix)    PREFIX="${2:?--prefix needs a path}"; shift 2 ;;
        --prefix=*)  PREFIX="${1#*=}"; shift ;;
        -h|--help)   usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 1 ;;
    esac
done

# Default prefix: system-wide when we can write there, per-user otherwise, so
# the script works without sudo instead of failing on it.
if [[ -z "$PREFIX" ]]; then
    if [[ $EUID -eq 0 ]]; then PREFIX="/usr/local"; else PREFIX="$HOME/.local"; fi
fi
BIN_DIR="$PREFIX/bin"
TARGET="$BIN_DIR/$BIN_NAME"

# True when writing to $1 would need elevation. Walks up to the nearest
# ancestor that actually exists - checking the immediate parent is wrong when
# the whole prefix has yet to be created, which is the common case for a fresh
# --prefix.
needs_sudo() {
    local p="$1"
    while [[ ! -e "$p" ]]; do
        local up; up="$(dirname "$p")"
        [[ "$up" == "$p" ]] && break
        p="$up"
    done
    [[ ! -w "$p" ]]
}

# Run a command, elevating only if the given destination requires it.
run_priv() {
    local dest="$1"; shift
    if [[ $EUID -eq 0 ]] || ! needs_sudo "$dest"; then
        "$@"
    else
        sudo "$@"
    fi
}

if [[ "$ACTION" == "uninstall" ]]; then
    if [[ -e "$TARGET" ]]; then
        run_priv "$TARGET" rm -f "$TARGET"
        echo "Removed $TARGET"
    else
        echo "Nothing to uninstall: $TARGET does not exist"
    fi
    exit 0
fi

command -v cargo >/dev/null 2>&1 || {
    echo "error: cargo not found. Install Rust from https://rustup.rs" >&2
    exit 1
}

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Building $BIN_NAME (release)..."
cargo build --release --manifest-path "$SRC_DIR/Cargo.toml" -p sfm-cli

BUILT="$SRC_DIR/target/release/$BIN_NAME"
[[ -x "$BUILT" ]] || { echo "error: build did not produce $BUILT" >&2; exit 1; }

run_priv "$BIN_DIR" mkdir -p "$BIN_DIR"
run_priv "$BIN_DIR" install -m 0755 "$BUILT" "$TARGET"
echo "Installed $TARGET"

if ! command -v "$BIN_NAME" >/dev/null 2>&1; then
    echo
    echo "note: $BIN_DIR is not on your PATH. Add it with:"
    echo "  echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.bashrc && exec \$SHELL"
else
    echo "Run '$BIN_NAME --help' to get started."
fi

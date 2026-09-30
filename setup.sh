#!/usr/bin/env bash
# Set up a complete music-decoder development environment.
#
# Installs system binaries (ffmpeg, fluidsynth, libsndfile), creates a
# Python 3.11 virtual environment at ./.venv, installs the package with
# dev extras, downloads the synthetic-fixture soundfont, and runs the
# `music-decoder doctor` health check.
#
# Usage:
#     source ./setup.sh         # run setup AND activate the venv in your shell
#     ./setup.sh                # run setup; prints activate command at the end
#     ./setup.sh --no-system    # skip system-package install (need sudo / brew)
#     ./setup.sh --no-soundfont # skip the ~6MB SF2 download
#     ./setup.sh --recreate     # delete and recreate the .venv
#
# Re-running is safe: each step is idempotent.

# ---- parent-shell state (kept around after the strict-mode subshell) -------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"

# Detect whether we're being sourced (works in bash and zsh) — when sourced,
# we activate the venv in the caller's shell at the end.
_setup_sourced=0
if [ -n "${BASH_VERSION:-}" ]; then
  (return 0 2>/dev/null) && _setup_sourced=1
elif [ -n "${ZSH_VERSION:-}" ]; then
  case "${ZSH_EVAL_CONTEXT:-}" in *:file*) _setup_sourced=1 ;; esac
fi

# ---- main work runs in a function, invoked from a strict-mode subshell -----
# A subshell isolates `set -euo pipefail` so it doesn't break the caller's
# interactive shell when this script is sourced.
_setup_main() {
  cd "$PROJECT_ROOT"

  PYTHON_BIN="${PYTHON_BIN:-}"
  SKIP_SYSTEM=0
  SKIP_SOUNDFONT=0
  RECREATE_VENV=0

  log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
  warn() { printf '\033[1;33m[setup]\033[0m %s\n' "$*" >&2; }
  err()  { printf '\033[1;31m[setup]\033[0m %s\n' "$*" >&2; }

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --no-system)    SKIP_SYSTEM=1 ;;
      --no-soundfont) SKIP_SOUNDFONT=1 ;;
      --recreate)     RECREATE_VENV=1 ;;
      --python)       shift; PYTHON_BIN="$1" ;;
      -h|--help)
        sed -n '2,15p' "$PROJECT_ROOT/setup.sh" | sed 's/^# \{0,1\}//'
        return 0
        ;;
      *)
        err "unknown argument: $1"
        return 2
        ;;
    esac
    shift
  done

  # --- 1. Detect OS / package manager --------------------------------------
  OS="$(uname -s)"
  case "$OS" in
    Darwin)  PLATFORM=macos ;;
    Linux)   PLATFORM=linux ;;
    *)       err "unsupported OS: $OS"; return 1 ;;
  esac
  log "platform: $PLATFORM"

  # --- 2. Install system binaries ------------------------------------------
  install_system_macos() {
    if ! command -v brew >/dev/null 2>&1; then
      err "Homebrew not found. Install from https://brew.sh and re-run, or pass --no-system."
      return 1
    fi
    local pkgs=(ffmpeg fluidsynth libsndfile)
    # pyproject pins >=3.11,<3.13 — install python@3.11 if neither is on PATH.
    if ! command -v python3.11 >/dev/null 2>&1 && ! command -v python3.12 >/dev/null 2>&1; then
      pkgs+=(python@3.11)
    fi
    log "brew install ${pkgs[*]}"
    brew install "${pkgs[@]}"
  }

  install_system_linux() {
    local sudo_cmd=""
    [[ $EUID -ne 0 ]] && sudo_cmd="sudo"
    if command -v apt-get >/dev/null 2>&1; then
      log "apt install ffmpeg libfluidsynth3 libsndfile1 build-essential pkg-config python3.11 python3.11-venv python3.11-dev"
      $sudo_cmd apt-get update
      $sudo_cmd apt-get install -y --no-install-recommends \
        ffmpeg libfluidsynth3 libsndfile1 \
        build-essential pkg-config \
        python3.11 python3.11-venv python3.11-dev
    elif command -v dnf >/dev/null 2>&1; then
      log "dnf install ffmpeg fluidsynth libsndfile python3.11"
      $sudo_cmd dnf install -y ffmpeg fluidsynth libsndfile python3.11 python3.11-devel gcc pkgconf-pkg-config
    elif command -v pacman >/dev/null 2>&1; then
      log "pacman -S ffmpeg fluidsynth libsndfile python"
      $sudo_cmd pacman -S --noconfirm --needed ffmpeg fluidsynth libsndfile python base-devel pkgconf
    else
      err "no supported package manager found (apt, dnf, pacman). Install ffmpeg + fluidsynth + libsndfile manually, or pass --no-system."
      return 1
    fi
  }

  if [[ $SKIP_SYSTEM -eq 0 ]]; then
    log "installing system binaries…"
    if [[ $PLATFORM == macos ]]; then
      install_system_macos
    else
      install_system_linux
    fi
  else
    log "skipping system-package install (--no-system)"
  fi

  # --- 3. Pick a Python interpreter ----------------------------------------
  pick_python() {
    if [[ -n "$PYTHON_BIN" ]]; then
      echo "$PYTHON_BIN"
      return
    fi
    for candidate in python3.11 python3.12 python3; do
      if command -v "$candidate" >/dev/null 2>&1; then
        local v
        v="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
        case "$v" in
          3.11|3.12) echo "$candidate"; return ;;
        esac
      fi
    done
    return 1
  }

  if ! PYTHON_BIN="$(pick_python)"; then
    err "could not find Python 3.11 or 3.12. Install one (brew install python@3.11) and re-run, or pass --python /path/to/python."
    return 1
  fi
  log "using interpreter: $PYTHON_BIN ($("$PYTHON_BIN" --version))"

  # --- 4. Create / refresh virtual environment -----------------------------
  if [[ $RECREATE_VENV -eq 1 && -d "$VENV_DIR" ]]; then
    log "removing existing venv at $VENV_DIR"
    rm -rf "$VENV_DIR"
  fi

  if [[ ! -d "$VENV_DIR" ]]; then
    log "creating venv at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  else
    log "reusing existing venv at $VENV_DIR"
  fi

  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"

  log "upgrading pip / wheel / setuptools"
  python -m pip install --upgrade pip wheel setuptools

  # madmom 0.16.x is an old source-only dependency that imports Cython during
  # its build setup. Its sdist also contains stale generated C files that do
  # not compile on Python 3.11, so we touch the Cython/Python sources to force
  # regeneration before installing it without dependency upgrades.
  log "installing legacy madmom build prerequisites"
  python -m pip install --upgrade hatchling "numpy<2" "cython<3" "scipy>=0.16" "mido>=1.2.8"

  local madmom_build_dir madmom_sdist madmom_src_dir madmom_install_rc
  madmom_build_dir="$(mktemp -d)"
  python -m pip download --no-build-isolation --no-deps --no-binary :all: --dest "$madmom_build_dir" "madmom>=0.16,<0.17"
  madmom_sdist=("${madmom_build_dir}"/madmom-*.tar.gz)
  madmom_src_dir="${madmom_build_dir}/src"
  mkdir "$madmom_src_dir"
  tar -xzf "${madmom_sdist[0]}" -C "$madmom_src_dir" --strip-components 1
  touch \
    "${madmom_src_dir}/madmom/audio/comb_filters.pyx" \
    "${madmom_src_dir}/madmom/features/beats_crf.pyx" \
    "${madmom_src_dir}/madmom/ml/hmm.pyx" \
    "${madmom_src_dir}/madmom/ml/nn/layers.py"

  log "pre-installing madmom without build isolation"
  if python -m pip install --no-build-isolation --no-deps "$madmom_src_dir"; then
    rm -rf "$madmom_build_dir"
  else
    madmom_install_rc=$?
    rm -rf "$madmom_build_dir"
    return "$madmom_install_rc"
  fi

  # --- 5. Install the package + dev extras ---------------------------------
  log "pip install -e .[dev]  (this may take several minutes for demucs/madmom/basic-pitch)"
  python -m pip install -e ".[dev]"

  # --- 6. Download soundfont (optional) ------------------------------------
  local soundfont_path="$PROJECT_ROOT/tests/fixtures/synthetic/soundfont/TimGM6mb.sf2"
  if [[ $SKIP_SOUNDFONT -eq 0 ]]; then
    if [[ -f "$soundfont_path" ]]; then
      log "soundfont already present: $soundfont_path"
    else
      log "downloading soundfont → $soundfont_path"
      if ! python "$PROJECT_ROOT/scripts/download_soundfont.py"; then
        warn "soundfont download failed — fluidsynth synthesis will fall back to sine waves."
        warn "Re-run later with: python scripts/download_soundfont.py"
      fi
    fi
  else
    log "skipping soundfont download (--no-soundfont)"
  fi

  # --- 7. Health check -----------------------------------------------------
  log "running: music-decoder doctor"
  if ! music-decoder doctor; then
    warn "doctor reported issues — see hints above. The environment is otherwise installed."
  fi
}

# Run the heavy work in an isolated subshell with strict mode.
( set -euo pipefail; _setup_main "$@" )
_setup_rc=$?
unset -f _setup_main

# ---- final step: activate venv (if sourced) or print the command ----------
if [ "$_setup_rc" -eq 0 ]; then
  if [ "$_setup_sourced" -eq 1 ] && [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck source=/dev/null
    . "$VENV_DIR/bin/activate"
    printf '\n\033[1;32m[setup]\033[0m venv activated in this shell — try: music-decoder --help\n'
  else
    cat <<EOF

[setup] done. To activate the venv in your current shell:

    source .venv/bin/activate

Tip: next time, run "source ./setup.sh" to auto-activate after setup.

Common commands:
    music-decoder analyze <youtube-url-or-file>
    music-decoder ui
    make test-fast
EOF
  fi
fi

# Return on sourced, exit on executed — preserves rc either way.
if [ "$_setup_sourced" -eq 1 ]; then
  unset _setup_sourced
  return $_setup_rc 2>/dev/null
else
  exit $_setup_rc
fi

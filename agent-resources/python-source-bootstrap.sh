#!/usr/bin/env sh
# SPDX-License-Identifier: Apache-2.0

set -eu

log() {
  printf '[python-source-bootstrap] %s\n' "$*"
}

cleanup_untracked_cython_sources() {
  if ! command -v git >/dev/null 2>&1 || ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    return 0
  fi

  git ls-files --others --exclude-standard | while IFS= read -r path; do
    case "$path" in
      *.c|*.cc|*.cpp|*.cxx)
        stem=${path%.*}
        if [ -f "$stem.pyx" ]; then
          rm -f "$path"
          log "removed generated Cython source $path"
        fi
        ;;
    esac
  done
}

configure_legacy_gcc_cflags() {
  if ! command -v gcc >/dev/null 2>&1; then
    return 0
  fi

  gcc_major=$(gcc -dumpfullversion -dumpversion 2>/dev/null | awk -F. '{print $1}')
  case "$gcc_major" in
    ''|*[!0-9]*)
      return 0
      ;;
  esac
  if [ "$gcc_major" -lt 14 ]; then
    return 0
  fi

  legacy_flags="-Wno-error=incompatible-pointer-types -Wno-incompatible-pointer-types"
  case " ${CFLAGS:-} " in
    *" -Wno-error=incompatible-pointer-types "*)
      ;;
    *)
      export CFLAGS="${CFLAGS:-} $legacy_flags"
      log "using GCC $gcc_major legacy CFLAGS for older generated C sources"
      ;;
  esac
}

if ! command -v python >/dev/null 2>&1; then
  log "python is unavailable; skipping"
  exit 0
fi

if [ ! -f setup.py ] && [ ! -f pyproject.toml ] && [ ! -f setup.cfg ]; then
  log "no Python packaging metadata found; skipping"
  exit 0
fi

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PYTHONDONTWRITEBYTECODE=1

configure_legacy_gcc_cflags

if [ -f setup.py ]; then
  log "building in-place extension modules with setup.py"
  python setup.py build_ext --inplace
  cleanup_untracked_cython_sources
fi

log "installing project in editable mode without dependency resolution"
python -m pip install --no-index --no-build-isolation --no-deps -e .
cleanup_untracked_cython_sources

log "completed"

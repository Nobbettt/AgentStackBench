#!/bin/sh
# SPDX-License-Identifier: Apache-2.0

set -eu

compat_root="${CONTEXTBENCH_GLIBC_COMPAT_ROOT:-${CONTEXTBENCH_RUNTIME_ROOT}/glibc-debian-trixie-x86_64}"
loader="${compat_root}/lib64/ld-linux-x86-64.so.2"
library_path="${compat_root}/lib"

if [ ! -x "$loader" ]; then
  echo "Memtrace glibc compatibility loader is missing or not executable: $loader" >&2
  exit 127
fi

for candidate in \
  "${NPM_CONFIG_PREFIX}/lib/node_modules/memtrace/node_modules/@memtrace/linux-x64/bin/memcore-server" \
  "${NPM_CONFIG_PREFIX}/lib/node_modules/memtrace/node_modules/@memtrace/linux-x64-noavx2/bin/memcore-server" \
  "${NPM_CONFIG_PREFIX}/lib/node_modules/@memtrace/linux-x64/bin/memcore-server" \
  "${NPM_CONFIG_PREFIX}/lib/node_modules/@memtrace/linux-x64-noavx2/bin/memcore-server"
do
  if [ -x "$candidate" ]; then
    exec "$loader" --library-path "$library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$candidate" "$@"
  fi
done

echo "Memtrace memcore-server binary was not found under ${NPM_CONFIG_PREFIX}/lib/node_modules" >&2
exit 127

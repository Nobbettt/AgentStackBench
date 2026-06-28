#!/bin/sh
# SPDX-License-Identifier: Apache-2.0

set -eu

if [ ! -f .git ]; then
  exit 0
fi

head_commit="$(git rev-parse HEAD)"
common_dir="$(git rev-parse --git-common-dir)"

case "$common_dir" in
  .git|"$(pwd)/.git")
    exit 0
    ;;
esac

rm .git
git init -q
mkdir -p .git/objects/info
printf '%s/objects\n' "$common_dir" > .git/objects/info/alternates
git reset -q --mixed "$head_commit"

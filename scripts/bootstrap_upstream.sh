#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${LINGBOT_UPSTREAM_ROOT:-$PROJECT_ROOT/.upstream/lingbot-vla-v2}"
REPOSITORY="$(awk '$1 == "repository:" {print $2}' "$PROJECT_ROOT/upstream.lock")"
REVISION="$(awk '$1 == "revision:" {print $2}' "$PROJECT_ROOT/upstream.lock")"

if [[ -d "$TARGET/.git" ]]; then
  ACTUAL="$(git -C "$TARGET" rev-parse HEAD)"
  if [[ "$ACTUAL" != "$REVISION" ]]; then
    printf 'Existing upstream checkout has the wrong revision: %s != %s\n' "$ACTUAL" "$REVISION" >&2
    exit 1
  fi
  printf '%s\n' "$TARGET"
  exit 0
fi

if [[ -e "$TARGET" ]]; then
  printf 'Target exists but is not a Git checkout: %s\n' "$TARGET" >&2
  exit 1
fi

mkdir -p -- "$(dirname -- "$TARGET")"
git clone --filter=blob:none "$REPOSITORY" "$TARGET"
git -C "$TARGET" checkout --detach "$REVISION"
printf '%s\n' "$TARGET"

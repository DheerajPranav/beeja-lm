#!/usr/bin/env bash
set -euo pipefail

printf 'Beeja LM — Claude Code Starter Kit\n'
printf 'Project: %s\n\n' "$(pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  printf 'Error: python3 was not found.\n' >&2
  exit 1
fi

python3 scripts/check_environment.py || true

if [ ! -d .git ]; then
  git init >/dev/null 2>&1 || true
fi

if ! command -v claude >/dev/null 2>&1; then
  cat <<'MSG'

Claude Code is not available on PATH.
Install or configure Claude Code, then run:

  claude

Inside the session:

  /context
  /beeja-lab bootstrap
MSG
  exit 0
fi

cat <<'MSG'

Starting Claude Code with the bootstrap skill.
Review any permission or workspace-trust prompts before accepting them.
MSG

exec claude "/beeja-lab bootstrap"

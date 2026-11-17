#!/usr/bin/env bash
# Local Mendix Agent launcher for Linux (uses the local .venv, no PATH changes)
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -x "$dir/.venv/bin/python" ]; then
  echo ".venv not found. Run: $dir/setup.sh" >&2
  exit 1
fi
exec "$dir/.venv/bin/python" "$dir/main.py" "$@"

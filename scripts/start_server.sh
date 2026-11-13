#!/usr/bin/env bash
# Starts the local Pokémon Showdown server in the foreground (Ctrl-C to stop).
#
# --no-security disables rate limiting and login checks, which is what lets
# bots play thousands of battles quickly. That is only acceptable because the
# server is bound to 127.0.0.1; this script refuses to start otherwise.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PS_DIR="$ROOT/vendor/pokemon-showdown"
CONFIG="$PS_DIR/config/config.js"
PORT="${PS_PORT:-8000}"

[ -f "$CONFIG" ] || { echo "error: server not installed; run scripts/setup_showdown.sh" >&2; exit 1; }
grep -q "^exports.bindaddress = '127.0.0.1';" "$CONFIG" || {
  echo "error: $CONFIG is not bound to 127.0.0.1; re-run scripts/setup_showdown.sh" >&2
  exit 1
}

cd "$PS_DIR"
exec node pokemon-showdown start --no-security "$PORT"

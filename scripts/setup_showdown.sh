#!/usr/bin/env bash
# Installs a local Pokémon Showdown server into vendor/pokemon-showdown.
#
# Idempotent: safe to re-run. Pins the simulator to a known commit so battle
# mechanics do not change underneath training runs, and forces the server to
# listen on loopback only (the upstream default is 0.0.0.0).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PS_DIR="$ROOT/vendor/pokemon-showdown"
PS_REPO="https://github.com/smogon/pokemon-showdown.git"
# Bump deliberately; re-run evaluation after bumping, since mechanics/sets may change.
PS_COMMIT="${PS_COMMIT:-e5d26285ab2cb644291dc516269c9927dc6ef993}"

command -v node >/dev/null || { echo "error: node is required (>=22.18)" >&2; exit 1; }
command -v npm >/dev/null || { echo "error: npm is required" >&2; exit 1; }

if [ ! -d "$PS_DIR/.git" ]; then
  echo "==> Cloning Pokémon Showdown @ ${PS_COMMIT:0:10}"
  mkdir -p "$PS_DIR"
  git -C "$PS_DIR" init --quiet
  git -C "$PS_DIR" remote add origin "$PS_REPO"
fi

if [ "$(git -C "$PS_DIR" rev-parse HEAD 2>/dev/null || true)" != "$PS_COMMIT" ]; then
  echo "==> Checking out ${PS_COMMIT:0:10}"
  git -C "$PS_DIR" fetch --quiet --depth 1 origin "$PS_COMMIT"
  git -C "$PS_DIR" checkout --quiet --detach FETCH_HEAD
fi

echo "==> Installing Node dependencies"
(cd "$PS_DIR" && npm install --no-audit --no-fund)

CONFIG="$PS_DIR/config/config.js"
[ -f "$CONFIG" ] || cp "$PS_DIR/config/config-example.js" "$CONFIG"

# Loopback only: the server runs with --no-security, so it must never be
# reachable from the LAN or the internet.
sed -i '' -E "s/^exports\.bindaddress = .*/exports.bindaddress = '127.0.0.1';/" "$CONFIG"
grep -q "^exports.bindaddress = '127.0.0.1';" "$CONFIG" \
  || { echo "error: failed to set bindaddress in $CONFIG" >&2; exit 1; }

echo "==> Done. Start the server with: scripts/start_server.sh"

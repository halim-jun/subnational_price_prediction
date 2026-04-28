#!/usr/bin/env bash
# Full build pipeline: export JSON from artifacts → next build → out/
# Used both locally and by Cloudflare Pages CI.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "[1/3] Exporting model artifacts to static JSON..."
cd "$PROJECT_ROOT"
python "$SCRIPT_DIR/scripts/export_data.py"

echo "[2/3] Installing frontend dependencies..."
cd "$SCRIPT_DIR/frontend"
if [ ! -d node_modules ]; then
  npm ci || npm install
fi

echo "[3/3] Building static site..."
npm run build

echo
echo "Done. Static site is at: $SCRIPT_DIR/frontend/out/"

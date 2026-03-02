#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

PY_BIN="${PY_BIN:-python3}"

$PY_BIN -m pip install -r "$ROOT_DIR/tools/local_services/requirements.txt"

# stop old
pkill -f "uvicorn tools.local_services.embedding_api:app" || true
pkill -f "uvicorn tools.local_services.mineru_compat_api:app" || true

nohup $PY_BIN -m uvicorn tools.local_services.embedding_api:app --host 127.0.0.1 --port 8005 > "$LOG_DIR/embedding_api.log" 2>&1 &
nohup $PY_BIN -m uvicorn tools.local_services.mineru_compat_api:app --host 127.0.0.1 --port 30924 > "$LOG_DIR/mineru_compat_api.log" 2>&1 &

echo "Started embedding API: http://127.0.0.1:8005"
echo "Started mineru API:    http://127.0.0.1:30924/file_parse"

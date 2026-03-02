#!/usr/bin/env bash
set -euo pipefail
pkill -f "uvicorn tools.local_services.embedding_api:app" || true
pkill -f "uvicorn tools.local_services.mineru_compat_api:app" || true
echo "Stopped local embedding/mineru services"

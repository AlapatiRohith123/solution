#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${APP_DIR:-/app}"
cd "$APP_DIR"
mkdir -p "$APP_DIR/artifacts"
python3 "$SCRIPT_DIR/train.py"

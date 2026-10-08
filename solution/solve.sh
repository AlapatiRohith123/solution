#!/bin/bash
set -e
APP_DIR="${APP_DIR:-/app}"
cd "$APP_DIR"
mkdir -p "$APP_DIR/artifacts"
python3 solution/train.py

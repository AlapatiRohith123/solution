#!/bin/bash
set -e
cd /app
mkdir -p /app/artifacts
python3 solution/train.py

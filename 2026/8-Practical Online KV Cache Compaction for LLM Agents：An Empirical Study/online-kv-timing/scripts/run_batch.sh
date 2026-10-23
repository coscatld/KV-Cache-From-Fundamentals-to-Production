#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m online_kv.selection --config configs/llama.json --output results/search
online-kv --config configs/llama.json --output results/test --split test --frozen results/search/frozen.json
python scripts/plot.py results/test

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
python -c 'import torch,triton,transformers; assert torch.cuda.is_available(), "CUDA required"'
python scripts/environment.py | tee results/environment-gpu.json
pytest -ra -s --junitxml=results/pytest-gpu.xml | tee results/pytest-gpu.txt
python benchmarks/run.py --dtype float32 --output results/gpu-float32.json --hf
python benchmarks/run.py --dtype float16 --output results/gpu-float16.json --hf
if python -c 'import torch; assert torch.cuda.is_bf16_supported()'; then
  python benchmarks/run.py --dtype bfloat16 --output results/gpu-bfloat16.json --hf
fi

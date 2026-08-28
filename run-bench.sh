#!/usr/bin/env bash
set -euo pipefail

bench_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$bench_dir/qwen_bench.py" run --config "$bench_dir/matrix.json" "$@"

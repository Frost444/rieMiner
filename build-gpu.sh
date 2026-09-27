#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
nvcc="${CUDACXX:-$(command -v nvcc || true)}"
if [ -z "$nvcc" ]; then
  printf 'CUDA nvcc not found. Set CUDACXX=/absolute/path/to/cuda/bin/nvcc.\n' >&2
  exit 1
fi
jobs=${BUILD_JOBS:-2}
cmake -S "$root/source/riecoin-network-backend-silver/src/network" -B "$root/build/network" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$root/build/network" --parallel "$jobs"
ctest --test-dir "$root/build/network" --output-on-failure
cmake -S "$root/source/riecoin-silver-r346" -B "$root/build/gpu" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$nvcc" -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-86-real;86-virtual}"
cmake --build "$root/build/gpu" --parallel "$jobs"
ctest --test-dir "$root/build/gpu" --output-on-failure
mkdir -p "$root/cli/riecoin-gpu/bin"
g++ -std=c++17 -O3 -pthread "$root/source/data-tools/data_generator.cpp" -lgmp -lcrypto -o "$root/cli/riecoin-gpu/bin/data-generator"
cp "$root/build/network/bin/riecoin-network-backend" "$root/cli/riecoin-gpu/bin/"
cp "$root/build/gpu/riecoin-r346" "$root/cli/riecoin-gpu/bin/"

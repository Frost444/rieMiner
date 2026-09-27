#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

// Host-only ABI between the stock-CGBN R202 translation unit and the isolated
// R=2^1216 / 38-word CUDA translation unit.  `candidates` is an array of
// cgbn_mem_t<1280>; void keeps the two incompatible CGBN instantiations from
// sharing a C++ type definition or leaking their compile-time radix choice.
extern "C" cudaError_t riecoin_r203_r1216_launch_segmented(
    const void* candidates, std::uint8_t* verdicts,
    const std::uint32_t* macro_counts, std::uint32_t macro_count,
    std::uint32_t macro_capacity, cudaStream_t stream);

extern "C" cudaError_t riecoin_r203_r1216_launch_oracle(
    const void* candidates, std::uint8_t* verdicts,
    std::uint32_t* normal_residues, std::uint32_t count,
    cudaStream_t stream);

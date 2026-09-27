#pragma once

// Include after the existing dynamic CGBN evaluator and RLOW queue bridge.
// Only Q0 arithmetic changes. Candidate generation, the sieve, queue coverage,
// tuple tests and network verification keep their existing semantics.
#include "riecoin_tile8_carrylate_core.cuh"
#include "riecoin_carrylate_prepare1152.hpp"

namespace riecoin_rlow_carrylate {
namespace core = riecoin_tile8_carrylate;
namespace prep = riecoin_carrylate_prepare1152;

// Rare-path counters only; the ordinary candidate path has no global atomic.
// 0: unsupported width/shape, 1: arithmetic invariant fallback, 2: exact
// low-product repairs inside the guarded high-product operator.
__device__ unsigned long long rare_paths[3];

// Bound the preparation temporaries to this call instead of retaining their
// register lifetime throughout the exponentiation. No CPU preparation/transfer.
__device__ __noinline__ bool prepare_shared(core::SharedInstance& state,
                                           uint32_t& bits) {
  prep::UInt1152 modulus{};
  for (uint32_t word = 0; word < core::kLimbs; ++word)
    modulus.limb[word] = state.modulus[word];
  prep::Prepared1152 prepared{};
  if (!prep::prepare(modulus, prepared) || prepared.bit_length < 3u)
    return false;
  bits = prepared.bit_length;
  for (uint32_t word = 0; word < core::kLimbs; ++word) {
    state.negative_inverse[word] = prepared.negative_inverse.limb[word];
    state.current[word] = prepared.two_r_mod_n.limb[word];
  }
  return true;
}

__device__ __forceinline__ bool evaluate(
    const cgbn_mem_t<1280u>* candidates, uint32_t slot,
    core::SharedInstance& state, bool& used_fast,
    uint32_t* normal_residue = nullptr) {
  const uint32_t rank = threadIdx.x & 7u;
  const uint32_t tile_base = (threadIdx.x & 31u) - rank;
  const uint32_t mask = 0xffu << tile_base;
  for (uint32_t word = rank; word < core::kLimbs; word += 8u)
    state.modulus[word] = candidates[slot]._limbs[word];
  __syncwarp(mask);
  uint32_t supported = 0u, bits = 0u;
  if (rank == 0u) {
    uint32_t high = 0u;
    for (uint32_t word = core::kLimbs; word < 40u; ++word)
      high |= candidates[slot]._limbs[word];
    supported = high == 0u && prepare_shared(state, bits) ? 1u : 0u;
  }
  supported = __shfl_sync(mask, supported, tile_base);
  bits = __shfl_sync(mask, bits, tile_base);
  __syncwarp(mask);
  used_fast = false;
  if (!supported) {
    if (rank == 0u) atomicAdd(rare_paths, 1ULL);
    return riecoin_cuda::evaluate_euler_jacobi_dynamic_1280_tpi8(candidates, slot);
  }

  uint32_t repairs = 0u, carry_failures = 0u, result_failures = 0u;
  // e=(n-1)/2, with its leading one already represented by exact 2R mod n.
  // Use the actual bit length: no normalization assumption n >= R/2.
  for (int32_t bit = static_cast<int32_t>(bits) - 3; bit >= 0; --bit) {
    core::montgomery_square_carrylate(state, mask, rank,
        repairs, carry_failures, result_failures);
    const uint32_t source_bit = static_cast<uint32_t>(bit) + 1u;
    if (rank == 0u &&
        ((state.modulus[source_bit >> 5u] >> (source_bit & 31u)) & 1u))
      core::modular_double(state.current, state.modulus);
    __syncwarp(mask);
  }
  core::from_montgomery_carrylate(state, mask, rank, repairs,
      carry_failures, result_failures);
  if (rank == 0u && repairs != 0u)
    atomicAdd(rare_paths + 2u, static_cast<unsigned long long>(repairs));
  if (__any_sync(mask, carry_failures != 0u || result_failures != 0u)) {
    // A fast-path anomaly cannot reject an otherwise valid candidate.
    if (rank == 0u) atomicAdd(rare_paths + 1u, 1ULL);
    return riecoin_cuda::evaluate_euler_jacobi_dynamic_1280_tpi8(candidates, slot);
  }

  uint32_t probable = 1u;
  if (rank == 0u) {
    const uint32_t mod8 = state.modulus[0] & 7u;
    const bool plus_one = mod8 == 1u || mod8 == 7u;
    for (uint32_t word = 0u; word < core::kLimbs; ++word) {
      const uint32_t expected = plus_one ? (word == 0u ? 1u : 0u)
          : (word == 0u ? state.modulus[0] - 1u : state.modulus[word]);
      if (state.current[word] != expected) probable = 0u;
      if (normal_residue != nullptr) normal_residue[word] = state.current[word];
    }
  }
  used_fast = true;
  return __shfl_sync(mask, probable, tile_base) != 0u;
}

// Closed exactness guard callable by the Horizon-owned stage before admission
// of any new arithmetic verdict. Residues are checked independently with GMP.
__global__ __launch_bounds__(128, 2) void oracle_kernel(
    const cgbn_mem_t<1280u>* candidates, uint8_t* verdicts,
    uint8_t* used_fast, uint32_t* residues, uint32_t count) {
  __shared__ core::SharedInstance shared[16];
  const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t slot = thread / 8u;
  if (slot >= count) return;
  bool fast = false;
  const bool probable = evaluate(candidates, slot, shared[threadIdx.x / 8u],
                                 fast, residues + slot * core::kLimbs);
  if ((thread & 7u) == 0u) {
    verdicts[slot] = probable ? 1u : 0u;
    used_fast[slot] = fast ? 1u : 0u;
  }
}
} // namespace riecoin_rlow_carrylate

namespace riecoin_rlow_bridge {
__global__ __launch_bounds__(128, 2) void rlow_segmented_q0_carrylate(
    const cgbn_mem_t<1280u>* candidates, uint8_t* verdicts,
    const uint32_t* macro_counts, uint32_t macro_count,
    uint32_t macro_capacity) {
  __shared__ riecoin_tile8_carrylate::SharedInstance shared[16];
  const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t slot = thread / 8u;
  const uint32_t macro = slot / macro_capacity;
  const uint32_t local = slot - macro * macro_capacity;
  if (macro >= macro_count) return;
  bool probable = false, used_fast = false;
  if (local < macro_counts[macro])
    probable = riecoin_rlow_carrylate::evaluate(candidates, slot,
        shared[threadIdx.x / 8u], used_fast);
  if ((thread & 7u) == 0u) verdicts[slot] = probable ? 1u : 0u;
}
} // namespace riecoin_rlow_bridge

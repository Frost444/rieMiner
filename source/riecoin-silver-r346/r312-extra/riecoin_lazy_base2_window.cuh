#pragma once

#include <cstdint>

// A different residue representation, not a different primality predicate.
// R = 2^1280. For odd n with h leading zero bits, choose k such that
// h >= 2^(k+1). A window value w is at most 2^k-1.
// REDC(x*x) < 2n whenever x*x < nR. Delaying normalization after the
// window's multiplication by 2^w gives x < 2^(w+1)n, hence
// x*x < 2^(2w+2)n*n < nR under that headroom condition.
// The shift cannot overflow R. The next REDC consumes the redundant residue;
// only the final Montgomery-to-normal conversion must be canonical.
// The old evaluator remains the fallback when this representation is invalid.
namespace riecoin_lazy_base2_window {

constexpr uint32_t kStorageBits = 1280u;
constexpr uint32_t kMaximumWindowBits = 6u;

#if defined(__CUDACC__)
__host__ __device__
#endif
constexpr uint32_t window_bits(uint32_t leading_zeroes) {
  uint32_t selected = 0u;
  for (uint32_t bits = 1u; bits <= kMaximumWindowBits; ++bits)
    if (leading_zeroes >= (1u << (bits + 1u))) selected = bits;
  return selected;
}

#if defined(__CUDACC__)
// Included after the existing CGBN evaluator; do not reinclude CGBN itself.
__device__ __forceinline__ bool evaluate(
    const cgbn_mem_t<kStorageBits>* candidates, uint32_t slot,
    uint32_t* normal_residue = nullptr, uint8_t* used_window = nullptr) {
  using context = cgbn_context_t<8u, riecoin_cuda::PrpCgbnParameters>;
  using env_type = cgbn_env_t<context, kStorageBits>;
  using bn = typename env_type::cgbn_t;
  context cgbn_context(cgbn_no_checks, nullptr, slot);
  env_type env(cgbn_context);
  bn modulus, residue;
  cgbn_load(env, modulus,
            const_cast<cgbn_mem_t<kStorageBits>*>(candidates + slot));
  const uint32_t leading_zeroes = cgbn_clz(env, modulus);
  const uint32_t bits = kStorageBits - leading_zeroes;
  const uint32_t width = window_bits(leading_zeroes);
  const uint32_t lane = threadIdx.x & 7u;
  if (used_window != nullptr && lane == 0u) used_window[slot] = 0u;
  if (width == 0u || bits < 3u ||
      (cgbn_get_ui32(env, modulus) & 1u) == 0u)
    return riecoin_cuda::evaluate_euler_jacobi_dynamic_1280_tpi8(candidates, slot);

  cgbn_set_ui32(env, residue, 1u);
  const uint32_t inverse = cgbn_bn2mont(env, residue, residue, modulus);
  uint32_t remaining = bits - 1u;  // e=(n-1)/2 = n >> 1 for odd n.
  bool first = true;
  while (remaining != 0u) {
    const uint32_t take = remaining < width ? remaining : width;
    if (!first) {
      for (uint32_t i = 0u; i < take; ++i)
        cgbn_mont_sqr(env, residue, residue, modulus, inverse);
    }
    // On the first window, squaring Montgomery one would do no useful work.
    const uint32_t value = cgbn_extract_bits_ui32(
        env, modulus, remaining - take + 1u, take);
    cgbn_shift_left(env, residue, residue, value);
    remaining -= take;
    first = false;
  }
  cgbn_mont2bn(env, residue, residue, modulus, inverse);
  if (normal_residue != nullptr)
    cgbn_store(env,
        reinterpret_cast<cgbn_mem_t<kStorageBits>*>(normal_residue) + slot, residue);
  if (used_window != nullptr && lane == 0u)
    used_window[slot] = static_cast<uint8_t>(width);
  return riecoin_cuda::passes_euler_jacobi(env, residue, modulus);
}

__global__ void oracle_kernel(const cgbn_mem_t<kStorageBits>* candidates,
    uint8_t* verdicts, uint8_t* used_window, uint32_t* residues, uint32_t count) {
  const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t slot = thread / 8u;
  if (slot >= count) return;
  const bool probable = evaluate(candidates, slot, residues, used_window);
  if ((thread & 7u) == 0u) verdicts[slot] = probable ? 1u : 0u;
}

__global__ void segmented_q0_kernel(
    const cgbn_mem_t<kStorageBits>* candidates, uint8_t* verdicts,
    const uint32_t* macro_counts, uint32_t macro_count,
    uint32_t macro_capacity) {
  const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t slot = thread / 8u;
  const uint32_t macro = slot / macro_capacity;
  if (macro >= macro_count) return;
  const uint32_t local = slot - macro * macro_capacity;
  const bool probable = local < macro_counts[macro] && evaluate(candidates, slot);
  if ((thread & 7u) == 0u) verdicts[slot] = probable ? 1u : 0u;
}
#endif

}  // namespace riecoin_lazy_base2_window

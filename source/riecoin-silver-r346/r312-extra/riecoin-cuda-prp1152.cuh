#pragma once

#include <cuda_runtime.h>
#include <cgbn/cgbn.h>

namespace riecoin_cuda {

constexpr uint32_t kPrpBlockThreads = 128u;

struct PrpCgbnParameters {
  static constexpr uint32_t TPB = kPrpBlockThreads;
  static constexpr uint32_t MAX_ROTATION = 4u;
  static constexpr uint32_t SHM_LIMIT = 0u;
  static constexpr bool CONSTANT_TIME = false;
};

template <class Env, class Bn>
__device__ __forceinline__ void double_modulo(
    Env& env, Bn& value, const Bn& modulus) {
  if (cgbn_compare(env, value, modulus) >= 0)
    cgbn_sub(env, value, value, modulus);
  const int32_t carry = cgbn_add(env, value, value, value);
  if (carry != 0 || cgbn_compare(env, value, modulus) >= 0)
    cgbn_sub(env, value, value, modulus);
}

template <class Env, class Bn>
__device__ __forceinline__ bool passes_euler_jacobi(
    Env& env, const Bn& residue, const Bn& modulus) {
  const uint32_t modulo_eight = cgbn_get_ui32(env, modulus) & 7u;
  if (modulo_eight == 1u || modulo_eight == 7u)
    return cgbn_equals_ui32(env, residue, 1u);
  Bn minus_one;
  cgbn_sub_ui32(env, minus_one, modulus, 1u);
  return cgbn_equals(env, residue, minus_one);
}

__device__ __forceinline__ bool evaluate_euler_jacobi_1152_tpi4(
    const cgbn_mem_t<1152u>* candidates, uint32_t instance) {
  using context = cgbn_context_t<4u, PrpCgbnParameters>;
  using env_type = cgbn_env_t<context, 1152u>;
  using bn = typename env_type::cgbn_t;
  context cgbn_context(cgbn_no_checks, nullptr, instance);
  env_type env(cgbn_context);
  bn modulus, residue, one_mont;
  cgbn_load(env, modulus,
            const_cast<cgbn_mem_t<1152u>*>(candidates + instance));
  if ((cgbn_get_ui32(env, modulus) & 1u) == 0u || cgbn_clz(env, modulus) != 0u)
    return false;
  cgbn_negate(env, one_mont, modulus);
  cgbn_set(env, residue, one_mont);
  const uint32_t inverse =
      -cgbn_binary_inverse_ui32(env, cgbn_get_ui32(env, modulus));
  uint32_t cached_word = UINT32_MAX;
  uint32_t exponent_word = 0u;
  for (int32_t bit = 1150; bit >= 0; --bit) {
    cgbn_mont_sqr(env, residue, residue, modulus, inverse);
    const uint32_t source = static_cast<uint32_t>(bit) + 1u;
    const uint32_t word = source >> 5u;
    if (word != cached_word) {
      exponent_word = cgbn_extract_bits_ui32(env, modulus, word * 32u, 32u);
      cached_word = word;
    }
    if (((exponent_word >> (source & 31u)) & 1u) != 0u)
      double_modulo(env, residue, modulus);
  }
  cgbn_mont2bn(env, residue, residue, modulus, inverse);
  return passes_euler_jacobi(env, residue, modulus);
}

__global__ void euler_jacobi_1152_tpi4_kernel(
    const cgbn_mem_t<1152u>* candidates,
    uint8_t* verdicts,
    uint32_t count) {
  const uint32_t global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t instance = global_thread / 4u;
  if (instance >= count) return;
  const bool probable = evaluate_euler_jacobi_1152_tpi4(candidates, instance);
  if ((global_thread & 3u) == 0u)
    verdicts[instance] = probable ? 1u : 0u;
}

__device__ __forceinline__ bool evaluate_euler_jacobi_dynamic_1280_tpi8(
    const cgbn_mem_t<1280u>* candidates, uint32_t instance) {
  using context = cgbn_context_t<8u, PrpCgbnParameters>;
  using env_type = cgbn_env_t<context, 1280u>;
  using bn = typename env_type::cgbn_t;
  context cgbn_context(cgbn_no_checks, nullptr, instance);
  env_type env(cgbn_context);
  bn modulus, residue;
  cgbn_load(env, modulus,
            const_cast<cgbn_mem_t<1280u>*>(candidates + instance));
  const uint32_t leading_zeroes = cgbn_clz(env, modulus);
  const uint32_t bit_length = 1280u - leading_zeroes;
  if ((cgbn_get_ui32(env, modulus) & 1u) == 0u ||
      bit_length < 3u || bit_length > 1280u)
    return false;
  cgbn_set_ui32(env, residue, 1u);
  const uint32_t inverse = cgbn_bn2mont(env, residue, residue, modulus);
  uint32_t cached_word = UINT32_MAX;
  uint32_t exponent_word = 0u;
  // The exponent is (n - 1) / 2.  For odd n its bit `bit` is exactly
  // modulus bit `bit + 1`; start at the actual top bit instead of assuming
  // the historical 1152-bit fixture.
  for (int32_t bit = static_cast<int32_t>(bit_length) - 2; bit >= 0; --bit) {
    cgbn_mont_sqr(env, residue, residue, modulus, inverse);
    const uint32_t source = static_cast<uint32_t>(bit) + 1u;
    const uint32_t word = source >> 5u;
    if (word != cached_word) {
      exponent_word = cgbn_extract_bits_ui32(env, modulus, word * 32u, 32u);
      cached_word = word;
    }
    if (((exponent_word >> (source & 31u)) & 1u) != 0u)
      double_modulo(env, residue, modulus);
  }
  cgbn_mont2bn(env, residue, residue, modulus, inverse);
  return passes_euler_jacobi(env, residue, modulus);
}

__global__ void euler_jacobi_dynamic_1280_tpi8_kernel(
    const cgbn_mem_t<1280u>* candidates,
    uint8_t* verdicts,
    uint32_t count) {
  const uint32_t global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t instance = global_thread / 8u;
  if (instance >= count) return;
  const bool probable = evaluate_euler_jacobi_dynamic_1280_tpi8(
      candidates, instance);
  if ((global_thread & 7u) == 0u)
    verdicts[instance] = probable ? 1u : 0u;
}

__global__ void euler_jacobi_dynamic_1280_tpi8_counted_kernel(
    const cgbn_mem_t<1280u>* candidates,
    uint8_t* verdicts,
    const uint32_t* device_count,
    uint32_t capacity) {
  const uint32_t global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t instance = global_thread / 8u;
  const uint32_t count = *device_count;
  if (instance >= count || instance >= capacity) return;
  const bool probable = evaluate_euler_jacobi_dynamic_1280_tpi8(
      candidates, instance);
  if ((global_thread & 7u) == 0u)
    verdicts[instance] = probable ? 1u : 0u;
}

}  // namespace riecoin_cuda

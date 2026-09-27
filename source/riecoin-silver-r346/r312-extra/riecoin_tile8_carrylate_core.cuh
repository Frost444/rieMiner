#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Exact reusable device atoms extracted mechanically from
// riecoin_tile8_carrylate1152.cu. The old normalized-only initialization
// and standalone benchmark are deliberately not part of this interface.
namespace riecoin_tile8_carrylate {

constexpr uint32_t kBits = 1152u;
constexpr uint32_t kLimbs = 36u;
constexpr uint32_t kProductLimbs = 72u;
constexpr uint32_t kTile = 8u;
constexpr uint32_t kBlockThreads = 128u;
constexpr uint32_t kInstancesPerBlock = kBlockThreads / kTile;
constexpr uint64_t kRadix = uint64_t{1} << 32u;
constexpr uint32_t kStepDomainMaximum = 37u;
constexpr uint32_t kStepNever = kStepDomainMaximum + 1u;

__host__ __device__ __forceinline__ constexpr uint32_t min_u32(
    uint32_t left, uint32_t right) {
  return left < right ? left : right;
}

constexpr uint64_t kSquareProducts = 666u;
constexpr uint64_t kInverseTruncatedProducts = 666u;
constexpr uint64_t kMulHighProducts = 630u;
constexpr uint64_t kCarryGuardProducts = 36u + 35u + 34u;
constexpr uint64_t kFallbackLowProducts = 666u;
constexpr uint64_t kFastProductsPerSquare =
    kSquareProducts + kInverseTruncatedProducts + kMulHighProducts +
    kCarryGuardProducts;
constexpr uint64_t kFastProductsFinalRedc =
    kInverseTruncatedProducts + kMulHighProducts + kCarryGuardProducts;
constexpr uint64_t kEulerJacobiSquares = 1150u;
constexpr uint64_t kFermatSquares = 1151u;
constexpr uint64_t kEulerJacobiProducts =
    kEulerJacobiSquares * kFastProductsPerSquare + kFastProductsFinalRedc;
constexpr uint64_t kFermatProducts =
    kFermatSquares * kFastProductsPerSquare + kFastProductsFinalRedc;

static_assert(kFastProductsPerSquare == 2067u, "carry-late product count drift");
static_assert(kEulerJacobiProducts == 2378451u, "Euler/Jacobi cost drift");
static_assert(kFermatProducts == 2380518u, "Fermat cost drift");

enum class Base2Semantic : uint32_t {
  FermatNMinusOne = 0u,
  EulerJacobiHalf = 1u,
};

struct UInt1152 {
  uint32_t limb[kLimbs];
};

struct Diagnostics {
  uint32_t shape_failures;
  uint32_t carry_guard_fallbacks;
  uint32_t carry_bound_failures;
  uint32_t result_bound_failures;
  uint32_t residue_is_one;
};

struct SharedInstance {
  uint32_t modulus[kLimbs];
  uint32_t negative_inverse[kLimbs];
  uint32_t current[kLimbs];
  uint32_t t0_or_p1[kLimbs];
  uint32_t t1_or_q[kLimbs];
};

struct U96 {
  uint64_t low;
  uint32_t high;
};

struct CarryStep {
  uint32_t low;
  uint32_t threshold;
};

__host__ __device__ __forceinline__ void add_u64(U96& value,
                                                  uint64_t addend) {
  const uint64_t before = value.low;
  value.low += addend;
  value.high += value.low < before ? 1u : 0u;
}

__host__ __device__ __forceinline__ void add_product(U96& value,
                                                      uint32_t left,
                                                      uint32_t right) {
  add_u64(value, static_cast<uint64_t>(left) * right);
}

__host__ __device__ __forceinline__ uint32_t u96_word(const U96& value,
                                                       uint32_t word) {
  if (word == 0u) return static_cast<uint32_t>(value.low);
  if (word == 1u) return static_cast<uint32_t>(value.low >> 32u);
  return value.high;
}

__device__ __forceinline__ uint64_t shuffle_u64(uint32_t mask, uint64_t value,
                                                uint32_t source_lane) {
  const uint32_t low = __shfl_sync(mask, static_cast<uint32_t>(value),
                                   source_lane);
  const uint32_t high = __shfl_sync(mask, static_cast<uint32_t>(value >> 32u),
                                    source_lane);
  return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32u);
}

__device__ __forceinline__ uint64_t shuffle_up_u64(uint32_t mask,
                                                   uint64_t value,
                                                   uint32_t delta) {
  const uint32_t low = __shfl_up_sync(mask, static_cast<uint32_t>(value), delta);
  const uint32_t high = __shfl_up_sync(mask,
      static_cast<uint32_t>(value >> 32u), delta);
  return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32u);
}

__host__ __device__ __forceinline__ uint32_t apply_step(
    const CarryStep& step, uint32_t input) {
  return step.low + (step.threshold <= kStepDomainMaximum &&
                     input >= step.threshold ? 1u : 0u);
}

__host__ __device__ __forceinline__ CarryStep compose_steps(
    const CarryStep& first, const CarryStep& second) {
  const uint32_t at_zero = apply_step(second, apply_step(first, 0u));
  const uint32_t at_max = apply_step(
      second, apply_step(first, kStepDomainMaximum));
  return CarryStep{at_zero, at_zero == at_max ? kStepNever : first.threshold};
}

__host__ __device__ __forceinline__ uint32_t pack_step(
    const CarryStep& step) {
  return step.low | (step.threshold << 6u);
}

__host__ __device__ __forceinline__ CarryStep unpack_step(uint32_t packed) {
  return CarryStep{packed & 63u, packed >> 6u};
}

__device__ __forceinline__ uint64_t raw_high(const U96& raw) {
  return (raw.low >> 32u) | (static_cast<uint64_t>(raw.high) << 32u);
}

// Normalize up to eight adjacent radix-2^32 coefficients.  For lane i>0,
// carry_i = high(raw_{i-1}) + delta_i and delta_i is in [0,37].  Each lane's
// recurrence is therefore a monotone one-step function; three prefix
// compositions replace five reductions for every coefficient.
__device__ __forceinline__ uint64_t normalize_tile(
    U96 raw, uint64_t carry_in, uint32_t tile_mask, uint32_t tile_rank,
    uint32_t active_count, uint32_t& digit, uint32_t& bound_failures) {
  const uint32_t tile_base = (threadIdx.x & 31u) - tile_rank;
  const uint64_t high = raw_high(raw);
  const uint64_t previous_high = shuffle_up_u64(tile_mask, high, 1u);

  CarryStep prefix{0u, kStepNever};
  if (tile_rank != 0u) {
    const uint64_t sum = static_cast<uint64_t>(static_cast<uint32_t>(raw.low)) +
                         previous_high;
    const uint32_t remainder = static_cast<uint32_t>(sum);
    const uint64_t distance = kRadix - remainder;
    prefix.low = static_cast<uint32_t>(sum >> 32u);
    prefix.threshold = distance <= kStepDomainMaximum
                           ? static_cast<uint32_t>(distance)
                           : kStepNever;
  }

#pragma unroll
  for (uint32_t offset = 1u; offset < kTile; offset <<= 1u) {
    const CarryStep earlier = unpack_step(__shfl_up_sync(
        tile_mask, pack_step(prefix), offset));
    if (tile_rank >= offset + 1u)
      prefix = compose_steps(earlier, prefix);
  }

  U96 first_total = raw;
  if (tile_rank == 0u) add_u64(first_total, carry_in);
  const uint64_t first_out = raw_high(first_total);
  uint32_t delta_one = tile_rank == 0u
      ? static_cast<uint32_t>(first_out - high) : 0u;
  delta_one = __shfl_sync(tile_mask, delta_one, tile_base);
  if (delta_one > kStepDomainMaximum) ++bound_failures;

  const uint32_t delta_out = tile_rank == 0u
      ? delta_one : apply_step(prefix, delta_one);
  const uint32_t previous_delta = __shfl_up_sync(tile_mask, delta_out, 1u);
  const uint64_t incoming = tile_rank == 0u
      ? carry_in : previous_high + previous_delta;
  U96 normalized = raw;
  add_u64(normalized, incoming);
  digit = static_cast<uint32_t>(normalized.low);
  const uint64_t outgoing = high + delta_out;
  const uint32_t last_lane = tile_base + active_count - 1u;
  return shuffle_u64(tile_mask, outgoing, last_lane);
}

__host__ __device__ __forceinline__ U96 square_coefficient(
    const uint32_t* value, uint32_t coefficient) {
  U96 result{0u, 0u};
  if (coefficient >= 2u * kLimbs - 1u) return result;
  const uint32_t first = coefficient >= kLimbs
      ? coefficient - (kLimbs - 1u) : 0u;
  const uint32_t last = min_u32(kLimbs - 1u, coefficient / 2u);
  for (uint32_t left = first; left <= last; ++left) {
    const uint32_t right = coefficient - left;
    const uint64_t product = static_cast<uint64_t>(value[left]) * value[right];
    add_u64(result, product);
    if (left != right) add_u64(result, product);
  }
  return result;
}

__host__ __device__ __forceinline__ U96 product_coefficient_host_device(
    const uint32_t* left, const uint32_t* right, uint32_t coefficient) {
  U96 result{0u, 0u};
  if (coefficient >= 2u * kLimbs - 1u) return result;
  const uint32_t first = coefficient >= kLimbs
      ? coefficient - (kLimbs - 1u) : 0u;
  const uint32_t last = min_u32(kLimbs - 1u, coefficient);
  for (uint32_t index = first; index <= last; ++index)
    add_product(result, left[index], right[coefficient - index]);
  return result;
}

__device__ __forceinline__ void square_full(
    const uint32_t* input, uint32_t* low, uint32_t* high,
    uint32_t tile_mask, uint32_t tile_rank, uint32_t& bound_failures) {
  uint64_t carry = 0u;
  for (uint32_t base = 0u; base < kProductLimbs; base += kTile) {
    const uint32_t coefficient = base + tile_rank;
    U96 raw = square_coefficient(input, coefficient);
    uint32_t digit = 0u;
    carry = normalize_tile(raw, carry, tile_mask, tile_rank, kTile,
                           digit, bound_failures);
    if (coefficient < kLimbs) low[coefficient] = digit;
    else high[coefficient - kLimbs] = digit;
    __syncwarp(tile_mask);
  }
  if (carry != 0u) ++bound_failures;
}

__device__ __forceinline__ void truncated_product(
    const uint32_t* left, const uint32_t* right, uint32_t* output,
    uint32_t tile_mask, uint32_t tile_rank, uint32_t& bound_failures) {
  uint64_t carry = 0u;
  for (uint32_t base = 0u; base < kLimbs; base += kTile) {
    const uint32_t active = min_u32(kTile, kLimbs - base);
    const uint32_t coefficient = base + tile_rank;
    U96 raw{0u, 0u};
    if (tile_rank < active)
      raw = product_coefficient_host_device(left, right, coefficient);
    uint32_t digit = 0u;
    carry = normalize_tile(raw, carry, tile_mask, tile_rank, active,
                           digit, bound_failures);
    if (tile_rank < active) output[coefficient] = digit;
    __syncwarp(tile_mask);
  }
}

struct GuardedCarry {
  uint64_t carry;
  bool ambiguous;
};

__host__ __device__ __forceinline__ GuardedCarry carry_from_top_three(
    const U96& coefficient35, const U96& coefficient34,
    const U96& coefficient33) {
  const uint64_t word1_sum = static_cast<uint64_t>(u96_word(coefficient33, 1u)) +
                              u96_word(coefficient34, 0u);
  const uint32_t fraction_word0 = u96_word(coefficient33, 0u);
  const uint32_t fraction_word1 = static_cast<uint32_t>(word1_sum);
  const uint64_t word2_sum = static_cast<uint64_t>(u96_word(coefficient33, 2u)) +
                              u96_word(coefficient34, 1u) +
                              u96_word(coefficient35, 0u) +
                              (word1_sum >> 32u);
  const uint32_t fraction_word2 = static_cast<uint32_t>(word2_sum);
  uint64_t integer = static_cast<uint64_t>(u96_word(coefficient35, 1u)) |
                     (static_cast<uint64_t>(u96_word(coefficient35, 2u)) << 32u);
  integer += u96_word(coefficient34, 2u) + (word2_sum >> 32u);

  // Omitted diagonals 0..32, scaled by B^3, are strictly below 33B+33.
  constexpr uint64_t tail_upper = 33u * kRadix + 33u;
  const uint64_t fraction_low = static_cast<uint64_t>(fraction_word0) |
      (static_cast<uint64_t>(fraction_word1) << 32u);
  bool ambiguous = false;
  if (fraction_word2 == UINT32_MAX && fraction_low != 0u) {
    const uint64_t distance = uint64_t{0} - fraction_low;
    ambiguous = distance <= tail_upper;
  }
  return GuardedCarry{integer, ambiguous};
}

__device__ __forceinline__ uint64_t exact_low_product_carry(
    const uint32_t* left, const uint32_t* right, uint32_t tile_mask,
    uint32_t tile_rank, uint32_t& bound_failures) {
  uint64_t carry = 0u;
  for (uint32_t base = 0u; base < kLimbs; base += kTile) {
    const uint32_t active = min_u32(kTile, kLimbs - base);
    const uint32_t coefficient = base + tile_rank;
    U96 raw{0u, 0u};
    if (tile_rank < active)
      raw = product_coefficient_host_device(left, right, coefficient);
    uint32_t discarded = 0u;
    carry = normalize_tile(raw, carry, tile_mask, tile_rank, active,
                           discarded, bound_failures);
  }
  return carry;
}

__device__ __forceinline__ void product_high_guarded(
    const uint32_t* left, const uint32_t* right, uint32_t* high_output,
    uint32_t tile_mask, uint32_t tile_rank, uint32_t& fallback_count,
    uint32_t& bound_failures) {
  U96 guard_raw{0u, 0u};
  if (tile_rank < 3u)
    guard_raw = product_coefficient_host_device(
        left, right, (kLimbs - 1u) - tile_rank);
  const uint32_t tile_base = (threadIdx.x & 31u) - tile_rank;
  U96 coefficient35{
      shuffle_u64(tile_mask, guard_raw.low, tile_base),
      __shfl_sync(tile_mask, guard_raw.high, tile_base)};
  U96 coefficient34{
      shuffle_u64(tile_mask, guard_raw.low, tile_base + 1u),
      __shfl_sync(tile_mask, guard_raw.high, tile_base + 1u)};
  U96 coefficient33{
      shuffle_u64(tile_mask, guard_raw.low, tile_base + 2u),
      __shfl_sync(tile_mask, guard_raw.high, tile_base + 2u)};
  GuardedCarry guarded{0u, false};
  if (tile_rank == 0u)
    guarded = carry_from_top_three(coefficient35, coefficient34, coefficient33);
  uint64_t carry = shuffle_u64(tile_mask, guarded.carry, tile_base);
  uint32_t ambiguous = __shfl_sync(
      tile_mask, guarded.ambiguous ? 1u : 0u, tile_base);
  if (ambiguous != 0u) {
    carry = exact_low_product_carry(
        left, right, tile_mask, tile_rank, bound_failures);
    if (tile_rank == 0u) ++fallback_count;
  }

  for (uint32_t base = kLimbs; base < kProductLimbs; base += kTile) {
    const uint32_t active = min_u32(kTile, kProductLimbs - base);
    const uint32_t coefficient = base + tile_rank;
    U96 raw{0u, 0u};
    if (tile_rank < active && coefficient < 2u * kLimbs - 1u)
      raw = product_coefficient_host_device(left, right, coefficient);
    uint32_t digit = 0u;
    carry = normalize_tile(raw, carry, tile_mask, tile_rank, active,
                           digit, bound_failures);
    if (tile_rank < active) high_output[coefficient - kLimbs] = digit;
    __syncwarp(tile_mask);
  }
  if (carry != 0u) ++bound_failures;
}

__device__ __forceinline__ int compare_words(const uint32_t* left,
                                              const uint32_t* right) {
#pragma unroll 1
  for (int32_t limb = static_cast<int32_t>(kLimbs) - 1; limb >= 0; --limb) {
    if (left[limb] < right[limb]) return -1;
    if (left[limb] > right[limb]) return 1;
  }
  return 0;
}

__device__ __forceinline__ void subtract_words(uint32_t* value,
                                                const uint32_t* modulus,
                                                uint64_t& top) {
  uint64_t borrow = 0u;
#pragma unroll 1
  for (uint32_t limb = 0u; limb < kLimbs; ++limb) {
    const uint64_t subtrahend = static_cast<uint64_t>(modulus[limb]) + borrow;
    const uint64_t minuend = value[limb];
    value[limb] = static_cast<uint32_t>(minuend - subtrahend);
    borrow = minuend < subtrahend ? 1u : 0u;
  }
  top -= borrow;
}

__device__ __forceinline__ void add_then_reduce(
    const uint32_t* left, const uint32_t* right, uint32_t carry_add,
    const uint32_t* modulus, uint32_t* output, uint32_t tile_mask,
    uint32_t tile_rank, uint32_t& bound_failures) {
  uint64_t carry = carry_add;
  for (uint32_t base = 0u; base < kLimbs; base += kTile) {
    const uint32_t active = min_u32(kTile, kLimbs - base);
    const uint32_t index = base + tile_rank;
    U96 raw{0u, 0u};
    if (tile_rank < active) {
      raw.low = right[index];
      if (left != nullptr) add_u64(raw, left[index]);
    }
    uint32_t digit = 0u;
    carry = normalize_tile(raw, carry, tile_mask, tile_rank, active,
                           digit, bound_failures);
    if (tile_rank < active) output[index] = digit;
    __syncwarp(tile_mask);
  }
  if (tile_rank == 0u) {
    if (carry > 1u) ++bound_failures;
    if (carry != 0u || compare_words(output, modulus) >= 0)
      subtract_words(output, modulus, carry);
    if (carry != 0u || compare_words(output, modulus) >= 0)
      ++bound_failures;
  }
  __syncwarp(tile_mask);
}

__device__ __forceinline__ uint32_t any_nonzero(
    const uint32_t* value, uint32_t tile_mask, uint32_t tile_rank) {
  uint32_t local = 0u;
  for (uint32_t index = tile_rank; index < kLimbs; index += kTile)
    local |= value[index];
  return __any_sync(tile_mask, local != 0u) ? 1u : 0u;
}

__device__ __forceinline__ void montgomery_square_carrylate(
    SharedInstance& state, uint32_t tile_mask, uint32_t tile_rank,
    uint32_t& fallback_count, uint32_t& carry_failures,
    uint32_t& result_failures) {
  square_full(state.current, state.t0_or_p1, state.t1_or_q,
              tile_mask, tile_rank, carry_failures);
  const uint32_t carry_add = any_nonzero(
      state.t0_or_p1, tile_mask, tile_rank);
  truncated_product(state.t0_or_p1, state.negative_inverse, state.current,
                    tile_mask, tile_rank, carry_failures);
  product_high_guarded(state.current, state.modulus, state.t0_or_p1,
                       tile_mask, tile_rank, fallback_count, carry_failures);
  add_then_reduce(state.t1_or_q, state.t0_or_p1, carry_add, state.modulus,
                  state.current, tile_mask, tile_rank, result_failures);
}

__device__ __forceinline__ void from_montgomery_carrylate(
    SharedInstance& state, uint32_t tile_mask, uint32_t tile_rank,
    uint32_t& fallback_count, uint32_t& carry_failures,
    uint32_t& result_failures) {
  const uint32_t carry_add = any_nonzero(state.current, tile_mask, tile_rank);
  truncated_product(state.current, state.negative_inverse, state.t1_or_q,
                    tile_mask, tile_rank, carry_failures);
  product_high_guarded(state.t1_or_q, state.modulus, state.t0_or_p1,
                       tile_mask, tile_rank, fallback_count, carry_failures);
  add_then_reduce(nullptr, state.t0_or_p1, carry_add, state.modulus,
                  state.current, tile_mask, tile_rank, result_failures);
}

__device__ __forceinline__ void modular_double(uint32_t* value,
                                                const uint32_t* modulus) {
  uint64_t carry = 0u;
#pragma unroll 1
  for (uint32_t limb = 0u; limb < kLimbs; ++limb) {
    const uint64_t doubled = (static_cast<uint64_t>(value[limb]) << 1u) | carry;
    value[limb] = static_cast<uint32_t>(doubled);
    carry = doubled >> 32u;
  }
  if (carry != 0u || compare_words(value, modulus) >= 0)
    subtract_words(value, modulus, carry);
}


} // namespace riecoin_tile8_carrylate


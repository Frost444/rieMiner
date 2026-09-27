#pragma once

#include <cstdint>

// Standalone preparation only: no CGBN, GMP, CUDA runtime, allocations or I/O.
// __host__/__device__ become active when a CUDA translation unit includes this
// file; ordinary C++ compilers see the same unsigned-integer implementation.
#if defined(__CUDACC__)
#define RIECOIN_CARRYLATE_PREP_HD __host__ __device__
#else
#define RIECOIN_CARRYLATE_PREP_HD
#endif

namespace riecoin_carrylate_prepare1152 {

constexpr std::uint32_t kBits = 1152u;
constexpr std::uint32_t kLimbs = kBits / 32u;
constexpr std::uint32_t kTriangularProducts = kLimbs * (kLimbs + 1u) / 2u;

struct UInt1152 {
  std::uint32_t limb[kLimbs];  // Little-endian, radix 2^32.
};

struct Prepared1152 {
  UInt1152 negative_inverse;  // -n^{-1} mod R, R = 2^1152.
  UInt1152 two_r_mod_n;       // 2R mod n, always in [0,n).
  std::uint32_t bit_length;
  std::uint32_t initial_doublings;
};

// Optional source-level work accounting, not a count of emitted instructions.
// 666 convolution products do NOT include the 35 coefficient-select low-word
// products or the ten low-word products in the scalar inverse seed.
struct WorkCounts {
  std::uint32_t convolution_mul32x32;
  std::uint32_t coefficient_mul32lo;
  std::uint32_t inverse_seed_mul32lo;
  std::uint32_t modular_doublings;
  std::uint32_t maximum_u96_high;
};

namespace detail {

struct U96 {
  std::uint64_t low;
  std::uint32_t high;
};

RIECOIN_CARRYLATE_PREP_HD inline void add_u64(U96& value,
                                             std::uint64_t addend) {
  const std::uint64_t before = value.low;
  value.low += addend;
  value.high += value.low < before ? 1u : 0u;
}

RIECOIN_CARRYLATE_PREP_HD inline std::uint32_t bit_length(
    const UInt1152& value) {
  for (std::uint32_t word = kLimbs; word != 0u; --word) {
    std::uint32_t top = value.limb[word - 1u];
    if (top == 0u) continue;
    std::uint32_t bits = 32u * (word - 1u);
    do {
      ++bits;
      top >>= 1u;
    } while (top != 0u);
    return bits;
  }
  return 0u;
}

RIECOIN_CARRYLATE_PREP_HD inline int compare(
    const UInt1152& left, const UInt1152& right) {
  for (std::uint32_t word = kLimbs; word != 0u; --word) {
    if (left.limb[word - 1u] < right.limb[word - 1u]) return -1;
    if (left.limb[word - 1u] > right.limb[word - 1u]) return 1;
  }
  return 0;
}

RIECOIN_CARRYLATE_PREP_HD inline std::uint32_t negative_inverse32(
    std::uint32_t odd) {
  // An odd number is its own inverse modulo 2. Each step doubles the number
  // of correct bits: 1 -> 2 -> 4 -> 8 -> 16 -> 32. Wraparound is intentional.
  std::uint32_t inverse = 1u;
  for (std::uint32_t step = 0u; step < 5u; ++step)
    inverse *= 2u - odd * inverse;
  return 0u - inverse;
}

RIECOIN_CARRYLATE_PREP_HD inline bool triangular_negative_inverse(
    const UInt1152& modulus, UInt1152& inverse, WorkCounts* work) {
  const std::uint32_t seed = negative_inverse32(modulus.limb[0]);
  if (work != nullptr) work->inverse_seed_mul32lo = 10u;

  // At coefficient k, carry + sum_{i=1..k} n_i*np_{k-i} is known.
  // Choose np_k = low32(known)*(-n_0^{-1}); adding n_0*np_k cancels its
  // low word exactly. The outgoing carry gives the next coefficient of
  // n*np+1. No full-width Newton products are needed.
  std::uint64_t carry = 1u;
  for (std::uint32_t coefficient = 0u; coefficient < kLimbs; ++coefficient) {
    U96 total{carry, 0u};
    for (std::uint32_t index = 1u; index <= coefficient; ++index) {
      add_u64(total, static_cast<std::uint64_t>(modulus.limb[index]) *
                       inverse.limb[coefficient - index]);
      if (work != nullptr) ++work->convolution_mul32x32;
    }
    const std::uint32_t digit = coefficient == 0u
        ? seed : static_cast<std::uint32_t>(total.low) * seed;
    if (work != nullptr && coefficient != 0u) ++work->coefficient_mul32lo;
    inverse.limb[coefficient] = digit;
    add_u64(total, static_cast<std::uint64_t>(modulus.limb[0]) * digit);
    if (work != nullptr) {
      ++work->convolution_mul32x32;
      if (total.high > work->maximum_u96_high)
        work->maximum_u96_high = total.high;
    }

    // At most 36 products plus a <36*2^32 carry: total <36*2^64.
    // U96 therefore has ample headroom; do not silently accept a violated
    // cancellation/carry invariant if this implementation is later changed.
    if (static_cast<std::uint32_t>(total.low) != 0u || total.high >= kLimbs)
      return false;
    carry = (total.low >> 32u) | (static_cast<std::uint64_t>(total.high) << 32u);
  }
  return true;
}

RIECOIN_CARRYLATE_PREP_HD inline bool double_reduced(
    UInt1152& value, const UInt1152& modulus) {
  // Precondition 0 <= value < modulus < R. Hence 2*value < 2*modulus,
  // and one subtraction suffices, INCLUDING when the top R bit overflows.
  std::uint64_t carry = 0u;
  for (std::uint32_t word = 0u; word < kLimbs; ++word) {
    const std::uint64_t doubled =
        (static_cast<std::uint64_t>(value.limb[word]) << 1u) | carry;
    value.limb[word] = static_cast<std::uint32_t>(doubled);
    carry = doubled >> 32u;
  }
  if (carry != 0u || compare(value, modulus) >= 0) {
    std::uint64_t borrow = 0u;
    for (std::uint32_t word = 0u; word < kLimbs; ++word) {
      const std::uint64_t minuend = value.limb[word];
      const std::uint64_t subtrahend =
          static_cast<std::uint64_t>(modulus.limb[word]) + borrow;
      value.limb[word] = static_cast<std::uint32_t>(minuend - subtrahend);
      borrow = minuend < subtrahend ? 1u : 0u;
    }
    if (carry != borrow) return false;
  }
  return true;
}

}  // namespace detail

// Accepts exactly odd 3 <= n <= 2^1152-1 (the type bounds the upper end).
// On failure, all output words/metadata are zero. Input may alias either
// UInt1152 output field: publication happens only after all reads of n.
// WorkCounts is optional and must not overlap the input or output objects.
RIECOIN_CARRYLATE_PREP_HD inline bool prepare(
    const UInt1152& modulus, Prepared1152& output, WorkCounts* work = nullptr) {
  if (work != nullptr) *work = WorkCounts{};
  const std::uint32_t bits = detail::bit_length(modulus);
  if (bits < 2u || (modulus.limb[0] & 1u) == 0u) {
    output = Prepared1152{};
    return false;
  }

  Prepared1152 prepared{};
  prepared.bit_length = bits;
  if (!detail::triangular_negative_inverse(modulus, prepared.negative_inverse,
                                           work)) {
    output = Prepared1152{};
    return false;
  }

  // For odd n>=3, 2^(bits-1) is STRICTLY below n. Start there, rather than
  // with R-n (which is not reduced when n<R/2), and reach exponent 1153.
  const std::uint32_t first_exponent = bits - 1u;
  prepared.two_r_mod_n.limb[first_exponent >> 5u] =
      std::uint32_t{1} << (first_exponent & 31u);
  for (std::uint32_t exponent = first_exponent; exponent < kBits + 1u;
       ++exponent) {
    if (!detail::double_reduced(prepared.two_r_mod_n, modulus)) {
      output = Prepared1152{};
      return false;
    }
    ++prepared.initial_doublings;
    if (work != nullptr) ++work->modular_doublings;
  }
  output = prepared;
  return true;
}

static_assert(kLimbs == 36u && kTriangularProducts == 666u,
              "1152-bit triangular preparation geometry changed");

}  // namespace riecoin_carrylate_prepare1152

#undef RIECOIN_CARRYLATE_PREP_HD

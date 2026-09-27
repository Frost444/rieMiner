#pragma once

#include <cstddef>
#include <cstdint>

// R141: keep the dense roots materialized, but represent every sparse prime by
// (P^-1 mod p, root for tuple member zero).  The six other roots are an affine
// recurrence because
//
//   root(offset) = -(base + offset) * P^-1 (mod p).
//
// The header is device resident and self-describing so cold reference kernels
// and the hot sparse kernel consume the exact same immutable representation.
namespace riecoin_compressed_root_storage {

#if defined(__CUDACC__)
#define RIECOIN_COMPRESSED_ROOT_HD __host__ __device__
#else
#define RIECOIN_COMPRESSED_ROOT_HD
#endif

constexpr std::uint32_t kMagic = 0x52313431u; // "R141"
constexpr std::uint32_t kVersion = 1u;
constexpr std::uint32_t kHeaderWords = 12u;
constexpr std::uint32_t kOffsetBase = 5u;

RIECOIN_COMPRESSED_ROOT_HD inline constexpr std::size_t words(
    std::uint32_t total, std::uint32_t dense) {
  return static_cast<std::size_t>(kHeaderWords) +
      static_cast<std::size_t>(dense) * 7u +
      static_cast<std::size_t>(total - dense) * 2u;
}

#if defined(__CUDACC__)
__device__ __forceinline__ std::uint32_t add_mod(
    std::uint32_t a, std::uint32_t b, std::uint32_t p) {
  const std::uint64_t sum = static_cast<std::uint64_t>(a) + b;
  return static_cast<std::uint32_t>(sum >= p ? sum - p : sum);
}

__device__ __forceinline__ std::uint32_t subtract_mod(
    std::uint32_t a, std::uint32_t b, std::uint32_t p) {
  return a >= b ? a - b : static_cast<std::uint32_t>(
      static_cast<std::uint64_t>(a) + p - b);
}

__device__ __forceinline__ std::uint32_t dense_count(
    const std::uint32_t* storage) {
  return storage[2u];
}

__device__ __forceinline__ std::uint32_t total_count(
    const std::uint32_t* storage) {
  return storage[3u];
}

__device__ __forceinline__ const std::uint32_t* sparse_inverses(
    const std::uint32_t* storage) {
  return storage + kHeaderWords +
      static_cast<std::size_t>(dense_count(storage)) * 7u;
}

__device__ __forceinline__ const std::uint32_t* sparse_roots0(
    const std::uint32_t* storage) {
  return sparse_inverses(storage) +
      (total_count(storage) - dense_count(storage));
}

__device__ __forceinline__ std::uint32_t advance_for_gap(
    std::uint32_t root, std::uint32_t gap,
    std::uint32_t inverse, std::uint32_t p) {
  const std::uint32_t step2 = add_mod(inverse, inverse, p);
  const std::uint32_t step4 = add_mod(step2, step2, p);
  const std::uint32_t step = gap == 2u ? step2
      : gap == 4u ? step4
      : gap == 6u ? add_mod(step2, step4, p)
      : 0u;
  // Both mainnet patterns use only 2/4/6 gaps.  A malformed immutable
  // header cannot silently reject work: return an impossible root that the
  // complete device verifier turns into a fatal mismatch before mining.
  return step == 0u ? p : subtract_mod(root, step, p);
}

// Dense producer kernels are statically restricted to the dense prefix.  Keep
// that hot access trivial even when the complete mixed-band accessor is kept
// out of line to bound nvcc's whole-translation-unit expansion.
__device__ __forceinline__ std::uint32_t dense_root_at(
    const std::uint32_t* storage, std::uint32_t prime_index,
    std::uint32_t member) {
  return storage[kHeaderWords +
      static_cast<std::size_t>(prime_index) * 7u + member];
}

#ifdef RIECOIN_RLOW_COMPRESSED_ROOT_NOINLINE
__device__ __noinline__
#else
__device__ __forceinline__
#endif
std::uint32_t root_at(
    const std::uint32_t* storage, std::uint32_t p,
    std::uint32_t prime_index, std::uint32_t member) {
  const std::uint32_t dense = dense_count(storage);
  if (prime_index < dense)
    return dense_root_at(storage, prime_index, member);
  const std::uint32_t sparse = prime_index - dense;
  const std::uint32_t inverse = sparse_inverses(storage)[sparse];
  std::uint32_t root = sparse_roots0(storage)[sparse];
  for (std::uint32_t m = 1u; m <= member; ++m) {
    const std::uint32_t gap = storage[kOffsetBase + m] -
        storage[kOffsetBase + m - 1u];
    root = advance_for_gap(root, gap, inverse, p);
  }
  return root;
}
#endif

} // namespace riecoin_compressed_root_storage
#undef RIECOIN_COMPRESSED_ROOT_HD

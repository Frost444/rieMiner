#pragma once

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <stdexcept>
#ifdef RIECOIN_RLOW_COMPRESSED_ROOTS
#include "riecoin_compressed_root_storage.cuh"
#endif

#if defined(__CUDACC__)
#define RLOW_EVENT_HD __host__ __device__
#else
#define RLOW_EVENT_HD
#endif

// Sparse factors are events in a whole 16-macro interval, not 16 independent
// prime scans. Keep the dense/shared-memory producer for p <= macro_positions;
// only the sparse band writes the final bitmap with commutative atomic AND.
// This is the SAME seven-member sieve, not a probabilistic candidate filter.
namespace rlow_sparse_event_sieve {

RLOW_EVENT_HD inline uint64_t first_hit(uint32_t root, uint32_t prime,
                                      uint64_t origin) {
  const uint32_t remainder = static_cast<uint32_t>(origin % prime);
  return root >= remainder ? root - remainder
                           : static_cast<uint64_t>(root) + prime - remainder;
}

inline void validate_interval(uint64_t origin, uint32_t positions,
                              uint32_t macros) {
  const uint64_t span = static_cast<uint64_t>(positions) * macros;
  if (positions == 0u || macros == 0u || (positions & 31u) != 0u ||
      span > UINT32_MAX || origin > UINT64_MAX - span)
    throw std::runtime_error("RLOW event sieve invalid aligned reservation");
}

RLOW_EVENT_HD inline uint32_t word_for_hit(uint64_t hit) {
  return static_cast<uint32_t>(hit >> 5u);
}

#if defined(__CUDACC__)
uint32_t cached_prime_count = 0u, cached_positions = 0u;
uint32_t dense_primes = 0u, sparse_primes = 0u;
const uint32_t* bound_primes = nullptr;
uint64_t scatter_launches = 0u;
double partition_host_ms = 0.0;

inline void reset() {
  cached_prime_count = cached_positions = dense_primes = sparse_primes = 0u;
  bound_primes = nullptr;
  scatter_launches = 0u;
  partition_host_ms = 0.0;
}

__global__ void find_dense_count(const uint32_t* primes, uint32_t count,
                                uint32_t positions, uint32_t* result) {
  // The existing exact prime/root producer supplies a sorted unique table.
  uint32_t low = 0u, high = count;
  while (low < high) {
    const uint32_t middle = low + (high - low) / 2u;
    if (primes[middle] <= positions) low = middle + 1u;
    else high = middle;
  }
  *result = low;
}

inline uint32_t partition(const uint32_t* primes, uint32_t count,
                          uint32_t positions, uint32_t* device_scratch) {
  if (bound_primes == nullptr) {
    const auto begin = std::chrono::steady_clock::now();
    find_dense_count<<<1u, 1u>>>(primes, count, positions, device_scratch);
    cuda_check(cudaGetLastError(), "RLOW event prime partition launch");
    cuda_check(cudaMemcpy(&dense_primes, device_scratch, sizeof(dense_primes),
                          cudaMemcpyDeviceToHost), "RLOW event prime partition");
    if (dense_primes > count)
      throw std::runtime_error("RLOW event prime partition out of range");
    cached_prime_count = count;
    cached_positions = positions;
    sparse_primes = count - dense_primes;
    bound_primes = primes;
    partition_host_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    std::cout << "phase=rlow_sparse_event_partition dense_primes=" << dense_primes
              << " sparse_primes=" << sparse_primes
              << " macro_positions=" << positions
              << " partition_host_ms=" << partition_host_ms << std::endl;
  }
  if (bound_primes != primes || count != cached_prime_count ||
      positions != cached_positions)
    throw std::runtime_error("RLOW event prime binding changed without reset");
  return dense_primes;
}

__global__ void scatter_sparse(uint32_t* survivors, const uint32_t* primes,
    const uint32_t* roots, uint32_t dense_count, uint32_t prime_count,
    uint32_t positions, uint32_t macros, uint64_t origin, uint32_t* error) {
  const uint32_t index = dense_count + blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= prime_count) return;
#ifdef RIECOIN_RLOW_COMPRESSED_ROOTS
  if (riecoin_compressed_root_storage::dense_count(roots) != dense_count ||
      riecoin_compressed_root_storage::total_count(roots) != prime_count) {
    atomicOr(error, 8u); return;
  }
#endif
  const uint32_t p = primes[index];
  if (p <= positions) { atomicOr(error, 1u); return; }
  const uint32_t remainder = static_cast<uint32_t>(origin % p);
  const uint64_t span = static_cast<uint64_t>(positions) * macros;
#ifdef RIECOIN_RLOW_COMPRESSED_ROOTS
  const uint32_t sparse_slot = index - dense_count;
  const uint32_t inverse =
      riecoin_compressed_root_storage::sparse_inverses(roots)[sparse_slot];
  uint32_t root =
      riecoin_compressed_root_storage::sparse_roots0(roots)[sparse_slot];
#endif
  uint32_t member_count = 7u;
#ifdef RIECOIN_A54_OPTIONAL_PRIME_LIMIT
  if (p > RIECOIN_A54_OPTIONAL_PRIME_LIMIT) member_count = 2u;
#endif
  for (uint32_t member = 0u; member < member_count; ++member) {
#ifdef RIECOIN_RLOW_COMPRESSED_ROOTS
    if (member != 0u) {
      const uint32_t gap = roots[
          riecoin_compressed_root_storage::kOffsetBase + member] - roots[
          riecoin_compressed_root_storage::kOffsetBase + member - 1u];
      root = riecoin_compressed_root_storage::advance_for_gap(
          root, gap, inverse, p);
    }
#else
    const uint32_t root = roots[static_cast<size_t>(index) * 7u + member];
#endif
    if (root >= p) { atomicOr(error, 2u); continue; }
    uint64_t hit = root >= remainder ? root - remainder
        : static_cast<uint64_t>(root) + p - remainder;
    for (; hit < span; hit += p)
      atomicAnd(survivors + word_for_hit(hit), ~(1u << (hit & 31u)));
  }
}
#endif
}  // namespace rlow_sparse_event_sieve
#undef RLOW_EVENT_HD

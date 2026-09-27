#pragma once

// R142 opt-in producer.  The final bitmap owns a 512-macro half-open factor
// interval, while the established queue and Q0 boundary remains exactly 64
// macros.  Dense private planes are reused for eight ordered 64-macro tiles;
// sparse primes traverse the complete 512-macro bitmap once.
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifndef RIECOIN_RLOW_BATCH_PRODUCER
#error "RLOW super sieve requires the certified batch producer"
#endif
#ifndef RIECOIN_RLOW_SPARSE_EVENT_SIEVE
#error "RLOW super sieve requires the exact sparse-event sieve"
#endif
#ifdef RIECOIN_RLOW_OVERLAP_PIPELINE
#error "RLOW super sieve is serialized; overlap is a separate causal branch"
#endif
#ifdef RIECOIN_RLOW_INCREMENTAL_REMAINDERS
#error "RLOW super sieve initially excludes recurrent sparse state"
#endif

namespace rlow_super_sieve {

#ifdef RIECOIN_RLOW_SUPER_SIEVE_MACROS
constexpr uint32_t kSuperMacros = RIECOIN_RLOW_SUPER_SIEVE_MACROS;
#else
constexpr uint32_t kSuperMacros = 512u;
#endif
constexpr uint32_t kTileMacros = rlow_batch_producer::kMacros;
constexpr uint32_t kSlices = kSuperMacros / kTileMacros;
static_assert(kTileMacros == 64u,
              "R142 keeps the queue/Q0 allocation and launch boundary at 64 macros");
static_assert(kSuperMacros == 512u && kSuperMacros % kTileMacros == 0u,
              "R142 is exactly eight ordered 64-macro tiles");

uint32_t* bitmap_words = nullptr;
uint32_t configured_words = 0u;
uint32_t next_slice = 0u;
uint64_t active_origin = 0u;
bool active = false;
bool cold_parity_checked = false;
std::array<cudaEvent_t, 7u> events{};

uint64_t builds = 0u;
uint64_t slices_emitted = 0u;
uint64_t discarded_slices = 0u;
uint64_t parity_words = 0u;
uint64_t parity_mismatches = 0u;
uint64_t parity_survivors_a = 0u;
uint64_t parity_survivors_b = 0u;
uint64_t parity_hash_a = 0u;
uint64_t parity_hash_b = 0u;
uint64_t parity_d2h_bytes = 0u;
size_t bitmap_bytes = 0u;
size_t device_total_bytes = 0u;
size_t device_free_before_bytes = 0u;
size_t device_free_high_water_bytes = 0u;
double dense_gpu_ms = 0.0;
double sparse_gpu_ms = 0.0;
double cold_build_gpu_ms = 0.0;
double warm_build_gpu_ms = 0.0;
double slice_scan_gpu_ms = 0.0;
double slice_pack_gpu_ms = 0.0;
double slice_metadata_gpu_ms = 0.0;
double cold_parity_ms = 0.0;

inline uint32_t host_popcount(uint32_t value) {
  uint32_t count = 0u;
  while (value != 0u) {
    value &= value - 1u;
    ++count;
  }
  return count;
}

inline uint64_t hash_words(uint64_t hash, const uint32_t* words, size_t count) {
  constexpr uint64_t kFnvPrime = 1099511628211ull;
  for (size_t i = 0u; i < count; ++i) {
    const uint32_t word = words[i];
    for (uint32_t shift = 0u; shift < 32u; shift += 8u) {
      hash ^= static_cast<uint8_t>(word >> shift);
      hash *= kFnvPrime;
    }
  }
  return hash;
}

inline std::string hex64(uint64_t value) {
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << value;
  return out.str();
}

inline void sample_device_high_water() {
  size_t free_bytes = 0u, total_bytes = 0u;
  cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes),
             "RLOW super sieve device memory sample");
  if (device_total_bytes == 0u) device_total_bytes = total_bytes;
  if (device_free_high_water_bytes == 0u || free_bytes < device_free_high_water_bytes)
    device_free_high_water_bytes = free_bytes;
}

inline void setup(uint32_t words) {
  if (bitmap_words != nullptr || configured_words != 0u || words == 0u ||
      rlow_batch_producer::configured_words != words ||
      rlow_batch_producer::bitmap_words == nullptr)
    throw std::runtime_error(
        "RLOW super sieve setup requires fresh state after 64-macro batch setup");
  try {
    cuda_check(cudaMemGetInfo(&device_free_before_bytes, &device_total_bytes),
               "RLOW super sieve pre-allocation memory sample");
    const uint64_t super_words = static_cast<uint64_t>(kSuperMacros) * words;
    if (super_words > UINT32_MAX || super_words > SIZE_MAX / sizeof(uint32_t))
      throw std::runtime_error("RLOW super sieve bitmap geometry overflow");
    bitmap_bytes = static_cast<size_t>(super_words) * sizeof(uint32_t);
    cuda_check(cudaMalloc(&bitmap_words, bitmap_bytes),
               "RLOW super sieve final bitmap");
    configured_words = words;
    for (auto& event : events)
      cuda_check(cudaEventCreate(&event), "RLOW super sieve phase event");
    sample_device_high_water();
    std::cout << "phase=rlow_super_sieve_ready super_macros=" << kSuperMacros
              << " tile_macros=" << kTileMacros
              << " slices=" << kSlices
              << " bitmap_bytes=" << bitmap_bytes
              << " queue_macros=" << rlow_queue::kMacros
              << " device_used_after_setup_bytes="
              << (device_total_bytes - device_free_high_water_bytes)
              << std::endl;
  } catch (...) {
    cudaError_t first = cudaSuccess;
    for (auto& event : events) {
      if (event != nullptr) {
        const cudaError_t result = cudaEventDestroy(event);
        if (first == cudaSuccess && result != cudaSuccess) first = result;
        event = nullptr;
      }
    }
    if (bitmap_words != nullptr) {
      const cudaError_t result = cudaFree(bitmap_words);
      if (first == cudaSuccess && result != cudaSuccess) first = result;
      bitmap_words = nullptr;
    }
    configured_words = 0u;
    throw;
  }
}

inline void cleanup() {
  if (active && next_slice < kSlices)
    discarded_slices += kSlices - next_slice;
  cudaError_t first = cudaSuccess;
  for (auto& event : events) {
    if (event != nullptr) {
      const cudaError_t result = cudaEventDestroy(event);
      if (first == cudaSuccess && result != cudaSuccess) first = result;
      event = nullptr;
    }
  }
  if (bitmap_words != nullptr) {
    const cudaError_t result = cudaFree(bitmap_words);
    if (first == cudaSuccess && result != cudaSuccess) first = result;
    bitmap_words = nullptr;
  }
  configured_words = 0u;
  active = false;
  next_slice = 0u;
  cuda_check(first, "RLOW super sieve cleanup");
}

inline void ensure_dense_scratch(uint32_t dense_count) {
  if (dense_count == 0u)
    throw std::runtime_error("RLOW super sieve requires a nonempty dense band");
  if (rlow_batch_producer::origin_remainders == nullptr) {
    const size_t entries = static_cast<size_t>(kTileMacros) * dense_count;
    if (entries > SIZE_MAX / sizeof(uint32_t))
      throw std::runtime_error("RLOW super sieve remainder geometry overflow");
    cuda_check(cudaMalloc(&rlow_batch_producer::origin_remainders,
                          entries * sizeof(uint32_t)),
               "RLOW super sieve dense tile remainders");
    rlow_batch_producer::allocated_primes = dense_count;
    sample_device_high_water();
    std::cout << "phase=rlow_super_sieve_remainders_ready primes=" << dense_count
              << " tile_macros=" << kTileMacros
              << " bytes=" << entries * sizeof(uint32_t)
              << " device_used_high_water_bytes="
              << (device_total_bytes - device_free_high_water_bytes)
              << std::endl;
  }
  if (rlow_batch_producer::allocated_primes != dense_count)
    throw std::runtime_error(
        "RLOW super sieve prime geometry changed without setup reset");
}

inline uint32_t dense_plane_count(uint32_t dense_count) {
  const uint64_t pairs = static_cast<uint64_t>(dense_count) * 7u;
  const uint32_t needed = static_cast<uint32_t>((pairs + 255u) / 256u);
  const uint32_t planes = rlow_batch_producer::maximum_planes < needed
      ? rlow_batch_producer::maximum_planes : needed;
  if (planes == 0u)
    throw std::runtime_error("RLOW super sieve has no dense private plane");
  return planes;
}

inline void launch_dense_tile(uint32_t word_count, uint32_t macro_positions,
    const uint32_t* primes, const uint32_t* roots, uint32_t dense_count,
    uint64_t tile_origin, uint32_t planes) {
  rlow_batch_producer::prepare_remainders<<<(dense_count + 255u) / 256u, 256u>>>(
      primes, dense_count, tile_origin, macro_positions,
      rlow_batch_producer::origin_remainders,
      rlow_batch_producer::terminal_status + 1u);
  rlow_batch_producer::root_planes<<<dim3(planes, kTileMacros), 256u,
      word_count * sizeof(uint32_t)>>>(
      rlow_batch_producer::private_planes, macro_positions, primes, roots,
      dense_count * 7u, rlow_batch_producer::origin_remainders, dense_count, 0u);
  rlow_batch_producer::word_owner<<<
      dim3(rlow_batch_producer::configured_chunks, kTileMacros), 256u>>>(
      rlow_batch_producer::bitmap_words, rlow_batch_producer::private_planes,
      word_count, planes, macro_positions, 0u);
}

#ifdef RIECOIN_RLOW_SUPER_SIEVE_EXACT_PARITY
inline void exact_cold_parity(uint32_t word_count, uint32_t macro_positions,
    const uint32_t* primes, const uint32_t* roots, uint32_t prime_count,
    uint32_t dense_count, uint64_t origin, uint32_t planes) {
  if (cold_parity_checked) return;
  const auto begin = std::chrono::steady_clock::now();
  const size_t tile_words = static_cast<size_t>(kTileMacros) * word_count;
  const size_t super_words = static_cast<size_t>(kSuperMacros) * word_count;
  std::vector<uint32_t> candidate(super_words);
  std::vector<uint32_t> reference(tile_words);
  cuda_check(cudaMemcpy(candidate.data(), bitmap_words, bitmap_bytes,
                        cudaMemcpyDeviceToHost),
             "RLOW super sieve candidate parity D2H");
  parity_d2h_bytes += bitmap_bytes;
  constexpr uint64_t kFnvOffset = 14695981039346656037ull;
  uint64_t candidate_hash = kFnvOffset;
  uint64_t reference_hash = kFnvOffset;
  uint64_t candidate_survivors = 0u;
  uint64_t reference_survivors = 0u;
  uint64_t mismatches = 0u;
  std::array<uint64_t, kSuperMacros> candidate_counts{};
  std::array<uint64_t, kSuperMacros> reference_counts{};
  candidate_hash = hash_words(candidate_hash, candidate.data(), candidate.size());
  for (uint32_t macro = 0u; macro < kSuperMacros; ++macro) {
    const size_t begin_word = static_cast<size_t>(macro) * word_count;
    for (uint32_t word = 0u; word < word_count; ++word)
      candidate_counts[macro] += host_popcount(candidate[begin_word + word]);
    candidate_survivors += candidate_counts[macro];
  }

  cuda_check(cudaMemset(rlow_batch_producer::terminal_status, 0,
                        2u * sizeof(uint32_t)),
             "RLOW super sieve reference status reset");
  for (uint32_t tile = 0u; tile < kSlices; ++tile) {
    const uint64_t tile_origin = origin +
        static_cast<uint64_t>(tile) * kTileMacros * macro_positions;
    launch_dense_tile(word_count, macro_positions, primes, roots, dense_count,
                      tile_origin, planes);
    rlow_sparse_event_sieve::scatter_sparse<<<
        (prime_count - dense_count + 255u) / 256u, 256u>>>(
        rlow_batch_producer::bitmap_words, primes, roots, dense_count,
        prime_count, macro_positions, kTileMacros, tile_origin,
        rlow_batch_producer::terminal_status + 1u);
    cuda_check(cudaMemcpy(reference.data(), rlow_batch_producer::bitmap_words,
                          tile_words * sizeof(uint32_t), cudaMemcpyDeviceToHost),
               "RLOW super sieve reference tile D2H");
    parity_d2h_bytes += tile_words * sizeof(uint32_t);
    reference_hash = hash_words(reference_hash, reference.data(), reference.size());
    const size_t candidate_base = static_cast<size_t>(tile) * tile_words;
    for (size_t index = 0u; index < tile_words; ++index)
      mismatches += candidate[candidate_base + index] != reference[index];
    for (uint32_t macro = 0u; macro < kTileMacros; ++macro) {
      const size_t begin_word = static_cast<size_t>(macro) * word_count;
      const uint32_t super_macro = tile * kTileMacros + macro;
      for (uint32_t word = 0u; word < word_count; ++word)
        reference_counts[super_macro] += host_popcount(reference[begin_word + word]);
      reference_survivors += reference_counts[super_macro];
    }
  }
  cuda_check(cudaGetLastError(), "RLOW super sieve parity kernels");
  uint32_t status[2]{};
  cuda_check(cudaMemcpy(status, rlow_batch_producer::terminal_status,
                        sizeof(status), cudaMemcpyDeviceToHost),
             "RLOW super sieve parity terminal guard");
  rlow_queue::added_d2h_bytes += parity_d2h_bytes + sizeof(status);
  mismatches += status[0] != 0u || status[1] != 0u;
  mismatches += candidate_counts != reference_counts;
  mismatches += candidate_survivors != reference_survivors;
  mismatches += candidate_hash != reference_hash;
  parity_words = super_words;
  parity_mismatches = mismatches;
  parity_survivors_a = reference_survivors;
  parity_survivors_b = candidate_survivors;
  parity_hash_a = reference_hash;
  parity_hash_b = candidate_hash;
  cold_parity_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - begin).count();
  rlow_batch_producer::cold_reference_ms += cold_parity_ms;
  std::cout << "phase=rlow_super_sieve_cold_parity"
            << " reference=A_8x64 candidate=B_1x512"
            << " words=" << parity_words
            << " survivors_a=" << parity_survivors_a
            << " survivors_b=" << parity_survivors_b
            << " hash_a=" << hex64(parity_hash_a)
            << " hash_b=" << hex64(parity_hash_b)
            << " mismatches=" << parity_mismatches
            << " order=macro_major_word_bit"
            << " d2h_bytes=" << parity_d2h_bytes
            << " cold_parity_ms=" << cold_parity_ms
            << " terminal=" << (parity_mismatches == 0u ? "PASS" : "FAIL")
            << std::endl;
  if (parity_mismatches != 0u)
    throw std::runtime_error(
        "RLOW super sieve 8x64/1x512 parity failed: no Q0 admitted");
  cold_parity_checked = true;
}
#endif

inline void build(uint32_t word_count, uint32_t macro_positions,
    const uint32_t* primes, const uint32_t* roots, uint32_t pair_count,
    uint64_t origin) {
  if (active || next_slice != 0u || configured_words != word_count ||
      pair_count == 0u || pair_count % 7u != 0u || primes == nullptr ||
      roots == nullptr)
    throw std::runtime_error("RLOW super sieve build binding/state mismatch");
  rlow_sparse_event_sieve::validate_interval(origin, macro_positions, kSuperMacros);
  const uint32_t prime_count = pair_count / 7u;
  const uint32_t dense_count = rlow_sparse_event_sieve::partition(
      primes, prime_count, macro_positions,
      rlow_batch_producer::terminal_status);
  if (dense_count == 0u || dense_count >= prime_count ||
      (macro_positions & 31u) != 0u)
    throw std::runtime_error(
        "RLOW super sieve requires aligned nonempty dense and sparse bands");
  ensure_dense_scratch(dense_count);
  const uint32_t planes = dense_plane_count(dense_count);
  cuda_check(cudaMemset(rlow_batch_producer::terminal_status, 0,
                        2u * sizeof(uint32_t)),
             "RLOW super sieve build status reset");
  cuda_check(cudaEventRecord(events[0]), "RLOW super sieve build begin");
  const size_t tile_words = static_cast<size_t>(kTileMacros) * word_count;
  for (uint32_t tile = 0u; tile < kSlices; ++tile) {
    const uint64_t tile_origin = origin +
        static_cast<uint64_t>(tile) * kTileMacros * macro_positions;
    launch_dense_tile(word_count, macro_positions, primes, roots, dense_count,
                      tile_origin, planes);
    cuda_check(cudaMemcpyAsync(bitmap_words + static_cast<size_t>(tile) * tile_words,
                               rlow_batch_producer::bitmap_words,
                               tile_words * sizeof(uint32_t),
                               cudaMemcpyDeviceToDevice),
               "RLOW super sieve dense tile publish");
  }
  cuda_check(cudaEventRecord(events[1]), "RLOW super sieve dense tiles complete");
  rlow_sparse_event_sieve::scatter_sparse<<<
      (prime_count - dense_count + 255u) / 256u, 256u>>>(
      bitmap_words, primes, roots, dense_count, prime_count, macro_positions,
      kSuperMacros, origin, rlow_batch_producer::terminal_status + 1u);
  ++rlow_sparse_event_sieve::scatter_launches;
  cuda_check(cudaEventRecord(events[2]), "RLOW super sieve sparse scatter complete");
  cuda_check(cudaGetLastError(), "RLOW super sieve build kernels");
  uint32_t status[2]{};
  cuda_check(cudaMemcpy(status, rlow_batch_producer::terminal_status,
                        sizeof(status), cudaMemcpyDeviceToHost),
             "RLOW super sieve build terminal guard");
  rlow_queue::added_d2h_bytes += sizeof(status);
  if (status[0] != 0u || status[1] != 0u)
    throw std::runtime_error(
        "RLOW super sieve invalid prime/root: no Q0 or result claim allowed");
  float dense_ms = 0.0f, sparse_ms = 0.0f;
  cuda_check(cudaEventElapsedTime(&dense_ms, events[0], events[1]),
             "RLOW super sieve dense timing");
  cuda_check(cudaEventElapsedTime(&sparse_ms, events[1], events[2]),
             "RLOW super sieve sparse timing");
  dense_gpu_ms += dense_ms;
  sparse_gpu_ms += sparse_ms;
  rlow_queue::sieve_gpu_ms += dense_ms + sparse_ms;
  if (builds == 0u) cold_build_gpu_ms += dense_ms + sparse_ms;
  else warm_build_gpu_ms += dense_ms + sparse_ms;
  ++builds;
  active_origin = origin;
  active = true;
  next_slice = 0u;
#ifdef RIECOIN_RLOW_SUPER_SIEVE_EXACT_PARITY
  exact_cold_parity(word_count, macro_positions, primes, roots, prime_count,
                    dense_count, origin, planes);
#endif
}

inline void fill_next_slice(uint32_t* words, uint32_t word_count,
    uint32_t macro_positions, const uint32_t* primes, const uint32_t* roots,
    uint32_t pair_count, uint64_t origin,
    const cgbn_mem_t<1280u>* base_origin,
    const cgbn_mem_t<1280u>* primorial,
    cgbn_mem_t<1280u>* candidates, uint64_t* factors,
    uint32_t* count, unsigned long long* total_count) {
  const auto host_begin = std::chrono::steady_clock::now();
  rlow_batch_producer::validate_geometry(
      word_count, macro_positions, origin);
  if (words == nullptr || base_origin == nullptr || primorial == nullptr ||
      candidates == nullptr || factors == nullptr || count == nullptr ||
      total_count == nullptr)
    throw std::runtime_error("RLOW super sieve slice binding mismatch");
  if (!active) build(word_count, macro_positions, primes, roots, pair_count, origin);
  const uint64_t tile_span = static_cast<uint64_t>(kTileMacros) * macro_positions;
  const uint64_t expected_origin = active_origin +
      static_cast<uint64_t>(next_slice) * tile_span;
  if (origin != expected_origin || next_slice >= kSlices)
    throw std::runtime_error(
        "RLOW super sieve requires ordered, gap-free 64-macro slice consumption");
  const size_t tile_words = static_cast<size_t>(kTileMacros) * word_count;
  const uint32_t* slice = bitmap_words +
      static_cast<size_t>(next_slice) * tile_words;
  const uint32_t all_words = kTileMacros * word_count;
  cuda_check(cudaMemset(rlow_batch_producer::overflow_flags, 0, all_words),
             "RLOW super sieve slice overflow reset");
  cuda_check(cudaMemset(rlow_batch_producer::terminal_status, 0,
                        2u * sizeof(uint32_t)),
             "RLOW super sieve slice status reset");
  cuda_check(cudaEventRecord(events[3]), "RLOW super sieve slice begin");
  rlow_batch_producer::scan_words<<<
      dim3(rlow_batch_producer::configured_chunks, kTileMacros), 256u>>>(
      slice, word_count, macro_positions,
      rlow_batch_producer::configured_chunks,
      rlow_batch_producer::word_prefix,
      rlow_batch_producer::chunk_counts, 0u);
  rlow_batch_producer::scan_chunks<<<dim3(1u, kTileMacros), 256u>>>(
      rlow_batch_producer::chunk_counts,
      rlow_batch_producer::configured_chunks,
      rlow_batch_producer::chunk_offsets, rlow_queue::macro_counts, 0u);
  cuda_check(cudaEventRecord(events[4]), "RLOW super sieve slice scan complete");
  rlow_batch_producer::pack_segments<<<
      dim3(rlow_batch_producer::configured_chunks, kTileMacros), 256u>>>(
      slice, word_count, macro_positions,
      rlow_batch_producer::configured_chunks,
      rlow_batch_producer::word_prefix,
      rlow_batch_producer::chunk_offsets, base_origin, primorial, candidates,
      factors, origin, rlow_batch_producer::overflow_flags, 0u);
  cuda_check(cudaEventRecord(events[5]), "RLOW super sieve slice pack complete");
  riecoin_rlow_bridge::rlow_scan_macro_counts<<<1u, 1u>>>(
      rlow_queue::macro_counts, kTileMacros, rlow_queue::macro_offsets, count);
  riecoin_rlow_bridge::rlow_reduce_u8_flags<<<1u, 256u>>>(
      rlow_batch_producer::overflow_flags, all_words,
      rlow_batch_producer::terminal_status);
  cuda_check(cudaEventRecord(events[6]), "RLOW super sieve slice metadata complete");
  cuda_check(cudaGetLastError(), "RLOW super sieve slice kernels");
  uint32_t status[2]{};
  const auto guard_begin = std::chrono::steady_clock::now();
  cuda_check(cudaMemcpy(status, rlow_batch_producer::terminal_status,
                        sizeof(status), cudaMemcpyDeviceToHost),
             "RLOW super sieve slice terminal guard");
  rlow_queue::safety_guard_host_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - guard_begin).count();
  rlow_queue::added_d2h_bytes += sizeof(status);
  if (status[0] != 0u || status[1] != 0u)
    throw std::runtime_error(
        "RLOW super sieve slice overflow: no Q0 or result claim allowed");
  rlow_queue::add_entered_count<<<1u, 1u>>>(count, total_count);
  cuda_check(cudaGetLastError(), "RLOW super sieve entered accounting");
  float elapsed = 0.0f;
  cuda_check(cudaEventElapsedTime(&elapsed, events[3], events[4]),
             "RLOW super sieve slice scan timing");
  slice_scan_gpu_ms += elapsed;
  rlow_queue::scan_gpu_ms += elapsed;
  cuda_check(cudaEventElapsedTime(&elapsed, events[4], events[5]),
             "RLOW super sieve slice pack timing");
  slice_pack_gpu_ms += elapsed;
  rlow_queue::pack_gpu_ms += elapsed;
  cuda_check(cudaEventElapsedTime(&elapsed, events[5], events[6]),
             "RLOW super sieve slice metadata timing");
  slice_metadata_gpu_ms += elapsed;
  rlow_queue::metadata_gpu_ms += elapsed;
  ++rlow_queue::fill_calls;
  ++slices_emitted;
  ++next_slice;
  if (next_slice == kSlices) {
    active = false;
    next_slice = 0u;
  }
  rlow_queue::fill_host_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - host_begin).count();
}

inline uint64_t super_span(uint32_t macro_positions) {
  const uint64_t span = static_cast<uint64_t>(kSuperMacros) * macro_positions;
  if (macro_positions == 0u || span > UINT32_MAX)
    throw std::runtime_error("RLOW super sieve span exceeds 32-bit bitmap mapping");
  return span;
}

}  // namespace rlow_super_sieve

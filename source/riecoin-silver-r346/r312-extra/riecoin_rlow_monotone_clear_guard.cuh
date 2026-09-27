#pragma once

// Independent guard, never invoked automatically and never part of mining data.
// CUDA integration: include at GLOBAL scope after the real sparse-event helper
// is defined (in queued.cu: after the anonymous namespace closes, before main).
// Invoke run_cuda_guard() only after UUID/PCI validation and cudaSetDevice in a
// future authorized Horizon canary with RIECOIN_RLOW_SKIP_CLEARED_BITS enabled.
// The kernel calls the ACTUAL rlow_sparse_event_sieve::clear_hit; there is no
// copied CUDA implementation. No recurrence or optimization is introduced here.
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#if defined(__CUDACC__) && defined(RIECOIN_RLOW_SKIP_CLEARED_BITS)
#include <cuda_runtime.h>
#endif

namespace riecoin_rlow_monotone_clear_guard {

constexpr std::uint32_t kWords = 513U;
constexpr std::uint32_t kRedzoneWords = 2U;
constexpr std::uint32_t kStorageWords = kWords + 2U * kRedzoneWords;
constexpr std::uint32_t kBitsPerWord = 16U;
constexpr std::uint32_t kHits = kWords * kBitsPerWord;
constexpr std::uint32_t kThreads = 128U;
constexpr std::uint32_t kBlocks = (kHits + kThreads - 1U) / kThreads;
static_assert(kWords > kThreads && kHits % kThreads != 0,
              "different blocks must hit each word, including a partial last block");

using Words = std::array<std::uint32_t, kStorageWords>;
struct Fixture {
  Words initial{};
  Words expected{};
  std::array<std::uint32_t, kHits> hits{};
};

inline Fixture make_fixture(unsigned pass) {
  if (pass > 1U) throw std::invalid_argument("monotone guard has exactly two passes");
  Fixture result;
  const std::array<std::uint32_t, 4> masks = pass == 0U
      ? std::array<std::uint32_t, 4>{{0xffffffffU, 0x0f0ff0f0U, 0U, 0x55555555U}}
      : std::array<std::uint32_t, 4>{{0xaaaaaaaaU, 0xf0f00f0fU, 0xffffffffU, 0xaaaaaaaaU}};
  for (std::uint32_t i = 0; i < kStorageWords; ++i)
    result.initial[i] = result.expected[i] = 0xbadc0ffeU ^ (i * 0x01010101U) ^ pass;
  for (std::uint32_t word = 0; word < kWords; ++word) {
    const auto initial = masks[word % masks.size()];
    result.initial[kRedzoneWords + word] = initial;
    // Independent closed-form oracle, NOT derived by replaying the hit list.
    // Pass 0 clears even bits once each; pass 1 clears odd bits once each.
    const auto retained = pass == 0U ? 0xaaaaaaaaU : 0x55555555U;
    result.expected[kRedzoneWords + word] = initial & retained;
    for (std::uint32_t slot = 0; slot < kBitsPerWord; ++slot)
      result.hits[slot * kWords + word] = word * 32U + 2U * slot + pass;
  }
  return result;
}

struct Comparison {
  std::uint32_t mismatches = 0U;
  std::uint32_t first_word = UINT32_MAX;
};
inline Comparison compare_all_words(const Words& observed, const Words& expected) {
  Comparison result;
  for (std::uint32_t word = 0; word < kStorageWords; ++word) {
    if (observed[word] == expected[word]) continue;
    if (result.mismatches == 0U) result.first_word = word;
    ++result.mismatches;
  }
  return result;
}

#if defined(__CUDACC__) && defined(RIECOIN_RLOW_SKIP_CLEARED_BITS)

// Include this file only after the actual sparse-event header has been parsed.
// Using an unknown/wrong namespace here intentionally fails compilation.
__global__ void scatter_unique_hits(std::uint32_t* bitmap,
                                   const std::uint32_t* hits) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kHits) return;
  const auto hit = hits[index];
  rlow_sparse_event_sieve::clear_hit(bitmap + (hit >> 5U), 1U << (hit & 31U));
}

inline void check_cuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string("monotone clear guard ") + operation +
                             ": " + cudaGetErrorString(error));
}
struct DeviceBuffer {
  std::uint32_t* pointer = nullptr;
  ~DeviceBuffer() { if (pointer != nullptr) (void)cudaFree(pointer); }
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  void release_checked() {
    if (pointer == nullptr) return;
    check_cuda(cudaFree(pointer), "free scratch");
    pointer = nullptr;
  }
};

struct Result {
  std::uint32_t passes = 0U;
  std::uint32_t words_checked = 0U;
  std::uint32_t mismatches = 0U;
  double host_ms = 0.0;
};

inline Result run_cuda_guard() {
  const auto begin = std::chrono::steady_clock::now();
  DeviceBuffer storage, hit_list;
  check_cuda(cudaMalloc(&storage.pointer, sizeof(Words)), "allocate bitmap scratch");
  check_cuda(cudaMalloc(&hit_list.pointer, kHits * sizeof(std::uint32_t)), "allocate hit scratch");
  std::uint32_t* const same_bitmap = storage.pointer + kRedzoneWords;
  Result result;
  for (unsigned pass = 0; pass < 2U; ++pass) {
    const auto fixture = make_fixture(pass);
    Words observed{};
    // Synchronous copies and the kernel all use the calling host thread's
    // default stream. In particular pass 1 overwrites these SAME addresses
    // after pass 0 completes, including prior zeros that must become ones.
    check_cuda(cudaMemcpy(storage.pointer, fixture.initial.data(), sizeof(Words),
                           cudaMemcpyHostToDevice), "reset same bitmap addresses");
    check_cuda(cudaMemcpy(hit_list.pointer, fixture.hits.data(), sizeof(fixture.hits),
                           cudaMemcpyHostToDevice), "copy unique hits");
    scatter_unique_hits<<<kBlocks, kThreads>>>(same_bitmap, hit_list.pointer);
    check_cuda(cudaGetLastError(), "scatter unique hits launch");
    check_cuda(cudaMemcpy(observed.data(), storage.pointer, sizeof(Words),
                           cudaMemcpyDeviceToHost), "read all words and redzones");
    const auto comparison = compare_all_words(observed, fixture.expected);
    result.words_checked += kStorageWords;
    result.mismatches += comparison.mismatches;
    ++result.passes;
    if (comparison.mismatches != 0U) {
      std::cout << "phase=rlow_monotone_clear_guard terminal=FAIL pass=" << pass
                << " mismatches=" << comparison.mismatches
                << " first_word=" << comparison.first_word
                << " got=" << observed[comparison.first_word]
                << " expected=" << fixture.expected[comparison.first_word] << std::endl;
      throw std::runtime_error("monotone clear changed exact bitmap; no PRP admitted");
    }
  }
  hit_list.release_checked();
  storage.release_checked();
  result.host_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - begin).count();
  std::cout << "phase=rlow_monotone_clear_guard terminal=PASS"
            << " passes=" << result.passes << " same_allocation=1"
            << " words_checked=" << result.words_checked
            << " unique_hits_per_pass=" << kHits
            << " blocks=" << kBlocks << " threads=" << kThreads
            << " mismatches=" << result.mismatches
            << " device_bytes=" << sizeof(Words) + kHits * sizeof(std::uint32_t)
            << " h2d_bytes=" << 2U * (sizeof(Words) + kHits * sizeof(std::uint32_t))
            << " d2h_bytes=" << 2U * sizeof(Words)
            << " host_ms=" << result.host_ms << std::endl;
  return result;
}
#endif
} // namespace riecoin_rlow_monotone_clear_guard

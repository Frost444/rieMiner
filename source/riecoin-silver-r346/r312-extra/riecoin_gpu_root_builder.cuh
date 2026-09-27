#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>
#include "riecoin_root_reciprocal.hpp"
#ifdef RIECOIN_RLOW_COMPRESSED_ROOTS
#include "riecoin_compressed_root_storage.cuh"
#endif

namespace riecoin_gpu_root_builder {
struct HighProduct {
    __device__ __forceinline__ unsigned long long operator()(unsigned long long a,
                                                            unsigned long long b) const {
        return __umul64hi(a, b);
    }
};

__global__ void build_roots(const std::uint32_t* primes, const std::uint32_t* inverses,
                            std::uint32_t count, const std::uint32_t* words,
                            std::uint32_t word_count, const std::uint32_t* offsets,
                            std::uint32_t* roots) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const std::uint32_t p = primes[i], inv = inverses[i];
    const std::uint64_t mu = UINT64_MAX / p;
    const std::uint32_t b = riecoin_root_reciprocal::word_remainder(words, word_count, p, mu, HighProduct{});
    for (unsigned m = 0; m < 7; ++m) {
        const auto rem = riecoin_root_reciprocal::reduce(std::uint64_t(b) + offsets[m], p, mu, HighProduct{});
        roots[std::size_t(i) * 7 + m] = riecoin_root_reciprocal::reduce(
            std::uint64_t(rem == 0 ? 0 : p - rem) * inv, p, mu, HighProduct{});
    }
}

#ifdef RIECOIN_RLOW_COMPRESSED_ROOTS
__global__ void build_compressed_roots(
    const std::uint32_t* primes, const std::uint32_t* inverses,
    std::uint32_t count, std::uint32_t dense_count,
    const std::uint32_t* words, std::uint32_t word_count,
    const std::uint32_t* offsets, std::uint32_t* storage) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const std::uint32_t p = primes[i], inv = inverses[i];
    const std::uint64_t mu = UINT64_MAX / p;
    const std::uint32_t b = riecoin_root_reciprocal::word_remainder(
        words, word_count, p, mu, HighProduct{});
    if (i < dense_count) {
        auto* dense = storage +
            riecoin_compressed_root_storage::kHeaderWords +
            static_cast<std::size_t>(i) * 7u;
        for (unsigned m = 0; m < 7; ++m) {
            const auto rem = riecoin_root_reciprocal::reduce(
                std::uint64_t(b) + offsets[m], p, mu, HighProduct{});
            dense[m] = riecoin_root_reciprocal::reduce(
                std::uint64_t(rem == 0u ? 0u : p - rem) * inv,
                p, mu, HighProduct{});
        }
    } else {
        const std::uint32_t sparse = i - dense_count;
        auto* inverse_out = storage +
            riecoin_compressed_root_storage::kHeaderWords +
            static_cast<std::size_t>(dense_count) * 7u;
        auto* root0_out = inverse_out + (count - dense_count);
        inverse_out[sparse] = inv;
        const auto rem = riecoin_root_reciprocal::reduce(
            std::uint64_t(b) + offsets[0], p, mu, HighProduct{});
        root0_out[sparse] = riecoin_root_reciprocal::reduce(
            std::uint64_t(rem == 0u ? 0u : p - rem) * inv,
            p, mu, HighProduct{});
    }
}

__global__ void verify_compressed_roots_exact(
    const std::uint32_t* primes, const std::uint32_t* inverses,
    std::uint32_t count, std::uint32_t dense_count,
    const std::uint32_t* words, std::uint32_t word_count,
    const std::uint32_t* offsets, const std::uint32_t* storage,
    unsigned long long* mismatches) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (i == 0u) {
        const auto expected_words = riecoin_compressed_root_storage::words(
            count, dense_count);
        if (storage[0] != riecoin_compressed_root_storage::kMagic ||
            storage[1] != riecoin_compressed_root_storage::kVersion ||
            storage[2] != dense_count || storage[3] != count ||
            storage[4] != expected_words)
            atomicAdd(mismatches, 1ull);
        for (unsigned m = 0; m < 7; ++m)
            if (storage[riecoin_compressed_root_storage::kOffsetBase + m] !=
                offsets[m]) atomicAdd(mismatches, 1ull);
    }
    const std::uint32_t p = primes[i], inv = inverses[i];
    if (p < 2u || inv == 0u || inv >= p) {
        atomicAdd(mismatches, 1ull);
        return;
    }
    if (i >= dense_count) {
        const std::uint32_t sparse = i - dense_count;
        if (riecoin_compressed_root_storage::sparse_inverses(storage)[sparse] != inv)
            atomicAdd(mismatches, 1ull);
    }
    std::uint64_t base_mod = 0u;
    for (std::uint32_t word = word_count; word != 0u; --word)
        base_mod = ((base_mod << 32u) | words[word - 1u]) % p;
    for (unsigned m = 0; m < 7; ++m) {
        const std::uint64_t rem = (base_mod + offsets[m]) % p;
        const std::uint32_t expected = static_cast<std::uint32_t>(
            ((rem == 0u ? 0u : p - rem) * std::uint64_t(inv)) % p);
        const std::uint32_t observed =
            riecoin_compressed_root_storage::root_at(storage, p, i, m);
        if (observed != expected || observed >= p)
            atomicAdd(mismatches, 1ull);
    }
}

#ifdef RIECOIN_RLOW_CANONICAL_DEVICE_TRANSFORM
__device__ __forceinline__ std::uint32_t canonical_effective_inverse(
    std::uint32_t p, const std::uint32_t* encoded,
    std::uint32_t index, std::uint32_t prefix_count,
    std::uint64_t multiplier, std::uint64_t second, std::uint64_t mu) {
    const auto value = encoded[index];
    if (index < prefix_count) return value;
    const auto multiplier_mod = riecoin_root_reciprocal::reduce(
        multiplier, p, mu, HighProduct{});
    const auto product = riecoin_root_reciprocal::reduce(
        std::uint64_t(value) * multiplier_mod, p, mu, HighProduct{});
    if (second == 1U) return product;
    const auto second_mod = riecoin_root_reciprocal::reduce(second,p,mu,HighProduct{});
    return riecoin_root_reciprocal::reduce(std::uint64_t(product)*second_mod,p,mu,HighProduct{});
}

__global__ void build_compressed_roots_canonical(
    const std::uint32_t* primes, const std::uint32_t* encoded_inverses,
    std::uint32_t count, std::uint32_t dense_count,
    std::uint32_t prefix_count, std::uint64_t multiplier, std::uint64_t second,
    const std::uint32_t* words, std::uint32_t word_count,
    const std::uint32_t* offsets, std::uint32_t* storage) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const std::uint32_t p = primes[i];
    const std::uint64_t mu = UINT64_MAX / p;
    const std::uint32_t inv = canonical_effective_inverse(
        p, encoded_inverses, i, prefix_count, multiplier, second, mu);
    const std::uint32_t b = riecoin_root_reciprocal::word_remainder(
        words, word_count, p, mu, HighProduct{});
    if (i < dense_count) {
        auto* dense = storage +
            riecoin_compressed_root_storage::kHeaderWords +
            static_cast<std::size_t>(i) * 7u;
        for (unsigned m = 0; m < 7; ++m) {
            const auto rem = riecoin_root_reciprocal::reduce(
                std::uint64_t(b) + offsets[m], p, mu, HighProduct{});
            dense[m] = riecoin_root_reciprocal::reduce(
                std::uint64_t(rem == 0u ? 0u : p - rem) * inv,
                p, mu, HighProduct{});
        }
    } else {
        const std::uint32_t sparse = i - dense_count;
        auto* inverse_out = storage +
            riecoin_compressed_root_storage::kHeaderWords +
            static_cast<std::size_t>(dense_count) * 7u;
        auto* root0_out = inverse_out + (count - dense_count);
        inverse_out[sparse] = inv;
        const auto rem = riecoin_root_reciprocal::reduce(
            std::uint64_t(b) + offsets[0], p, mu, HighProduct{});
        root0_out[sparse] = riecoin_root_reciprocal::reduce(
            std::uint64_t(rem == 0u ? 0u : p - rem) * inv,
            p, mu, HighProduct{});
    }
}

__global__ void verify_compressed_roots_canonical_exact(
    const std::uint32_t* primes, const std::uint32_t* encoded_inverses,
    std::uint32_t count, std::uint32_t dense_count,
    std::uint32_t prefix_count, std::uint64_t multiplier, std::uint64_t second,
    const std::uint32_t* primorial_words, std::uint32_t primorial_word_count,
    const std::uint32_t* base_words, std::uint32_t base_word_count,
    const std::uint32_t* offsets, const std::uint32_t* storage,
    unsigned long long* mismatches) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (i == 0u) {
        const auto expected_words = riecoin_compressed_root_storage::words(
            count, dense_count);
        if (storage[0] != riecoin_compressed_root_storage::kMagic ||
            storage[1] != riecoin_compressed_root_storage::kVersion ||
            storage[2] != dense_count || storage[3] != count ||
            storage[4] != expected_words)
            atomicAdd(mismatches, 1ull);
        for (unsigned m = 0; m < 7; ++m)
            if (storage[riecoin_compressed_root_storage::kOffsetBase + m] !=
                offsets[m]) atomicAdd(mismatches, 1ull);
    }
    const std::uint32_t p = primes[i];
    const std::uint64_t mu = UINT64_MAX / p;
    const std::uint32_t inv = canonical_effective_inverse(
        p, encoded_inverses, i, prefix_count, multiplier, second, mu);
    if (p < 2u || inv == 0u || inv >= p) {
        atomicAdd(mismatches, 1ull);
        return;
    }
    std::uint64_t primorial_mod = 0u;
    for (std::uint32_t word = primorial_word_count; word != 0u; --word)
        primorial_mod = ((primorial_mod << 32u) |
            primorial_words[word - 1u]) % p;
    if (primorial_mod == 0u || primorial_mod * inv % p != 1u)
        atomicAdd(mismatches, 1ull);
    if (i >= dense_count) {
        const std::uint32_t sparse = i - dense_count;
        if (riecoin_compressed_root_storage::sparse_inverses(storage)[sparse] != inv)
            atomicAdd(mismatches, 1ull);
    }
    std::uint64_t base_mod = 0u;
    for (std::uint32_t word = base_word_count; word != 0u; --word)
        base_mod = ((base_mod << 32u) | base_words[word - 1u]) % p;
    for (unsigned m = 0; m < 7; ++m) {
        const std::uint64_t rem = (base_mod + offsets[m]) % p;
        const std::uint32_t expected = static_cast<std::uint32_t>(
            ((rem == 0u ? 0u : p - rem) * std::uint64_t(inv)) % p);
        const std::uint32_t observed =
            riecoin_compressed_root_storage::root_at(storage, p, i, m);
        if (observed != expected || observed >= p)
            atomicAdd(mismatches, 1ull);
    }
}
#endif
#endif

// Deliberately independent from build_roots: ordinary integer remainder is
// slower than the reciprocal reducer, but it makes a complete live correctness
// gate without returning the root table across PCIe.  The widened recurrence is
// exact for every uint32_t prime because (p - 1) * 2^32 + UINT32_MAX < 2^64.
__global__ void verify_roots_exact(const std::uint32_t* primes,
                                   const std::uint32_t* inverses,
                                   std::uint32_t count,
                                   const std::uint32_t* words,
                                   std::uint32_t word_count,
                                   const std::uint32_t* offsets,
                                   const std::uint32_t* roots,
                                   unsigned long long* mismatches) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const std::uint32_t p = primes[i], inv = inverses[i];
    if (p < 2u || inv == 0u || inv >= p) {
        atomicAdd(mismatches, 1ull);
        return;
    }
    std::uint64_t base_mod = 0u;
    for (std::uint32_t word = word_count; word != 0u; --word)
        base_mod = ((base_mod << 32u) | words[word - 1u]) % p;
    for (unsigned m = 0; m < 7; ++m) {
        const std::uint32_t root = roots[std::size_t(i) * 7u + m];
        const std::uint64_t rem = (base_mod + offsets[m]) % p;
        const std::uint32_t expected = static_cast<std::uint32_t>(
            ((rem == 0u ? 0u : p - rem) * std::uint64_t(inv)) % p);
        if (root != expected || root >= p) atomicAdd(mismatches, 1ull);
    }
}

inline void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
struct Buffer {
    std::uint32_t* data = nullptr;
    explicit Buffer(std::size_t words) { check(cudaMalloc(&data, words * sizeof(std::uint32_t)), "gpu roots alloc"); }
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

struct CounterBuffer {
    unsigned long long* data = nullptr;
    CounterBuffer() { check(cudaMalloc(&data, sizeof(*data)), "gpu roots counter alloc"); }
    ~CounterBuffer() { if (data) cudaFree(data); }
    CounterBuffer(const CounterBuffer&) = delete;
    CounterBuffer& operator=(const CounterBuffer&) = delete;
};

// Move-only ownership crosses the precompute/setup boundary once.  release_*()
// hands the final allocations to the unchanged stage cleanup path.
struct DeviceRootTable {
    std::uint32_t* primes = nullptr;
    std::uint32_t* first_factors = nullptr;
    std::uint32_t prime_count = 0u;
    std::uint32_t dense_prime_count = 0u;
    std::uint64_t root_storage_words = 0u;
    std::uint64_t exact_roots_checked = 0u;
    DeviceRootTable() = default;
    ~DeviceRootTable() {
        if (primes) cudaFree(primes);
        if (first_factors) cudaFree(first_factors);
    }
    DeviceRootTable(const DeviceRootTable&) = delete;
    DeviceRootTable& operator=(const DeviceRootTable&) = delete;
    DeviceRootTable(DeviceRootTable&& other) noexcept { *this = std::move(other); }
    DeviceRootTable& operator=(DeviceRootTable&& other) noexcept {
        if (this == &other) return *this;
        if (primes) cudaFree(primes);
        if (first_factors) cudaFree(first_factors);
        primes = other.primes;
        first_factors = other.first_factors;
        prime_count = other.prime_count;
        dense_prime_count = other.dense_prime_count;
        root_storage_words = other.root_storage_words;
        exact_roots_checked = other.exact_roots_checked;
        other.primes = nullptr;
        other.first_factors = nullptr;
        other.prime_count = 0u;
        other.dense_prime_count = 0u;
        other.root_storage_words = 0u;
        other.exact_roots_checked = 0u;
        return *this;
    }
    std::uint32_t* release_primes() noexcept {
        auto* value = primes;
        primes = nullptr;
        return value;
    }
    std::uint32_t* release_first_factors() noexcept {
        auto* value = first_factors;
        first_factors = nullptr;
        return value;
    }
};
}

namespace riecoin_prime_root_cache {
// Opt-in wrapper renames the original CPU roots_for to cpu_roots_for before
// including this adapter. Cache key, validation, prime basis and inverse values
// are unchanged. Roots still depend on the COMPLETE current target.
inline exact::RootTable roots_for(const Basis& basis, mpz_srcptr base,
                                 const std::array<std::uint32_t, 7>& offsets) {
    using namespace riecoin_gpu_root_builder;
    const auto begin = std::chrono::steady_clock::now();
    if (basis.first > basis.all_primes.size() ||
        basis.inverses.size() != basis.all_primes.size() - basis.first || mpz_sgn(base) < 0)
        throw std::invalid_argument("invalid GPU root geometry");
    const std::size_t n = basis.inverses.size();
    if (n == 0 || mpz_sizeinbase(base, 2) > 4096) {
        std::cout << "phase=gpu_roots fallback=cpu reason=unsupported_shape" << std::endl;
        return cpu_roots_for(basis, base, offsets);
    }
    if (n > UINT32_MAX || n > std::vector<std::uint32_t>().max_size() / 7)
        throw std::length_error("GPU root count overflow");
    exact::RootTable out;
    out.primes.assign(basis.all_primes.begin() + basis.first, basis.all_primes.end());
    out.first_factors.resize(n * 7);
    std::vector<std::uint32_t> words((mpz_sizeinbase(base, 2) + 31) / 32, 0);
    std::size_t exported = 0;
    mpz_export(words.data(), &exported, -1, sizeof(std::uint32_t), 0, 0, base);
    Buffer p(n), inv(n), target(words.size()), offset(7), root(n * 7);
    check(cudaMemcpy(p.data, out.primes.data(), n*4, cudaMemcpyHostToDevice), "gpu roots primes");
    check(cudaMemcpy(inv.data, basis.inverses.data(), n*4, cudaMemcpyHostToDevice), "gpu roots inverse");
    check(cudaMemcpy(target.data, words.data(), words.size()*4, cudaMemcpyHostToDevice), "gpu roots target");
    check(cudaMemcpy(offset.data, offsets.data(), 7*4, cudaMemcpyHostToDevice), "gpu roots offsets");
    build_roots<<<static_cast<unsigned>((n + 255) / 256), 256>>>(p.data, inv.data,
        static_cast<std::uint32_t>(n), target.data, static_cast<std::uint32_t>(words.size()), offset.data, root.data);
    check(cudaGetLastError(), "gpu roots launch");
    check(cudaMemcpy(out.first_factors.data(), root.data, n*7*4, cudaMemcpyDeviceToHost), "gpu roots return");
    // Preserve the existing full bitmap CPU oracle (it uses this full root
    // table). In addition, independently verify spread-out live roots with GMP.
    const std::size_t samples = (std::min)(n, std::size_t(512));
    for (std::size_t s = 0; s < samples; ++s) {
        const std::size_t i = samples == 1 ? 0 : s * (n - 1) / (samples - 1);
        const auto prime = out.primes[i];
        const auto b = static_cast<std::uint32_t>(mpz_fdiv_ui(base, prime));
        for (unsigned m = 0; m < 7; ++m) {
            const std::uint64_t rem = (std::uint64_t(b) + offsets[m]) % prime;
            const auto expected = static_cast<std::uint32_t>((rem == 0 ? 0 : prime - rem) * basis.inverses[i] % prime);
            if (out.first_factors[i*7+m] != expected) throw std::runtime_error("gpu_root_live_gmp_mismatch");
        }
    }
    std::cout << "phase=gpu_roots primes=" << n << " gmp_roots_checked=" << samples*7
              << " mismatches=0 extra_h2d_bytes=" << (n*8 + words.size()*4 + 28)
              << " extra_d2h_bytes=" << n*28
              << " ms=" << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()
              << " timing_scope=allocation_copy_kernel_copy_gmp_guard cleanup_charged_in_parent=1" << std::endl;
    return out;
}

inline riecoin_gpu_root_builder::DeviceRootTable roots_for_device(
    const Basis& basis, mpz_srcptr base,
    const std::array<std::uint32_t, 7>& offsets
#ifdef RIECOIN_RLOW_COMPRESSED_ROOTS
    , std::uint32_t dense_limit
#endif
    ) {
    using namespace riecoin_gpu_root_builder;
    const auto begin = std::chrono::steady_clock::now();
#ifdef RIECOIN_RLOW_CANONICAL_DEVICE_TRANSFORM
    const bool direct = basis.exact_direct_geometry && basis.first > 0U &&
        basis.first == basis.requested_first && basis.prefix_inverse_count == 0U &&
        basis.inverse_multiplier == 1U && basis.inverse_multiplier_second == 1U;
    if ((!direct && (basis.first != RIECOIN_RLOW_PRIME_ROOT_CACHE_CANONICAL_FIRST ||
        basis.requested_first < RIECOIN_RLOW_PRIME_ROOT_CACHE_CANONICAL_MIN_FIRST)) ||
        basis.requested_first > basis.first ||
        basis.prefix_inverse_count != basis.first - basis.requested_first ||
        basis.prefix_inverse_count > basis.prefix_inverses.size() ||
        basis.inverses.size() != basis.all_primes.size() - basis.first ||
        basis.requested_primorial_words.empty() || mpz_sgn(base) < 0)
        throw std::invalid_argument(
            "invalid canonical device-resident GPU root geometry");
    const std::size_t prefix_n = basis.prefix_inverse_count;
    const std::size_t prime_first = basis.requested_first;
    const std::size_t n = prefix_n + basis.inverses.size();
#else
    if (basis.first > basis.all_primes.size() ||
        basis.inverses.size() != basis.all_primes.size() - basis.first ||
        mpz_sgn(base) < 0)
        throw std::invalid_argument("invalid device-resident GPU root geometry");
    const std::size_t n = basis.inverses.size();
    const std::size_t prime_first = basis.first;
#endif
    if (n == 0u || n > UINT32_MAX || n > SIZE_MAX / (7u * sizeof(std::uint32_t)) ||
        mpz_sizeinbase(base, 2) > 4096u)
        throw std::invalid_argument("unsupported device-resident GPU root shape");
#ifdef RIECOIN_RLOW_COMPRESSED_ROOTS
    if (offsets[0] != 0u)
        throw std::invalid_argument("compressed roots require zero first offset");
    for (std::size_t m = 1u; m < offsets.size(); ++m) {
        const auto gap = offsets[m] - offsets[m - 1u];
        if (offsets[m] <= offsets[m - 1u] ||
            (gap != 2u && gap != 4u && gap != 6u))
            throw std::invalid_argument("compressed roots unsupported pattern gap");
    }
    const auto prime_begin = basis.all_primes.begin() + prime_first;
    const auto dense_end = std::upper_bound(
        prime_begin, basis.all_primes.end(), dense_limit);
    const std::size_t dense_n = static_cast<std::size_t>(
        std::distance(prime_begin, dense_end));
    if (dense_n == 0u || dense_n >= n || dense_n > UINT32_MAX)
        throw std::invalid_argument("compressed roots require dense and sparse bands");
    const auto storage_words = riecoin_compressed_root_storage::words(
        static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(dense_n));
    if (storage_words > UINT32_MAX || storage_words > SIZE_MAX / sizeof(std::uint32_t))
        throw std::length_error("compressed root storage overflow");
#endif
    std::vector<std::uint32_t> words((mpz_sizeinbase(base, 2) + 31u) / 32u, 0u);
    std::size_t exported = 0u;
    mpz_export(words.data(), &exported, -1, sizeof(std::uint32_t), 0, 0, base);

    DeviceRootTable out;
    try {
        check(cudaMalloc(&out.primes, n * sizeof(std::uint32_t)),
              "device roots final primes alloc");
#ifdef RIECOIN_RLOW_COMPRESSED_ROOTS
        check(cudaMalloc(&out.first_factors,
                         storage_words * sizeof(std::uint32_t)),
              "device roots compressed storage alloc");
        out.dense_prime_count = static_cast<std::uint32_t>(dense_n);
        out.root_storage_words = storage_words;
#else
        check(cudaMalloc(&out.first_factors, n * 7u * sizeof(std::uint32_t)),
              "device roots final factors alloc");
        out.dense_prime_count = static_cast<std::uint32_t>(n);
        out.root_storage_words = static_cast<std::uint64_t>(n) * 7u;
#endif
        out.prime_count = static_cast<std::uint32_t>(n);
        Buffer inverses(n), target(words.size()), device_offsets(7u);
#ifdef RIECOIN_RLOW_CANONICAL_DEVICE_TRANSFORM
        Buffer requested_primorial(basis.requested_primorial_words.size());
#endif
        CounterBuffer mismatch;
        check(cudaMemcpy(out.primes, basis.all_primes.data() + prime_first,
                         n * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
              "device roots primes");
#ifdef RIECOIN_RLOW_CANONICAL_DEVICE_TRANSFORM
        if (prefix_n != 0u)
            check(cudaMemcpy(inverses.data, basis.prefix_inverses.data(),
                             prefix_n * sizeof(std::uint32_t),
                             cudaMemcpyHostToDevice),
                  "device roots prefix inverses");
        check(cudaMemcpy(inverses.data + prefix_n, basis.inverses.data(),
                         basis.inverses.size() * sizeof(std::uint32_t),
                         cudaMemcpyHostToDevice),
              "device roots canonical inverses");
        check(cudaMemcpy(requested_primorial.data,
                         basis.requested_primorial_words.data(),
                         basis.requested_primorial_words.size() *
                             sizeof(std::uint32_t),
                         cudaMemcpyHostToDevice),
              "device roots requested primorial");
#else
        check(cudaMemcpy(inverses.data, basis.inverses.data(),
                         n * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
              "device roots inverses");
#endif
        check(cudaMemcpy(target.data, words.data(),
                         words.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
              "device roots target");
        check(cudaMemcpy(device_offsets.data, offsets.data(),
                         offsets.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
              "device roots offsets");
        check(cudaMemset(mismatch.data, 0, sizeof(*mismatch.data)),
              "device roots mismatch reset");
        const auto blocks = static_cast<unsigned>((n + 255u) / 256u);
#ifdef RIECOIN_RLOW_COMPRESSED_ROOTS
        std::array<std::uint32_t,
            riecoin_compressed_root_storage::kHeaderWords> header{};
        header[0] = riecoin_compressed_root_storage::kMagic;
        header[1] = riecoin_compressed_root_storage::kVersion;
        header[2] = out.dense_prime_count;
        header[3] = out.prime_count;
        header[4] = static_cast<std::uint32_t>(storage_words);
        std::copy(offsets.begin(), offsets.end(),
                  header.begin() + riecoin_compressed_root_storage::kOffsetBase);
        check(cudaMemcpy(out.first_factors, header.data(), sizeof(header),
                         cudaMemcpyHostToDevice),
              "device roots compressed header");
#ifdef RIECOIN_RLOW_CANONICAL_DEVICE_TRANSFORM
        build_compressed_roots_canonical<<<blocks, 256u>>>(
            out.primes, inverses.data, out.prime_count,
            out.dense_prime_count, static_cast<std::uint32_t>(prefix_n),
            basis.inverse_multiplier, basis.inverse_multiplier_second, target.data,
            static_cast<std::uint32_t>(words.size()), device_offsets.data,
            out.first_factors);
#else
        build_compressed_roots<<<blocks, 256u>>>(out.primes, inverses.data,
            out.prime_count, out.dense_prime_count, target.data,
            static_cast<std::uint32_t>(words.size()), device_offsets.data,
            out.first_factors);
#endif
        check(cudaGetLastError(), "device compressed roots build launch");
#ifdef RIECOIN_RLOW_CANONICAL_DEVICE_TRANSFORM
        verify_compressed_roots_canonical_exact<<<blocks, 256u>>>(
            out.primes, inverses.data, out.prime_count,
            out.dense_prime_count, static_cast<std::uint32_t>(prefix_n),
            basis.inverse_multiplier, basis.inverse_multiplier_second, requested_primorial.data,
            static_cast<std::uint32_t>(
                basis.requested_primorial_words.size()),
            target.data, static_cast<std::uint32_t>(words.size()),
            device_offsets.data, out.first_factors, mismatch.data);
#else
        verify_compressed_roots_exact<<<blocks, 256u>>>(
            out.primes, inverses.data, out.prime_count,
            out.dense_prime_count, target.data,
            static_cast<std::uint32_t>(words.size()), device_offsets.data,
            out.first_factors, mismatch.data);
#endif
        check(cudaGetLastError(),
              "device compressed roots exact verification launch");
#else
        build_roots<<<blocks, 256u>>>(out.primes, inverses.data,
            out.prime_count, target.data, static_cast<std::uint32_t>(words.size()),
            device_offsets.data, out.first_factors);
        check(cudaGetLastError(), "device roots build launch");
        verify_roots_exact<<<blocks, 256u>>>(out.primes, inverses.data,
            out.prime_count, target.data, static_cast<std::uint32_t>(words.size()),
            device_offsets.data, out.first_factors, mismatch.data);
        check(cudaGetLastError(), "device roots exact verification launch");
#endif
        unsigned long long mismatches = 0u;
        check(cudaMemcpy(&mismatches, mismatch.data, sizeof(mismatches),
                         cudaMemcpyDeviceToHost),
              "device roots exact verification result");
        if (mismatches != 0u)
            throw std::runtime_error("device_root_complete_exact_relation_mismatch");
        out.exact_roots_checked = static_cast<std::uint64_t>(n) * 7u;
        std::cout << "phase=gpu_roots_device_resident primes=" << n
                  << " exact_roots_checked=" << out.exact_roots_checked
#ifdef RIECOIN_RLOW_CANONICAL_DEVICE_TRANSFORM
                  << " exact_inverses_checked=" << n
                  << " canonical_first=" << basis.first
                  << " requested_first=" << basis.requested_first
                  << " prefix_inverses=" << prefix_n
                  << " inverse_multiplier=" << basis.inverse_multiplier
#endif
                  << " mismatches=0 final_device_bytes="
                  << n * sizeof(std::uint32_t) +
                     out.root_storage_words * sizeof(std::uint32_t)
#ifdef RIECOIN_RLOW_COMPRESSED_ROOTS
                  << " representation=dense7_sparse_seed_inverse"
                  << " dense_primes=" << out.dense_prime_count
                  << " sparse_primes=" << out.prime_count - out.dense_prime_count
                  << " root_storage_words=" << out.root_storage_words
#else
                  << " representation=dense7"
#endif
                  << " extra_h2d_bytes="
                  << (n * 8u + words.size() * 4u + offsets.size() * 4u
#ifdef RIECOIN_RLOW_CANONICAL_DEVICE_TRANSFORM
                      + basis.requested_primorial_words.size() * 4u
#endif
                     )
                  << " extra_d2h_bytes=" << sizeof(mismatches)
                  << " ms=" << std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - begin).count()
                  << " timing_scope=allocation_copy_build_complete_exact_guard"
                  << std::endl;
        return out;
    } catch (...) {
        // out owns and releases any final allocation completed before failure.
        throw;
    }
}
}

#pragma once

#ifdef RIECOIN_RLOW_COMPRESSED_ROOTS
#include "riecoin_compressed_root_storage.cuh"
#endif

// R-LOW: a block owns a complete macro bitmap, never a global bitmap word.
// Prime/root generation is real; no synthetic strike tape or replay is used.
// The second operator transposes the private planes into one exact bitmap.
// All prime members are preserved. The PRP workspace is independently bounded.
#ifdef RIECOIN_RLOW_QUEUE_MACROS
constexpr uint32_t kRlowPrpCapacity = RIECOIN_RLOW_QUEUE_MACROS * 4096u;
#else
constexpr uint32_t kRlowPrpCapacity = 65536u;
#endif
__device__ unsigned long long rlow_strict_prefix_device[6];
uint64_t rlow_exact_sevens = 0u;
#ifdef RIECOIN_RLOW_SCIENCE_Q9
uint64_t rlow_exact_eights = 0u;
uint64_t rlow_exact_nines = 0u;
#endif
uint32_t* rlow_private_planes = nullptr;
uint32_t rlow_plane_count = 0u;
uint32_t rlow_plane_words = 0u;

__global__ void rlow_private_root_planes(
    uint32_t* planes, uint32_t candidate_count, const uint32_t* primes,
    const uint32_t* first_factors, uint32_t pair_count, uint64_t factor_origin) {
  extern __shared__ uint32_t composite[];
  const uint32_t words = (candidate_count + 31u) >> 5u;
  for (uint32_t word = threadIdx.x; word < words; word += blockDim.x)
    composite[word] = 0u;
  __syncthreads();
  for (uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x;
       pair < pair_count; pair += blockDim.x * gridDim.x) {
    const uint32_t p = primes[pair / 7u];
#ifdef RIECOIN_A54_OPTIONAL_PRIME_LIMIT
    if (p > RIECOIN_A54_OPTIONAL_PRIME_LIMIT && pair % 7u >= 2u) continue;
#endif
    const uint32_t origin_mod = static_cast<uint32_t>(factor_origin % p);
#ifdef RIECOIN_RLOW_COMPRESSED_ROOTS
    const uint32_t root = riecoin_compressed_root_storage::root_at(
        first_factors, p, pair / 7u, pair % 7u);
#else
    const uint32_t root = first_factors[pair];
#endif
    uint64_t hit = root >= origin_mod ? root - origin_mod
        : static_cast<uint64_t>(root) + p - origin_mod;
    for (; hit < candidate_count; hit += p)
      atomicOr(composite + (hit >> 5u), 1u << (hit & 31u));
  }
  __syncthreads();
  for (uint32_t word = threadIdx.x; word < words; word += blockDim.x)
    planes[static_cast<size_t>(blockIdx.x) * words + word] = composite[word];
}

__global__ void rlow_private_word_owner(
    uint32_t* survivors, const uint32_t* planes, uint32_t words,
    uint32_t plane_count, uint32_t candidate_count) {
  const uint32_t word = blockIdx.x * blockDim.x + threadIdx.x;
  if (word >= words) return;
  uint32_t composite = 0u;
  for (uint32_t plane = 0u; plane < plane_count; ++plane)
    composite |= planes[static_cast<size_t>(plane) * words + word];
  uint32_t value = ~composite;
  if (word + 1u == words && (candidate_count & 31u) != 0u)
    value &= (1u << (candidate_count & 31u)) - 1u;
  survivors[word] = value;
}

void rlow_sieve_launch(uint32_t* words, uint32_t candidate_count,
                       const uint32_t* primes, const uint32_t* roots,
                       uint32_t pair_count, uint64_t factor_origin) {
  const uint32_t word_count = (candidate_count + 31u) >> 5u;
  if (candidate_count > 8u * 65536u || pair_count == 0u)
    throw std::runtime_error("RLOW private macro geometry unsupported");
  if (rlow_private_planes == nullptr) {
    int ordinal = 0;
    cudaDeviceProp properties{};
    cuda_check(cudaGetDevice(&ordinal), "RLOW current device");
    cuda_check(cudaGetDeviceProperties(&properties, ordinal), "RLOW properties");
    rlow_plane_count = std::max(1u, std::min(
        static_cast<uint32_t>(properties.multiProcessorCount) * 2u,
        (pair_count + 255u) / 256u));
    rlow_plane_words = word_count;
    const size_t bytes = static_cast<size_t>(word_count) * rlow_plane_count * 4u;
    cuda_check(cudaMalloc(&rlow_private_planes, bytes), "RLOW private planes");
    cuda_check(cudaFuncSetAttribute(rlow_private_root_planes,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(word_count * 4u)), "RLOW private shared opt-in");
    std::cout << "phase=rlow_private_macro_ready macro_positions="
              << candidate_count << " planes=" << rlow_plane_count
              << " plane_bytes=" << bytes << " shared_bytes=" << word_count * 4u
              << " prp_workspace=" << kRlowPrpCapacity
              << " global_bitmap_atomics=0 hot_path_reschedules=0" << std::endl;
  }
  if (word_count != rlow_plane_words)
    throw std::runtime_error("RLOW macro shape changed without ownership reset");
  rlow_private_root_planes<<<rlow_plane_count, 256u, word_count * 4u>>>(
      rlow_private_planes, candidate_count, primes, roots, pair_count, factor_origin);
  cuda_check(cudaGetLastError(), "RLOW private root planes launch");
  rlow_private_word_owner<<<(word_count + 255u) / 256u, 256u>>>(
      words, rlow_private_planes, word_count, rlow_plane_count, candidate_count);
}

bool rlow_pci_equal(const std::string& expected, const cudaDeviceProp& p) {
  unsigned domain = 0u, bus = 0u, device = 0u, function = 0u;
  char extra = 0;
  if (std::sscanf(expected.c_str(), "%x:%x:%x.%x%c", &domain, &bus,
                  &device, &function, &extra) != 4) return false;
  return domain == static_cast<unsigned>(p.pciDomainID) &&
         bus == static_cast<unsigned>(p.pciBusID) &&
         device == static_cast<unsigned>(p.pciDeviceID) && function == 0u;
}

std::string rlow_funnel_json(uint64_t entered, uint64_t q0_passed,
                            double complete_wall_seconds) {
  std::array<unsigned long long, 6> strict{};
  cuda_check(cudaMemcpyFromSymbol(strict.data(), rlow_strict_prefix_device,
      sizeof(strict)), "RLOW strict prefix counts D2H");
  std::array<uint64_t, 8> counts{};
  counts[0] = entered;
  counts[1] = q0_passed;
  for (size_t i = 0u; i < strict.size(); ++i) counts[i + 2u] = strict[i];
  for (size_t i = 1u; i < counts.size(); ++i)
    if (counts[i] > counts[i - 1u])
      throw std::runtime_error("RLOW strict prefix conservation failed");
  if (rlow_exact_sevens > counts.back())
    throw std::runtime_error("RLOW exact Q7 exceeds GPU all-seven candidates");
#ifdef RIECOIN_RLOW_SCIENCE_Q9
  if (rlow_exact_nines > rlow_exact_eights ||
      rlow_exact_eights > rlow_exact_sevens)
    throw std::runtime_error("RLOW strict Q7/Q8/Q9 conservation failed");
#endif
  std::ostringstream out;
  out << std::fixed << std::setprecision(6)
      << "{\"operator\":\"direct_roots_private_macro_unique_word_owner\""
      << ",\"global_bitmap_atomics\":0,\"hot_path_reschedules\":0"
      << ",\"prp_workspace\":" << kRlowPrpCapacity
      << ",\"strict_prefix_counts\":[";
  for (size_t i = 0u; i < counts.size(); ++i) {
    if (i) out << ',';
    out << counts[i];
  }
  out << "],\"r_vector\":[";
  for (size_t i = 0u; i < 7u; ++i) {
    if (i) out << ',';
    if (counts[i + 1u] == 0u) out << "null";
    else out << static_cast<double>(counts[i]) / counts[i + 1u];
  }
  out << "],\"r_global\":";
  if (counts.back() == 0u) out << "null";
  else out << std::pow(static_cast<double>(entered) / counts.back(), 1.0 / 7.0);
  out << ",\"q7_exact\":" << rlow_exact_sevens
      << ",\"q7_exact_s\":" << (complete_wall_seconds > 0.0
          ? rlow_exact_sevens / complete_wall_seconds : 0.0)
#ifdef RIECOIN_RLOW_SCIENCE_Q9
      << ",\"q8_strict_verified\":" << rlow_exact_eights
      << ",\"q8_strict_verified_s\":" << (complete_wall_seconds > 0.0
          ? rlow_exact_eights / complete_wall_seconds : 0.0)
      << ",\"q9_strict_verified\":" << rlow_exact_nines
      << ",\"q9_strict_verified_s\":" << (complete_wall_seconds > 0.0
          ? rlow_exact_nines / complete_wall_seconds : 0.0)
      << ",\"q8_q9_scope\":\"independent_gmp64_rare_tail_not_primality_certificate\""
#endif
      << ",\"accepted\":0,\"accepted_s\":0"
      << ",\"network_boundary\":\"not_connected_pre_submit\"}";
  return out.str();
}

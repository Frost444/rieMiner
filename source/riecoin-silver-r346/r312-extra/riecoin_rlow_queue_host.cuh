#pragma once

// Fixed, independently owned macro segments remove the per-macro PRP
// boundary. Sixteen new ranges feed one Q0 launch and one tuple launch.
// No queue slot or factor range is replayed; holes are explicitly invalid.
namespace rlow_queue {
constexpr uint32_t kMacros = riecoin_rlow_bridge::kQueueMacroCount;
constexpr uint32_t kMacroCapacity = riecoin_rlow_bridge::kQueueMacroCapacity;
constexpr uint32_t kCapacity = kMacros * kMacroCapacity;
uint32_t* word_prefix = nullptr;
uint32_t* chunk_counts = nullptr;
uint32_t* chunk_offsets = nullptr;
uint32_t* macro_counts = nullptr;
uint32_t* macro_offsets = nullptr;
uint32_t* physical_slots = nullptr;
uint32_t* overflow_total = nullptr;
uint8_t* overflow_flags = nullptr;
bool cold_checked = false;
std::array<cudaEvent_t, kMacros * 4u> fill_events{};
cudaEvent_t metadata_end{};
double fill_host_ms = 0.0, sieve_gpu_ms = 0.0, scan_gpu_ms = 0.0;
double pack_gpu_ms = 0.0, metadata_gpu_ms = 0.0, safety_guard_host_ms = 0.0;
double cold_host_ms = 0.0, cold_copy_host_ms = 0.0, cold_exact_cpu_ms = 0.0;
uint64_t fill_calls = 0u, added_d2h_bytes = 0u;

void setup(uint32_t words) {
  if (words == 0u || word_prefix || chunk_counts || chunk_offsets ||
      macro_counts || macro_offsets || physical_slots || overflow_total ||
      overflow_flags || metadata_end ||
      std::any_of(fill_events.begin(), fill_events.end(),
                  [](cudaEvent_t event) { return event != nullptr; }))
    throw std::runtime_error("RLOW queue setup requires released epoch state");
  // These receipts and the cold construction check belong to THIS target/job.
  cold_checked = false;
  fill_host_ms = sieve_gpu_ms = scan_gpu_ms = pack_gpu_ms = 0.0;
  metadata_gpu_ms = safety_guard_host_ms = 0.0;
  cold_host_ms = cold_copy_host_ms = cold_exact_cpu_ms = 0.0;
  fill_calls = added_d2h_bytes = 0u;
  const uint32_t chunks = (words + 255u) / 256u;
  cuda_check(cudaMalloc(&word_prefix, words * 4u), "RLOW queue word prefixes");
  cuda_check(cudaMalloc(&chunk_counts, chunks * 4u), "RLOW queue chunk counts");
  cuda_check(cudaMalloc(&chunk_offsets, chunks * 4u), "RLOW queue chunk offsets");
  cuda_check(cudaMalloc(&macro_counts, kMacros * 4u), "RLOW queue macro counts");
  cuda_check(cudaMalloc(&macro_offsets, kMacros * 4u), "RLOW queue macro offsets");
  cuda_check(cudaMalloc(&physical_slots, 4u), "RLOW queue physical slots");
  cuda_check(cudaMalloc(&overflow_total, 4u), "RLOW queue overflow total");
  cuda_check(cudaMalloc(&overflow_flags, words), "RLOW queue overflow flags");
  const uint32_t slots = kCapacity;
  cuda_check(cudaMemcpy(physical_slots, &slots, 4u, cudaMemcpyHostToDevice),
             "RLOW queue capacity upload");
  for (auto& event : fill_events)
    cuda_check(cudaEventCreate(&event), "RLOW fill phase event");
  cuda_check(cudaEventCreate(&metadata_end), "RLOW metadata phase event");
}

__global__ void add_entered_count(const uint32_t* count,
                                 unsigned long long* total) {
  if (blockIdx.x == 0u && threadIdx.x == 0u) *total += *count;
}

__global__ void sample_rejected_segments(
    const uint8_t* verdicts, const uint64_t* factors, const uint32_t* counts,
    uint32_t sample_limit, uint64_t* sample_factors,
    unsigned long long* rejected_counts) {
  const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= kCapacity) return;
  const uint32_t macro = slot / kMacroCapacity;
  if (slot % kMacroCapacity >= counts[macro] || verdicts[slot] != 0u) return;
  const unsigned long long output = atomicAdd(rejected_counts, 1ull);
  if (output < sample_limit) sample_factors[output] = factors[slot];
}

void fill(uint32_t* words, uint32_t word_count, uint32_t macro_positions,
          const uint32_t* primes, const uint32_t* roots, uint32_t pair_count,
          uint64_t origin, const cgbn_mem_t<1280u>* base_origin,
          const cgbn_mem_t<1280u>* primorial,
          cgbn_mem_t<1280u>* candidates, uint64_t* factors,
          uint32_t* count, unsigned long long* total_count) {
  using namespace riecoin_rlow_bridge;
  const auto host_begin = std::chrono::steady_clock::now();
  const uint32_t chunks = (word_count + 255u) / 256u;
  cuda_check(cudaMemset(overflow_flags, 0, word_count), "RLOW queue overflow reset");
  for (uint32_t macro = 0u; macro < kMacros; ++macro) {
    const uint64_t macro_origin = origin + static_cast<uint64_t>(macro) * macro_positions;
    cuda_check(cudaEventRecord(fill_events[macro * 4u]), "RLOW fill phase begin");
    rlow_sieve_launch(words, macro_positions, primes, roots, pair_count, macro_origin);
    cuda_check(cudaEventRecord(fill_events[macro * 4u + 1u]), "RLOW fill phase sieve");
    rlow_scan_bitmap_words<<<chunks, 256u>>>(words, word_count, macro_positions,
                                            word_prefix, chunk_counts);
    rlow_scan_chunk_counts<<<1u, 256u>>>(chunk_counts, chunks, chunk_offsets,
                                        macro_counts + macro);
    cuda_check(cudaEventRecord(fill_events[macro * 4u + 2u]), "RLOW fill phase scan");
    rlow_compact_bitmap_segment<<<chunks, 256u>>>(
        words, word_count, macro_positions, word_prefix, chunk_offsets,
        base_origin, primorial, candidates, factors, origin, macro,
        kMacroCapacity, kCapacity, overflow_flags);
    cuda_check(cudaEventRecord(fill_events[macro * 4u + 3u]), "RLOW fill phase pack");
  }
  rlow_scan_macro_counts<<<1u, 1u>>>(macro_counts, kMacros, macro_offsets, count);
  rlow_reduce_u8_flags<<<1u, 256u>>>(overflow_flags, word_count, overflow_total);
  cuda_check(cudaEventRecord(metadata_end), "RLOW fill metadata complete");
  cuda_check(cudaGetLastError(), "RLOW queue fill kernels");
  uint32_t overflow = 0u;
  const auto guard_begin = std::chrono::steady_clock::now();
  cuda_check(cudaMemcpy(&overflow, overflow_total, 4u, cudaMemcpyDeviceToHost),
             "RLOW queue overflow guard");
  safety_guard_host_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - guard_begin).count();
  added_d2h_bytes += 4u;
  if (overflow != 0u)
    throw std::runtime_error("RLOW macro segment overflow: no PRP or result claim allowed");
  add_entered_count<<<1u, 1u>>>(count, total_count);
  cuda_check(cudaGetLastError(), "RLOW queue exact entered accounting");
  for (uint32_t macro = 0u; macro < kMacros; ++macro) {
    float elapsed = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed, fill_events[macro * 4u],
        fill_events[macro * 4u + 1u]), "RLOW sieve subphase time");
    sieve_gpu_ms += elapsed;
    cuda_check(cudaEventElapsedTime(&elapsed, fill_events[macro * 4u + 1u],
        fill_events[macro * 4u + 2u]), "RLOW scan subphase time");
    scan_gpu_ms += elapsed;
    cuda_check(cudaEventElapsedTime(&elapsed, fill_events[macro * 4u + 2u],
        fill_events[macro * 4u + 3u]), "RLOW pack subphase time");
    pack_gpu_ms += elapsed;
  }
  float metadata_ms = 0.0f;
  cuda_check(cudaEventElapsedTime(&metadata_ms, fill_events.back(), metadata_end),
             "RLOW metadata subphase time");
  metadata_gpu_ms += metadata_ms;
  ++fill_calls;
  fill_host_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - host_begin).count();
}

bool cpu_euler_jacobi(mpz_srcptr n) {
  mpz_t exponent, residue, base_two, minus_one;
  mpz_inits(exponent, residue, base_two, minus_one, nullptr);
  mpz_sub_ui(exponent, n, 1u);
  mpz_set(minus_one, exponent);
  mpz_fdiv_q_2exp(exponent, exponent, 1u);
  mpz_set_ui(base_two, 2u);
  mpz_powm(residue, base_two, exponent, n);
  const unsigned mod8 = static_cast<unsigned>(mpz_fdiv_ui(n, 8u));
  const bool pass = mod8 == 1u || mod8 == 7u
      ? mpz_cmp_ui(residue, 1u) == 0 : mpz_cmp(residue, minus_one) == 0;
  mpz_clears(exponent, residue, base_two, minus_one, nullptr);
  return pass;
}

// One cold construction gate, charged to E2E time. Every populated candidate
// is checked exactly; only three PRP verdicts per segment are CPU sampled.
// Rejected-path sampling is not a global false-negative proof.
void check_first_queue(const cgbn_mem_t<1280u>* candidates,
                       const uint64_t* factors, const uint8_t* verdicts,
                       const uint32_t* entered_count, mpz_srcptr base,
                       mpz_srcptr primorial, uint64_t origin,
                       uint32_t macro_positions) {
  if (cold_checked) return;
  const auto cold_begin = std::chrono::steady_clock::now();
  std::array<uint32_t, kMacros> counts{};
  std::vector<cgbn_mem_t<1280u>> packed(kCapacity);
  std::vector<uint64_t> actual_factors(kCapacity);
  std::vector<uint8_t> actual_verdicts(kCapacity);
  uint32_t entered = 0u;
  const auto copy_begin = std::chrono::steady_clock::now();
  cuda_check(cudaMemcpy(counts.data(), macro_counts, sizeof(counts), cudaMemcpyDeviceToHost),
             "RLOW queue cold macro counts");
  cuda_check(cudaMemcpy(packed.data(), candidates, packed.size() * sizeof(packed[0]),
                         cudaMemcpyDeviceToHost), "RLOW queue cold candidates");
  cuda_check(cudaMemcpy(actual_factors.data(), factors, actual_factors.size() * 8u,
                         cudaMemcpyDeviceToHost), "RLOW queue cold factors");
  cuda_check(cudaMemcpy(actual_verdicts.data(), verdicts, actual_verdicts.size(),
                         cudaMemcpyDeviceToHost), "RLOW queue cold verdicts");
  cuda_check(cudaMemcpy(&entered, entered_count, 4u, cudaMemcpyDeviceToHost),
             "RLOW queue cold entered count");
  const auto exact_begin = std::chrono::steady_clock::now();
  cold_copy_host_ms += std::chrono::duration<double, std::milli>(exact_begin - copy_begin).count();
  added_d2h_bytes += sizeof(counts) + packed.size() * sizeof(packed[0]) +
      actual_factors.size() * 8u + actual_verdicts.size() + 4u;
  uint64_t checked = 0u, sampled = 0u, mismatch = 0u;
  mpz_t factor, expected, actual;
  mpz_inits(factor, expected, actual, nullptr);
  for (uint32_t macro = 0u; macro < kMacros; ++macro) {
    if (counts[macro] > kMacroCapacity)
      throw std::runtime_error("RLOW cold macro count exceeds fixed segment");
    const uint64_t begin = origin + static_cast<uint64_t>(macro) * macro_positions;
    uint64_t previous = 0u;
    for (uint32_t local = 0u; local < kMacroCapacity; ++local) {
      const uint32_t slot = macro * kMacroCapacity + local;
      if (local >= counts[macro]) {
        mismatch += actual_verdicts[slot] != 0u;
        continue;
      }
      const uint64_t value = actual_factors[slot];
      mismatch += value < begin || value >= begin + macro_positions ||
                  (local != 0u && value <= previous);
      previous = value;
      mpz_import(factor, 1u, -1, sizeof(value), 0, 0, &value);
      mpz_mul(expected, factor, primorial);
      mpz_add(expected, expected, base);
      mpz_import(actual, 40u, -1, sizeof(uint32_t), 0, 0, packed[slot]._limbs);
      mismatch += mpz_cmp(actual, expected) != 0;
      ++checked;
      if (local == 0u || local == counts[macro] / 2u || local + 1u == counts[macro]) {
        mismatch += (actual_verdicts[slot] != 0u) != cpu_euler_jacobi(expected);
        ++sampled;
      }
    }
  }
  mpz_clears(factor, expected, actual, nullptr);
  mismatch += checked != entered;
  std::cout << "phase=rlow_queue_cold_gate candidates=" << checked
            << " sampled_prp=" << sampled << " mismatches=" << mismatch
            << " global_false_negative_claim=0" << std::endl;
  if (mismatch != 0u) throw std::runtime_error("RLOW segmented queue exactness gate failed");
  cold_exact_cpu_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - exact_begin).count();
  cold_host_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - cold_begin).count();
  cold_checked = true;
}

std::string observed_funnel(uint64_t entered, uint64_t passed, double seconds,
                            double consumer_ms, double rare_exact_ms) {
  std::string result = rlow_funnel_json(entered, passed, seconds);
  const std::string old_name = "direct_roots_private_macro_unique_word_owner";
  const auto name = result.find(old_name);
  if (name != std::string::npos)
    result.replace(name, old_name.size(), "private_macro_segmented_queue_fill");
  std::ostringstream metrics;
  metrics << std::fixed << std::setprecision(6)
      << ",\"queue_macros\":" << kMacros
      << ",\"macro_segment_capacity\":" << kMacroCapacity
      << ",\"prp_boundary\":\"disjoint_macro_segments\""
      << ",\"queue_fill_calls\":" << fill_calls
      << ",\"assembly_host_ms\":" << fill_host_ms
      << ",\"sieve_gpu_ms\":" << sieve_gpu_ms
      << ",\"scan_gpu_ms\":" << scan_gpu_ms
      << ",\"pack_gpu_ms\":" << pack_gpu_ms
      << ",\"queue_metadata_gpu_ms\":" << metadata_gpu_ms
      << ",\"overflow_guard_host_ms\":" << safety_guard_host_ms
      << ",\"cold_queue_host_ms\":" << cold_host_ms
      << ",\"cold_queue_copy_host_ms\":" << cold_copy_host_ms
      << ",\"cold_queue_exact_cpu_ms\":" << cold_exact_cpu_ms
      << ",\"added_d2h_bytes\":" << added_d2h_bytes
      << ",\"added_setup_h2d_bytes\":4,\"added_hot_h2d_bytes\":0"
      << ",\"producer_blocked_by_consumer_ms\":" << consumer_ms
      << ",\"rare_exact_ms\":" << rare_exact_ms
      << ",\"submit_ms\":null,\"submit_scope\":\"network_backend_outside_stage\""
      << ",\"timing_scope\":\"host_and_gpu_subphases_overlap_do_not_sum\"";
  result.insert(result.size() - 1u, metrics.str());
  return result;
}

void release() {
  cudaError_t first = cudaSuccess;
  auto release_pointer = [&first](auto*& pointer) {
    if (pointer != nullptr) {
      const auto result = cudaFree(pointer);
      if (first == cudaSuccess && result != cudaSuccess) first = result;
      pointer = nullptr;
    }
  };
  release_pointer(word_prefix); release_pointer(chunk_counts);
  release_pointer(chunk_offsets); release_pointer(macro_counts);
  release_pointer(macro_offsets); release_pointer(physical_slots);
  release_pointer(overflow_total); release_pointer(overflow_flags);
  auto release_event = [&first](cudaEvent_t& event) {
    if (event != nullptr) {
      const auto result = cudaEventDestroy(event);
      if (first == cudaSuccess && result != cudaSuccess) first = result;
      event = nullptr;
    }
  };
  for (auto& event : fill_events) release_event(event);
  release_event(metadata_end);
  cold_checked = false;
  cuda_check(first, "RLOW queue epoch release");
}
}  // namespace rlow_queue

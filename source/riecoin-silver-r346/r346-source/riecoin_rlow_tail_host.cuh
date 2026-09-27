#pragma once

// Included inside the stage namespace after Options, CUDA checks and RLOW atoms.
// One stage process owns one immutable network job. No CPU candidate generation
// occurs in the hot path; only bounded queue receipts cross the PCIe boundary.
#include "riecoin_rlow_q1_tail_queue.cuh"
#ifdef RIECOIN_RLOW_Q1_INPUT_BATCH
#include "riecoin_q1_frame_batch.hpp"
#endif

namespace rlow_tail_host {
using namespace rlow_q1_tail;
Controller* controller = nullptr;
Binding binding{};
DeviceQueue queue{};
DeviceCounters counters{};
cgbn_mem_t<1280u>* immutable_base = nullptr;
const cgbn_mem_t<1280u>* immutable_primorial = nullptr;
const uint32_t* member_offsets = nullptr;
cgbn_mem_t<1280u>* workspace = nullptr;
uint8_t* verdicts = nullptr;
bool r292_active=false;
uint64_t r292_calls=0,r292_tests=0;
uint32_t* prefixes = nullptr;
uint32_t* chunks = nullptr;
uint32_t* offsets = nullptr;
uint32_t* accepted_count = nullptr;
std::array<cudaEvent_t, 4> events{};
std::array<unsigned long long, 6> physical_tests{};
uint32_t q1_offset = 0u;
uint64_t append_calls = 0u, flush_calls = 0u, maximum_pending = 0u;
uint64_t maximum_flush_age_us = 0u, deadline_late_flushes = 0u;
double q1_gpu_ms = 0.0, tail_gpu_ms = 0.0;
double q1_prp_gpu_ms = 0.0, tail_prp_gpu_ms = 0.0;
double cold_guard_ms = 0.0;
#ifdef RIECOIN_RLOW_TAIL_TRACE
struct TraceRecord {
  uint64_t sequence = 0u, host_begin_us = 0u, host_elapsed_us = 0u;
  uint32_t kind = 0u, input = 0u, accepted = 0u;
  uint32_t pending_before = 0u, pending_after = 0u;
  float gpu_ms = 0.0f, prp_gpu_ms = 0.0f;
};
std::vector<TraceRecord> trace_records;
uint64_t trace_sequence = 0u;
uint64_t trace_steady_anchor_us = 0u, trace_wall_anchor_unix_us = 0u;
bool trace_overflow = false;
constexpr size_t kTraceRecordLimit = 4096u;
#endif
#ifdef RIECOIN_RLOW_Q1_INPUT_BATCH
riecoin_q1_frame_batch::Batch* input_batch = nullptr;
uint64_t* input_factors = nullptr;
std::array<cudaEvent_t, 2> input_copy_events{};
uint64_t input_copy_bytes = 0u, input_flushes = 0u, input_max_pending = 0u;
uint64_t input_max_age_us = 0u;
uint32_t input_threshold = 0u;
double input_copy_gpu_ms = 0.0;
#ifdef RIECOIN_RLOW_DEFER_Q1_COPY_RECEIPT
bool input_copy_timing_pending = false;
#endif
#endif

inline uint64_t now_us() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline void check_error(const char* where) {
  uint32_t error = 0u;
  cuda_check(cudaMemcpy(&error, queue.error, sizeof(error), cudaMemcpyDeviceToHost), where);
  if (error != 0u) {
    if (controller != nullptr) controller->fail();
    throw std::runtime_error(std::string(where) + ": fatal queue/pack error=" + std::to_string(error));
  }
}

inline std::array<float, 2> phase_timings() {
  std::array<float, 2> elapsed{};
  cuda_check(cudaEventElapsedTime(&elapsed[0], events[0], events[3]), "R12 total phase elapsed");
  cuda_check(cudaEventElapsedTime(&elapsed[1], events[1], events[2]), "R12 PRP phase elapsed");
  return elapsed;
}

inline void add_timings(double& total, double& prp, const std::array<float, 2>& elapsed) {
  total += elapsed[0];
  prp += elapsed[1];
}

#ifdef RIECOIN_RLOW_TAIL_TRACE
inline void record_trace(uint32_t kind, uint64_t host_begin_us, uint32_t input,
    uint32_t accepted, uint32_t pending_before, uint32_t pending_after,
    const std::array<float, 2>& elapsed) {
  if (trace_records.size() == kTraceRecordLimit) { trace_overflow = true; return; }
  trace_records.push_back(TraceRecord{++trace_sequence, host_begin_us,
      now_us() - host_begin_us, kind, input, accepted, pending_before,
      pending_after, elapsed[0], elapsed[1]});
}
#endif

inline void launch_member_prp(uint32_t count) {
  if (count == 0u) throw std::runtime_error("R12 empty member PRP launch");
  if(r292_active){
    cuda_check(r298_member_dispatch(workspace,verdicts,nullptr,count,cudaStreamPerThread),"R298 exact member SoS");
    ++r292_calls;r292_tests+=count;return;
  }
  const uint32_t blocks = (count * 8u + riecoin_cuda::kPrpBlockThreads - 1u) /
                          riecoin_cuda::kPrpBlockThreads;
#ifdef RIECOIN_RLOW_LAZY_BASE2_TAIL
#ifndef RIECOIN_RLOW_LAZY_BASE2_Q0
#error "Lazy members require the verified lazy base2 evaluator and cold oracle"
#endif
  // Exactly the already-oracled operator, with optional audit outputs off.
  // No new exponentiation kernel or weaker primality predicate.
  riecoin_lazy_base2_window::oracle_kernel<<<blocks, riecoin_cuda::kPrpBlockThreads>>>(
      workspace, verdicts, nullptr, nullptr, count);
#else
  riecoin_cuda::euler_jacobi_dynamic_1280_tpi8_kernel
      <<<blocks, riecoin_cuda::kPrpBlockThreads>>>(workspace, verdicts, count);
#endif
  cuda_check(cudaGetLastError(), "R12 member PRP launch");
}

inline void cold_pack_guard(const Options& options) {
  const auto start = std::chrono::steady_clock::now();
  std::vector<uint64_t> factors;
  const uint64_t room = options.factor_max - options.factor_origin;
  for (const uint64_t delta : {0ull, 1ull, 31ull, 524287ull, 524288ull,
                              0xffffffffull, 0x100000000ull, 0x100000001ull})
    if (delta < room) factors.push_back(options.factor_origin + delta);
  if (room != 0u) factors.push_back(options.factor_max - 1u);
  if (factors.empty()) throw std::runtime_error("R12 empty pack guard domain");
  const uint32_t count = static_cast<uint32_t>(factors.size());
  std::array<uint32_t, 6> host_offsets{};
  cuda_check(cudaMemcpy(host_offsets.data(), member_offsets, sizeof(host_offsets), cudaMemcpyDeviceToHost),
             "R12 guard member offsets");
  cuda_check(cudaMemcpy(queue.factors, factors.data(), factors.size() * sizeof(uint64_t), cudaMemcpyHostToDevice),
             "R12 guard factors");
  cuda_check(cudaMemcpy(queue.count, &count, sizeof(count), cudaMemcpyHostToDevice), "R12 guard count");
  mpz_t b, p, f, expected, actual;
  mpz_inits(b, p, f, expected, actual, nullptr);
  mpz_import(b, 40u, -1, sizeof(uint32_t), 0, 0, binding.base_at_anchor.data());
  mpz_import(p, 40u, -1, sizeof(uint32_t), 0, 0, binding.primorial.data());
  uint64_t checked = 0u, mismatches = 0u;
#ifdef RIECOIN_RLOW_LAZY_BASE2_TAIL
  uint64_t predicate_checked = 0u, predicate_mismatches = 0u;
#endif
  auto compare = [&](const std::vector<cgbn_mem_t<1280u>>& packed, bool tail) {
    for (size_t i = 0u; i < packed.size(); ++i) {
      const size_t row = tail ? i / 5u : i;
      const size_t member = tail ? i % 5u + 1u : 0u;
      const uint64_t relative = factors[row] - binding.factor_anchor;
      mpz_import(f, 1u, -1, sizeof(relative), 0, 0, &relative);
      mpz_mul(expected, p, f); mpz_add(expected, expected, b);
      mpz_add_ui(expected, expected, host_offsets[member]);
      mpz_import(actual, 40u, -1, sizeof(uint32_t), 0, 0, packed[i]._limbs);
      ++checked;
      if (mpz_sizeinbase(expected, 2) > 1280u || mpz_cmp(actual, expected) != 0) ++mismatches;
    }
#ifdef RIECOIN_RLOW_LAZY_BASE2_TAIL
    if (mismatches != 0u)
      throw std::runtime_error("lazy member pack mismatch before PRP guard");
    cuda_check(cudaMemset(verdicts, 0xa5, packed.size()), "lazy member verdict poison");
    launch_member_prp(static_cast<uint32_t>(packed.size()));
    std::vector<uint8_t> observed(packed.size(), 0xa5);
    cuda_check(cudaMemcpy(observed.data(), verdicts, observed.size(), cudaMemcpyDeviceToHost),
               "lazy member cold predicate receipt");
    for (size_t i = 0u; i < packed.size(); ++i) {
      mpz_import(expected, 40u, -1, sizeof(uint32_t), 0, 0, packed[i]._limbs);
      // Reuse scratch temporaries only after the independent packing check.
      mpz_sub_ui(f, expected, 1u); mpz_fdiv_q_2exp(f, f, 1u);
      mpz_set_ui(actual, 2u); mpz_powm(actual, actual, f, expected);
      const auto mod8 = mpz_fdiv_ui(expected, 8u);
      const bool plus = mod8 == 1u || mod8 == 7u;
      mpz_sub_ui(f, expected, 1u);
      const bool probable = plus ? mpz_cmp_ui(actual, 1u) == 0 : mpz_cmp(actual, f) == 0;
      ++predicate_checked;
      if (observed[i] != static_cast<uint8_t>(probable)) ++predicate_mismatches;
    }
#endif
  };
  try {
    construct_q1<<<(count + 255u) / 256u, 256u>>>(queue.factors, count, immutable_base,
        immutable_primorial, binding.factor_anchor, q1_offset, workspace, queue.error);
    cuda_check(cudaGetLastError(), "R12 cold Q1 pack");
    check_error("R12 cold Q1 overflow");
    std::vector<cgbn_mem_t<1280u>> first(count);
    cuda_check(cudaMemcpy(first.data(), workspace, first.size() * sizeof(first[0]), cudaMemcpyDeviceToHost),
               "R12 cold Q1 candidates");
    compare(first, false);
    construct_tail<<<(5u * count + 255u) / 256u, 256u>>>(queue, count, immutable_base,
        immutable_primorial, binding.factor_anchor, member_offsets, workspace);
    cuda_check(cudaGetLastError(), "R12 cold optional pack");
    check_error("R12 cold optional overflow");
    std::vector<cgbn_mem_t<1280u>> tail(5u * count);
    cuda_check(cudaMemcpy(tail.data(), workspace, tail.size() * sizeof(tail[0]), cudaMemcpyDeviceToHost),
               "R12 cold optional candidates");
    compare(tail, true);
  } catch (...) {
    mpz_clears(b, p, f, expected, actual, nullptr); throw;
  }
  mpz_clears(b, p, f, expected, actual, nullptr);
  cuda_check(cudaMemset(queue.count, 0, sizeof(uint32_t)), "R12 cold queue reset");
  cold_guard_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  std::cout << "phase=rlow_q1_tail_cold_pack checked=" << checked << " mismatches=" << mismatches
            << " guard_ms=" << cold_guard_ms << " terminal=" << (mismatches == 0u ? "PASS" : "FAIL") << std::endl;
  if (mismatches != 0u) throw std::runtime_error("R12 GPU pack differs from GMP: no PRP admitted");
#ifdef RIECOIN_RLOW_LAZY_BASE2_TAIL
  std::cout << "phase=rlow_lazy_member_cold_guard checked=" << predicate_checked
            << " mismatches=" << predicate_mismatches
            << " terminal=" << (predicate_mismatches == 0u ? "PASS" : "FAIL") << std::endl;
  if (predicate_mismatches != 0u)
    throw std::runtime_error("lazy member GPU predicate differs from GMP");
#endif
}

inline void setup(const Options& options, uint32_t capacity,
    const cgbn_mem_t<1280u>* initial_base, const cgbn_mem_t<1280u>* primorial,
    const uint32_t* tuple_offsets, cgbn_mem_t<1280u>* member_workspace,
    uint8_t* member_verdicts, unsigned long long* observed_pass,
    unsigned long long* active, unsigned long long* rejected, uint64_t* samples,
    uint64_t* complete, unsigned long long* complete_count, uint32_t* fatal_error) {
  if (controller != nullptr || capacity == 0u || capacity > UINT32_MAX / 6u)
    throw std::runtime_error("R12 invalid or duplicate setup");
  physical_tests.fill(0u);
  append_calls = flush_calls = maximum_pending = 0u;
  maximum_flush_age_us = deadline_late_flushes = 0u;
  q1_gpu_ms = tail_gpu_ms = q1_prp_gpu_ms = tail_prp_gpu_ms = 0.0;
  cold_guard_ms = 0.0;
#ifdef RIECOIN_RLOW_TAIL_TRACE
  trace_records.clear(); trace_records.reserve(kTraceRecordLimit);
  trace_sequence = 0u; trace_overflow = false;
  trace_steady_anchor_us = now_us();
  trace_wall_anchor_unix_us = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count());
#endif
#ifdef RIECOIN_RLOW_Q1_INPUT_BATCH
  input_copy_bytes = input_flushes = input_max_pending = input_max_age_us = 0u;
  input_threshold = 0u;
  input_copy_gpu_ms = 0.0;
#ifdef RIECOIN_RLOW_DEFER_Q1_COPY_RECEIPT
  input_copy_timing_pending = false;
#endif
#endif
  binding.work_id = options.work_id;
  binding.template_id = options.template_id;
  binding.generation = 1u;
  binding.factor_anchor = options.factor_origin;
  immutable_primorial = primorial; member_offsets = tuple_offsets;
  workspace = member_workspace; verdicts = member_verdicts;
  queue.capacity = capacity; queue.error = fatal_error;
  cuda_check(cudaMalloc(&immutable_base, sizeof(cgbn_mem_t<1280u>)), "R12 immutable base allocation");
  cuda_check(cudaMemcpy(immutable_base, initial_base, sizeof(cgbn_mem_t<1280u>), cudaMemcpyDeviceToDevice),
             "R12 freeze base at anchor");
  cuda_check(cudaMemcpy(binding.base_at_anchor.data(), initial_base, sizeof(cgbn_mem_t<1280u>), cudaMemcpyDeviceToHost),
             "R12 bind base");
  cuda_check(cudaMemcpy(binding.primorial.data(), primorial, sizeof(cgbn_mem_t<1280u>), cudaMemcpyDeviceToHost),
             "R12 bind primorial");
  cuda_check(cudaMemcpy(&q1_offset, tuple_offsets, sizeof(q1_offset), cudaMemcpyDeviceToHost), "R12 Q1 offset");
  cuda_check(cudaMalloc(&queue.factors, static_cast<size_t>(capacity) * sizeof(uint64_t)), "R12 factors allocation");
  cuda_check(cudaMalloc(&queue.count, sizeof(uint32_t)), "R12 count allocation");
  cuda_check(cudaMemset(queue.count, 0, sizeof(uint32_t)), "R12 count reset");
  cuda_check(cudaMalloc(&prefixes, static_cast<size_t>(capacity) * sizeof(uint32_t)), "R12 prefixes allocation");
  const uint32_t chunk_capacity = (capacity + 255u) / 256u;
  cuda_check(cudaMalloc(&chunks, static_cast<size_t>(chunk_capacity) * sizeof(uint32_t)), "R12 chunks allocation");
  cuda_check(cudaMalloc(&offsets, static_cast<size_t>(chunk_capacity) * sizeof(uint32_t)), "R12 offsets allocation");
  cuda_check(cudaMalloc(&accepted_count, sizeof(uint32_t)), "R12 accepted count allocation");
  cuda_check(cudaMalloc(&counters.tested, 6u * sizeof(unsigned long long)), "R12 physical tests allocation");
  cuda_check(cudaMemset(counters.tested, 0, 6u * sizeof(unsigned long long)), "R12 physical tests reset");
  counters.observed_pass = observed_pass; counters.active = active;
  cuda_check(cudaGetSymbolAddress(reinterpret_cast<void**>(&counters.strict), rlow_strict_prefix_device),
             "R12 strict prefix address");
  counters.rejected = rejected; counters.samples = samples;
  counters.sample_limit = options.rejected_audit_per_stage;
  counters.complete = complete; counters.complete_count = complete_count; counters.complete_capacity = capacity;
  for (auto& event : events) cuda_check(cudaEventCreate(&event), "R12 event allocation");
  std::array<uint32_t,6> r292_offsets{};
  cuda_check(cudaMemcpy(r292_offsets.data(),tuple_offsets,sizeof(r292_offsets),cudaMemcpyDeviceToHost),"R298 member offsets bounds");
  const auto r292_bound=r292::bounds(binding.base_at_anchor.data(),binding.primorial.data(),
    options.factor_max-options.factor_origin,r292_offsets.data(),r292::enabled);
  r292_active=r292_bound.admitted;r292_calls=r292_tests=0;
  std::cout<<"phase=r298_tail_route active="<<r292_active<<" minimum_bits="<<r292_bound.minimum_bits
    <<" maximum_bits="<<r292_bound.maximum_bits<<" whole_reservation=1 offsets=6 complete_fallback=1"<<std::endl;
  cold_pack_guard(options);
  // Cold guard launches are independently GMP-checked, not productive tests.
  r292_calls=r292_tests=0;
  controller = new Controller(capacity, std::min(capacity, 512u), 250000u);
  controller->bind(binding, options.factor_origin);
#ifdef RIECOIN_RLOW_Q1_INPUT_BATCH
  cuda_check(cudaMalloc(&input_factors, static_cast<size_t>(capacity) * sizeof(uint64_t)),
             "Q1 batch factor allocation");
  for (auto& event : input_copy_events)
    cuda_check(cudaEventCreate(&event), "Q1 batch copy event allocation");
  // Exercise the real device copy offsets at 32/64-bit boundaries. This is
  // cold scratch data only; no factor or proof is admitted by this guard.
  const std::array<uint64_t, 4> copy_fixture{{0xffffffffull, 0x100000000ull,
      0xaaaa5555cccc3333ull, 0xffffffffffffffffull}};
  std::array<uint64_t, 4> copy_observed{};
  const uint32_t copy_count = std::min(capacity, 4u);
  const uint32_t copy_first = (copy_count + 1u) / 2u;
  auto* copy_source = reinterpret_cast<uint64_t*>(workspace);
  cuda_check(cudaMemcpy(copy_source, copy_fixture.data(), copy_count * sizeof(uint64_t),
      cudaMemcpyHostToDevice), "Q1 batch cold copy source");
  cuda_check(cudaMemcpy(input_factors, copy_source, copy_first * sizeof(uint64_t),
      cudaMemcpyDeviceToDevice), "Q1 batch cold copy first frame");
  if (copy_count != copy_first)
    cuda_check(cudaMemcpy(input_factors + copy_first, copy_source + copy_first,
        (copy_count - copy_first) * sizeof(uint64_t), cudaMemcpyDeviceToDevice),
        "Q1 batch cold copy second frame");
  cuda_check(cudaMemcpy(copy_observed.data(), input_factors, copy_count * sizeof(uint64_t),
      cudaMemcpyDeviceToHost), "Q1 batch cold copy receipt");
  for (uint32_t i = 0; i != copy_count; ++i)
    if (copy_observed[i] != copy_fixture[i])
      throw std::runtime_error("Q1 batch cold copy mismatch");
  std::cout << "phase=rlow_q1_input_copy_guard checked=" << copy_count
            << " mismatches=0 terminal=PASS errors=0" << std::endl;
  input_threshold = std::min(capacity, (r298::batch_enabled&&r292_active)?16384u:4096u);
  input_batch = new riecoin_q1_frame_batch::Batch(
      capacity, input_threshold, 250000u, options.factor_origin);
  std::cout << "phase=rlow_q1_input_batch_ready capacity=" << capacity
            << " threshold=" << input_threshold
            << " max_wait_us=250000 ordering=lossless_contiguous_frames" << std::endl;
#endif
  std::cout << "phase=rlow_q1_tail_ready capacity=" << capacity << " threshold=512 max_wait_us=250000"
            << " binding=immutable_job_base physical_unknowns=explicit" << std::endl;
}

inline void flush(bool force = false) {
  if (controller == nullptr || controller->pending() == 0u) return;
  const uint64_t trace_host_begin_us = now_us();
  const uint64_t now = now_us();
  if (!force && !controller->due(now)) return;
  const uint64_t deadline = controller->deadline_us();
  const uint64_t oldest = deadline - 250000u;
  maximum_flush_age_us = std::max(maximum_flush_age_us, now - oldest);
  if (now > deadline) ++deadline_late_flushes;
  const auto ticket = controller->begin_drain(binding, now, force);
  const uint32_t count = controller->pending();
  cuda_check(cudaEventRecord(events[0]), "R12 optional start");
  construct_tail<<<(5u * count + 255u) / 256u, 256u>>>(queue, count, immutable_base,
      immutable_primorial, binding.factor_anchor, member_offsets, workspace);
  cuda_check(cudaGetLastError(), "R12 optional construction");
  check_error("R12 optional pack guard");
  cuda_check(cudaEventRecord(events[1]), "R12 optional PRP start");
  launch_member_prp(5u * count);
  cuda_check(cudaEventRecord(events[2]), "R12 optional PRP end");
  reduce_tail<<<(count + 255u) / 256u, 256u>>>(verdicts, count, queue, counters);
  commit_drain_queue<<<1u, 1u>>>(queue, count);
  cuda_check(cudaEventRecord(events[3]), "R12 optional end");
  cuda_check(cudaGetLastError(), "R12 optional kernels");
  check_error("R12 optional receipt");
  uint32_t remaining = UINT32_MAX;
  cuda_check(cudaMemcpy(&remaining, queue.count, sizeof(remaining), cudaMemcpyDeviceToHost), "R12 drain count receipt");
  if (remaining != 0u) { controller->fail(); throw std::runtime_error("R12 incomplete GPU drain"); }
  controller->commit_drain(ticket, count);
  const auto elapsed = phase_timings();
  add_timings(tail_gpu_ms, tail_prp_gpu_ms, elapsed); ++flush_calls;
#ifdef RIECOIN_RLOW_TAIL_TRACE
  record_trace(2u, trace_host_begin_us, count, 0u, count, 0u, elapsed);
#endif
}

inline void process_frame(const uint64_t* factors, uint32_t count, uint64_t frame_begin, uint64_t frame_end) {
  if (controller == nullptr) throw std::runtime_error("R12 append before setup");
  const uint64_t trace_host_begin_us = now_us();
  auto append_time = now_us();
  if (!controller->can_append(count, append_time)) {
    flush(true);
    append_time = now_us();
  }
  const Frame frame{frame_begin, frame_end};
  // One decision instant: crossing the oldest-entry deadline between a
  // successful admission check and ticket creation cannot fail the stage.
  const auto ticket = controller->begin_append(binding, frame, count, append_time);
  if (count == 0u) { controller->commit_append(ticket, 0u, now_us()); return; }
  cuda_check(cudaEventRecord(events[0]), "R12 Q1 start");
  construct_q1<<<(count + 255u) / 256u, 256u>>>(factors, count, immutable_base,
      immutable_primorial, binding.factor_anchor, q1_offset, workspace, queue.error);
  cuda_check(cudaGetLastError(), "R12 Q1 construction");
  check_error("R12 Q1 pack guard");
  cuda_check(cudaEventRecord(events[1]), "R12 Q1 PRP start");
  launch_member_prp(count);
  cuda_check(cudaEventRecord(events[2]), "R12 Q1 PRP end");
  const uint32_t blocks = (count + 255u) / 256u;
  riecoin_rlow_bridge::rlow_scan_flags<<<blocks, 256u>>>(verdicts, count, prefixes, chunks);
  riecoin_rlow_bridge::rlow_scan_chunk_counts<<<1u, 256u>>>(chunks, blocks, offsets, accepted_count);
  append_q1<<<blocks, 256u>>>(factors, verdicts, count, prefixes, offsets, accepted_count,
      ticket.pending_before, frame, queue, counters);
  commit_append_queue<<<1u, 1u>>>(queue, ticket.pending_before, accepted_count);
  cuda_check(cudaEventRecord(events[3]), "R12 Q1 end");
  cuda_check(cudaGetLastError(), "R12 Q1 append kernels");
  check_error("R12 Q1 append receipt");
  uint32_t accepted = 0u, total = 0u;
  cuda_check(cudaMemcpy(&accepted, accepted_count, sizeof(accepted), cudaMemcpyDeviceToHost), "R12 Q1 accepted receipt");
  cuda_check(cudaMemcpy(&total, queue.count, sizeof(total), cudaMemcpyDeviceToHost), "R12 Q1 count receipt");
  if (accepted > count || total != ticket.pending_before + accepted) {
    controller->fail(); throw std::runtime_error("R12 Q1 count conservation failure");
  }
  controller->commit_append(ticket, accepted, now_us());
  maximum_pending = std::max<uint64_t>(maximum_pending, total);
  const auto elapsed = phase_timings();
  add_timings(q1_gpu_ms, q1_prp_gpu_ms, elapsed); ++append_calls;
#ifdef RIECOIN_RLOW_TAIL_TRACE
  record_trace(1u, trace_host_begin_us, count, accepted, ticket.pending_before,
      total, elapsed);
#endif
  flush(false);
}

#ifdef RIECOIN_RLOW_Q1_INPUT_BATCH
inline void settle_input_copy_timing() {
#ifdef RIECOIN_RLOW_DEFER_Q1_COPY_RECEIPT
  if (!input_copy_timing_pending) return;
  cuda_check(cudaEventSynchronize(input_copy_events[1]), "Q1 batch deferred copy receipt");
  float elapsed = 0.0f;
  cuda_check(cudaEventElapsedTime(&elapsed, input_copy_events[0], input_copy_events[1]),
             "Q1 batch deferred copy elapsed");
  input_copy_gpu_ms += elapsed;
  input_copy_timing_pending = false;
#endif
}

inline void flush_input(bool force) {
  if (input_batch == nullptr || !input_batch->has_span()) return;
  const auto now = now_us();
  if (!force && !input_batch->due(now)) return;
  const auto span = input_batch->snapshot();
  if (span.count != 0u)
    input_max_age_us = std::max(input_max_age_us, now - input_batch->oldest_us());
  process_frame(input_factors, span.count, span.begin, span.end);
#ifdef RIECOIN_RLOW_DEFER_Q1_COPY_RECEIPT
  // process_frame's exact D2H receipts have already completed every earlier
  // default-stream copy.  Charge timing here instead of stalling each append.
  settle_input_copy_timing();
#endif
  input_batch->commit(span);
  ++input_flushes;
}
#endif

inline void append(const uint64_t* factors, uint32_t count,
                   uint64_t frame_begin, uint64_t frame_end) {
#ifdef RIECOIN_RLOW_Q1_INPUT_BATCH
  if (input_batch == nullptr) throw std::runtime_error("Q1 input batch before setup");
  input_batch->validate(frame_begin, frame_end, count);
  if (input_batch->needs_room(count)) flush_input(true);
  if (count != 0u) {
#ifdef RIECOIN_RLOW_DEFER_Q1_COPY_RECEIPT
    if (input_batch->copy_offset() == 0u)
      cuda_check(cudaEventRecord(input_copy_events[0]), "Q1 batch deferred copy begin");
#else
    cuda_check(cudaEventRecord(input_copy_events[0]), "Q1 batch copy begin");
#endif
    cuda_check(cudaMemcpyAsync(input_factors + input_batch->copy_offset(), factors,
        static_cast<size_t>(count) * sizeof(uint64_t), cudaMemcpyDeviceToDevice),
        "Q1 batch factor copy");
    cuda_check(cudaEventRecord(input_copy_events[1]), "Q1 batch copy end");
#ifdef RIECOIN_RLOW_DEFER_Q1_COPY_RECEIPT
    input_copy_timing_pending = true;
#else
    cuda_check(cudaEventSynchronize(input_copy_events[1]), "Q1 batch copy receipt");
    float elapsed = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed, input_copy_events[0], input_copy_events[1]),
               "Q1 batch copy elapsed");
    input_copy_gpu_ms += elapsed;
#endif
    input_copy_bytes += static_cast<uint64_t>(count) * sizeof(uint64_t);
  }
  input_batch->append(frame_begin, frame_end, count, now_us());
  input_max_pending = std::max<uint64_t>(input_max_pending, input_batch->copy_offset());
  flush_input(false);
#else
  process_frame(factors, count, frame_begin, frame_end);
#endif
}

inline uint64_t pending_total() {
  uint64_t pending = controller == nullptr ? 0u : controller->pending();
#ifdef RIECOIN_RLOW_Q1_INPUT_BATCH
  if (input_batch != nullptr) pending += input_batch->copy_offset();
#endif
  return pending;
}

inline void finish(uint64_t expected_cursor, uint64_t q0_passed) {
#ifdef RIECOIN_RLOW_Q1_INPUT_BATCH
  flush_input(true);
  if (input_batch == nullptr || input_batch->has_span() ||
      input_batch->snapshot().end != expected_cursor ||
      input_copy_bytes != q0_passed * sizeof(uint64_t))
    throw std::runtime_error("Q1 input final frame/count conservation failure");
#endif
  flush(true);
  if (controller == nullptr || !controller->drained() || controller->cursor() != expected_cursor)
    throw std::runtime_error("R12 final pending/frame conservation failure");
  check_error("R12 terminal error guard");
  cuda_check(cudaMemcpy(physical_tests.data(), counters.tested, sizeof(physical_tests), cudaMemcpyDeviceToHost),
             "R12 physical test totals");
  unsigned long long q1_passed = 0u;
  cuda_check(cudaMemcpy(&q1_passed, counters.observed_pass, sizeof(q1_passed), cudaMemcpyDeviceToHost),
             "R12 Q1 survivor total");
  if (physical_tests[0] != q0_passed || controller->drained_count() != q1_passed)
    throw std::runtime_error("R12 first-stage conservation failure");
  for (size_t i = 1u; i < physical_tests.size(); ++i)
    if (physical_tests[i] != q1_passed) throw std::runtime_error("R12 missing/duplicate optional test");
  uint64_t r292_expected=0;for(const auto n:physical_tests)r292_expected+=n;
  if(r292_tests!=(r292_active?r292_expected:0))throw std::runtime_error("R298 member kernel test conservation");
  std::cout<<"phase=r298_tail_complete active="<<r292_active<<" calls="<<r292_calls
    <<" tests="<<r292_tests<<" expected="<<r292_expected<<" errors=0"<<std::endl;
  std::cout << "phase=rlow_q1_tail_drained q1_tests=" << physical_tests[0]
            << " optional_rows=" << q1_passed << " flushes=" << flush_calls
            << " max_pending=" << maximum_pending << " pending=0 errors=0"
            << " max_flush_age_us=" << maximum_flush_age_us
             << " deadline_late_flushes=" << deadline_late_flushes << std::endl;
#ifdef RIECOIN_RLOW_TAIL_TRACE
  for (const auto& record : trace_records) {
    std::cout << "phase=rlow_tail_trace sequence=" << record.sequence
              << " kind=" << (record.kind == 1u ? "q1" : "optional_tail")
              << " host_begin_us=" << record.host_begin_us
              << " host_elapsed_us=" << record.host_elapsed_us
              << " input=" << record.input << " accepted=" << record.accepted
              << " pending_before=" << record.pending_before
              << " pending_after=" << record.pending_after
              << " gpu_ms=" << record.gpu_ms
              << " prp_gpu_ms=" << record.prp_gpu_ms << std::endl;
  }
  std::cout << "phase=rlow_tail_trace_terminal records=" << trace_records.size()
            << " overflow=" << (trace_overflow ? 1 : 0)
            << " steady_anchor_us=" << trace_steady_anchor_us
            << " wall_anchor_unix_us=" << trace_wall_anchor_unix_us << std::endl;
#endif
#ifdef RIECOIN_RLOW_Q1_INPUT_BATCH
  std::cout << "phase=rlow_q1_input_drained input_flushes=" << input_flushes
            << " input_max_pending=" << input_max_pending
            << " input_max_age_us=" << input_max_age_us
            << " copy_gpu_ms=" << input_copy_gpu_ms
            << " copy_bytes=" << input_copy_bytes << " pending=0 errors=0" << std::endl;
#endif
}

inline uint64_t physical_test_total() {
  uint64_t total = 0u;
  for (const auto count : physical_tests) total += count;
  return total;
}

inline void cleanup() {
  auto release = [](auto*& pointer) {
    if (pointer != nullptr) { cuda_check(cudaFree(pointer), "R12 cleanup"); pointer = nullptr; }
  };
  release(immutable_base); release(queue.factors); release(queue.count);
  release(prefixes); release(chunks); release(offsets); release(accepted_count);
  release(counters.tested);
#ifdef RIECOIN_RLOW_Q1_INPUT_BATCH
  release(input_factors);
  for (auto& event : input_copy_events)
    if (event != nullptr) { cuda_check(cudaEventDestroy(event), "Q1 batch event cleanup"); event = nullptr; }
  delete input_batch; input_batch = nullptr;
#endif
  for (auto& event : events)
    if (event != nullptr) { cuda_check(cudaEventDestroy(event), "R12 event cleanup"); event = nullptr; }
  delete controller; controller = nullptr;
  binding = {};
  immutable_primorial = nullptr;
  member_offsets = nullptr;
  workspace = nullptr;
  verdicts = nullptr;
  queue = {};
  counters = {};
  q1_offset = 0u;
  // Timings and terminal physical_test totals survive cleanup for the receipt.
}
}  // namespace rlow_tail_host

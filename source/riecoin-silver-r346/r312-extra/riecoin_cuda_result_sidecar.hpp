#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

// A11: immutable append-prefix receipts. The compute stream publishes a pinned
// snapshot, not a host wait. One consumer owns copies, output and verifier input.
// Full telemetry slots coalesce observations, NEVER results: factors stay in the
// append-only device journal, and the mandatory final snapshot drains its prefix.
// No cudaLaunchHostFunc callback invokes CUDA (forbidden by the CUDA API).
namespace riecoin_result_sidecar {
inline void checked(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
struct Counters {
  unsigned long long complete{}, entries{};
  std::array<unsigned long long, 6> active{}, tested{}, passed{};
};
struct Frame {
  double elapsed_s{}, prp_gpu_ms{};
  std::uint64_t loops{}, positions{}, total_passes{}, factor_begin{}, factor_next{}, pending{};
  std::uint32_t last_passes{};
  bool fixed_entries{};
  std::uint64_t entries{};
};
struct Sources {
  const unsigned long long *complete{}, *entries{}, *active{}, *tested{}, *passed{};
  const std::uint64_t* factors{};
  std::uint64_t capacity{};
};
class Collector {
  static constexpr std::size_t slots_count = 4, factor_chunk = 256;
  struct Slot { Counters* counters{}; Frame frame{}; cudaEvent_t ready{}; bool busy{}; };
  std::array<Slot, slots_count> slots_{};
  std::array<std::size_t, slots_count> queue_{};
  std::uint64_t head_{}, tail_{}, drained_{}, snapshots_{}, coalesced_{};
  Sources sources_;
  std::function<void(std::uint64_t)> admit_;
  const std::atomic<std::uint64_t>& journaled_;
  std::uint64_t* scratch_{};
  cudaStream_t copy_stream_{};
  int device_{};
  std::mutex mutex_;
  std::condition_variable wake_, space_;
  std::thread worker_;
  bool stop_{};
  std::exception_ptr failure_;

  void cleanup() noexcept {
    for (auto& s : slots_) {
      // An enqueue failure may leave a copy in flight before publication.
      if (s.busy && s.ready) cudaEventSynchronize(s.ready);
      if (s.ready) cudaEventDestroy(s.ready);
      if (s.counters) cudaFreeHost(s.counters);
    }
    if (scratch_) cudaFreeHost(scratch_);
    if (copy_stream_) cudaStreamDestroy(copy_stream_);
  }
  void consume(Slot& s) {
    checked(cudaEventSynchronize(s.ready), "sidecar snapshot ready");
    const auto& c = *s.counters;
    if (c.complete < drained_ || c.complete > sources_.capacity)
      throw std::runtime_error("sidecar immutable result frontier invalid");
    while (drained_ < c.complete) {
      const std::size_t count = static_cast<std::size_t>(
          (std::min)(c.complete - drained_, static_cast<unsigned long long>(factor_chunk)));
      checked(cudaMemcpyAsync(scratch_, sources_.factors + drained_, count * sizeof(*scratch_),
                              cudaMemcpyDeviceToHost, copy_stream_), "sidecar factor prefix copy");
      checked(cudaStreamSynchronize(copy_stream_), "sidecar factor copy ready");
      for (std::size_t i = 0; i < count; ++i) admit_(scratch_[i]);
      drained_ += count;
    }
    const auto& f = s.frame;
    std::ostringstream out;
    out << "phase=q0_measure_progress elapsed_s=" << f.elapsed_s
        << " loops=" << f.loops << " positions=" << f.positions << " q0_entries=" << c.entries
        << " q0_candidates_per_s=" << (f.elapsed_s > 0 ? c.entries / f.elapsed_s : 0)
        << " q0_positions_per_s=" << (f.elapsed_s > 0 ? f.positions / f.elapsed_s : 0)
        << " q0_prp_per_s=" << (f.prp_gpu_ms > 0 ? 1000.0 * c.entries / f.prp_gpu_ms : 0)
        << " q0_survivors_per_loop=" << (f.loops ? c.entries / f.loops : 0)
        << " q0_passes_last_loop=" << f.last_passes << " q0_passes_total=" << f.total_passes
        << " tuple_counts=" << c.entries << ',' << f.total_passes;
    for (auto n : c.active) out << ',' << n;
    out << " exact_results_journaled=" << journaled_.load()
        << " factor_origin_begin=" << f.factor_begin << " factor_origin_next=" << f.factor_next
        << " tuple_operator=mandatory_q1_then_queued_optional5 tuple_pending=" << f.pending
        << " member_tested=";
    for (std::size_t i = 0; i < 6; ++i) out << (i ? "," : "") << c.tested[i];
    out << " member_prp_passed=";
    for (std::size_t i = 0; i < 6; ++i) out << (i ? "," : "") << c.passed[i];
    out << " result_sidecar=1 snapshot_scope=immutable_prefix";
    std::cout << out.str() << std::endl;
    ++snapshots_;
  }
  void run() noexcept {
    try {
      checked(cudaSetDevice(device_), "sidecar thread device");
      for (;;) {
        std::size_t index;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          wake_.wait(lock, [&] { return stop_ || head_ != tail_; });
          if (head_ == tail_) return;
          index = queue_[head_++ % slots_count];
        }
        consume(slots_[index]);
        { std::lock_guard<std::mutex> lock(mutex_); slots_[index].busy = false; }
        space_.notify_one();
      }
    } catch (...) {
      { std::lock_guard<std::mutex> lock(mutex_); failure_ = std::current_exception(); }
      space_.notify_all();
    }
  }
public:
  Collector(Sources sources, std::function<void(std::uint64_t)> admit,
            const std::atomic<std::uint64_t>& journaled)
      : sources_(sources), admit_(std::move(admit)), journaled_(journaled) {
    try {
      checked(cudaGetDevice(&device_), "sidecar device identity");
      checked(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking), "sidecar copy stream");
      checked(cudaMallocHost(reinterpret_cast<void**>(&scratch_), factor_chunk * sizeof(*scratch_)),
              "sidecar pinned factor buffer");
      for (auto& s : slots_) {
        checked(cudaMallocHost(reinterpret_cast<void**>(&s.counters), sizeof(Counters)), "sidecar pinned counters");
        checked(cudaEventCreateWithFlags(&s.ready, cudaEventDisableTiming), "sidecar event");
      }
      worker_ = std::thread([this] { run(); });
    } catch (...) { cleanup(); throw; }
  }
  Collector(const Collector&) = delete;
  Collector& operator=(const Collector&) = delete;
  ~Collector() { try { finish(); } catch (...) {} cleanup(); }
  bool submit(Frame frame, bool mandatory = false) {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto free_slot = [&] {
      for (std::size_t i = 0; i < slots_count; ++i) if (!slots_[i].busy) return i;
      return slots_count;
    };
    if (mandatory) space_.wait(lock, [&] { return failure_ || free_slot() != slots_count; });
    if (failure_) std::rethrow_exception(failure_);
    if (stop_) throw std::runtime_error("sidecar submit after final drain");
    const std::size_t index = free_slot();
    if (index == slots_count) { ++coalesced_; return false; }
    auto& s = slots_[index];
    s.busy = true;
    s.frame = frame;
    // All source-counter copies are ordered on the compute thread's stream.
    // That stream may proceed immediately afterward. Host storage stays pinned
    // and owned until the worker consumes the completion event.
    try {
      auto copy = [&](void* to, const void* from, std::size_t bytes) {
        checked(cudaMemcpyAsync(to, from, bytes, cudaMemcpyDeviceToHost, cudaStreamPerThread),
                "sidecar snapshot enqueue");
      };
      copy(&s.counters->complete, sources_.complete, sizeof(s.counters->complete));
      if (frame.fixed_entries) s.counters->entries = frame.entries;
      else copy(&s.counters->entries, sources_.entries, sizeof(s.counters->entries));
      copy(s.counters->active.data(), sources_.active, sizeof(s.counters->active));
      copy(s.counters->tested.data(), sources_.tested, sizeof(s.counters->tested));
      copy(s.counters->passed.data(), sources_.passed, sizeof(s.counters->passed));
      checked(cudaEventRecord(s.ready, cudaStreamPerThread), "sidecar snapshot commit");
    } catch (...) {
      // Failure-only cleanup: never free pinned buffers with pending DMA.
      cudaStreamSynchronize(cudaStreamPerThread);
      s.busy = false;
      throw;
    }
    queue_[tail_++ % slots_count] = index;
    lock.unlock();
    wake_.notify_one();
    return true;
  }
  void finish() {
    if (!worker_.joinable()) return;
    { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
    wake_.notify_all();
    worker_.join();
    if (failure_) std::rethrow_exception(failure_);
    if (head_ != tail_) throw std::runtime_error("sidecar pending snapshots at drain");
  }
  std::uint64_t drained() const { return drained_; } // Only after finish().
  std::uint64_t snapshots() const { return snapshots_; }
  std::uint64_t coalesced() const { return coalesced_; }
};
} // namespace riecoin_result_sidecar

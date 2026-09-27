#pragma once
// A14: owns Q1/optional-member CUDA work on a distinct host thread/PTDS.
// Producer grants immutable factor-buffer ownership through an event. Buffers
// return only after the consumer's stream completed their last read. No proof
// or counter from a later factor frontier can enter an earlier receipt.
namespace riecoin_tail_rail {
class Rail {
  using Frame = riecoin_result_sidecar::Frame;
  static constexpr std::size_t count_slots = 4;
  struct Slot {
    std::uint64_t* factors{};
    unsigned long long* entries{};
    cudaEvent_t ready{};
    Frame frame{};
    std::uint64_t begin{}, end{};
    std::uint32_t count{};
    bool busy{}, publish{};
#ifdef RIECOIN_RLOW_ASYNC_FRONTIER
    std::array<cudaEvent_t,7> phases{};
    std::array<cudaEvent_t,6> producer_phases{};
    std::uint32_t* receipt{}; // pass count, producer overflow, invalid prime
#endif
  };
  std::array<Slot, count_slots> slots_{};
  std::array<std::size_t, count_slots> queue_{};
  std::size_t head_{}, tail_{};
  std::uint32_t capacity_{};
  const unsigned long long* entries_{};
  riecoin_result_sidecar::Collector& collector_;
  std::chrono::steady_clock::time_point started_;
  std::mutex mutex_;
  std::condition_variable work_, space_;
  std::thread worker_;
  std::exception_ptr error_;
  bool stop_{};
  int device_{};
  std::uint64_t admitted_{}, consumed_{}, max_pending_{};
  double enqueue_wait_ms_{};
#ifdef RIECOIN_RLOW_ASYNC_FRONTIER
  std::size_t reserved_ = count_slots;
  cudaEvent_t epoch_{};
  std::array<cudaEvent_t,6> original_producer_events_{};
  std::array<double,6> phase_ms_{};
  std::uint64_t passes_{};
#endif

  void clean() noexcept {
    for (auto& s : slots_) {
      if (s.busy && s.ready) cudaEventSynchronize(s.ready);
      if (s.ready) cudaEventDestroy(s.ready);
      if (s.entries) cudaFreeHost(s.entries);
      if (s.factors) cudaFree(s.factors);
#ifdef RIECOIN_RLOW_ASYNC_FRONTIER
      if(s.receipt) cudaFreeHost(s.receipt);
      for(auto e:s.phases) if(e) cudaEventDestroy(e);
      for(auto e:s.producer_phases) if(e) cudaEventDestroy(e);
#endif
    }
#ifdef RIECOIN_RLOW_ASYNC_FRONTIER
    if(epoch_) cudaEventDestroy(epoch_);
#endif
  }
  void run() noexcept {
    try {
      cuda_check(cudaSetDevice(device_), "tail rail thread device");
      Frame last{};
      bool have_last = false;
      for (;;) {
        std::size_t index;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          work_.wait(lock, [&]{ return stop_ || head_ != tail_; });
          if (head_ == tail_) break;
          index = queue_[head_++ % count_slots];
        }
        auto& s = slots_[index];
        cuda_check(cudaEventSynchronize(s.ready), "tail rail immutable frame ready");
#ifdef RIECOIN_RLOW_ASYNC_FRONTIER
        if(s.receipt[0]>capacity_ || s.receipt[1] || s.receipt[2])
          throw std::runtime_error("async frontier guard failure: no frame admitted");
        s.count=s.receipt[0]; passes_+=s.count;
        s.frame.total_passes=passes_; s.frame.last_passes=s.count;
        float delta=0;
        for(std::size_t i=0;i<6;++i) {
          cuda_check(cudaEventElapsedTime(&delta,s.phases[i],s.phases[i+1]),"async Q0 phase");
          phase_ms_[i]+=delta;
        }
        s.frame.prp_gpu_ms=phase_ms_[3];
        cuda_check(cudaEventElapsedTime(&delta,epoch_,s.ready),"async completed frontier clock");
        s.frame.elapsed_s=delta/1000.0;
        double* producer_times[]={&rlow_batch_producer::remainder_gpu_ms,&rlow_queue::sieve_gpu_ms,
          &rlow_queue::scan_gpu_ms,&rlow_queue::pack_gpu_ms,&rlow_queue::metadata_gpu_ms};
        for(std::size_t i=0;i<5;++i) {
          cuda_check(cudaEventElapsedTime(&delta,s.producer_phases[i],s.producer_phases[i+1]),"async producer phase");
          *producer_times[i]+=delta;
        }
#endif
        rlow_tail_host::flush(false);
        rlow_tail_host::append(s.factors, s.count, s.begin, s.end);
        // append may defer input copies: readiness is a buffer-lifetime
        // condition, not a wait on the main search thread/stream.
        cuda_check(cudaStreamSynchronize(cudaStreamPerThread), "tail rail frame consumed");
        last = s.frame;
        last.fixed_entries = true;
        last.entries = *s.entries;
        last.pending = rlow_tail_host::pending_total();
        have_last = true;
        if (s.publish) collector_.submit(last);
        {
          std::lock_guard<std::mutex> lock(mutex_);
          s.busy = false;
          ++consumed_;
        }
        space_.notify_one();
      }
      if (!have_last) throw std::runtime_error("tail rail empty claimed window");
      rlow_tail_host::finish(last.factor_next, last.total_passes);
      last.elapsed_s = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - started_).count();
      last.pending = 0;
      collector_.submit(last, true);
    } catch (...) {
      // Ensure outstanding consumer reads are complete even on failure before
      // root unwinds and reclaims immutable device slots.
      cudaStreamSynchronize(cudaStreamPerThread);
      { std::lock_guard<std::mutex> lock(mutex_); error_ = std::current_exception(); }
      space_.notify_all();
    }
  }
public:
  Rail(std::uint32_t capacity, const unsigned long long* entries,
       riecoin_result_sidecar::Collector& collector,
       std::chrono::steady_clock::time_point started)
      : capacity_(capacity), entries_(entries), collector_(collector), started_(started) {
    try {
      cuda_check(cudaGetDevice(&device_), "tail rail producer device");
#ifdef RIECOIN_RLOW_ASYNC_FRONTIER
      cuda_check(cudaEventCreate(&epoch_),"async epoch event");
      cuda_check(cudaEventRecord(epoch_),"async epoch begin");
      original_producer_events_=rlow_batch_producer::events;
#endif
      for (auto& s : slots_) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&s.factors),
            static_cast<std::size_t>(capacity) * sizeof(std::uint64_t)), "tail rail factor slot");
        cuda_check(cudaMallocHost(reinterpret_cast<void**>(&s.entries), sizeof(*s.entries)),
            "tail rail entry receipt");
#ifdef RIECOIN_RLOW_ASYNC_FRONTIER
#ifdef RIECOIN_RLOW_BLOCKING_READY
        // Same device event, timing and immutable-frame ownership. Only the
        // host wait sleeps instead of burning a CPU while Q0 is running.
        cuda_check(cudaEventCreateWithFlags(&s.ready,cudaEventBlockingSync), "tail rail blocking ready event");
#else
        cuda_check(cudaEventCreate(&s.ready), "tail rail ready event");
#endif
        cuda_check(cudaMallocHost(reinterpret_cast<void**>(&s.receipt),3*sizeof(std::uint32_t)),"async pinned guard");
        for(auto& e:s.phases) cuda_check(cudaEventCreate(&e),"async Q0 event slot");
        for(auto& e:s.producer_phases) cuda_check(cudaEventCreate(&e),"async producer event slot");
#else
        cuda_check(cudaEventCreateWithFlags(&s.ready, cudaEventDisableTiming), "tail rail ready event");
#endif
      }
      worker_ = std::thread([this]{ run(); });
    } catch (...) { clean(); throw; }
  }
  Rail(const Rail&) = delete;
#ifdef RIECOIN_RLOW_ASYNC_FRONTIER
  std::array<cudaEvent_t,7>& acquire() {
    const auto begin=std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mutex_);
    auto free_slot=[&]{for(std::size_t i=0;i<count_slots;++i)if(!slots_[i].busy)return i;return count_slots;};
    space_.wait(lock,[&]{return error_ || free_slot()!=count_slots;});
    enqueue_wait_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    if(error_)std::rethrow_exception(error_);
    if(reserved_!=count_slots || stop_)throw std::runtime_error("async slot ownership violation");
    reserved_=free_slot(); slots_[reserved_].busy=true;
    rlow_batch_producer::events=slots_[reserved_].producer_phases;
    return slots_[reserved_].phases;
  }
  void submit_async(const std::uint64_t* factors,const std::uint32_t* count,
                    std::uint64_t begin,std::uint64_t end,Frame frame,bool publish) {
    if(reserved_==count_slots || begin>=end)throw std::runtime_error("async unowned submission");
    auto& s=slots_[reserved_];s.frame=frame;s.begin=begin;s.end=end;s.publish=publish;
    cuda_check(cudaMemcpyAsync(s.factors,factors,static_cast<std::size_t>(capacity_)*sizeof(*factors),cudaMemcpyDeviceToDevice,cudaStreamPerThread),"async immutable factors");
    cuda_check(cudaMemcpyAsync(s.receipt,count,sizeof(*count),cudaMemcpyDeviceToHost,cudaStreamPerThread),"async pass receipt");
    cuda_check(cudaMemcpyAsync(s.receipt+1,rlow_batch_producer::terminal_status,2*sizeof(*count),cudaMemcpyDeviceToHost,cudaStreamPerThread),"async producer guard receipt");
    cuda_check(cudaMemcpyAsync(s.entries,entries_,sizeof(*s.entries),cudaMemcpyDeviceToHost,cudaStreamPerThread),"async entry receipt");
    cuda_check(cudaEventRecord(s.ready,cudaStreamPerThread),"async frame publish");
    {std::lock_guard<std::mutex> lock(mutex_);queue_[tail_++%count_slots]=reserved_;reserved_=count_slots;++admitted_;max_pending_=(std::max)(max_pending_,admitted_-consumed_);}
    work_.notify_one();
  }
  std::uint64_t total_passes()const{return passes_;}
  const std::array<double,6>& timings()const{return phase_ms_;}
#endif
  Rail& operator=(const Rail&) = delete;
  ~Rail() {
#ifdef RIECOIN_RLOW_ASYNC_FRONTIER
    if(reserved_!=count_slots) cudaStreamSynchronize(cudaStreamPerThread);
#endif
    try { finish(); } catch (...) {} clean();
  }
  void submit(const std::uint64_t* factors, std::uint32_t count,
              std::uint64_t begin, std::uint64_t end, Frame frame, bool publish) {
    if (count > capacity_ || begin >= end) throw std::runtime_error("tail rail invalid frame");
    const auto waiting = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mutex_);
    auto available = [&] {
      for (std::size_t i=0; i<count_slots; ++i) if (!slots_[i].busy) return i;
      return count_slots;
    };
    space_.wait(lock, [&]{ return error_ || available() != count_slots; });
    enqueue_wait_ms_ += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - waiting).count();
    if (error_) std::rethrow_exception(error_);
    if (stop_) throw std::runtime_error("tail rail append after stop");
    const auto index = available();
    auto& s = slots_[index];
    s.busy = true; s.frame=frame; s.begin=begin; s.end=end; s.count=count; s.publish=publish;
    try {
      if (count) cuda_check(cudaMemcpyAsync(s.factors, factors,
          static_cast<std::size_t>(count)*sizeof(std::uint64_t), cudaMemcpyDeviceToDevice,
          cudaStreamPerThread), "tail rail immutable frame copy");
      cuda_check(cudaMemcpyAsync(s.entries, entries_, sizeof(*s.entries),
          cudaMemcpyDeviceToHost, cudaStreamPerThread), "tail rail frontier receipt");
      cuda_check(cudaEventRecord(s.ready, cudaStreamPerThread), "tail rail frame publish");
    } catch (...) {
      cudaStreamSynchronize(cudaStreamPerThread); s.busy=false; throw;
    }
    queue_[tail_++ % count_slots]=index;
    ++admitted_;
    max_pending_=(std::max)(max_pending_, admitted_-consumed_);
    lock.unlock();
    work_.notify_one();
  }
  void finish() {
    if (!worker_.joinable()) return;
    { std::lock_guard<std::mutex> lock(mutex_); stop_=true; }
    work_.notify_one();
    worker_.join();
#ifdef RIECOIN_RLOW_ASYNC_FRONTIER
    rlow_batch_producer::events=original_producer_events_;
#endif
    if (error_) std::rethrow_exception(error_);
    if (admitted_ != consumed_ || head_ != tail_)
      throw std::runtime_error("tail rail frame conservation failure");
    std::cout << "phase=cuda_tail_rail_drained admitted=" << admitted_
              << " consumed=" << consumed_ << " max_pending=" << max_pending_
              << " enqueue_wait_ms=" << enqueue_wait_ms_ << " errors=0 pending=0" << std::endl;
  }
};
} // namespace riecoin_tail_rail

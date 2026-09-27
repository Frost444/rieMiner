#pragma once
// Included inside rlow_batch_producer. Only cold frontier construction moves;
// candidate packing, dense worklists and all admission semantics stay owned by
// the calling thread. Its handshake forbids concurrent host-global mutation.
struct R259Bank {
  uint32_t *bits=nullptr, *occupied=nullptr, *status=nullptr, *reference=nullptr;
  uint64_t origin=0;
  uint32_t groups=0;
  const uint32_t *primes=nullptr, *roots=nullptr;
  cudaEvent_t last_use=nullptr, begin=nullptr, ready=nullptr;
  bool used=false;
  void capture() {
    bits=frontier_bits; occupied=frontier_occupied; status=frontier_status;
    reference=frontier_reference; origin=frontier_origin;
    groups=frontier_cached_groups; primes=frontier_primes; roots=frontier_roots;
  }
  void bind() const {
    frontier_bits=bits; frontier_occupied=occupied; frontier_status=status;
    frontier_reference=reference; frontier_origin=origin;
    frontier_cached_groups=groups; frontier_primes=primes; frontier_roots=roots;
  }
  bool contains(uint64_t o, uint64_t span, const uint32_t* p, const uint32_t* r) const {
    return groups && p==primes && r==roots && o>=origin &&
      o-origin<span*groups && (o-origin)%span==0;
  }
};
static R259Bank r259_banks[2];
static bool r259_enabled=false, r259_initialized=false, r259_pending=false;
static uint64_t r259_limit=0, r259_built=0, r259_adopted=0;
static unsigned r259_active=0;
static int r259_device=0;
static double r259_gpu_ms=0, r259_wait_ms=0;
static unsigned r259_audited=0;
static std::function<void(uint64_t)> r260_deferred_builder;
static uint64_t r260_deferred_span=0;
static cudaStream_t r266_producer=nullptr,r266_build_stream=cudaStreamPerThread;
static bool r266_recycle_checked=false;
// R270 changes only the host submission granularity. A future bank is still
// published atomically; partial preparation never changes admission semantics.
static bool r270_quantized=false;
static unsigned r273_style=1; // 1=R270 dense slices, 2=R272 shared phases.
static unsigned r270_build_slice=UINT32_MAX, r270_next_slice=0;
static uint64_t r270_next_origin=0, r270_quanta=0;
static std::function<void(uint64_t)> r270_builder;

inline void r259_audit(uint64_t origin) {
  const char* path=std::getenv("ASTRA_R259_BITMAP_AUDIT");
  if(!path || !*path || origin!=frontier_origin || r259_audited>=3) return;
  const size_t words=static_cast<size_t>(configured_words)*kMacros*frontier_cached_groups;
  if(words>16777216u)throw std::runtime_error("R259 audit byte bound");
  std::vector<uint32_t> exact(words);
  cuda_check(cudaMemcpy(exact.data(),frontier_bits,words*4u,cudaMemcpyDeviceToHost),"R259 complete bank audit");
  const std::string file=std::string(path)+"\\bank-"+std::to_string(origin)+".bin";
  std::ofstream out(file,std::ios::binary|std::ios::out);
  if(!out || !out.write(reinterpret_cast<const char*>(exact.data()),words*4u))
    throw std::runtime_error("R259 audit write failed");
  out.close(); if(!out)throw std::runtime_error("R259 audit close failed");
  ++r259_audited;
}

inline void r259_configure(uint64_t limit) {
  const char* flag=std::getenv("ASTRA_R259_PREFETCH");
  if (flag && std::strcmp(flag,"0") && std::strcmp(flag,"1"))
    throw std::runtime_error("R259 invalid prefetch flag");
  r259_enabled=flag && !std::strcmp(flag,"1"); r259_limit=limit;
  const char* quantum=std::getenv("ASTRA_R270_QUANTUM");
  if(quantum && std::strcmp(quantum,"0") && std::strcmp(quantum,"1"))
    throw std::runtime_error("R270 invalid quantum flag");
  r270_quantized=quantum && !std::strcmp(quantum,"1");
  if(r270_quantized && !r259_enabled)
    throw std::runtime_error("R270 quantization requires prefetch");
  if (r259_enabled && frontier_groups!=8u)
    throw std::runtime_error("R259 qualification requires eight-slice frontier");
  cuda_check(cudaGetDevice(&r259_device),"R259 producer device");
  std::cout<<"phase=r259_prefetch enabled="<<r259_enabled<<" limit="<<limit
    <<" candidate_pipeline_unchanged=1 telemetry_scope=end_to_end\n";
}
inline void r259_collect(unsigned bank) {
  if(r270_quantized && r270_next_slice!=8u)
    throw std::runtime_error("R270 incomplete bank cannot be collected");
  const auto start=std::chrono::steady_clock::now();
  cuda_check(cudaEventSynchronize(r259_banks[bank].ready),"R259 bank ready");
  r259_wait_ms+=std::chrono::duration<double,std::milli>(
    std::chrono::steady_clock::now()-start).count();
  float ms=0;
  cuda_check(cudaEventElapsedTime(&ms,r259_banks[bank].begin,r259_banks[bank].ready),
    "R259 producer elapsed (overlaps consumer; not additive)");
  r259_gpu_ms+=ms;
}
inline void r259_resolve(uint64_t origin,uint64_t span,const uint32_t* primes,const uint32_t* roots) {
  if (!r259_enabled || !r259_initialized) return;
  if (r259_banks[r259_active].contains(origin,span,primes,roots)) return;
  const unsigned next=1u-r259_active;
  // The inherited calendar itself requires complete contiguous frontiers.
  // A wrong origin or binding must fail, never consume an unrelated bank.
  if (!r259_pending || !r259_banks[next].contains(origin,span,primes,roots) ||
      origin!=r259_banks[next].origin)
    throw std::runtime_error("R259 frontier generation/binding mismatch");
  r259_collect(next);
  r259_banks[next].bind(); r259_active=next; r259_pending=false; ++r259_adopted;
}
inline void r259_finish(std::function<void(uint64_t)> build,uint64_t span) {
  if (!r259_enabled || !build) return;
  if (!r259_initialized) {
    r259_banks[0].capture(); r259_initialized=true;
    cuda_check(cudaStreamCreateWithFlags(&r266_producer,cudaStreamNonBlocking),"R266 producer stream");
    const size_t words=static_cast<size_t>(configured_words)*kMacros*frontier_groups;
    auto& extra=r259_banks[1];
    cuda_check(cudaMalloc(&extra.bits,words*4u),"R259 second bitmap bank");
    cuda_check(cudaMalloc(&extra.occupied,((words+31u)/32u)*4u),"R259 second occupancy");
    cuda_check(cudaMalloc(&extra.status,4u),"R259 second sticky guard");
    if (frontier_audit) cuda_check(cudaMalloc(&extra.reference,words*4u),"R259 second audit shadow");
    for (auto& bank:r259_banks) {
      cuda_check(cudaEventCreateWithFlags(&bank.last_use,cudaEventDisableTiming),"R259 bank lifetime");
      cuda_check(cudaEventCreate(&bank.begin),"R259 producer begin");
      cuda_check(cudaEventCreate(&bank.ready),"R259 producer ready");
    }
  }
  auto& current=r259_banks[r259_active]; current.capture();
  cuda_check(cudaEventRecord(current.last_use),"R259 last bitmap read"); current.used=true;
  // Close recycled-bank ownership BEFORE Q0 is queued. A host-side dependency
  // check after Q0 can flush its WDDM command batch and defeat concurrency.
  r266_recycle_checked=false;
  if(!r259_pending){
    auto& target=r259_banks[1u-r259_active];
    if(target.used)cuda_check(cudaEventSynchronize(target.last_use),"R266 recycled lifetime before PRP");
    r266_recycle_checked=true;
  }
  r260_deferred_builder=std::move(build); r260_deferred_span=span;
}
inline void r270_submit_quantum() {
  if(!r270_builder || r270_next_slice>=8u || !r259_pending)
    throw std::runtime_error("R270 invalid quantum ownership");
  const R259Bank saved=r259_banks[r259_active];
  auto& bank=r259_banks[1u-r259_active];
  bank.bind();r266_build_stream=r266_producer;r270_build_slice=r270_next_slice;
  try {
    if(r270_next_slice==0u)
      cuda_check(cudaEventRecord(bank.begin,r266_producer),"R270 future begin");
    r270_builder(r270_next_origin);
    cuda_check(cudaGetLastError(),"R270 partial future kernels");
    bank.capture();
    ++r270_next_slice;++r270_quanta;
    if(r270_next_slice==8u){
      cuda_check(cudaEventRecord(bank.ready,r266_producer),"R270 complete future ready");
      const char* inject=std::getenv("ASTRA_R259_INJECT_GENERATION");
      if(inject && !std::strcmp(inject,"1") && r259_built==0)
        bank.origin+=r260_deferred_span;
      ++r259_built;r270_builder={};
    }
  } catch (...) {
    r270_build_slice=UINT32_MAX;r266_build_stream=cudaStreamPerThread;
    saved.bind();throw;
  }
  r270_build_slice=UINT32_MAX;r266_build_stream=cudaStreamPerThread;saved.bind();
}
// R260 changes only the submission boundary. Record the bank's last-read event
// inside fill; submit future production AFTER current PRP is already queued.
inline void r260_submit_after_prp() {
  if(!r259_enabled || !r260_deferred_builder) return;
  const auto build=std::move(r260_deferred_builder);
  const uint64_t span=r260_deferred_span;
  auto& current=r259_banks[r259_active];
  if (r259_pending) {
    if(r270_quantized && r270_next_slice<8u)r270_submit_quantum();
    return;
  }
  const uint64_t extent=span*current.groups;
  if (current.origin>UINT64_MAX-extent) throw std::runtime_error("R259 next bank overflow");
  const uint64_t next_origin=current.origin+extent;
  if (next_origin>=r259_limit) return; // Do not generate unrequested future banks.
  const unsigned target=1u-r259_active;
  const R259Bank saved=current;
  if(r270_quantized){
    if(!r266_recycle_checked||!r266_producer)
      throw std::runtime_error("R270 recycled bank not qualified");
    r270_next_origin=next_origin;r270_next_slice=0;r270_builder=build;
    r259_pending=true;r270_submit_quantum();return;
  }
  {
    if(!r266_recycle_checked||!r266_producer)throw std::runtime_error("R266 recycled bank not qualified");
    // Submit both streams from the same host thread. Scratch/calendar belong
    // to the producer stream; current Q0 only reads its already packed inputs.
    auto& bank=r259_banks[target];
    bank.bind();
    r266_build_stream=r266_producer;
    try {
      cuda_check(cudaEventRecord(bank.begin,r266_producer),"R259 future sieve begin");
      build(next_origin);
      cuda_check(cudaGetLastError(),"R259 future kernels");
      bank.capture();
      cuda_check(cudaEventRecord(bank.ready,r266_producer),"R259 future sieve ready");
    } catch (...) {r266_build_stream=cudaStreamPerThread;saved.bind();throw;}
    r266_build_stream=cudaStreamPerThread;saved.bind();
  }
  const char* inject=std::getenv("ASTRA_R259_INJECT_GENERATION");
  if(inject && !std::strcmp(inject,"1") && r259_built==0)
    r259_banks[target].origin+=span; // Closed gate: must reject, never silently rebind.
  r259_pending=true; ++r259_built;
}
inline void r259_shutdown() {
  if (!r259_initialized) return;
  std::exception_ptr error;
  try {
    if (r259_pending && (!r270_quantized || r270_next_slice==8u))
      r259_collect(1u-r259_active);
    cuda_check(cudaStreamSynchronize(r266_producer),"R266 producer drain");
  } catch (...) { error=std::current_exception(); }
  std::cout<<"phase=r259_complete prefetched="<<r259_built<<" adopted="<<r259_adopted
    <<" unused="<<(r259_built-r259_adopted)<<" producer_gpu_ms="<<r259_gpu_ms
    <<" ready_wait_ms="<<r259_wait_ms<<" producer_time_not_additive=1 host_submitters=1"
    <<" quantum="<<r270_quantized<<" submitted_quanta="<<r270_quanta
    <<" partial_discarded="<<(r270_quantized&&r259_pending&&r270_next_slice<8u?1:0)<<"\n";
  // Restore the original allocation for inherited cleanup; release only extra.
  r259_banks[0].bind();
  auto& extra=r259_banks[1];
  const auto release=[&](cudaError_t result,const char* label){try{cuda_check(result,label);}catch(...){if(!error)error=std::current_exception();}};
  for (auto ptr:{extra.bits,extra.occupied,extra.status,extra.reference}) if(ptr) release(cudaFree(ptr),"R266 extra bank release");
  for (auto& bank:r259_banks) for (auto event:{bank.last_use,bank.begin,bank.ready})
    if(event) release(cudaEventDestroy(event),"R266 event release");
  if(r266_producer)release(cudaStreamDestroy(r266_producer),"R266 stream release");
  r266_producer=nullptr;r266_build_stream=cudaStreamPerThread;r266_recycle_checked=false;
  r270_builder={};r260_deferred_builder={};r270_build_slice=UINT32_MAX;
  r259_initialized=false; r259_pending=false;
  if(error) std::rethrow_exception(error);
}

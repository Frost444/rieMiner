#pragma once

// Include AFTER private_macro + queue_host. No second CGBN include.
// Integration: setup(words) after rlow_queue::setup(words); replace only fill;
// cleanup() before wall_end. Counts/segments/cold GMP gate remain queue-owned.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <functional>
#include "r174_phase_recurrence.cuh"
#ifdef RIECOIN_RLOW_COMPRESSED_ROOTS
#include "riecoin_compressed_root_storage.cuh"
#endif
#ifdef RIECOIN_RLOW_SPARSE_EVENT_SIEVE
#include "riecoin_rlow_sparse_event_sieve.cuh"
#endif

#if defined(__CUDACC__)
#define RLOW_BATCH_HD __host__ __device__
#else
#define RLOW_BATCH_HD
#endif

namespace rlow_batch_producer {
#ifdef RIECOIN_RLOW_QUEUE_MACROS
constexpr uint32_t kMacros = RIECOIN_RLOW_QUEUE_MACROS;
#else
constexpr uint32_t kMacros = 16u;
#endif
constexpr uint32_t kMacroCapacity = 4096u;
constexpr uint32_t kCapacity = kMacros * kMacroCapacity;
static_assert(kMacros >= 16u && kMacros <= 64u && kMacros % 16u == 0u,
              "unsupported batch macro geometry");
constexpr uint32_t kMaximumPositions = 8u * 65536u;

inline void validate_geometry(uint32_t words, uint32_t positions, uint64_t origin) {
  if (positions == 0u || positions > kMaximumPositions ||
      words != (positions + 31u) / 32u)
    throw std::runtime_error("RLOW batch unsupported macro geometry");
  const uint64_t span = static_cast<uint64_t>(kMacros) * positions;
  // The caller advances the base by this full half-open reservation.
  if (origin > UINT64_MAX - span || span - 1u > UINT32_MAX)
    throw std::runtime_error("RLOW batch factor reservation overflow");
}

RLOW_BATCH_HD inline size_t word_index(uint32_t macro, uint32_t word, uint32_t words) {
  return static_cast<size_t>(macro) * words + word;
}

// Inputs are reduced modulo p. The widened add handles p near 2^32 exactly.
RLOW_BATCH_HD inline uint32_t advance_remainder(uint32_t current, uint32_t step,
                                               uint32_t p) {
  const uint64_t sum = static_cast<uint64_t>(current) + step;
  return static_cast<uint32_t>(sum >= p ? sum - p : sum);
}

// This mapping is shared by CUDA packing and the CPU oracle tests. Arithmetic
// construction itself deliberately remains the existing bridge pack atom.
RLOW_BATCH_HD inline bool map_output(uint64_t origin, uint32_t positions,
    uint32_t macro, uint32_t word, uint32_t bit, uint32_t rank,
    uint32_t& slot, uint64_t& relative, uint64_t& factor) {
  if (macro >= kMacros || rank >= kMacroCapacity || bit >= 32u) return false;
  const uint64_t local = static_cast<uint64_t>(word) * 32u + bit;
  if (local >= positions) return false;
  relative = static_cast<uint64_t>(macro) * positions + local;
  if (relative > UINT32_MAX || origin > UINT64_MAX - relative) return false;
  slot = macro * kMacroCapacity + rank;
  factor = origin + relative;
  return true;
}

#if defined(__CUDACC__)
static_assert(kMacros == rlow_queue::kMacros &&
              kMacroCapacity == rlow_queue::kMacroCapacity,
              "RLOW batch/consumer segment geometry mismatch");
static_assert(riecoin_rlow_bridge::kBlockThreads == 256u, "scan block mismatch");

uint32_t* bitmap_words = nullptr;
uint32_t* private_planes = nullptr;
uint32_t* word_prefix = nullptr;
uint32_t* chunk_counts = nullptr;
uint32_t* chunk_offsets = nullptr;
uint32_t* origin_remainders = nullptr;
uint8_t* overflow_flags = nullptr;
uint32_t* terminal_status = nullptr;  // [pack overflow count, invalid prime]
uint32_t configured_words = 0u, configured_chunks = 0u;
uint32_t maximum_planes = 0u, allocated_primes = 0u;
std::array<cudaEvent_t, 6u> events{};
double remainder_gpu_ms = 0.0;
double cold_reference_ms = 0.0;
bool cold_reference_checked = false;

// A93: future dense bitmaps are the final bitmaps, not extra sparse masks.
uint32_t frontier_groups = 8u, frontier_cached_groups = 0u;
uint32_t *frontier_bits = nullptr, *frontier_occupied = nullptr, *frontier_status = nullptr;
uint32_t* frontier_reference = nullptr;
bool frontier_audit = false;
uint32_t frontier_cutover = 33554432u, frontier_middle = UINT32_MAX;
bool frontier_diagnostic = false, frontier_diagnosed = false;
uint32_t* frontier_nonempty = nullptr;
std::array<cudaEvent_t,4u> frontier_diag_events{};
uint64_t frontier_origin = 0u;
const uint32_t *frontier_primes = nullptr, *frontier_roots = nullptr;
// R178: isolated full-miner integration. Immutable roots remain the oracle.
uint32_t* r178_phases = nullptr;
uint32_t r178_capacity = 0u, r178_begin = 0u, r178_count = 0u, r178_span = 0u;
uint64_t r178_expected = 0u, r178_initializations = 0u, r178_continuations = 0u, r178_fallbacks = 0u;
const uint32_t *r178_primes_binding = nullptr, *r178_roots_binding = nullptr;
bool r178_enabled = false, r178_valid = false;

#include "r259_prefetch.cuh"

// Resource cleanup on both normal completion and partial setup failure.
inline void cleanup() {
  r259_shutdown();
  cudaError_t first = cudaSuccess;
  auto release = [&first](auto*& pointer) {
    if (pointer != nullptr) {
      const cudaError_t result = cudaFree(pointer);
      if (first == cudaSuccess && result != cudaSuccess) first = result;
      pointer = nullptr;
    }
  };
  release(bitmap_words); release(private_planes); release(word_prefix);
  release(chunk_counts); release(chunk_offsets); release(origin_remainders);
  release(overflow_flags); release(terminal_status);
  release(frontier_bits); release(frontier_occupied); release(frontier_status);
  release(frontier_reference);
  if(r178_enabled)std::cout<<"phase=r178_owner initializations="<<r178_initializations
      <<" continuations="<<r178_continuations<<" shape_fallbacks="<<r178_fallbacks
      <<" phase_bytes="<<static_cast<uint64_t>(r178_capacity)*4u<<std::endl;
  release(r178_phases);
  r178_capacity=r178_begin=r178_count=r178_span=0u;
  r178_expected=r178_initializations=r178_continuations=r178_fallbacks=0u;
  r178_primes_binding=r178_roots_binding=nullptr;r178_valid=r178_enabled=false;
  release(frontier_nonempty);
  for(auto& e:frontier_diag_events){if(e){cudaEventDestroy(e);e=nullptr;}}
  frontier_middle=UINT32_MAX;frontier_diagnosed=false;
  frontier_cached_groups = 0u; frontier_primes = frontier_roots = nullptr;
  for (auto& event : events) {
    if (event != nullptr) {
      const cudaError_t result = cudaEventDestroy(event);
      if (first == cudaSuccess && result != cudaSuccess) first = result;
      event = nullptr;
    }
  }
  configured_words = configured_chunks = maximum_planes = allocated_primes = 0u;
  cold_reference_checked = false;
  cuda_check(first, "RLOW batch cleanup");
}

__global__ void prepare_remainders(const uint32_t* primes, uint32_t prime_count,
    uint64_t origin, uint32_t positions, uint32_t* remainders, uint32_t* error) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= prime_count) return;
  const uint32_t p = primes[index];
  if (p < 2u) { atomicOr(error, 1u); return; }
  uint32_t current = static_cast<uint32_t>(origin % p);
  const uint32_t step = positions % p;
  for (uint32_t macro = 0u; macro < kMacros; ++macro) {
    remainders[static_cast<size_t>(macro) * prime_count + index] = current;
    if (macro + 1u != kMacros) current = advance_remainder(current, step, p);
  }
}

// Identical private-plane strike ownership to rlow_private_root_planes.
// grid.y is a disjoint macro; only origin%p has been hoisted to the kernel above.
__global__ void root_planes(uint32_t* planes, uint32_t positions,
    const uint32_t* primes, const uint32_t* roots, uint32_t pair_count,
    const uint32_t* remainders, uint32_t prime_count) {
  extern __shared__ uint32_t composite[];
  const uint32_t words = (positions + 31u) / 32u;
  const uint32_t macro = blockIdx.y;
  for (uint32_t word = threadIdx.x; word < words; word += blockDim.x)
    composite[word] = 0u;
  __syncthreads();
  for (uint64_t pair = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       pair < pair_count; pair += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
    const uint32_t prime_index = static_cast<uint32_t>(pair / 7u);
    const uint32_t p = primes[prime_index];
#ifdef RIECOIN_A54_OPTIONAL_PRIME_LIMIT
    if (p > RIECOIN_A54_OPTIONAL_PRIME_LIMIT && pair % 7u >= 2u) continue;
#endif
    if (p < 2u) continue;  // prepare_remainders already makes this fatal.
    const uint32_t origin_mod = remainders[static_cast<size_t>(macro) * prime_count + prime_index];
#ifdef RIECOIN_RLOW_COMPRESSED_ROOTS
    const uint32_t root = riecoin_compressed_root_storage::dense_root_at(
        roots, prime_index, static_cast<uint32_t>(pair % 7u));
#else
    const uint32_t root = roots[pair];
#endif
    uint64_t hit = root >= origin_mod ? root - origin_mod
        : static_cast<uint64_t>(root) + p - origin_mod;
    for (; hit < positions; hit += p)
      atomicOr(composite + (hit >> 5u), 1u << (hit & 31u));
  }
  __syncthreads();
  const size_t plane = static_cast<size_t>(macro) * gridDim.x + blockIdx.x;
  for (uint32_t word = threadIdx.x; word < words; word += blockDim.x)
    planes[plane * words + word] = composite[word];
}

__global__ void word_owner(uint32_t* survivors, const uint32_t* planes,
    uint32_t words, uint32_t plane_count, uint32_t positions) {
  const uint32_t word = blockIdx.x * blockDim.x + threadIdx.x;
  if (word >= words) return;
  const uint32_t macro = blockIdx.y;
  uint32_t composite_word = 0u;
  for (uint32_t plane = 0u; plane < plane_count; ++plane)
    composite_word |= planes[(static_cast<size_t>(macro) * plane_count + plane) * words + word];
  survivors[word_index(macro, word, words)] = ~composite_word &
      riecoin_rlow_bridge::valid_word_mask(word, positions);
}

__global__ void compare_reference_words(const uint32_t* batch,
    const uint32_t* reference, uint32_t words, uint32_t macro, uint32_t* mismatches) {
  const uint32_t word = blockIdx.x * blockDim.x + threadIdx.x;
  if (word < words && batch[word_index(macro, word, words)] != reference[word])
    atomicAdd(mismatches, 1u);
}

__global__ void frontier_occupancy(const uint32_t* bits,uint32_t* occupied,uint32_t count,uint32_t* nonempty=nullptr) {
  const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
  const uint32_t present=__ballot_sync(0xffffffffu,i<count && bits[i]!=0u);
  if ((threadIdx.x&31u)==0u && i<count){occupied[i/32u]=present;if(nonempty)atomicAdd(nonempty,__popc(present));}
}

__global__ void frontier_compare(const uint32_t* a,const uint32_t* b,uint32_t count,uint32_t* error) {
  const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count && a[i]!=b[i])atomicOr(error,16u);
}

__global__ void frontier_scatter(uint32_t* bits,const uint32_t* occupied,
    const uint32_t* primes,const uint32_t* roots,uint32_t dense,uint32_t count,uint32_t begin,uint32_t end,
    uint64_t origin,uint64_t span,uint32_t* error) {
  const uint32_t i=begin+blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=end)return;
  if(riecoin_compressed_root_storage::dense_count(roots)!=dense ||
     riecoin_compressed_root_storage::total_count(roots)!=count){atomicOr(error,8u);return;}
  const uint32_t p=primes[i];if(p<2u){atomicOr(error,1u);return;}
  const uint32_t remainder=static_cast<uint32_t>(origin%p);
  const uint32_t inverse=riecoin_compressed_root_storage::sparse_inverses(roots)[i-dense];
  uint32_t root=riecoin_compressed_root_storage::sparse_roots0(roots)[i-dense];
  uint32_t members=7u;
#ifdef RIECOIN_A54_OPTIONAL_PRIME_LIMIT
  if(p>RIECOIN_A54_OPTIONAL_PRIME_LIMIT)members=2u;
#endif
  for(uint32_t member=0;member<members;++member){
    if(member){const uint32_t gap=roots[riecoin_compressed_root_storage::kOffsetBase+member]-roots[riecoin_compressed_root_storage::kOffsetBase+member-1u];
      root=riecoin_compressed_root_storage::advance_for_gap(root,gap,inverse,p);}
    if(root>=p){atomicOr(error,2u);continue;}
    uint64_t hit=root>=remainder?root-remainder:static_cast<uint64_t>(root)+p-remainder;
    for(;hit<span;hit+=p){const uint32_t word=static_cast<uint32_t>(hit>>5u);
      if(!occupied || ((occupied[word>>5u]>>(word&31u))&1u))atomicAnd(bits+word,~(1u<<(hit&31u)));}
  }
}


__global__ void r178_scatter(uint32_t* bits,const uint32_t* occupied,
    const uint32_t* primes,const uint32_t* roots,uint32_t dense,uint32_t count,uint32_t begin,uint32_t end,
    uint64_t origin,uint64_t span,uint32_t* error,uint32_t* phases,bool initialize) {
  const uint32_t i=begin+blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=end)return;
  if(riecoin_compressed_root_storage::dense_count(roots)!=dense ||
     riecoin_compressed_root_storage::total_count(roots)!=count){atomicOr(error,8u);return;}
  const uint32_t p=primes[i];if(p<2u){atomicOr(error,1u);return;}
  if(p<=33554432u || !span || span>268435456u || origin>UINT64_MAX-span){atomicOr(error,32u);return;}
  const uint32_t inverse=riecoin_compressed_root_storage::sparse_inverses(roots)[i-dense];
  const uint32_t immutable_root=riecoin_compressed_root_storage::sparse_roots0(roots)[i-dense];
  if(immutable_root>=p || inverse>=p){atomicOr(error,64u);return;}
  uint32_t root=initialize?r174::first(immutable_root,p,origin):phases[i-begin];
  if(root>=p){atomicOr(error,128u);return;}
  const uint32_t next_phase=r174::advance_default_frontier(root,p,static_cast<uint32_t>(span));
#ifdef ASTRA_R248_EMIT_OPTIONAL
  ASTRA_R248_EMIT_OPTIONAL(root,p,inverse,origin,span);
#endif
  uint32_t members=7u;
#ifdef RIECOIN_A54_OPTIONAL_PRIME_LIMIT
  if(p>RIECOIN_A54_OPTIONAL_PRIME_LIMIT)members=2u;
#endif
  for(uint32_t member=0;member<members;++member){
    if(member){const uint32_t gap=roots[riecoin_compressed_root_storage::kOffsetBase+member]-roots[riecoin_compressed_root_storage::kOffsetBase+member-1u];
      root=riecoin_compressed_root_storage::advance_for_gap(root,gap,inverse,p);}
    if(root>=p){atomicOr(error,2u);continue;}
    uint64_t hit=root;
    for(;hit<span;hit+=p){const uint32_t word=static_cast<uint32_t>(hit>>5u);
      if(!occupied || ((occupied[word>>5u]>>(word&31u))&1u))atomicAnd(bits+word,~(1u<<(hit&31u)));}
  }
  phases[i-begin]=next_phase;
}

inline void r178_dispatch(const uint32_t* primes,const uint32_t* roots,uint32_t dense,
    uint32_t count,uint32_t begin,uint64_t origin,uint64_t span,cudaStream_t stream=cudaStreamPerThread) {
  const uint32_t tail=count-begin;
  if(tail>r178_capacity){
    if(r178_phases)cuda_check(cudaFree(r178_phases),"R178 phase resize release");
    r178_phases=nullptr;r178_valid=false;r178_capacity=0u;
    cuda_check(cudaMalloc(&r178_phases,static_cast<size_t>(tail)*4u),"R178 phase allocation");
    r178_capacity=tail;
  }
  const bool initialize=!r178_valid || r178_primes_binding!=primes || r178_roots_binding!=roots ||
      r178_begin!=begin || r178_count!=count || r178_span!=span || r178_expected!=origin;
  r178_scatter<<<(tail+255u)/256u,256u,0,stream>>>(frontier_bits,frontier_occupied,primes,roots,
      dense,count,begin,count,origin,span,frontier_status,r178_phases,initialize);
  cuda_check(cudaGetLastError(),"R178 full-frontier launch");
  // This owner advances on the same ordered CUDA stream. Sticky frontier
  // errors remain on the unchanged fail-closed path; no partial result repair.
  r178_valid=true;r178_primes_binding=primes;r178_roots_binding=roots;
  r178_begin=begin;r178_count=count;r178_span=static_cast<uint32_t>(span);r178_expected=origin+span;
  if(initialize)++r178_initializations;else ++r178_continuations;
}

__global__ void scan_words(const uint32_t* words, uint32_t word_count,
    uint32_t positions, uint32_t chunks, uint32_t* prefixes, uint32_t* counts) {
  using Scan = cub::BlockScan<uint32_t, 256u>;
  __shared__ typename Scan::TempStorage storage;
  const uint32_t word = blockIdx.x * 256u + threadIdx.x;
  uint32_t mask = 0u;
  if (word < word_count)
    mask = words[word_index(blockIdx.y, word, word_count)] &
        riecoin_rlow_bridge::valid_word_mask(word, positions);
  uint32_t prefix = 0u, population = 0u;
  const uint32_t value = static_cast<uint32_t>(__popc(mask));
  Scan(storage).ExclusiveSum(value, prefix, population);
  if (word < word_count) prefixes[word_index(blockIdx.y, word, word_count)] = prefix;
  if (threadIdx.x == 0u)
    counts[static_cast<size_t>(blockIdx.y) * chunks + blockIdx.x] = population;
}

// At most 64 chunks under the original private-macro geometry: one CUB block
// per macro, no host prefix, no cross-macro dependency or append atomic.
__global__ void scan_chunks(const uint32_t* counts, uint32_t chunks,
    uint32_t* offsets, uint32_t* macro_counts) {
  using Scan = cub::BlockScan<uint32_t, 256u>;
  __shared__ typename Scan::TempStorage storage;
  const size_t base = static_cast<size_t>(blockIdx.y) * chunks;
  const uint32_t value = threadIdx.x < chunks ? counts[base + threadIdx.x] : 0u;
  uint32_t prefix = 0u, total = 0u;
  Scan(storage).ExclusiveSum(value, prefix, total);
  if (threadIdx.x < chunks) offsets[base + threadIdx.x] = prefix;
  if (threadIdx.x == 0u) macro_counts[blockIdx.y] = total;
}

__global__ void pack_segments(const uint32_t* words, uint32_t word_count,
    uint32_t positions, uint32_t chunks, const uint32_t* prefixes,
    const uint32_t* offsets, const cgbn_mem_t<1280u>* base_origin,
    const cgbn_mem_t<1280u>* primorial, cgbn_mem_t<1280u>* candidates,
    uint64_t* factors, uint64_t origin, uint8_t* errors) {
  const uint32_t word = blockIdx.x * 256u + threadIdx.x;
  if (word >= word_count) return;
  const uint32_t macro = blockIdx.y;
  const size_t index = word_index(macro, word, word_count);
  const uint32_t original = words[index] & riecoin_rlow_bridge::valid_word_mask(word, positions);
  uint32_t mask = original;
  const uint32_t rank_base = offsets[static_cast<size_t>(macro) * chunks + word / 256u] + prefixes[index];
  while (mask != 0u) {
    const uint32_t bit = static_cast<uint32_t>(__ffs(mask) - 1);
    const uint32_t rank = rank_base + __popc(original & ((1u << bit) - 1u));
    uint32_t slot = 0u;
    uint64_t relative = 0u, factor = 0u;
    if (!map_output(origin, positions, macro, word, bit, rank, slot, relative, factor)) {
      errors[index] = 1u;
    } else {
      factors[slot] = factor;
      if (riecoin_rlow_bridge::rlow_pack_candidate(candidates + slot, base_origin,
                                                  primorial, relative))
        errors[index] = 1u;
    }
    mask &= mask - 1u;
  }
}

inline void setup(uint32_t words) {
  if (configured_words != 0u || words == 0u || words > kMaximumPositions / 32u ||
      rlow_queue::macro_counts == nullptr || rlow_queue::macro_offsets == nullptr)
    throw std::runtime_error("RLOW batch setup requires fresh state and queue setup");
  // Per-epoch accounting: no attribution of previous-target work to this job.
  remainder_gpu_ms = cold_reference_ms = 0.0;
  cold_reference_checked = false;
#ifdef RIECOIN_RLOW_SPARSE_EVENT_SIEVE
  rlow_sparse_event_sieve::reset();
#endif
  try {
    int ordinal = 0;
    cudaDeviceProp properties{};
    cuda_check(cudaGetDevice(&ordinal), "RLOW batch device");
    cuda_check(cudaGetDeviceProperties(&properties, ordinal), "RLOW batch properties");
    maximum_planes = static_cast<uint32_t>(properties.multiProcessorCount) * 2u;
    if (maximum_planes == 0u) throw std::runtime_error("RLOW batch no multiprocessors");
    const char* plane_env=std::getenv("ASTRA_A95_PLANES");
    uint32_t requested_planes=32u;
    if(plane_env){char* end=nullptr;const unsigned long parsed=std::strtoul(plane_env,&end,10);
      if(end==plane_env || *end!='\0' || parsed>1024u)throw std::runtime_error("A95 invalid plane cap");
      requested_planes=static_cast<uint32_t>(parsed);}
    if(requested_planes && requested_planes<maximum_planes)maximum_planes=requested_planes;
    std::cout<<"phase=a95_plane_cap requested="<<requested_planes<<" active="<<maximum_planes
      <<" macros="<<kMacros<<" prime_pairs_preserved=1"<<std::endl;
    configured_words = words;
    configured_chunks = (words + 255u) / 256u;
    const size_t all_words = static_cast<size_t>(kMacros) * words;
    const size_t all_chunks = static_cast<size_t>(kMacros) * configured_chunks;
    const size_t plane_bytes = all_words * maximum_planes * sizeof(uint32_t);
    cuda_check(cudaMalloc(&bitmap_words, all_words * 4u), "RLOW batch bitmaps");
    cuda_check(cudaMalloc(&private_planes, plane_bytes), "RLOW batch private planes");
    cuda_check(cudaMalloc(&word_prefix, all_words * 4u), "RLOW batch word prefixes");
    cuda_check(cudaMalloc(&chunk_counts, all_chunks * 4u), "RLOW batch chunk counts");
    cuda_check(cudaMalloc(&chunk_offsets, all_chunks * 4u), "RLOW batch chunk offsets");
    cuda_check(cudaMalloc(&overflow_flags, all_words), "RLOW batch overflow flags");
    cuda_check(cudaMalloc(&terminal_status, 2u * sizeof(uint32_t)), "RLOW batch terminal status");
    const char* r178_env=std::getenv("ASTRA_R178_PHASES");
    if(r178_env && std::strcmp(r178_env,"0")!=0 && std::strcmp(r178_env,"1")!=0)
      throw std::runtime_error("R178 phase setting must be 0 or 1");
    r178_enabled=r178_env && std::strcmp(r178_env,"1")==0;
    std::cout<<"phase=r178_configuration enabled="<<r178_enabled<<std::endl;
    const char* frontier_env=std::getenv("ASTRA_A93_FRONTIER_GROUPS");
    frontier_groups=0u;
    if(frontier_env){char* end=nullptr;const unsigned long parsed=std::strtoul(frontier_env,&end,10);
      if(end==frontier_env || *end!='\0' || parsed>32u)throw std::runtime_error("A93 invalid frontier setting");
      frontier_groups=static_cast<uint32_t>(parsed);}
    if(frontier_groups!=0u && frontier_groups!=4u && frontier_groups!=8u && frontier_groups!=16u && frontier_groups!=32u)
      throw std::runtime_error("A93 unsupported frontier group count");
    if(frontier_groups){
      const size_t future_words=all_words*frontier_groups;
      if(future_words>UINT32_MAX || future_words*32u>UINT32_MAX)throw std::runtime_error("A93 frontier exceeds geometry");
      cuda_check(cudaMalloc(&frontier_bits,future_words*4u),"A93 frontier bitmaps");
      cuda_check(cudaMalloc(&frontier_occupied,((future_words+31u)/32u)*4u),"A93 immutable occupancy");
      cuda_check(cudaMalloc(&frontier_status,4u),"A93 sticky cache error");
      const char* audit_env=std::getenv("ASTRA_A93_AUDIT");
      frontier_audit=audit_env && std::strcmp(audit_env,"1")==0;
      if(audit_env && !frontier_audit && std::strcmp(audit_env,"0")!=0)
        throw std::runtime_error("A93 invalid audit setting");
      if(frontier_audit)cuda_check(cudaMalloc(&frontier_reference,future_words*4u),"A93 audit-only shadow");
    }
    const char* cut=std::getenv("ASTRA_A94_CUTOVER");
    if(cut){char* end=nullptr;const unsigned long long parsed=std::strtoull(cut,&end,10);
      if(end==cut || *end!='\0' || parsed>UINT32_MAX)throw std::runtime_error("A94 invalid cutover");
      frontier_cutover=static_cast<uint32_t>(parsed);}
    const char* diag=std::getenv("ASTRA_A94_DIAGNOSTIC");
    frontier_diagnostic=diag && std::strcmp(diag,"1")==0;
    if(diag && !frontier_diagnostic && std::strcmp(diag,"0")!=0)throw std::runtime_error("A94 invalid diagnostic flag");
    if(frontier_diagnostic){cuda_check(cudaMalloc(&frontier_nonempty,4u),"A94 diagnostic count");
      for(auto& e:frontier_diag_events)cuda_check(cudaEventCreate(&e),"A94 diagnostic event");}
    std::cout<<"phase=a93_frontier groups="<<frontier_groups<<" no_sparse_mask=1 no_merge=1 audit="<<frontier_audit<<std::endl;
    std::cout<<"phase=a94_hybrid cutover="<<frontier_cutover<<" diagnostic="<<frontier_diagnostic<<std::endl;
    for (auto& event : events) cuda_check(cudaEventCreate(&event), "RLOW batch phase event");
    cuda_check(cudaFuncSetAttribute(root_planes, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(words * 4u)), "RLOW batch shared opt-in");
    std::cout << "phase=rlow_batch_ready macros=" << kMacros
              << " words_per_macro=" << words << " maximum_planes=" << maximum_planes
              << " plane_bytes=" << plane_bytes
              << " other_bytes=" << all_words * 9u + all_chunks * 8u + 8u
              << " remainder_bytes=deferred_to_first_fill" << std::endl;
  } catch (...) {
    try { cleanup(); } catch (...) {}
    throw;
  }
}

inline void fill(uint32_t* words, uint32_t word_count, uint32_t macro_positions,
    const uint32_t* primes, const uint32_t* roots, uint32_t pair_count,
    uint64_t origin, const cgbn_mem_t<1280u>* base_origin,
    const cgbn_mem_t<1280u>* primorial, cgbn_mem_t<1280u>* candidates,
    uint64_t* factors, uint32_t* count, unsigned long long* total_count) {
  const auto host_begin = std::chrono::steady_clock::now();
  std::function<void(uint64_t)> r259_builder;
  // Legacy one-macro scratch is used only for the one-time exact reference gate.
  validate_geometry(word_count, macro_positions, origin);
  if (configured_words != word_count || pair_count == 0u || pair_count % 7u != 0u ||
      words == nullptr || primes == nullptr || roots == nullptr || base_origin == nullptr || primorial == nullptr ||
      candidates == nullptr || factors == nullptr || count == nullptr || total_count == nullptr)
    throw std::runtime_error("RLOW batch fill binding/shape mismatch");
  const uint32_t prime_count = pair_count / 7u;
  uint32_t remainder_prime_count = prime_count;
#ifdef RIECOIN_RLOW_SPARSE_EVENT_SIEVE
  const uint32_t dense_count = rlow_sparse_event_sieve::partition(
      primes, prime_count, macro_positions, terminal_status);
  const bool use_sparse_events = dense_count != 0u && dense_count < prime_count &&
      (macro_positions & 31u) == 0u;
  if (use_sparse_events) {
    rlow_sparse_event_sieve::validate_interval(origin, macro_positions, kMacros);
    remainder_prime_count = dense_count;
  }
#endif
  if (origin_remainders == nullptr) {
    const size_t bytes = static_cast<size_t>(kMacros) * remainder_prime_count * 4u;
    cuda_check(cudaMalloc(&origin_remainders, bytes), "RLOW batch origin remainders");
    allocated_primes = remainder_prime_count;
    std::cout << "phase=rlow_batch_remainders_ready primes=" << remainder_prime_count
              << " bytes=" << bytes << " origin_div64_per_prime=1"
              << " timing_scope=first_fill_wall" << std::endl;
  }
  if (allocated_primes != remainder_prime_count)
    throw std::runtime_error("RLOW batch prime geometry changed without setup reset");
  const uint32_t dense_pair_count = remainder_prime_count * 7u;
  const uint32_t planes_needed = static_cast<uint32_t>((static_cast<uint64_t>(dense_pair_count) + 255u) / 256u);
  const uint32_t planes = maximum_planes < planes_needed ? maximum_planes : planes_needed;
  const uint32_t all_words = kMacros * word_count;
  cuda_check(cudaMemset(overflow_flags, 0, all_words), "RLOW batch overflow reset");
  cuda_check(cudaMemset(terminal_status, 0, 8u), "RLOW batch status reset");
  cuda_check(cudaEventRecord(events[0]), "RLOW batch begin");
  const uint32_t* active_bitmap=bitmap_words;
  if(frontier_groups && use_sparse_events){
    const uint64_t span=static_cast<uint64_t>(macro_positions)*kMacros;
    r259_resolve(origin,span,primes,roots);
    if(frontier_middle==UINT32_MAX){
      rlow_sparse_event_sieve::find_dense_count<<<1u,1u>>>(primes,prime_count,frontier_cutover,frontier_status);
      cuda_check(cudaMemcpy(&frontier_middle,frontier_status,4u,cudaMemcpyDeviceToHost),"A94 cold band partition");
      if(frontier_middle<dense_count)frontier_middle=dense_count;
      if(frontier_middle>prime_count)throw std::runtime_error("A94 partition overflow");
    }
    const bool hit=frontier_cached_groups && frontier_primes==primes && frontier_roots==roots &&
        origin>=frontier_origin && origin-frontier_origin<span*frontier_cached_groups &&
        (origin-frontier_origin)%span==0u;
    cuda_check(cudaEventRecord(events[1]),"A93 preparation charged to sieve");
    const auto build_frontier=[=](uint64_t origin){
      const cudaStream_t stream=r266_build_stream;
      const bool quantum=r270_build_slice!=UINT32_MAX;
      const bool phased=r273_style==2;
      frontier_cached_groups=frontier_groups;
      if((UINT64_MAX-origin)/span<frontier_cached_groups)
        frontier_cached_groups=static_cast<uint32_t>((UINT64_MAX-origin)/span);
      if(!frontier_cached_groups)throw std::runtime_error("A93 cannot form safe frontier");
      if(quantum && (frontier_cached_groups!=8u || r270_build_slice>=8u))
        throw std::runtime_error("R270 partial frontier geometry unsupported");
      frontier_origin=origin;frontier_primes=primes;frontier_roots=roots;
      if(!quantum || r270_build_slice==0u)
        cuda_check(cudaMemsetAsync(frontier_status,0,4u,stream),"A93 cache error reset");
      const bool diagnose=frontier_diagnostic && !frontier_diagnosed;
      if(quantum && diagnose)throw std::runtime_error("R270 partial diagnostic unsupported");
      if(diagnose){cuda_check(cudaMemsetAsync(frontier_nonempty,0,4u,stream),"A94 stats reset");
        cuda_check(cudaEventRecord(frontier_diag_events[0],stream),"A94 diagnostic begin");}
      for(uint32_t group=0;group<frontier_cached_groups;++group){
        // R272: prepare all dense slices first, then spread the expensive
        // shared bands over later Q0s. Never append all shared work at the end.
        if(quantum && (phased?r270_build_slice!=0u:group!=r270_build_slice))continue;
        const uint64_t next=origin+static_cast<uint64_t>(group)*span;
        prepare_remainders<<<(remainder_prime_count+255u)/256u,256u,0,stream>>>(
            primes,remainder_prime_count,next,macro_positions,origin_remainders,frontier_status);
        root_planes<<<dim3(planes,kMacros),256u,word_count*4u,stream>>>(
            private_planes,macro_positions,primes,roots,dense_pair_count,origin_remainders,remainder_prime_count);
        word_owner<<<dim3(configured_chunks,kMacros),256u,0,stream>>>(
            frontier_bits+static_cast<size_t>(group)*all_words,private_planes,word_count,planes,macro_positions);
        // The audit shadow must precede ALL new strikes, including this band.
        if(frontier_audit)cuda_check(cudaMemcpyAsync(frontier_reference+static_cast<size_t>(group)*all_words,
            frontier_bits+static_cast<size_t>(group)*all_words,static_cast<size_t>(all_words)*4u,
            cudaMemcpyDeviceToDevice,stream),"A94 pristine dense audit snapshot");
        if(frontier_middle>dense_count)frontier_scatter<<<(frontier_middle-dense_count+255u)/256u,256u,0,stream>>>(
            frontier_bits+static_cast<size_t>(group)*all_words,nullptr,primes,roots,dense_count,prime_count,
            dense_count,frontier_middle,next,span,frontier_status);
      }
      // Each phase still owns one ordered command prefix of this exact bank.
      if(!quantum || phased || r270_build_slice==7u){
      const uint32_t future_words=all_words*frontier_cached_groups;
      if(diagnose)cuda_check(cudaEventRecord(frontier_diag_events[1],stream),"A94 dense band done");
      if(!quantum || !phased || r270_build_slice==0u)
        frontier_occupancy<<<(future_words+255u)/256u,256u,0,stream>>>(frontier_bits,frontier_occupied,future_words,diagnose?frontier_nonempty:nullptr);
      if(diagnose)cuda_check(cudaEventRecord(frontier_diag_events[2],stream),"A94 occupancy done");
#ifdef ASTRA_R197_FRONTIER_PROBE
      // Isolated measurement executable only; undefined in every production build.
      ASTRA_R197_FRONTIER_PROBE(frontier_bits,frontier_occupied,primes,roots,
          dense_count,prime_count,frontier_middle,origin,span*frontier_cached_groups);
#endif
      if(prime_count>frontier_middle && (!quantum || !phased || r270_build_slice==1u)){
        const uint64_t future_span=span*frontier_cached_groups;
        if(r178_enabled && frontier_cutover>=33554432u && future_span && future_span<=268435456u &&
            origin<=UINT64_MAX-future_span) {
          r178_dispatch(primes,roots,dense_count,prime_count,frontier_middle,origin,future_span,stream);
        }else{
          r178_valid=false;if(r178_enabled)++r178_fallbacks;
          frontier_scatter<<<(prime_count-frontier_middle+255u)/256u,256u,0,stream>>>(
              frontier_bits,frontier_occupied,primes,roots,dense_count,prime_count,
              frontier_middle,prime_count,origin,future_span,frontier_status);
        }
      }
#ifdef ASTRA_R209_FRONTIER_APPLY
      if(!quantum || !phased)ASTRA_R209_FRONTIER_APPLY(frontier_bits,origin,span*frontier_cached_groups,frontier_status,stream);
      else if(r270_build_slice>=2u && r270_build_slice<=6u)
        ASTRA_R272_PHASE_APPLY(frontier_bits,origin,span*frontier_cached_groups,frontier_status,r270_build_slice-2u,stream);
#elif defined(ASTRA_R205_APPLY)
      ASTRA_R205_APPLY(frontier_bits,frontier_occupied,origin,span*frontier_cached_groups);
#elif defined(ASTRA_R201_APPLY)
      ASTRA_R201_APPLY(frontier_bits,origin,span*frontier_cached_groups,true);
#endif
      if(diagnose){
        cuda_check(cudaEventRecord(frontier_diag_events[3],stream),"A94 shared band done");
        cuda_check(cudaEventSynchronize(frontier_diag_events[3]),"A94 diagnostic-only cold sync");
        float ms[3]{};for(uint32_t d=0;d<3;++d)cuda_check(cudaEventElapsedTime(&ms[d],frontier_diag_events[d],frontier_diag_events[d+1]),"A94 phase diagnostic");
        uint32_t occupied=0;cuda_check(cudaMemcpy(&occupied,frontier_nonempty,4u,cudaMemcpyDeviceToHost),"A94 cold occupancy");
        std::cout<<"phase=a94_autopsy cutover="<<frontier_cutover<<" middle_primes="<<frontier_middle-dense_count
          <<" shared_primes="<<prime_count-frontier_middle<<" nonempty_words="<<occupied<<" total_words="<<future_words
          <<" dense_medium_ms="<<ms[0]<<" occupancy_ms="<<ms[1]<<" shared_ms="<<ms[2]<<std::endl;
        frontier_diagnosed=true;
      }
      if(frontier_audit && (!quantum || r270_build_slice==7u)){
        for(uint32_t g=0;g<frontier_cached_groups;++g)
          rlow_sparse_event_sieve::scatter_sparse<<<(prime_count-dense_count+255u)/256u,256u,0,stream>>>(
              frontier_reference+static_cast<size_t>(g)*all_words,primes,roots,dense_count,prime_count,
              macro_positions,kMacros,origin+static_cast<uint64_t>(g)*span,frontier_status);
        frontier_compare<<<(future_words+255u)/256u,256u,0,stream>>>(frontier_reference,frontier_bits,future_words,frontier_status);
      }
      if(!quantum || r270_build_slice==7u)++rlow_sparse_event_sieve::scatter_launches;
      }
    };
    if(!hit)build_frontier(origin);
    r259_builder=build_frontier;
    r259_audit(origin);
    const uint32_t group=static_cast<uint32_t>((origin-frontier_origin)/span);
    active_bitmap=frontier_bits+static_cast<size_t>(group)*all_words;
    cuda_check(cudaMemcpyAsync(terminal_status+1u,frontier_status,4u,cudaMemcpyDeviceToDevice),
               "A93 preserve future cache errors");
  }else{
  prepare_remainders<<<(remainder_prime_count + 255u) / 256u, 256u>>>(
      primes, remainder_prime_count, origin, macro_positions, origin_remainders, terminal_status + 1u);
  cuda_check(cudaEventRecord(events[1]), "RLOW batch remainders complete");
  root_planes<<<dim3(planes, kMacros), 256u, word_count * 4u>>>(
      private_planes, macro_positions, primes, roots, dense_pair_count, origin_remainders, remainder_prime_count);
  word_owner<<<dim3(configured_chunks, kMacros), 256u>>>(
      bitmap_words, private_planes, word_count, planes, macro_positions);
#ifdef RIECOIN_RLOW_SPARSE_EVENT_SIEVE
  if (use_sparse_events) {
    rlow_sparse_event_sieve::scatter_sparse<<<(prime_count - dense_count + 255u) / 256u, 256u>>>(
        bitmap_words, primes, roots, dense_count, prime_count, macro_positions,
        kMacros, origin, terminal_status + 1u);
    ++rlow_sparse_event_sieve::scatter_launches;
  }
#endif
  }
  cuda_check(cudaEventRecord(events[2]), "RLOW batch sieve complete");
  scan_words<<<dim3(configured_chunks, kMacros), 256u>>>(
      active_bitmap, word_count, macro_positions, configured_chunks, word_prefix, chunk_counts);
  scan_chunks<<<dim3(1u, kMacros), 256u>>>(chunk_counts, configured_chunks,
                                        chunk_offsets, rlow_queue::macro_counts);
  cuda_check(cudaEventRecord(events[3]), "RLOW batch scan complete");
  pack_segments<<<dim3(configured_chunks, kMacros), 256u>>>(
      active_bitmap, word_count, macro_positions, configured_chunks, word_prefix, chunk_offsets,
      base_origin, primorial, candidates, factors, origin, overflow_flags);
  cuda_check(cudaEventRecord(events[4]), "RLOW batch pack complete");
  riecoin_rlow_bridge::rlow_scan_macro_counts<<<1u, 1u>>>(
      rlow_queue::macro_counts, kMacros, rlow_queue::macro_offsets, count);
  riecoin_rlow_bridge::rlow_reduce_u8_flags<<<1u, 256u>>>(
      overflow_flags, all_words, terminal_status);
  cuda_check(cudaEventRecord(events[5]), "RLOW batch metadata complete");
  cuda_check(cudaGetLastError(), "RLOW batch fill kernels");
  uint32_t status[2]{};
#ifdef RIECOIN_RLOW_ASYNC_FRONTIER
  if (!cold_reference_checked) {
#endif
  const auto guard_begin = std::chrono::steady_clock::now();
  cuda_check(cudaMemcpy(status, terminal_status, sizeof(status), cudaMemcpyDeviceToHost),
             "RLOW batch terminal guard");
  rlow_queue::safety_guard_host_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - guard_begin).count();
  rlow_queue::added_d2h_bytes += sizeof(status);
  if (status[0] != 0u || status[1] != 0u)
    throw std::runtime_error("RLOW batch overflow/invalid prime: no PRP or result claim allowed");
#ifdef RIECOIN_RLOW_ASYNC_FRONTIER
  }
#endif
  if (!cold_reference_checked) {
    const auto reference_begin = std::chrono::steady_clock::now();
    cuda_check(cudaMemset(terminal_status, 0, 4u), "RLOW batch reference reset");
    bool reference_grouped = false;
#ifdef ASTRA_R203_COMPARE
    reference_grouped = ASTRA_R203_COMPARE(active_bitmap,macro_positions,
        primes,roots,pair_count,origin,kMacros,terminal_status);
#endif
    for (uint32_t macro = 0u; !reference_grouped && macro < kMacros; ++macro) {
      const uint64_t macro_origin = origin + static_cast<uint64_t>(macro) * macro_positions;
      rlow_sieve_launch(words, macro_positions, primes, roots, pair_count, macro_origin);
#ifdef ASTRA_R201_APPLY
      ASTRA_R201_APPLY(words,macro_origin,macro_positions,false);
#endif
      compare_reference_words<<<configured_chunks, 256u>>>(
          active_bitmap, words, word_count, macro, terminal_status);
    }
    cuda_check(cudaGetLastError(), "RLOW batch reference kernels");
    uint32_t mismatches = 0u;
    cuda_check(cudaMemcpy(&mismatches, terminal_status, sizeof(mismatches), cudaMemcpyDeviceToHost),
               "RLOW batch complete bitmap comparison");
    rlow_queue::added_d2h_bytes += sizeof(mismatches);
    cold_reference_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - reference_begin).count();
    std::cout << "phase=rlow_batch_cold_reference macros=" << kMacros
              << " words=" << all_words << " mismatches=" << mismatches
              << " cold_reference_ms=" << cold_reference_ms
              << " terminal=" << (mismatches == 0u ? "PASS" : "FAIL") << std::endl;
    if (mismatches != 0u)
      throw std::runtime_error("RLOW batch bitmap differs from original sieve: no PRP admitted");
    cold_reference_checked = true;
#ifdef ASTRA_R211_REFERENCE_DONE
    ASTRA_R211_REFERENCE_DONE();
#endif
  }
  rlow_queue::add_entered_count<<<1u, 1u>>>(count, total_count);
  cuda_check(cudaGetLastError(), "RLOW batch entered accounting");
#ifndef RIECOIN_RLOW_ASYNC_FRONTIER
  double* timings[] = {&remainder_gpu_ms, &rlow_queue::sieve_gpu_ms,
      &rlow_queue::scan_gpu_ms, &rlow_queue::pack_gpu_ms, &rlow_queue::metadata_gpu_ms};
  for (uint32_t phase = 0u; phase < 5u; ++phase) {
    float elapsed = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed, events[phase], events[phase + 1u]),
               "RLOW batch phase timing");
    *timings[phase] += elapsed;
  }
#endif
  ++rlow_queue::fill_calls;
  r259_finish(r259_builder,static_cast<uint64_t>(macro_positions)*kMacros);
  rlow_queue::fill_host_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - host_begin).count();
}
#endif  // __CUDACC__
}  // namespace rlow_batch_producer
#undef RLOW_BATCH_HD

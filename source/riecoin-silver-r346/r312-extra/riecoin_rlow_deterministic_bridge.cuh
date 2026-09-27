#pragma once

// R-LOW successor bridge (isolated, no changes to the vertical prototype).
//
// The private-root sieve is allowed to use shared-memory atomicOr while each
// block owns its complete bitmap plane.  Everything after the unique-word
// owner is deliberately deterministic: bitmap words are block-scanned, queue
// macros have fixed disjoint ranges, candidates are written in word/bit order,
// and PRP/funnel outputs are compacted with the same scan primitive.  There is
// no global bitmap atomic and no 65k-position scan in this bridge.
//
// Include after riecoin-cuda-prp1152.cuh (the dynamic 1280-bit evaluator is
// used by rlow_segmented_q0_prp).  The host side owns allocation, memset and
// exact replay gates; this file contains only reusable CUDA atoms.

#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
// CGBN is already included by riecoin-cuda-prp1152.cuh. Its upstream header
// is not idempotent and must not be included a second time here.

#include <cstdint>

namespace riecoin_rlow_bridge {

constexpr uint32_t kBlockThreads = 256u;
// A measured deep macro may contain only about 0.9k Q0 survivors.  Sixteen
// successive disjoint macros fill a 65,536-slot PRP workspace (about 14k Q0
// entries at the observed rate) while retaining a bounded per-macro overflow
// gate.  Both values are launch parameters below; these are the first canary
// geometry, not a correctness constant.
constexpr uint32_t kQueueMacroCapacity = 4096u;
#ifdef RIECOIN_RLOW_QUEUE_MACROS
constexpr uint32_t kQueueMacroCount = RIECOIN_RLOW_QUEUE_MACROS;
#else
constexpr uint32_t kQueueMacroCount = 16u;
#endif
static_assert(kQueueMacroCount >= 16u && kQueueMacroCount <= 64u &&
              kQueueMacroCount % 16u == 0u, "unsupported queue macro geometry");
constexpr uint32_t kQueueCapacity =
    kQueueMacroCapacity * kQueueMacroCount;

__device__ __forceinline__ uint32_t valid_word_mask(
    uint32_t word, uint32_t candidate_count) {
  const uint64_t first = static_cast<uint64_t>(word) * 32u;
  if (first >= candidate_count) return 0u;
  const uint32_t remaining = candidate_count - static_cast<uint32_t>(first);
  if (remaining >= 32u) return 0xffffffffu;
  return (1u << remaining) - 1u;
}

// Stable exclusive scan of bitmap-word populations.  One block owns 256
// words; only shared-memory CUB state is used.  `chunk_counts` is later
// scanned by rlow_scan_chunk_counts, so no global counter is touched here.
__global__ void rlow_scan_bitmap_words(
    const uint32_t* survivor_words, uint32_t word_count,
    uint32_t candidate_count, uint32_t* word_prefix,
    uint32_t* chunk_counts) {
  using Scan = cub::BlockScan<uint32_t, kBlockThreads>;
  __shared__ typename Scan::TempStorage scan_storage;
  const uint32_t word = blockIdx.x * kBlockThreads + threadIdx.x;
  uint32_t mask = 0u;
  if (word < word_count)
    mask = survivor_words[word] & valid_word_mask(word, candidate_count);
  const uint32_t population = __popc(mask);
  uint32_t prefix = 0u;
  uint32_t block_population = 0u;
  Scan(scan_storage).ExclusiveSum(population, prefix, block_population);
  if (word < word_count) word_prefix[word] = prefix;
  if (threadIdx.x == 0u) chunk_counts[blockIdx.x] = block_population;
}

// Exact composition of an additional stage-local root sieve.  This is the
// useful r0 lever when the all-member PTL is already expensive: pass only the
// Q0 roots for a higher prime band, then intersect word-for-word before any
// PRP.  The owner is unique and no bitmap atomic is needed.
__global__ void rlow_intersect_bitmap_words(
    uint32_t* survivor_words, const uint32_t* stage_words,
    uint32_t word_count, uint32_t candidate_count) {
  const uint32_t word = blockIdx.x * kBlockThreads + threadIdx.x;
  if (word >= word_count) return;
  survivor_words[word] &= stage_words[word] &
                          valid_word_mask(word, candidate_count);
}

// The number of chunks is at most ceil((8*65536/32)/256)=64 for the intended
// queue.  The loop also keeps this atom valid for the larger offline fixtures.
__global__ void rlow_scan_chunk_counts(
    const uint32_t* chunk_counts, uint32_t chunk_count,
    uint32_t* chunk_offsets, uint32_t* total_count) {
  using Scan = cub::BlockScan<uint32_t, kBlockThreads>;
  __shared__ typename Scan::TempStorage scan_storage;
  __shared__ uint32_t carry;
  if (threadIdx.x == 0u) carry = 0u;
  __syncthreads();
  for (uint32_t begin = 0u; begin < chunk_count;
       begin += kBlockThreads) {
    const uint32_t index = begin + threadIdx.x;
    const uint32_t value = index < chunk_count ? chunk_counts[index] : 0u;
    uint32_t prefix = 0u;
    uint32_t block_total = 0u;
    Scan(scan_storage).ExclusiveSum(value, prefix, block_total);
    __syncthreads();
    if (index < chunk_count) chunk_offsets[index] = carry + prefix;
    // Every warp must finish reading carry before lane zero advances it.
    __syncthreads();
    if (threadIdx.x == 0u) carry += block_total;
    __syncthreads();
  }
  if (threadIdx.x == 0u) *total_count = carry;
}

__device__ __forceinline__ bool rlow_pack_candidate(
    cgbn_mem_t<1280u>* destination,
    const cgbn_mem_t<1280u>* base_origin,
    const cgbn_mem_t<1280u>* primorial,
    uint64_t relative_factor) {
  if (relative_factor > UINT32_MAX) return true;
  uint64_t carry = 0u;
#pragma unroll
  for (uint32_t limb = 0u; limb < 40u; ++limb) {
    const uint64_t value =
        static_cast<uint64_t>(primorial->_limbs[limb]) * relative_factor +
        static_cast<uint64_t>(base_origin->_limbs[limb]) + carry;
    destination->_limbs[limb] = static_cast<uint32_t>(value);
    carry = value >> 32u;
  }
  return carry != 0u || relative_factor > UINT32_MAX;
}

// Compact one sieve bitmap into a fixed queue segment.  The segment is
// deterministic (`macro_index * macro_capacity`) and therefore permits
// successive macro production without an append atomic.  Candidate values
// preserve the absolute factor range beginning at factor_origin.
// The caller must keep base_origin anchored at the first macro for the whole
// queue and advance the production base once by macro_count*candidate_count
// after the queue has drained.  This is the range-conservation boundary.
__global__ void rlow_compact_bitmap_segment(
    const uint32_t* survivor_words, uint32_t word_count,
    uint32_t candidate_count, const uint32_t* word_prefix,
    const uint32_t* chunk_offsets, const cgbn_mem_t<1280u>* base_origin,
    const cgbn_mem_t<1280u>* primorial, cgbn_mem_t<1280u>* candidates,
    uint64_t* factors, uint64_t factor_origin, uint32_t macro_index,
    uint32_t macro_capacity, uint32_t workspace_capacity,
    uint8_t* overflow_flags) {
  const uint32_t word = blockIdx.x * kBlockThreads + threadIdx.x;
  if (word >= word_count) return;
  uint32_t mask = survivor_words[word] & valid_word_mask(word, candidate_count);
  const uint32_t chunk = word / kBlockThreads;
  const uint32_t rank_base = chunk_offsets[chunk] + word_prefix[word];
  while (mask != 0u) {
    const uint32_t bit = static_cast<uint32_t>(__ffs(mask) - 1);
    const uint32_t local_rank = rank_base + __popc(
        (survivor_words[word] & valid_word_mask(word, candidate_count)) &
        ((1u << bit) - 1u));
    const uint64_t relative_factor =
        static_cast<uint64_t>(macro_index) * candidate_count +
        static_cast<uint64_t>(word) * 32u + bit;
    const uint32_t output = macro_index * macro_capacity + local_rank;
    if (output >= workspace_capacity ||
        local_rank >= macro_capacity || relative_factor > UINT32_MAX) {
      overflow_flags[word] = 1u;
    } else {
      factors[output] = factor_origin + relative_factor;
      if (rlow_pack_candidate(candidates + output, base_origin, primorial,
                              relative_factor))
        overflow_flags[word] = 1u;
    }
    mask &= mask - 1u;
  }
}

// Deterministic prefix over the bounded macro counts.  This is intentionally a
// single thread: it is bounded, auditable, and removes the fixed one-macro PRP
// boundary without introducing any append counter.
__global__ void rlow_scan_macro_counts(
    const uint32_t* macro_counts, uint32_t macro_count,
    uint32_t* macro_offsets, uint32_t* total_count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  uint32_t prefix = 0u;
  for (uint32_t macro = 0u; macro < macro_count; ++macro) {
    macro_offsets[macro] = prefix;
    prefix += macro_counts[macro];
  }
  *total_count = prefix;
}

// Each segment is physically disjoint.  A counted PRP launch over the packed
// queue is therefore replaced by this segmented launch, which executes only
// valid entries while retaining enough resident threads to fill the device.
__global__ void rlow_segmented_q0_prp(
    const cgbn_mem_t<1280u>* candidates, uint8_t* verdicts,
    const uint32_t* macro_counts, uint32_t macro_count,
    uint32_t macro_capacity) {
  const uint32_t global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t slot = global_thread / 8u;
  const uint32_t macro = slot / macro_capacity;
  const uint32_t local = slot - macro * macro_capacity;
  const uint32_t queue_slots = macro_count * macro_capacity;
  const bool valid = macro < macro_count && local < macro_counts[macro];
  // CGBN is cooperative: all eight lanes in a valid instance must evaluate.
  // Only the final byte store is restricted to the instance's first lane.
  const bool probable = valid &&
      riecoin_cuda::evaluate_euler_jacobi_dynamic_1280_tpi8(candidates, slot);
  if ((global_thread & 7u) == 0u && slot < queue_slots)
    verdicts[slot] = probable ? 1u : 0u;
}

// Generic stable flag scan/compaction.  Use this for Q0 passes, complete
// tuple factors, or any other binary funnel stage.  Invalid queue slots must
// be represented by a zero flag by the producer.
__global__ void rlow_scan_flags(
    const uint8_t* flags, uint32_t count, uint32_t* index_prefix,
    uint32_t* chunk_counts) {
  using Scan = cub::BlockScan<uint32_t, kBlockThreads>;
  __shared__ typename Scan::TempStorage scan_storage;
  const uint32_t index = blockIdx.x * kBlockThreads + threadIdx.x;
  const uint32_t value = index < count && flags[index] != 0u ? 1u : 0u;
  uint32_t prefix = 0u;
  uint32_t block_total = 0u;
  Scan(scan_storage).ExclusiveSum(value, prefix, block_total);
  if (index < count) index_prefix[index] = prefix;
  if (threadIdx.x == 0u) chunk_counts[blockIdx.x] = block_total;
}

__global__ void rlow_compact_flagged_factors(
    const uint8_t* flags, const uint64_t* source_factors, uint32_t count,
    const uint32_t* index_prefix, const uint32_t* chunk_offsets,
    uint64_t* destination_factors, uint32_t capacity,
    uint8_t* overflow_flags) {
  const uint32_t index = blockIdx.x * kBlockThreads + threadIdx.x;
  if (index >= count || flags[index] == 0u) return;
  const uint32_t output = chunk_offsets[index / kBlockThreads] +
                          index_prefix[index];
  if (output >= capacity) {
    overflow_flags[index] = 1u;
    return;
  }
  destination_factors[output] = source_factors[index];
}

// Stage-local exact filter for a compact factor queue.  Generate a private
// bitmap for one member/high-prime band, then run this over the compact queue
// before constructing that member's 1280-bit PRP workspace.  It charges only
// one bitmap lookup per queued factor and never appends with an atomic.
__global__ void rlow_filter_factor_list(
    const uint64_t* factors, uint32_t count, uint64_t factor_origin,
    uint32_t macro_index, uint32_t macro_positions,
    const uint32_t* stage_survivor_words, uint32_t candidate_count,
    uint8_t* flags) {
  const uint32_t index = blockIdx.x * kBlockThreads + threadIdx.x;
  if (index >= count) return;
  const uint64_t factor = factors[index];
  const uint64_t range_begin = factor_origin +
      static_cast<uint64_t>(macro_index) * macro_positions;
  const uint64_t relative = factor >= range_begin ? factor - range_begin : UINT64_MAX;
  const bool in_range = relative < candidate_count;
  const uint32_t word = static_cast<uint32_t>(relative >> 5u);
  const uint32_t bit = static_cast<uint32_t>(relative & 31u);
  flags[index] = in_range &&
      ((stage_survivor_words[word] >> bit) & 1u) != 0u ? 1u : 0u;
}

// Tuple construction has a per-output overflow byte instead of a global
// atomic counter.  The host can reduce the bytes with rlow_reduce_u8_flags.
__global__ void rlow_construct_tuple_members(
    const uint64_t* pass_factors, uint32_t pass_count,
    const cgbn_mem_t<1280u>* base_origin,
    const cgbn_mem_t<1280u>* primorial, const uint32_t* member_offsets,
    uint64_t factor_origin, cgbn_mem_t<1280u>* members,
    uint8_t* overflow_flags) {
  const uint32_t output = blockIdx.x * kBlockThreads + threadIdx.x;
  const uint32_t total = pass_count * 6u;
  if (output >= total) return;
  const uint32_t pass = output / 6u;
  const uint32_t member = output - pass * 6u;
  const uint64_t factor = pass_factors[pass];
  if (factor < factor_origin || factor - factor_origin > UINT32_MAX) {
    overflow_flags[output] = 1u;
    return;
  }
  const uint64_t relative = factor - factor_origin;
  uint64_t carry = 0u;
#pragma unroll
  for (uint32_t limb = 0u; limb < 40u; ++limb) {
    const uint64_t value =
        static_cast<uint64_t>(primorial->_limbs[limb]) * relative +
        static_cast<uint64_t>(base_origin->_limbs[limb]) + carry;
    members[output]._limbs[limb] = static_cast<uint32_t>(value);
    carry = value >> 32u;
  }
  uint64_t add = member_offsets[member];
#pragma unroll
  for (uint32_t limb = 0u; limb < 40u && add != 0u; ++limb) {
    const uint64_t value =
        static_cast<uint64_t>(members[output]._limbs[limb]) + add;
    members[output]._limbs[limb] = static_cast<uint32_t>(value);
    add = value >> 32u;
  }
  if (carry != 0u || add != 0u) overflow_flags[output] = 1u;
}

// Classify one Q0-passed factor into the exact Riecoin v1 prefix relation:
// member 0 (the second constellation member) is mandatory; later members may
// contain a hole only while five total probable primes remain reachable.
// `active_mask` is min-five viability, NOT the all-prime consecutive prefix.
__global__ void rlow_classify_tuple(
    const uint8_t* member_verdicts, uint32_t pass_count,
    uint8_t* member_masks, uint8_t* active_masks, uint8_t* reject_stages,
    uint8_t* complete_flags) {
  const uint32_t index = blockIdx.x * kBlockThreads + threadIdx.x;
  if (index >= pass_count) return;
  uint8_t member_mask = 0u;
#pragma unroll
  for (uint32_t member = 0u; member < 6u; ++member)
    if (member_verdicts[index * 6u + member] != 0u)
      member_mask |= static_cast<uint8_t>(1u << member);

  uint32_t probable_prime_count = 1u;  // Q0 already passed.
  uint8_t active_mask = 0u;
  uint8_t reject_stage = 0u;
  bool alive = true;
#pragma unroll
  for (uint32_t member = 0u; member < 6u; ++member) {
    if (!alive) break;
    const bool passed = (member_mask & (1u << member)) != 0u;
    if (passed) ++probable_prime_count;
    const uint32_t remaining = 5u - member;
    const bool viable = (member != 0u || passed) &&
                        probable_prime_count + remaining >= 5u;
    if (viable) {
      active_mask |= static_cast<uint8_t>(1u << member);
    } else {
      reject_stage = static_cast<uint8_t>(member + 1u);
      alive = false;
    }
  }
  member_masks[index] = member_mask;
  active_masks[index] = active_mask;
  reject_stages[index] = reject_stage;
  complete_flags[index] = alive ? 1u : 0u;
}

// Deterministic funnel reduction.  The first rejected stage (0) is Q0; stages
// 1..6 are tuple members, and complete_count is the PRP all-seven/at-least-five
// set that must still pass the bounded exact GMP verifier.
__global__ void rlow_reduce_funnel(
    const uint8_t* member_masks, const uint8_t* active_masks,
    const uint8_t* reject_stages, const uint8_t* complete_flags,
    uint32_t pass_count, const uint32_t* entered_count,
    const uint32_t* q0_pass_count, uint64_t* member_pass_counts,
    uint64_t* active_counts, uint64_t* rejected_counts,
    uint64_t* complete_count, uint64_t* strict_prefix_counts) {
  if (blockIdx.x != 0u) return;
  __shared__ uint64_t partial[18][kBlockThreads];
  uint64_t local[18]{};
  for (uint32_t index = threadIdx.x; index < pass_count;
       index += kBlockThreads) {
    const uint8_t members = member_masks[index];
    const uint8_t active = active_masks[index];
    bool strict_alive = true;
#pragma unroll
    for (uint32_t member = 0u; member < 6u; ++member) {
      local[member] += (members & (1u << member)) != 0u;
      local[6u + member] += (active & (1u << member)) != 0u;
      strict_alive = strict_alive && (members & (1u << member)) != 0u;
      local[12u + member] += strict_alive;
    }
  }
#pragma unroll
  for (uint32_t lane = 0u; lane < 18u; ++lane)
    partial[lane][threadIdx.x] = local[lane];
  __syncthreads();
  for (uint32_t stride = kBlockThreads / 2u; stride != 0u; stride >>= 1u) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (uint32_t lane = 0u; lane < 18u; ++lane)
        partial[lane][threadIdx.x] += partial[lane][threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x != 0u) return;
  for (uint32_t member = 0u; member < 6u; ++member) {
    member_pass_counts[member] = partial[member][0];
    active_counts[member] = partial[6u + member][0];
    strict_prefix_counts[member] = partial[12u + member][0];
  }
  for (uint32_t stage = 0u; stage < 7u; ++stage) rejected_counts[stage] = 0u;
  const uint64_t entered = *entered_count;
  const uint64_t passed = *q0_pass_count;
  rejected_counts[0] = entered >= passed ? entered - passed : UINT64_MAX;
  // The per-item stage histogram is kept separate from the 13 compact
  // counters above so the stage identity remains exact and inspectable.
  for (uint32_t index = 0u; index < pass_count; ++index) {
    const uint8_t stage = reject_stages[index];
    if (stage >= 1u && stage <= 6u) ++rejected_counts[stage];
  }
  *complete_count = 0u;
  for (uint32_t index = 0u; index < pass_count; ++index)
    *complete_count += complete_flags[index] != 0u ? 1u : 0u;
}

// Bounded deterministic reduction for overflow byte arrays.  This is useful
// as a terminal gate after candidate/member construction and compaction.
__global__ void rlow_reduce_u8_flags(
    const uint8_t* flags, uint32_t count, uint32_t* result) {
  __shared__ uint32_t partial[kBlockThreads];
  uint32_t local = 0u;
  for (uint32_t index = threadIdx.x; index < count; index += kBlockThreads)
    local += flags[index] != 0u ? 1u : 0u;
  partial[threadIdx.x] = local;
  __syncthreads();
  for (uint32_t stride = kBlockThreads / 2u; stride != 0u; stride >>= 1u) {
    if (threadIdx.x < stride)
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0u) *result = partial[0];
}

}  // namespace riecoin_rlow_bridge

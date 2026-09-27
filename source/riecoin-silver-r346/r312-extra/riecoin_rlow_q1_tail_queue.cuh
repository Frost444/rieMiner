#pragma once

// Isolated R12 atom. Include after the existing bridge/CGBN evaluator, once.
// API: immutable Binding + Controller on host; DeviceQueue and GPU atoms below.
// Same stream, one writer: construct_q1 -> existing PRP -> bridge flag/chunk
// scans -> append_q1 -> commit_append_queue -> checked scalar receipt.
// Flush: construct_tail -> existing PRP(5*pending) -> reduce_tail ->
// commit_drain_queue -> checked scalar receipt -> Controller::commit_drain.
// Root owns allocations, launches, synchronization, error checks and samples.
// Check construction/error status BEFORE either PRP launch. Never zero unknown
// verdicts to pretend they were tested. Marginal unknown[i]=Q0passed-tested[i].
// No CPU factor scan; no host rebuilding of candidates. All GPU errors fatal.
// The 250ms deadline is an admission/poll contract, NOT GPU preemption: the
// caller must schedule a flush by deadline, not block across it in another kernel.
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef __CUDACC__
#define RLOW_Q1_HD __host__ __device__
#else
#define RLOW_Q1_HD
#endif

namespace rlow_q1_tail {
constexpr uint32_t kLimbs = 40u;
using Words = std::array<uint32_t, kLimbs>;
struct Binding {
  std::string work_id, template_id;
  uint64_t generation = 0u;
  uint64_t factor_anchor = 0u;
  Words base_at_anchor{}, primorial{};
  bool operator==(const Binding& other) const {
    return work_id == other.work_id && template_id == other.template_id && generation == other.generation &&
        factor_anchor == other.factor_anchor && base_at_anchor == other.base_at_anchor &&
        primorial == other.primorial;
  }
};
struct Frame { uint64_t begin = 0u, end = 0u; };
struct Ticket {
  uint64_t generation = 0u, serial = 0u;
  Frame frame{};
  uint32_t pending_before = 0u, upper_bound = 0u;
};

// Host bookkeeping never owns the factors themselves. A failed/unknown GPU
// transaction poisons the controller; it cannot be retried or silently reset.
class Controller {
 public:
  Controller(uint32_t capacity, uint32_t threshold, uint64_t wait_us = 250000u)
      : capacity_(capacity), threshold_(threshold), wait_us_(wait_us) {
    if (capacity == 0u || capacity > UINT32_MAX / 5u || threshold == 0u ||
        threshold > capacity || wait_us == 0u || wait_us > 250000u)
      throw std::runtime_error("R12 invalid queue geometry/deadline");
  }
  void bind(const Binding& binding, uint64_t first_frame) {
    healthy();
    if (state_ != Idle || pending_ != 0u || first_frame < binding.factor_anchor ||
        binding.generation == 0u || binding.work_id.empty() || binding.template_id.empty() ||
        binding.primorial == Words{} ||
        (bound_ && (binding.generation <= binding_.generation || binding.work_id == binding_.work_id)))
      throw std::runtime_error("R12 rebind requires drained queue and fresh immutable job");
    binding_ = binding; bound_ = true; cursor_ = first_frame;
  }
  bool due(uint64_t now_us) const {
    healthy();
    if (pending_ == 0u) return false;
    if (now_us < oldest_us_) throw std::runtime_error("R12 clock moved backwards");
    return pending_ >= threshold_ || now_us - oldest_us_ >= wait_us_;
  }
  bool can_append(uint32_t upper_bound, uint64_t now_us) const {
    healthy();
    return bound_ && state_ == Idle && upper_bound <= capacity_ - pending_ && !due(now_us);
  }
  Ticket begin_append(const Binding& binding, Frame frame, uint32_t upper_bound, uint64_t now_us) {
    same_job(binding);
    if (frame.begin != cursor_ || frame.end <= frame.begin ||
        upper_bound > frame.end - frame.begin || !can_append(upper_bound, now_us))
      throw std::runtime_error("R12 append requires contiguous frame/space/deadline; flush first");
    if (serial_ == UINT64_MAX) throw std::runtime_error("R12 ticket serial overflow");
    ticket_ = {binding.generation, ++serial_, frame, pending_, upper_bound};
    state_ = Appending;
    return ticket_;
  }
  void commit_append(const Ticket& ticket, uint32_t accepted, uint64_t now_us) {
    exact_ticket(ticket, Appending);
    if (accepted > ticket.upper_bound || accepted > capacity_ - pending_ ||
        now_us > UINT64_MAX - wait_us_) {
      poisoned_ = true; throw std::runtime_error("R12 append receipt overflow");
    }
    if (pending_ != 0u && now_us < oldest_us_) {
      poisoned_ = true; throw std::runtime_error("R12 append clock regression");
    }
    if (pending_ == 0u && accepted != 0u) oldest_us_ = now_us;
    pending_ += accepted; cursor_ = ticket.frame.end; state_ = Idle;
  }
  Ticket begin_drain(const Binding& binding, uint64_t now_us, bool force = false) {
    same_job(binding);
    if (state_ != Idle || pending_ == 0u || (!force && !due(now_us)))
      throw std::runtime_error("R12 drain not ready");
    if (serial_ == UINT64_MAX) throw std::runtime_error("R12 ticket serial overflow");
    ticket_ = {binding.generation, ++serial_, {cursor_, cursor_}, pending_, pending_};
    state_ = Draining;
    return ticket_;
  }
  void commit_drain(const Ticket& ticket, uint32_t processed) {
    exact_ticket(ticket, Draining);
    if (processed != pending_ || processed > UINT64_MAX - drained_) {
      poisoned_ = true; throw std::runtime_error("R12 incomplete drain receipt");
    }
    drained_ += processed; pending_ = 0u; oldest_us_ = 0u; state_ = Idle;
  }
  void fail() { poisoned_ = true; }
  bool drained() const { healthy(); return state_ == Idle && pending_ == 0u; }
  uint32_t pending() const { return pending_; }
  uint64_t cursor() const { return cursor_; }
  uint64_t drained_count() const { return drained_; }
  uint64_t deadline_us() const { healthy(); return pending_ == 0u ? 0u : oldest_us_ + wait_us_; }
 private:
  enum State { Idle, Appending, Draining };
  uint32_t capacity_, threshold_, pending_ = 0u;
  uint64_t wait_us_, oldest_us_ = 0u, cursor_ = 0u, serial_ = 0u, drained_ = 0u;
  bool bound_ = false, poisoned_ = false;
  State state_ = Idle;
  Binding binding_{};
  Ticket ticket_{};
  void healthy() const { if (poisoned_) throw std::runtime_error("R12 poisoned transaction"); }
  void same_job(const Binding& binding) const {
    healthy();
    if (!bound_ || !(binding == binding_)) throw std::runtime_error("R12 stale/mutated job binding");
  }
  void exact_ticket(const Ticket& ticket, State expected) const {
    healthy();
    if (state_ != expected || ticket.generation != ticket_.generation ||
        ticket.serial != ticket_.serial || ticket.frame.begin != ticket_.frame.begin ||
        ticket.frame.end != ticket_.frame.end || ticket.pending_before != ticket_.pending_before ||
        ticket.upper_bound != ticket_.upper_bound)
      throw std::runtime_error("R12 stale/duplicate transaction receipt");
  }
};

struct Classification {
  uint8_t active = 0u, strict = 0u, reject_stage = 0u, prime_count = 1u;
  bool complete = false, resolved = true;
};
// Tail bits are Q1..Q6; Q0 already passed. Unknown bits are not composites.
RLOW_Q1_HD inline Classification classify(uint8_t passed, uint8_t known) {
  Classification out;
  bool prefix = true;
  for (uint32_t member = 0u; member < 6u; ++member) {
    const uint8_t bit = static_cast<uint8_t>(1u << member);
    if (prefix) {
      if ((known & bit) == 0u) { out.resolved = false; return out; }
      prefix = (passed & bit) != 0u;
      if (prefix) out.strict |= bit;
    }
  }
  for (uint32_t member = 0u; member < 6u; ++member) {
    const uint8_t bit = static_cast<uint8_t>(1u << member);
    if ((known & bit) == 0u) { out.resolved = false; return out; }
    const bool pass = (passed & bit) != 0u;
    out.prime_count = static_cast<uint8_t>(out.prime_count + (pass ? 1u : 0u));
    if ((member == 0u && !pass) || out.prime_count + (5u - member) < 5u) {
      out.reject_stage = static_cast<uint8_t>(member + 1u); return out;
    }
    out.active |= bit;
  }
  out.complete = true;
  return out;
}

// Exact base + P*uint64_relative + offset, with explicit 1280-bit overflow.
// Two 32-bit scalar products avoid the old pack's relative<=UINT32_MAX limit.
// Destination must not alias P. On false, output is invalid and cannot enter PRP.
RLOW_Q1_HD inline bool pack_relative64(uint32_t* output, const uint32_t* base,
    const uint32_t* primorial, uint64_t relative, uint32_t offset) {
  const uint32_t low = static_cast<uint32_t>(relative);
  const uint32_t high = static_cast<uint32_t>(relative >> 32u);
  uint64_t carry = 0u;
  for (uint32_t limb = 0u; limb < kLimbs; ++limb) {
    const uint64_t value = static_cast<uint64_t>(primorial[limb]) * low + base[limb] + carry;
    output[limb] = static_cast<uint32_t>(value); carry = value >> 32u;
  }
  bool overflow = carry != 0u || (high != 0u && primorial[kLimbs - 1u] != 0u);
  carry = 0u;
  if (high != 0u)
    for (uint32_t limb = 1u; limb < kLimbs; ++limb) {
      const uint64_t value = static_cast<uint64_t>(primorial[limb - 1u]) * high + output[limb] + carry;
      output[limb] = static_cast<uint32_t>(value); carry = value >> 32u;
    }
  overflow = overflow || carry != 0u;
  carry = offset;
  for (uint32_t limb = 0u; limb < kLimbs && carry != 0u; ++limb) {
    const uint64_t value = static_cast<uint64_t>(output[limb]) + carry;
    output[limb] = static_cast<uint32_t>(value); carry = value >> 32u;
  }
  return !overflow && carry == 0u;
}

#ifdef __CUDACC__
struct DeviceQueue {
  uint64_t* factors;
  uint32_t* count;
  uint32_t capacity;
  uint32_t* error;  // shared fatal flag; never reset while pending/in flight
};
struct DeviceCounters {
  unsigned long long* tested;        // [6], physical tests only
  unsigned long long* observed_pass; // [6], NOT uncensored marginals
  unsigned long long* active;        // [6], exact min5 viability
  unsigned long long* strict;        // [6], exact all-prime prefix
  unsigned long long* rejected;      // [7], stage0 remains Q0-owned
  uint64_t* samples;                 // [7*sample_limit]
  uint32_t sample_limit;
  uint64_t* complete;
  unsigned long long* complete_count;
  uint32_t complete_capacity;
};

__global__ void construct_q1(const uint64_t* factors, uint32_t count,
    const cgbn_mem_t<1280u>* immutable_base, const cgbn_mem_t<1280u>* immutable_primorial,
    uint64_t factor_anchor, uint32_t q1_offset, cgbn_mem_t<1280u>* candidates, uint32_t* error) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  if (factors[index] < factor_anchor || !pack_relative64(candidates[index]._limbs,
      immutable_base->_limbs, immutable_primorial->_limbs, factors[index] - factor_anchor, q1_offset))
    atomicOr(error, 1u);
}

// Flags and the existing bridge's stable word/chunk prefixes determine each
// output slot. One thread owns each factor; queue count publishes only afterward.
__global__ void append_q1(const uint64_t* factors, const uint8_t* q1_flags,
    uint32_t count, const uint32_t* index_prefix, const uint32_t* chunk_offsets,
    const uint32_t* accepted_count, uint32_t pending_before, Frame frame,
    DeviceQueue queue, DeviceCounters counters) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  if (*queue.count != pending_before || pending_before > queue.capacity ||
      *accepted_count > queue.capacity - pending_before ||
      factors[index] < frame.begin || factors[index] >= frame.end || q1_flags[index] > 1u) {
    atomicOr(queue.error, 2u); return;
  }
  atomicAdd(counters.tested, 1ull);
  if (q1_flags[index] == 0u) {
    const unsigned long long sample = atomicAdd(counters.rejected + 1u, 1ull);
    if (sample < counters.sample_limit) counters.samples[counters.sample_limit + sample] = factors[index];
    return;
  }
  const uint32_t rank = chunk_offsets[index / 256u] + index_prefix[index];
  if (rank >= *accepted_count) { atomicOr(queue.error, 4u); return; }
  queue.factors[pending_before + rank] = factors[index];
  atomicAdd(counters.observed_pass, 1ull);
  atomicAdd(counters.active, 1ull);
  atomicAdd(counters.strict, 1ull);
}

__global__ void commit_append_queue(DeviceQueue queue, uint32_t pending_before,
                                    const uint32_t* accepted_count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || *queue.error != 0u) return;
  if (*queue.count != pending_before || pending_before > queue.capacity ||
      *accepted_count > queue.capacity - pending_before) { atomicOr(queue.error, 8u); return; }
  *queue.count = pending_before + *accepted_count;
}

__global__ void construct_tail(DeviceQueue queue, uint32_t expected_count,
    const cgbn_mem_t<1280u>* immutable_base, const cgbn_mem_t<1280u>* immutable_primorial,
    uint64_t factor_anchor, const uint32_t* member_offsets6,
    cgbn_mem_t<1280u>* members) {
  const uint32_t output = blockIdx.x * blockDim.x + threadIdx.x;
  if (expected_count > queue.capacity || expected_count > UINT32_MAX / 5u ||
      *queue.count != expected_count) { if (output == 0u) atomicOr(queue.error, 16u); return; }
  if (output >= expected_count * 5u) return;
  const uint64_t factor = queue.factors[output / 5u];
  if (factor < factor_anchor || !pack_relative64(members[output]._limbs,
      immutable_base->_limbs, immutable_primorial->_limbs, factor - factor_anchor,
      member_offsets6[output % 5u + 1u])) atomicOr(queue.error, 32u);
}

__global__ void reduce_tail(const uint8_t* verdicts5, uint32_t expected_count,
                            DeviceQueue queue, DeviceCounters counters) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= expected_count) return;
  if (*queue.count != expected_count || expected_count > queue.capacity) {
    atomicOr(queue.error, 64u); return;
  }
  uint8_t mask = 1u;  // Q1 was actually tested and passed before admission.
  for (uint32_t member = 1u; member < 6u; ++member) {
    if (verdicts5[index * 5u + member - 1u] > 1u) { atomicOr(queue.error, 512u); return; }
    atomicAdd(counters.tested + member, 1ull);
    if (verdicts5[index * 5u + member - 1u] != 0u) {
      mask |= static_cast<uint8_t>(1u << member);
      atomicAdd(counters.observed_pass + member, 1ull);
    }
  }
  const Classification result = classify(mask, 63u);
  for (uint32_t member = 1u; member < 6u; ++member) {
    if ((result.active & (1u << member)) != 0u) atomicAdd(counters.active + member, 1ull);
    if ((result.strict & (1u << member)) != 0u) atomicAdd(counters.strict + member, 1ull);
  }
  if (!result.complete) {
    const uint32_t stage = result.reject_stage;
    const unsigned long long sample = atomicAdd(counters.rejected + stage, 1ull);
    if (sample < counters.sample_limit)
      counters.samples[static_cast<size_t>(stage) * counters.sample_limit + sample] = queue.factors[index];
  } else {
    const unsigned long long output = atomicAdd(counters.complete_count, 1ull);
    if (output >= counters.complete_capacity) atomicOr(queue.error, 128u);
    else counters.complete[output] = queue.factors[index];
  }
}

__global__ void commit_drain_queue(DeviceQueue queue, uint32_t expected_count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || *queue.error != 0u) return;
  if (*queue.count != expected_count) { atomicOr(queue.error, 256u); return; }
  *queue.count = 0u;
}
#endif
}  // namespace rlow_q1_tail
#undef RLOW_Q1_HD

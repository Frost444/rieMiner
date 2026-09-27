#pragma once
#include <cstdint>
#include <stdexcept>

// Lossless host descriptor for a device-resident FIFO of Q0-pass factors.
// It changes only when Q1 is executed, never which factors are admitted.
namespace riecoin_q1_frame_batch {
struct Span {
  std::uint64_t begin = 0, end = 0;
  std::uint32_t count = 0;
};
class Batch {
 public:
  Batch(std::uint32_t capacity, std::uint32_t threshold,
        std::uint64_t wait_us, std::uint64_t anchor)
      : capacity_(capacity), threshold_(threshold), wait_us_(wait_us),
        span_{anchor, anchor, 0} {
    if (!capacity || !threshold || threshold > capacity || !wait_us)
      throw std::invalid_argument("Q1 batch invalid limits");
  }
  void validate(std::uint64_t begin, std::uint64_t end,
                std::uint32_t count) const {
    if (begin != span_.end || end <= begin || count > end - begin ||
        count > capacity_)
      throw std::runtime_error("Q1 batch invalid/noncontiguous frame");
  }
  bool needs_room(std::uint32_t count) const {
    if (count > capacity_) throw std::runtime_error("Q1 frame exceeds capacity");
    return count > capacity_ - span_.count;
  }
  std::uint32_t copy_offset() const { return span_.count; }
  bool has_span() const { return span_.end != span_.begin; }
  Span snapshot() const { return span_; }
  std::uint64_t oldest_us() const { return oldest_us_; }
  void append(std::uint64_t begin, std::uint64_t end,
              std::uint32_t count, std::uint64_t now_us) {
    validate(begin, end, count);
    if (needs_room(count)) throw std::runtime_error("Q1 batch must drain first");
    if (span_.count && now_us < oldest_us_)
      throw std::runtime_error("Q1 batch nonmonotonic clock");
    if (!span_.count && count) oldest_us_ = now_us;
    span_.count += count;
    span_.end = end;
  }
  bool due(std::uint64_t now_us) const {
    if (!has_span()) return false;
    if (!span_.count || span_.count >= threshold_) return true;
    if (now_us < oldest_us_) throw std::runtime_error("Q1 batch clock reversed");
    return now_us - oldest_us_ >= wait_us_;
  }
  void commit(const Span& consumed) {
    if (!has_span() || consumed.begin != span_.begin || consumed.end != span_.end ||
        consumed.count != span_.count)
      throw std::runtime_error("Q1 batch partial/stale drain receipt");
    span_.begin = span_.end;
    span_.count = 0;
    oldest_us_ = 0;
  }
 private:
  std::uint32_t capacity_, threshold_;
  std::uint64_t wait_us_, oldest_us_ = 0;
  Span span_;
};
}  // namespace riecoin_q1_frame_batch

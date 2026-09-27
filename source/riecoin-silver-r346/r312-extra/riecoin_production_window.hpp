#pragma once

#include <cmath>
#include <cstdint>

// A time limit is not authority to cross a nonce reservation. Exhaustion is a
// normal boundary ONLY after the complete reserved interval has been consumed;
// the caller must then drain queues and run its unchanged exactness checks.
namespace riecoin_production_window {
enum class Next { invalid, batch, drain };

inline Next next(std::uint64_t begin, std::uint64_t cursor,
                 std::uint64_t end, std::uint64_t bundle) noexcept {
  if (bundle == 0 || begin >= end || cursor < begin || cursor > end)
    return Next::invalid;
  if ((end - begin) % bundle != 0 || (cursor - begin) % bundle != 0)
    return Next::invalid; // a partial bundle is NOT silently dropped
  return cursor == end ? Next::drain : Next::batch;
}

inline bool complete(double measured_seconds, double requested_seconds,
                     bool reservation_exhausted, std::uint64_t begin,
                     std::uint64_t consumed_end, std::uint64_t reserved_end) noexcept {
  if (!std::isfinite(measured_seconds) || measured_seconds <= 0 ||
      !std::isfinite(requested_seconds) || requested_seconds <= 0 ||
      begin >= consumed_end || consumed_end > reserved_end)
    return false;
  if (reservation_exhausted && consumed_end != reserved_end) return false;
  return measured_seconds >= requested_seconds || reservation_exhausted;
}
} // namespace riecoin_production_window

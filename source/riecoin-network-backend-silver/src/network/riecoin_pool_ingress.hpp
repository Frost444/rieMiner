#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace riecoin::network {

// A submission barrier for a nonblocking, newline-framed pool connection.
// Reader returns bytes, -1 for would-block, or 0 for a closed connection.
// Only an empty socket AND an empty partial-frame buffer release submissions.
// A fragmented clean_jobs notification must not be overtaken by a proof.
class PoolIngress {
 public:
  static constexpr std::size_t max_frame_bytes = 1024U * 1024U;
  static constexpr std::size_t default_turn_bytes = 256U * 1024U;

  template <class Reader, class Handler>
  bool drain(Reader&& read, Handler&& handle,
             std::size_t turn_bytes = default_turn_bytes) {
    if (turn_bytes == 0U) throw std::invalid_argument("empty pool receive budget");
    std::array<char, 8192> buffer{};
    std::size_t consumed = 0U;
    while (consumed < turn_bytes) {
      const auto capacity = std::min(buffer.size(), turn_bytes - consumed);
      const std::ptrdiff_t count = read(buffer.data(), capacity);
      if (count == -1) return framing_.empty();
      if (count == 0) throw std::runtime_error("pool connection closed");
      if (count < -1 || static_cast<std::size_t>(count) > capacity)
        throw std::runtime_error("invalid pool receive result");
      consumed += static_cast<std::size_t>(count);
      framing_.append(buffer.data(), static_cast<std::size_t>(count));
      std::size_t newline = 0U;
      while ((newline = framing_.find('\n')) != std::string::npos) {
        if (newline > max_frame_bytes)
          throw std::runtime_error("pool framing buffer limit exceeded");
        std::string line = framing_.substr(0U, newline);
        framing_.erase(0U, newline + 1U);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) handle(std::move(line));
      }
      if (framing_.size() > max_frame_bytes)
        throw std::runtime_error("pool framing buffer limit exceeded");
    }
    // The reader has not reported would-block yet. More notifications may be
    // waiting, so retain the proof queue and resume reception next turn.
    return false;
  }

  std::size_t buffered_bytes() const noexcept { return framing_.size(); }

 private:
  std::string framing_;
};

}  // namespace riecoin::network

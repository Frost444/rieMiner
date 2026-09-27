#pragma once

#include <boost/json/value.hpp>

#include <cstdint>
#include <chrono>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace riecoin::network {

inline constexpr const char* kEventSchema = "riecoin-network-event-v1";
inline constexpr const char* kUserAgent = "rieMinerHorizon/2609";

struct Outbound {
  std::string wire;
  std::string purpose;
  bool contains_secret{false};

  Outbound(std::string wire_value, std::string purpose_value, bool secret)
      : wire(std::move(wire_value)), purpose(std::move(purpose_value)),
        contains_secret(secret) {}
  ~Outbound() { wipe(); }
  Outbound(const Outbound&) = delete;
  Outbound& operator=(const Outbound&) = delete;
  Outbound(Outbound&& other) noexcept
      : wire(std::move(other.wire)), purpose(std::move(other.purpose)),
        contains_secret(other.contains_secret) {
    other.contains_secret = false;
  }
  Outbound& operator=(Outbound&& other) noexcept {
    if (this != &other) {
      wipe();
      wire = std::move(other.wire);
      purpose = std::move(other.purpose);
      contains_secret = other.contains_secret;
      other.contains_secret = false;
    }
    return *this;
  }

 private:
  void wipe() noexcept {
    if (!contains_secret || wire.empty()) return;
    volatile char* bytes = wire.data();
    for (std::size_t i = 0; i < wire.size(); ++i) bytes[i] = 0;
  }
};

struct WorkAssignment {
  std::string work_id;
  std::string template_id;
  std::string pool_job_id;
  std::string extra_nonce2_hex;
  std::string ntime_submit_hex;
  std::string target_hex;
  std::uint32_t target_bits{0};
  // Consensus v1: candidate - target must be strictly below 2^this value.
  // Zero is reserved for explicit local fixtures, never live allocations.
  std::uint32_t target_offset_bits{0};
  std::uint32_t primorial_number{0};
  std::uint32_t height{0};
  std::uint32_t pattern{0};
  std::uint64_t generation{0};
  std::uint64_t clean_epoch{0};
  std::uint64_t factor_origin{0};
  std::uint32_t positions{0};
};

struct Dispatch {
  std::vector<Outbound> outbound;
  std::vector<boost::json::object> events;
  bool work_available{false};
  bool invalidate_current_work{false};
};

class StratumSession {
 public:
  using Clock = std::chrono::steady_clock;
  static constexpr std::chrono::seconds submission_ack_timeout{30};
  StratumSession(std::string username, std::string password,
                 std::uint16_t difficulty_offset);
  ~StratumSession();

  StratumSession(const StratumSession&) = delete;
  StratumSession& operator=(const StratumSession&) = delete;

  Dispatch start();
  Dispatch ingest(const std::string& line, Clock::time_point now = Clock::now());

  bool authorized() const noexcept { return authorized_; }
  bool subscribed() const noexcept { return subscribed_; }
  bool fatal() const noexcept { return fatal_; }
  bool has_template() const noexcept { return template_.has_value(); }
  std::uint64_t pending_submissions() const noexcept;
  bool submissions_expired(Clock::time_point now = Clock::now()) const noexcept;
  bool reconnect_allowed() const noexcept { return pending_submissions() == 0U; }
  std::vector<boost::json::object> unknown_submission_outcomes(
      const std::string& reason, Clock::time_point now = Clock::now()) const;
  std::uint16_t extra_nonce2_length() const noexcept {
    return extra_nonce2_length_;
  }

  WorkAssignment allocate_work(const std::vector<std::uint8_t>& extra_nonce2,
                               std::uint64_t factor_origin,
                               std::uint32_t positions);
  Dispatch submit(const WorkAssignment& work,
                  const std::string& nonce_v1_uint256_hex,
                  std::uint32_t prime_count, Clock::time_point now = Clock::now());

  boost::json::object status_event(const std::string& phase) const;

 private:
  enum class PendingKind { subscribe, authorize, submit };
  struct Pending {
    PendingKind kind;
    std::string work_id;
    Clock::time_point submitted_at{};
  };
  struct Template {
    std::string pool_job_id;
    std::vector<std::uint8_t> previous_hash;
    std::vector<std::uint8_t> coinbase1;
    std::vector<std::uint8_t> coinbase2;
    std::vector<std::vector<std::uint8_t>> merkle_branches;
    std::uint32_t version{0};
    std::uint32_t bits{0};
    std::uint64_t ntime{0};
    std::uint32_t height{0};
    std::uint32_t pattern{0};
    bool clean_jobs{false};
  };

  std::uint64_t next_id_{0};
  std::uint64_t next_work_id_{0};
  std::uint64_t generation_{0};
  std::uint64_t clean_epoch_{0};
  std::uint64_t accepted_{0};
  std::uint64_t rejected_{0};
  std::uint64_t stale_{0};
  std::uint64_t local_stale_{0};
  std::uint64_t errors_{0};
  std::uint64_t submitted_{0};
  std::string username_;
  std::string password_;
  std::uint16_t difficulty_offset_{0};
  bool subscribed_{false};
  bool authorized_{false};
  bool fatal_{false};
  std::vector<std::uint8_t> extra_nonce1_;
  std::uint16_t extra_nonce2_length_{0};
  std::map<std::uint64_t, Pending> pending_;
  std::optional<Template> template_;

  boost::json::object event(const std::string& phase) const;
  Dispatch protocol_error(const std::string& error_class);
};

std::vector<std::uint8_t> hex_decode(const std::string& text,
                                     std::optional<std::size_t> exact_bytes = {});
std::string hex_encode(const std::vector<std::uint8_t>& bytes);
std::string canonical_hex(const std::string& text, std::size_t exact_bytes);

}  // namespace riecoin::network

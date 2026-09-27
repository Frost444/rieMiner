#pragma once

#include "riecoin_stratum_protocol.hpp"

#include <boost/json/object.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace riecoin::network {

struct StageConfig {
  std::filesystem::path executable;
  std::filesystem::path live_state;
  std::string gpu_uuid;
  std::string gpu_pci;
  std::uint32_t device_index{0};
  std::uint32_t primorial_number{0};
  std::uint64_t primorial_offset{0};
  std::uint16_t difficulty_offset{0};
  std::uint32_t production_session_seconds{0};
  std::uint64_t production_max_batches{1};
  std::uint32_t rejected_audit_per_stage{8};
  std::vector<std::string> prefix_arguments;
};

struct ExactShare {
  std::string nonce_v1_uint256_hex;
  std::uint32_t prime_count{0};
  // True only when this exact record was already durably admitted by the live
  // reader. Terminal reconciliation must not enqueue or submit it a second time.
  bool previously_admitted{false};
};

struct StageOutcome {
  WorkAssignment work;
  std::vector<ExactShare> shares;
  boost::json::object terminal_event;
};

struct StageProof {
  WorkAssignment work;
  ExactShare share;
  boost::json::object admission_event;
};

// The same verifier serves live admissions and terminal replay. Its cache is
// private: callers cannot manufacture a cached proof to bypass GMP. A missing
// live watermark uses complete() only, preserving old stage/Silver semantics.
class StageJournalReader {
 public:
  StageJournalReader(StageConfig config, WorkAssignment work,
      std::filesystem::path journal, std::filesystem::path admissions = {});
  ~StageJournalReader();
  StageJournalReader(const StageJournalReader&) = delete;
  StageJournalReader& operator=(const StageJournalReader&) = delete;
  std::optional<ExactShare> admit_next(std::uint64_t committed_count);
  void verify_admitted_prefix() const;
  StageOutcome complete(const std::filesystem::path& summary,
                        bool publish_new_admissions = true,
                        bool operator_drain_requested = false);
  const std::string& runtime_sha256() const noexcept;
  std::uint64_t admitted_count() const noexcept;
  std::uint64_t replay_count() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

enum class StageRetryDisposition { cancelled, completed };
// A pre-existing crash (including an external kill) is never a retryable pool
// invalidation. Kept pure so races and unconfirmed waits have closed fixtures.
StageRetryDisposition stage_retry_disposition(bool stop_confirmed,
    bool own_termination_succeeded, std::uint32_t exit_code);
// Inspect already-written guard/error evidence, not whether unfinished work
// passed every guard. Partial results remain unvalidated and never submitted.
void require_no_stage_failure_evidence(const std::string& line);
std::string stage_executable_sha256(const std::filesystem::path& path);

class StageBridge {
 public:
  explicit StageBridge(StageConfig config);
  ~StageBridge();
  StageBridge(const StageBridge&) = delete;
  StageBridge& operator=(const StageBridge&) = delete;

  void start(const WorkAssignment& work);
  bool running() const noexcept;
  void request_operator_drain();
  std::vector<boost::json::object> drain_progress_events();
  std::optional<StageProof> poll_exact_proof();
  std::optional<StageOutcome> poll();
  std::optional<boost::json::object> cancel(const std::string& reason,
                                           bool require_retry_proof = false);

  static StageOutcome validate_completed_journal(
      const StageConfig& config, const WorkAssignment& work,
      const std::filesystem::path& journal,
      const std::filesystem::path& summary = {});

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

std::string make_nonce_v1_hex(std::uint16_t primorial_number,
                              std::uint64_t factor,
                              std::uint64_t primorial_offset,
                              std::uint16_t difficulty_offset);
std::string reconstruct_candidate_hex(const std::string& target_hex,
                                      std::uint32_t primorial_number,
                                      std::uint64_t factor,
                                      std::uint64_t primorial_offset);
// Select geometry for the entire reserved range, not just its first candidate.
// These setup/rare-result checks do not add primality work to the hot path.
std::uint32_t select_consensus_primorial(const WorkAssignment& work,
                                       std::uint32_t maximum_primorial_number,
                                       std::uint64_t primorial_offset,
                                       std::uint64_t factor_end);
void require_consensus_factor_window(const WorkAssignment& work,
                                     std::uint32_t primorial_number,
                                     std::uint64_t primorial_offset,
                                     std::uint64_t factor_end);
void require_consensus_nonce(const WorkAssignment& work,
                             const std::string& nonce_hex,
                             std::uint16_t difficulty_offset);
// Relay measured stage counters without converting prefix survival into work
// that the physical executor did not perform. This never reports acceptance.
void append_stage_observability(boost::json::object& event,
                                const boost::json::object& summary);
// Parse physical/censored member telemetry. R12 never falls back to six tests
// per Q0 pass when its explicit counters are missing or malformed.
void append_live_member_observability(
    boost::json::object& event,
    const std::map<std::string, std::string>& fields);
// Live rate includes elapsed bridge wall time, not only the CUDA hot window.
void append_live_wall_rate(boost::json::object& event,
                          std::uint64_t entered,
                          double component_candidates_s,
                          double bridge_elapsed_s);

}  // namespace riecoin::network

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

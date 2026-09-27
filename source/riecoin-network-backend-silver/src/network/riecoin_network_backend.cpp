#include "riecoin_stratum_protocol.hpp"
#include "riecoin_stage_bridge.hpp"
#include "riecoin_pool_ingress.hpp"

#include <boost/json.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <bcrypt.h>
#else
#include "rc2_posix.hpp"
#define _popen popen
#define _pclose pclose
#endif

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using boost::multiprecision::cpp_int;
using riecoin::network::Dispatch;
using riecoin::network::Outbound;
using riecoin::network::StratumSession;
using riecoin::network::StageBridge;
using riecoin::network::StageConfig;
using riecoin::network::WorkAssignment;

namespace {

// Payout and endpoint are user-owned configuration, cross-checked against the credential file.

std::optional<fs::path> g_protocol_event_journal;

struct PoolConfig {
  std::string host;
  std::uint16_t port{0};
  std::string payout_address;
  fs::path credential_source;
  std::string gpu_uuid;
  std::string gpu_pci;
  bool network_enabled{false};
  std::string observability_gate;
};

struct Credentials {
  std::string username;
  std::string password;
  ~Credentials() {
    if (!password.empty()) SecureZeroMemory(password.data(), password.size());
  }
};

struct Options {
  fs::path config{"C:/Gapcoin/Riecoin-CUDA/config/riecoin-pool.json"};
  fs::path stage_exe;
  fs::path live_state{"C:/Gapcoin/Riecoin-CUDA/state/live-network"};
  std::uint32_t positions{65536U};
  std::uint32_t primorial_number{114U};
  std::uint64_t primorial_offset{114023297140211ULL};
  std::uint16_t difficulty_offset{0U};
  std::uint32_t device_index{0U};
  std::uint32_t production_session_seconds{0U};
  std::uint64_t production_max_batches{1U};
  std::uint32_t rejected_audit_per_stage{8U};
  std::uint64_t factor_origin{0U};
  std::uint64_t live_stage_limit{0U};
  std::string control_token;
  bool live{false};
};

class FatalNetworkError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// An owned UI handoff drains the current stage and its ACKs; it never kills
// a worker merely because a sampled pending counter happened to be zero.
class GracefulStop {
 public:
  using Clock = std::chrono::steady_clock;
  bool requested() const noexcept { return !request_id_.empty(); }
  const std::string& request_id() const noexcept { return request_id_; }
  bool ready(bool stage_running, std::uint64_t pending) const noexcept {
    return requested() && !stage_running && pending == 0U;
  }
  bool expired(Clock::time_point now) const noexcept {
    return requested() && now - started_ >= std::chrono::seconds(120);
  }
  bool observe(const std::string& text, std::uint64_t pid,
               const std::string& token, Clock::time_point now) {
    if (requested() || token.empty() || text.size() > 4096U) return false;
    try {
      const auto value = boost::json::parse(text);
      const auto& object = value.as_object();
      if (object.at("schema").as_string() != "riecoin-graceful-stop-v1" ||
          object.at("action").as_string() != "drain" ||
          object.at("backend_pid").to_number<std::uint64_t>() != pid ||
          object.at("control_token").as_string() != token)
        return false;
      const std::string id(object.at("request_id").as_string().c_str());
      if (id.empty() || id.size() > 128U ||
          !std::all_of(id.begin(), id.end(), [](unsigned char c) {
            return std::isalnum(c) || c == '-' || c == '_' || c == ':';
          })) return false;
      request_id_ = id;
      started_ = now;
      return true;
    } catch (const std::exception&) { return false; }
  }
  bool poll_file(const fs::path& path, const std::string& token) {
    if (requested() || token.empty()) return false;
    std::error_code error;
    const auto bytes = fs::file_size(path, error);
    if (error || bytes == 0U || bytes > 4096U) return false;
    std::ifstream input(path, std::ios::binary);
    std::string text(static_cast<std::size_t>(bytes), '\0');
    if (!input.read(text.data(), static_cast<std::streamsize>(bytes))) return false;
    return observe(text, GetCurrentProcessId(), token, Clock::now());
  }
 private:
  std::string request_id_;
  Clock::time_point started_{};
};

enum class StageDispatchAction { none, start, replace };

StageDispatchAction stage_dispatch_action(const Dispatch& dispatch,
                                           bool authorized,
                                           bool has_template,
                                           bool stage_running,
                                           bool allow_stage_start = true) noexcept {
  if (!allow_stage_start || !dispatch.work_available || !authorized || !has_template)
    return StageDispatchAction::none;
  if (!stage_running) return StageDispatchAction::start;
  return dispatch.invalidate_current_work ? StageDispatchAction::replace
                                          : StageDispatchAction::none;
}

Dispatch submit_validated_proof(StratumSession& session, const WorkAssignment& work,
    const riecoin::network::ExactShare& proof,
    StratumSession::Clock::time_point now = StratumSession::Clock::now()) {
  auto dispatch = session.submit(work, proof.nonce_v1_uint256_hex, proof.prime_count, now);
  for (auto& event : dispatch.events) {
    // handle_dispatch flushes this request_id -> (work,nonce) binding before
    // touching the socket. Later ACKs keep the protocol's exact request ID.
    event["nonce_v1_uint256_hex"] = proof.nonce_v1_uint256_hex;
    event["template_id"] = work.template_id;
    event["clean_epoch"] = work.clean_epoch;
  }
  return dispatch;
}

std::string trim(std::string value) {
  const auto keep = [](unsigned char c) { return !std::isspace(c); };
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), keep));
  value.erase(std::find_if(value.rbegin(), value.rend(), keep).base(), value.end());
  return value;
}

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::string power_of_two_target(std::uint32_t bits) {
  if (bits == 0U) throw std::runtime_error("zero target width");
  static constexpr char digits[] = "1248";
  const std::uint32_t exponent = bits - 1U;
  return std::string(1U, digits[exponent & 3U]) +
         std::string(exponent / 4U, '0');
}

std::uint64_t parse_u64(const std::string& value, const char* name) {
  std::size_t end = 0;
  const auto parsed = std::stoull(value, &end, 10);
  if (end != value.size()) throw std::runtime_error(std::string("invalid ") + name);
  return parsed;
}

Options parse_options(int argc, char** argv) {
  Options result;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto need = [&](const char* name) -> std::string {
      if (++i >= argc) throw std::runtime_error(std::string("missing ") + name);
      return argv[i];
    };
    if (arg == "--live") result.live = true;
    else if (arg == "--config") result.config = need("--config");
    else if (arg == "--stage-exe") result.stage_exe = need("--stage-exe");
    else if (arg == "--live-state") result.live_state = need("--live-state");
    else if (arg == "--positions")
      result.positions = static_cast<std::uint32_t>(parse_u64(need("--positions"), "positions"));
    else if (arg == "--primorial-number")
      result.primorial_number = static_cast<std::uint32_t>(
          parse_u64(need("--primorial-number"), "primorial number"));
    else if (arg == "--primorial-offset")
      result.primorial_offset = parse_u64(need("--primorial-offset"), "primorial offset");
    else if (arg == "--difficulty-offset")
      result.difficulty_offset = static_cast<std::uint16_t>(
          parse_u64(need("--difficulty-offset"), "difficulty offset"));
    else if (arg == "--device")
      result.device_index = static_cast<std::uint32_t>(
          parse_u64(need("--device"), "device index"));
    else if (arg == "--production-session-seconds")
      result.production_session_seconds = static_cast<std::uint32_t>(
          parse_u64(need("--production-session-seconds"), "production session seconds"));
    else if (arg == "--production-max-batches")
      result.production_max_batches = parse_u64(
          need("--production-max-batches"), "production max batches");
    else if (arg == "--rejected-audit-per-stage")
      result.rejected_audit_per_stage = static_cast<std::uint32_t>(
          parse_u64(need("--rejected-audit-per-stage"), "rejected audit per stage"));
    else if (arg == "--factor-origin")
      result.factor_origin = parse_u64(need("--factor-origin"), "factor origin");
    else if (arg == "--control-token")
      result.control_token = need("--control-token");
    else if (arg == "--live-stage-limit") {
      const auto value = need("--live-stage-limit");
      if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("invalid live stage limit");
      result.live_stage_limit = parse_u64(value, "live stage limit");
    }
    else if (arg == "--help") {
      std::cout << "riecoin-network-backend --self-test | --config-check | "
                   "--live "
                   "--config PATH --stage-exe PATH [--live-state PATH] "
                   "[--positions N --primorial-number N --primorial-offset N "
                   "--difficulty-offset N --device N --factor-origin N "
                   "--production-session-seconds N --production-max-batches N "
                    "--rejected-audit-per-stage N --live-stage-limit N --control-token ID]\n";
      std::exit(0);
    } else throw std::runtime_error("unknown option");
  }
  if (static_cast<unsigned int>(result.live) != 1U)
    throw std::runtime_error("choose exactly one execution mode");
  if (result.positions == 0U || result.primorial_number == 0U ||
      result.primorial_number > 65535U || result.difficulty_offset > 2047U)
    throw std::runtime_error("invalid bounded mining geometry");
  if ((result.production_session_seconds != 0U &&
       result.production_session_seconds < 30U) ||
      result.production_max_batches == 0U ||
      result.rejected_audit_per_stage > 1024U || result.device_index > 31U)
    throw std::runtime_error("invalid bounded production configuration");
  return result;
}

boost::json::value read_json(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("required JSON file unavailable");
  std::ostringstream bytes;
  bytes << input.rdbuf();
  boost::system::error_code ec;
  auto value = boost::json::parse(bytes.str(), ec);
  if (ec || !value.is_object()) throw std::runtime_error("required JSON file invalid");
  return value;
}

PoolConfig load_pool_config(const fs::path& path) {
  const auto value = read_json(path);
  const auto& object = value.as_object();
  if (object.at("schema").as_string() != "riecoin-pool-config-v1" ||
      object.at("network").as_string() != "mainnet")
    throw std::runtime_error("pool config schema/network mismatch");
  PoolConfig result;
  result.host = std::string(object.at("host").as_string().c_str());
  const auto port = object.at("port").to_number<std::uint64_t>();
  if (port == 0U || port > 65535U) throw std::runtime_error("invalid configured pool port");
  result.port = static_cast<std::uint16_t>(port);
  result.payout_address = std::string(object.at("payout_address").as_string().c_str());
  result.gpu_uuid = std::string(object.at("gpu_uuid").as_string().c_str());
  result.gpu_pci = std::string(object.at("gpu_pci").as_string().c_str());
  result.network_enabled = object.at("network_mining_enabled").as_bool();
  result.observability_gate = std::string(object.at("observability_gate").as_string().c_str());
  const fs::path source(std::string(object.at("credential_source").as_string().c_str()));
  result.credential_source = source.is_absolute() ? source : path.parent_path() / source;
  if (result.host.empty() || result.host.size() > 253U || result.port == 0U ||
      result.payout_address.empty() || result.payout_address.size() > 128U ||
      result.host.find_first_of("\r\n\t /\\") != std::string::npos ||
      result.payout_address.find_first_of("\r\n\t ") != std::string::npos)
    throw std::runtime_error("invalid configured pool endpoint or payout address");
  return result;
}

Credentials load_credentials(const fs::path& path, const PoolConfig& config) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("protected credential source unavailable");
  std::map<std::string, std::string> values;
  std::string line;
  while (std::getline(input, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#' || line[0] == ';') continue;
    const auto equals = line.find('=');
    if (equals == std::string::npos) continue;
    std::string key = lower(trim(line.substr(0, equals)));
    std::replace(key.begin(), key.end(), '-', '_');
    values[key] = trim(line.substr(equals + 1U));
    SecureZeroMemory(line.data(), line.size());
  }
  Credentials result;
  for (const char* key : {"username", "worker", "account", "user"})
    if (result.username.empty() && values.count(key)) result.username = values[key];
  for (const char* key : {"password", "token", "secret", "pass"})
    if (result.password.empty() && values.count(key)) result.password = values[key];
  std::string bound_host;
  for (const char* key : {"host", "pool_host"})
    if (bound_host.empty() && values.count(key)) bound_host = values[key];
  std::string bound_port;
  if (values.count("port")) bound_port = values["port"];
  std::string bound_address;
  for (const char* key : {"payout_address", "payoutaddress", "address"})
    if (bound_address.empty() && values.count(key)) bound_address = values[key];
  const std::string mode = values.count("mode") ? lower(values["mode"]) : std::string{};
  for (auto& entry : values)
    if (entry.first != "username" && entry.first != "worker" &&
        entry.first != "account" && entry.first != "user")
      SecureZeroMemory(entry.second.data(), entry.second.size());
  if (result.username.empty() || result.password.empty())
    throw std::runtime_error("protected credential source has unsupported key names");
  if (mode != "pool" || bound_host != config.host ||
      parse_u64(bound_port, "credential pool port") != config.port ||
      bound_address != config.payout_address)
    throw std::runtime_error("credential source endpoint/payout binding mismatch");
  return result;
}

bool durable_protocol_event(const boost::json::object& event) {
  const auto* phase = event.if_contains("phase");
  if (phase == nullptr || !phase->is_string()) return false;
  return phase->as_string() != "q0_measure_progress";
}

void append_protocol_event(const boost::json::object& event) {
  if (!g_protocol_event_journal || !durable_protocol_event(event)) return;
  fs::create_directories(g_protocol_event_journal->parent_path());
  HANDLE file = CreateFileW(g_protocol_event_journal->c_str(), FILE_APPEND_DATA,
                            FILE_SHARE_READ, nullptr, OPEN_ALWAYS,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE)
    throw std::runtime_error("protocol event journal open failed");
  std::string line = boost::json::serialize(event) + "\n";
  DWORD written = 0;
  const bool ok = WriteFile(file, line.data(), static_cast<DWORD>(line.size()),
                            &written, nullptr) &&
                  written == line.size() && FlushFileBuffers(file);
  CloseHandle(file);
  if (!ok) throw std::runtime_error("protocol event journal append failed");
}

void emit(boost::json::object event) {
  event["backend_pid"] = static_cast<std::uint64_t>(GetCurrentProcessId());
  event["event_tick_ms"] = static_cast<std::uint64_t>(GetTickCount64());
  append_protocol_event(event);
  std::cout << boost::json::serialize(event) << '\n' << std::flush;
}

void emit_all(std::vector<boost::json::object>& events) {
  for (auto& event : events) emit(std::move(event));
}

void emit_closed_session_status(boost::json::object event) {
  event["active_pending_submissions"] = 0U;
  event["outcomes_unknown"] = event.if_contains("pending_submissions")
      ? event.at("pending_submissions") : boost::json::value(0U);
  emit(std::move(event));
}

bool send_all(SOCKET socket, const Outbound& message) {
  std::size_t sent = 0;
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(5);
  while (sent < message.wire.size()) {
    const auto remaining = std::chrono::duration_cast<std::chrono::microseconds>(
        deadline - std::chrono::steady_clock::now());
    if (remaining.count() <= 0) return false;
    fd_set writable, exceptional;
    FD_ZERO(&writable); FD_ZERO(&exceptional);
    FD_SET(socket, &writable); FD_SET(socket, &exceptional);
    timeval timeout{
        static_cast<long>(remaining.count() / 1000000),
        static_cast<long>(remaining.count() % 1000000)};
    const int ready = select(static_cast<int>(socket)+1, nullptr, &writable, &exceptional, &timeout);
    if (ready <= 0 || FD_ISSET(socket, &exceptional)) return false;
    const std::size_t request = std::min<std::size_t>(
        message.wire.size() - sent,
        static_cast<std::size_t>(std::numeric_limits<int>::max()));
    const int chunk = static_cast<int>(send(socket, message.wire.data() + sent,
                           static_cast<int>(request),
#ifdef _WIN32
                           0
#else
                           MSG_NOSIGNAL
#endif
                           ));
    if (chunk == SOCKET_ERROR) {
      const int error = WSAGetLastError();
      if (error == WSAEWOULDBLOCK || error == WSAEINTR) continue;
      return false;
    }
    if (chunk == 0) return false;
    sent += static_cast<std::size_t>(chunk);
  }
  return true;
}

std::vector<std::uint8_t> counter_bytes(cpp_int value, std::size_t width) {
  std::vector<std::uint8_t> result(width, 0U);
  for (std::size_t i = width; i-- > 0;) {
    result[i] = static_cast<std::uint8_t>((value & 255).convert_to<unsigned>());
    value >>= 8;
  }
  return result;
}

void atomic_write(const fs::path& path, const std::string& bytes) {
  fs::create_directories(path.parent_path());
  fs::path temporary = path;
  temporary += ".tmp." + std::to_string(GetCurrentProcessId()) + "." +
      std::to_string(GetTickCount64());
  HANDLE file = CreateFileW(temporary.c_str(), GENERIC_WRITE, 0, nullptr,
                            CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE)
    throw std::runtime_error("cannot create live-state transaction");
  bool written = true;
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const DWORD request = static_cast<DWORD>(std::min<std::size_t>(
        bytes.size() - offset, std::numeric_limits<DWORD>::max()));
    DWORD completed = 0;
    if (!WriteFile(file, bytes.data() + offset, request, &completed, nullptr) ||
        completed == 0U) {
      written = false;
      break;
    }
    offset += completed;
  }
  const bool flushed = written && FlushFileBuffers(file);
  CloseHandle(file);
  if (!flushed) {
    DeleteFileW(temporary.c_str());
    throw std::runtime_error("cannot durably flush live state");
  }
  if (!MoveFileExW(temporary.c_str(), path.c_str(),
                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
    throw std::runtime_error("cannot publish live state atomically");
}

struct SessionTerminalReceipt {
  boost::json::object snapshot;
  std::vector<boost::json::object> unknown_outcomes;
  bool reconnect_allowed{false};
};

SessionTerminalReceipt persist_session_before_reset(
    const fs::path& live_state, const std::string& connection_id,
    const StratumSession& session, const std::string& reason,
    bool allow_reconnect, StratumSession::Clock::time_point now = StratumSession::Clock::now()) {
  if (connection_id.empty() ||
      connection_id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_") !=
          std::string::npos)
    throw FatalNetworkError("invalid session receipt identity");
  SessionTerminalReceipt receipt;
  receipt.snapshot = session.status_event("session_ended");
  receipt.unknown_outcomes = session.unknown_submission_outcomes(reason, now);
  receipt.reconnect_allowed = allow_reconnect && session.reconnect_allowed();
  auto& snapshot = receipt.snapshot;
  snapshot["connection_id"] = connection_id;
  snapshot["error_class"] = reason;
  // An operator stop with every ACK resolved is not a backend failure.
  snapshot["backend_errors"] =
      reason == "operator_graceful_stop" && receipt.unknown_outcomes.empty() ? 0U : 1U;
  snapshot["terminal_status"] = receipt.unknown_outcomes.empty()
      ? "disconnected_no_pending" : "outcomes_unknown";
  snapshot["outcomes_unknown"] = receipt.unknown_outcomes.size();
  snapshot["active_pending_submissions"] = 0U;
  snapshot["reconnect_allowed"] = receipt.reconnect_allowed;
  snapshot["resubmission_allowed"] = false;
  snapshot["ack_timeout_s"] = StratumSession::submission_ack_timeout.count();
  const fs::path archive = live_state / "session-terminals" / (connection_id + ".json");
  snapshot["terminal_receipt"] = archive.string();
  boost::json::array outcomes;
  for (auto& outcome : receipt.unknown_outcomes) {
    outcome["connection_id"] = connection_id;
    outcome["terminal_receipt"] = archive.string();
    outcomes.push_back(outcome);
  }
  snapshot["unknown_submission_outcomes"] = std::move(outcomes);
  if (fs::exists(archive)) throw FatalNetworkError("session terminal receipt already exists");
  const std::string encoded = boost::json::serialize(snapshot) + "\n";
  // Immutable receipt first; then the compact authoritative active snapshot.
  // Both writes flush and atomically publish before callers reset the session.
  atomic_write(archive, encoded);
  atomic_write(live_state / "session-active.json", encoded);
  return receipt;
}

void emit_session_terminal(SessionTerminalReceipt& receipt) {
  emit(receipt.snapshot);
  emit_all(receipt.unknown_outcomes);
}

class RangeLedger {
 public:
  explicit RangeLedger(fs::path path, std::uint64_t initial_factor_origin = 0U)
      : path_(std::move(path)), initial_factor_origin_(initial_factor_origin) {
#ifdef _WIN32
    mutex_ = CreateMutexW(nullptr, FALSE,
                          L"Local\\RiecoinNetworkRangeLedgerV1");
#else
    mutex_ = rc2_open_lock(path_);
#endif
    if (mutex_ == nullptr) throw std::runtime_error("range ledger mutex unavailable");
  }
  ~RangeLedger() { if (mutex_) CloseHandle(mutex_); }
  RangeLedger(const RangeLedger&) = delete;
  RangeLedger& operator=(const RangeLedger&) = delete;

  std::pair<std::vector<std::uint8_t>, std::uint64_t> reserve(
      std::size_t nonce_width, std::uint64_t positions) {
    const DWORD wait = WaitForSingleObject(mutex_, 10000U);
    if (wait != WAIT_OBJECT_0 && wait != WAIT_ABANDONED)
      throw std::runtime_error("range ledger lock timeout");
    struct Unlock {
      HANDLE handle;
      ~Unlock() { ReleaseMutex(handle); }
    } unlock{mutex_};
    cpp_int nonce_counter = 0;
    std::uint64_t factor_origin = initial_factor_origin_;
    if (fs::exists(path_)) {
      const auto state = read_json(path_).as_object();
      if ((state.at("schema").as_string() != "riecoin-live-range-ledger-v1" &&
           state.at("schema").as_string() != "riecoin-live-range-ledger-v2") ||
          state.at("nonce_width").to_number<std::uint64_t>() != nonce_width)
        throw std::runtime_error("range ledger schema/nonce width mismatch");
      nonce_counter = cpp_int(std::string(state.at("next_extra_nonce2").as_string().c_str()));
      // A fresh extranonce2 creates a fresh header/target. Factors are LOCAL
      // to that target, not a second ever-growing global namespace. Preserve
      // the durable nonce counter on v1 migration, never its invalid origin.
    } else {
      std::vector<std::uint8_t> seed(nonce_width, 0U);
#ifdef _WIN32
      if (BCryptGenRandom(nullptr, seed.data(), static_cast<ULONG>(seed.size()),
                          BCRYPT_USE_SYSTEM_PREFERRED_RNG) < 0)
#else
      if(seed.size()>static_cast<std::size_t>(std::numeric_limits<int>::max())||
         RAND_bytes(seed.data(),static_cast<int>(seed.size()))!=1)
#endif
        throw std::runtime_error("secure nonce seed unavailable");
      seed[0] &= 0x7fU;  // Reserve at least half the namespace for monotonic use.
      for (const auto byte : seed) { nonce_counter <<= 8; nonce_counter += byte; }
    }
    const cpp_int modulus = cpp_int(1) << (8U * nonce_width);
    if (nonce_counter < 0 || nonce_counter >= modulus)
      throw std::runtime_error("extraNonce2 namespace exhausted");
    if (factor_origin > std::numeric_limits<std::uint64_t>::max() - positions)
      throw std::runtime_error("factor namespace exhausted");
    const auto nonce = counter_bytes(nonce_counter, nonce_width);
    const cpp_int next_nonce = nonce_counter + 1;
    boost::json::object next{{"schema", "riecoin-live-range-ledger-v2"},
                             {"nonce_width", nonce_width},
                             {"next_extra_nonce2", next_nonce.str()},
                             {"next_factor_origin", initial_factor_origin_},
                             {"factor_scope", "unique_extra_nonce2_target"},
                             {"last_reserved_positions", positions}};
    atomic_write(path_, boost::json::serialize(next) + "\n");
    return {nonce, factor_origin};
  }

 private:
  fs::path path_;
  std::uint64_t initial_factor_origin_{0U};
  HANDLE mutex_{nullptr};
};

class E2eWindow {
 public:
  std::optional<boost::json::object> add(
      const boost::json::object& batch,
      const boost::json::object& protocol_status) {
    const auto bits = batch.at("target_bits").to_number<std::uint64_t>();
    if (target_bits_ != 0U && target_bits_ != bits) reset();
    target_bits_ = bits;
    const double wall = batch.at("batch_wall_s").to_number<double>();
    const auto entered = batch.at("q0_entered").to_number<std::uint64_t>();
    const auto passed = batch.at("q0_passed").to_number<std::uint64_t>();
    const double component =
        batch.at("component_q0_candidates_s").to_number<double>();
    wall_seconds_ += wall;
    positions_ += batch.at("positions").to_number<std::uint64_t>();
    q0_entered_ += entered;
    q0_passed_ += passed;
    exact_results_ += batch.at("exact_results").to_number<std::uint64_t>();
    if (component > 0.0) component_seconds_ += static_cast<double>(entered) / component;
    ++batches_;
    if (wall_seconds_ < 30.0) return {};

    const double e2e_c_s = wall_seconds_ > 0.0
        ? static_cast<double>(q0_entered_) / wall_seconds_ : 0.0;
    boost::json::object summary{
        {"schema", "riecoin-network-summary-v1"},
        {"measurement_layer", "completed_stage_wall_pre_submit"},
        {"rate_scope", "completed_stage_wall_excludes_connection_interruptions_and_acks"},
        {"phase", "network_summary"}, {"target_bits", target_bits_},
        {"e2e_c_s", e2e_c_s},
        {"component_q0_candidates_s", component_seconds_ > 0.0
            ? boost::json::value(static_cast<double>(q0_entered_) / component_seconds_)
            : boost::json::value(nullptr)},
        {"q0_entered", q0_entered_}, {"q0_passed", q0_passed_},
        {"positions", positions_}, {"exact_results", exact_results_},
        {"accepted", protocol_status.at("accepted")},
        {"rejected", protocol_status.at("rejected")},
        {"stale", protocol_status.at("stale")},
        {"errors", protocol_status.at("errors")},
        {"window", boost::json::object{{"measured_s", wall_seconds_},
                                        {"batches", batches_}}},
        {"conservation", boost::json::object{
            {"range_positions", positions_}, {"q0_entered", q0_entered_},
            {"q0_passed", q0_passed_}, {"validated_results", exact_results_},
            {"pass", true}}},
        {"terminal_status", "pass"}};
    if (q0_passed_ > 0U) {
      const double measured_r = static_cast<double>(q0_entered_) /
                                static_cast<double>(q0_passed_);
      summary["r"] = measured_r;
      summary["r_defined"] = true;
      summary["b_d"] = 86400.0 * e2e_c_s / std::pow(measured_r, 7.0);
    } else {
      summary["r"] = nullptr;
      summary["r_defined"] = false;
      summary["b_d"] = nullptr;
    }
    reset();
    return summary;
  }

 private:
  void reset() {
    target_bits_ = 0U;
    wall_seconds_ = 0.0;
    component_seconds_ = 0.0;
    positions_ = q0_entered_ = q0_passed_ = exact_results_ = batches_ = 0U;
  }
  std::uint64_t target_bits_{0};
  std::uint64_t positions_{0};
  std::uint64_t q0_entered_{0};
  std::uint64_t q0_passed_{0};
  std::uint64_t exact_results_{0};
  std::uint64_t batches_{0};
  double wall_seconds_{0.0};
  double component_seconds_{0.0};
};

bool stage_supports_network_batch(const fs::path& stage) {
  if (stage.empty() || !fs::is_regular_file(stage)) return false;
  // Runtime DLLs are deployed beside the stage; the launcher supplies only bundle paths.
#ifdef _WIN32
  const std::string command = '"' + stage.string() + "\" --help 2>&1";
#else
  // Quote a POSIX shell argument, including apostrophes; no user string is code.
  std::string command="'";
  for(char c:stage.string()) command += c=='\'' ? "'\"'\"'" : std::string(1,c);
  command += "' --help 2>&1";
#endif
  FILE* pipe = _popen(command.c_str(), "r");
  if (pipe == nullptr) return false;
  std::string output;
  char buffer[512];
  while (fgets(buffer, sizeof(buffer), pipe)) {
    output += buffer;
    if (output.size() > 64U * 1024U) break;
  }
  const int exit_code = _pclose(pipe);
  return exit_code == 0 && output.find("--factor-origin") != std::string::npos &&
         output.find("--network-batch") != std::string::npos;
}

void publish_dispatch(const Options& options, const PoolConfig& config,
                      const WorkAssignment& work) {
  boost::json::object dispatch{
      {"schema", "riecoin-live-stage-dispatch-v1"},
      {"phase", "work_dispatched"}, {"work_id", work.work_id},
      {"template_id", work.template_id}, {"height", work.height},
      {"pool_job_id", work.pool_job_id},
      {"extra_nonce2_hex", work.extra_nonce2_hex},
      {"ntime_submit_hex", work.ntime_submit_hex},
      {"generation", work.generation}, {"clean_epoch", work.clean_epoch},
      {"pattern", work.pattern}, {"target_hex", work.target_hex},
      {"target_bits", work.target_bits}, {"factor_origin", work.factor_origin},
      {"target_offset_bits", work.target_offset_bits},
      {"primorial_number", work.primorial_number},
      {"positions", work.positions},
      {"gpu_uuid", config.gpu_uuid}, {"gpu_pci", config.gpu_pci},
      {"exactness", "waiting"}, {"terminal_status", "running"}};
  const fs::path path = options.live_state / "dispatch" /
                        fs::path(work.work_id + ".json");
  atomic_write(path, boost::json::serialize(dispatch) + "\n");
  emit(std::move(dispatch));
}

void start_next_batch(StratumSession& session, RangeLedger& ledger,
                       StageBridge& stage, const Options& options,
                       const PoolConfig& config) {
  std::uint64_t reserved_positions = options.positions;
  if (options.production_session_seconds != 0U) {
    if (options.production_max_batches >
        std::numeric_limits<std::uint64_t>::max() / reserved_positions)
      throw std::runtime_error("production range reservation overflow");
    reserved_positions *= options.production_max_batches;
  }
  auto reservation = ledger.reserve(session.extra_nonce2_length(),
                                    reserved_positions);
  auto work = session.allocate_work(reservation.first,
                                          reservation.second,
                                          options.positions);
  if (reserved_positions > std::numeric_limits<std::uint64_t>::max() -
                               work.factor_origin)
    throw std::runtime_error("production factor reservation overflow");
  work.primorial_number = riecoin::network::select_consensus_primorial(
      work, options.primorial_number, options.primorial_offset,
      work.factor_origin + reserved_positions);
  riecoin::network::require_consensus_factor_window(work, work.primorial_number,
      options.primorial_offset, work.factor_origin + reserved_positions);
  publish_dispatch(options, config, work);
  stage.start(work);
  emit({{"schema", riecoin::network::kEventSchema}, {"phase", "stage_started"},
        {"work_id", work.work_id}, {"template_id", work.template_id},
        {"factor_origin", work.factor_origin}, {"positions", work.positions},
        {"accepted", 0}, {"rejected", 0}, {"stale", 0}, {"errors", 0},
        {"terminal_status", "running"}});
}

void handle_dispatch(SOCKET socket, Dispatch dispatch, StratumSession& session,
                     RangeLedger& ledger, StageBridge& stage,
                      const Options& options, const PoolConfig& config,
                      bool& initial_authorize_attempted,
                      bool& ever_authorized, bool allow_new_work = true) {
  const bool submission_guard_tripped = std::any_of(
      dispatch.events.begin(), dispatch.events.end(), [](const auto& event) {
        const auto* suspended = event.if_contains("submissions_suspended");
        return suspended && suspended->is_bool() && suspended->as_bool();
      });
  emit_all(dispatch.events);
  for (const auto& outbound : dispatch.outbound) {
    if (outbound.purpose == "authorize") {
      if (initial_authorize_attempted && !ever_authorized)
        throw FatalNetworkError("authorize retry blocked before first success");
      initial_authorize_attempted = true;
    }
    if (!send_all(socket, outbound)) throw std::runtime_error("pool send failed");
  }
  if (session.authorized()) ever_authorized = true;
  if (session.fatal())
    throw FatalNetworkError(submission_guard_tripped
        ? "pool rejected share; submission circuit opened"
        : "pool authorization/protocol failure");
  if (!allow_new_work) {
    // A clean notify still invalidates computation, never its in-flight ACKs.
    if (dispatch.invalidate_current_work && stage.running())
      if (auto cancelled = stage.cancel("pool_work_invalidated_during_drain"))
        emit(std::move(*cancelled));
    return;
  }
  switch (stage_dispatch_action(dispatch, session.authorized(),
                                session.has_template(), stage.running())) {
    case StageDispatchAction::start:
      start_next_batch(session, ledger, stage, options, config);
      break;
    case StageDispatchAction::replace: {
      for (auto& progress : stage.drain_progress_events()) emit(std::move(progress));
      const auto cancelled = stage.cancel("pool_work_invalidated");
      if (!cancelled || stage.running())
        throw std::runtime_error("stage remained active after invalidation");
      emit(*cancelled);  // Durable evidence precedes a fresh nonce reservation.
      start_next_batch(session, ledger, stage, options, config);
      break;
    }
    case StageDispatchAction::none:
      break;
  }
}

SOCKET connect_pool(const PoolConfig& config) {
#ifdef _WIN32
  WSADATA data{};
  if (WSAStartup(MAKEWORD(2, 2), &data) != 0)
    throw std::runtime_error("Winsock initialization failed");
#endif
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;
  addrinfo* addresses = nullptr;
  const std::string port = std::to_string(config.port);
  if (getaddrinfo(config.host.c_str(), port.c_str(), &hints, &addresses) != 0)
    throw std::runtime_error("pool DNS resolution failed");
  SOCKET result = INVALID_SOCKET;
  for (auto* address = addresses; address; address = address->ai_next) {
    result = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
    if (result != INVALID_SOCKET &&
        connect(result, address->ai_addr, static_cast<int>(address->ai_addrlen)) == 0)
      break;
    if (result != INVALID_SOCKET) closesocket(result);
    result = INVALID_SOCKET;
  }
  freeaddrinfo(addresses);
  if (result == INVALID_SOCKET) throw std::runtime_error("pool connection failed");
#ifndef _WIN32
  if(result>=FD_SETSIZE||fcntl(result,F_SETFD,FD_CLOEXEC)!=0){
    close(result);throw std::runtime_error("pool descriptor not usable");
  }
#endif
  return result;
}

int live(const Options& requested_options) {
  Options options = requested_options;
  const PoolConfig config = load_pool_config(options.config);
  if (!config.network_enabled ||
      config.observability_gate != "ready-full-horizon-network-runtime")
    throw std::runtime_error("live network gate is closed");
  if (!stage_supports_network_batch(options.stage_exe))
    throw std::runtime_error("stage lacks mandatory --network-batch/--factor-origin contract");
  g_protocol_event_journal = options.live_state / "protocol-events.jsonl";
  Credentials credentials = load_credentials(config.credential_source, config);
  RangeLedger ledger(options.live_state / "range-ledger.json",
                     options.factor_origin);
  E2eWindow e2e_window;
  std::uint32_t backoff_seconds = 1U;
  bool initial_authorize_attempted = false;
  bool ever_authorized = false;
  GracefulStop stop;
#ifndef _WIN32
  // CLI lifetime is a lease: closing/killing its console cannot orphan mining.
  const auto cli_parent_pid = getppid();
#endif
  std::uint64_t connection_number = 0U;
  for (;;) {
    SOCKET socket = INVALID_SOCKET;
    std::unique_ptr<StageBridge> stage;
    std::unique_ptr<StratumSession> protocol;
    const auto connection_begin = std::chrono::steady_clock::now();
    const std::string connection_id = std::to_string(GetCurrentProcessId()) + "-" +
        std::to_string(GetTickCount64()) + "-" + std::to_string(connection_number++);
    try {
      protocol = std::make_unique<StratumSession>(credentials.username,
          credentials.password, options.difficulty_offset);
      auto& session = *protocol;
      StageConfig stage_config;
      stage_config.executable = options.stage_exe;
      stage_config.live_state = options.live_state;
      stage_config.gpu_uuid = config.gpu_uuid;
      stage_config.gpu_pci = config.gpu_pci;
      stage_config.device_index = options.device_index;
      stage_config.primorial_number = options.primorial_number;
      stage_config.primorial_offset = options.primorial_offset;
      stage_config.difficulty_offset = options.difficulty_offset;
      stage_config.production_session_seconds = options.production_session_seconds;
      stage_config.production_max_batches = options.production_max_batches;
      stage_config.rejected_audit_per_stage = options.rejected_audit_per_stage;
      stage = std::make_unique<StageBridge>(std::move(stage_config));
      socket = connect_pool(config);
      u_long nonblocking = 1UL;
      if (ioctlsocket(socket, FIONBIO, &nonblocking) != 0)
        throw std::runtime_error("pool nonblocking mode failed");
      emit({{"schema", riecoin::network::kEventSchema}, {"phase", "connected"},
            {"host", config.host}, {"port", config.port},
            {"accepted", 0}, {"rejected", 0}, {"stale", 0}, {"errors", 0},
            {"terminal_status", "running"}});
      handle_dispatch(socket, session.start(), session, ledger, *stage,
                      options, config, initial_authorize_attempted,
                      ever_authorized);
      riecoin::network::PoolIngress ingress;
      struct PendingLiveProof {
        WorkAssignment work;
        riecoin::network::ExactShare share;
      };
      std::deque<PendingLiveProof> pending_live_proofs;
      const auto handshake_begin = std::chrono::steady_clock::now();
      auto last_pool_message = handshake_begin;
      std::optional<std::chrono::steady_clock::time_point> subscribed_at;
      std::optional<std::chrono::steady_clock::time_point> authorized_at;
      const auto receive_pool = [&]() {
        return ingress.drain(
            [&](char* data, std::size_t capacity) -> std::ptrdiff_t {
              const int bytes = recv(socket, data, static_cast<int>(capacity), 0);
              if (bytes == SOCKET_ERROR) {
                const int error = WSAGetLastError();
                if (error == WSAEWOULDBLOCK) return -1;
                throw std::runtime_error("pool receive failed; winsock=" + std::to_string(error));
              }
              if (bytes > 0) last_pool_message = std::chrono::steady_clock::now();
              return bytes;
            },
            [&](const std::string& line) {
              handle_dispatch(socket, session.ingest(line), session, ledger,
                  *stage, options, config, initial_authorize_attempted,
                  ever_authorized, !stop.requested());
            });
      };
      const auto flush_submissions = [&]() {
        for (unsigned i = 0U; i < 8U; ++i) {
          // Verification/receipt writes can outlast the preceding socket poll.
          // Consume clean_jobs before constructing/registering ANY submission.
          // On a partial frame or receive-budget boundary, proofs stay queued.
          if (!receive_pool()) return;
          std::optional<Dispatch> submission;
          if (!pending_live_proofs.empty()) {
            const auto& proof = pending_live_proofs.front();
            submission = submit_validated_proof(session, proof.work, proof.share);
            pending_live_proofs.pop_front();
          }
          if (!submission) return;
          handle_dispatch(socket, std::move(*submission), session, ledger, *stage,
              options, config, initial_authorize_attempted, ever_authorized, !stop.requested());
        }
      };
      for (;;) {
        bool parent_drain = false;
#ifndef _WIN32
        if (!stop.requested() && getppid() != cli_parent_pid) {
          parent_drain = stop.observe(boost::json::serialize(boost::json::object{
              {"schema", "riecoin-graceful-stop-v1"}, {"action", "drain"},
              {"backend_pid", static_cast<std::uint64_t>(GetCurrentProcessId())},
              {"control_token", options.control_token}, {"request_id", "cli-parent-exited"}}),
              GetCurrentProcessId(), options.control_token, GracefulStop::Clock::now());
        }
#endif
        if (parent_drain || stop.poll_file(options.live_state / "stop-request.json", options.control_token)) {
          stage->request_operator_drain();
          auto draining = session.status_event("graceful_draining");
          draining["control_request_id"] = stop.request_id();
          draining["terminal_status"] = "finishing_stage_and_pending_acks";
          emit(std::move(draining));
        }
        if (stop.expired(GracefulStop::Clock::now()))
          throw FatalNetworkError("graceful_stop_deadline_exceeded");
        if (session.submissions_expired())
          throw FatalNetworkError("submission_ack_deadline_exceeded");
        fd_set readable;
        FD_ZERO(&readable);
        FD_SET(socket, &readable);
        timeval timeout{0, 200000};
        const int ready = select(static_cast<int>(socket)+1, &readable, nullptr, nullptr, &timeout);
        if (ready == SOCKET_ERROR) throw std::runtime_error("pool select failed");
        if (ready > 0 && FD_ISSET(socket, &readable)) (void)receive_pool();
        flush_submissions();
        for (auto& progress : stage->drain_progress_events()) emit(std::move(progress));
        // Bounded replay quantum keeps notify/ACK reception live. Only complete
        // records below the child's fsync watermark may cross this boundary.
        const auto proof_turn_started = std::chrono::steady_clock::now();
        for (unsigned proof_index = 0U; proof_index < 8U; ++proof_index) {
          if (!pending_live_proofs.empty()) break;
          if (proof_index != 0U && std::chrono::steady_clock::now() - proof_turn_started >=
                                      std::chrono::milliseconds(50)) break;
          auto proof = stage->poll_exact_proof();
          if (!proof) break;
          emit(proof->admission_event);  // Per-work receipt already flushed as well.
          pending_live_proofs.push_back({std::move(proof->work), std::move(proof->share)});
          flush_submissions();
        }
        if (auto outcome = pending_live_proofs.empty() ? stage->poll() : std::nullopt) {
          emit(outcome->terminal_event);
            if (auto summary = e2e_window.add(
                    outcome->terminal_event,
                    session.status_event("protocol_status")))
              emit(std::move(*summary));
            for (const auto& share : outcome->shares) {
              if (share.previously_admitted) continue;
              pending_live_proofs.push_back({outcome->work, share});
            }
            flush_submissions();
            start_next_batch(session, ledger, *stage, options, config);
        }
        flush_submissions();
        const auto now = std::chrono::steady_clock::now();
        if (session.submissions_expired(now))
          throw FatalNetworkError("submission_ack_deadline_exceeded");
        if (stop.ready(stage->running(), session.pending_submissions()) && pending_live_proofs.empty()) {
          auto receipt = persist_session_before_reset(options.live_state, connection_id,
              session, "operator_graceful_stop", false, now);
          emit_session_terminal(receipt);
          auto stopped = receipt.snapshot;
          stopped["phase"] = "stopped";
          stopped["control_request_id"] = stop.request_id();
          stopped["terminal_status"] = "graceful_stopped";
          emit_closed_session_status(std::move(stopped));
          closesocket(socket);
          return 0;
        }
        if (session.subscribed() && !subscribed_at) subscribed_at = now;
        if (session.authorized() && !authorized_at) authorized_at = now;
        if (!session.subscribed() && now - handshake_begin > std::chrono::seconds(20))
          throw std::runtime_error("pool subscribe deadline exceeded");
        if (session.subscribed() && !session.authorized() && subscribed_at &&
            now - *subscribed_at > std::chrono::seconds(20))
          throw FatalNetworkError("pool authorize deadline exceeded");
        if (session.authorized() && !session.has_template() && authorized_at &&
            now - *authorized_at > std::chrono::seconds(60))
          throw std::runtime_error("pool first-notify deadline exceeded");
        if (now - last_pool_message > std::chrono::seconds(120))
          throw std::runtime_error("pool inactivity deadline exceeded");
      }
    } catch (const FatalNetworkError& error) {
      if (protocol) {
        auto receipt = persist_session_before_reset(options.live_state, connection_id,
            *protocol, error.what(), false);
        emit_session_terminal(receipt);
      }
      if (stage) {
        if (auto cancelled = stage->cancel("fatal_pool_error"))
          emit(std::move(*cancelled));
      }
      if (socket != INVALID_SOCKET) closesocket(socket);
      auto stopped = protocol ? protocol->status_event("stopped") : boost::json::object{
          {"schema", riecoin::network::kEventSchema}, {"phase", "stopped"},
          {"submitted", 0}, {"accepted", 0}, {"rejected", 0}, {"stale", 0},
          {"errors", 0}, {"pending_submissions", 0}};
      stopped["error_class"] = error.what();
      stopped["backend_errors"] = 1U;
      if (stopped.at("errors").to_number<std::uint64_t>() == 0U) stopped["errors"] = 1U;
      stopped["terminal_status"] = "operator_relaunch_required";
      emit_closed_session_status(std::move(stopped));
      return 1;
    } catch (const std::exception& error) {
      std::optional<SessionTerminalReceipt> receipt;
      if (protocol) {
        receipt = persist_session_before_reset(options.live_state, connection_id, *protocol, error.what(), !stop.requested());
        emit_session_terminal(*receipt);
      }
      if (stage) {
        if (auto cancelled = stage->cancel("connection_or_stage_error"))
          emit(std::move(*cancelled));
      }
      if (socket != INVALID_SOCKET) closesocket(socket);
      if (receipt && (!receipt->reconnect_allowed || stop.requested())) {
        auto stopped = receipt->snapshot;
        stopped["phase"] = "stopped";
        stopped["terminal_status"] = receipt->unknown_outcomes.empty()
            ? "operator_stop_interrupted" : "outcomes_unknown";
        if (stopped.at("errors").to_number<std::uint64_t>() == 0U) stopped["errors"] = 1U;
        emit_closed_session_status(std::move(stopped));
        return 1;
      }
      const auto stable_seconds = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now() - connection_begin).count();
      if (stable_seconds >= 60) backoff_seconds = 1U;
      auto reconnecting = receipt ? receipt->snapshot : boost::json::object{
          {"schema", riecoin::network::kEventSchema}, {"submitted", 0},
          {"accepted", 0}, {"rejected", 0}, {"stale", 0}, {"errors", 0},
          {"pending_submissions", 0}};
      reconnecting["phase"] = "reconnecting";
      reconnecting["error_class"] = error.what();
      reconnecting["backoff_s"] = backoff_seconds;
      reconnecting["backend_errors"] = 1U;
      if (reconnecting.at("errors").to_number<std::uint64_t>() == 0U) reconnecting["errors"] = 1U;
      reconnecting["terminal_status"] = "retrying";
      emit_closed_session_status(std::move(reconnecting));
      Sleep(backoff_seconds * 1000U);
      backoff_seconds = std::min<std::uint32_t>(30U, backoff_seconds * 2U);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    return live(options);
  } catch (const std::exception& error) {
    // All credential-bearing wire messages are deliberately excluded from
    // exceptions and logs. Only bounded local error classes reach this sink.
    emit({{"schema", riecoin::network::kEventSchema}, {"phase", "error"},
          {"error_class", error.what()}, {"accepted", 0}, {"rejected", 0},
          {"stale", 0}, {"errors", 1}, {"terminal_status", "FAIL"}});
    return 1;
  }
}

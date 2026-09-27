#include "riecoin_stratum_protocol.hpp"
#include "riecoin_stage_bridge.hpp"

#include <boost/json.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#else
#include "rc2_posix.hpp"
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace riecoin::network {
namespace {

using boost::multiprecision::cpp_int;

constexpr std::array<std::uint64_t, 7> kPattern0{0, 2, 4, 2, 4, 6, 2};
constexpr std::array<std::uint64_t, 7> kPattern1{0, 2, 6, 4, 2, 4, 2};

std::uint8_t hex_digit(char c) {
  if (c >= '0' && c <= '9') return static_cast<std::uint8_t>(c - '0');
  if (c >= 'a' && c <= 'f') return static_cast<std::uint8_t>(c - 'a' + 10);
  if (c >= 'A' && c <= 'F') return static_cast<std::uint8_t>(c - 'A' + 10);
  throw std::runtime_error("invalid hexadecimal field");
}

std::uint64_t parse_hex_u64(const std::string& text, std::size_t max_digits) {
  if (text.empty() || text.size() > max_digits)
    throw std::runtime_error("hex integer has invalid width");
  std::uint64_t value = 0;
  for (const char c : text) value = (value << 4U) | hex_digit(c);
  return value;
}

std::array<std::uint8_t, 32> sha256(const std::vector<std::uint8_t>& data) {
#ifdef _WIN32
  BCRYPT_ALG_HANDLE algorithm = nullptr;
  BCRYPT_HASH_HANDLE hash = nullptr;
  DWORD object_size = 0, hash_size = 0, bytes = 0;
  auto failed = [](NTSTATUS s) { return s < 0; };
  if (failed(BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM,
                                         nullptr, 0)))
    throw std::runtime_error("SHA256 provider unavailable");
  const auto close_algorithm = [&] { BCryptCloseAlgorithmProvider(algorithm, 0); };
  if (failed(BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
                               reinterpret_cast<PUCHAR>(&object_size),
                               sizeof(object_size), &bytes, 0)) ||
      failed(BCryptGetProperty(algorithm, BCRYPT_HASH_LENGTH,
                               reinterpret_cast<PUCHAR>(&hash_size),
                               sizeof(hash_size), &bytes, 0)) || hash_size != 32U) {
    close_algorithm();
    throw std::runtime_error("SHA256 provider properties invalid");
  }
  std::vector<std::uint8_t> object(object_size);
  std::array<std::uint8_t, 32> digest{};
  if (failed(BCryptCreateHash(algorithm, &hash, object.data(), object_size,
                              nullptr, 0, 0))) {
    close_algorithm();
    throw std::runtime_error("SHA256 hash creation failed");
  }
  const auto destroy_hash = [&] { BCryptDestroyHash(hash); };
  if ((!data.empty() && failed(BCryptHashData(
          hash, const_cast<PUCHAR>(data.data()),
          static_cast<ULONG>(data.size()), 0))) ||
      failed(BCryptFinishHash(hash, digest.data(),
                              static_cast<ULONG>(digest.size()), 0))) {
    destroy_hash();
    close_algorithm();
    throw std::runtime_error("SHA256 hashing failed");
  }
  destroy_hash();
  close_algorithm();
  return digest;
#else
  std::array<std::uint8_t,32> digest{};unsigned length=0;
  if(EVP_Digest(data.data(),data.size(),digest.data(),&length,EVP_sha256(),nullptr)!=1||length!=32)
    throw std::runtime_error("SHA256 hashing failed");
  return digest;
#endif
}

std::array<std::uint8_t, 32> sha256d(const std::vector<std::uint8_t>& data) {
  const auto first = sha256(data);
  return sha256(std::vector<std::uint8_t>(first.begin(), first.end()));
}

void append_le(std::vector<std::uint8_t>& out, std::uint64_t value,
               std::size_t bytes) {
  for (std::size_t i = 0; i < bytes; ++i)
    out.push_back(static_cast<std::uint8_t>(value >> (8U * i)));
}

std::string cpp_int_hex(const cpp_int& value) {
  if (value == 0) return "0";
  std::ostringstream out;
  out << std::hex << value;
  return out.str();
}

cpp_int little_endian_integer(const std::array<std::uint8_t, 32>& bytes) {
  cpp_int value = 0;
  for (std::size_t i = bytes.size(); i-- > 0;) {
    value <<= 8;
    value += bytes[i];
  }
  return value;
}

std::uint32_t select_pattern(const boost::json::array& patterns) {
  for (const auto& pattern_value : patterns) {
    if (!pattern_value.is_array()) continue;
    const auto& pattern = pattern_value.as_array();
    if (pattern.size() != kPattern0.size()) continue;
    bool p0 = true, p1 = true;
    for (std::size_t i = 0; i < pattern.size(); ++i) {
      if (!pattern[i].is_int64() && !pattern[i].is_uint64()) {
        p0 = p1 = false;
        break;
      }
      const auto item = pattern[i].to_number<std::uint64_t>();
      p0 = p0 && item == kPattern0[i];
      p1 = p1 && item == kPattern1[i];
    }
    if (p0) return 0;
    if (p1) return 1;
  }
  throw std::runtime_error("pool offered no supported mainnet pattern");
}

bool is_null_or_absent(const boost::json::object& object, const char* key) {
  const auto* value = object.if_contains(key);
  return value == nullptr || value->is_null();
}

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool response_is_stale(const boost::json::object& object) {
  const auto* error = object.if_contains("error");
  if (error == nullptr || error->is_null()) return false;
  const std::string text = lower_copy(boost::json::serialize(*error));
  return text.find("stale") != std::string::npos ||
         text.find("obsolete") != std::string::npos ||
         text.find("job not found") != std::string::npos;
}

std::string bounded_pool_text(std::string text, const std::string& secret) {
  if (!secret.empty()) {
    std::size_t at = 0;
    while ((at = text.find(secret, at)) != std::string::npos) {
      text.replace(at, secret.size(), "[redacted]");
      at += 10U;
    }
  }
  for (char& c : text)
    if (static_cast<unsigned char>(c) < 32U) c = ' ';
  if (text.size() > 384U) text.resize(384U);
  return text;
}

void append_pool_error(boost::json::object& output,
                        const boost::json::object& response,
                        const std::string& secret) {
  const auto* error = response.if_contains("error");
  if (error == nullptr || error->is_null()) {
    output["pool_error_message"] = "result_false_without_error_detail";
    return;
  }
  const boost::json::value* code = nullptr;
  const boost::json::value* message = nullptr;
  if (error->is_array()) {
    const auto& values = error->as_array();
    if (!values.empty()) code = &values[0];
    if (values.size() > 1U) message = &values[1];
  } else if (error->is_object()) {
    code = error->as_object().if_contains("code");
    message = error->as_object().if_contains("message");
  } else if (error->is_string()) {
    message = error;
  }
  if (code && (code->is_int64() || code->is_uint64()))
    output["pool_error_code"] = *code;
  if (message && message->is_string())
    output["pool_error_message"] = bounded_pool_text(
        std::string(message->as_string().c_str()), secret);
  else
    output["pool_error_message"] = "nonstandard_pool_error_shape";
}

boost::json::value parse_json(const std::string& line) {
  if (line.empty() || line.size() > 1024U * 1024U)
    throw std::runtime_error("pool JSON line has invalid size");
  boost::system::error_code ec;
  auto value = boost::json::parse(line, ec);
  if (ec || !value.is_object()) throw std::runtime_error("invalid pool JSON");
  return value;
}

std::string json_wire(boost::json::object object) {
  return boost::json::serialize(object) + "\n";
}

}  // namespace

std::vector<std::uint8_t> hex_decode(const std::string& text,
                                     std::optional<std::size_t> exact_bytes) {
  if ((text.size() & 1U) != 0U)
    throw std::runtime_error("hexadecimal field has odd width");
  if (exact_bytes && text.size() != *exact_bytes * 2U)
    throw std::runtime_error("hexadecimal field has wrong width");
  std::vector<std::uint8_t> result(text.size() / 2U);
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = static_cast<std::uint8_t>((hex_digit(text[2U * i]) << 4U) |
                                          hex_digit(text[2U * i + 1U]));
  return result;
}

std::string hex_encode(const std::vector<std::uint8_t>& bytes) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result(bytes.size() * 2U, '0');
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    result[2U * i] = digits[bytes[i] >> 4U];
    result[2U * i + 1U] = digits[bytes[i] & 15U];
  }
  return result;
}

std::string canonical_hex(const std::string& text, std::size_t exact_bytes) {
  return hex_encode(hex_decode(text, exact_bytes));
}

StratumSession::StratumSession(std::string username, std::string password,
                               std::uint16_t difficulty_offset)
    : username_(std::move(username)), password_(std::move(password)),
      difficulty_offset_(difficulty_offset) {
  if (username_.empty() || password_.empty())
    throw std::runtime_error("pool credentials are empty");
  if (difficulty_offset_ > 2047U)
    throw std::runtime_error("difficulty offset is out of nonce-v1 range");
}

StratumSession::~StratumSession() {
  if (!password_.empty()) SecureZeroMemory(password_.data(), password_.size());
}

boost::json::object StratumSession::event(const std::string& phase) const {
  return {{"schema", kEventSchema}, {"phase", phase},
          {"accepted", accepted_}, {"rejected", rejected_},
          {"stale", stale_}, {"errors", errors_},
          {"submitted", submitted_}, {"local_stale", local_stale_},
          {"pending_submissions", pending_submissions()}};
}

std::uint64_t StratumSession::pending_submissions() const noexcept {
  std::uint64_t count = 0U;
  for (const auto& entry : pending_)
    if (entry.second.kind == PendingKind::submit) ++count;
  return count;
}

bool StratumSession::submissions_expired(Clock::time_point now) const noexcept {
  for (const auto& entry : pending_)
    if (entry.second.kind == PendingKind::submit &&
        now - entry.second.submitted_at >= submission_ack_timeout)
      return true;
  return false;
}

std::vector<boost::json::object> StratumSession::unknown_submission_outcomes(
    const std::string& reason, Clock::time_point now) const {
  std::vector<boost::json::object> results;
  for (const auto& entry : pending_) {
    if (entry.second.kind != PendingKind::submit) continue;
    auto result = event("outcomes_unknown");
    result["request_id"] = entry.first;
    result["work_id"] = entry.second.work_id;
    result["terminal_status"] = "outcomes_unknown";
    result["error_class"] = reason;
    result["pool_outcome"] = "unknown";
    result["outcome_unknown"] = true;
    result["active_pending_submissions"] = 0U;
    result["backend_errors"] = 1U;
    result["resubmission_allowed"] = false;
    result["ack_timeout_s"] = submission_ack_timeout.count();
    result["submission_age_s"] =
        std::chrono::duration<double>(now - entry.second.submitted_at).count();
    result["ack_deadline_expired"] =
        now - entry.second.submitted_at >= submission_ack_timeout;
    results.push_back(std::move(result));
  }
  return results;
}

boost::json::object StratumSession::status_event(const std::string& phase) const {
  auto result = event(phase);
  result["subscribed"] = subscribed_;
  result["authorized"] = authorized_;
  result["generation"] = generation_;
  result["clean_epoch"] = clean_epoch_;
  boost::json::array pending_submissions;
  for (const auto& entry : pending_)
    if (entry.second.kind == PendingKind::submit)
      pending_submissions.push_back(boost::json::object{
          {"request_id", entry.first}, {"work_id", entry.second.work_id}});
  result["pending_submission_requests"] = std::move(pending_submissions);
  return result;
}

Dispatch StratumSession::protocol_error(const std::string& error_class) {
  ++errors_;
  fatal_ = true;
  Dispatch result;
  auto e = event("error");
  e["error_class"] = error_class;
  result.events.push_back(std::move(e));
  return result;
}

Dispatch StratumSession::start() {
  const auto id = next_id_++;
  pending_.emplace(id, Pending{PendingKind::subscribe, {}});
  boost::json::object request{{"id", id}, {"method", "mining.subscribe"},
                              {"params", boost::json::array{kUserAgent}}};
  Dispatch result;
  result.outbound.push_back({json_wire(std::move(request)), "subscribe", false});
  result.events.push_back(event("subscribe_sent"));
  return result;
}

Dispatch StratumSession::ingest(const std::string& line, Clock::time_point now) {
  // Deadline is exclusive: an ACK first processed at exactly t_submit + 30 s
  // is too late. Keep its pending ID so terminal accounting records UNKNOWN.
  // Notifications never refresh this per-submission clock.
  if (submissions_expired(now)) return protocol_error("submission_ack_deadline_exceeded");
  try {
    const auto parsed = parse_json(line);
    const auto& object = parsed.as_object();
    if (const auto* method_value = object.if_contains("method")) {
      if (!method_value->is_string()) return protocol_error("method_not_string");
      const std::string method(method_value->as_string().c_str());
      if (method == "client.show_message") {
        Dispatch result;
        auto e = event("pool_message");
        const auto* params = object.if_contains("params");
        if (params && params->is_array() && !params->as_array().empty() &&
            params->as_array().front().is_string())
          e["message"] = bounded_pool_text(
              std::string(params->as_array().front().as_string().c_str()), password_);
        result.events.push_back(std::move(e));
        return result;
      }
      if (method != "mining.notify") {
        Dispatch result;
        auto e = event("pool_method_ignored");
        e["method"] = method;
        result.events.push_back(std::move(e));
        return result;
      }
      const auto* params_value = object.if_contains("params");
      if (params_value == nullptr || !params_value->is_array())
        return protocol_error("notify_params_missing");
      const auto& p = params_value->as_array();
      if (p.size() < 11U) return protocol_error("notify_params_short");
      Template next;
      next.pool_job_id = std::string(p[0].as_string().c_str());
      next.previous_hash = hex_decode(std::string(p[1].as_string().c_str()), 32U);
      next.coinbase1 = hex_decode(std::string(p[2].as_string().c_str()));
      next.coinbase2 = hex_decode(std::string(p[3].as_string().c_str()));
      if (next.coinbase1.size() < 46U)
        throw std::runtime_error("coinbase1 too short for height field");
      if (!p[4].is_array()) throw std::runtime_error("merkle branch list invalid");
      for (const auto& branch : p[4].as_array())
        next.merkle_branches.push_back(
            hex_decode(std::string(branch.as_string().c_str()), 32U));
      next.version = static_cast<std::uint32_t>(
          parse_hex_u64(std::string(p[5].as_string().c_str()), 8U));
      next.bits = static_cast<std::uint32_t>(
          parse_hex_u64(std::string(p[6].as_string().c_str()), 8U));
      next.ntime = parse_hex_u64(std::string(p[7].as_string().c_str()), 16U);
      next.clean_jobs = p[8].as_bool();
      if (p[9].to_number<std::int64_t>() != 1)
        throw std::runtime_error("unsupported pow version");
      next.pattern = select_pattern(p[10].as_array());
      const std::uint8_t height_bytes = next.coinbase1[42];
      if (height_bytes == 0U || height_bytes > 3U)
        throw std::runtime_error("unsupported coinbase height width");
      next.height = 0;
      for (std::uint8_t i = 0; i < height_bytes; ++i)
        next.height |= static_cast<std::uint32_t>(next.coinbase1[43U + i])
                       << (8U * i);
      const bool previous_hash_changed =
          template_.has_value() && next.previous_hash != template_->previous_hash;
      const bool invalidate_current_work =
          next.clean_jobs || previous_hash_changed;
      ++generation_;
      if (invalidate_current_work) ++clean_epoch_;
      template_ = std::move(next);
      Dispatch result;
      auto e = event("work_received");
      e["template_id"] = template_->pool_job_id + ":" + std::to_string(generation_);
      e["height"] = template_->height;
      e["pattern"] = template_->pattern;
      e["clean_jobs"] = template_->clean_jobs;
      e["previous_hash_changed"] = previous_hash_changed;
      e["invalidate_current_work"] = invalidate_current_work;
      e["generation"] = generation_;
      e["terminal_status"] = "not_submitting_until_unique_factor_origin_stage";
      result.events.push_back(std::move(e));
      result.work_available = authorized_;
      result.invalidate_current_work = invalidate_current_work;
      return result;
    }

    const auto* id_value = object.if_contains("id");
    if (id_value == nullptr || id_value->is_null() ||
        (!id_value->is_int64() && !id_value->is_uint64()))
      return protocol_error("response_id_invalid");
    const auto id = id_value->to_number<std::uint64_t>();
    auto pending_it = pending_.find(id);
    // rieMiner's official local TestServer historically answers authorize with
    // id=0. Accept that single legacy shape only before authorization and only
    // when exactly one authorize request is pending; submissions remain strict.
    if (pending_it == pending_.end() && id == 0U && subscribed_ && !authorized_) {
      auto legacy = pending_.end();
      for (auto it = pending_.begin(); it != pending_.end(); ++it) {
        if (it->second.kind != PendingKind::authorize) continue;
        if (legacy != pending_.end()) { legacy = pending_.end(); break; }
        legacy = it;
      }
      pending_it = legacy;
    }
    if (pending_it == pending_.end()) return protocol_error("response_id_unknown");
    const Pending pending = pending_it->second;
    pending_.erase(pending_it);

    if (pending.kind == PendingKind::subscribe) {
      if (!is_null_or_absent(object, "error"))
        return protocol_error("subscribe_refused");
      const auto* result_value = object.if_contains("result");
      if (result_value == nullptr || !result_value->is_array())
        return protocol_error("subscribe_result_invalid");
      const auto& result_array = result_value->as_array();
      if (result_array.size() < 3U || !result_array[1].is_string())
        return protocol_error("subscribe_result_short");
      extra_nonce1_ = hex_decode(std::string(result_array[1].as_string().c_str()));
      extra_nonce2_length_ = static_cast<std::uint16_t>(
          result_array[2].to_number<std::uint64_t>());
      if (extra_nonce2_length_ == 0U || extra_nonce2_length_ > 16U)
        return protocol_error("extra_nonce2_width_unsupported");
      subscribed_ = true;
      const auto authorize_id = next_id_++;
      pending_.emplace(authorize_id, Pending{PendingKind::authorize, {}});
      boost::json::object request{
          {"id", authorize_id}, {"method", "mining.authorize"},
          {"params", boost::json::array{username_, password_}}};
      Dispatch dispatch;
      dispatch.outbound.push_back(
          {json_wire(std::move(request)), "authorize", true});
      dispatch.events.push_back(event("subscribed"));
      return dispatch;
    }

    const auto* result_value = object.if_contains("result");
    const bool success = result_value != nullptr && result_value->is_bool() &&
                         result_value->as_bool() &&
                         is_null_or_absent(object, "error");
    if (pending.kind == PendingKind::authorize) {
      if (!success) return protocol_error("authorization_refused");
      authorized_ = true;
      Dispatch dispatch;
      dispatch.events.push_back(event("authorized"));
      dispatch.work_available = template_.has_value();
      return dispatch;
    }

    Dispatch dispatch;
    if (success) {
      ++accepted_;
      auto e = event("share_accepted");
      e["work_id"] = pending.work_id;
      e["request_id"] = id;
      dispatch.events.push_back(std::move(e));
    } else if (response_is_stale(object)) {
      ++stale_;
      auto e = event("share_stale");
      e["work_id"] = pending.work_id;
      e["request_id"] = id;
      append_pool_error(e, object, password_);
      dispatch.events.push_back(std::move(e));
    } else {
      ++rejected_;
      auto e = event("share_rejected");
      e["work_id"] = pending.work_id;
      e["request_id"] = id;
      append_pool_error(e, object, password_);
      e["submissions_suspended"] = true;
      fatal_ = true;  // No reconnect loop that keeps sending invalid shares.
      dispatch.events.push_back(std::move(e));
    }
    return dispatch;
  } catch (const std::exception&) {
    return protocol_error("malformed_or_unsupported_pool_message");
  }
}

WorkAssignment StratumSession::allocate_work(
    const std::vector<std::uint8_t>& extra_nonce2,
    std::uint64_t factor_origin, std::uint32_t positions) {
  if (!authorized_ || !template_) throw std::runtime_error("no authorized work template");
  if (extra_nonce2.size() != extra_nonce2_length_)
    throw std::runtime_error("extraNonce2 allocation width mismatch");
  if (positions == 0U || factor_origin >
          std::numeric_limits<std::uint64_t>::max() - positions)
    throw std::runtime_error("factor range is empty or wraps");

  std::vector<std::uint8_t> coinbase;
  coinbase.insert(coinbase.end(), template_->coinbase1.begin(), template_->coinbase1.end());
  coinbase.insert(coinbase.end(), extra_nonce1_.begin(), extra_nonce1_.end());
  coinbase.insert(coinbase.end(), extra_nonce2.begin(), extra_nonce2.end());
  coinbase.insert(coinbase.end(), template_->coinbase2.begin(), template_->coinbase2.end());
  auto merkle = sha256d(coinbase);
  for (const auto& branch : template_->merkle_branches) {
    std::vector<std::uint8_t> pair(merkle.begin(), merkle.end());
    pair.insert(pair.end(), branch.begin(), branch.end());
    merkle = sha256d(pair);
  }

  std::vector<std::uint8_t> header;
  header.reserve(80U);
  append_le(header, template_->version, 4U);
  for (std::size_t word = 0; word < 8U; ++word)
    for (std::size_t byte = 0; byte < 4U; ++byte)
      header.push_back(template_->previous_hash[word * 4U + (3U - byte)]);
  header.insert(header.end(), merkle.begin(), merkle.end());
  append_le(header, template_->ntime, 8U);
  append_le(header, template_->bits, 4U);
  if (header.size() != 80U) throw std::runtime_error("internal header width error");
  const auto header_hash = sha256d(header);

  std::uint64_t bits64 = template_->bits;
  bits64 += static_cast<std::uint64_t>(difficulty_offset_) << 13U;
  bits64 = std::min<std::uint64_t>(bits64, 0xffffffffULL);
  const auto difficulty_integer = static_cast<std::uint32_t>(bits64 / 256U);
  if (difficulty_integer < 264U)
    throw std::runtime_error("pool difficulty cannot form Riecoin v1 target");
  const std::uint32_t df = static_cast<std::uint32_t>(bits64 & 255U);
  const std::uint64_t mantissa = 256ULL +
      (10ULL * df * df * df + 7383ULL * df * df +
       5840720ULL * df + 3997440ULL) / (1ULL << 23U);
  cpp_int target = mantissa;
  target <<= 256U;
  target += little_endian_integer(header_hash);
  target <<= (difficulty_integer - 264U);

  WorkAssignment work;
  work.extra_nonce2_hex = hex_encode(extra_nonce2);
  work.work_id = "net-" + std::to_string(generation_) + "-" +
                 work.extra_nonce2_hex + "-" +
                 std::to_string(factor_origin) + "-" +
                 std::to_string(next_work_id_++);
  work.template_id = template_->pool_job_id + ":" + std::to_string(generation_);
  work.pool_job_id = template_->pool_job_id;
  std::ostringstream ntime;
  ntime << std::hex << std::setfill('0') << std::setw(16) << template_->ntime;
  work.ntime_submit_hex = ntime.str();
  work.target_hex = cpp_int_hex(target);
  work.target_bits = static_cast<std::uint32_t>(boost::multiprecision::msb(target) + 1U);
  work.target_offset_bits = difficulty_integer - 264U;
  work.height = template_->height;
  work.pattern = template_->pattern;
  work.generation = generation_;
  work.clean_epoch = clean_epoch_;
  work.factor_origin = factor_origin;
  work.positions = positions;
  return work;
}

Dispatch StratumSession::submit(const WorkAssignment& work,
                                const std::string& nonce_v1_uint256_hex,
                                std::uint32_t prime_count, Clock::time_point now) {
  if (submissions_expired(now)) return protocol_error("submission_ack_deadline_exceeded");
  Dispatch dispatch;
  if (prime_count < 5U || canonical_hex(nonce_v1_uint256_hex, 32U) !=
                            nonce_v1_uint256_hex) {
    ++errors_;
    auto e = event("error");
    e["error_class"] = "inexact_or_invalid_share_record";
    e["work_id"] = work.work_id;
    dispatch.events.push_back(std::move(e));
    return dispatch;
  }
  if (work.clean_epoch < clean_epoch_) {
    ++stale_;
    ++local_stale_;
    auto e = event("share_stale");
    e["work_id"] = work.work_id;
    e["reason"] = "locally_invalidated_by_clean_jobs";
    dispatch.events.push_back(std::move(e));
    return dispatch;
  }
  try {
    require_consensus_nonce(work, nonce_v1_uint256_hex, difficulty_offset_);
  } catch (const std::exception& error) {
    ++errors_;
    auto e = event("error");
    e["error_class"] = "consensus_submission_guard";
    e["reason"] = error.what();
    e["work_id"] = work.work_id;
    dispatch.events.push_back(std::move(e));
    return dispatch;  // Never send a locally invalid nonce to the pool.
  }
  const auto id = next_id_++;
  pending_.emplace(id, Pending{PendingKind::submit, work.work_id, now});
  boost::json::object request{
      {"id", id}, {"method", "mining.submit"},
      {"params", boost::json::array{username_, work.pool_job_id,
                                    work.extra_nonce2_hex,
                                    work.ntime_submit_hex,
                                    nonce_v1_uint256_hex}}};
  ++submitted_;
  dispatch.outbound.push_back({json_wire(std::move(request)), "submit", true});
  auto e = event("share_submitted");
  e["work_id"] = work.work_id;
  e["request_id"] = id;
  e["prime_count"] = prime_count;
  dispatch.events.push_back(std::move(e));
  return dispatch;
}

}  // namespace riecoin::network

#include "riecoin_stage_bridge.hpp"

#include <boost/json.hpp>
#include <boost/json/basic_parser_impl.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#else
#include "rc2_posix.hpp"
#endif
#include <gmp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace fs = std::filesystem;

namespace riecoin::network {
namespace {

using boost::multiprecision::cpp_int;

constexpr std::array<std::array<std::uint32_t, 7>, 2> kOffsets{{
    {{0, 2, 6, 8, 12, 18, 20}},
    {{0, 2, 8, 12, 14, 18, 20}},
}};

constexpr const char* kQueuedOptionalOperator = "mandatory_q1_then_queued_optional5";
constexpr DWORD kCancelledStageExitCode = 0xC000013AU;
constexpr std::uintmax_t kJournalLimit = 16U * 1024U * 1024U;
constexpr std::size_t kJournalRecordLimit = 1024U * 1024U;
constexpr std::size_t kJournalReadChunk = 64U * 1024U;

bool failure_counter(const std::string& key) {
  const auto ends = [&key](const char* suffix) {
    const std::string tail(suffix);
    return key.size() >= tail.size() &&
        key.compare(key.size() - tail.size(), tail.size(), tail) == 0;
  };
  return key == "mismatches" || key == "mismatch_words" || key == "errors" ||
      key == "drops" || key == "duplicates" || key == "prime_false_negatives" ||
      ends("_mismatches") || ends("_mismatch_words") || ends("_false_negatives") ||
      ends("_errors") || ends("_drops") || ends("_duplicates");
}

bool failure_status(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value == "fail" || value == "failed" || value == "failed_closed" ||
         value == "error" || value == "mismatch";
}

std::uint64_t strict_counter(const std::string& input) {
  if (input.empty()) throw std::runtime_error("empty stage member counter");
  std::uint64_t result = 0U;
  for (const unsigned char character : input) {
    if (character < '0' || character > '9')
      throw std::runtime_error("stage member counter must be an unsigned integer");
    const std::uint64_t digit = character - '0';
    if (result > ((std::numeric_limits<std::uint64_t>::max)() - digit) / 10U)
      throw std::runtime_error("stage member counter overflow");
    result = result * 10U + digit;
  }
  return result;
}

std::uint64_t strict_json_counter(const boost::json::value& input) {
  if (input.is_uint64()) return input.as_uint64();
  if (input.is_int64() && input.as_int64() >= 0)
    return static_cast<std::uint64_t>(input.as_int64());
  throw std::runtime_error("stage member JSON counter must be an unsigned integer");
}

void inspect_failure_value(const boost::json::value& value) {
  if (value.is_object()) {
    for (const auto& field : value.as_object()) {
      const std::string key(field.key());
      const auto& item = field.value();
      if ((failure_counter(key) && strict_json_counter(item) != 0U) ||
          (key == "pass" && item.is_bool() && !item.as_bool()) ||
          (item.is_string() && failure_status(std::string(item.as_string()))))
        throw std::runtime_error("stage_invalidation_observed_failure");
      if (key == "error" && !item.is_null() &&
          !(item.is_int64() && item.as_int64() == 0) &&
          !(item.is_uint64() && item.as_uint64() == 0U))
        throw std::runtime_error("stage_invalidation_observed_error");
      inspect_failure_value(item);
    }
  } else if (value.is_array()) {
    for (const auto& item : value.as_array()) inspect_failure_value(item);
  }
}

boost::json::array parse_member_csv(const std::string& input) {
  boost::json::array result;
  std::size_t start = 0U;
  for (unsigned member = 0U; member < 6U; ++member) {
    const auto comma = input.find(',', start);
    if ((member < 5U) != (comma != std::string::npos))
      throw std::runtime_error("stage member counter width mismatch");
    result.push_back(strict_counter(input.substr(start,
        comma == std::string::npos ? std::string::npos : comma - start)));
    if (comma != std::string::npos) start = comma + 1U;
  }
  return result;
}

void append_physical_members(boost::json::object& event,
    const boost::json::array& tested, const boost::json::array& passed,
    std::uint64_t q0_passed, bool censored_operator) {
  if (tested.size() != 6U || passed.size() != 6U)
    throw std::runtime_error("stage member PRP counter width mismatch");
  boost::json::array physical_ratios, unknown;
  bool censored = censored_operator;
  for (std::size_t member = 0U; member < 6U; ++member) {
    const auto n = strict_json_counter(tested[member]);
    const auto p = strict_json_counter(passed[member]);
    if (p > n || n > q0_passed)
      throw std::runtime_error("stage member PRP conservation failed");
    unknown.push_back(q0_passed - n);
    censored = censored || n < q0_passed;
    if (p == 0U) physical_ratios.push_back(nullptr);
    else physical_ratios.push_back(static_cast<double>(n) / static_cast<double>(p));
  }
  event["member_tested"] = tested;
  event["member_prp_passed"] = passed;
  event["member_unknown"] = std::move(unknown);
  event["member_unknown_scope"] = "not_tested_not_classified_composite";
  event["member_r_emp"] = std::move(physical_ratios);
  event["member_tested_scope"] = "physical_prp_tests_not_prefix_survivors";
  event["member_pass_scope"] = censored
      ? "observed_conditionally_tested_not_all_q0_passes"
      : "observed_all_q0_passes";
}

std::array<std::uint8_t, 32> sha256(const std::vector<std::uint8_t>& bytes) {
#ifdef _WIN32
  BCRYPT_ALG_HANDLE algorithm = nullptr;
  BCRYPT_HASH_HANDLE hash = nullptr;
  DWORD object_size = 0, hash_size = 0, returned = 0;
  const auto failed = [](NTSTATUS status) { return status < 0; };
  if (failed(BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM,
                                         nullptr, 0)))
    throw std::runtime_error("stage hash provider unavailable");
  if (failed(BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
                               reinterpret_cast<PUCHAR>(&object_size),
                               sizeof(object_size), &returned, 0)) ||
      failed(BCryptGetProperty(algorithm, BCRYPT_HASH_LENGTH,
                               reinterpret_cast<PUCHAR>(&hash_size),
                               sizeof(hash_size), &returned, 0)) || hash_size != 32U) {
    BCryptCloseAlgorithmProvider(algorithm, 0);
    throw std::runtime_error("stage hash provider invalid");
  }
  std::vector<std::uint8_t> object(object_size);
  std::array<std::uint8_t, 32> digest{};
  if (failed(BCryptCreateHash(algorithm, &hash, object.data(), object_size,
                              nullptr, 0, 0)) ||
      (!bytes.empty() && failed(BCryptHashData(
          hash, const_cast<PUCHAR>(bytes.data()),
          static_cast<ULONG>(bytes.size()), 0))) ||
      failed(BCryptFinishHash(hash, digest.data(),
                              static_cast<ULONG>(digest.size()), 0))) {
    if (hash) BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(algorithm, 0);
    throw std::runtime_error("stage hashing failed");
  }
  BCryptDestroyHash(hash);
  BCryptCloseAlgorithmProvider(algorithm, 0);
  return digest;
#else
  std::array<std::uint8_t, 32> digest{}; unsigned length=0;
  if(EVP_Digest(bytes.data(),bytes.size(),digest.data(),&length,EVP_sha256(),nullptr)!=1||length!=32)
    throw std::runtime_error("stage hashing failed");
  return digest;
#endif
}

std::string hash_bytes_hex(const std::vector<std::uint8_t>& bytes) {
  const auto digest = sha256(bytes);
  return hex_encode(std::vector<std::uint8_t>(digest.begin(), digest.end()));
}

std::string hash_file_hex(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("stage executable unavailable");
  input.seekg(0, std::ios::end);
  const auto length = input.tellg();
  if (length <= 0 || length > 512LL * 1024LL * 1024LL)
    throw std::runtime_error("stage executable size invalid");
  input.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
  input.read(reinterpret_cast<char*>(bytes.data()), length);
  if (!input) throw std::runtime_error("stage executable read failed");
  return hash_bytes_hex(bytes);
}

std::string gpu_identity_hash(const StageConfig& config) {
  const std::string identity = config.gpu_uuid + "|" + config.gpu_pci +
      "|device=" + std::to_string(config.device_index);
  return hash_bytes_hex(std::vector<std::uint8_t>(identity.begin(), identity.end()));
}

std::uint64_t reserved_factor_end(const StageConfig& config,
                                  const WorkAssignment& work) {
  const std::uint64_t batches = config.production_session_seconds == 0U
      ? 1U : config.production_max_batches;
  if (batches == 0U ||
      work.positions > (std::numeric_limits<std::uint64_t>::max() -
                        work.factor_origin) / batches)
    throw std::runtime_error("stage factor reservation overflow");
  return work.factor_origin + static_cast<std::uint64_t>(work.positions) * batches;
}

#ifdef _WIN32
std::wstring widen(const std::string& value) {
  if (value.empty()) return {};
  const int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                         value.data(),
                                         static_cast<int>(value.size()),
                                         nullptr, 0);
  if (length <= 0) throw std::runtime_error("UTF-8 command argument invalid");
  std::wstring result(static_cast<std::size_t>(length), L'\0');
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                      static_cast<int>(value.size()), result.data(), length);
  return result;
}

std::wstring quote_argument(const std::wstring& argument) {
  if (argument.empty()) return L"\"\"";
  if (argument.find_first_of(L" \t\n\v\"") == std::wstring::npos) return argument;
  std::wstring result = L"\"";
  std::size_t backslashes = 0;
  for (const wchar_t c : argument) {
    if (c == L'\\') {
      ++backslashes;
    } else if (c == L'\"') {
      result.append(backslashes * 2U + 1U, L'\\');
      result.push_back(c);
      backslashes = 0;
    } else {
      result.append(backslashes, L'\\');
      backslashes = 0;
      result.push_back(c);
    }
  }
  result.append(backslashes * 2U, L'\\');
  result.push_back(L'\"');
  return result;
}

std::wstring command_line(const std::vector<std::string>& arguments) {
  std::wstring result;
  for (const auto& argument : arguments) {
    if (!result.empty()) result.push_back(L' ');
    result += quote_argument(widen(argument));
  }
  return result;
}

#endif
cpp_int parse_hex_integer(const std::string& value) {
  if (value.empty()) throw std::runtime_error("empty integer in stage journal");
  (void)hex_decode(value.size() % 2U == 0U ? value : "0" + value);
  cpp_int result = 0;
  for (const char c : value) {
    result <<= 4;
    if (c >= '0' && c <= '9') result += c - '0';
    else if (c >= 'a' && c <= 'f') result += c - 'a' + 10;
    else if (c >= 'A' && c <= 'F') result += c - 'A' + 10;
  }
  return result;
}

std::string integer_hex(const cpp_int& value) {
  std::ostringstream output;
  output << std::hex << value;
  return output.str();
}

std::uint64_t decimal_u64(const boost::json::value& value) {
  std::string text;
  if (value.is_string()) text = std::string(value.as_string().c_str());
  else if (value.is_uint64()) return value.as_uint64();
  else if (value.is_int64() && value.as_int64() >= 0)
    return static_cast<std::uint64_t>(value.as_int64());
  else throw std::runtime_error("stage journal integer type invalid");
  std::size_t end = 0;
  const auto parsed = std::stoull(text, &end, 10);
  if (end != text.size()) throw std::runtime_error("stage journal decimal invalid");
  return parsed;
}

cpp_int primorial(std::uint32_t count) {
  cpp_int result = 1;
  std::uint32_t found = 0;
  for (std::uint32_t candidate = 2; found < count; ++candidate) {
    bool prime = true;
    for (std::uint32_t divisor = 2;
         static_cast<std::uint64_t>(divisor) * divisor <= candidate; ++divisor)
      if (candidate % divisor == 0U) { prime = false; break; }
    if (prime) { result *= candidate; ++found; }
  }
  return result;
}

bool replay_probable_prime(const cpp_int& n) {
  const std::string hex = integer_hex(n);
  mpz_t value;
  mpz_init(value);
  if (mpz_set_str(value, hex.c_str(), 16) != 0) {
    mpz_clear(value);
    throw std::runtime_error("GMP replay import failed");
  }
  const bool probable = mpz_probab_prime_p(value, 32) != 0;
  mpz_clear(value);
  return probable;
}

std::uint32_t replay_prime_count(const cpp_int& first, std::uint32_t pattern) {
  if (pattern >= kOffsets.size()) throw std::runtime_error("stage pattern invalid");
  std::uint32_t count = 0;
  for (std::size_t member = 0; member < kOffsets[pattern].size(); ++member) {
    if (replay_probable_prime(first + kOffsets[pattern][member])) {
      ++count;
    } else if (member < 2U || count +
               (kOffsets[pattern].size() - member - 1U) < 5U) {
      return count;
    }
  }
  return count;
}

std::vector<boost::json::object> read_journal(const fs::path& path) {
  if (!fs::is_regular_file(path) || fs::file_size(path) > kJournalLimit)
    throw std::runtime_error("stage journal missing or oversized");
  std::ifstream input(path, std::ios::binary);
  std::vector<boost::json::object> records;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line.size() > kJournalRecordLimit)
      throw std::runtime_error("stage journal record invalid");
    boost::system::error_code error;
    auto value = boost::json::parse(line, error);
    if (error || !value.is_object())
      throw std::runtime_error("stage journal JSON invalid");
    records.push_back(value.as_object());
  }
  if (!input.eof()) throw std::runtime_error("stage journal read failed");
  return records;
}

struct BoundedJournalLine {
  std::string text;
  std::uintmax_t offset{0U};
  bool terminated{false};
};

std::ifstream open_bounded_journal(const fs::path& path) {
  if (!fs::is_regular_file(path) || fs::file_size(path) > kJournalLimit)
    throw std::runtime_error("stage journal missing or oversized");
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("stage journal open failed");
  return input;
}

std::optional<BoundedJournalLine> read_bounded_journal_line(
    std::ifstream& input, std::uintmax_t& consumed) {
  BoundedJournalLine line;
  line.offset = consumed;
  char character = '\0';
  while (input.get(character)) {
    if (++consumed > kJournalLimit)
      throw std::runtime_error("stage journal grew beyond read bound");
    if (character == '\n') {
      line.terminated = true;
      if (!line.text.empty() && line.text.back() == '\r') line.text.pop_back();
      return line;
    }
    if (line.text.size() == kJournalRecordLimit)
      throw std::runtime_error("stage journal record oversized");
    line.text.push_back(character);
  }
  if (input.bad() || !input.eof())
    throw std::runtime_error("stage journal read failed");
  if (consumed == line.offset) return {};
  return line;
}

void require_string(const boost::json::object& record, const char* key,
                    const std::string& expected) {
  const auto* value = record.if_contains(key);
  if (value == nullptr || !value->is_string() || value->as_string() != expected)
    throw std::runtime_error("stage journal binding mismatch");
}

void require_ascii_case_insensitive(const boost::json::object& record,
                                    const char* key,
                                    const std::string& expected) {
  const auto* value = record.if_contains(key);
  if (value == nullptr || !value->is_string())
    throw std::runtime_error(std::string("stage journal binding mismatch: ") + key);
  auto canonical = [](std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    return text;
  };
  if (canonical(std::string(value->as_string().c_str())) != canonical(expected))
    throw std::runtime_error(std::string("stage journal binding mismatch: ") + key);
}

struct InterruptedJournalInspection {
  std::uint64_t complete_records{0U};
  std::uintmax_t tail_offset{0U};
  std::uintmax_t tail_bytes{0U};
  std::string tail_sha256;
};

// An interrupted object can already contain a complete failure field. Use the
// JSON tokenizer (including escapes and nesting), not substring matching or a
// fabricated closing brace, to inspect every value that was actually written.
struct InterruptedFailureHandler {
  static constexpr std::size_t max_array_size = kJournalRecordLimit;
  static constexpr std::size_t max_object_size = kJournalRecordLimit;
  static constexpr std::size_t max_string_size = kJournalRecordLimit;
  static constexpr std::size_t max_key_size = kJournalRecordLimit;
  using Error = boost::system::error_code;
  using View = boost::json::string_view;
  std::string key, key_parts, string_parts;

  bool observe(boost::json::value value) {
    boost::json::object field;
    field.emplace(key, std::move(value));
    inspect_failure_value(field);
    key.clear();
    return true;
  }
  bool on_document_begin(Error&) { return true; }
  bool on_document_end(Error&) { return true; }
  bool on_array_begin(Error&) { return observe(boost::json::array{}); }
  bool on_array_end(std::size_t, Error&) { return true; }
  bool on_object_begin(Error&) { return observe(boost::json::object{}); }
  bool on_object_end(std::size_t, Error&) { return true; }
  bool on_key_part(View part, std::size_t, Error&) {
    key_parts.append(part.data(), part.size());
    return true;
  }
  bool on_key(View part, std::size_t, Error&) {
    key = std::move(key_parts);
    key_parts.clear();
    key.append(part.data(), part.size());
    return true;
  }
  bool on_string_part(View part, std::size_t, Error&) {
    string_parts.append(part.data(), part.size());
    return true;
  }
  bool on_string(View part, std::size_t, Error&) {
    string_parts.append(part.data(), part.size());
    const auto value = std::move(string_parts);
    string_parts.clear();
    return observe(boost::json::value(value));
  }
  bool on_number_part(View, Error&) { return true; }
  bool on_int64(std::int64_t value, View, Error&) { return observe(value); }
  bool on_uint64(std::uint64_t value, View, Error&) { return observe(value); }
  bool on_double(double value, View, Error&) { return observe(value); }
  bool on_bool(bool value, Error&) { return observe(value); }
  bool on_null(Error&) { return observe(nullptr); }
  bool on_comment_part(View, Error&) { return true; }
  bool on_comment(View, Error&) { return true; }
};

void inspect_incomplete_failure_evidence(const std::string& prefix) {
  boost::json::basic_parser<InterruptedFailureHandler> parser(boost::json::parse_options{});
  boost::system::error_code error;
  (void)parser.write_some(false, prefix.data(), prefix.size(), error);
  if (error != boost::json::error::incomplete)
    throw std::runtime_error("interrupted stage journal suffix is not incomplete JSON");
}

// Only an owned, confirmed interruption can leave an uncommitted final line.
// All complete lines and every observed committed proof remain strict. A valid
// JSON object lacking LF is still not admitted, but cannot conceal an error.
InterruptedJournalInspection inspect_interrupted_journal(
    const fs::path& path, const WorkAssignment& work, std::uint64_t committed) {
  auto input = open_bounded_journal(path);
  std::uintmax_t consumed = 0U;
  InterruptedJournalInspection result;
  while (const auto line = read_bounded_journal_line(input, consumed)) {
    boost::system::error_code error;
    const auto value = boost::json::parse(line->text, error);
    if (line->terminated || !error) {
      if (error || !value.is_object())
        throw std::runtime_error("stage journal JSON invalid");
      const auto& record = value.as_object();
      require_string(record, "work_id", work.work_id);
      require_string(record, "template_id", work.template_id);
      inspect_failure_value(record);
      if (line->terminated) {
        if (result.complete_records < committed)
          require_string(record, "schema", "riecoin-exact-result-v1");
        ++result.complete_records;
        continue;
      }
    } else {
      const auto first = line->text.find_first_not_of(" \t\r");
      if (error != boost::json::error::incomplete || first == std::string::npos ||
          line->text[first] != '{')
        throw std::runtime_error("interrupted stage journal suffix is not a JSON prefix");
      inspect_incomplete_failure_evidence(line->text);
    }
    if (result.complete_records < committed)
      throw std::runtime_error("committed proof journal truncated during invalidation");
    result.tail_offset = line->offset;
    result.tail_bytes = consumed - line->offset;
    result.tail_sha256 = hash_bytes_hex(std::vector<std::uint8_t>(
        line->text.begin(), line->text.end()));
  }
  if (result.complete_records < committed)
    throw std::runtime_error("committed proof missing during invalidation");
  return result;
}

struct ProofContext {
  StageConfig config;
  WorkAssignment work;
  std::string runtime_hash;
  std::string gpu_hash;
  cpp_int prime_product;
  cpp_int candidate_base;
  std::uint64_t factor_end{0U};

  ProofContext(StageConfig value, WorkAssignment assignment)
      : config(std::move(value)), work(std::move(assignment)) {
    if (work.primorial_number != 0U) config.primorial_number = work.primorial_number;
    if (work.work_id.empty() || work.template_id.empty() || work.positions == 0U)
      throw std::runtime_error("stage proof work identity missing");
    factor_end = reserved_factor_end(config, work);
    require_consensus_factor_window(work, config.primorial_number,
                                    config.primorial_offset, factor_end);
    runtime_hash = hash_file_hex(config.executable);
    gpu_hash = gpu_identity_hash(config);
    const cpp_int target = parse_hex_integer(work.target_hex);
    prime_product = primorial(config.primorial_number);
    // Consensus advances a FULL primorial at exact divisibility as well.
    candidate_base = target + prime_product - target % prime_product +
                     config.primorial_offset;
  }
};

void require_proof_binding(const ProofContext& context,
                           const boost::json::object& record) {
  require_string(record, "work_id", context.work.work_id);
  require_string(record, "template_id", context.work.template_id);
  require_string(record, "runtime_sha256", context.runtime_hash);
  require_string(record, "gpu_identity_sha256", context.gpu_hash);
  require_string(record, "target_hex", context.work.target_hex);
  if (strict_json_counter(record.at("pattern")) != context.work.pattern ||
      strict_json_counter(record.at("primorial_number")) != context.config.primorial_number ||
      decimal_u64(record.at("primorial_offset")) != context.config.primorial_offset)
    throw std::runtime_error("stage geometry binding mismatch");
}

ExactShare validate_individual_proof(const ProofContext& context,
                                     const boost::json::object& record) {
  require_proof_binding(context, record);
  inspect_failure_value(record);
  require_string(record, "schema", "riecoin-exact-result-v1");
  require_string(record, "event", "exact_constellation");
  require_string(record, "proof_state", "riecoin_v1_min5_exact_32_rounds");
  require_string(record, "terminal_status", "proof_complete");
  require_string(record, "submission_status", "not_attempted");
  const auto factor = decimal_u64(record.at("factor"));
  if (factor < context.work.factor_origin || factor >= context.factor_end)
    throw std::runtime_error("share factor outside reserved work range");
  const cpp_int candidate = context.candidate_base + context.prime_product * factor;
  require_string(record, "candidate_hex", integer_hex(candidate));
  const std::string nonce(record.at("nonce_v1_uint256_hex").as_string());
  if (nonce != make_nonce_v1_hex(static_cast<std::uint16_t>(context.config.primorial_number),
          factor, context.config.primorial_offset, context.config.difficulty_offset))
    throw std::runtime_error("share nonce binding/uniqueness failed");
  const auto replay_count = replay_prime_count(candidate, context.work.pattern);
  if (replay_count < 5U || strict_json_counter(record.at("prime_count")) != replay_count)
    throw std::runtime_error("share Q0/Q1 mandatory and min5 replay failed");
  return {nonce, replay_count, false};
}

// A durable per-work identity precedes any in-memory queue/wire action. Never
// recover by blindly replaying this file: a process loss leaves send outcomes
// unknown, and the normal range ledger allocates a fresh header on relaunch.
void append_proof_admission(const fs::path& path, const ProofContext& context,
    const boost::json::object& record, const ExactShare& share, bool live) {
  if (path.empty()) throw std::runtime_error("live proof admission journal missing");
  const auto encoded_record = boost::json::serialize(record);
  const auto record_hash = hash_bytes_hex(std::vector<std::uint8_t>(
      encoded_record.begin(), encoded_record.end()));
  boost::json::object receipt{{"schema", "riecoin-proof-admission-v1"},
      {"phase", "share_proof_admitted"}, {"work_id", context.work.work_id},
      {"template_id", context.work.template_id}, {"clean_epoch", context.work.clean_epoch},
      {"runtime_sha256", context.runtime_hash}, {"gpu_identity_sha256", context.gpu_hash},
      {"nonce_v1_uint256_hex", share.nonce_v1_uint256_hex}, {"prime_count", share.prime_count},
      {"factor", record.at("factor")}, {"proof_record_sha256", record_hash},
      {"proof_scope", "individual_gmp32_min5_with_holes_not_stage_coverage"},
      {"stage_complete", !live}, {"submission_status", "not_sent"},
      {"terminal_status", "individually_validated"}};
  const std::string bytes = boost::json::serialize(receipt) + "\n";
  if (fs::exists(path) && fs::file_size(path) > kJournalLimit - bytes.size())
    throw std::runtime_error("proof admission journal oversized");
  HANDLE file = CreateFileW(path.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ, nullptr,
                             OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE)
    throw std::runtime_error("proof admission journal open failed");
  struct Close { HANDLE file; ~Close() { CloseHandle(file); } } close{file};
  DWORD written = 0U;
  if (!WriteFile(file, bytes.data(), static_cast<DWORD>(bytes.size()), &written, nullptr) ||
      written != bytes.size() || !FlushFileBuffers(file))
    throw std::runtime_error("proof admission journal durable write failed");
}

boost::json::object validate_summary(const fs::path& path,
                                     const StageConfig& config,
                                     const WorkAssignment& work,
                                     std::size_t validated_shares,
                                     bool operator_drain_requested = false) {
  if (path.empty() || !fs::exists(path))
    throw std::runtime_error("mandatory network-batch summary missing");
  const auto records = read_journal(path);
  if (records.size() != 1U) throw std::runtime_error("stage summary invalid");
  const auto& summary = records[0];
  require_string(summary, "schema", "riecoin-q0-benchmark-v1");
  require_string(summary, "work_id", work.work_id);
  require_string(summary, "template_id", work.template_id);
  require_string(summary, "terminal_status", "pass");
  const bool production = config.production_session_seconds != 0U;
  if (production)
    require_string(summary, "measurement_layer",
                   "production_multibatch_pre_submit");
  const auto& gpu = summary.at("gpu").as_object();
  // NVIDIA UUID text is case-insensitive. The CUDA runtime emits `gpu-...`
  // while NVML/UI inventory emits `GPU-...`; preserve identity semantics
  // instead of rejecting a valid terminal result on presentation casing.
  require_ascii_case_insensitive(gpu, "uuid", config.gpu_uuid);
  require_string(gpu, "pci", config.gpu_pci);
  const auto& geometry = summary.at("geometry").as_object();
  const std::uint64_t factor_begin =
      geometry.at("factor_origin_begin").to_number<std::uint64_t>();
  const std::uint64_t factor_end =
      geometry.at("factor_origin_end").to_number<std::uint64_t>();
  const std::uint64_t positions =
      geometry.at("positions").to_number<std::uint64_t>();
  const std::uint64_t batch_positions = geometry.if_contains("batch_positions")
      ? geometry.at("batch_positions").to_number<std::uint64_t>() : positions;
  const std::uint64_t batches = geometry.if_contains("batches")
      ? geometry.at("batches").to_number<std::uint64_t>() : 1U;
  if (batches == 0U ||
      batch_positions > std::numeric_limits<std::uint64_t>::max() / batches ||
      positions > std::numeric_limits<std::uint64_t>::max() - factor_begin)
    throw std::runtime_error("stage summary geometry overflow");
  if (geometry.at("target_bits").to_number<std::uint64_t>() != work.target_bits ||
      geometry.at("pattern").to_number<std::uint64_t>() != work.pattern ||
      geometry.at("primorial_number").to_number<std::uint64_t>() !=
          config.primorial_number ||
      decimal_u64(geometry.at("primorial_offset")) != config.primorial_offset ||
      factor_begin != work.factor_origin || batch_positions != work.positions ||
      positions != batches * batch_positions ||
      factor_end != factor_begin + positions ||
      factor_end > reserved_factor_end(config, work) ||
      (!production && (batches != 1U || positions != work.positions)))
    throw std::runtime_error("network-batch summary geometry mismatch");
  const auto& counts = summary.at("counts").as_object();
  if (counts.at("proofs").to_number<std::uint64_t>() != validated_shares ||
      counts.at("errors").to_number<std::uint64_t>() != 0U)
    throw std::runtime_error("network-batch summary count mismatch");
  (void)summary.at("rates").as_object().at("q0_candidates_s").to_number<double>();
  (void)summary.at("rates").as_object().at("positions_s").to_number<double>();
  (void)summary.at("rates").as_object().at("prp_s_secondary").to_number<double>();
  (void)summary.at("rates").as_object().at("prp_per_candidate").to_number<double>();
  const auto& window = summary.at("window").as_object();
  const double measured_s = window.at("measured_s").to_number<double>();
  bool exhausted = false;
  bool drained_prefix = false;
  if (const auto* reason = window.if_contains("completion_reason")) {
    const auto& text = reason->as_string();
    drained_prefix = text == "cancelled_consumed_prefix" && operator_drain_requested;
    if (text != "duration" && text != "reservation_exhausted" && !drained_prefix)
      throw std::runtime_error("unknown production window completion reason");
    exhausted = text == "reservation_exhausted";
  }
  if (drained_prefix && (!production || positions == 0U ||
      !std::isfinite(measured_s) || measured_s <= 0.0))
    throw std::runtime_error("invalid operator-drained consumed prefix");
  // RC3:OPERATOR-DRAIN changes only the duration obligation, never geometry,
  // proof replay, rejected-candidate audits or exactness/conservation checks.
  if (production && !drained_prefix && !riecoin_production_window::complete(
          measured_s, config.production_session_seconds, exhausted,
          factor_begin, factor_end, reserved_factor_end(config, work)))
    throw std::runtime_error("production window incomplete or reservation inconsistent");
  const auto& exactness = summary.at("exactness").as_object();
  for (const char* key : {"sieve_mismatch_words", "q0_mismatches",
                          "dynamic_bit_parity_mismatches",
                          "dynamic_bit_prime_false_negatives",
                          "replay_mismatch_words", "replay_verdict_mismatches",
                          "nonce_mismatches", "prime_false_negatives",
                          "drops", "duplicates"})
    if (exactness.at(key).to_number<std::uint64_t>() != 0U)
      throw std::runtime_error("stage exactness summary failed");
  if (production) {
    const auto& audit = exactness.at("rejected_audit").as_object();
    if (audit.at("bound_per_stage").to_number<std::uint64_t>() !=
            config.rejected_audit_per_stage ||
        audit.at("detected_exact_share_false_negatives").to_number<std::uint64_t>() != 0U ||
        audit.at("nonce_mismatches").to_number<std::uint64_t>() != 0U ||
        audit.at("global_false_negative_claim").as_bool())
      throw std::runtime_error("bounded rejected-path audit failed");
    const auto& conservation = summary.at("conservation").as_object();
    if (!conservation.at("pass").as_bool() ||
        conservation.at("range_positions").to_number<std::uint64_t>() != positions ||
        conservation.at("exact_results").to_number<std::uint64_t>() !=
            validated_shares)
      throw std::runtime_error("production conservation summary failed");
    const auto& funnel = summary.at("funnel").as_object();
    if (funnel.at("q0_entered").to_number<std::uint64_t>() !=
            counts.at("q0_entries").to_number<std::uint64_t>() ||
        funnel.at("q0_passed").to_number<std::uint64_t>() !=
            counts.at("q0_passed").to_number<std::uint64_t>() ||
        funnel.at("exact_shares").to_number<std::uint64_t>() != validated_shares)
      throw std::runtime_error("production funnel summary failed");
  }
  return summary;
}

}  // namespace

std::string stage_executable_sha256(const fs::path& path) {
  return hash_file_hex(path);
}

StageRetryDisposition stage_retry_disposition(bool stop_confirmed,
    bool own_termination_succeeded, std::uint32_t exit_code) {
  if (!stop_confirmed || exit_code == STILL_ACTIVE)
    throw std::runtime_error("stage_invalidation_stop_unconfirmed");
  if (exit_code == 0U) return StageRetryDisposition::completed;
  if (own_termination_succeeded && exit_code == kCancelledStageExitCode)
    return StageRetryDisposition::cancelled;
  throw std::runtime_error("stage_failed_before_invalidation");
}

void require_no_stage_failure_evidence(const std::string& line) {
  if (line.size() > 1024U * 1024U)
    throw std::runtime_error("stage_invalidation_evidence_line_oversized");
  const auto first = line.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return;
  if (line[first] == '{') {
    const auto record = boost::json::parse(line);
    if (!record.is_object())
      throw std::runtime_error("stage_invalidation_evidence_invalid");
    inspect_failure_value(record);
    return;
  }
  std::istringstream tokens(line);
  std::string token;
  while (tokens >> token) {
    const auto equals = token.find('=');
    if (equals == std::string::npos) continue;
    const auto key = token.substr(0U, equals);
    const auto value = token.substr(equals + 1U);
    if (failure_status(value) ||
        (failure_counter(key) && strict_counter(value) != 0U) ||
        (key == "error" && value != "0"))
      throw std::runtime_error("stage_invalidation_observed_failure");
  }
}

std::string make_nonce_v1_hex(std::uint16_t primorial_number,
                              std::uint64_t factor,
                              std::uint64_t primorial_offset,
                              std::uint16_t difficulty_offset) {
  std::array<std::uint8_t, 32> bytes{};
  const std::uint16_t version = static_cast<std::uint16_t>(
      2U + (static_cast<std::uint32_t>(difficulty_offset) << 5U));
  bytes[0] = static_cast<std::uint8_t>(version);
  bytes[1] = static_cast<std::uint8_t>(version >> 8U);
  for (std::size_t i = 0; i < 8U; ++i) {
    bytes[2U + i] = static_cast<std::uint8_t>(primorial_offset >> (8U * i));
    bytes[14U + i] = static_cast<std::uint8_t>(factor >> (8U * i));
  }
  bytes[30] = static_cast<std::uint8_t>(primorial_number);
  bytes[31] = static_cast<std::uint8_t>(primorial_number >> 8U);
  std::vector<std::uint8_t> reversed(bytes.rbegin(), bytes.rend());
  return hex_encode(reversed);
}

std::string reconstruct_candidate_hex(const std::string& target_hex,
                                      std::uint32_t primorial_number,
                                      std::uint64_t factor,
                                      std::uint64_t primorial_offset) {
  const cpp_int target = parse_hex_integer(target_hex);
  const cpp_int prime_product = primorial(primorial_number);
  const cpp_int alignment = prime_product - (target % prime_product);
  return integer_hex(target + alignment + prime_product * factor +
                     primorial_offset);
}

void require_consensus_factor_window(const WorkAssignment& work,
                                     std::uint32_t primorial_number,
                                     std::uint64_t primorial_offset,
                                     std::uint64_t factor_end) {
  if (work.target_offset_bits == 0U) return;  // Explicit local fixtures only.
  if (work.target_offset_bits > 4096U ||
      work.target_bits != work.target_offset_bits + 265U ||
      primorial_number == 0U || primorial_number > 65535U ||
      factor_end <= work.factor_origin)
    throw std::runtime_error("invalid consensus factor window metadata");
  const cpp_int target = parse_hex_integer(work.target_hex);
  const cpp_int limit = cpp_int(1) << work.target_offset_bits;
  if (target <= 0 || target % limit != 0 ||
      boost::multiprecision::msb(target) + 1U != work.target_bits)
    throw std::runtime_error("consensus target/offset binding mismatch");
  const cpp_int product = primorial(primorial_number);
  if (product > limit)
    throw std::runtime_error("primorial_exceeds_consensus_target_window");
  const cpp_int last_offset = product - target % product +
      product * (factor_end - 1U) + primorial_offset;
  if (last_offset >= limit)
    throw std::runtime_error("candidate_offset_exceeds_consensus_target_window");
}

std::uint32_t select_consensus_primorial(const WorkAssignment& work,
                                       std::uint32_t maximum_primorial_number,
                                       std::uint64_t primorial_offset,
                                       std::uint64_t factor_end) {
  if (work.target_offset_bits == 0U || work.target_offset_bits > 4096U ||
      work.target_bits != work.target_offset_bits + 265U ||
      maximum_primorial_number == 0U || maximum_primorial_number > 65535U ||
      factor_end <= work.factor_origin)
    throw std::runtime_error("cannot select unbound live consensus geometry");
  const cpp_int target = parse_hex_integer(work.target_hex);
  const cpp_int limit = cpp_int(1) << work.target_offset_bits;
  if (target <= 0 || target % limit != 0 ||
      boost::multiprecision::msb(target) + 1U != work.target_bits)
    throw std::runtime_error("consensus target/offset binding mismatch");
  std::vector<cpp_int> products{cpp_int(1)};
  for (std::uint32_t value = 2; products.size() <= maximum_primorial_number; ++value) {
    bool prime = true;
    for (std::uint32_t divisor = 2;
         static_cast<std::uint64_t>(divisor) * divisor <= value; ++divisor)
      if (value % divisor == 0U) { prime = false; break; }
    if (!prime) continue;
    products.push_back(products.back() * value);
    // Larger primorials can no longer fit even with the smallest alignment.
    if (products.back() > limit ||
        products.back() * (factor_end - 1U) + 1U + primorial_offset >= limit)
      break;
  }
  for (std::size_t count = products.size() - 1U; count > 0U; --count) {
    const cpp_int& product = products[count];
    if (product > limit) continue;
    const cpp_int last_offset = product - target % product +
        product * (factor_end - 1U) + primorial_offset;
    if (last_offset < limit) return static_cast<std::uint32_t>(count);
  }
  throw std::runtime_error("no primorial fits the reserved consensus window");
}

void require_consensus_nonce(const WorkAssignment& work,
                             const std::string& nonce_hex,
                             std::uint16_t difficulty_offset) {
  if (work.target_offset_bits == 0U)
    throw std::runtime_error("local_fixture_cannot_be_submitted");
  auto bytes = hex_decode(nonce_hex, 32U);
  std::reverse(bytes.begin(), bytes.end());
  const std::uint16_t version = bytes[0] | (std::uint16_t(bytes[1]) << 8U);
  if (version != 2U + (std::uint32_t(difficulty_offset) << 5U))
    throw std::runtime_error("nonce_difficulty_version_mismatch");
  // This producer deliberately uses uint64 factors/offsets. Reject nonzero
  // high limbs instead of silently truncating the consensus 128/96-bit fields.
  for (std::size_t i = 10U; i < 14U; ++i)
    if (bytes[i] != 0U) throw std::runtime_error("nonce_offset_high_limb_nonzero");
  for (std::size_t i = 22U; i < 30U; ++i)
    if (bytes[i] != 0U) throw std::runtime_error("nonce_factor_high_limb_nonzero");
  std::uint64_t factor = 0U, offset = 0U;
  for (std::size_t i = 0; i < 8U; ++i) {
    offset |= std::uint64_t(bytes[2U + i]) << (8U * i);
    factor |= std::uint64_t(bytes[14U + i]) << (8U * i);
  }
  const std::uint32_t count = bytes[30] | (std::uint32_t(bytes[31]) << 8U);
  if (factor == std::numeric_limits<std::uint64_t>::max() ||
      factor < work.factor_origin ||
      (work.primorial_number != 0U && count != work.primorial_number))
    throw std::runtime_error("nonce_factor_or_primorial_binding_mismatch");
  WorkAssignment single = work;
  single.factor_origin = factor;
  require_consensus_factor_window(single, count, offset, factor + 1U);
}

struct StageJournalReader::Impl {
  Impl(StageConfig config, WorkAssignment work, fs::path path, fs::path receipt)
      : context(std::move(config), std::move(work)), journal(std::move(path)),
        admissions(std::move(receipt)) {
    if (!admissions.empty() && fs::exists(admissions))
      throw std::runtime_error("proof admission identity already exists; replay forbidden");
  }
  ProofContext context;
  fs::path journal;
  fs::path admissions;
  std::uintmax_t offset{0U};
  std::string partial;
  std::uint64_t committed{0U};
  std::uint64_t replays{0U};
  bool poisoned{false};
  bool finished{false};
  struct CachedProof { std::string record; ExactShare share; };
  std::vector<CachedProof> cache;
  std::set<std::string> nonces;

  std::optional<std::string> next_line() {
    if (!fs::is_regular_file(journal))
      throw std::runtime_error("committed proof journal missing");
    const auto size = fs::file_size(journal);
    if (size > kJournalLimit || size < offset)
      throw std::runtime_error("proof journal oversized or truncated after admission");
    auto newline = partial.find('\n');
    if (newline == std::string::npos && size > offset) {
      const auto bytes = static_cast<std::size_t>((std::min)(
          size - offset, static_cast<std::uintmax_t>(kJournalReadChunk)));
      std::ifstream input(journal, std::ios::binary);
      input.seekg(static_cast<std::streamoff>(offset));
      std::string chunk(bytes, '\0');
      input.read(chunk.data(), static_cast<std::streamsize>(bytes));
      if (input.gcount() != static_cast<std::streamsize>(bytes))
        throw std::runtime_error("proof journal incremental read failed");
      offset += bytes;
      partial += chunk;
      newline = partial.find('\n');
    }
    if (newline == std::string::npos) {
      if (partial.size() > kJournalRecordLimit)
        throw std::runtime_error("proof journal unterminated record oversized");
      return {};  // No LF means no admission, even when the JSON already parses.
    }
    if (newline == 0U || newline > kJournalRecordLimit)
      throw std::runtime_error("proof journal record invalid");
    std::string line = partial.substr(0U, newline);
    partial.erase(0U, newline + 1U);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return line;
  }
};

StageJournalReader::StageJournalReader(StageConfig config, WorkAssignment work,
    fs::path journal, fs::path admissions)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(work),
                                  std::move(journal), std::move(admissions))) {}
StageJournalReader::~StageJournalReader() = default;
const std::string& StageJournalReader::runtime_sha256() const noexcept {
  return impl_->context.runtime_hash;
}
std::uint64_t StageJournalReader::admitted_count() const noexcept {
  return static_cast<std::uint64_t>(impl_->cache.size());
}
std::uint64_t StageJournalReader::replay_count() const noexcept { return impl_->replays; }

void StageJournalReader::verify_admitted_prefix() const {
  if (impl_->poisoned) throw std::runtime_error("proof reader is closed");
  if (impl_->cache.empty()) return;
  auto input = open_bounded_journal(impl_->journal);
  std::uintmax_t consumed = 0U;
  // Read exactly the admitted records, with per-record and aggregate byte caps;
  // never parse a writer's uncommitted suffix as part of this prefix check.
  for (const auto& cached : impl_->cache) {
    const auto line = read_bounded_journal_line(input, consumed);
    if (!line || !line->terminated)
      throw std::runtime_error("admitted proof missing or truncated in retained journal");
    boost::system::error_code error;
    const auto value = boost::json::parse(line->text, error);
    if (error || !value.is_object() || boost::json::serialize(value) != cached.record)
      throw std::runtime_error("admitted proof changed before work invalidation");
  }
}

std::optional<ExactShare> StageJournalReader::admit_next(std::uint64_t committed_count) {
  try {
    if (impl_->poisoned || impl_->finished)
      throw std::runtime_error("proof reader is closed");
    if (committed_count < impl_->committed || committed_count > 65536U ||
        committed_count > impl_->context.factor_end - impl_->context.work.factor_origin)
      throw std::runtime_error("committed proof watermark regressed or exceeds capacity");
    impl_->committed = committed_count;
    if (admitted_count() == committed_count) return {};
    const auto line = impl_->next_line();
    if (!line) return {};
    boost::system::error_code error;
    auto value = boost::json::parse(*line, error);
    if (error || !value.is_object()) throw std::runtime_error("proof journal JSON invalid");
    const auto& record = value.as_object();
    const std::string nonce(record.at("nonce_v1_uint256_hex").as_string());
    if (impl_->nonces.count(nonce) != 0U)
      throw std::runtime_error("duplicate proof nonce in committed journal");
    auto share = validate_individual_proof(impl_->context, record);
    ++impl_->replays;
    append_proof_admission(impl_->admissions, impl_->context, record, share, true);
    share.previously_admitted = true;
    impl_->cache.push_back({boost::json::serialize(record), share});
    impl_->nonces.insert(nonce);
    return share;
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

struct StageBridge::Impl {
  explicit Impl(StageConfig value) : config(std::move(value)) {}
  StageConfig config;
  bool operator_drain_requested{false};
  PROCESS_INFORMATION process{};
  HANDLE job{nullptr};
  HANDLE output{INVALID_HANDLE_VALUE};
  std::optional<WorkAssignment> work;
  fs::path journal;
  fs::path summary;
  fs::path log;
  fs::path admissions;
  std::unique_ptr<StageJournalReader> proofs;
  std::optional<std::uint64_t> committed_proofs;
  std::uintmax_t progress_offset{0};
  std::string progress_partial;
  std::chrono::steady_clock::time_point started{};

  void close_handles() {
    if (process.hThread) { CloseHandle(process.hThread); process.hThread = nullptr; }
    if (process.hProcess) { CloseHandle(process.hProcess); process.hProcess = nullptr; }
    if (job) { CloseHandle(job); job = nullptr; }
    if (output != INVALID_HANDLE_VALUE) { CloseHandle(output); output = INVALID_HANDLE_VALUE; }
  }
};

StageBridge::StageBridge(StageConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {
  if (!fs::is_regular_file(impl_->config.executable))
    throw std::runtime_error("stage executable unavailable");
  if ((impl_->config.production_session_seconds != 0U &&
       impl_->config.production_session_seconds < 30U) ||
      impl_->config.production_max_batches == 0U ||
      impl_->config.rejected_audit_per_stage > 1024U)
    throw std::runtime_error("invalid bounded production stage configuration");
}

StageBridge::~StageBridge() {
  if (running()) {
    TerminateProcess(impl_->process.hProcess, 0xC000013AU);
    WaitForSingleObject(impl_->process.hProcess, 5000U);
  }
  impl_->close_handles();
}

bool StageBridge::running() const noexcept {
  return impl_->process.hProcess != nullptr;
}

void StageBridge::request_operator_drain() {
  if (!running() || impl_->operator_drain_requested) return;
  const auto signal = impl_->summary.parent_path() / "operator-drain.request";
  std::ofstream output(signal, std::ios::binary | std::ios::trunc);
  output << "operator drain at exact bundle boundary\n";
  output.close();
  if (!output) throw std::runtime_error("cannot publish operator drain request");
  impl_->operator_drain_requested = true;
}

void StageBridge::start(const WorkAssignment& work) {
  if (running()) throw std::runtime_error("stage already running");
  impl_->operator_drain_requested = false;
  if (work.primorial_number != 0U)
    impl_->config.primorial_number = work.primorial_number;
  require_consensus_factor_window(work, impl_->config.primorial_number,
      impl_->config.primorial_offset, reserved_factor_end(impl_->config, work));
  impl_->started = std::chrono::steady_clock::now();
  impl_->progress_offset = 0U;
  impl_->progress_partial.clear();
  impl_->committed_proofs.reset();
  const cpp_int target_value = parse_hex_integer(work.target_hex);
  if (target_value <= 0 ||
      static_cast<std::uint32_t>(boost::multiprecision::msb(target_value) + 1U) !=
          work.target_bits)
    throw std::runtime_error("stage target width binding mismatch");
  if (work.target_bits > 4096U)
    throw std::runtime_error("stage target width exceeds supported 4096 bits");
  const fs::path directory = impl_->config.live_state / "runs" / work.work_id;
  fs::create_directories(directory);
  if (fs::exists(directory / "operator-drain.request"))
    throw std::runtime_error("stage work identity has an existing drain request");
  impl_->journal = directory / "results.jsonl";
  impl_->summary = directory / "q0-summary.json";
  impl_->log = directory / "stage.log";
  impl_->admissions = directory / "proof-admissions.jsonl";
  for (const auto& path : {impl_->journal, impl_->summary, impl_->log, impl_->admissions})
    if (fs::exists(path)) throw std::runtime_error("stage work identity already exists");

  impl_->proofs = std::make_unique<StageJournalReader>(impl_->config, work,
                                                     impl_->journal, impl_->admissions);
  const std::string runtime_hash = impl_->proofs->runtime_sha256();
  const std::string gpu_hash = gpu_identity_hash(impl_->config);
  std::vector<std::string> arguments{impl_->config.executable.string()};
  arguments.insert(arguments.end(), impl_->config.prefix_arguments.begin(),
                   impl_->config.prefix_arguments.end());
  const std::uint64_t factor_end = reserved_factor_end(impl_->config, work);
  std::vector<std::string> mining;
  if (impl_->config.production_session_seconds != 0U) {
    mining.insert(mining.end(), {
        "--production-session-seconds",
        std::to_string(impl_->config.production_session_seconds),
        "--rejected-audit-per-stage",
        std::to_string(impl_->config.rejected_audit_per_stage)});
  } else {
    mining.push_back("--network-batch");
  }
  const std::vector<std::string> common{
      "--candidates", std::to_string(work.positions),
      "--factor-origin", std::to_string(work.factor_origin),
      "--factor-max", std::to_string(factor_end),
      "--device", std::to_string(impl_->config.device_index),
      "--pattern", std::to_string(work.pattern),
      "--primorial-number", std::to_string(impl_->config.primorial_number),
      "--primorial-offset", std::to_string(impl_->config.primorial_offset),
      "--difficulty-offset", std::to_string(impl_->config.difficulty_offset),
      "--target-hex", work.target_hex,
      "--results-jsonl", impl_->journal.string(),
      "--q0-summary-json", impl_->summary.string(),
      "--work-id", work.work_id,
      "--template-id", work.template_id,
      "--runtime-sha256", runtime_hash,
      "--gpu-identity-sha256", gpu_hash,
      "--expect-uuid", impl_->config.gpu_uuid,
      "--expect-pci", impl_->config.gpu_pci};
  mining.insert(mining.end(), common.begin(), common.end());
  arguments.insert(arguments.end(), mining.begin(), mining.end());
#ifdef _WIN32
  std::wstring command = command_line(arguments);

  SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
  impl_->output = CreateFileW(impl_->log.c_str(), GENERIC_WRITE,
                              FILE_SHARE_READ, &security, CREATE_NEW,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
  if (impl_->output == INVALID_HANDLE_VALUE)
    throw std::runtime_error("stage log creation failed");
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  startup.dwFlags = STARTF_USESTDHANDLES;
  startup.hStdOutput = impl_->output;
  startup.hStdError = impl_->output;
  startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
  impl_->job = CreateJobObjectW(nullptr, nullptr);
  if (!impl_->job) {
    impl_->close_handles();
    throw std::runtime_error("stage kill-on-close job creation failed");
  }
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION job_info{};
  job_info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  if (!SetInformationJobObject(impl_->job, JobObjectExtendedLimitInformation,
                               &job_info, sizeof(job_info))) {
    impl_->close_handles();
    throw std::runtime_error("stage kill-on-close job configuration failed");
  }
  if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE,
                      CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr,
                      directory.c_str(), &startup,
                      &impl_->process)) {
    impl_->close_handles();
    throw std::runtime_error("stage process creation failed");
  }
  if (!AssignProcessToJobObject(impl_->job, impl_->process.hProcess)) {
    TerminateProcess(impl_->process.hProcess, 0xC000013AU);
    WaitForSingleObject(impl_->process.hProcess, 5000U);
    impl_->close_handles();
    throw std::runtime_error("stage kill-on-close job assignment failed");
  }
  if (ResumeThread(impl_->process.hThread) == static_cast<DWORD>(-1)) {
    TerminateProcess(impl_->process.hProcess, 0xC000013AU);
    WaitForSingleObject(impl_->process.hProcess, 5000U);
    impl_->close_handles();
    throw std::runtime_error("stage resume failed");
  }
#else
  // RC2:STAGE-POSIX — direct argv exec, pidfd identity, process-group cancellation.
  impl_->process.hProcess = rc2_spawn(arguments,directory,impl_->log);
#endif
  impl_->work = work;
}

void append_live_member_observability(
    boost::json::object& event,
    const std::map<std::string, std::string>& fields) {
  const auto q0_field = fields.find("q0_passes_total");
  if (q0_field == fields.end()) throw std::runtime_error("stage Q0 pass counter missing");
  const auto q0_passed = strict_counter(q0_field->second);
  const auto operator_field = fields.find("tuple_operator");
  const auto tested_field = fields.find("member_tested");
  const auto passed_field = fields.find("member_prp_passed");
  const auto pending_field = fields.find("tuple_pending");
  const bool r12 = operator_field != fields.end() &&
      operator_field->second == kQueuedOptionalOperator;
  const bool explicit_fields = tested_field != fields.end() || passed_field != fields.end();
  if (operator_field != fields.end() && operator_field->second.empty())
    throw std::runtime_error("empty stage tuple operator");
  if ((operator_field != fields.end() || explicit_fields) &&
      (tested_field == fields.end() || passed_field == fields.end()))
    throw std::runtime_error("stage tuple operator requires explicit member counters");
  if (r12 && pending_field == fields.end())
    throw std::runtime_error("R12 pending tuple counter missing");
  std::optional<std::uint64_t> pending;
  if (pending_field != fields.end()) {
    pending = strict_counter(pending_field->second);
    if (*pending > q0_passed) throw std::runtime_error("stage pending exceeds Q0 passes");
  }
  if (explicit_fields) {
    append_physical_members(event, parse_member_csv(tested_field->second),
                            parse_member_csv(passed_field->second), q0_passed, r12);
  } else {
    // Unchanged legacy ABI: no tuple operator and no explicit member counters.
    boost::json::array tested;
    for (unsigned member = 0U; member < 6U; ++member) tested.push_back(q0_passed);
    event["member_tested"] = std::move(tested);
    event["member_tested_scope"] = "all_six_per_q0_pass_current_stage_v1";
  }
  if (operator_field != fields.end()) event["tuple_operator"] = operator_field->second;
  if (pending) event["tuple_pending"] = *pending;
}

void append_stage_observability(boost::json::object& event,
                                const boost::json::object& summary) {
  std::string tuple_operator;
  if (const auto* operator_value = summary.if_contains("tuple_operator")) {
    const auto& name = operator_value->as_object().at("name").as_string();
    tuple_operator.assign(name.data(), name.size());
    if (tuple_operator.empty()) throw std::runtime_error("empty stage tuple operator");
  }
  const bool r12 = tuple_operator == kQueuedOptionalOperator;
  if (!tuple_operator.empty()) {
    const auto* funnel = summary.if_contains("funnel");
    if (funnel == nullptr || !funnel->is_object() ||
        !funnel->as_object().contains("member_tested") ||
        !funnel->as_object().contains("member_prp_passed"))
      throw std::runtime_error("stage tuple operator requires explicit terminal member counters");
  }
  if (const auto* model = summary.if_contains("sieve_model")) {
    event["sieve_model"] = *model;
    for (const auto& field : model->as_object())
      event[std::string(field.key())] = field.value();
  }
  if (const auto* counts_value = summary.if_contains("counts")) {
    const auto& counts = counts_value->as_object();
    const auto* classifier_value = counts.if_contains("tuple_counts");
    const auto* declared_funnel = summary.if_contains("funnel");
    const bool min5_classifier = declared_funnel != nullptr &&
        declared_funnel->is_object() &&
        declared_funnel->as_object().if_contains("active_after_member") != nullptr;
    if (classifier_value != nullptr && min5_classifier) {
      const auto& classifier = classifier_value->as_array();
      if (classifier.size() != 8U)
        throw std::runtime_error("stage min-five classifier width mismatch");
      boost::json::array ratios;
      for (std::size_t i = 0U; i < 7U; ++i) {
        const auto before = classifier[i].to_number<std::uint64_t>();
        const auto after = classifier[i + 1U].to_number<std::uint64_t>();
        if (after > before)
          throw std::runtime_error("stage min-five classifier conservation failed");
        if (after == 0U) ratios.push_back(nullptr);
        else ratios.push_back(static_cast<double>(before) / static_cast<double>(after));
      }
      event["classifier_counts"] = classifier;
      event["classifier_r_vector"] = std::move(ratios);
      event["classifier_scope"] = "minimum_five_viability_with_late_holes_not_q7";
      const auto entered = classifier.front().to_number<std::uint64_t>();
      const auto complete = classifier.back().to_number<std::uint64_t>();
      event["r_star_min5_classifier"] = complete == 0U
          ? boost::json::value(nullptr)
          : boost::json::value(std::pow(static_cast<double>(entered) /
                                       static_cast<double>(complete), 1.0 / 7.0));
      if (const auto* proof_value = counts.if_contains("proofs")) {
        const auto exact = proof_value->to_number<std::uint64_t>();
        if (exact > complete)
          throw std::runtime_error("stage exact min-five exceeds classifier completion");
        event["r_star_min5_exact"] = exact == 0U
            ? boost::json::value(nullptr)
            : boost::json::value(std::pow(static_cast<double>(entered) /
                                         static_cast<double>(exact), 1.0 / 7.0));
        event["r_star_min5_scope"] = "seventh_root_of_entered_per_exact_min5_not_q7";
      }
    }
  }
  if (const auto* funnel_value = summary.if_contains("funnel")) {
    const auto& funnel = funnel_value->as_object();
    if (funnel.contains("member_tested") && !funnel.contains("member_prp_passed"))
      throw std::runtime_error("stage explicit member pass counters missing");
    if (const auto* passed_value = funnel.if_contains("member_prp_passed")) {
      const auto& passed = passed_value->as_array();
      const auto q0_passed = strict_json_counter(summary.at("counts").as_object().at("q0_passed"));
      boost::json::array tested;
      if (const auto* explicit_tested = funnel.if_contains("member_tested")) {
        tested = explicit_tested->as_array();
      } else {
        // Silver/private-macro ABI v1 physically tests all six members for
        // every Q0 pass. The min-five active funnel is NOT the tested count.
        for (unsigned member = 0U; member < 6U; ++member)
          tested.push_back(q0_passed);
      }
      append_physical_members(event, tested, passed, q0_passed, r12);
    }
  }
  if (!tuple_operator.empty()) {
    event["tuple_operator"] = tuple_operator;
    const auto& metadata = summary.at("tuple_operator").as_object();
    if (const auto* pending_value = metadata.if_contains("pending")) {
      const auto pending = strict_json_counter(*pending_value);
      const auto q0_passed = strict_json_counter(summary.at("counts").as_object().at("q0_passed"));
      if (pending > q0_passed || (r12 && pending != 0U))
        throw std::runtime_error("stage terminal tuple queue not drained");
      event["tuple_pending"] = pending;
    }
  }
  const auto* rlow_value = summary.if_contains("rlow");
  if (rlow_value == nullptr) return;
  const auto& rlow = rlow_value->as_object();
  const auto& prefix = rlow.at("strict_prefix_counts").as_array();
  const auto& ratios = rlow.at("r_vector").as_array();
  if (prefix.size() != 8U || ratios.size() != 7U)
    throw std::runtime_error("RLOW prefix/r-vector width mismatch");
  for (std::size_t i = 0U; i < 7U; ++i) {
    const auto before = prefix[i].to_number<std::uint64_t>();
    const auto after = prefix[i + 1U].to_number<std::uint64_t>();
    if (after > before) throw std::runtime_error("RLOW prefix conservation failed");
    if (after == 0U) {
      if (!ratios[i].is_null())
        throw std::runtime_error("RLOW zero-denominator ratio must be unknown");
    } else {
      const double expected = static_cast<double>(before) / static_cast<double>(after);
      const double observed = ratios[i].to_number<double>();
      if (!std::isfinite(observed) ||
          std::abs(observed - expected) > std::max(0.00001, expected * 0.000001))
        throw std::runtime_error("RLOW ratio does not match measured prefix");
    }
  }
  const auto q7_exact = rlow.at("q7_exact").to_number<std::uint64_t>();
  if (q7_exact > prefix.back().to_number<std::uint64_t>())
    throw std::runtime_error("RLOW exact Q7 exceeds the measured all-seven queue");
  event["rlow"] = rlow;
  event["strict_prefix_counts"] = prefix;
  event["r_vector"] = ratios;
  event["q7_strict_r_vector"] = ratios;
  event["r_vector_scope"] = "consecutive_prime_prefix_for_all_seven_not_min5";
  event["r_global"] = rlow.at("r_global");
  event["q7_exact"] = q7_exact;
  event["q7_exact_s"] = rlow.at("q7_exact_s");
  event["r_star_q7_exact"] = q7_exact == 0U
      ? boost::json::value(nullptr)
      : boost::json::value(std::pow(
          static_cast<double>(prefix.front().to_number<std::uint64_t>()) /
          static_cast<double>(q7_exact), 1.0 / 7.0));
  event["q7_metric_boundary"] = "exact_before_network_submission";
}

void append_live_wall_rate(boost::json::object& event,
                          std::uint64_t entered,
                          double component_candidates_s,
                          double bridge_elapsed_s) {
  if (!std::isfinite(bridge_elapsed_s) || bridge_elapsed_s <= 0.0 ||
      !std::isfinite(component_candidates_s) || component_candidates_s < 0.0)
    throw std::runtime_error("invalid live wall-rate observation");
  event["component_q0_candidates_s"] = component_candidates_s;
  event["bridge_elapsed_s"] = bridge_elapsed_s;
  event["e2e_c_s"] = static_cast<double>(entered) / bridge_elapsed_s;
  event["rate_scope"] = "entered_candidates_per_bridge_wall_second_pre_submit";
}

std::vector<boost::json::object> StageBridge::drain_progress_events() {
  std::vector<boost::json::object> events;
  if (!impl_->work || impl_->log.empty() || !fs::exists(impl_->log)) return events;
  const std::uintmax_t size = fs::file_size(impl_->log);
  if (size > kJournalLimit || size < impl_->progress_offset)
    throw std::runtime_error("stage progress log oversized or truncated");
  if (size == impl_->progress_offset) return events;
  std::ifstream input(impl_->log, std::ios::binary);
  input.seekg(static_cast<std::streamoff>(impl_->progress_offset));
  std::string chunk(static_cast<std::size_t>(size - impl_->progress_offset), '\0');
  input.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
  const auto consumed = static_cast<std::uintmax_t>(input.gcount());
  impl_->progress_offset += consumed;
  chunk.resize(static_cast<std::size_t>(consumed));
  impl_->progress_partial += chunk;
  std::size_t newline = 0U;
  while ((newline = impl_->progress_partial.find('\n')) != std::string::npos) {
    std::string line = impl_->progress_partial.substr(0U, newline);
    impl_->progress_partial.erase(0U, newline + 1U);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    require_no_stage_failure_evidence(line);
    const bool rlow_terminal = line.rfind("phase=rlow_e2e_terminal ", 0U) == 0U;
    if (!rlow_terminal && line.rfind("phase=q0_measure_progress ", 0U) != 0U) continue;
    std::map<std::string, std::string> fields;
    std::istringstream tokens(line);
    std::string token;
    while (tokens >> token) {
      const auto equals = token.find('=');
      if (equals != std::string::npos) {
        const auto inserted = fields.emplace(token.substr(0U, equals), token.substr(equals + 1U));
        const auto& key = inserted.first->first;
        if (!inserted.second && (key == "tuple_operator" || key == "member_tested" ||
            key == "member_prp_passed" || key == "tuple_pending" || key == "q0_passes_total" ||
            key == "exact_results_journaled"))
          throw std::runtime_error("duplicate stage member telemetry field");
      }
    }
    auto number = [&](const char* key) -> double {
      const auto it = fields.find(key);
      if (it == fields.end()) throw std::runtime_error("stage progress field missing");
      return std::stod(it->second);
    };
    auto integer = [&](const char* key) -> std::uint64_t {
      const auto it = fields.find(key);
      if (it == fields.end()) throw std::runtime_error("stage progress field missing");
      return static_cast<std::uint64_t>(std::stoull(it->second));
    };
    if (rlow_terminal) {
      const auto raw = fields.find("observed_funnel");
      if (raw == fields.end()) throw std::runtime_error("RLOW terminal funnel missing");
      const auto parsed = boost::json::parse(raw->second);
      boost::json::object event{{"schema", kEventSchema},
          {"phase", "rlow_e2e_terminal"},
          {"measurement_layer", "production_multibatch_pre_submit"},
          {"terminal_status", "awaiting_exact_replay"},
          {"work_id", impl_->work->work_id},
          {"template_id", impl_->work->template_id},
          {"gpu_uuid", impl_->config.gpu_uuid},
          {"device_index", impl_->config.device_index},
          {"e2e_c_s", number("e2e_c_s")}};
      append_stage_observability(event, boost::json::object{{"rlow", parsed}});
      append_live_wall_rate(event,
          parsed.as_object().at("strict_prefix_counts").as_array().front()
              .to_number<std::uint64_t>(),
          number("e2e_c_s"),
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        impl_->started).count());
      events.push_back(std::move(event));
      continue;
    }
    const double q0_entered = number("q0_entries");
    const auto q0_passed_count = strict_counter(fields.at("q0_passes_total"));
    const double q0_passed = static_cast<double>(q0_passed_count);
    boost::json::object event{{"schema", kEventSchema},
                              {"phase", "q0_measure_progress"},
                              {"measurement_layer", "production_multibatch_pre_submit"},
                              {"terminal_status", "running"},
                               {"work_id", impl_->work->work_id},
                               {"template_id", impl_->work->template_id},
                               {"target_bits", impl_->work->target_bits},
                               {"gpu_uuid", impl_->config.gpu_uuid},
                              {"device_index", impl_->config.device_index},
                              {"elapsed_s", number("elapsed_s")},
                              {"positions", integer("positions")},
                               {"q0_entered", static_cast<std::uint64_t>(q0_entered)},
                               {"q0_passed", q0_passed_count},
                               {"q0_candidates_per_s", number("q0_candidates_per_s")},
                               {"component_q0_candidates_s", number("q0_candidates_per_s")},
                               {"positions_per_s", number("q0_positions_per_s")},
                               {"q0_positions_per_s", number("q0_positions_per_s")},
                               {"prp_s_secondary", number("q0_prp_per_s")},
                               {"q0_prp_per_s", number("q0_prp_per_s")},
                              {"tuple_counts", fields.at("tuple_counts")},
                              {"proofs", integer("exact_results_journaled")},
                              {"factor_origin_begin", integer("factor_origin_begin")},
                              {"factor_origin_next", integer("factor_origin_next")},
                              {"accepted", 0}, {"rejected", 0},
                              {"stale", 0}, {"errors", 0}};
    append_live_wall_rate(event, static_cast<std::uint64_t>(q0_entered),
        number("q0_candidates_per_s"),
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      impl_->started).count());
    for (const char* key : {"actual_prime_limit", "configured_prime_limit",
                            "prime_count", "prime_product", "r_model",
                            "r_model_bits"}) {
      if (fields.find(key) != fields.end()) event[key] = number(key);
    }
    if (q0_passed > 0.0) {
      const double measured_r = q0_entered / q0_passed;
      event["r"] = measured_r;
      event["r_defined"] = true;
      event["b_d"] = 86400.0 * event.at("e2e_c_s").to_number<double>() /
                       std::pow(measured_r, 7.0);
      event["b_d_scope"] = "uniform_member_density_estimate_not_observed_blocks";
    } else {
      event["r"] = nullptr;
      event["r_defined"] = false;
      event["b_d"] = nullptr;
    }
    append_live_member_observability(event, fields);
    const auto committed = strict_counter(fields.at("exact_results_journaled"));
    if ((impl_->committed_proofs && committed < *impl_->committed_proofs) ||
        committed > q0_passed_count)
      throw std::runtime_error("stage committed proof watermark invalid");
    impl_->committed_proofs = committed;
    events.push_back(std::move(event));
  }
  return events;
}

std::optional<StageProof> StageBridge::poll_exact_proof() {
  if (!impl_->work || !impl_->proofs || !impl_->committed_proofs ||
      impl_->config.production_session_seconds == 0U)
    return {};  // Old/non-streaming stages retain the complete terminal path.
  const auto process_state = WaitForSingleObject(impl_->process.hProcess, 0U);
  if (process_state == WAIT_OBJECT_0) return {};  // Terminal/global guards run first.
  if (process_state != WAIT_TIMEOUT)
    throw std::runtime_error("live proof stage state unavailable");
  const auto share = impl_->proofs->admit_next(*impl_->committed_proofs);
  if (!share) return {};
  StageProof proof{*impl_->work, *share, {}};
  proof.admission_event = {{"schema", kEventSchema}, {"phase", "share_proof_admitted"},
      {"work_id", proof.work.work_id}, {"template_id", proof.work.template_id},
      {"clean_epoch", proof.work.clean_epoch}, {"runtime_sha256", impl_->proofs->runtime_sha256()},
      {"nonce_v1_uint256_hex", share->nonce_v1_uint256_hex}, {"prime_count", share->prime_count},
      {"proof_scope", "individual_gmp32_min5_with_holes_not_stage_coverage"},
      {"stage_complete", false}, {"admitted_results", impl_->proofs->admitted_count()},
      {"committed_results", *impl_->committed_proofs},
      {"admission_journal", impl_->admissions.string()},
      {"submission_status", "not_sent"}, {"terminal_status", "individually_validated"}};
  return proof;
}

std::optional<boost::json::object> StageBridge::cancel(const std::string& reason,
                                                      bool require_retry_proof) {
  if (!running()) return {};
  const WorkAssignment cancelled = *impl_->work;
  bool own_termination = false;
  DWORD exit_code = STILL_ACTIVE;
  StageRetryDisposition disposition = StageRetryDisposition::cancelled;
  std::optional<InterruptedJournalInspection> interrupted_journal;
  if (require_retry_proof) {
    const DWORD initial_wait = WaitForSingleObject(impl_->process.hProcess, 0U);
    if (initial_wait != WAIT_OBJECT_0 && initial_wait != WAIT_TIMEOUT)
      throw std::runtime_error("stage_invalidation_process_state_unavailable");
    if (initial_wait == WAIT_TIMEOUT)
      own_termination = TerminateProcess(impl_->process.hProcess,
                                         kCancelledStageExitCode) != FALSE;
    const DWORD stopped = WaitForSingleObject(impl_->process.hProcess, 5000U);
    if (!GetExitCodeProcess(impl_->process.hProcess, &exit_code))
      throw std::runtime_error("stage_invalidation_exit_status_unavailable");
    disposition = stage_retry_disposition(stopped == WAIT_OBJECT_0,
                                         own_termination, exit_code);
  } else {
    TerminateProcess(impl_->process.hProcess, kCancelledStageExitCode);
    WaitForSingleObject(impl_->process.hProcess, 5000U);
  }
  impl_->close_handles();
  if (require_retry_proof) {
    // The child is stopped: inspect the entire bounded artifact, including a
    // failure flushed after the last live-progress poll. Never erase/rewrite it.
    if (!fs::is_regular_file(impl_->log) ||
        fs::file_size(impl_->log) > 16U * 1024U * 1024U)
      throw std::runtime_error("stage_invalidation_log_missing_or_oversized");
    std::ifstream log(impl_->log, std::ios::binary);
    std::string line;
    while (std::getline(log, line)) require_no_stage_failure_evidence(line);
    if (!log.eof()) throw std::runtime_error("stage_invalidation_log_read_failed");
    impl_->proofs->verify_admitted_prefix();
    const auto inspect_complete_artifact = [&](const fs::path& path) {
      if (!fs::exists(path)) return;  // A genuinely unfinished stage may have none.
      for (const auto& record : read_journal(path)) {
        require_string(record, "work_id", cancelled.work_id);
        require_string(record, "template_id", cancelled.template_id);
        inspect_failure_value(record);
      }
    };
    if (disposition == StageRetryDisposition::cancelled) {
      const auto committed = impl_->committed_proofs.value_or(0U);
      if (fs::exists(impl_->journal))
        interrupted_journal = inspect_interrupted_journal(impl_->journal, cancelled, committed);
      else if (committed != 0U)
        throw std::runtime_error("committed proof journal missing during invalidation");
    } else {
      inspect_complete_artifact(impl_->journal);
    }
    // The summary is atomically published, not an append-in-progress journal.
    // Its existing strict parser and the completed-stage validator are unchanged.
    inspect_complete_artifact(impl_->summary);
    if (disposition == StageRetryDisposition::completed)
      (void)impl_->proofs->complete(impl_->summary, false);
  }
  impl_->work.reset();
  boost::json::object event{{"schema", kEventSchema}, {"phase", "work_cancelled"},
                              {"work_id", cancelled.work_id},
                              {"template_id", cancelled.template_id},
                              {"factor_origin", cancelled.factor_origin},
                               {"positions", cancelled.positions},
                               {"reason", reason}, {"terminal_status", "interrupted"}};
  const auto admitted = impl_->proofs ? impl_->proofs->admitted_count() : 0U;
  event["individually_admitted_results"] = admitted;
  event["admission_journal"] = impl_->admissions.string();
  if (reason.rfind("pool_work_invalidated", 0U) == 0U) {
    event["unadmitted_results_status"] = "obsolete_unadmitted_not_submitted";
    event["committed_results_seen"] = impl_->committed_proofs
        ? boost::json::value(*impl_->committed_proofs) : boost::json::value(nullptr);
    event["unadmitted_committed_results_seen"] = impl_->committed_proofs
        ? boost::json::value(*impl_->committed_proofs - admitted) : boost::json::value(nullptr);
    event["result_journal"] = impl_->journal.string();
  }
  if (require_retry_proof) {
    event["stop_confirmed"] = true;
    event["retry_safe"] = true;
    event["own_termination_succeeded"] = own_termination;
    event["stage_exit_code"] = static_cast<std::uint64_t>(exit_code);
    event["completed_before_invalidation"] = disposition == StageRetryDisposition::completed;
    event["stage_wall_s"] = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - impl_->started).count();
    event["failure_evidence_checked"] = true;
    event["partial_results_status"] = admitted == 0U
        ? "retained_not_admitted_not_submitted"
        : "individual_proofs_admitted_stage_coverage_not_credited";
    event["stage_log"] = impl_->log.string();
    event["result_journal"] = impl_->journal.string();
    event["stage_summary"] = impl_->summary.string();
    if (interrupted_journal && interrupted_journal->tail_bytes != 0U) {
      event["uncommitted_journal_tail_offset"] = interrupted_journal->tail_offset;
      event["uncommitted_journal_tail_bytes"] = interrupted_journal->tail_bytes;
      event["uncommitted_journal_tail_sha256"] = interrupted_journal->tail_sha256;
      event["uncommitted_journal_tail_status"] = "retained_not_admitted_not_submitted";
    }
  }
  return event;
}

std::optional<StageOutcome> StageBridge::poll() {
  if (!running() || WaitForSingleObject(impl_->process.hProcess, 0U) == WAIT_TIMEOUT)
    return {};
  DWORD exit_code = 1U;
  if (!GetExitCodeProcess(impl_->process.hProcess, &exit_code))
    throw std::runtime_error("stage exit status unavailable");
  impl_->close_handles();
  impl_->work.reset();
  if (exit_code != 0U) throw std::runtime_error("stage process failed closed");
  auto outcome = impl_->proofs->complete(impl_->summary, true,
                                       impl_->operator_drain_requested);
  const double wall_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - impl_->started).count();
  const double q0_entered = outcome.terminal_event.at("q0_entered").to_number<double>();
  const double q0_passed = outcome.terminal_event.at("q0_passed").to_number<double>();
  const double e2e_c_s = wall_seconds > 0.0 ? q0_entered / wall_seconds : 0.0;
  outcome.terminal_event["batch_wall_s"] = wall_seconds;
  outcome.terminal_event["e2e_c_s"] = e2e_c_s;
  if (q0_passed > 0.0) {
    const double measured_r = q0_entered / q0_passed;
    outcome.terminal_event["r"] = measured_r;
    outcome.terminal_event["r_defined"] = true;
    outcome.terminal_event["b_d"] =
        86400.0 * e2e_c_s / std::pow(measured_r, 7.0);
  } else {
    outcome.terminal_event["r"] = nullptr;
    outcome.terminal_event["r_defined"] = false;
    outcome.terminal_event["b_d"] = nullptr;
  }
  return outcome;
}

StageOutcome StageJournalReader::complete(const fs::path& summary,
                                          bool publish_new_admissions,
                                          bool operator_drain_requested) {
  try {
  if (impl_->poisoned || impl_->finished)
    throw std::runtime_error("proof reader is closed");
  const auto& context = impl_->context;
  const auto& config = context.config;
  const auto& work = context.work;
  const auto records = read_journal(impl_->journal);
  bool terminal_seen = false;
  std::uint64_t terminal_interesting = 0;
  std::uint64_t terminal_journaled = 0;
  std::set<std::string> nonces;
  StageOutcome outcome;
  outcome.work = work;
  for (std::size_t record_index = 0; record_index < records.size(); ++record_index) {
    const auto& record = records[record_index];
    require_proof_binding(context, record);
    const std::string schema(record.at("schema").as_string().c_str());
    if (schema == "riecoin-terminal-v1") {
      if (terminal_seen || record_index + 1U != records.size())
        throw std::runtime_error("stage terminal must be unique and last");
      terminal_seen = true;
      require_string(record, "proof_state", "exact_pass");
      require_string(record, "terminal_status", "completed");
      if (record.at("errors").to_number<std::uint64_t>() != 0U)
        throw std::runtime_error("stage terminal exactness failed");
      terminal_interesting = record.at("interesting_results").to_number<std::uint64_t>();
      terminal_journaled = record.at("results_journaled").to_number<std::uint64_t>();
      continue;
    }
    const std::string nonce(record.at("nonce_v1_uint256_hex").as_string().c_str());
    if (!nonces.insert(nonce).second)
      throw std::runtime_error("share nonce binding/uniqueness failed");
    if (record_index < impl_->cache.size()) {
      const auto& cached = impl_->cache[record_index];
      if (cached.record != boost::json::serialize(record))
        throw std::runtime_error("previously admitted proof changed in terminal journal");
      outcome.shares.push_back(cached.share);  // Same full record, no second GMP replay.
    } else {
      outcome.shares.push_back(validate_individual_proof(context, record));
      ++impl_->replays;
    }
  }
  if (!terminal_seen) throw std::runtime_error("stage terminal record missing");
  if (terminal_interesting != outcome.shares.size() ||
      terminal_journaled != outcome.shares.size() ||
      outcome.shares.size() < impl_->cache.size() ||
      terminal_journaled < impl_->committed)
    throw std::runtime_error("stage terminal/share count conservation failed");
  const auto validated_summary = validate_summary(summary, config, work,
      outcome.shares.size(), operator_drain_requested);
  const auto& rates = validated_summary.at("rates").as_object();
  const auto& counts = validated_summary.at("counts").as_object();
  const auto& geometry = validated_summary.at("geometry").as_object();
  const auto completed_end = geometry.at("factor_origin_end").to_number<std::uint64_t>();
  for (std::size_t index = 0U; index < outcome.shares.size(); ++index)
    if (decimal_u64(records[index].at("factor")) >= completed_end)
      throw std::runtime_error("admitted proof lies outside completed stage coverage");
  const std::uint64_t total_positions =
      geometry.at("positions").to_number<std::uint64_t>();
  outcome.terminal_event = {
      {"schema", kEventSchema}, {"phase", "stage_complete"},
      {"work_id", work.work_id}, {"template_id", work.template_id},
      {"target_bits", work.target_bits},
      {"factor_origin", work.factor_origin},
      {"factor_origin_end", geometry.at("factor_origin_end")},
      {"positions", total_positions}, {"batch_positions", work.positions},
      {"batches", geometry.at("batches")},
      {"device_index", config.device_index}, {"gpu_uuid", config.gpu_uuid},
      {"proofs", outcome.shares.size()}, {"exactness", "pass"},
      {"component_q0_candidates_s", rates.at("q0_candidates_s")},
      {"positions_per_s", rates.at("positions_s")},
      {"prp_s_secondary", rates.at("prp_s_secondary")},
      {"prp_per_candidate", rates.at("prp_per_candidate")},
      {"r_emp", rates.at("r_emp")},
      {"r_emp_defined", rates.at("r_emp_defined")},
      {"survivors", counts.at("survivors")},
      {"q0_entered", counts.at("q0_entries")},
      {"q0_passed", counts.at("q0_passed")},
      {"exact_results", outcome.shares.size()},
      {"tuple_counts", counts.at("tuple_counts")},
      {"funnel", validated_summary.if_contains("funnel")
          ? validated_summary.at("funnel") : boost::json::value(nullptr)},
      {"rejected_audit", validated_summary.at("exactness").as_object().if_contains("rejected_audit")
          ? validated_summary.at("exactness").as_object().at("rejected_audit")
          : boost::json::value(nullptr)},
      {"conservation", validated_summary.if_contains("conservation")
          ? validated_summary.at("conservation")
          : boost::json::value(boost::json::object{
              {"range_positions", total_positions},
              {"terminal_results", terminal_journaled},
              {"validated_results", outcome.shares.size()}, {"pass", true}})},
      {"terminal_status", "completed"}};
  append_stage_observability(outcome.terminal_event, validated_summary);
  outcome.terminal_event["individually_admitted_before_terminal"] = admitted_count();
  outcome.terminal_event["backend_gmp_replays"] = replay_count();
  if (publish_new_admissions && !impl_->admissions.empty()) {
    for (std::size_t index = impl_->cache.size(); index < outcome.shares.size(); ++index)
      append_proof_admission(impl_->admissions, context, records[index], outcome.shares[index], false);
    outcome.terminal_event["admission_journal"] = impl_->admissions.string();
  }
  impl_->finished = true;
  return outcome;
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

StageOutcome StageBridge::validate_completed_journal(
    const StageConfig& config, const WorkAssignment& work,
    const fs::path& journal, const fs::path& summary) {
  StageJournalReader reader(config, work, journal);
  return reader.complete(summary, false);
}

}  // namespace riecoin::network

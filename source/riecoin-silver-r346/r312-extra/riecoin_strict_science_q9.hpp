#pragma once

#include <gmp.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

namespace riecoin_strict_science_q9 {

constexpr std::array<uint32_t, 7> kPattern0Q7{{0u, 2u, 6u, 8u, 12u, 18u, 20u}};
constexpr std::array<uint32_t, 7> kPattern1Q7{{0u, 2u, 8u, 12u, 14u, 18u, 20u}};
constexpr std::array<int32_t, 8> kPattern0Q8{{0, 2, 6, 8, 12, 18, 20, 26}};
constexpr std::array<int32_t, 9> kPattern0Q9{{0, 2, 6, 8, 12, 18, 20, 26, 30}};
constexpr std::array<int32_t, 8> kPattern1Q8{{-6, 0, 2, 8, 12, 14, 18, 20}};
constexpr std::array<int32_t, 9> kPattern1Q9{{-10, -6, 0, 2, 8, 12, 14, 18, 20}};

struct Result {
  uint32_t length = 0u;
  uint32_t offset_count = 0u;
  std::array<int32_t, 9> offsets{};
  bool q7_replayed = false;
};

inline bool probable_prime_at_offset(mpz_srcptr first, int32_t offset,
                                     int rounds = 64) {
  mpz_t value;
  mpz_init_set(value, first);
  if (offset >= 0) {
    mpz_add_ui(value, value, static_cast<unsigned long>(offset));
  } else {
    mpz_sub_ui(value, value,
               static_cast<unsigned long>(-static_cast<int64_t>(offset)));
  }
  const bool probable_prime = mpz_probab_prime_p(value, rounds) != 0;
  mpz_clear(value);
  return probable_prime;
}

template <std::size_t N>
inline void set_offsets(Result& result, const std::array<int32_t, N>& offsets) {
  result.offsets.fill(0);
  for (std::size_t i = 0; i < N; ++i) result.offsets[i] = offsets[i];
  result.offset_count = static_cast<uint32_t>(N);
  result.length = static_cast<uint32_t>(N);
}

inline Result verify(mpz_srcptr first, uint32_t pattern,
                     const std::array<uint32_t, 7>& consensus_offsets,
                     int rounds = 64) {
  Result result;
  if (pattern > 1u) return result;
  const auto& expected = pattern == 0u ? kPattern0Q7 : kPattern1Q7;
  if (consensus_offsets != expected) return result;

  for (const uint32_t offset : expected) {
    if (!probable_prime_at_offset(first, static_cast<int32_t>(offset), rounds))
      return result;
  }
  result.q7_replayed = true;
  result.length = 7u;
  result.offset_count = 7u;
  for (std::size_t i = 0; i < expected.size(); ++i)
    result.offsets[i] = static_cast<int32_t>(expected[i]);

  const int32_t q8_extension = pattern == 0u ? 26 : -6;
  if (!probable_prime_at_offset(first, q8_extension, rounds)) return result;
  if (pattern == 0u) set_offsets(result, kPattern0Q8);
  else set_offsets(result, kPattern1Q8);

  const int32_t q9_extension = pattern == 0u ? 30 : -10;
  if (!probable_prime_at_offset(first, q9_extension, rounds)) return result;
  if (pattern == 0u) set_offsets(result, kPattern0Q9);
  else set_offsets(result, kPattern1Q9);
  return result;
}

inline std::string json_fields(const Result& result) {
  std::ostringstream out;
  out << ",\"strict_science_length\":" << result.length
      << ",\"strict_science_offsets\":[";
  for (uint32_t i = 0u; i < result.offset_count; ++i) {
    if (i != 0u) out << ',';
    out << result.offsets[i];
  }
  out << "]"
      << ",\"strict_science_q7_replayed\":"
      << (result.q7_replayed ? "true" : "false")
      << ",\"strict_science_verifier\":\"gmp_probab_prime_64_rare_tail\""
      << ",\"strict_science_scope\":\"independent_probable_prime_replay_not_certificate\"";
  return out.str();
}

}  // namespace riecoin_strict_science_q9

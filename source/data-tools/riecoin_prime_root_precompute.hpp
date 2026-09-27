#pragma once

#include <gmp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

// CPU-only, stateless setup. No template, target, primorial or root cache.
// Integration points in riecoin-rlow-e2e-queued.cu: primes_to(), then the
// prime-major/member-minor first_factors loop after base and primorial exist.
namespace riecoin_prime_root_precompute {

inline std::vector<std::uint32_t> primes_to(std::uint32_t limit) {
  std::vector<std::uint32_t> primes;
  if (limit < 2U) return primes;
  // Capacity is a hint only: it never bounds the sieve or its output.
  const auto capacity = limit < 64U ? static_cast<std::size_t>(limit) :
      static_cast<std::size_t>(1.15 * static_cast<double>(limit) /
                               std::log(static_cast<double>(limit))) + 64U;
  primes.reserve(capacity);
  primes.push_back(2U);
  if (limit < 3U) return primes;

  std::uint32_t root = static_cast<std::uint32_t>(std::sqrt(static_cast<double>(limit)));
  while ((std::uint64_t(root) + 1U) * (std::uint64_t(root) + 1U) <= limit) ++root;
  while (std::uint64_t(root) * root > limit) --root;
  std::vector<std::uint8_t> small(static_cast<std::size_t>(root / 2U) + 1U, 0U);
  std::vector<std::uint32_t> factors;
  for (std::uint32_t p = 3U; p <= root; p += 2U) {
    if (small[p / 2U] != 0U) continue;
    factors.push_back(p);
    for (std::uint64_t n = std::uint64_t(p) * p; n <= root; n += 2U * p)
      small[static_cast<std::size_t>(n / 2U)] = 1U;
  }

  constexpr std::size_t kSegmentOdds = 32768U;
  std::vector<std::uint8_t> composite(kSegmentOdds);
  const std::uint64_t last = (std::uint64_t(limit) - 1U) | 1U;
  for (std::uint64_t low = 3U; low <= last;) {
    const auto high = (std::min)(last, low + 2U * (kSegmentOdds - 1U));
    const auto count = static_cast<std::size_t>((high - low) / 2U + 1U);
    std::fill_n(composite.begin(), count, std::uint8_t{0U});
    for (const auto p : factors) {
      const std::uint64_t square = std::uint64_t(p) * p;
      if (square > high) break;
      std::uint64_t first = (std::max)(square, ((low + p - 1U) / p) * p);
      if ((first & 1U) == 0U) first += p;
      for (std::uint64_t index = (first - low) / 2U; index < count; index += p)
        composite[static_cast<std::size_t>(index)] = 1U;
    }
    for (std::size_t index = 0U; index < count; ++index)
      if (composite[index] == 0U)
        primes.push_back(static_cast<std::uint32_t>(low + 2U * index));
    low = high + 2U;
  }
  return primes;
}

inline std::uint32_t inverse_mod(std::uint32_t value, std::uint32_t modulus) {
  if (modulus < 2U) throw std::invalid_argument("prime modulus must be at least two");
  std::int64_t t = 0, next_t = 1;
  // Euclidean remainders/quotients are uint32, even though signed Bezout
  // coefficients need int64. Do not pay for signed 64-bit division here.
  std::uint32_t r = modulus, next_r = value;
  while (next_r != 0) {
    const std::uint32_t quotient = r / next_r;
    const auto updated_t = t - std::int64_t(quotient) * next_t;
    const auto updated_r = r % next_r;
    t = next_t;
    next_t = updated_t;
    r = next_r;
    next_r = updated_r;
  }
  if (r != 1) throw std::runtime_error("non-invertible primorial residue");
  if (t < 0) t += modulus;
  return static_cast<std::uint32_t>(t);
}

inline std::uint32_t add_mod(std::uint32_t left, std::uint32_t right,
                             std::uint32_t modulus) noexcept {
  // Inputs are canonical. Their sum is below 2p, including uint32 edge p.
  const std::uint64_t sum = std::uint64_t(left) + right;
  return static_cast<std::uint32_t>(sum >= modulus ? sum - modulus : sum);
}

struct RootTable {
  std::vector<std::uint32_t> primes;
  std::vector<std::uint32_t> first_factors;
};

inline RootTable roots_for(const std::vector<std::uint32_t>& all_primes,
                          std::size_t first_prime,
                          mpz_srcptr primorial, mpz_srcptr base,
                          const std::array<std::uint32_t, 7>& offsets) {
  if (first_prime > all_primes.size() || mpz_sgn(primorial) <= 0)
    throw std::invalid_argument("invalid prime-root geometry");
  const auto count = all_primes.size() - first_prime;
  RootTable output;
  if (count > output.first_factors.max_size() / offsets.size())
    throw std::length_error("prime-root table size overflow");
  output.primes.assign(all_primes.begin() + static_cast<std::ptrdiff_t>(first_prime),
                       all_primes.end());
  output.first_factors.resize(count * offsets.size());

  std::array<std::uint32_t, 6> steps{};
  bool small_even_gaps = offsets.front() == 0U;
  for (std::size_t member = 1U; member < offsets.size(); ++member) {
    const auto gap = offsets[member] - offsets[member - 1U];
    const bool supported = offsets[member] >= offsets[member - 1U] &&
                           (gap == 2U || gap == 4U || gap == 6U);
    small_even_gaps = small_even_gaps && supported;
    if (supported) steps[member - 1U] = gap / 2U - 1U;
  }

  for (std::size_t index = 0U; index < count; ++index) {
    const auto p = output.primes[index];
    if (p < 2U) throw std::invalid_argument("invalid prime-root modulus");
    const auto base_mod = static_cast<std::uint32_t>(mpz_fdiv_ui(base, p));
    const auto inverse = inverse_mod(
        static_cast<std::uint32_t>(mpz_fdiv_ui(primorial, p)), p);
    auto* roots = output.first_factors.data() + index * offsets.size();
    if (!small_even_gaps) {
      // Exact generic fallback; no shape is silently filtered out.
      for (std::size_t member = 0U; member < offsets.size(); ++member) {
        const auto residue = (std::uint64_t(base_mod) + offsets[member]) % p;
        const auto negative = residue == 0U ? 0U : p - residue;
        roots[member] = static_cast<std::uint32_t>((negative * inverse) % p);
      }
      continue;
    }

    // P*r(o)+B+o == 0 (mod p). With I=P^-1 (mod p),
    // r(o+g) == r(o)-g*I (mod p). Thus one product reduction for r(0)
    // and canonical additions for gaps 2/4/6 recover the same seven roots.
    // This changes neither prime coverage nor the order of any output word.
    const auto negative = base_mod == 0U ? 0U : p - base_mod;
    std::uint32_t current = static_cast<std::uint32_t>((std::uint64_t(negative) * inverse) % p);
    const auto twice_inverse = add_mod(inverse, inverse, p);
    const auto step2 = twice_inverse == 0U ? 0U : p - twice_inverse;
    const auto step4 = add_mod(step2, step2, p);
    const std::array<std::uint32_t, 3> delta{{step2, step4, add_mod(step2, step4, p)}};
    roots[0] = current;
    for (std::size_t member = 1U; member < offsets.size(); ++member) {
      current = add_mod(current, delta[steps[member - 1U]], p);
      roots[member] = current;
    }
  }
  return output;
}

}  // namespace riecoin_prime_root_precompute

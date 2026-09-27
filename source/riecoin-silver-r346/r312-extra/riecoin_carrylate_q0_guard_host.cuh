#pragma once

// Include AFTER cuda_check, CUDA/GMP/common standard headers and the carry-late
// wrapper. In particular, do NOT include CGBN again: it is not idempotent here.
// Call once after the stage wall clock starts and the selected GPU is validated,
// before admitting mining candidates. The caller resets rare_paths afterwards;
// this oracle deliberately exercises unsupported inputs as well as the fast path.

namespace riecoin_carrylate_q0_guard_detail {

constexpr std::uint32_t kResidueWords = 36u;
constexpr std::uint32_t kCandidateWords = 40u;
constexpr std::uint32_t kCorpusSize = 100u;

class GmpInteger {
 public:
  GmpInteger() { mpz_init(value); }
  ~GmpInteger() { mpz_clear(value); }
  GmpInteger(const GmpInteger&) = delete;
  GmpInteger& operator=(const GmpInteger&) = delete;
  mpz_t value;
};

template <class T>
class DeviceAllocation {
 public:
  DeviceAllocation(std::size_t count, const char* operation) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)),
               operation);
  }
  ~DeviceAllocation() noexcept {
    if (pointer_ != nullptr) (void)cudaFree(pointer_);
  }
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  T* get() const noexcept { return pointer_; }
  void release_checked(const char* operation) {
    if (pointer_ == nullptr) return;
    const cudaError_t status = cudaFree(pointer_);
    if (status == cudaSuccess) pointer_ = nullptr;
    cuda_check(status, operation);
  }
 private:
  T* pointer_ = nullptr;
};

struct Fixture {
  cgbn_mem_t<1280u> candidate{};
  std::array<std::uint32_t, kResidueWords> residue{};
  std::uint8_t verdict = 0u;
  std::uint8_t used_fast = 0u;
  std::string label;
};

struct Report {
  std::uint32_t total = 0u;
  std::uint32_t checked = 0u;
  std::uint32_t mismatches = 0u;
  std::uint32_t fast = 0u;
  std::uint32_t fallback = 0u;
};

inline void append_fixture(std::vector<Fixture>& corpus, mpz_srcptr modulus,
                           const std::string& label, bool prime_fixture = false) {
  if (mpz_sgn(modulus) < 0 || mpz_sizeinbase(modulus, 2) > 1280u)
    throw std::runtime_error("carry-late guard fixture exceeds the input type");
  Fixture fixture;
  fixture.label = label;
  std::size_t words = 0u;
  mpz_export(fixture.candidate._limbs, &words, -1, sizeof(std::uint32_t),
             0, 0, modulus);
  if (words > kCandidateWords)
    throw std::runtime_error("carry-late guard input packing overflow");

  GmpInteger packed, exponent, base, residue, expected;
  mpz_import(packed.value, kCandidateWords, -1, sizeof(std::uint32_t),
             0, 0, fixture.candidate._limbs);
  if (mpz_cmp(packed.value, modulus) != 0)
    throw std::runtime_error("carry-late guard input packing mismatch");

  const std::size_t bits = mpz_sizeinbase(modulus, 2);
  // Match the existing dynamic Euler contract, NOT the helper that special-
  // cases small primality: 0,1,2,3 and all even inputs must return false.
  const bool valid_shape = bits >= 3u && mpz_odd_p(modulus) != 0;
  fixture.used_fast = valid_shape && bits <= 1152u ? 1u : 0u;
  if (valid_shape) {
    mpz_sub_ui(exponent.value, modulus, 1u);
    mpz_fdiv_q_2exp(exponent.value, exponent.value, 1u);
    mpz_set_ui(base.value, 2u);
    mpz_powm(residue.value, base.value, exponent.value, modulus);
    const int jacobi = mpz_jacobi(base.value, modulus);
    if (jacobi == 1) mpz_set_ui(expected.value, 1u);
    else if (jacobi == -1) mpz_sub_ui(expected.value, modulus, 1u);
    else throw std::runtime_error("carry-late guard unexpected Jacobi symbol");
    fixture.verdict = mpz_cmp(residue.value, expected.value) == 0 ? 1u : 0u;
    if (fixture.used_fast != 0u) {
      words = 0u;
      mpz_export(fixture.residue.data(), &words, -1, sizeof(std::uint32_t),
                 0, 0, residue.value);
      if (words > kResidueWords)
        throw std::runtime_error("carry-late guard expected residue overflow");
    }
  }
  if (prime_fixture && fixture.verdict != 1u)
    throw std::runtime_error("carry-late guard positive prime fixture changed");
  corpus.push_back(std::move(fixture));
}

inline std::vector<Fixture> make_corpus() {
  std::vector<Fixture> corpus;
  corpus.reserve(kCorpusSize);
  GmpInteger modulus, power;

  // 64 cases: every real width, all four odd classes mod 8, sparse lower
  // boundaries and dense upper boundaries. Includes both sides of R/2 and R.
  for (std::uint32_t bits = 1137u; bits <= 1152u; ++bits) {
    mpz_set_ui(power.value, 1u);
    mpz_mul_2exp(power.value, power.value, bits - 1u);
    for (unsigned long delta : {1ul, 3ul}) {
      mpz_add_ui(modulus.value, power.value, delta);
      append_fixture(corpus, modulus.value, "lower-" + std::to_string(bits) +
                     "+" + std::to_string(delta));
    }
    mpz_mul_2exp(power.value, power.value, 1u);
    for (unsigned long delta : {3ul, 1ul}) {
      mpz_sub_ui(modulus.value, power.value, delta);
      append_fixture(corpus, modulus.value, "upper-" + std::to_string(bits) +
                     "-" + std::to_string(delta));
    }
  }

  // 16 small inputs: nine supported odd inputs and seven invalid shapes.
  for (unsigned long fixture_value : {0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul,
                                      8ul, 9ul, 11ul, 13ul, 15ul, 17ul, 25ul, 31ul}) {
    mpz_set_ui(modulus.value, fixture_value);
    append_fixture(corpus, modulus.value, "small-" + std::to_string(fixture_value));
  }

  // Frozen constants p = 2^(bits-1) + addend. Six supported fixtures reproduce
  // the CPU preparation test's mpz_nextprime(2^(bits-1)+0x12345); two more
  // exercise a TRUE fallback verdict. All eight were checked offline with
  // mpz_probab_prime_p(p,64). No prime search or primality retest at stage time.
  struct PrimeFixture { std::uint32_t bits; unsigned long addend; };
  constexpr PrimeFixture primes[] = {
      {1137u, 0x12b1ful}, {1139u, 0x12543ul}, {1145u, 0x12403ul},
      {1150u, 0x1285dul}, {1151u, 0x126dful}, {1152u, 0x124fdul},
      {1153u, 0x126e5ul}, {1280u, 0x12c3bul}};
  for (const PrimeFixture& prime : primes) {
    mpz_set_ui(modulus.value, 1u);
    mpz_mul_2exp(modulus.value, modulus.value, prime.bits - 1u);
    mpz_add_ui(modulus.value, modulus.value, prime.addend);
    append_fixture(corpus, modulus.value,
                   "frozen-prime-" + std::to_string(prime.bits), true);
  }

  // Four dense carry patterns, including high coefficients and low n_0 values
  // that make the inverse coefficients nontrivial across the whole radix.
  constexpr std::uint32_t patterns[4][2] = {
      {0xffffffffu, 3u}, {0xffffffffu, 0x80000001u},
      {0xaaaaaaaau, 5u}, {0x55555555u, 7u}};
  for (std::uint32_t index = 0u; index < 4u; ++index) {
    std::array<std::uint32_t, kResidueWords> limbs{};
    limbs.fill(patterns[index][0]);
    limbs[0] = patterns[index][1];
    mpz_import(modulus.value, limbs.size(), -1, sizeof(std::uint32_t),
               0, 0, limbs.data());
    append_fixture(corpus, modulus.value, "dense-carry-" + std::to_string(index));
  }

  // Five unsupported odd widths, plus three full-width even shapes. Fallback
  // does not export a 36-word residue; only its route and Euler bool are checked.
  for (std::uint32_t bits : {1153u, 1154u, 1184u, 1279u, 1280u}) {
    mpz_set_ui(modulus.value, 1u);
    mpz_mul_2exp(modulus.value, modulus.value, bits - 1u);
    mpz_add_ui(modulus.value, modulus.value, 3u);
    append_fixture(corpus, modulus.value, "fallback-odd-" + std::to_string(bits));
  }
  for (std::uint32_t bits : {1139u, 1152u, 1280u}) {
    mpz_set_ui(modulus.value, 1u);
    mpz_mul_2exp(modulus.value, modulus.value, bits);
    mpz_sub_ui(modulus.value, modulus.value, 2u);
    append_fixture(corpus, modulus.value, "fallback-even-" + std::to_string(bits));
  }
  if (corpus.size() != kCorpusSize)
    throw std::runtime_error("carry-late guard corpus size drift");
  return corpus;
}

inline void print_report(const Report& report,
                          std::chrono::steady_clock::time_point begin,
                          bool passed) {
  const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - begin).count();
  std::ostringstream line;
  line << std::fixed << std::setprecision(3)
       << "phase=carrylate_q0_guard total=" << report.total
       << " checked=" << report.checked << " mismatches=" << report.mismatches
       << " fast=" << report.fast << " fallback=" << report.fallback
       << " ms=" << ms << " terminal=" << (passed ? "PASS" : "FAIL")
       << " errors=" << (passed ? 0 : 1)
       << " timing_scope=wall_including_gmp_alloc_copy_kernel_verify_cleanup";
  std::cout << line.str() << std::endl;
}

}  // namespace riecoin_carrylate_q0_guard_detail

inline void run_carrylate_q0_guard() {
  namespace guard = riecoin_carrylate_q0_guard_detail;
  const auto begin = std::chrono::steady_clock::now();
  guard::Report report;
  try {
    // Keep all resources inside this scope so PASS/FAIL wall time includes
    // cleanup. Destructors also release every earlier allocation if a later
    // allocation, launch, copy or comparison throws.
    const std::vector<guard::Fixture> corpus = guard::make_corpus();
    report.total = static_cast<std::uint32_t>(corpus.size());
    std::vector<cgbn_mem_t<1280u>> inputs(corpus.size());
    std::vector<std::uint8_t> verdicts(corpus.size(), 0xa5u);
    std::vector<std::uint8_t> used_fast(corpus.size(), 0xa5u);
    std::vector<std::uint32_t> residues(corpus.size() * guard::kResidueWords,
                                       0xa5a5a5a5u);
    for (std::size_t index = 0u; index < corpus.size(); ++index)
      inputs[index] = corpus[index].candidate;
    guard::DeviceAllocation<cgbn_mem_t<1280u>> d_inputs(
        inputs.size(), "carry-late guard allocate candidates");
    guard::DeviceAllocation<std::uint8_t> d_verdicts(
        verdicts.size(), "carry-late guard allocate verdicts");
    guard::DeviceAllocation<std::uint8_t> d_used_fast(
        used_fast.size(), "carry-late guard allocate routes");
    guard::DeviceAllocation<std::uint32_t> d_residues(
        residues.size(), "carry-late guard allocate residues");
    cuda_check(cudaMemcpy(d_inputs.get(), inputs.data(),
        inputs.size() * sizeof(inputs.front()), cudaMemcpyHostToDevice),
        "carry-late guard copy candidates");
    cuda_check(cudaMemset(d_verdicts.get(), 0xa5, verdicts.size()),
               "carry-late guard poison verdicts");
    cuda_check(cudaMemset(d_used_fast.get(), 0xa5, used_fast.size()),
               "carry-late guard poison routes");
    cuda_check(cudaMemset(d_residues.get(), 0xa5,
        residues.size() * sizeof(residues.front())),
        "carry-late guard poison residues");
    const std::uint32_t blocks = (report.total * 8u + 127u) / 128u;
    riecoin_rlow_carrylate::oracle_kernel<<<blocks, 128u>>>(
        d_inputs.get(), d_verdicts.get(), d_used_fast.get(), d_residues.get(),
        report.total);
    cuda_check(cudaGetLastError(), "carry-late guard oracle launch");
    cuda_check(cudaDeviceSynchronize(), "carry-late guard oracle execution");
    cuda_check(cudaMemcpy(verdicts.data(), d_verdicts.get(), verdicts.size(),
        cudaMemcpyDeviceToHost), "carry-late guard copy verdicts");
    cuda_check(cudaMemcpy(used_fast.data(), d_used_fast.get(), used_fast.size(),
        cudaMemcpyDeviceToHost), "carry-late guard copy routes");
    cuda_check(cudaMemcpy(residues.data(), d_residues.get(),
        residues.size() * sizeof(residues.front()), cudaMemcpyDeviceToHost),
        "carry-late guard copy residues");

    std::string first_mismatch;
    for (std::size_t index = 0u; index < corpus.size(); ++index) {
      const guard::Fixture& expected = corpus[index];
      if (used_fast[index] == 1u) ++report.fast;
      if (used_fast[index] == 0u) ++report.fallback;
      bool mismatch = used_fast[index] != expected.used_fast ||
                      verdicts[index] != expected.verdict;
      if (expected.used_fast != 0u) {
        for (std::uint32_t word = 0u; word < guard::kResidueWords; ++word)
          mismatch = mismatch || residues[index * guard::kResidueWords + word] !=
                                  expected.residue[word];
      }
      if (mismatch) {
        ++report.mismatches;
        if (first_mismatch.empty()) first_mismatch = expected.label;
      }
      ++report.checked;
    }
    if (report.mismatches != 0u)
      throw std::runtime_error("carry-late Q0 guard mismatches=" +
          std::to_string(report.mismatches) + " first_case=" + first_mismatch);

    // Surface cleanup errors on the successful path. During exception unwinding
    // the nonthrowing destructors still attempt all remaining frees.
    d_residues.release_checked("carry-late guard free residues");
    d_used_fast.release_checked("carry-late guard free routes");
    d_verdicts.release_checked("carry-late guard free verdicts");
    d_inputs.release_checked("carry-late guard free candidates");
  } catch (...) {
    guard::print_report(report, begin, false);
    throw;
  }
  guard::print_report(report, begin, true);
}

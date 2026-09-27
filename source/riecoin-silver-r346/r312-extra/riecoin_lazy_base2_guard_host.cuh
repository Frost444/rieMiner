#pragma once

// Included after cuda_check, the ordinary CUDA/GMP/standard headers and the
// lazy-window operator. Cold guard only: its complete cost belongs to E2E.
namespace riecoin_lazy_base2_guard {

constexpr std::uint32_t kWords = 40u;
struct Integer {
  mpz_t value;
  Integer() { mpz_init(value); }
  ~Integer() { mpz_clear(value); }
  Integer(const Integer&) = delete;
  Integer& operator=(const Integer&) = delete;
};
template<class T> struct Device {
  T* data = nullptr;
  explicit Device(std::size_t count) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)),
               "lazy base2 guard allocation");
  }
  ~Device() noexcept { if (data) (void)cudaFree(data); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  void release() {
    if (!data) return;
    const cudaError_t status = cudaFree(data);
    if (status == cudaSuccess) data = nullptr;
    cuda_check(status, "lazy base2 guard release");
  }
};
struct Fixture {
  cgbn_mem_t<1280u> input{};
  std::array<std::uint32_t, kWords> residue{};
  std::uint8_t verdict = 0u;
  std::uint8_t window = 0u;
  std::string label;
};

inline void append(std::vector<Fixture>& fixtures, mpz_srcptr modulus,
                   const std::string& label, bool positive = false) {
  if (mpz_sgn(modulus) < 0 || mpz_sizeinbase(modulus, 2) > 1280u)
    throw std::runtime_error("lazy base2 guard input width");
  Fixture f;
  f.label = label;
  std::size_t words = 0;
  mpz_export(f.input._limbs, &words, -1, sizeof(std::uint32_t), 0, 0, modulus);
  if (words > kWords) throw std::runtime_error("lazy base2 guard packing");
  Integer packed, exponent, base, residue, expected;
  mpz_import(packed.value, kWords, -1, sizeof(std::uint32_t), 0, 0,
             f.input._limbs);
  if (mpz_cmp(packed.value, modulus) != 0)
    throw std::runtime_error("lazy base2 guard packing roundtrip");
  const std::uint32_t bits = static_cast<std::uint32_t>(mpz_sizeinbase(modulus, 2));
  if (bits >= 3u && mpz_odd_p(modulus)) {
    f.window = static_cast<std::uint8_t>(
        riecoin_lazy_base2_window::window_bits(1280u - bits));
    mpz_fdiv_q_2exp(exponent.value, modulus, 1u);
    mpz_set_ui(base.value, 2u);
    mpz_powm(residue.value, base.value, exponent.value, modulus);
    const int symbol = mpz_jacobi(base.value, modulus);
    if (symbol == 1) mpz_set_ui(expected.value, 1u);
    else if (symbol == -1) mpz_sub_ui(expected.value, modulus, 1u);
    else throw std::runtime_error("lazy base2 guard Jacobi");
    f.verdict = mpz_cmp(residue.value, expected.value) == 0 ? 1u : 0u;
    mpz_export(f.residue.data(), &words, -1, sizeof(std::uint32_t), 0, 0,
               residue.value);
    if (words > kWords) throw std::runtime_error("lazy base2 guard residue width");
  }
  if (positive && f.verdict != 1u)
    throw std::runtime_error("lazy base2 positive fixture changed");
  fixtures.push_back(std::move(f));
}

inline std::vector<Fixture> corpus() {
  std::vector<Fixture> result;
  result.reserve(400u);
  Integer n, power;
  for (std::uint32_t bits : {31u,32u,33u,63u,64u,65u,127u,128u,129u,
       511u,512u,513u,799u,800u,831u,832u,943u,944u,1023u,1024u,1055u,
       1087u,1088u,1097u,1103u,1106u,1114u,1120u,1136u,1137u,1138u,1139u,
       1140u,1141u,1142u,1143u,1144u,1145u,1146u,1147u,1148u,1151u,1152u,1153u,
       1183u,1184u,1215u,1216u,1217u,1247u,1248u,1249u,1263u,1264u,1265u,
       1271u,1272u,1273u,1275u,1276u,1277u,1279u,1280u}) {
    mpz_set_ui(power.value, 1u);
    mpz_mul_2exp(power.value, power.value, bits - 1u);
    for (unsigned long delta : {1ul, 3ul}) {
      mpz_add_ui(n.value, power.value, delta);
      append(result, n.value, "lower-" + std::to_string(bits) + "-" +
             std::to_string(delta));
    }
    mpz_mul_2exp(power.value, power.value, 1u);
    for (unsigned long delta : {1ul, 3ul}) {
      mpz_sub_ui(n.value, power.value, delta);
      append(result, n.value, "upper-" + std::to_string(bits) + "-" +
             std::to_string(delta));
    }
  }
  for (unsigned long value : {0ul,1ul,2ul,3ul,4ul,5ul,6ul,7ul,8ul,9ul,
                              11ul,13ul,15ul,17ul,25ul,31ul}) {
    mpz_set_ui(n.value, value);
    append(result, n.value, "small-" + std::to_string(value));
  }
  // All six-bit values, placed across a 32-bit boundary; dense upper fixtures
  // above additionally exercise repeated w=63 windows near the headroom limit.
  for (std::uint32_t value = 0; value != 64u; ++value) {
    mpz_set_ui(n.value, value);
    mpz_mul_2exp(n.value, n.value, 29u);
    mpz_setbit(n.value, 1105u);
    mpz_setbit(n.value, 0u);
    append(result, n.value, "crossword-" + std::to_string(value));
  }
  std::uint32_t random = 0x8d216b57u;
  for (std::uint32_t index = 0; index != 64u; ++index) {
    constexpr std::uint32_t widths[] = {1097u,1106u,1151u,1152u,1153u,
                                       1216u,1248u,1264u};
    const std::uint32_t bits = widths[index % 8u];
    std::array<std::uint32_t, kWords> limbs{};
    for (auto& word : limbs) {
      random ^= random << 13u;
      random ^= random >> 17u;
      random ^= random << 5u;
      word = random;
    }
    mpz_import(n.value, kWords, -1, sizeof(std::uint32_t), 0, 0, limbs.data());
    mpz_fdiv_r_2exp(n.value, n.value, bits);
    mpz_setbit(n.value, bits - 1u);
    mpz_setbit(n.value, 0u);
    append(result, n.value, "dense-random-" + std::to_string(index));
  }
  // Frozen positive inputs shared with the earlier exact oracle; no nextprime
  // or search is introduced into the production path.
  struct Prime { std::uint32_t bits; unsigned long addend; };
  constexpr Prime primes[] = {{1137u,0x12b1ful},{1139u,0x12543ul},
      {1145u,0x12403ul},{1150u,0x1285dul},{1151u,0x126dful},
      {1152u,0x124fdul},{1153u,0x126e5ul},{1280u,0x12c3bul}};
  for (const auto& prime : primes) {
    mpz_set_ui(n.value, 1u);
    mpz_mul_2exp(n.value, n.value, prime.bits - 1u);
    mpz_add_ui(n.value, n.value, prime.addend);
    append(result, n.value, "positive-" + std::to_string(prime.bits), true);
  }
  return result;
}
}  // namespace riecoin_lazy_base2_guard

inline void run_lazy_base2_guard() {
  namespace guard = riecoin_lazy_base2_guard;
  const auto start = std::chrono::steady_clock::now();
  std::uint32_t checked = 0, mismatches = 0, fast = 0, fallback = 0;
  std::array<std::uint32_t, 7u> windows{};
  auto report = [&](bool pass) {
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "phase=lazy_base2_guard checked=" << checked
              << " mismatches=" << mismatches << " fast=" << fast
              << " fallback=" << fallback << " residue_words=40 ms=" << ms
              << " terminal=" << (pass ? "PASS" : "FAIL")
              << " errors=" << (pass ? 0 : 1)
              << " timing_scope=wall_including_gmp_alloc_copy_kernel_verify_cleanup"
              << std::endl;
  };
  try {
    const auto cases = guard::corpus();
    std::vector<cgbn_mem_t<1280u>> input(cases.size());
    std::vector<std::uint8_t> verdict(cases.size()), route(cases.size());
    std::vector<std::uint32_t> residue(cases.size() * guard::kWords);
    for (std::size_t i = 0; i != cases.size(); ++i) input[i] = cases[i].input;
    guard::Device<cgbn_mem_t<1280u>> d_input(input.size());
    guard::Device<std::uint8_t> d_verdict(verdict.size()), d_route(route.size());
    guard::Device<std::uint32_t> d_residue(residue.size());
    cuda_check(cudaMemcpy(d_input.data, input.data(), input.size() * sizeof(input[0]),
        cudaMemcpyHostToDevice), "lazy base2 guard input");
    cuda_check(cudaMemset(d_verdict.data, 0xa5, verdict.size()), "lazy guard poison bool");
    cuda_check(cudaMemset(d_route.data, 0xa5, route.size()), "lazy guard poison route");
    cuda_check(cudaMemset(d_residue.data, 0xa5, residue.size() * sizeof(residue[0])),
               "lazy guard poison residue");
    const auto count = static_cast<std::uint32_t>(cases.size());
    riecoin_lazy_base2_window::oracle_kernel<<<(count * 8u + 127u) / 128u, 128u>>>(
        d_input.data, d_verdict.data, d_route.data, d_residue.data, count);
    cuda_check(cudaGetLastError(), "lazy base2 oracle launch");
    cuda_check(cudaDeviceSynchronize(), "lazy base2 oracle execution");
    cuda_check(cudaMemcpy(verdict.data(), d_verdict.data, verdict.size(),
        cudaMemcpyDeviceToHost), "lazy base2 oracle bool");
    cuda_check(cudaMemcpy(route.data(), d_route.data, route.size(),
        cudaMemcpyDeviceToHost), "lazy base2 oracle route");
    cuda_check(cudaMemcpy(residue.data(), d_residue.data,
        residue.size() * sizeof(residue[0]), cudaMemcpyDeviceToHost),
        "lazy base2 oracle residue");
    std::string first_mismatch;
    for (std::size_t i = 0; i != cases.size(); ++i) {
      bool mismatch = route[i] != cases[i].window || verdict[i] != cases[i].verdict;
      if (route[i] < windows.size()) ++windows[route[i]];
      if (route[i] == 0u) ++fallback;
      else ++fast;
      if (cases[i].window != 0u)
        for (std::uint32_t word = 0; word != guard::kWords; ++word)
          mismatch = mismatch || residue[i * guard::kWords + word] !=
                                  cases[i].residue[word];
      if (mismatch) {
        ++mismatches;
        if (first_mismatch.empty()) first_mismatch = cases[i].label;
      }
      ++checked;
    }
    if (mismatches != 0u)
      throw std::runtime_error("lazy base2 oracle mismatch first=" + first_mismatch);
    for (std::uint32_t width = 0; width != windows.size(); ++width)
      if (windows[width] == 0u)
        throw std::runtime_error("lazy base2 missing window coverage");
    d_residue.release(); d_route.release(); d_verdict.release(); d_input.release();
  } catch (...) { report(false); throw; }
  report(true);
}

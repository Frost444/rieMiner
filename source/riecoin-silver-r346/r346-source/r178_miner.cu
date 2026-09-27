#define RIECOIN_RLOW_PRIME_ROOT_CACHE_MAX_BYTES 536870912u
#define RIECOIN_A54_OPTIONAL_PRIME_LIMIT 100000000u
// BEGIN INLINED astra-riecoin-wait-policy-a27.cu
// One binary, runtime-selectable host wait policy; unchanged A25 device code.
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <iostream>

static cudaError_t astra_a27_set_device(int device) {
    const cudaError_t selected=::cudaSetDevice(device);
    if(selected!=cudaSuccess)return selected;
    const char* mode=std::getenv("ASTRA_CUDA_WAIT_POLICY");
    if(!mode || std::strcmp(mode,"auto")==0)return cudaSuccess;
    unsigned int schedule=0;
    if(std::strcmp(mode,"block")==0)schedule=cudaDeviceScheduleBlockingSync;
    else if(std::strcmp(mode,"yield")==0)schedule=cudaDeviceScheduleYield;
    else return cudaErrorInvalidValue;
    unsigned int flags=0;
    const cudaError_t got=::cudaGetDeviceFlags(&flags);
    if(got!=cudaSuccess)return got;
    // CUDA 12.8 permits changing the initialized current device's flags.
    // Preserve all non-scheduler flags; allocations, queues and kernels are unchanged.
    const cudaError_t changed=::cudaSetDeviceFlags((flags&~cudaDeviceScheduleMask)|schedule);
    if(changed==cudaSuccess)std::cout<<"phase=host_wait_policy mode="<<mode<<" device="<<device<<std::endl;
    return changed;
}
#define cudaSetDevice astra_a27_set_device
// BEGIN INLINED astra-riecoin-blocking-ready-a25.cu
// Isolate host wait policy from arithmetic/register changes (A23 not included).
#define RIECOIN_RLOW_BLOCKING_READY
// BEGIN INLINED astra-riecoin-wide1280-tpi4-a19.cu
// A16 scheduling retained; exact full-width group specialization, opt-in only.
#define RIECOIN_RLOW_WIDE_TPI4
// BEGIN INLINED astra-riecoin-async-frontier-a16.cu
// Event-owned ring: completed counters, guards, timings and survivor work
// leave the submission thread. Full-width arithmetic and cold gates unchanged.
#define RIECOIN_RLOW_ASYNC_EXACT_VERIFY
#define RIECOIN_RLOW_RESULT_SIDECAR
#define RIECOIN_RLOW_CUDA_TAIL_RAIL
#define RIECOIN_RLOW_ASYNC_FRONTIER
// BEGIN INLINED riecoin-rlow-e2e-r202-canonicalp114-min109-compressedconsumers-width1148-asyncexact.cu
// R202 completes the compressed-root consumer closure exposed sequentially by
// the exact P109 gate. R201 repaired the warmup private-macro reader; R202 also
// teaches the unchanged R194 dense batch and sparse-event readers to consume
// the same self-describing mixed dense/sparse table. No arithmetic, queue,
// capacity, PRP, verifier, or result-contract change is introduced here.
// BEGIN INLINED riecoin-rlow-e2e-r201-canonicalp114-min109-devicecompressed-width1148-asyncexact.cu
// R201 closes the exact R200 consumer boundary exposed by the P109 gate.
// R200 correctly built and verified the compressed device-resident root table,
// while the inherited R194 private-macro sieve still indexed roots as a dense
// seven-word table. The only additional source dependency in this candidate is
// the compressed-root-aware private-macro consumer; the productive R194 engine
// and the canonical P114 representation remain otherwise unchanged.
// BEGIN INLINED riecoin-rlow-e2e-r200-canonicalp114-min109-devicecompressed-width1148-asyncexact.cu
// R200 is an isolated closure repair of R199.  It keeps the certified R194
// producer/queue implementation byte-for-byte and opts only its historical
// five-field telemetry call into the newer device-root integration unit.
// The productive change remains one exact P114 basis for P109..P114, fused
// into fully verified device-resident compressed roots before sieve use.
#define RIECOIN_RLOW_QUEUE_FUNNEL_V1_COMPAT
#define RIECOIN_RLOW_PRIME_ROOT_CACHE_CANONICAL_FIRST 114u
#define RIECOIN_RLOW_PRIME_ROOT_CACHE_CANONICAL_MIN_FIRST 107u
#define RIECOIN_RLOW_CANONICAL_DEVICE_TRANSFORM
#define RIECOIN_RLOW_COMPRESSED_ROOTS
// BEGIN INLINED riecoin-rlow-e2e-r194-residentroots-width1148-asyncexact.cu
// R194/C5: remove prime/root construction from the productive job boundary.
// The immutable SHA-verified prime/inverse cache is loaded once per process;
// exact roots are constructed directly into their final device-resident table.
// R172 arithmetic, complete <=1148-bit routing and async exact verification
// remain unchanged.  This wrapper is isolated and is not a Silver mutation.
#include <gmp.h>
#include "riecoin-cuda-prp1152.cuh"

#define RIECOIN_RLOW_FAST_PRECOMPUTE
#define RIECOIN_RLOW_PRIME_ROOT_CACHE
#define RIECOIN_RLOW_DEVICE_ROOTS

// Preserve the CPU implementation as an exact oracle while selecting the
// device-resident producer at the queued engine boundary.
#define roots_for cpu_roots_for
#include "riecoin_prime_root_cache.hpp"
#undef roots_for
#undef small
#include "riecoin_gpu_root_builder.cuh"

#define RIECOIN_RLOW_NARROW_MAX_INPUT_BITS 1148u
#define RIECOIN_RLOW_NARROW_LAZY_Q0
#define RIECOIN_RLOW_ASYNC_EXACT_VERIFY
// BEGIN INLINED riecoin-rlow-e2e-deeproot64.cu
// Same 100M/64-macro exact GPU pipeline. Change only stateless CPU setup:
// identical ordered primes/roots, segmented odd sieve and affine root steps.
#define RIECOIN_RLOW_FAST_PRECOMPUTE
// BEGIN INLINED riecoin-rlow-e2e-deepqueue64.cu
// Keep the exact 100M sieve and every candidate. Aggregate 64 disjoint macros
// before the unchanged Q0 PRP to compensate for the lower survivor density.
// Segment capacity/overflow guards are unchanged; all extra memory/setup/cold
// verification is charged to E2E. This does not alter the residue geometry.
#define RIECOIN_RLOW_QUEUE_MACROS 64u
#define RIECOIN_RLOW_DEFAULT_PRIME_LIMIT 100000000u
#define RIECOIN_RLOW_BATCH_PRODUCER
#define RIECOIN_RLOW_Q1_TAIL_QUEUE
#define RIECOIN_RLOW_Q1_INPUT_BATCH
#define RIECOIN_RLOW_LAZY_BASE2_Q0
#define RIECOIN_RLOW_SPARSE_EVENT_SIEVE
// BEGIN INLINED riecoin-rlow-e2e-queued.cu
#include <cuda_runtime.h>
#ifdef RIECOIN_RLOW_RESULT_SIDECAR
#if !defined(RIECOIN_RLOW_ASYNC_EXACT_VERIFY) || !defined(RIECOIN_RLOW_Q1_TAIL_QUEUE)
#error "Result sidecar requires the qualified async verifier and Q1-tail path"
#endif
#include "riecoin_cuda_result_sidecar.hpp"
#endif
#include <gmp.h>
#ifdef RIECOIN_RLOW_SCIENCE_Q9
#include "riecoin_strict_science_q9.hpp"
#endif
#include "riecoin_production_window.hpp"
#ifdef RIECOIN_RLOW_FAST_PRECOMPUTE
#include "riecoin_prime_root_precompute.hpp"
#endif
#ifdef RIECOIN_RLOW_PRIME_ROOT_CACHE
#ifndef RIECOIN_RLOW_FAST_PRECOMPUTE
#error "Prime/inverse cache requires the exact fast-precompute path"
#endif
#include "riecoin_prime_root_cache.hpp"
#endif
#if defined(RIECOIN_RLOW_DEVICE_ROOTS) && !defined(RIECOIN_RLOW_PRIME_ROOT_CACHE)
#error "Device-resident roots require the exact prime/inverse cache basis"
#endif
#if defined(RIECOIN_RLOW_OVERLAP_PIPELINE) && !defined(RIECOIN_RLOW_BATCH_PRODUCER)
#error "The overlap pipeline requires disjoint batch-producer macro ownership"
#endif
#include "riecoin-cuda-prp1152.cuh"
#include "riecoin_rlow_deterministic_bridge.cuh"
#ifdef RIECOIN_RLOW_CARRYLATE_Q0
#include "riecoin_rlow_carrylate_q0.cuh"
#endif
#ifdef RIECOIN_RLOW_LAZY_BASE2_Q0
#ifdef RIECOIN_RLOW_CARRYLATE_Q0
#error "Select exactly one Q0 operator"
#endif
#ifdef RIECOIN_RLOW_WIDE_TPI4
#include "riecoin_lazy_base2_tpi4.cuh"
#else
#include "riecoin_lazy_base2_window.cuh"
#endif
#endif
#ifdef RIECOIN_RLOW_SPARSE_FAMILY_Q0
#include "riecoin_sparse_scalar_q0.cuh"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// Experimental resident host envelopes may request an epoch replacement only
// at the already exact bundle synchronization boundary below.  The production
// one-shot build does not define this hook, so its behaviour is unchanged.
#ifndef RIECOIN_RLOW_RESIDENT_BUNDLE_STOP_REQUESTED
#define RIECOIN_RLOW_RESIDENT_BUNDLE_STOP_REQUESTED() false
#endif

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace {

constexpr std::array<std::array<uint32_t, 7>, 2> kMainnetPatterns{{
    {{0, 2, 4, 2, 4, 6, 2}},
    {{0, 2, 6, 4, 2, 4, 2}},
}};

struct Options {
  uint32_t candidates = 1u << 20;
#ifdef RIECOIN_RLOW_DEFAULT_PRIME_LIMIT
  uint32_t prime_limit = 200000000u; // A54c default only; explicit CLI still wins.
#else
  uint32_t prime_limit = 10000000;
#endif
  uint32_t pattern = 0;
  uint32_t primorial_number = 8;
  uint32_t initial_target_bits = 0;
  uint32_t target_bits = 1152;
  uint32_t difficulty_offset = 0;
  uint32_t oracle_threads = 1;
  uint32_t q0_duration_seconds = 0;
  uint32_t production_session_seconds = 0;
  uint32_t rejected_audit_per_stage = 8;
  bool network_batch = false;
  uint64_t factor_max = 16ull * (1ull << 25u);
  uint64_t factor_origin = 0;
  uint64_t primorial_offset = 114023297140211ull;
  int device = 0;
  bool codec_self_test = false;
  std::string expected_uuid;
  std::string expected_pci = "00000000:04:00.0";
  std::string target_hex;
  std::string results_jsonl;
  std::string q0_summary_json;
  std::string work_id;
  std::string template_id;
  std::string runtime_sha256;
  std::string gpu_identity_sha256;
#ifdef RIECOIN_RLOW_PRIME_ROOT_CACHE
  std::string prime_root_cache_dir;
  bool prime_root_cache_dir_explicit = false;
#endif
};

std::string json_escape(const std::string& input) {
  std::ostringstream escaped;
  for (const unsigned char c : input) {
    switch (c) {
      case '"': escaped << "\\\""; break;
      case '\\': escaped << "\\\\"; break;
      case '\b': escaped << "\\b"; break;
      case '\f': escaped << "\\f"; break;
      case '\n': escaped << "\\n"; break;
      case '\r': escaped << "\\r"; break;
      case '\t': escaped << "\\t"; break;
      default:
        if (c < 0x20u) {
          escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                  << static_cast<unsigned int>(c) << std::dec;
        } else {
          escaped << static_cast<char>(c);
        }
    }
  }
  return escaped.str();
}

bool append_jsonl_durable(const std::string& path, const std::string& line) {
  if (path.empty()) return false;
  FILE* file = std::fopen(path.c_str(), "ab");
  if (file == nullptr) return false;
  const std::string encoded = line + '\n';
  const bool wrote = std::fwrite(encoded.data(), 1, encoded.size(), file) == encoded.size();
  const bool flushed = std::fflush(file) == 0;
#ifdef _WIN32
  const bool committed = flushed && _commit(_fileno(file)) == 0;
#else
  const bool committed = flushed && fsync(fileno(file)) == 0;
#endif
  const bool closed = std::fclose(file) == 0;
  return wrote && flushed && committed && closed;
}

bool write_json_atomic(const std::string& path, const std::string& json) {
  if (path.empty()) return false;
  const std::string temporary = path + ".tmp";
  FILE* file = std::fopen(temporary.c_str(), "wb");
  if (file == nullptr) return false;
  const bool wrote = std::fwrite(json.data(), 1, json.size(), file) == json.size();
  const bool newline = std::fwrite("\n", 1, 1, file) == 1;
  const bool flushed = std::fflush(file) == 0;
#ifdef _WIN32
  const bool committed = flushed && _commit(_fileno(file)) == 0;
#else
  const bool committed = flushed && fsync(fileno(file)) == 0;
#endif
  const bool closed = std::fclose(file) == 0;
  if (!(wrote && newline && flushed && committed && closed)) {
    std::remove(temporary.c_str());
    return false;
  }
#ifdef _WIN32
  return MoveFileExA(temporary.c_str(), path.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
  return std::rename(temporary.c_str(), path.c_str()) == 0;
#endif
}

void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

uint32_t parse_u32(const char* text, const char* name) {
  char* end = nullptr;
  const auto value = std::strtoull(text, &end, 0);
  if (end == text || *end != '\0' || value > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error(std::string("invalid ") + name);
  return static_cast<uint32_t>(value);
}

uint64_t parse_u64(const char* text, const char* name) {
  char* end = nullptr;
  const auto value = std::strtoull(text, &end, 0);
  if (end == text || *end != '\0')
    throw std::runtime_error(std::string("invalid ") + name);
  return static_cast<uint64_t>(value);
}

std::string mpz_hex(const mpz_t value) {
  const size_t digits = mpz_sizeinbase(value, 16);
  std::vector<char> buffer(digits + 2u, '\0');
  if (mpz_get_str(buffer.data(), 16, value) == nullptr)
    throw std::runtime_error("mpz hex conversion failed");
  return std::string(buffer.data());
}

void mpz_add_u64(mpz_t result, const mpz_t source, uint64_t value) {
  mpz_t wide;
  mpz_init(wide);
  mpz_import(wide, 1, -1, sizeof(value), 0, 0, &value);
  mpz_add(result, source, wide);
  mpz_clear(wide);
}

void mpz_set_u64(mpz_t result, uint64_t value) {
  mpz_import(result, 1, -1, sizeof(value), 0, 0, &value);
}

void pack_candidate_1280(cgbn_mem_t<1280u>& packed, const mpz_t value) {
  std::fill(std::begin(packed._limbs), std::end(packed._limbs), 0u);
  size_t written = 0u;
  mpz_export(packed._limbs, &written, -1, sizeof(uint32_t), 0, 0, value);
  const size_t bit_length = mpz_sizeinbase(value, 2);
  if (written > 40u || mpz_sgn(value) <= 0 || bit_length < 3u ||
      bit_length > 1280u)
    throw std::runtime_error("Q0 candidate escaped dynamic 1280-bit bucket");
}

void pack_unsigned_1280(cgbn_mem_t<1280u>& packed, const mpz_t value) {
  std::fill(std::begin(packed._limbs), std::end(packed._limbs), 0u);
  size_t written = 0u;
  mpz_export(packed._limbs, &written, -1, sizeof(uint32_t), 0, 0, value);
  if (written > 40u || mpz_sgn(value) < 0)
    throw std::runtime_error("unsigned value escaped 1280-bit bucket");
}

void initialize_target(mpz_t target, const Options& options) {
  if (!options.target_hex.empty()) {
    if (mpz_set_str(target, options.target_hex.c_str(), 16) != 0)
      throw std::runtime_error("invalid target hex");
    return;
  }
  mpz_set_ui(target, 1u);
  mpz_mul_2exp(target, target, options.target_bits - 1u);
}

Options parse_options(int argc, char** argv) {
  Options result;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto need = [&](const char* name) -> const char* {
      if (++i >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return argv[i];
    };
    if (arg == "--candidates") result.candidates = parse_u32(need("--candidates"), "candidate count");
    else if (arg == "--prime-limit") result.prime_limit = parse_u32(need("--prime-limit"), "prime limit");
#ifdef RIECOIN_RLOW_PRIME_ROOT_CACHE
    else if (arg == "--prime-root-cache-dir") {
      result.prime_root_cache_dir = need("--prime-root-cache-dir");
      result.prime_root_cache_dir_explicit = true;
    }
#endif
    else if (arg == "--pattern") result.pattern = parse_u32(need("--pattern"), "pattern");
    else if (arg == "--primorial-number") result.primorial_number = parse_u32(need("--primorial-number"), "primorial number");
    else if (arg == "--initial-target-bits") result.initial_target_bits = parse_u32(need("--initial-target-bits"), "initial target bits");
    else if (arg == "--target-bits") result.target_bits = parse_u32(need("--target-bits"), "target bits");
    else if (arg == "--factor-max") result.factor_max = parse_u64(need("--factor-max"), "factor max");
    else if (arg == "--factor-origin") result.factor_origin = parse_u64(need("--factor-origin"), "factor origin");
    else if (arg == "--primorial-offset") result.primorial_offset = parse_u64(need("--primorial-offset"), "primorial offset");
    else if (arg == "--difficulty-offset") result.difficulty_offset = parse_u32(need("--difficulty-offset"), "difficulty offset");
    else if (arg == "--oracle-threads") result.oracle_threads = parse_u32(need("--oracle-threads"), "oracle threads");
    else if (arg == "--q0-duration-seconds") result.q0_duration_seconds = parse_u32(need("--q0-duration-seconds"), "Q0 duration");
    else if (arg == "--production-session-seconds") result.production_session_seconds = parse_u32(need("--production-session-seconds"), "production session duration");
    else if (arg == "--rejected-audit-per-stage") result.rejected_audit_per_stage = parse_u32(need("--rejected-audit-per-stage"), "rejected audit bound");
    else if (arg == "--network-batch") result.network_batch = true;
    else if (arg == "--codec-self-test") result.codec_self_test = true;
    else if (arg == "--device") result.device = static_cast<int>(parse_u32(need("--device"), "device"));
    else if (arg == "--expect-uuid") result.expected_uuid = need("--expect-uuid");
    else if (arg == "--expect-pci") result.expected_pci = need("--expect-pci");
    else if (arg == "--target-hex") result.target_hex = need("--target-hex");
    else if (arg == "--results-jsonl") result.results_jsonl = need("--results-jsonl");
    else if (arg == "--q0-summary-json") result.q0_summary_json = need("--q0-summary-json");
    else if (arg == "--work-id") result.work_id = need("--work-id");
    else if (arg == "--template-id") result.template_id = need("--template-id");
    else if (arg == "--runtime-sha256") result.runtime_sha256 = need("--runtime-sha256");
    else if (arg == "--gpu-identity-sha256") result.gpu_identity_sha256 = need("--gpu-identity-sha256");
    else if (arg == "--help") {
      std::cout << "riecoin-cuda-sieve-prototype [--candidates N] [--prime-limit P] "
                   "[--pattern 0|1] [--primorial-number N] [--primorial-offset O] "
                   "[--initial-target-bits B --factor-max F] "
                   "[--factor-origin F0] "
                   "[--difficulty-offset D] [--device D] [--expect-uuid GPU-...] "
                    "[--oracle-threads N] "
                    "[--target-bits B | --target-hex HEX] [--codec-self-test] "
                    "[--network-batch --factor-origin F0 --q0-summary-json PATH] "
                    "[--production-session-seconds S>=30 "
                    "--rejected-audit-per-stage N] "
                    "[--results-jsonl PATH --work-id ID --template-id ID "
                    "--q0-summary-json PATH "
                    "--runtime-sha256 SHA256 --gpu-identity-sha256 SHA256]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + arg);
    }
  }
  if (result.candidates == 0 || result.prime_limit < 23 || result.pattern > 1 ||
      result.primorial_number == 0 || result.primorial_number > 65535 ||
      result.difficulty_offset > 2047 || result.factor_max == 0 ||
      result.target_bits == 0 || result.oracle_threads == 0 ||
      result.oracle_threads > 128 || result.rejected_audit_per_stage > 1024u)
    throw std::runtime_error("invalid bounded fixture geometry");
  if (result.factor_origin > std::numeric_limits<uint64_t>::max() - result.candidates ||
      result.factor_origin + result.candidates > result.factor_max)
    throw std::runtime_error("factor origin window exceeds factor max");
  if (!result.codec_self_test &&
      (result.results_jsonl.empty() || result.work_id.empty() ||
       result.template_id.empty() || result.runtime_sha256.empty()))
    throw std::runtime_error("durable result identity arguments are mandatory");
  if (!result.codec_self_test && result.gpu_identity_sha256.empty())
    throw std::runtime_error("durable GPU identity hash is mandatory");
  if (result.q0_duration_seconds != 0u && result.q0_duration_seconds < 30u)
    throw std::runtime_error("Q0 measurement window must be at least 30 seconds");
  if (result.production_session_seconds != 0u &&
      result.production_session_seconds < 30u)
    throw std::runtime_error("production session must be at least 30 seconds");
  if (result.q0_duration_seconds != 0u && result.q0_summary_json.empty())
    throw std::runtime_error("Q0 terminal summary path is mandatory");
  if (result.network_batch && result.q0_summary_json.empty())
    throw std::runtime_error("network-batch Q0 summary path is mandatory");
  if (result.network_batch && result.q0_duration_seconds != 0u)
    throw std::runtime_error("single-batch and duration modes are mutually exclusive");
  if ((result.production_session_seconds != 0u) &&
      (result.network_batch || result.q0_duration_seconds != 0u))
    throw std::runtime_error("production session is a distinct execution mode");
  if (result.production_session_seconds != 0u && result.q0_summary_json.empty())
    throw std::runtime_error("production terminal summary path is mandatory");
  if (result.production_session_seconds != 0u &&
      ((result.factor_max - result.factor_origin) % result.candidates != 0u))
    throw std::runtime_error("production factor reservation must contain whole batches");
  if (result.production_session_seconds != 0u && result.candidates > (1u << 19u))
    throw std::runtime_error("production tuple workspace exceeds bounded capacity");
  if (result.network_batch && result.factor_origin + result.candidates != result.factor_max)
    throw std::runtime_error("network-batch must reserve exactly one factor window");
  if (!result.codec_self_test && (result.expected_uuid.empty() || result.expected_pci.empty()))
    throw std::runtime_error("RLOW UUID and PCI guards are mandatory");
  return result;
}

std::vector<uint32_t> primes_to(uint32_t limit) {
#ifdef RIECOIN_RLOW_FAST_PRECOMPUTE
  return riecoin_prime_root_precompute::primes_to(limit);
#else
  std::vector<bool> composite(static_cast<size_t>(limit) + 1, false);
  std::vector<uint32_t> primes;
  for (uint32_t p = 2; p <= limit; ++p) {
    if (composite[p]) continue;
    primes.push_back(p);
    if (static_cast<uint64_t>(p) * p <= limit)
      for (uint32_t n = p * p; n <= limit; n += p) composite[n] = true;
  }
  return primes;
#endif
}

uint32_t derive_primorial_number(const std::vector<uint32_t>& prime_table,
                                 uint32_t initial_target_bits,
                                 uint64_t factor_max) {
  mpz_t limit, factor, primorial, next;
  mpz_inits(limit, factor, primorial, next, nullptr);
  mpz_set_ui(limit, 1u);
  mpz_mul_2exp(limit, limit, initial_target_bits);
  mpz_sub_ui(limit, limit, 1u);
  mpz_import(factor, 1, -1, sizeof(factor_max), 0, 0, &factor_max);
  mpz_fdiv_q(limit, limit, factor);
  mpz_set_ui(primorial, 1u);
  uint32_t number = 0;
  for (size_t i = 0; i < prime_table.size(); ++i) {
    mpz_mul_ui(next, primorial, prime_table[i]);
    if (mpz_cmp(next, limit) >= 0) {
      number = static_cast<uint32_t>(i);
      break;
    }
    mpz_set(primorial, next);
    number = static_cast<uint32_t>(i + 1u);
  }
  mpz_clears(limit, factor, primorial, next, nullptr);
  if (number == 0 || number >= prime_table.size())
    throw std::runtime_error("unable to derive bounded primorial number");
  return number;
}

uint32_t inverse_mod(uint32_t value, uint32_t modulus) {
  int64_t t = 0, new_t = 1;
  int64_t r = modulus, new_r = value;
  while (new_r != 0) {
    const int64_t q = r / new_r;
    const int64_t next_t = t - q * new_t;
    const int64_t next_r = r - q * new_r;
    t = new_t;
    new_t = next_t;
    r = new_r;
    new_r = next_r;
  }
  if (r != 1) throw std::runtime_error("non-invertible primorial residue");
  if (t < 0) t += modulus;
  return static_cast<uint32_t>(t);
}

std::array<uint32_t, 7> cumulative_pattern(uint32_t index) {
  std::array<uint32_t, 7> result{};
  uint32_t sum = 0;
  for (size_t i = 0; i < result.size(); ++i) {
    sum += kMainnetPatterns[index][i];
    result[i] = sum;
  }
  return result;
}

#include "riecoin_rlow_private_macro.cuh"
#include "riecoin_rlow_queue_host.cuh"
#ifdef RIECOIN_RLOW_BATCH_PRODUCER
#include "r178_batch_producer.cuh"
#endif
#ifdef RIECOIN_RLOW_SUPER_SIEVE
#include "riecoin_rlow_super_sieve.cuh"
#endif
#ifdef RIECOIN_RLOW_Q1_TAIL_QUEUE
#include "riecoin_rlow_tail_host.cuh"
#ifdef RIECOIN_RLOW_CUDA_TAIL_RAIL
#include "riecoin_cuda_tail_rail.hpp"
#endif
#endif
#ifdef RIECOIN_RLOW_CARRYLATE_Q0
#include "riecoin_carrylate_q0_guard_host.cuh"
#endif
#ifdef RIECOIN_RLOW_LAZY_BASE2_Q0
#include "riecoin_lazy_base2_guard_host.cuh"
#endif
#ifdef RIECOIN_RLOW_NARROW_LAZY_Q0
#include "riecoin_narrow_lazy_q0.cuh"
#endif

__global__ void mark_composites(uint32_t* words,
                                uint32_t candidate_count,
                                const uint32_t* primes,
                                const uint32_t* first_factors,
                                uint32_t pair_count,
                                uint64_t factor_origin) {
  const uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= pair_count) return;
  const uint32_t p = primes[pair / 7u];
#ifdef RIECOIN_A54_OPTIONAL_PRIME_LIMIT
  if (p > RIECOIN_A54_OPTIONAL_PRIME_LIMIT && pair % 7u >= 2u) return;
#endif
  const uint32_t origin_mod = static_cast<uint32_t>(factor_origin % p);
#ifdef RIECOIN_RLOW_COMPRESSED_ROOTS
  const uint32_t root = riecoin_compressed_root_storage::root_at(
      first_factors, p, pair / 7u, pair % 7u);
#else
  const uint32_t root = first_factors[pair];
#endif
  uint32_t f = root >= origin_mod
      ? root - origin_mod
      : root + p - origin_mod;
  while (f < candidate_count) {
    atomicAnd(words + (f >> 5u), ~(1u << (f & 31u)));
    f += p;
  }
}

__global__ void compare_complete_bitmaps(const uint32_t* candidate,
                                         const uint32_t* reference,
                                         uint32_t word_count,
                                         uint32_t* mismatches) {
  const uint32_t word = blockIdx.x * blockDim.x + threadIdx.x;
  if (word < word_count && candidate[word] != reference[word])
    atomicAdd(mismatches, 1u);
}

// One warp owns one survivor bitmap word. It compacts active factors and
// constructs exact 1280-bit containers for the 1152-bit Riecoin candidates in
// one device pass. No candidate or bitmap crosses the PCIe boundary.
__global__ void compact_construct_q0_candidates(
    const uint32_t* words,
    uint32_t word_count,
    uint32_t candidate_count,
    const cgbn_mem_t<1280u>* base,
    const cgbn_mem_t<1280u>* primorial,
    cgbn_mem_t<1280u>* candidates,
    uint64_t* factors,
    uint64_t factor_origin,
    uint32_t capacity,
    uint32_t* batch_count,
    unsigned long long* total_count,
    uint32_t* overflow_count) {
  const uint32_t global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t lane = global_thread & 31u;
  const uint32_t word_index = global_thread >> 5u;
  const uint32_t active_mask = __activemask();
  uint32_t word = word_index < word_count ? words[word_index] : 0u;
  word = __shfl_sync(active_mask, word, 0u);
  if (word_index < word_count && word_index * 32u + 32u > candidate_count) {
    const uint32_t valid = candidate_count - word_index * 32u;
    word &= valid == 32u ? 0xffffffffu : ((1u << valid) - 1u);
  }
  const uint32_t pop = __popc(word);
  uint32_t base_index = 0u;
  if (lane == 0u && pop != 0u) {
    base_index = atomicAdd(batch_count, pop);
    atomicAdd(total_count, static_cast<unsigned long long>(pop));
  }
  base_index = __shfl_sync(active_mask, base_index, 0u);
  if (((word >> lane) & 1u) == 0u) return;
  const uint32_t factor = word_index * 32u + lane;
  if (factor >= candidate_count) return;
  const uint32_t rank = __popc(word & ((lane == 0u) ? 0u : ((1u << lane) - 1u)));
  const uint32_t output = base_index + rank;
  if (output >= capacity) {
    atomicAdd(overflow_count, 1u);
    return;
  }
  factors[output] = factor_origin + factor;
  uint64_t carry = 0u;
#pragma unroll
  for (uint32_t limb = 0u; limb < 40u; ++limb) {
    const uint64_t value =
        static_cast<uint64_t>(primorial->_limbs[limb]) * factor +
        static_cast<uint64_t>(base->_limbs[limb]) + carry;
    candidates[output]._limbs[limb] = static_cast<uint32_t>(value);
    carry = value >> 32u;
  }
  if (carry != 0u) atomicAdd(overflow_count, 1u);
}

__global__ void compact_q0_passes(const uint8_t* verdicts,
                                  const uint64_t* factors,
                                  const uint32_t* batch_count,
                                  uint32_t capacity,
                                  uint64_t* pass_factors,
                                  uint32_t* pass_count) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t count = *batch_count;
  if (index >= count || index >= capacity || verdicts[index] == 0u) return;
  const uint32_t output = atomicAdd(pass_count, 1u);
  pass_factors[output] = factors[index];
}

__global__ void sample_q0_rejects(
    const uint8_t* verdicts,
    const uint64_t* factors,
    const uint32_t* batch_count,
    uint32_t capacity,
    uint32_t sample_limit,
    uint64_t* sample_factors,
    unsigned long long* rejected_counts) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t count = *batch_count;
  if (index >= count || index >= capacity || verdicts[index] != 0u) return;
  const unsigned long long slot = atomicAdd(rejected_counts, 1ull);
  if (slot < sample_limit)
    sample_factors[slot] = factors[index];
}

__global__ void construct_tuple_members(
    const uint64_t* pass_factors,
    uint32_t pass_count,
    const cgbn_mem_t<1280u>* base,
    const cgbn_mem_t<1280u>* primorial,
    const uint32_t* member_offsets,
    uint64_t factor_origin,
    cgbn_mem_t<1280u>* members,
    uint32_t* overflow_count) {
  const uint32_t output = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t total = pass_count * 6u;
  if (output >= total) return;
  const uint32_t pass_index = output / 6u;
  const uint32_t member = output % 6u;
  const uint64_t absolute_factor = pass_factors[pass_index];
  if (absolute_factor < factor_origin ||
      absolute_factor - factor_origin > UINT32_MAX) {
    atomicAdd(overflow_count, 1u);
    return;
  }
  const uint32_t local_factor =
      static_cast<uint32_t>(absolute_factor - factor_origin);
  uint64_t carry = 0u;
#pragma unroll
  for (uint32_t limb = 0u; limb < 40u; ++limb) {
    const uint64_t value =
        static_cast<uint64_t>(primorial->_limbs[limb]) * local_factor +
        static_cast<uint64_t>(base->_limbs[limb]) + carry;
    members[output]._limbs[limb] = static_cast<uint32_t>(value);
    carry = value >> 32u;
  }
  uint64_t add = member_offsets[member];
#pragma unroll
  for (uint32_t limb = 0u; limb < 40u && add != 0u; ++limb) {
    const uint64_t value =
        static_cast<uint64_t>(members[output]._limbs[limb]) + add;
    members[output]._limbs[limb] = static_cast<uint32_t>(value);
    add = value >> 32u;
  }
  if (carry != 0u || add != 0u) atomicAdd(overflow_count, 1u);
}

__global__ void reduce_tuple_candidates(
    const uint8_t* member_verdicts,
    const uint64_t* pass_factors,
    uint32_t pass_count,
    uint32_t sample_limit,
    unsigned long long* member_pass_counts,
    unsigned long long* active_after_member,
    unsigned long long* rejected_counts,
    uint64_t* sample_factors,
    uint64_t* complete_factors,
    uint32_t complete_capacity,
    unsigned long long* complete_count,
    uint32_t* overflow_count) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pass_count) return;
  uint8_t verdicts[6];
#pragma unroll
  for (uint32_t member = 0u; member < 6u; ++member) {
    verdicts[member] = member_verdicts[index * 6u + member];
    if (verdicts[member] != 0u)
      atomicAdd(member_pass_counts + member, 1ull);
  }

  bool rlow_prefix_alive = true;
#pragma unroll
  for (uint32_t member = 0u; member < 6u; ++member) {
    rlow_prefix_alive = rlow_prefix_alive && verdicts[member] != 0u;
    if (rlow_prefix_alive) atomicAdd(rlow_strict_prefix_device + member, 1ull);
  }
  uint32_t probable_prime_count = 1u;  // Q0 already passed.
#pragma unroll
  for (uint32_t member = 0u; member < 6u; ++member) {
    if (verdicts[member] != 0u) ++probable_prime_count;
    const uint32_t remaining = 5u - member;
    const bool viable = member != 0u || verdicts[member] != 0u;
    if (!viable || probable_prime_count + remaining < 5u) {
      const uint32_t stage = member + 1u;
      const unsigned long long slot = atomicAdd(rejected_counts + stage, 1ull);
      if (slot < sample_limit)
        sample_factors[stage * sample_limit + slot] = pass_factors[index];
      return;
    }
    atomicAdd(active_after_member + member, 1ull);
  }
  if (probable_prime_count < 5u) {
    atomicAdd(overflow_count, 1u);
    return;
  }
  const unsigned long long output = atomicAdd(complete_count, 1ull);
  if (output < complete_capacity)
    complete_factors[output] = pass_factors[index];
  else
    atomicAdd(overflow_count, 1u);
}

__global__ void advance_q0_base(cgbn_mem_t<1280u>* base,
                                const cgbn_mem_t<1280u>* primorial,
                                uint32_t factor_stride,
                                uint32_t* overflow_count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  uint64_t carry = 0u;
#pragma unroll
  for (uint32_t limb = 0u; limb < 40u; ++limb) {
    const uint64_t value =
        static_cast<uint64_t>(primorial->_limbs[limb]) * factor_stride +
        static_cast<uint64_t>(base->_limbs[limb]) + carry;
    base->_limbs[limb] = static_cast<uint32_t>(value);
    carry = value >> 32u;
  }
  if (carry != 0u) atomicAdd(overflow_count, 1u);
}

void cpu_mark(std::vector<uint32_t>& words,
              uint32_t count,
              const std::vector<uint32_t>& primes,
              const std::vector<uint32_t>& first_factors,
              uint64_t factor_origin) {
  for (size_t pi = 0; pi < primes.size(); ++pi) {
    const uint32_t p = primes[pi];
    for (size_t member = 0; member < 7; ++member) {
#ifdef RIECOIN_A54_OPTIONAL_PRIME_LIMIT
      if (p > RIECOIN_A54_OPTIONAL_PRIME_LIMIT && member >= 2u) continue;
#endif
      const uint32_t first = first_factors[pi * 7 + member];
      const uint32_t origin_mod = static_cast<uint32_t>(factor_origin % p);
      const uint32_t local = first >= origin_mod ? first - origin_mod
                                                 : first + p - origin_mod;
      for (uint32_t f = local; f < count; f += p)
        words[f >> 5u] &= ~(1u << (f & 31u));
    }
  }
}

bool oracle_constellation(const mpz_t first, const std::array<uint32_t, 7>& offsets,
                           uint32_t iterations) {
  mpz_t n;
  mpz_init(n);
  for (uint32_t offset : offsets) {
    mpz_add_ui(n, first, offset);
    if (mpz_probab_prime_p(n, static_cast<int>(iterations)) == 0) {
      mpz_clear(n);
      return false;
    }
  }
  mpz_clear(n);
  return true;
}

bool gmp_euler_jacobi_base_two(const mpz_t n) {
  if (mpz_cmp_ui(n, 3u) <= 0 || mpz_even_p(n) != 0)
    return mpz_cmp_ui(n, 2u) == 0;
  mpz_t exponent, residue, expected;
  mpz_inits(exponent, residue, expected, nullptr);
  mpz_sub_ui(exponent, n, 1u);
  mpz_fdiv_q_2exp(exponent, exponent, 1u);
  mpz_set_ui(residue, 2u);
  mpz_powm(residue, residue, exponent, n);
  const unsigned long modulo_eight = mpz_fdiv_ui(n, 8u);
  bool result = false;
  if (modulo_eight == 1u || modulo_eight == 7u) {
    result = mpz_cmp_ui(residue, 1u) == 0;
  } else {
    mpz_sub_ui(expected, n, 1u);
    result = mpz_cmp(residue, expected) == 0;
  }
  mpz_clears(exponent, residue, expected, nullptr);
  return result;
}

struct DynamicBitParityResult {
  uint64_t samples = 0;
  uint64_t mismatches = 0;
  uint64_t prime_false_negatives = 0;
  uint32_t minimum_bits = 0;
  uint32_t maximum_bits = 0;
};

DynamicBitParityResult run_dynamic_bit_parity_gate() {
  constexpr std::array<uint32_t, 4> kBitLengths{{1152u, 1153u, 1154u, 1160u}};
  std::vector<cgbn_mem_t<1280u>> packed;
  std::vector<uint8_t> expected;
  std::vector<uint8_t> known_prime;
  packed.reserve(kBitLengths.size() * 2u);
  expected.reserve(kBitLengths.size() * 2u);
  known_prime.reserve(kBitLengths.size() * 2u);

  mpz_t value, seed;
  mpz_inits(value, seed, nullptr);
  for (const uint32_t bits : kBitLengths) {
    // One proven prime and one deterministic odd integer at every boundary.
    mpz_set_ui(seed, 1u);
    mpz_mul_2exp(seed, seed, bits - 1u);
    mpz_add_ui(seed, seed, 0x5a5bu + bits);
    mpz_nextprime(value, seed);
    if (mpz_sizeinbase(value, 2) != bits)
      throw std::runtime_error("dynamic parity prime escaped requested bit length");
    packed.emplace_back();
    pack_candidate_1280(packed.back(), value);
    expected.push_back(gmp_euler_jacobi_base_two(value) ? 1u : 0u);
    known_prime.push_back(1u);

    mpz_set_ui(value, 1u);
    mpz_mul_2exp(value, value, bits - 1u);
    mpz_add_ui(value, value, 0x12345u + 2u * bits);
    if (mpz_even_p(value) != 0) mpz_add_ui(value, value, 1u);
    if (mpz_sizeinbase(value, 2) != bits)
      throw std::runtime_error("dynamic parity sample escaped requested bit length");
    packed.emplace_back();
    pack_candidate_1280(packed.back(), value);
    expected.push_back(gmp_euler_jacobi_base_two(value) ? 1u : 0u);
    known_prime.push_back(0u);
  }
  mpz_clears(value, seed, nullptr);

  cgbn_mem_t<1280u>* d_candidates = nullptr;
  uint8_t* d_verdicts = nullptr;
  std::vector<uint8_t> verdicts(packed.size(), 0u);
  cuda_check(cudaMalloc(&d_candidates, packed.size() * sizeof(packed.front())),
             "cudaMalloc dynamic parity candidates");
  cuda_check(cudaMalloc(&d_verdicts, verdicts.size() * sizeof(uint8_t)),
             "cudaMalloc dynamic parity verdicts");
  cuda_check(cudaMemcpy(d_candidates, packed.data(),
                        packed.size() * sizeof(packed.front()),
                        cudaMemcpyHostToDevice),
             "copy dynamic parity candidates");
  const uint32_t blocks = static_cast<uint32_t>(
      (packed.size() * 8u + riecoin_cuda::kPrpBlockThreads - 1u) /
      riecoin_cuda::kPrpBlockThreads);
  riecoin_cuda::euler_jacobi_dynamic_1280_tpi8_kernel
      <<<blocks, riecoin_cuda::kPrpBlockThreads>>>(
          d_candidates, d_verdicts, static_cast<uint32_t>(packed.size()));
  cuda_check(cudaGetLastError(), "dynamic parity kernel launch");
  cuda_check(cudaMemcpy(verdicts.data(), d_verdicts,
                        verdicts.size() * sizeof(uint8_t),
                        cudaMemcpyDeviceToHost),
             "copy dynamic parity verdicts");
  cuda_check(cudaFree(d_verdicts), "cudaFree dynamic parity verdicts");
  cuda_check(cudaFree(d_candidates), "cudaFree dynamic parity candidates");

  DynamicBitParityResult result;
  result.samples = verdicts.size();
  result.minimum_bits = kBitLengths.front();
  result.maximum_bits = kBitLengths.back();
  for (size_t i = 0; i < verdicts.size(); ++i) {
    if (verdicts[i] != expected[i]) ++result.mismatches;
    if (known_prime[i] != 0u && verdicts[i] == 0u)
      ++result.prime_false_negatives;
  }
  return result;
}

uint32_t share_prime_count(const mpz_t first,
                           const std::array<uint32_t, 7>& offsets,
                           uint32_t iterations,
                           std::array<uint64_t, 8>* tuple_counts) {
  mpz_t n;
  mpz_init(n);
  uint32_t prime_count = 0;
  if (tuple_counts != nullptr) ++(*tuple_counts)[0];
  for (size_t member = 0; member < offsets.size(); ++member) {
    mpz_add_ui(n, first, offsets[member]);
    if (mpz_probab_prime_p(n, static_cast<int>(iterations)) != 0) {
      ++prime_count;
      if (tuple_counts != nullptr) ++(*tuple_counts)[prime_count];
    } else {
      // Riecoin v1 requires the first two pattern members. Later holes are
      // allowed, but the share must still reach at least five primes of seven.
      if (member < 2 || prime_count + (offsets.size() - member - 1u) < 5u) {
        mpz_clear(n);
        return prime_count;
      }
    }
  }
  mpz_clear(n);
  return prime_count;
}

#ifdef RIECOIN_RLOW_SCIENCE_Q9
riecoin_strict_science_q9::Result strict_science_result(
    const mpz_t first, uint32_t pattern,
    const std::array<uint32_t, 7>& offsets,
    uint32_t exact_prime_count) {
  if (exact_prime_count != 7u) return {};
  return riecoin_strict_science_q9::verify(first, pattern, offsets, 64);
}

void account_strict_science(const riecoin_strict_science_q9::Result& result) {
  if (result.length >= 7u) ++rlow_exact_sevens;
  if (result.length >= 8u) ++rlow_exact_eights;
  if (result.length >= 9u) ++rlow_exact_nines;
}
#endif

struct RiecoinNonceV1 {
  std::array<uint8_t, 32> bytes{};
};

std::string nonce_uint256_hex(const RiecoinNonceV1& nonce) {
  std::ostringstream text;
  text << std::hex << std::setfill('0');
  for (auto it = nonce.bytes.rbegin(); it != nonce.bytes.rend(); ++it)
    text << std::setw(2) << static_cast<unsigned int>(*it);
  return text.str();
}

uint8_t hex_nibble(char value) {
  if (value >= '0' && value <= '9') return static_cast<uint8_t>(value - '0');
  if (value >= 'a' && value <= 'f') return static_cast<uint8_t>(value - 'a' + 10);
  if (value >= 'A' && value <= 'F') return static_cast<uint8_t>(value - 'A' + 10);
  throw std::runtime_error("invalid nonce hex digit");
}

RiecoinNonceV1 nonce_from_uint256_hex(const std::string& hex) {
  if (hex.size() != 64) throw std::runtime_error("nonce hex must contain 32 bytes");
  RiecoinNonceV1 nonce;
  for (size_t raw_index = 0; raw_index < nonce.bytes.size(); ++raw_index) {
    const size_t text_index = (nonce.bytes.size() - 1u - raw_index) * 2u;
    nonce.bytes[raw_index] = static_cast<uint8_t>(
        (hex_nibble(hex[text_index]) << 4u) | hex_nibble(hex[text_index + 1u]));
  }
  return nonce;
}

RiecoinNonceV1 encode_nonce_v1(uint16_t primorial_number,
                               uint64_t primorial_factor,
                               uint64_t primorial_offset,
                               uint16_t difficulty_offset) {
  RiecoinNonceV1 nonce;
  const uint16_t version = static_cast<uint16_t>(2u + (difficulty_offset << 5u));
  nonce.bytes[0] = static_cast<uint8_t>(version);
  nonce.bytes[1] = static_cast<uint8_t>(version >> 8u);
  for (uint32_t i = 0; i < 8; ++i) {
    nonce.bytes[2 + i] = static_cast<uint8_t>(primorial_offset >> (8u * i));
    nonce.bytes[14 + i] = static_cast<uint8_t>(primorial_factor >> (8u * i));
  }
  nonce.bytes[30] = static_cast<uint8_t>(primorial_number);
  nonce.bytes[31] = static_cast<uint8_t>(primorial_number >> 8u);
  return nonce;
}

void decode_nonce_v1(const RiecoinNonceV1& nonce,
                     uint16_t& primorial_number,
                     mpz_t primorial_factor,
                     mpz_t primorial_offset,
                     uint16_t& difficulty_offset) {
  const uint16_t version = static_cast<uint16_t>(nonce.bytes[0]) |
                           static_cast<uint16_t>(nonce.bytes[1]) << 8u;
  if ((version & 31u) != 2u) throw std::runtime_error("nonce is not Riecoin PoW v1");
  difficulty_offset = static_cast<uint16_t>((version & 65504u) >> 5u);
  primorial_number = static_cast<uint16_t>(nonce.bytes[30]) |
                     static_cast<uint16_t>(nonce.bytes[31]) << 8u;
  mpz_import(primorial_factor, 16, -1, sizeof(uint8_t), 0, 0,
             nonce.bytes.data() + 14);
  mpz_import(primorial_offset, 12, -1, sizeof(uint8_t), 0, 0,
             nonce.bytes.data() + 2);
}

void reconstruct_candidate_from_nonce(mpz_t result,
                                      const mpz_t target,
                                      const std::vector<uint32_t>& prime_table,
                                      const RiecoinNonceV1& nonce) {
  uint16_t primorial_number = 0, difficulty_offset = 0;
  mpz_t primorial, factor, offset, target_mod;
  mpz_inits(primorial, factor, offset, target_mod, nullptr);
  decode_nonce_v1(nonce, primorial_number, factor, offset, difficulty_offset);
  if (primorial_number > prime_table.size()) {
    mpz_clears(primorial, factor, offset, target_mod, nullptr);
    throw std::runtime_error("nonce primorial number exceeds prime table");
  }
  mpz_set_ui(primorial, 1u);
  for (uint16_t i = 0; i < primorial_number; ++i)
    mpz_mul_ui(primorial, primorial, prime_table[i]);
  mpz_mod(target_mod, target, primorial);
  mpz_sub(result, primorial, target_mod);
  mpz_add(result, result, target);
  mpz_addmul(result, factor, primorial);
  mpz_add(result, result, offset);
  mpz_clears(primorial, factor, offset, target_mod, nullptr);
}

int run_codec_self_test(const Options& options) {
  const auto prime_table = primes_to(options.prime_limit);
  const uint32_t primorial_number = options.initial_target_bits == 0
      ? options.primorial_number
      : derive_primorial_number(prime_table, options.initial_target_bits,
                                options.factor_max);
  if (primorial_number > prime_table.size())
    throw std::runtime_error("codec prime table is too short");

  mpz_t target, primorial, target_mod, base, direct, decoded, factor;
  mpz_inits(target, primorial, target_mod, base, direct, decoded, factor, nullptr);
  initialize_target(target, options);
  mpz_set_ui(primorial, 1u);
  for (uint32_t i = 0; i < primorial_number; ++i)
    mpz_mul_ui(primorial, primorial, prime_table[i]);
  mpz_mod(target_mod, target, primorial);
  mpz_sub(base, primorial, target_mod);
  mpz_add(base, base, target);
  mpz_add_u64(base, base, options.primorial_offset);
  const std::array<uint64_t, 5> samples{{0, 1, 17, 65535, 0x123456789abcdef0ull}};
  uint64_t mismatches = 0;
  std::string first_codec_delta = "0";
  for (uint64_t sample : samples) {
    mpz_import(factor, 1, -1, sizeof(sample), 0, 0, &sample);
    mpz_mul(direct, factor, primorial);
    mpz_add(direct, direct, base);
    const auto nonce = encode_nonce_v1(
        static_cast<uint16_t>(primorial_number), sample,
        options.primorial_offset, static_cast<uint16_t>(options.difficulty_offset));
    reconstruct_candidate_from_nonce(decoded, target, prime_table, nonce);
    if (mpz_cmp(direct, decoded) != 0) {
      if (first_codec_delta == "0") {
        mpz_sub(decoded, decoded, direct);
        first_codec_delta = mpz_hex(decoded);
      }
      ++mismatches;
    }
    mismatches += nonce.bytes[0] != static_cast<uint8_t>(2u + (options.difficulty_offset << 5u));
    mismatches += nonce.bytes[30] != static_cast<uint8_t>(primorial_number);
  }
  const auto genesis_nonce = nonce_from_uint256_hex(
      "00000000000000000000000000000000000000000000002990adb3a701960002");
  uint16_t genesis_primorial_number = 0, genesis_difficulty_offset = 0;
  mpz_t genesis_factor, genesis_offset, expected_genesis_offset;
  mpz_inits(genesis_factor, genesis_offset, expected_genesis_offset, nullptr);
  decode_nonce_v1(genesis_nonce, genesis_primorial_number, genesis_factor,
                  genesis_offset, genesis_difficulty_offset);
  mismatches += genesis_primorial_number != 0;
  mismatches += genesis_difficulty_offset != 0;
  mismatches += mpz_cmp_ui(genesis_factor, 0u) != 0;
  const uint64_t expected_offset_u64 = 11699549762945430ull;
  mpz_import(expected_genesis_offset, 1, -1, sizeof(expected_offset_u64), 0, 0,
             &expected_offset_u64);
  mismatches += mpz_cmp(genesis_offset, expected_genesis_offset) != 0;
  mpz_clears(genesis_factor, genesis_offset, expected_genesis_offset, nullptr);
  std::cout << "RIECOIN_NONCE_CODEC_V1\n"
            << "codec_samples=" << samples.size()
            << " codec_mismatches=" << mismatches
            << " primorial_number=" << primorial_number
            << " primorial_bits=" << mpz_sizeinbase(primorial, 2)
            << " target_bits=" << mpz_sizeinbase(target, 2) << '\n'
            << "first_codec_delta_hex=" << first_codec_delta << '\n'
            << "core_testnet_genesis_codec=" << (mismatches == 0 ? "PASS" : "FAIL") << '\n'
            << "cuda_initialized=0 rpc=0 submission=0\n"
            << "verdict=" << (mismatches == 0 ? "PASS" : "FAIL") << '\n';
  mpz_clears(target, primorial, target_mod, base, direct, decoded, factor, nullptr);
  return mismatches == 0 ? 0 : 2;
}

}  // namespace

#ifdef RIECOIN_RLOW_SKIP_CLEARED_BITS
#include "riecoin_rlow_monotone_clear_guard.cuh"
#endif

int main(int argc, char** argv) {
 try {
  Options options = parse_options(argc, argv);
#ifdef RIECOIN_A54_OPTIONAL_PRIME_LIMIT
  std::cout << "phase=a54_selective_sieve optional_prime_limit="
            << RIECOIN_A54_OPTIONAL_PRIME_LIMIT << " mandatory_prime_limit="
            << options.prime_limit << " mandatory_members=2" << std::endl;
#endif
#if defined(RIECOIN_RLOW_SPARSE_FAMILY_Q0) || defined(RIECOIN_RLOW_P31_FAMILY)
  // The backend's ordinary Silver setting is intentionally replaced by this
  // explicitly encoded candidate family. The resulting nonce still carries
  // the real primorial number and is replayed from that nonce before submit.
  options.primorial_number=31u;
  std::cout<<"phase=p31_family_bound primorial_number=31 submitted_nonce_bound=1"
           <<" q0_operator="
#ifdef RIECOIN_RLOW_SPARSE_FAMILY_Q0
           <<"scalar_sparse"
#else
           <<"silver_narrow_lazy"
#endif
           <<std::endl;
#endif
  if (options.codec_self_test) return run_codec_self_test(options);
  if (options.production_session_seconds < 30u)
    throw std::runtime_error("RLOW queued executable requires production >=30 seconds");
  const bool production_mode = options.production_session_seconds != 0u;
  const bool device_pipeline_mode = production_mode || options.network_batch ||
                                    options.q0_duration_seconds != 0u;
  const auto wall_begin = std::chrono::steady_clock::now();
  const int requested_device = options.device;
  auto uuid_for = [](const cudaUUID_t& raw_uuid) {
    std::ostringstream text;
    text << "GPU-" << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
      text << std::setw(2)
           << static_cast<unsigned int>(static_cast<unsigned char>(raw_uuid.bytes[i]));
      if (i == 3 || i == 5 || i == 7 || i == 9) text << '-';
    }
    std::string result = text.str();
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return result;
  };
  std::string expected_uuid = options.expected_uuid;
  std::transform(expected_uuid.begin(), expected_uuid.end(), expected_uuid.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (!expected_uuid.empty()) {
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    bool found = false;
    for (int ordinal = 0; ordinal < device_count; ++ordinal) {
      cudaDeviceProp probe{};
      cuda_check(cudaGetDeviceProperties(&probe, ordinal), "cudaGetDeviceProperties probe");
      if (uuid_for(probe.uuid) == expected_uuid) {
        options.device = ordinal;
        found = true;
        break;
      }
    }
    if (!found) throw std::runtime_error("authorized GPU UUID not visible to CUDA runtime");
  }
  cudaDeviceProp rlow_identity_probe{};
  cuda_check(cudaGetDeviceProperties(&rlow_identity_probe, options.device), "RLOW identity before allocation");
  if (uuid_for(rlow_identity_probe.uuid) != expected_uuid ||
      !rlow_pci_equal(options.expected_pci, rlow_identity_probe))
    throw std::runtime_error("RLOW UUID/PCI identity mismatch; allocation and kernels forbidden");
  std::cout << "phase=rlow_identity_pass uuid=" << expected_uuid
            << " pci=" << options.expected_pci << std::endl;
  cuda_check(cudaSetDevice(options.device), "cudaSetDevice");
#ifdef RIECOIN_RLOW_SKIP_CLEARED_BITS
  riecoin_rlow_monotone_clear_guard::run_cuda_guard();
#endif
  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, options.device), "cudaGetDeviceProperties");
  const cudaUUID_t raw_uuid = prop.uuid;
  const std::string actual_uuid = uuid_for(raw_uuid);
  std::ostringstream actual_pci_text;
  actual_pci_text << std::hex << std::setfill('0')
                  << std::setw(8) << prop.pciDomainID << ':'
                  << std::setw(2) << prop.pciBusID << ':'
                  << std::setw(2) << prop.pciDeviceID << ".0";
  const std::string actual_pci = actual_pci_text.str();
  if (!expected_uuid.empty() && actual_uuid != expected_uuid)
    throw std::runtime_error("authorized GPU UUID mismatch: requested_device=" +
                             std::to_string(options.device) + " actual=" + actual_uuid);

  DynamicBitParityResult dynamic_bit_parity;
#ifdef RIECOIN_RLOW_LAZY_BASE2_Q0
  run_lazy_base2_guard();
#endif
#ifdef RIECOIN_RLOW_CARRYLATE_Q0
  run_carrylate_q0_guard();
  const unsigned long long carrylate_zero_paths[3]{};
  cuda_check(cudaMemcpyToSymbol(riecoin_rlow_carrylate::rare_paths,
      carrylate_zero_paths, sizeof(carrylate_zero_paths)), "carrylate reset guard counters");
#endif
  if (options.network_batch || production_mode) {
    dynamic_bit_parity = run_dynamic_bit_parity_gate();
    std::cout << "phase=dynamic_bit_parity_complete"
              << " samples=" << dynamic_bit_parity.samples
              << " min_bits=" << dynamic_bit_parity.minimum_bits
              << " max_bits=" << dynamic_bit_parity.maximum_bits
              << " mismatches=" << dynamic_bit_parity.mismatches
              << " prime_false_negatives="
              << dynamic_bit_parity.prime_false_negatives << std::endl;
    if (dynamic_bit_parity.mismatches != 0u ||
        dynamic_bit_parity.prime_false_negatives != 0u)
      throw std::runtime_error("dynamic 1280-bit Euler-Jacobi parity gate failed");
  }

  const auto offsets = cumulative_pattern(options.pattern);
  const auto precompute_begin = std::chrono::steady_clock::now();

#ifdef RIECOIN_RLOW_PRIME_ROOT_CACHE
  auto all_primes = riecoin_prime_root_cache::primorial_seed(
      options.prime_limit, options.primorial_number,
      options.initial_target_bits, options.factor_max);
  const double cache_seed_ms = riecoin_prime_root_cache::elapsed_ms(precompute_begin);
#else
  auto all_primes = primes_to(options.prime_limit);
#endif
  if (options.initial_target_bits != 0)
    options.primorial_number = derive_primorial_number(
        all_primes, options.initial_target_bits, options.factor_max);
  if (options.primorial_number >= all_primes.size())
    throw std::runtime_error("prime limit does not cover requested primorial number");

  mpz_t target, primorial, target_mod, base;
  mpz_inits(target, primorial, target_mod, base, nullptr);
  initialize_target(target, options);
  mpz_set_ui(primorial, 1u);
  for (uint32_t i = 0; i < options.primorial_number; ++i)
    mpz_mul_ui(primorial, primorial, all_primes[i]);
  mpz_mod(target_mod, target, primorial);
  mpz_sub(base, primorial, target_mod);
  mpz_add(base, base, target);
  mpz_add_u64(base, base, options.primorial_offset);
#ifdef RIECOIN_RLOW_NARROW_LAZY_Q0
  const bool narrow_q0 = riecoin_narrow_lazy_q0::covers_reservation(base,primorial,options.factor_max);
  if(narrow_q0)riecoin_narrow_lazy_q0::guard();
  std::cout<<"phase=q0_width_route storage_bits="<<(narrow_q0?1152:1280)
           <<" tpi="<<(narrow_q0?4:
#ifdef RIECOIN_RLOW_WIDE_TPI4
           4
#else
           8
#endif
           )<<" whole_reservation_bound=1"<<std::endl;
#endif
  const std::string target_hex_canonical = mpz_hex(target);
  mpz_t r283_upper;mpz_init(r283_upper);
  mpz_import(r283_upper,1,-1,sizeof(options.factor_max),0,0,&options.factor_max);
  mpz_mul(r283_upper,r283_upper,primorial);mpz_add(r283_upper,r283_upper,base);
  const auto r283_bits=mpz_sizeinbase(r283_upper,2);
  const bool r283_active=r283::enabled&&r283::qualified&&r283_bits>=r322::minimum_bits()&&r283_bits<=1184u&&
    mpz_sizeinbase(base,2)>=3u&&mpz_odd_p(base)&&mpz_even_p(primorial);
  mpz_clear(r283_upper);
  std::uint32_t *r283_slots=nullptr,*r283_total=nullptr;
  if(r283_active){
    cuda_check(cudaGetSymbolAddress(reinterpret_cast<void**>(&r283_slots),riecoin_lazy_base2_window::r249_slots),"R283 slot binding");
    cuda_check(cudaGetSymbolAddress(reinterpret_cast<void**>(&r283_total),riecoin_lazy_base2_window::r249_total),"R283 total binding");
  }
  std::cout<<"phase=r283_route active="<<r283_active<<" maximum_bits="<<r283_bits<<" dense_worklist=1 complete_fallback=1"<<std::endl;
  const std::string primorial_hex_canonical = mpz_hex(primorial);
  std::ostringstream offsets_json_text;
  offsets_json_text << '[';
  for (size_t i = 0; i < offsets.size(); ++i) {
    if (i != 0) offsets_json_text << ',';
    offsets_json_text << offsets[i];
  }
  offsets_json_text << ']';
  const std::string offsets_json = offsets_json_text.str();
#ifdef ASTRA_R345_PREPARE
  // P/B/offsets are immutable until the join. A failed main branch waits for
  // the worker before unwinding; no device state is consumed before readiness.
  r345::Preparation r345_setup;
  if(r345::requested())r345_setup.start(options.device,[&]{
    ASTRA_R201_INITIALIZE(primorial,base,offsets,options.prime_limit);
  });
#endif

  std::vector<uint32_t> primes;
  std::vector<uint32_t> first_factors;
  std::size_t sieve_prime_count = 0u;
#ifdef RIECOIN_RLOW_DEVICE_ROOTS
  riecoin_gpu_root_builder::DeviceRootTable device_roots;
#endif
#ifdef RIECOIN_RLOW_PRIME_ROOT_CACHE
  {
    std::filesystem::path cache_dir;
    try {
      cache_dir = options.prime_root_cache_dir_explicit
          ? std::filesystem::path(options.prime_root_cache_dir)
          : riecoin_prime_root_cache::default_directory();
    } catch (const std::exception&) { /* unavailable temp directory: stateless */ }
    riecoin_prime_root_cache::Metrics cache_metrics;
    auto cache_basis = riecoin_prime_root_cache::acquire(
        cache_dir, options.prime_limit, options.primorial_number,
        primorial, all_primes, cache_metrics);
    const auto roots_begin = std::chrono::steady_clock::now();
#ifdef RIECOIN_RLOW_DEVICE_ROOTS
    device_roots = riecoin_prime_root_cache::roots_for_device(
        cache_basis, base, offsets
#ifdef RIECOIN_RLOW_COMPRESSED_ROOTS
        , options.candidates
#endif
        );
    sieve_prime_count = device_roots.prime_count;
#else
    auto exact_roots = riecoin_prime_root_cache::roots_for(cache_basis, base, offsets);
#endif
    const double root_ms = riecoin_prime_root_cache::elapsed_ms(roots_begin);
    all_primes.swap(cache_basis.all_primes); // retain full nonce reconstruction table
#ifndef RIECOIN_RLOW_DEVICE_ROOTS
    primes.swap(exact_roots.primes);
    first_factors.swap(exact_roots.first_factors);
#endif
    std::cout << "phase=prime_inverse_cache status=" << cache_metrics.status
              << " cache_path=" << std::quoted(cache_metrics.cache_path)
              << " published=" << cache_metrics.published
              << " bytes=" << cache_metrics.file_bytes
              << " seed_ms=" << cache_seed_ms
              << " key_ms=" << cache_metrics.key_ms
              << " load_ms=" << cache_metrics.load_ms
              << " verify_ms=" << cache_metrics.verify_ms
              << " build_ms=" << cache_metrics.build_ms
              << " publish_ms=" << cache_metrics.publish_ms
              << " roots_ms=" << root_ms << std::endl;
  }
#elif defined(RIECOIN_RLOW_FAST_PRECOMPUTE)
  auto exact_roots = riecoin_prime_root_precompute::roots_for(
      all_primes, options.primorial_number, primorial, base, offsets);
  primes.swap(exact_roots.primes);
  first_factors.swap(exact_roots.first_factors);
#else
  for (size_t prime_index = options.primorial_number;
       prime_index < all_primes.size(); ++prime_index) {
    const uint32_t p = all_primes[prime_index];
    const uint32_t base_mod = static_cast<uint32_t>(mpz_fdiv_ui(base, p));
    const uint32_t inv = inverse_mod(static_cast<uint32_t>(mpz_fdiv_ui(primorial, p)), p);
    primes.push_back(p);
    for (uint32_t offset : offsets) {
      const uint32_t residue = (base_mod + offset % p) % p;
      const uint32_t neg = residue == 0 ? 0 : p - residue;
      first_factors.push_back(static_cast<uint32_t>((static_cast<uint64_t>(neg) * inv) % p));
    }
  }
#endif
#ifndef RIECOIN_RLOW_DEVICE_ROOTS
  sieve_prime_count = primes.size();
#endif
  if (sieve_prime_count > UINT32_MAX / 7u)
    throw std::length_error("sieve root pair count overflow");
#ifdef ASTRA_R201_INITIALIZE
#ifdef ASTRA_R345_PREPARE
  if(r345_setup.active())r345_setup.join();
  else {
    const auto r345_serial_begin=std::chrono::steady_clock::now();
    ASTRA_R201_INITIALIZE(primorial,base,offsets,options.prime_limit);
    std::cout<<"phase=r345_readiness mode=serial secondary_ms="<<r345::ms(r345_serial_begin)<<" errors=0"<<std::endl;
  }
#else
  ASTRA_R201_INITIALIZE(primorial,base,offsets,options.prime_limit);
#endif
#endif
  const auto precompute_end = std::chrono::steady_clock::now();
  std::cout << "phase=precompute_complete ms="
            << std::chrono::duration<double, std::milli>(precompute_end - precompute_begin).count()
            << " primes=" << sieve_prime_count << std::endl;

  const size_t word_count = (static_cast<size_t>(options.candidates) + 31u) / 32u;
  std::vector<uint32_t> initial(word_count, 0xffffffffu);
  if ((options.candidates & 31u) != 0)
    initial.back() = (1u << (options.candidates & 31u)) - 1u;
  std::vector<uint32_t> gpu_words(initial), cpu_words(initial);

  const auto gpu_setup_begin = std::chrono::steady_clock::now();
  uint32_t *d_words = nullptr, *d_primes = nullptr, *d_first = nullptr;
  cuda_check(cudaMalloc(&d_words, word_count * sizeof(uint32_t)), "cudaMalloc words");
#ifdef RIECOIN_RLOW_DEVICE_ROOTS
  d_primes = device_roots.release_primes();
  d_first = device_roots.release_first_factors();
  if (d_primes == nullptr || d_first == nullptr)
    throw std::runtime_error("device-resident root ownership transfer failed");
#else
  cuda_check(cudaMalloc(&d_primes, primes.size() * sizeof(uint32_t)), "cudaMalloc primes");
  cuda_check(cudaMalloc(&d_first, first_factors.size() * sizeof(uint32_t)), "cudaMalloc factors");
#endif
  cuda_check(cudaMemcpy(d_words, initial.data(), word_count * sizeof(uint32_t), cudaMemcpyHostToDevice), "copy words");
#ifndef RIECOIN_RLOW_DEVICE_ROOTS
  cuda_check(cudaMemcpy(d_primes, primes.data(), primes.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "copy primes");
  cuda_check(cudaMemcpy(d_first, first_factors.data(), first_factors.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "copy factors");
#endif
  const auto gpu_setup_end = std::chrono::steady_clock::now();
  std::cout << "phase=gpu_setup_complete ms="
            << std::chrono::duration<double, std::milli>(gpu_setup_end - gpu_setup_begin).count()
            << std::endl;

  cudaEvent_t begin{}, end{};
  cuda_check(cudaEventCreate(&begin), "event begin");
  cuda_check(cudaEventCreate(&end), "event end");
  cuda_check(cudaEventRecord(begin), "record begin");
  const uint32_t pair_count = static_cast<uint32_t>(sieve_prime_count * 7u);
  rlow_sieve_launch(
      d_words, options.candidates, d_primes, d_first, pair_count,
      options.factor_origin);
  cuda_check(cudaGetLastError(), "mark_composites launch");
  cuda_check(cudaEventRecord(end), "record end");
  cuda_check(cudaEventSynchronize(end), "synchronize kernel");
  float gpu_ms = 0.0f;
  cuda_check(cudaEventElapsedTime(&gpu_ms, begin, end), "elapsed time");
  std::cout << "phase=kernel_complete ms=" << gpu_ms << std::endl;
  const auto d2h_begin = std::chrono::steady_clock::now();
  cuda_check(cudaMemcpy(gpu_words.data(), d_words, word_count * sizeof(uint32_t), cudaMemcpyDeviceToHost), "copy survivors");
  const auto d2h_end = std::chrono::steady_clock::now();
  std::cout << "phase=d2h_complete ms="
            << std::chrono::duration<double, std::milli>(d2h_end - d2h_begin).count()
            << std::endl;

  const auto cpu_begin = std::chrono::steady_clock::now();
  size_t mismatch_words = 0u;
#ifdef RIECOIN_RLOW_DEVICE_ROOTS
  uint32_t* d_reference_words = nullptr;
  uint32_t* d_reference_mismatches = nullptr;
  cuda_check(cudaMalloc(&d_reference_words, word_count * sizeof(uint32_t)),
             "device sieve reference bitmap alloc");
  cuda_check(cudaMalloc(&d_reference_mismatches, sizeof(uint32_t)),
             "device sieve reference mismatch alloc");
  cuda_check(cudaMemcpy(d_reference_words, initial.data(),
                        word_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
             "device sieve reference bitmap init");
  cuda_check(cudaMemset(d_reference_mismatches, 0, sizeof(uint32_t)),
             "device sieve reference mismatch reset");
  mark_composites<<<(pair_count + 255u) / 256u, 256u>>>(
      d_reference_words, options.candidates, d_primes, d_first, pair_count,
      options.factor_origin);
  compare_complete_bitmaps<<<(static_cast<uint32_t>(word_count) + 255u) / 256u, 256u>>>(
      d_words, d_reference_words, static_cast<uint32_t>(word_count),
      d_reference_mismatches);
  cuda_check(cudaGetLastError(), "device complete sieve reference launch");
  uint32_t device_mismatches = 0u;
  cuda_check(cudaMemcpy(&device_mismatches, d_reference_mismatches,
                        sizeof(device_mismatches), cudaMemcpyDeviceToHost),
             "device complete sieve reference result");
  mismatch_words = device_mismatches;
  cuda_check(cudaFree(d_reference_mismatches), "device sieve reference mismatch free");
  cuda_check(cudaFree(d_reference_words), "device sieve reference bitmap free");
#else
  cpu_mark(cpu_words, options.candidates, primes, first_factors,
           options.factor_origin);
#endif
  const auto cpu_end = std::chrono::steady_clock::now();
  const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_begin).count();
#ifdef RIECOIN_RLOW_DEVICE_ROOTS
  std::cout << "phase=device_reference_complete ms=" << cpu_ms
            << " complete_bitmap_words=" << word_count
            << " mismatches=" << mismatch_words << std::endl;
#else
  std::cout << "phase=cpu_reference_complete ms=" << cpu_ms << std::endl;
  for (size_t i = 0; i < word_count; ++i)
    mismatch_words += gpu_words[i] != cpu_words[i];
#endif

  const auto compaction_begin = std::chrono::steady_clock::now();
  std::vector<uint32_t> survivors;
  survivors.reserve(options.candidates / 32u);
  for (uint32_t f = 0; f < options.candidates; ++f)
    if ((gpu_words[f >> 5u] >> (f & 31u)) & 1u) survivors.push_back(f);
  const auto compaction_end = std::chrono::steady_clock::now();
  std::cout << "phase=compaction_complete ms="
            << std::chrono::duration<double, std::milli>(compaction_end - compaction_begin).count()
            << " survivors=" << survivors.size() << std::endl;

  std::vector<uint8_t> gpu_q0_verdicts;
  std::vector<cgbn_mem_t<1280u>> q0_packed;
  cgbn_mem_t<1280u>* d_q0_candidates = nullptr;
  uint32_t* d_q0_initial_words = nullptr;
  cgbn_mem_t<1280u>* d_q0_base = nullptr;
  cgbn_mem_t<1280u>* d_q0_primorial = nullptr;
  uint8_t* d_q0_verdicts = nullptr;
  uint64_t* d_q0_factors = nullptr;
  uint64_t* d_q0_pass_factors = nullptr;
  uint32_t* d_q0_count = nullptr;
  uint32_t* d_q0_pass_count = nullptr;
  uint32_t* d_q0_overflow_count = nullptr;
  unsigned long long* d_q0_total_count = nullptr;
  cgbn_mem_t<1280u>* d_tuple_members = nullptr;
  uint8_t* d_tuple_verdicts = nullptr;
  uint32_t* d_tuple_offsets = nullptr;
  unsigned long long* d_tuple_member_pass_counts = nullptr;
  unsigned long long* d_tuple_active_counts = nullptr;
  unsigned long long* d_rejected_counts = nullptr;
  uint64_t* d_rejected_sample_factors = nullptr;
  uint64_t* d_complete_tuple_factors = nullptr;
  unsigned long long* d_complete_tuple_count = nullptr;
  uint32_t* d_tuple_overflow_count = nullptr;
  double q0_pack_ms = 0.0;
  double q0_gpu_ms = 0.0;
  uint64_t q0_measured_loops = 0u;
  bool q0_reservation_exhausted = false;
  bool q0_resident_cancelled = false; // Only set after a complete, synchronized bundle.
  uint64_t q0_total_positions = 0u;
  uint64_t q0_total_entries = 0u;
  uint64_t q0_total_passes = 0u;
  uint64_t q0_replay_mismatch_words = 0u;
  uint64_t q0_replay_verdict_mismatches = 0u;
  uint64_t q0_drops = 0u;
  uint64_t q0_duplicates = 0u;
  double q0_measured_ms = 0.0;
  double q0_reset_ms = 0.0;
  double q0_sieve_ms = 0.0;
  double q0_d2h_replay_ms = 0.0;
  double q0_compact_pack_ms = 0.0;
  double q0_h2d_candidates_ms = 0.0;
  double q0_prp_gpu_ms = 0.0;
  double q0_verdict_d2h_ms = 0.0;
  double tuple_gpu_ms = 0.0;
  double exact_verify_ms = 0.0;
  uint64_t gpu_complete_tuples = 0u;
  uint64_t verifier_rejected_tuples = 0u;
  uint64_t rejected_audit_samples = 0u;
  uint64_t rejected_audit_exact_share_false_negatives = 0u;
  uint64_t rejected_audit_nonce_mismatches = 0u;
  std::array<uint64_t, 6> production_member_pass_counts{};
  std::array<uint64_t, 6> production_active_counts{};
  std::array<uint64_t, 7> production_rejected_counts{};
  if (device_pipeline_mode) {
    const size_t q0_base_bits = mpz_sizeinbase(base, 2);
    if (q0_base_bits < 3u || q0_base_bits > 1280u)
      throw std::runtime_error("Q0 GPU mode requires candidates within the dynamic 1280-bit envelope");
    if (survivors.empty())
      throw std::runtime_error("Q0 fixture produced no sieve survivors");
    const auto pack_begin = std::chrono::steady_clock::now();
    q0_packed.resize(survivors.size());
    mpz_t q0_candidate, q0_factor, q0_base_origin;
    mpz_inits(q0_candidate, q0_factor, q0_base_origin, nullptr);
    mpz_set_u64(q0_factor, options.factor_origin);
    mpz_mul(q0_factor, q0_factor, primorial);
    mpz_add(q0_base_origin, base, q0_factor);
    for (size_t index = 0u; index < survivors.size(); ++index) {
      mpz_set_ui(q0_factor, survivors[index]);
      mpz_mul(q0_factor, q0_factor, primorial);
      mpz_add(q0_candidate, q0_base_origin, q0_factor);
      pack_candidate_1280(q0_packed[index], q0_candidate);
    }
    q0_pack_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - pack_begin).count();
    gpu_q0_verdicts.resize(survivors.size());
    cgbn_mem_t<1280u> q0_base_packed{}, q0_primorial_packed{};
    pack_unsigned_1280(q0_base_packed, q0_base_origin);
    mpz_clears(q0_candidate, q0_factor, q0_base_origin, nullptr);
    pack_unsigned_1280(q0_primorial_packed, primorial);
    const size_t q0_capacity = kRlowPrpCapacity;
    if (survivors.size() > q0_capacity)
      throw std::runtime_error("RLOW cold survivor count exceeds independent PRP capacity");
    cuda_check(cudaMalloc(&d_q0_candidates,
                          q0_capacity * sizeof(q0_packed[0])),
               "cudaMalloc Q0 candidates");
    cuda_check(cudaMalloc(&d_q0_initial_words,
                          word_count * sizeof(uint32_t)),
               "cudaMalloc Q0 initial bitmap");
    cuda_check(cudaMalloc(&d_q0_verdicts, q0_capacity),
               "cudaMalloc Q0 verdicts");
    cuda_check(cudaMalloc(&d_q0_factors, q0_capacity * sizeof(uint64_t)),
               "cudaMalloc Q0 factors");
    cuda_check(cudaMalloc(&d_q0_pass_factors, q0_capacity * sizeof(uint64_t)),
               "cudaMalloc Q0 pass factors");
    cuda_check(cudaMalloc(&d_q0_base, sizeof(q0_base_packed)),
               "cudaMalloc Q0 base");
    cuda_check(cudaMalloc(&d_q0_primorial, sizeof(q0_primorial_packed)),
               "cudaMalloc Q0 primorial");
    cuda_check(cudaMalloc(&d_q0_count, sizeof(uint32_t)),
               "cudaMalloc Q0 count");
    cuda_check(cudaMalloc(&d_q0_pass_count, sizeof(uint32_t)),
               "cudaMalloc Q0 pass count");
    cuda_check(cudaMalloc(&d_q0_overflow_count, sizeof(uint32_t)),
               "cudaMalloc Q0 overflow count");
    cuda_check(cudaMalloc(&d_q0_total_count, sizeof(unsigned long long)),
               "cudaMalloc Q0 total count");
    if (production_mode) {
      const size_t tuple_member_capacity = q0_capacity * 6u;
      const size_t rejected_sample_slots = std::max<size_t>(
          1u, static_cast<size_t>(options.rejected_audit_per_stage) * 7u);
      std::array<uint32_t, 6> tuple_offsets{};
      for (size_t i = 0; i < tuple_offsets.size(); ++i)
        tuple_offsets[i] = offsets[i + 1u];
      cuda_check(cudaMalloc(&d_tuple_members,
                            tuple_member_capacity * sizeof(q0_packed[0])),
                 "cudaMalloc production tuple members");
      cuda_check(cudaMalloc(&d_tuple_verdicts, tuple_member_capacity),
                 "cudaMalloc production tuple verdicts");
      cuda_check(cudaMalloc(&d_tuple_offsets,
                            tuple_offsets.size() * sizeof(uint32_t)),
                 "cudaMalloc production tuple offsets");
      cuda_check(cudaMemcpy(d_tuple_offsets, tuple_offsets.data(),
                            tuple_offsets.size() * sizeof(uint32_t),
                            cudaMemcpyHostToDevice),
                 "copy production tuple offsets");
      cuda_check(cudaMalloc(&d_tuple_member_pass_counts,
                            6u * sizeof(unsigned long long)),
                 "cudaMalloc production member pass counts");
      cuda_check(cudaMalloc(&d_tuple_active_counts,
                            6u * sizeof(unsigned long long)),
                 "cudaMalloc production active counts");
      cuda_check(cudaMalloc(&d_rejected_counts,
                            7u * sizeof(unsigned long long)),
                 "cudaMalloc production rejected counts");
      cuda_check(cudaMalloc(&d_rejected_sample_factors,
                            rejected_sample_slots * sizeof(uint64_t)),
                 "cudaMalloc production rejected audit samples");
      cuda_check(cudaMalloc(&d_complete_tuple_factors,
                            q0_capacity * sizeof(uint64_t)),
                 "cudaMalloc production complete tuples");
      cuda_check(cudaMalloc(&d_complete_tuple_count,
                            sizeof(unsigned long long)),
                 "cudaMalloc production complete tuple count");
      cuda_check(cudaMalloc(&d_tuple_overflow_count, sizeof(uint32_t)),
                 "cudaMalloc production tuple overflow count");
    }
    cuda_check(cudaMemcpy(d_q0_base, &q0_base_packed, sizeof(q0_base_packed),
                          cudaMemcpyHostToDevice), "copy Q0 base");
    cuda_check(cudaMemcpy(d_q0_primorial, &q0_primorial_packed,
                          sizeof(q0_primorial_packed), cudaMemcpyHostToDevice),
               "copy Q0 primorial");
    cuda_check(cudaMemcpy(d_q0_initial_words, initial.data(),
                          word_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
               "copy Q0 initial bitmap");
    cuda_check(cudaMemcpy(d_q0_candidates, q0_packed.data(),
                          q0_packed.size() * sizeof(q0_packed[0]),
                          cudaMemcpyHostToDevice), "copy Q0 candidates");
    const uint32_t q0_blocks =
        (static_cast<uint32_t>(q0_packed.size()) + 15u) / 16u;
    cudaEvent_t q0_begin{}, q0_end{};
    cuda_check(cudaEventCreate(&q0_begin), "Q0 event begin");
    cuda_check(cudaEventCreate(&q0_end), "Q0 event end");
    cuda_check(cudaEventRecord(q0_begin), "Q0 record begin");
    riecoin_cuda::euler_jacobi_dynamic_1280_tpi8_kernel
        <<<q0_blocks, riecoin_cuda::kPrpBlockThreads>>>(
            d_q0_candidates, d_q0_verdicts,
            static_cast<uint32_t>(q0_packed.size()));
    cuda_check(cudaGetLastError(), "Q0 launch");
    cuda_check(cudaEventRecord(q0_end), "Q0 record end");
    cuda_check(cudaEventSynchronize(q0_end), "Q0 sync");
    float measured_q0_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&measured_q0_ms, q0_begin, q0_end),
               "Q0 elapsed");
    q0_gpu_ms = measured_q0_ms;
    cuda_check(cudaMemcpy(gpu_q0_verdicts.data(), d_q0_verdicts,
                          gpu_q0_verdicts.size(), cudaMemcpyDeviceToHost),
               "copy Q0 verdicts");
    cudaEventDestroy(q0_begin);
    cudaEventDestroy(q0_end);
    std::cout << "phase=q0_warmup_complete entries=" << survivors.size()
              << " pack_ms=" << q0_pack_ms
              << " gpu_ms=" << q0_gpu_ms << std::endl;
  }

  std::atomic<uint64_t> next_survivor{0}, accepted_count{0},
      result_journaled_count{0}, result_journal_error_count{0},
      nonce_mismatch_count{0}, checked_count{0}, q0_mismatch_count{0},
      q0_prime_false_negative_count{0};
  std::array<std::atomic<uint64_t>, 8> tuple_counts{};
  std::mutex checkpoint_mutex;
  std::mutex result_journal_mutex;
  std::vector<std::string> pending_result_records;
  const auto oracle_begin = std::chrono::steady_clock::now();
  std::cout << "phase=oracle_begin checked=0 survivors=" << survivors.size()
            << " threads=" << options.oracle_threads << std::endl;
  const uint64_t checkpoint_stride = 256u;
  auto oracle_worker = [&]() {
    mpz_t candidate, factor, nonce_candidate;
    mpz_inits(candidate, factor, nonce_candidate, nullptr);
    std::array<uint64_t, 8> local_tuple_counts{};
    while (true) {
      const uint64_t survivor_index = next_survivor.fetch_add(1u);
      if (survivor_index >= survivors.size()) break;
      const uint32_t local_factor = survivors[static_cast<size_t>(survivor_index)];
      const uint64_t f = options.factor_origin + local_factor;
      mpz_set_u64(factor, f);
      mpz_mul(factor, factor, primorial);
      mpz_add(candidate, base, factor);
      const auto nonce = encode_nonce_v1(
          static_cast<uint16_t>(options.primorial_number), f,
          options.primorial_offset, static_cast<uint16_t>(options.difficulty_offset));
      reconstruct_candidate_from_nonce(nonce_candidate, target, all_primes, nonce);
      if (mpz_cmp(candidate, nonce_candidate) != 0)
        nonce_mismatch_count.fetch_add(1u);
      const bool gpu_q0 = !gpu_q0_verdicts.empty() &&
          gpu_q0_verdicts[static_cast<size_t>(survivor_index)] != 0u;
      if (!gpu_q0_verdicts.empty()) {
        const bool expected_q0 = gmp_euler_jacobi_base_two(candidate);
        if (gpu_q0 != expected_q0) q0_mismatch_count.fetch_add(1u);
        if (mpz_probab_prime_p(candidate, 1) != 0 && !gpu_q0)
          q0_prime_false_negative_count.fetch_add(1u);
      }
      if (options.network_batch && !gpu_q0) {
        ++local_tuple_counts[0];
        const uint64_t checked = checked_count.fetch_add(1u) + 1u;
        if (checked % checkpoint_stride == 0u) {
          const std::lock_guard<std::mutex> lock(checkpoint_mutex);
          const auto checkpoint = std::chrono::steady_clock::now();
          std::cout << "phase=oracle_progress checked=" << checked_count.load()
                    << " accepted=" << accepted_count.load()
                    << " ms=" << std::chrono::duration<double, std::milli>(
                           checkpoint - oracle_begin).count() << std::endl;
        }
        continue;
      }
      const uint32_t screened_prime_count =
          share_prime_count(candidate, offsets, 1, &local_tuple_counts);
      const uint32_t exact_prime_count = screened_prime_count >= 5u
          ? share_prime_count(candidate, offsets, 32, nullptr)
          : 0u;
#ifdef RIECOIN_RLOW_SCIENCE_Q9
      const auto strict_science = strict_science_result(
          candidate, options.pattern, offsets, exact_prime_count);
#endif
      if (exact_prime_count >= 5u) {
        accepted_count.fetch_add(1u);
        std::ostringstream result_record;
        result_record
            << "{\"schema\":\"riecoin-exact-result-v1\""
            << ",\"event\":\"exact_constellation\""
            << ",\"coin\":\"riecoin\""
            << ",\"work_id\":\"" << json_escape(options.work_id) << "\""
            << ",\"template_id\":\"" << json_escape(options.template_id) << "\""
            << ",\"gpu_uuid\":\"" << json_escape(actual_uuid) << "\""
            << ",\"gpu_identity_sha256\":\"" << json_escape(options.gpu_identity_sha256) << "\""
            << ",\"runtime_sha256\":\"" << json_escape(options.runtime_sha256) << "\""
            << ",\"factor\":\"" << f << "\""
            << ",\"nonce_v1_uint256_hex\":\"" << nonce_uint256_hex(nonce) << "\""
            << ",\"candidate_hex\":\"" << mpz_hex(candidate) << "\""
            << ",\"pattern\":" << options.pattern
            << ",\"offsets\":" << offsets_json
            << ",\"primorial_number\":" << options.primorial_number
            << ",\"primorial_hex\":\"" << primorial_hex_canonical << "\""
            << ",\"primorial_offset\":\"" << options.primorial_offset << "\""
             << ",\"target_hex\":\"" << target_hex_canonical << "\""
            << ",\"prime_count\":" << exact_prime_count;
#ifdef RIECOIN_RLOW_SCIENCE_Q9
        result_record << riecoin_strict_science_q9::json_fields(strict_science);
#endif
        result_record
            << ",\"proof_state\":\"riecoin_v1_min5_exact_32_rounds\""
            << ",\"terminal_status\":\"proof_complete\""
            << ",\"submission_status\":\"not_attempted\""
            << ",\"accepted\":false,\"rejected\":false,\"stale\":false}"
            ;
        const std::lock_guard<std::mutex> lock(result_journal_mutex);
        if (options.network_batch) {
          pending_result_records.push_back(result_record.str());
        } else if (append_jsonl_durable(options.results_jsonl, result_record.str())) {
          result_journaled_count.fetch_add(1u);
        } else {
          result_journal_error_count.fetch_add(1u);
        }
      }
      const uint64_t checked = checked_count.fetch_add(1u) + 1u;
      if (checked % checkpoint_stride == 0u) {
        const std::lock_guard<std::mutex> lock(checkpoint_mutex);
        const auto checkpoint = std::chrono::steady_clock::now();
        std::cout << "phase=oracle_progress checked=" << checked_count.load()
                  << " accepted=" << accepted_count.load()
                  << " ms=" << std::chrono::duration<double, std::milli>(checkpoint - oracle_begin).count()
                  << std::endl;
      }
    }
    for (size_t i = 0; i < local_tuple_counts.size(); ++i)
      tuple_counts[i].fetch_add(local_tuple_counts[i]);
    mpz_clears(candidate, factor, nonce_candidate, nullptr);
  };
  std::vector<std::thread> oracle_workers;
  if (!production_mode) {
    oracle_workers.reserve(options.oracle_threads);
    for (uint32_t i = 0; i < options.oracle_threads; ++i)
      oracle_workers.emplace_back(oracle_worker);
    for (auto& worker : oracle_workers) worker.join();
  } else {
    std::cout << "phase=debug_oracle_removed checked=0 policy="
                 "gpu_tuple_then_rare_cpu_exact" << std::endl;
  }
  uint64_t accepted = accepted_count.load();
  uint64_t result_journaled = result_journaled_count.load();
  uint64_t result_journal_errors = result_journal_error_count.load();
  uint64_t nonce_roundtrip_mismatches = nonce_mismatch_count.load();
  const auto oracle_end = std::chrono::steady_clock::now();
  const double oracle_ms = std::chrono::duration<double, std::milli>(oracle_end - oracle_begin).count();

  const bool q0_warmup_exact = q0_mismatch_count.load() == 0u &&
      q0_prime_false_negative_count.load() == 0u;
  std::set<uint64_t> live_verified_factors;
  uint64_t live_complete_drained = 0u;
  auto verify_and_journal_complete = [&](uint64_t factor_value) {
    if (!live_verified_factors.insert(factor_value).second) return;
    const auto exact_begin = std::chrono::steady_clock::now();
    mpz_t exact_candidate, exact_factor, nonce_candidate;
    mpz_inits(exact_candidate, exact_factor, nonce_candidate, nullptr);
    mpz_set_u64(exact_factor, factor_value);
    mpz_mul(exact_factor, exact_factor, primorial);
    mpz_add(exact_candidate, base, exact_factor);
    const auto nonce = encode_nonce_v1(
        static_cast<uint16_t>(options.primorial_number), factor_value,
        options.primorial_offset,
        static_cast<uint16_t>(options.difficulty_offset));
    reconstruct_candidate_from_nonce(nonce_candidate, target, all_primes, nonce);
    if (mpz_cmp(exact_candidate, nonce_candidate) != 0)
      nonce_mismatch_count.fetch_add(1u);
    const uint32_t exact_prime_count =
        share_prime_count(exact_candidate, offsets, 32u, nullptr);
#ifdef RIECOIN_RLOW_SCIENCE_Q9
    const auto strict_science = strict_science_result(
        exact_candidate, options.pattern, offsets, exact_prime_count);
    account_strict_science(strict_science);
#else
    if (exact_prime_count == 7u) ++rlow_exact_sevens;
#endif
    if (exact_prime_count < 5u) {
      ++verifier_rejected_tuples;
    } else {
      ++accepted;
      accepted_count.store(accepted);
      std::ostringstream result_record;
      result_record
          << "{\"schema\":\"riecoin-exact-result-v1\""
          << ",\"event\":\"exact_constellation\""
          << ",\"coin\":\"riecoin\""
          << ",\"work_id\":\"" << json_escape(options.work_id) << "\""
          << ",\"template_id\":\"" << json_escape(options.template_id) << "\""
          << ",\"gpu_uuid\":\"" << json_escape(actual_uuid) << "\""
          << ",\"gpu_identity_sha256\":\""
          << json_escape(options.gpu_identity_sha256) << "\""
          << ",\"runtime_sha256\":\"" << json_escape(options.runtime_sha256) << "\""
          << ",\"factor\":\"" << factor_value << "\""
          << ",\"nonce_v1_uint256_hex\":\"" << nonce_uint256_hex(nonce) << "\""
          << ",\"candidate_hex\":\"" << mpz_hex(exact_candidate) << "\""
          << ",\"pattern\":" << options.pattern
          << ",\"offsets\":" << offsets_json
          << ",\"primorial_number\":" << options.primorial_number
          << ",\"primorial_hex\":\"" << primorial_hex_canonical << "\""
          << ",\"primorial_offset\":\"" << options.primorial_offset << "\""
          << ",\"target_hex\":\"" << target_hex_canonical << "\""
          << ",\"prime_count\":" << exact_prime_count;
#ifdef RIECOIN_RLOW_SCIENCE_Q9
      result_record << riecoin_strict_science_q9::json_fields(strict_science);
#endif
      result_record
          << ",\"proof_state\":\"riecoin_v1_min5_exact_32_rounds\""
          << ",\"terminal_status\":\"proof_complete\""
          << ",\"submission_status\":\"not_attempted\""
          << ",\"accepted\":false,\"rejected\":false,\"stale\":false}";
      if (append_jsonl_durable(options.results_jsonl, result_record.str()))
        result_journaled_count.fetch_add(1u);
      else
        result_journal_error_count.fetch_add(1u);
    }
    mpz_clears(exact_candidate, exact_factor, nonce_candidate, nullptr);
    exact_verify_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - exact_begin).count();
  };
#ifdef RIECOIN_RLOW_ASYNC_EXACT_VERIFY
  constexpr size_t kAsyncExactQueueCapacity = 256u;
  std::mutex async_exact_mutex;
  std::condition_variable async_exact_wake;
  std::condition_variable async_exact_space;
  std::deque<uint64_t> async_exact_queue;
  bool async_exact_stop = false;
  std::exception_ptr async_exact_failure;
  uint64_t async_exact_enqueued = 0u;
  uint64_t async_exact_completed = 0u;
  uint64_t async_exact_max_pending = 0u;
  double async_exact_enqueue_wait_ms = 0.0;
  double async_exact_drain_wait_ms = 0.0;
  double async_exact_worker_ms = 0.0;
  std::thread async_exact_worker;
  struct AsyncExactThreadGuard {
    std::thread* worker;
    std::mutex* mutex;
    std::condition_variable* wake;
    bool* stop;
    ~AsyncExactThreadGuard() {
      if (worker == nullptr || !worker->joinable()) return;
      {
        const std::lock_guard<std::mutex> lock(*mutex);
        *stop = true;
      }
      wake->notify_all();
      worker->join();
    }
  } async_exact_guard{&async_exact_worker, &async_exact_mutex,
                      &async_exact_wake, &async_exact_stop};
  const auto start_async_exact_worker = [&]() {
    if (!production_mode || async_exact_worker.joinable()) return;
    async_exact_worker = std::thread([&]() {
      for (;;) {
        uint64_t factor_value = 0u;
        {
          std::unique_lock<std::mutex> lock(async_exact_mutex);
          async_exact_wake.wait(lock, [&]() {
            return async_exact_stop || !async_exact_queue.empty();
          });
          if (async_exact_queue.empty()) {
            if (async_exact_stop) return;
            continue;
          }
          factor_value = async_exact_queue.front();
          async_exact_queue.pop_front();
        }
        async_exact_space.notify_one();
        try {
          const auto worker_begin = std::chrono::steady_clock::now();
          verify_and_journal_complete(factor_value);
          async_exact_worker_ms += std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - worker_begin).count();
          ++async_exact_completed;
        } catch (...) {
          {
            const std::lock_guard<std::mutex> lock(async_exact_mutex);
            async_exact_failure = std::current_exception();
            async_exact_stop = true;
          }
          async_exact_wake.notify_all();
          async_exact_space.notify_all();
          return;
        }
      }
    });
  };
  const auto enqueue_async_exact = [&](uint64_t factor_value) {
    const auto wait_begin = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(async_exact_mutex);
    async_exact_space.wait(lock, [&]() {
      return async_exact_queue.size() < kAsyncExactQueueCapacity ||
             async_exact_failure != nullptr;
    });
    async_exact_enqueue_wait_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wait_begin).count();
    if (async_exact_failure != nullptr)
      std::rethrow_exception(async_exact_failure);
    async_exact_queue.push_back(factor_value);
    ++async_exact_enqueued;
    async_exact_max_pending = std::max<uint64_t>(
        async_exact_max_pending, async_exact_queue.size());
    lock.unlock();
    async_exact_wake.notify_one();
  };
  const auto finish_async_exact_worker = [&]() {
    if (!async_exact_worker.joinable()) return;
    const auto drain_begin = std::chrono::steady_clock::now();
    {
      const std::lock_guard<std::mutex> lock(async_exact_mutex);
      async_exact_stop = true;
    }
    async_exact_wake.notify_all();
    async_exact_worker.join();
    async_exact_drain_wait_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - drain_begin).count();
    if (async_exact_failure != nullptr)
      std::rethrow_exception(async_exact_failure);
    if (async_exact_completed != async_exact_enqueued ||
        !async_exact_queue.empty())
      throw std::runtime_error("async exact verifier conservation failure");
  };
#endif
  if (device_pipeline_mode) {
    if (mismatch_words != 0u || nonce_roundtrip_mismatches != 0u ||
        !q0_warmup_exact)
      throw std::runtime_error("Q0 warmup exactness gate failed; measured run forbidden");

    if (options.candidates > (1u << 24u))
      throw std::runtime_error("Q0 device-resident prototype capacity exceeds safe launch bound");
    const uint32_t compact_threads = static_cast<uint32_t>(word_count * 32u);
    const uint32_t compact_blocks = (compact_threads + 127u) / 128u;
    // Live batches advance to different absolute factor windows. Their sieve
    // survivor counts are not constant, so capacity must be a proof-safe bound,
    // not the warmup popcount. The counted kernel gates inactive instances.
    const uint32_t q0_capacity_bound = kRlowPrpCapacity;
    const uint32_t prp_blocks = (q0_capacity_bound * 8u + 127u) / 128u;
#ifdef RIECOIN_RLOW_WIDE_TPI4
#ifdef RIECOIN_RLOW_PACKED_DISPATCH
    const uint32_t main_prp_blocks = std::min((q0_capacity_bound*4u+127u)/128u,
        static_cast<uint32_t>(prop.multiProcessorCount)*6u);
#else
    const uint32_t main_prp_blocks = (q0_capacity_bound * 4u + 127u) / 128u;
#endif
#else
    const uint32_t main_prp_blocks = prp_blocks;
#endif
    const uint32_t pass_blocks = (q0_capacity_bound + 255u) / 256u;

    // Cold exactness gate for the new device-side representation. This is the
    // only path that copies candidates back; it is outside the measured window.
    const auto cold_pipeline_begin = std::chrono::steady_clock::now();
    cuda_check(cudaMemset(d_q0_count, 0, sizeof(uint32_t)), "Q0 cold count reset");
    cuda_check(cudaMemset(d_q0_pass_count, 0, sizeof(uint32_t)), "Q0 cold pass reset");
    cuda_check(cudaMemset(d_q0_overflow_count, 0, sizeof(uint32_t)), "Q0 cold overflow reset");
    cuda_check(cudaMemset(d_q0_total_count, 0, sizeof(unsigned long long)), "Q0 cold total reset");
    cuda_check(cudaMemcpy(d_words, d_q0_initial_words, word_count * sizeof(uint32_t),
                          cudaMemcpyDeviceToDevice), "Q0 cold bitmap reset");
    rlow_sieve_launch(
        d_words, options.candidates, d_primes, d_first, pair_count,
        options.factor_origin);
    compact_construct_q0_candidates<<<compact_blocks, 128u>>>(
        d_words, static_cast<uint32_t>(word_count), options.candidates,
        d_q0_base, d_q0_primorial, d_q0_candidates, d_q0_factors,
        options.factor_origin, q0_capacity_bound,
        d_q0_count, d_q0_total_count, d_q0_overflow_count);
    riecoin_cuda::euler_jacobi_dynamic_1280_tpi8_counted_kernel
        <<<prp_blocks, riecoin_cuda::kPrpBlockThreads>>>(
            d_q0_candidates, d_q0_verdicts, d_q0_count, q0_capacity_bound);
    compact_q0_passes<<<pass_blocks, 256u>>>(
        d_q0_verdicts, d_q0_factors, d_q0_count, q0_capacity_bound,
        d_q0_pass_factors, d_q0_pass_count);
    cuda_check(cudaGetLastError(), "Q0 cold device pipeline launch");
    cuda_check(cudaDeviceSynchronize(), "Q0 cold device pipeline sync");
    const double cold_pipeline_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - cold_pipeline_begin).count();

    uint32_t cold_count = 0u, cold_pass_count = 0u, cold_overflow = 0u;
    cuda_check(cudaMemcpy(&cold_count, d_q0_count, sizeof(cold_count),
                          cudaMemcpyDeviceToHost), "Q0 cold count D2H");
    cuda_check(cudaMemcpy(&cold_pass_count, d_q0_pass_count, sizeof(cold_pass_count),
                          cudaMemcpyDeviceToHost), "Q0 cold pass count D2H");
    cuda_check(cudaMemcpy(&cold_overflow, d_q0_overflow_count, sizeof(cold_overflow),
                          cudaMemcpyDeviceToHost), "Q0 cold overflow D2H");
    if (cold_count != survivors.size() || cold_pass_count > cold_count || cold_overflow != 0u)
      throw std::runtime_error("Q0 cold device conservation failed");
    std::vector<uint64_t> cold_factors(cold_count), cold_pass_factors(cold_pass_count);
    std::vector<cgbn_mem_t<1280u>> cold_candidates(cold_count);
    cuda_check(cudaMemcpy(cold_factors.data(), d_q0_factors,
                          cold_factors.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost),
               "Q0 cold factors D2H");
    cuda_check(cudaMemcpy(cold_candidates.data(), d_q0_candidates,
                          cold_candidates.size() * sizeof(cold_candidates[0]),
                          cudaMemcpyDeviceToHost), "Q0 cold candidates D2H");
    cuda_check(cudaMemcpy(cold_pass_factors.data(), d_q0_pass_factors,
                          cold_pass_factors.size() * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost), "Q0 cold pass factors D2H");
    mpz_t cold_candidate, cold_factor;
    mpz_inits(cold_candidate, cold_factor, nullptr);
    cgbn_mem_t<1280u> cold_expected{};
    for (size_t index = 0u; index < cold_factors.size(); ++index) {
      mpz_set_u64(cold_factor, cold_factors[index]);
      mpz_mul(cold_factor, cold_factor, primorial);
      mpz_add(cold_candidate, base, cold_factor);
      pack_candidate_1280(cold_expected, cold_candidate);
      q0_replay_mismatch_words += std::memcmp(
          &cold_expected, &cold_candidates[index], sizeof(cold_expected)) != 0;
    }
    mpz_clears(cold_candidate, cold_factor, nullptr);
    std::vector<uint64_t> expected_pass_factors;
    for (size_t index = 0u; index < survivors.size(); ++index)
      if (gpu_q0_verdicts[index] != 0u)
        expected_pass_factors.push_back(options.factor_origin + survivors[index]);
    std::sort(cold_pass_factors.begin(), cold_pass_factors.end());
    std::sort(expected_pass_factors.begin(), expected_pass_factors.end());
    if (cold_pass_factors != expected_pass_factors)
      ++q0_replay_verdict_mismatches;
    if (q0_replay_mismatch_words != 0u || q0_replay_verdict_mismatches != 0u)
      throw std::runtime_error("Q0 device candidate/parity gate failed");

    if (options.network_batch) {
      for (const std::string& record : pending_result_records) {
        if (append_jsonl_durable(options.results_jsonl, record))
          result_journaled_count.fetch_add(1u);
        else
          result_journal_error_count.fetch_add(1u);
      }
      result_journaled = result_journaled_count.load();
      result_journal_errors = result_journal_error_count.load();
      q0_total_entries = cold_count;
      q0_total_passes = cold_pass_count;
      q0_total_positions = options.candidates;
      q0_measured_loops = 1u;
      q0_measured_ms = cold_pipeline_ms;
      q0_prp_gpu_ms = q0_gpu_ms;
      q0_drops = cold_overflow;
      q0_duplicates = 0u;
      std::cout << "phase=network_batch_q0_complete measured_ms="
                << q0_measured_ms << " q0_entries=" << q0_total_entries
                << " q0_passes=" << cold_pass_count
                << " factor_origin_begin=" << options.factor_origin
                << " factor_origin_end="
                << (options.factor_origin + options.candidates) << std::endl;
    } else {
    cuda_check(cudaMemset(d_q0_total_count, 0, sizeof(unsigned long long)),
               "Q0 measured total reset");
    cuda_check(cudaMemset(d_q0_overflow_count, 0, sizeof(uint32_t)),
               "Q0 measured overflow reset");
    if (production_mode) {
      cuda_check(cudaMemset(d_tuple_member_pass_counts, 0,
                            6u * sizeof(unsigned long long)),
                 "production member pass counts reset");
      cuda_check(cudaMemset(d_tuple_active_counts, 0,
                            6u * sizeof(unsigned long long)),
                 "production active counts reset");
      cuda_check(cudaMemset(d_rejected_counts, 0,
                            7u * sizeof(unsigned long long)),
                 "production rejected counts reset");
      cuda_check(cudaMemset(d_complete_tuple_count, 0,
                            sizeof(unsigned long long)),
                 "production complete tuple count reset");
      cuda_check(cudaMemset(d_tuple_overflow_count, 0, sizeof(uint32_t)),
                 "production tuple overflow reset");
    }
    const uint64_t rlow_bundle_positions =
        static_cast<uint64_t>(rlow_queue::kMacros) * options.candidates;
    if (rlow_bundle_positions > UINT32_MAX || !production_mode)
      throw std::runtime_error("RLOW queued path requires bounded production mode");
#ifdef RIECOIN_RLOW_SUPER_SIEVE
    const uint64_t rlow_super_positions =
        rlow_super_sieve::super_span(options.candidates);
    if ((options.factor_max - options.factor_origin) % rlow_super_positions != 0u)
      throw std::runtime_error(
          "RLOW super sieve rejects ragged factor windows before any Q0 claim");
#endif
    std::array<cudaEvent_t, 7> q0_events{};
    for (auto& event : q0_events)
      cuda_check(cudaEventCreate(&event), "Q0 phase event create");
    rlow_queue::setup(static_cast<uint32_t>(word_count));
#ifdef RIECOIN_RLOW_BATCH_PRODUCER
    rlow_batch_producer::setup(static_cast<uint32_t>(word_count));
    rlow_batch_producer::r259_configure(options.factor_max);
#endif
#ifdef RIECOIN_RLOW_SUPER_SIEVE
    rlow_super_sieve::setup(static_cast<uint32_t>(word_count));
#endif
    const std::array<unsigned long long, 6> rlow_zero_prefix{};
    cuda_check(cudaMemcpyToSymbol(rlow_strict_prefix_device, rlow_zero_prefix.data(),
                                 sizeof(rlow_zero_prefix)), "RLOW strict prefix reset");
#ifdef RIECOIN_RLOW_Q1_TAIL_QUEUE
    rlow_tail_host::setup(options, q0_capacity_bound, d_q0_base, d_q0_primorial,
        d_tuple_offsets, d_tuple_members, d_tuple_verdicts, d_tuple_member_pass_counts,
        d_tuple_active_counts, d_rejected_counts, d_rejected_sample_factors,
        d_complete_tuple_factors, d_complete_tuple_count, d_tuple_overflow_count);
#endif
#ifdef RIECOIN_RLOW_ASYNC_EXACT_VERIFY
    start_async_exact_worker();
#endif
#ifdef RIECOIN_RLOW_RESULT_SIDECAR
    riecoin_result_sidecar::Collector result_sidecar(
        {d_complete_tuple_count, d_q0_total_count, d_tuple_active_counts,
         rlow_tail_host::counters.tested, rlow_tail_host::counters.observed_pass,
         d_complete_tuple_factors, q0_capacity_bound},
        enqueue_async_exact, result_journaled_count);
#endif
    const auto measured_begin = std::chrono::steady_clock::now();
#ifdef RIECOIN_RLOW_CUDA_TAIL_RAIL
    riecoin_tail_rail::Rail tail_rail(q0_capacity_bound, d_q0_total_count,
                                     result_sidecar, measured_begin);
#endif
    auto last_progress = measured_begin;
#ifdef RIECOIN_RLOW_PIPELINE_TRACE
    struct PipelineTraceRecord {
      uint64_t sequence = 0u, host_begin_us = 0u, host_elapsed_us = 0u;
      uint32_t kind = 0u;
      uint64_t value0 = 0u, value1 = 0u;
    };
    constexpr size_t kPipelineTraceLimit = 32768u;
    std::vector<PipelineTraceRecord> pipeline_trace_records;
    pipeline_trace_records.reserve(kPipelineTraceLimit);
    uint64_t pipeline_trace_sequence = 0u;
    bool pipeline_trace_overflow = false;
    const auto pipeline_trace_now_us = []() -> uint64_t {
      return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count());
    };
    const uint64_t pipeline_trace_steady_anchor_us = pipeline_trace_now_us();
    const uint64_t pipeline_trace_wall_anchor_unix_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    const auto pipeline_trace_record = [&](uint32_t kind, uint64_t host_begin_us,
                                            uint64_t value0 = 0u,
                                            uint64_t value1 = 0u) {
      if (pipeline_trace_records.size() == kPipelineTraceLimit) {
        pipeline_trace_overflow = true;
        return;
      }
      pipeline_trace_records.push_back(PipelineTraceRecord{
          ++pipeline_trace_sequence, host_begin_us,
          pipeline_trace_now_us() - host_begin_us, kind, value0, value1});
    };
    uint64_t pipeline_trace_post_sync_begin_us = 0u;
#endif
    uint64_t measured_factor_origin = options.factor_origin;
    const uint32_t measurement_seconds = production_mode
        ? options.production_session_seconds : options.q0_duration_seconds;
    bool resident_epoch_stop_requested = false;
    do {
#ifdef RIECOIN_RLOW_PIPELINE_TRACE
      const uint64_t pipeline_trace_loop_entry_us = pipeline_trace_now_us();
      if (pipeline_trace_post_sync_begin_us != 0u)
        pipeline_trace_record(3u, pipeline_trace_post_sync_begin_us);
      const uint64_t pipeline_trace_pre_gpu_begin_us = pipeline_trace_loop_entry_us;
#endif
      const auto reservation_next = riecoin_production_window::next(
          options.factor_origin, measured_factor_origin,
          options.factor_max, rlow_bundle_positions);
      if (reservation_next == riecoin_production_window::Next::invalid)
        throw std::runtime_error("RLOW queue factor reservation invalid or partial");
      if (reservation_next == riecoin_production_window::Next::drain) {
        q0_reservation_exhausted = true;
        break; // continue through finish(), exact replay and terminal conservation
      }
#ifdef RIECOIN_RLOW_Q1_TAIL_QUEUE
#ifndef RIECOIN_RLOW_CUDA_TAIL_RAIL
      rlow_tail_host::flush(false);
#endif
#endif
#ifdef RIECOIN_RLOW_PIPELINE_TRACE
      pipeline_trace_record(1u, pipeline_trace_pre_gpu_begin_us);
      const uint64_t pipeline_trace_q0_gpu_begin_us = pipeline_trace_now_us();
#endif
#ifdef RIECOIN_RLOW_ASYNC_FRONTIER
      auto& q0_events=tail_rail.acquire();
#endif
      cuda_check(cudaEventRecord(q0_events[0]), "Q0 phase record begin");
      cuda_check(cudaMemcpy(d_words, d_q0_initial_words,
                            word_count * sizeof(uint32_t), cudaMemcpyDeviceToDevice),
                 "Q0 measured bitmap reset");
      cuda_check(cudaMemset(d_q0_count, 0, sizeof(uint32_t)), "Q0 measured count reset");
      cuda_check(cudaMemset(d_q0_pass_count, 0, sizeof(uint32_t)), "Q0 measured pass reset");
      cuda_check(cudaEventRecord(q0_events[1]), "Q0 phase record reset");
#ifdef RIECOIN_RLOW_BATCH_PRODUCER
#ifdef RIECOIN_RLOW_OVERLAP_PIPELINE
      cuda_check(cudaStreamWaitEvent(rlow_batch_producer::producer(), q0_events[1], 0u),
                 "RLOW overlap producer reset dependency");
      rlow_batch_producer::enqueue_fill(
          d_words, static_cast<uint32_t>(word_count), options.candidates,
          d_primes, d_first, pair_count, measured_factor_origin,
          d_q0_base, d_q0_primorial, d_q0_candidates, d_q0_factors,
          d_q0_count, d_q0_total_count);
      for (uint32_t group = 0u;
           group < rlow_batch_producer::kOverlapGroups; ++group) {
        const uint32_t macro_base =
            group * rlow_batch_producer::kMacrosPerOverlapGroup;
        const uint32_t slot_base = macro_base * rlow_queue::kMacroCapacity;
        const uint32_t group_capacity =
            rlow_batch_producer::kMacrosPerOverlapGroup *
            rlow_queue::kMacroCapacity;
        rlow_batch_producer::begin_consumer_group(group);
#ifdef RIECOIN_RLOW_SPARSE_FAMILY_Q0
        riecoin_sparse_scalar_q0::segmented<<<
            (group_capacity + 127u) / 128u, 128u, 0,
            rlow_batch_producer::consumer()>>>(
            d_q0_candidates + slot_base, d_q0_verdicts + slot_base,
            rlow_queue::macro_counts + macro_base,
            rlow_batch_producer::kMacrosPerOverlapGroup,
            rlow_queue::kMacroCapacity);
#else
#ifdef RIECOIN_RLOW_CARRYLATE_Q0
        riecoin_rlow_bridge::rlow_segmented_q0_carrylate
#elif defined(RIECOIN_RLOW_LAZY_BASE2_Q0)
#ifdef RIECOIN_RLOW_NARROW_LAZY_Q0
        if (narrow_q0) {
          riecoin_narrow_lazy_q0::segmented<<<
              (group_capacity * 4u + 127u) / 128u, 128u, 0,
              rlow_batch_producer::consumer()>>>(
              d_q0_candidates + slot_base, d_q0_verdicts + slot_base,
              rlow_queue::macro_counts + macro_base,
              rlow_batch_producer::kMacrosPerOverlapGroup,
              rlow_queue::kMacroCapacity);
        } else
#endif
        riecoin_lazy_base2_window::segmented_q0_kernel
#else
        riecoin_rlow_bridge::rlow_segmented_q0_prp
#endif
            <<<((group_capacity *
#ifdef RIECOIN_RLOW_WIDE_TPI4
                   4u +
#else
                   8u +
#endif
                   riecoin_cuda::kPrpBlockThreads - 1u) /
                  riecoin_cuda::kPrpBlockThreads),
                riecoin_cuda::kPrpBlockThreads, 0,
                rlow_batch_producer::consumer()>>>(
                d_q0_candidates + slot_base, d_q0_verdicts + slot_base,
                rlow_queue::macro_counts + macro_base,
                rlow_batch_producer::kMacrosPerOverlapGroup,
                rlow_queue::kMacroCapacity);
#endif
        cuda_check(cudaGetLastError(), "RLOW overlap group PRP launch");
        rlow_batch_producer::end_consumer_group(group);
      }
      rlow_batch_producer::publish_consumer_done();
      rlow_batch_producer::finish_fill(
          d_words, static_cast<uint32_t>(word_count), options.candidates,
          d_primes, d_first, pair_count, measured_factor_origin);
      q0_prp_gpu_ms += rlow_batch_producer::finish_consumer_and_bundle();
      cuda_check(cudaEventRecord(q0_events[2]),
                 "RLOW overlap producer/consumer complete");
      cuda_check(cudaEventRecord(q0_events[3]),
                 "RLOW overlap queue compact already fused");
      cuda_check(cudaEventRecord(q0_events[4]),
                 "RLOW overlap PRP already complete");
#else
#ifdef RIECOIN_RLOW_SUPER_SIEVE
      rlow_super_sieve::fill_next_slice(
          d_words, static_cast<uint32_t>(word_count), options.candidates,
          d_primes, d_first, pair_count, measured_factor_origin,
          d_q0_base, d_q0_primorial, d_q0_candidates, d_q0_factors,
          d_q0_count, d_q0_total_count);
#else
      rlow_batch_producer::fill(d_words, static_cast<uint32_t>(word_count),
          options.candidates, d_primes, d_first, pair_count, measured_factor_origin,
          d_q0_base, d_q0_primorial, d_q0_candidates, d_q0_factors,
          d_q0_count, d_q0_total_count);
#endif
#endif
#else
      rlow_queue::fill(d_words, static_cast<uint32_t>(word_count),
          options.candidates, d_primes, d_first, pair_count, measured_factor_origin,
          d_q0_base, d_q0_primorial, d_q0_candidates, d_q0_factors,
          d_q0_count, d_q0_total_count);
#endif
#ifndef RIECOIN_RLOW_OVERLAP_PIPELINE
      cuda_check(cudaEventRecord(q0_events[2]), "RLOW private sieve/scan/pack complete");
      cuda_check(cudaEventRecord(q0_events[3]), "RLOW queue compact already fused");
#ifdef ASTRA_R249_DENSE_Q0
      if(r283_active||!narrow_q0){
        riecoin_lazy_base2_window::prepare_dense_q0<<<rlow_queue::kMacros,128u>>>(
            d_q0_verdicts,rlow_queue::macro_counts,rlow_queue::kMacros,rlow_queue::kMacroCapacity);
        cuda_check(cudaGetLastError(),"R249 exact dense worklist");
      }
#endif
#ifdef RIECOIN_RLOW_SPARSE_FAMILY_Q0
      riecoin_sparse_scalar_q0::segmented<<<(q0_capacity_bound+127u)/128u,128u>>>(
          d_q0_candidates,d_q0_verdicts,rlow_queue::macro_counts,
          rlow_queue::kMacros,rlow_queue::kMacroCapacity);
#else
      if(r283_active){
        cuda_check(r294_dense_dispatch(d_q0_candidates,d_q0_verdicts,
          r283_slots,r283_total,q0_capacity_bound,cudaStreamPerThread),"R283 dense SoS Q0");
      }else{
#ifdef RIECOIN_RLOW_CARRYLATE_Q0
      riecoin_rlow_bridge::rlow_segmented_q0_carrylate
#elif defined(RIECOIN_RLOW_LAZY_BASE2_Q0)
#ifdef RIECOIN_RLOW_NARROW_LAZY_Q0
      if(narrow_q0) {
        riecoin_narrow_lazy_q0::segmented<<<(q0_capacity_bound*4u+127u)/128u,128u>>>(
            d_q0_candidates,d_q0_verdicts,rlow_queue::macro_counts,
            rlow_queue::kMacros,rlow_queue::kMacroCapacity);
      } else
#endif
      riecoin_lazy_base2_window::segmented_q0_kernel
#else
      riecoin_rlow_bridge::rlow_segmented_q0_prp
#endif
          <<<main_prp_blocks, riecoin_cuda::kPrpBlockThreads>>>(
              d_q0_candidates, d_q0_verdicts, rlow_queue::macro_counts,
              rlow_queue::kMacros, rlow_queue::kMacroCapacity);
      }
#endif
      cuda_check(cudaGetLastError(), "RLOW filled queue PRP launch");
      cuda_check(cudaEventRecord(q0_events[4]), "Q0 phase record PRP");
#ifdef RIECOIN_RLOW_BATCH_PRODUCER
      rlow_batch_producer::r260_submit_after_prp();
#endif
#endif
      compact_q0_passes<<<pass_blocks, 256u>>>(
          d_q0_verdicts, d_q0_factors, rlow_queue::physical_slots, q0_capacity_bound,
          d_q0_pass_factors, d_q0_pass_count);
      cuda_check(cudaGetLastError(), "RLOW segmented pass compact launch");
      if (production_mode) {
        const uint32_t sample_blocks = (q0_capacity_bound + 255u) / 256u;
        rlow_queue::sample_rejected_segments<<<sample_blocks, 256u>>>(
            d_q0_verdicts, d_q0_factors, rlow_queue::macro_counts,
            options.rejected_audit_per_stage, d_rejected_sample_factors,
            d_rejected_counts);
        cuda_check(cudaGetLastError(), "Q0 rejected audit sample launch");
      }
      cuda_check(cudaEventRecord(q0_events[5]), "Q0 phase record pass compact");
#ifndef RIECOIN_RLOW_ASYNC_FRONTIER
      cuda_check(cudaEventSynchronize(q0_events[5]), "Q0 pass compact sync");
#endif
#ifdef RIECOIN_RLOW_PIPELINE_TRACE
      pipeline_trace_record(2u, pipeline_trace_q0_gpu_begin_us);
      const uint64_t pipeline_trace_receipt_tail_begin_us = pipeline_trace_now_us();
#endif
      uint32_t batch_q0_passes = 0u;
#ifdef RIECOIN_RLOW_ASYNC_FRONTIER
      if(!rlow_queue::cold_checked) {
#endif
      cuda_check(cudaMemcpy(&batch_q0_passes, d_q0_pass_count,
                            sizeof(batch_q0_passes), cudaMemcpyDeviceToHost),
                 "Q0 batch pass count D2H");
      if (batch_q0_passes > q0_capacity_bound)
        throw std::runtime_error("Q0 pass count exceeds bounded capacity");
      rlow_queue::check_first_queue(d_q0_candidates, d_q0_factors, d_q0_verdicts,
          d_q0_count, base, primorial, measured_factor_origin, options.candidates);
#ifdef RIECOIN_RLOW_ASYNC_FRONTIER
      }
#endif
      if (production_mode) {
#ifndef RIECOIN_RLOW_ASYNC_FRONTIER
        q0_total_passes += batch_q0_passes;
#endif
#ifdef RIECOIN_RLOW_Q1_TAIL_QUEUE
#ifndef RIECOIN_RLOW_CUDA_TAIL_RAIL
        rlow_tail_host::append(d_q0_pass_factors, batch_q0_passes,
            measured_factor_origin, measured_factor_origin + rlow_bundle_positions);
#endif
#else
        if (batch_q0_passes != 0u) {
          const uint32_t tuple_members = batch_q0_passes * 6u;
          construct_tuple_members<<<(tuple_members + 255u) / 256u, 256u>>>(
              d_q0_pass_factors, batch_q0_passes, d_q0_base,
              d_q0_primorial, d_tuple_offsets, measured_factor_origin,
              d_tuple_members, d_tuple_overflow_count);
          cuda_check(cudaGetLastError(), "production tuple construct launch");
          const uint32_t tuple_prp_blocks =
              (tuple_members * 8u + riecoin_cuda::kPrpBlockThreads - 1u) /
              riecoin_cuda::kPrpBlockThreads;
          riecoin_cuda::euler_jacobi_dynamic_1280_tpi8_kernel
              <<<tuple_prp_blocks, riecoin_cuda::kPrpBlockThreads>>>(
                  d_tuple_members, d_tuple_verdicts, tuple_members);
          cuda_check(cudaGetLastError(), "production tuple PRP launch");
          reduce_tuple_candidates<<<(batch_q0_passes + 255u) / 256u, 256u>>>(
              d_tuple_verdicts, d_q0_pass_factors, batch_q0_passes,
              options.rejected_audit_per_stage,
              d_tuple_member_pass_counts, d_tuple_active_counts,
              d_rejected_counts, d_rejected_sample_factors,
              d_complete_tuple_factors, q0_capacity_bound,
              d_complete_tuple_count, d_tuple_overflow_count);
          cuda_check(cudaGetLastError(), "production tuple reduction launch");
        }
#endif
      }
      advance_q0_base<<<1u, 1u>>>(d_q0_base, d_q0_primorial,
          static_cast<uint32_t>(rlow_bundle_positions), d_q0_overflow_count);
      cuda_check(cudaGetLastError(), "Q0 measured base advance launch");
      cuda_check(cudaEventRecord(q0_events[6]), "Q0 phase record complete");
#ifndef RIECOIN_RLOW_ASYNC_FRONTIER
      cuda_check(cudaEventSynchronize(q0_events[6]), "Q0 measured pipeline sync");
#endif
#ifdef RIECOIN_RLOW_PIPELINE_TRACE
      pipeline_trace_record(4u, pipeline_trace_receipt_tail_begin_us,
                            batch_q0_passes, q0_total_passes);
      pipeline_trace_post_sync_begin_us = pipeline_trace_now_us();
#endif
      resident_epoch_stop_requested =
          RIECOIN_RLOW_RESIDENT_BUNDLE_STOP_REQUESTED();
      q0_resident_cancelled = resident_epoch_stop_requested;
#ifndef RIECOIN_RLOW_ASYNC_FRONTIER
      float stage_ms = 0.0f;
      cuda_check(cudaEventElapsedTime(&stage_ms, q0_events[0], q0_events[1]), "Q0 reset elapsed");
      q0_reset_ms += stage_ms;
#ifdef RIECOIN_RLOW_PIPELINE_TRACE
      pipeline_trace_record(10u, pipeline_trace_now_us(),
          static_cast<uint64_t>(stage_ms * 1000.0f), q0_measured_loops);
#endif
#ifdef RIECOIN_RLOW_OVERLAP_PIPELINE
      q0_sieve_ms += rlow_batch_producer::overlap_last_remainder_gpu_ms +
                     rlow_batch_producer::overlap_last_sieve_gpu_ms;
      q0_compact_pack_ms += rlow_batch_producer::overlap_last_scan_pack_gpu_ms +
                            rlow_batch_producer::overlap_last_metadata_gpu_ms;
#else
      cuda_check(cudaEventElapsedTime(&stage_ms, q0_events[1], q0_events[2]), "Q0 sieve elapsed");
      q0_sieve_ms += stage_ms;
#ifdef RIECOIN_RLOW_PIPELINE_TRACE
      pipeline_trace_record(11u, pipeline_trace_now_us(),
          static_cast<uint64_t>(stage_ms * 1000.0f), q0_measured_loops);
#endif
      cuda_check(cudaEventElapsedTime(&stage_ms, q0_events[2], q0_events[3]), "Q0 compact elapsed");
      q0_compact_pack_ms += stage_ms;
#ifdef RIECOIN_RLOW_PIPELINE_TRACE
      pipeline_trace_record(12u, pipeline_trace_now_us(),
          static_cast<uint64_t>(stage_ms * 1000.0f), q0_measured_loops);
#endif
      cuda_check(cudaEventElapsedTime(&stage_ms, q0_events[3], q0_events[4]), "Q0 PRP elapsed");
      q0_prp_gpu_ms += stage_ms;
#ifdef RIECOIN_RLOW_PIPELINE_TRACE
      pipeline_trace_record(13u, pipeline_trace_now_us(),
          static_cast<uint64_t>(stage_ms * 1000.0f), q0_measured_loops);
#endif
#endif
      cuda_check(cudaEventElapsedTime(&stage_ms, q0_events[4], q0_events[5]), "Q0 pass compact elapsed");
      q0_verdict_d2h_ms += stage_ms;
#ifdef RIECOIN_RLOW_PIPELINE_TRACE
      pipeline_trace_record(14u, pipeline_trace_now_us(),
          static_cast<uint64_t>(stage_ms * 1000.0f), q0_measured_loops);
#endif
      cuda_check(cudaEventElapsedTime(&stage_ms, q0_events[5], q0_events[6]), "production tuple elapsed");
      if (production_mode) tuple_gpu_ms += stage_ms;
#ifdef RIECOIN_RLOW_PIPELINE_TRACE
      pipeline_trace_record(15u, pipeline_trace_now_us(),
          static_cast<uint64_t>(stage_ms * 1000.0f), q0_measured_loops);
#endif
#endif // synchronous event harvesting; async rail owns its event slots
      q0_measured_loops += rlow_queue::kMacros;
      q0_total_positions += rlow_bundle_positions;
      measured_factor_origin += rlow_bundle_positions;
      const auto now = std::chrono::steady_clock::now();
#ifdef RIECOIN_RLOW_CUDA_TAIL_RAIL
#ifdef RIECOIN_RLOW_ASYNC_FRONTIER
      tail_rail.submit_async(d_q0_pass_factors,d_q0_pass_count,
#else
      tail_rail.submit(d_q0_pass_factors, batch_q0_passes,
#endif
          measured_factor_origin - rlow_bundle_positions, measured_factor_origin,
          {std::chrono::duration<double>(now - measured_begin).count(), q0_prp_gpu_ms,
           q0_measured_loops, q0_total_positions, q0_total_passes, options.factor_origin,
           measured_factor_origin, 0u, batch_q0_passes},
          std::chrono::duration<double>(now - last_progress).count() >= 0.25);
#endif
      if (std::chrono::duration<double>(now - last_progress).count() >= 0.25) {
#ifdef RIECOIN_RLOW_RESULT_SIDECAR
#ifndef RIECOIN_RLOW_CUDA_TAIL_RAIL
        result_sidecar.submit({
            std::chrono::duration<double>(now - measured_begin).count(), q0_prp_gpu_ms,
            q0_measured_loops, q0_total_positions, q0_total_passes, options.factor_origin,
            measured_factor_origin, rlow_tail_host::pending_total(), batch_q0_passes});
#endif
#else
        if (production_mode) {
#ifdef RIECOIN_RLOW_PIPELINE_TRACE
          const uint64_t pipeline_trace_complete_count_begin_us =
              pipeline_trace_now_us();
          const uint64_t pipeline_trace_complete_before = live_complete_drained;
#endif
          unsigned long long live_complete_count = 0u;
          cuda_check(cudaMemcpy(&live_complete_count, d_complete_tuple_count,
                                sizeof(live_complete_count), cudaMemcpyDeviceToHost),
                     "production live complete count D2H");
#ifdef RIECOIN_RLOW_PIPELINE_TRACE
          pipeline_trace_record(5u, pipeline_trace_complete_count_begin_us,
                                live_complete_count,
                                live_complete_count >= pipeline_trace_complete_before
                                    ? live_complete_count - pipeline_trace_complete_before
                                    : 0u);
#endif
          if (live_complete_count > q0_capacity_bound)
            throw std::runtime_error("production live complete count overflow");
          if (live_complete_count > live_complete_drained) {
            std::vector<uint64_t> new_complete_factors(
                static_cast<size_t>(live_complete_count - live_complete_drained));
#ifdef RIECOIN_RLOW_PIPELINE_TRACE
            const uint64_t pipeline_trace_complete_factors_begin_us =
                pipeline_trace_now_us();
#endif
            cuda_check(cudaMemcpy(
                new_complete_factors.data(),
                d_complete_tuple_factors + live_complete_drained,
                new_complete_factors.size() * sizeof(uint64_t),
                cudaMemcpyDeviceToHost),
                "production live complete factors D2H");
#ifdef RIECOIN_RLOW_PIPELINE_TRACE
            pipeline_trace_record(6u, pipeline_trace_complete_factors_begin_us,
                                  new_complete_factors.size(),
                                  new_complete_factors.size() * sizeof(uint64_t));
            const uint64_t pipeline_trace_exact_journal_begin_us =
                pipeline_trace_now_us();
#endif
            for (const uint64_t factor_value : new_complete_factors) {
#ifdef RIECOIN_RLOW_ASYNC_EXACT_VERIFY
              enqueue_async_exact(factor_value);
#else
              verify_and_journal_complete(factor_value);
#endif
            }
#ifdef RIECOIN_RLOW_PIPELINE_TRACE
            pipeline_trace_record(7u, pipeline_trace_exact_journal_begin_us,
                                  new_complete_factors.size(),
                                  result_journaled_count.load());
#endif
            live_complete_drained = live_complete_count;
          }
        }
        const double elapsed_s = std::chrono::duration<double>(now - measured_begin).count();
#ifdef RIECOIN_RLOW_PIPELINE_TRACE
        const uint64_t pipeline_trace_telemetry_begin_us = pipeline_trace_now_us();
#endif
        unsigned long long live_q0_entries = 0u;
        uint32_t live_passes = batch_q0_passes;
        cuda_check(cudaMemcpy(&live_q0_entries, d_q0_total_count,
                              sizeof(live_q0_entries), cudaMemcpyDeviceToHost),
                   "Q0 live total telemetry D2H");
        const double live_candidates_s = elapsed_s > 0.0
            ? static_cast<double>(live_q0_entries) / elapsed_s : 0.0;
        const double live_positions_s = elapsed_s > 0.0
            ? static_cast<double>(q0_total_positions) / elapsed_s : 0.0;
        const double live_prp_s = q0_prp_gpu_ms > 0.0
            ? 1000.0 * static_cast<double>(live_q0_entries) / q0_prp_gpu_ms : 0.0;
        std::array<unsigned long long, 6> live_active_counts{};
        if (production_mode) {
          cuda_check(cudaMemcpy(live_active_counts.data(), d_tuple_active_counts,
                                sizeof(live_active_counts), cudaMemcpyDeviceToHost),
                     "production live tuple telemetry D2H");
        }
#ifdef RIECOIN_RLOW_Q1_TAIL_QUEUE
        std::array<unsigned long long, 6> live_member_tested{}, live_member_passed{};
        cuda_check(cudaMemcpy(live_member_tested.data(), rlow_tail_host::counters.tested,
                              sizeof(live_member_tested), cudaMemcpyDeviceToHost),
                   "R12 live physical test telemetry D2H");
        cuda_check(cudaMemcpy(live_member_passed.data(), rlow_tail_host::counters.observed_pass,
                              sizeof(live_member_passed), cudaMemcpyDeviceToHost),
                   "R12 live observed pass telemetry D2H");
#endif
#ifdef RIECOIN_RLOW_PIPELINE_TRACE
        pipeline_trace_record(8u, pipeline_trace_telemetry_begin_us,
                              live_q0_entries, q0_total_passes);
        const uint64_t pipeline_trace_stdout_begin_us = pipeline_trace_now_us();
#endif
        std::cout << "phase=q0_measure_progress elapsed_s=" << elapsed_s
                  << " loops=" << q0_measured_loops
                  << " positions=" << q0_total_positions
                  << " q0_entries=" << live_q0_entries
                  << " q0_candidates_per_s=" << live_candidates_s
                  << " q0_positions_per_s=" << live_positions_s
                  << " q0_prp_per_s=" << live_prp_s
                  << " q0_survivors_per_loop="
                  << (q0_measured_loops == 0u ? 0u : live_q0_entries / q0_measured_loops)
                  << " q0_passes_last_loop=" << live_passes
                  << " q0_passes_total="
                  << (production_mode ? q0_total_passes : live_passes)
                  << " tuple_counts=" << live_q0_entries << ','
                  << (production_mode ? q0_total_passes : live_passes) << ','
                  << live_active_counts[0] << ',' << live_active_counts[1] << ','
                  << live_active_counts[2] << ',' << live_active_counts[3] << ','
                  << live_active_counts[4] << ',' << live_active_counts[5]
                  << " exact_results_journaled="
                  << result_journaled_count.load()
                  << " factor_origin_begin=" << options.factor_origin
                  << " factor_origin_next=" << measured_factor_origin;
#ifdef RIECOIN_RLOW_Q1_TAIL_QUEUE
        std::cout << " tuple_operator=mandatory_q1_then_queued_optional5"
                  << " tuple_pending=" << rlow_tail_host::pending_total()
                  << " member_tested=";
        for (size_t member = 0u; member < live_member_tested.size(); ++member)
          std::cout << (member == 0u ? "" : ",") << live_member_tested[member];
        std::cout << " member_prp_passed=";
        for (size_t member = 0u; member < live_member_passed.size(); ++member)
          std::cout << (member == 0u ? "" : ",") << live_member_passed[member];
#endif
        std::cout << std::endl;
#ifdef RIECOIN_RLOW_PIPELINE_TRACE
        pipeline_trace_record(9u, pipeline_trace_stdout_begin_us,
                              q0_measured_loops, result_journaled_count.load());
#endif
#endif // RIECOIN_RLOW_RESULT_SIDECAR
        last_progress = now;
      }
#ifdef RIECOIN_RLOW_SUPER_SIEVE
    // A 512-macro sparse traversal is one indivisible claimed unit.  Duration
    // expiry or a resident stop may prevent the next super-bundle, but must not
    // discard any of the eight already-built 64-macro slices.  Re-enter once at
    // the exact reservation end so the normal drain path records exhaustion.
    } while (rlow_super_sieve::active ||
             measured_factor_origin == options.factor_max ||
             (!resident_epoch_stop_requested && std::chrono::duration<double>(
                  std::chrono::steady_clock::now() - measured_begin).count() <
              static_cast<double>(measurement_seconds)));
#else
    } while (!resident_epoch_stop_requested && std::chrono::duration<double>(
                 std::chrono::steady_clock::now() - measured_begin).count() <
             static_cast<double>(measurement_seconds));
#endif
#ifdef RIECOIN_RLOW_PIPELINE_TRACE
    if (pipeline_trace_post_sync_begin_us != 0u)
      pipeline_trace_record(3u, pipeline_trace_post_sync_begin_us);
#endif
#ifdef RIECOIN_RLOW_Q1_TAIL_QUEUE
    // All queued mandatory positives are consumed before publication and all
    // final optional work is charged to the measured/window wall boundary.
#ifdef RIECOIN_RLOW_CUDA_TAIL_RAIL
    tail_rail.finish();
#ifdef RIECOIN_RLOW_ASYNC_FRONTIER
    q0_total_passes=tail_rail.total_passes();
    const auto& async_times=tail_rail.timings();
    q0_reset_ms=async_times[0];q0_sieve_ms=async_times[1];
    q0_compact_pack_ms=async_times[2];q0_prp_gpu_ms=async_times[3];
    q0_verdict_d2h_ms=async_times[4];
#endif
#else
    rlow_tail_host::finish(measured_factor_origin, q0_total_passes);
#endif
    tuple_gpu_ms = rlow_tail_host::q1_gpu_ms + rlow_tail_host::tail_gpu_ms;
#ifdef RIECOIN_RLOW_Q1_INPUT_BATCH
    tuple_gpu_ms += rlow_tail_host::input_copy_gpu_ms;
#endif
#endif
#ifdef RIECOIN_RLOW_RESULT_SIDECAR
    // One final immutable frontier after the GPU tail, before verifier drain.
    // Unobserved intermediate telemetry never removes a candidate/result.
#ifndef RIECOIN_RLOW_CUDA_TAIL_RAIL
    result_sidecar.submit({
        std::chrono::duration<double>(std::chrono::steady_clock::now() - measured_begin).count(),
        q0_prp_gpu_ms, q0_measured_loops, q0_total_positions, q0_total_passes,
        options.factor_origin, measured_factor_origin, rlow_tail_host::pending_total(), 0u}, true);
#endif
    result_sidecar.finish();
    live_complete_drained = result_sidecar.drained();
    std::cout << "phase=result_sidecar_drained results=" << live_complete_drained
              << " snapshots=" << result_sidecar.snapshots()
              << " telemetry_coalesced=" << result_sidecar.coalesced()
              << " pending=0 errors=0" << std::endl;
#endif
#ifdef RIECOIN_RLOW_ASYNC_EXACT_VERIFY
    finish_async_exact_worker();
#endif
    q0_measured_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - measured_begin).count();
#ifdef RIECOIN_RLOW_ASYNC_EXACT_VERIFY
    std::cout << "phase=async_exact_verify_drained enqueued="
              << async_exact_enqueued
              << " completed=" << async_exact_completed
              << " max_pending=" << async_exact_max_pending
              << " capacity=" << kAsyncExactQueueCapacity
              << " enqueue_wait_ms=" << async_exact_enqueue_wait_ms
              << " drain_wait_ms=" << async_exact_drain_wait_ms
              << " worker_ms=" << async_exact_worker_ms
              << " errors=0 pending=0" << std::endl;
#endif
#ifdef RIECOIN_RLOW_PIPELINE_TRACE
    for (const auto& record : pipeline_trace_records) {
      const char* kind = "unknown";
      switch (record.kind) {
        case 1u: kind = "pre_gpu_host"; break;
        case 2u: kind = "q0_gpu_to_pass_sync"; break;
        case 3u: kind = "post_sync_host"; break;
        case 4u: kind = "pass_receipt_and_tail"; break;
        case 5u: kind = "live_complete_count_d2h"; break;
        case 6u: kind = "complete_factors_d2h"; break;
        case 7u: kind = "exact_verify_and_journal"; break;
        case 8u: kind = "telemetry_d2h"; break;
        case 9u: kind = "progress_stdout"; break;
        // value0 is existing CUDA-event elapsed microseconds, not host time.
        // No new CUDA events or synchronizations are introduced by A13.
        case 10u: kind = "gpu_event_reset_us"; break;
        case 11u: kind = "gpu_event_sieve_us"; break;
        case 12u: kind = "gpu_event_pack_us"; break;
        case 13u: kind = "gpu_event_prp_us"; break;
        case 14u: kind = "gpu_event_pass_compact_us"; break;
        case 15u: kind = "gpu_event_receipt_tail_us"; break;
      }
      std::cout << "phase=rlow_pipeline_trace sequence=" << record.sequence
                << " kind=" << kind
                << " host_begin_us=" << record.host_begin_us
                << " host_elapsed_us=" << record.host_elapsed_us
                << " value0=" << record.value0
                << " value1=" << record.value1 << '\n';
    }
    std::cout << "phase=rlow_pipeline_trace_terminal records="
              << pipeline_trace_records.size()
              << " overflow=" << (pipeline_trace_overflow ? 1 : 0)
              << " steady_anchor_us=" << pipeline_trace_steady_anchor_us
              << " wall_anchor_unix_us=" << pipeline_trace_wall_anchor_unix_us
              << std::endl;
#endif
    for (auto& event : q0_events) cudaEventDestroy(event);
    uint32_t final_batch_count = 0u, final_pass_count = 0u, final_overflow = 0u;
    unsigned long long measured_total_count = 0u;
    const auto terminal_d2h_begin = std::chrono::steady_clock::now();
    cuda_check(cudaMemcpy(&final_batch_count, d_q0_count, sizeof(final_batch_count),
                          cudaMemcpyDeviceToHost), "Q0 terminal count D2H");
    cuda_check(cudaMemcpy(&final_pass_count, d_q0_pass_count, sizeof(final_pass_count),
                          cudaMemcpyDeviceToHost), "Q0 terminal pass count D2H");
    cuda_check(cudaMemcpy(&final_overflow, d_q0_overflow_count, sizeof(final_overflow),
                          cudaMemcpyDeviceToHost), "Q0 terminal overflow D2H");
    cuda_check(cudaMemcpy(&measured_total_count, d_q0_total_count,
                          sizeof(measured_total_count), cudaMemcpyDeviceToHost),
               "Q0 terminal total D2H");
    q0_total_entries = measured_total_count;
    q0_drops = final_overflow;
    if (final_batch_count > q0_capacity_bound || final_overflow != 0u)
      ++q0_replay_mismatch_words;
    if (production_mode) {
      unsigned long long complete_count = 0u;
      uint32_t tuple_overflow = 0u;
      std::array<unsigned long long, 6> member_pass_counts{};
      std::array<unsigned long long, 6> active_counts{};
      std::array<unsigned long long, 7> rejected_counts{};
      cuda_check(cudaMemcpy(&complete_count, d_complete_tuple_count,
                            sizeof(complete_count), cudaMemcpyDeviceToHost),
                 "production complete tuple count D2H");
      cuda_check(cudaMemcpy(&tuple_overflow, d_tuple_overflow_count,
                            sizeof(tuple_overflow), cudaMemcpyDeviceToHost),
                 "production tuple overflow D2H");
      cuda_check(cudaMemcpy(member_pass_counts.data(),
                            d_tuple_member_pass_counts,
                            member_pass_counts.size() *
                                sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost),
                 "production member pass counts D2H");
      cuda_check(cudaMemcpy(active_counts.data(), d_tuple_active_counts,
                            active_counts.size() * sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost),
                 "production active counts D2H");
      cuda_check(cudaMemcpy(rejected_counts.data(), d_rejected_counts,
                            rejected_counts.size() * sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost),
                 "production rejected counts D2H");
      gpu_complete_tuples = complete_count;
      q0_drops += tuple_overflow;
      if (complete_count > q0_capacity_bound) {
        ++q0_replay_mismatch_words;
        complete_count = q0_capacity_bound;
      }
      std::vector<uint64_t> complete_factors(
          static_cast<size_t>(complete_count));
      cuda_check(cudaMemcpy(complete_factors.data(), d_complete_tuple_factors,
                            complete_factors.size() * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost),
                 "production complete tuple factors D2H");
      for (size_t i = 0; i < member_pass_counts.size(); ++i) {
        production_member_pass_counts[i] = member_pass_counts[i];
        production_active_counts[i] = active_counts[i];
      }
      for (size_t i = 0; i < rejected_counts.size(); ++i)
        production_rejected_counts[i] = rejected_counts[i];

      std::vector<uint64_t> audit_factors(
          static_cast<size_t>(options.rejected_audit_per_stage) * 7u);
      if (!audit_factors.empty())
        cuda_check(cudaMemcpy(audit_factors.data(), d_rejected_sample_factors,
                              audit_factors.size() * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost),
                   "production rejected audit factors D2H");
      q0_d2h_replay_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - terminal_d2h_begin).count();

      std::sort(complete_factors.begin(), complete_factors.end());
      for (size_t i = 0; i < complete_factors.size(); ++i) {
        const uint64_t factor_value = complete_factors[i];
        if (factor_value < options.factor_origin ||
            factor_value >= measured_factor_origin)
          ++q0_replay_verdict_mismatches;
        if (i != 0u && complete_factors[i - 1u] == factor_value)
          ++q0_duplicates;
      }

      const auto exact_begin = std::chrono::steady_clock::now();
      mpz_t exact_candidate, exact_factor, nonce_candidate;
      mpz_inits(exact_candidate, exact_factor, nonce_candidate, nullptr);
      for (const uint64_t factor_value : complete_factors) {
        if (live_verified_factors.find(factor_value) != live_verified_factors.end())
          continue;
        mpz_set_u64(exact_factor, factor_value);
        mpz_mul(exact_factor, exact_factor, primorial);
        mpz_add(exact_candidate, base, exact_factor);
        const auto nonce = encode_nonce_v1(
            static_cast<uint16_t>(options.primorial_number), factor_value,
            options.primorial_offset,
            static_cast<uint16_t>(options.difficulty_offset));
        reconstruct_candidate_from_nonce(nonce_candidate, target, all_primes,
                                         nonce);
        if (mpz_cmp(exact_candidate, nonce_candidate) != 0)
          nonce_mismatch_count.fetch_add(1u);
        const uint32_t exact_prime_count =
            share_prime_count(exact_candidate, offsets, 32u, nullptr);
#ifdef RIECOIN_RLOW_SCIENCE_Q9
        const auto strict_science = strict_science_result(
            exact_candidate, options.pattern, offsets, exact_prime_count);
        account_strict_science(strict_science);
#else
        if (exact_prime_count == 7u) ++rlow_exact_sevens;
#endif
        if (exact_prime_count < 5u) {
          ++verifier_rejected_tuples;
          continue;
        }
        ++accepted;
        std::ostringstream result_record;
        result_record
            << "{\"schema\":\"riecoin-exact-result-v1\""
            << ",\"event\":\"exact_constellation\""
            << ",\"coin\":\"riecoin\""
            << ",\"work_id\":\"" << json_escape(options.work_id) << "\""
            << ",\"template_id\":\"" << json_escape(options.template_id) << "\""
            << ",\"gpu_uuid\":\"" << json_escape(actual_uuid) << "\""
            << ",\"gpu_identity_sha256\":\""
            << json_escape(options.gpu_identity_sha256) << "\""
            << ",\"runtime_sha256\":\""
            << json_escape(options.runtime_sha256) << "\""
            << ",\"factor\":\"" << factor_value << "\""
            << ",\"nonce_v1_uint256_hex\":\""
            << nonce_uint256_hex(nonce) << "\""
            << ",\"candidate_hex\":\"" << mpz_hex(exact_candidate) << "\""
            << ",\"pattern\":" << options.pattern
            << ",\"offsets\":" << offsets_json
            << ",\"primorial_number\":" << options.primorial_number
            << ",\"primorial_hex\":\"" << primorial_hex_canonical << "\""
            << ",\"primorial_offset\":\"" << options.primorial_offset << "\""
            << ",\"target_hex\":\"" << target_hex_canonical << "\""
            << ",\"prime_count\":" << exact_prime_count;
#ifdef RIECOIN_RLOW_SCIENCE_Q9
        result_record << riecoin_strict_science_q9::json_fields(strict_science);
#endif
        result_record
            << ",\"proof_state\":\"riecoin_v1_min5_exact_32_rounds\""
            << ",\"terminal_status\":\"proof_complete\""
            << ",\"submission_status\":\"not_attempted\""
            << ",\"accepted\":false,\"rejected\":false,\"stale\":false}";
        pending_result_records.push_back(result_record.str());
      }

      for (size_t stage = 0; stage < rejected_counts.size(); ++stage) {
        const size_t sample_count = static_cast<size_t>(std::min<unsigned long long>(
            rejected_counts[stage], options.rejected_audit_per_stage));
        rejected_audit_samples += sample_count;
        for (size_t sample = 0; sample < sample_count; ++sample) {
          const uint64_t factor_value =
              audit_factors[stage * options.rejected_audit_per_stage + sample];
          mpz_set_u64(exact_factor, factor_value);
          mpz_mul(exact_factor, exact_factor, primorial);
          mpz_add(exact_candidate, base, exact_factor);
          const auto nonce = encode_nonce_v1(
              static_cast<uint16_t>(options.primorial_number), factor_value,
              options.primorial_offset,
              static_cast<uint16_t>(options.difficulty_offset));
          reconstruct_candidate_from_nonce(nonce_candidate, target, all_primes,
                                           nonce);
          if (mpz_cmp(exact_candidate, nonce_candidate) != 0)
            ++rejected_audit_nonce_mismatches;
          if (share_prime_count(exact_candidate, offsets, 32u, nullptr) >= 5u)
            ++rejected_audit_exact_share_false_negatives;
        }
      }
      mpz_clears(exact_candidate, exact_factor, nonce_candidate, nullptr);
      exact_verify_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - exact_begin).count();
      accepted_count.store(accepted);
      if (rejected_audit_exact_share_false_negatives == 0u &&
          rejected_audit_nonce_mismatches == 0u && q0_drops == 0u &&
          q0_duplicates == 0u && q0_replay_mismatch_words == 0u &&
          q0_replay_verdict_mismatches == 0u) {
        for (const std::string& record : pending_result_records) {
          if (append_jsonl_durable(options.results_jsonl, record))
            result_journaled_count.fetch_add(1u);
          else
            result_journal_error_count.fetch_add(1u);
        }
      }
      result_journaled = result_journaled_count.load();
      result_journal_errors = result_journal_error_count.load();
    } else {
      std::vector<uint64_t> terminal_pass_factors(final_pass_count);
      cuda_check(cudaMemcpy(terminal_pass_factors.data(), d_q0_pass_factors,
                            terminal_pass_factors.size() * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost), "Q0 terminal proofs D2H");
      q0_d2h_replay_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - terminal_d2h_begin).count();
      q0_total_passes = final_pass_count;
      std::sort(terminal_pass_factors.begin(), terminal_pass_factors.end());
      const uint64_t last_origin = measured_factor_origin - options.candidates;
      for (size_t i = 0; i < terminal_pass_factors.size(); ++i) {
        const uint64_t factor_value = terminal_pass_factors[i];
        if (factor_value < last_origin || factor_value >= measured_factor_origin)
          ++q0_replay_verdict_mismatches;
        if (i != 0u && terminal_pass_factors[i - 1u] == factor_value)
          ++q0_duplicates;
      }
    }
    std::cout << "phase=q0_measure_complete measured_ms=" << q0_measured_ms
              << " loops=" << q0_measured_loops
              << " q0_entries=" << q0_total_entries << std::endl;
    }
  }
  nonce_roundtrip_mismatches = nonce_mismatch_count.load();
#ifdef RIECOIN_RLOW_Q1_TAIL_QUEUE
  rlow_tail_host::cleanup();
#endif
#ifdef RIECOIN_RLOW_BATCH_PRODUCER
#ifdef RIECOIN_RLOW_DUE_BUCKET_FRONTIER
  // The producer cleanup prints and releases the device frontier.  Preserve
  // the terminal proof receipt before that ownership boundary; re-snapshotting
  // after release returns the fail-closed default and corrupts both the final
  // verdict and JSON telemetry even when conservation passed on device.
  const auto future_frontier_receipt =
      rlow_sparse_event_sieve::snapshot_future_frontier();
#endif
#ifdef RIECOIN_RLOW_SUPER_SIEVE
  rlow_super_sieve::cleanup();
#endif
  rlow_batch_producer::cleanup();
#ifdef ASTRA_R201_CLEANUP
  ASTRA_R201_CLEANUP();
#endif
#endif
#ifdef RIECOIN_RLOW_CARRYLATE_Q0
  unsigned long long carrylate_paths[3]{};
  cuda_check(cudaMemcpyFromSymbol(carrylate_paths, riecoin_rlow_carrylate::rare_paths,
      sizeof(carrylate_paths)), "carrylate measured path counters");
  std::cout << "phase=carrylate_q0_paths unsupported=" << carrylate_paths[0]
            << " invariant_fallback=" << carrylate_paths[1]
            << " exact_low_repairs=" << carrylate_paths[2] << std::endl;
#endif
  const auto wall_end = std::chrono::steady_clock::now();
  const double precompute_ms = std::chrono::duration<double, std::milli>(precompute_end - precompute_begin).count();
  const double gpu_setup_ms = std::chrono::duration<double, std::milli>(gpu_setup_end - gpu_setup_begin).count();
  const double d2h_ms = std::chrono::duration<double, std::milli>(d2h_end - d2h_begin).count();
  const double compaction_ms = std::chrono::duration<double, std::milli>(compaction_end - compaction_begin).count();
  const double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
  std::array<uint64_t, 8> final_tuple_counts{};
  for (size_t i = 0; i < final_tuple_counts.size(); ++i)
    final_tuple_counts[i] = tuple_counts[i].load();
  // Funnel convention: [N0 entering Q0, Q0 pass, viable after Q1..Q6].
  // Legacy single-batch mode retains its historical deeper CPU counters.
  if (options.network_batch || production_mode) {
    final_tuple_counts[0] = q0_total_entries;
    final_tuple_counts[1] = q0_total_passes;
  }
  if (production_mode)
    for (size_t i = 0; i < production_active_counts.size(); ++i)
      final_tuple_counts[i + 2u] = production_active_counts[i];

  mpz_t positive_candidate;
  mpz_init_set_ui(positive_candidate, 11u);
  const bool positive_oracle = oracle_constellation(positive_candidate, cumulative_pattern(0), 31);
  mpz_t positive_share_candidate;
  mpz_init_set_ui(positive_share_candidate, 5u);
  const uint32_t positive_share_prime_count =
      share_prime_count(positive_share_candidate, cumulative_pattern(0), 31, nullptr);

  const double q0_candidates_per_s = q0_measured_ms > 0.0
      ? 1000.0 * static_cast<double>(q0_total_entries) / q0_measured_ms : 0.0;
  const double q0_positions_per_s = q0_measured_ms > 0.0
      ? 1000.0 * static_cast<double>(q0_total_positions) / q0_measured_ms : 0.0;
#ifdef RIECOIN_RLOW_Q1_TAIL_QUEUE
  const uint64_t total_prp_tests = q0_total_entries +
      (production_mode ? rlow_tail_host::physical_test_total() : 0u);
  const double total_prp_gpu_ms = q0_prp_gpu_ms + rlow_tail_host::q1_prp_gpu_ms +
      rlow_tail_host::tail_prp_gpu_ms;
#else
  const uint64_t total_prp_tests = q0_total_entries +
      (production_mode ? 6u * q0_total_passes : 0u);
  const double total_prp_gpu_ms = q0_prp_gpu_ms + tuple_gpu_ms;
#endif
  const double q0_prp_per_s = total_prp_gpu_ms > 0.0
      ? 1000.0 * static_cast<double>(total_prp_tests) / total_prp_gpu_ms : 0.0;
  const double prp_per_candidate = q0_total_entries > 0u
      ? static_cast<double>(total_prp_tests) /
            static_cast<double>(q0_total_entries)
      : 0.0;
  const double stage_e2e_c_s = wall_ms > 0.0
      ? 1000.0 * static_cast<double>(q0_total_entries) / wall_ms : 0.0;
  const bool r_emp_defined = q0_total_passes != 0u;
  const double r_emp = r_emp_defined
      ? static_cast<double>(q0_total_entries) /
            static_cast<double>(q0_total_passes)
      : 0.0;
  const uint32_t required_window_seconds = production_mode
      ? options.production_session_seconds : options.q0_duration_seconds;
  const bool q0_range_ok = options.factor_origin < options.factor_max &&
      q0_total_positions <= options.factor_max - options.factor_origin;
  // A fresh-work cancellation is a valid consumed-prefix boundary, not a
  // claim that the timed benchmark completed. Every queued result is drained
  // before this point; all existing exactness/conservation checks still apply.
  const bool q0_cancelled_prefix_ok = q0_resident_cancelled && q0_range_ok &&
      q0_total_positions > 0u && q0_measured_ms > 0.0 &&
      std::isfinite(q0_measured_ms) &&
      q0_total_positions % (static_cast<uint64_t>(options.candidates) *
                           rlow_queue::kMacros) == 0u;
  const bool q0_window_ok = required_window_seconds == 0u ||
      q0_cancelled_prefix_ok ||
      (q0_range_ok && riecoin_production_window::complete(
          q0_measured_ms / 1000.0, required_window_seconds,
          q0_reservation_exhausted, options.factor_origin,
          options.factor_origin + q0_total_positions, options.factor_max));
  uint64_t tuple_rejected_total = 0u;
  for (size_t i = 1u; i < production_rejected_counts.size(); ++i)
    tuple_rejected_total += production_rejected_counts[i];
  bool active_monotonic = true;
  uint64_t prior_active = q0_total_passes;
  for (const uint64_t active : production_active_counts) {
    active_monotonic = active_monotonic && active <= prior_active;
    prior_active = active;
  }
  const bool production_conservation = !production_mode ||
      (q0_total_entries <= q0_total_positions &&
       production_rejected_counts[0] + q0_total_passes == q0_total_entries &&
       tuple_rejected_total + gpu_complete_tuples == q0_total_passes &&
       production_active_counts.back() == gpu_complete_tuples &&
       gpu_complete_tuples == accepted + verifier_rejected_tuples &&
       active_monotonic &&
       rejected_audit_exact_share_false_negatives == 0u &&
       rejected_audit_nonce_mismatches == 0u);
  const uint64_t sieve_rejected = q0_total_positions >= q0_total_entries
      ? q0_total_positions - q0_total_entries : 0u;
  bool proof_ok = mismatch_words == 0 && nonce_roundtrip_mismatches == 0 &&
                  q0_mismatch_count.load() == 0u &&
                  q0_prime_false_negative_count.load() == 0u &&
                  dynamic_bit_parity.mismatches == 0u &&
                  dynamic_bit_parity.prime_false_negatives == 0u &&
                  q0_replay_mismatch_words == 0u &&
                  q0_replay_verdict_mismatches == 0u &&
                  q0_drops == 0u && q0_duplicates == 0u && q0_window_ok &&
                  production_conservation &&
                  (!options.network_batch ||
                   (q0_total_passes == final_tuple_counts[1] &&
                    q0_total_passes <= q0_total_entries)) &&
                   positive_oracle && positive_share_prime_count == 6u &&
                   result_journal_errors == 0 && result_journaled == accepted;
#ifdef RIECOIN_RLOW_DUE_BUCKET_FRONTIER
  proof_ok = proof_ok && future_frontier_receipt.pass;
#endif
#ifdef RIECOIN_RLOW_OVERLAP_PIPELINE
  const bool overlap_join_conservation =
      rlow_batch_producer::overlap_bundles * rlow_queue::kMacros ==
      q0_measured_loops;
  proof_ok = proof_ok && overlap_join_conservation;
#endif
  const std::string r_emp_json = r_emp_defined ? std::to_string(r_emp) : "null";
  std::string rlow_observed_funnel = rlow_queue::observed_funnel(
      q0_total_entries, q0_total_passes, wall_ms / 1000.0,
      q0_prp_gpu_ms, tuple_gpu_ms
#ifndef RIECOIN_RLOW_QUEUE_FUNNEL_V1_COMPAT
      , exact_verify_ms
#endif
      );
#ifdef RIECOIN_RLOW_SPARSE_EVENT_SIEVE
  // The inherited private producer has no global bitmap atomics. The sparse
  // successor does: never carry that old zero into this artifact's telemetry.
  const std::string old_atomic_field = ",\"global_bitmap_atomics\":0";
  const auto old_atomic_at = rlow_observed_funnel.find(old_atomic_field);
  if (old_atomic_at == std::string::npos)
    throw std::runtime_error("sparse producer inherited metric schema changed");
  rlow_observed_funnel.replace(old_atomic_at, old_atomic_field.size(),
      rlow_sparse_event_sieve::scatter_launches != 0u
          ? ",\"global_bitmap_atomics_used\":true,\"global_bitmap_atomic_count\":null"
          : ",\"global_bitmap_atomics_used\":false,\"global_bitmap_atomic_count\":0");
#endif
#ifdef RIECOIN_RLOW_OVERLAP_PIPELINE
  {
    std::ostringstream overlap_metrics;
    overlap_metrics << std::fixed << std::setprecision(6)
        << ",\"overlap_pipeline\":{\"groups\":"
        << rlow_batch_producer::kOverlapGroups
        << ",\"macros_per_group\":"
        << rlow_batch_producer::kMacrosPerOverlapGroup
        << ",\"bundles\":" << rlow_batch_producer::overlap_bundles
        << ",\"producer_gpu_ms\":"
        << rlow_batch_producer::overlap_producer_gpu_ms
        << ",\"consumer_q0_gpu_ms\":"
        << rlow_batch_producer::overlap_consumer_gpu_ms
        << ",\"bundle_wall_ms\":"
        << rlow_batch_producer::overlap_bundle_wall_ms
        << ",\"realized_overlap_ms\":"
        << rlow_batch_producer::overlap_realized_ms
        << ",\"join_conservation\":"
        << (overlap_join_conservation ? "true" : "false")
        << ",\"publication\":\"event_after_disjoint_group_pack\"}";
    rlow_observed_funnel.insert(rlow_observed_funnel.size() - 1u,
                                overlap_metrics.str());
  }
#endif
  std::cout << "phase=rlow_e2e_terminal e2e_c_s=" << stage_e2e_c_s
            << " q7_exact=" << rlow_exact_sevens;
#ifdef RIECOIN_RLOW_SCIENCE_Q9
  std::cout << " q8_strict_verified=" << rlow_exact_eights
            << " q9_strict_verified=" << rlow_exact_nines;
#endif
  std::cout << " observed_funnel=" << rlow_observed_funnel << std::endl;

  if (device_pipeline_mode) {
    std::ostringstream summary;
    summary << std::fixed << std::setprecision(6)
        << "{\"schema\":\"riecoin-q0-benchmark-v1\""
        << ",\"measurement_layer\":\""
        << (production_mode ? "production_multibatch_pre_submit" :
            (options.network_batch ? "network_batch_pre_submit" : "offline_q0"))
        << "\""
        << ",\"run_id\":\"" << json_escape(options.work_id + "-q0") << "\""
        << ",\"work_id\":\"" << json_escape(options.work_id) << "\""
        << ",\"template_id\":\"" << json_escape(options.template_id) << "\""
        << ",\"gpu\":{\"uuid\":\"" << json_escape(actual_uuid)
        << "\",\"pci\":\"" << json_escape(actual_pci)
        << "\",\"runtime_sha256\":\"" << json_escape(options.runtime_sha256) << "\"}"
        << ",\"geometry\":{\"target_bits\":" << mpz_sizeinbase(target, 2)
        << ",\"pattern\":" << options.pattern
        << ",\"offsets\":" << offsets_json
        << ",\"primorial_number\":" << options.primorial_number
        << ",\"primorial_offset\":\"" << options.primorial_offset << "\""
        << ",\"factor_origin_begin\":" << options.factor_origin
        << ",\"factor_origin_end\":"
        << (options.factor_origin + q0_measured_loops *
            static_cast<uint64_t>(options.candidates))
        << ",\"positions\":"
        << (production_mode ? q0_total_positions : options.candidates)
        << ",\"batch_positions\":" << options.candidates
        << ",\"batches\":" << q0_measured_loops << "}"
        << ",\"rates\":{\"positions_s\":" << q0_positions_per_s
        << ",\"q0_candidates_s\":" << q0_candidates_per_s
        << ",\"prp_s_secondary\":" << q0_prp_per_s
        << ",\"prp_per_candidate\":" << prp_per_candidate
        << ",\"stage_e2e_c_s\":" << stage_e2e_c_s
        << ",\"r_emp\":" << r_emp_json
        << ",\"r_emp_defined\":" << (r_emp_defined ? "true" : "false") << "}"
        << ",\"rlow\":" << rlow_observed_funnel
#ifdef RIECOIN_RLOW_Q1_TAIL_QUEUE
        << ",\"tuple_operator\":{\"name\":\"mandatory_q1_then_queued_optional5\""
        << ",\"cold_pack_guard_ms\":" << rlow_tail_host::cold_guard_ms
        << ",\"q1_gpu_ms\":" << rlow_tail_host::q1_gpu_ms
        << ",\"optional_gpu_ms\":" << rlow_tail_host::tail_gpu_ms
        << ",\"q1_prp_gpu_ms\":" << rlow_tail_host::q1_prp_gpu_ms
        << ",\"optional_prp_gpu_ms\":" << rlow_tail_host::tail_prp_gpu_ms
        << ",\"physical_tests\":" << rlow_tail_host::physical_test_total()
        << ",\"optional_unknown_per_member\":"
        << (q0_total_passes - rlow_tail_host::physical_tests[1])
        << ",\"member_pass_scope\":\"observed_conditionally_tested_not_all_q0_passes\""
        << ",\"flushes\":" << rlow_tail_host::flush_calls
        << ",\"max_pending\":" << rlow_tail_host::maximum_pending
        << ",\"max_flush_age_us\":" << rlow_tail_host::maximum_flush_age_us
        << ",\"deadline_late_flushes\":" << rlow_tail_host::deadline_late_flushes
        << ",\"pending\":0,\"conservation\":true}"
#endif
#ifdef RIECOIN_RLOW_BATCH_PRODUCER
#ifdef RIECOIN_RLOW_SUPER_SIEVE
        << ",\"producer_operator\":{\"name\":\"split_dense64_sparse_super512\""
        << ",\"queue_macros\":" << rlow_queue::kMacros
        << ",\"super_macros\":" << rlow_super_sieve::kSuperMacros
        << ",\"super_builds\":" << rlow_super_sieve::builds
        << ",\"slices_emitted\":" << rlow_super_sieve::slices_emitted
        << ",\"discarded_slices\":" << rlow_super_sieve::discarded_slices
        << ",\"bitmap_bytes\":" << rlow_super_sieve::bitmap_bytes
        << ",\"device_used_high_water_bytes\":"
        << (rlow_super_sieve::device_total_bytes -
            rlow_super_sieve::device_free_high_water_bytes)
        << ",\"dense_gpu_ms\":" << rlow_super_sieve::dense_gpu_ms
        << ",\"sparse_gpu_ms\":" << rlow_super_sieve::sparse_gpu_ms
        << ",\"cold_build_gpu_ms\":" << rlow_super_sieve::cold_build_gpu_ms
        << ",\"warm_build_gpu_ms\":" << rlow_super_sieve::warm_build_gpu_ms
        << ",\"slice_scan_gpu_ms\":" << rlow_super_sieve::slice_scan_gpu_ms
        << ",\"slice_pack_gpu_ms\":" << rlow_super_sieve::slice_pack_gpu_ms
        << ",\"slice_metadata_gpu_ms\":" << rlow_super_sieve::slice_metadata_gpu_ms
        << ",\"cold_parity_ms\":" << rlow_super_sieve::cold_parity_ms
        << ",\"parity_words\":" << rlow_super_sieve::parity_words
        << ",\"parity_mismatches\":" << rlow_super_sieve::parity_mismatches
        << ",\"parity_survivors_a\":" << rlow_super_sieve::parity_survivors_a
        << ",\"parity_survivors_b\":" << rlow_super_sieve::parity_survivors_b
        << ",\"parity_hash_a\":\"" << rlow_super_sieve::hex64(rlow_super_sieve::parity_hash_a) << "\""
        << ",\"parity_hash_b\":\"" << rlow_super_sieve::hex64(rlow_super_sieve::parity_hash_b) << "\""
        << ",\"parity_d2h_bytes\":" << rlow_super_sieve::parity_d2h_bytes
        << ",\"ragged_policy\":\"reject_before_first_claim\"}"
#else
        << ",\"producer_operator\":{\"name\":\"grid_private_macro_batch\""
        << ",\"macros\":" << rlow_batch_producer::kMacros
        << ",\"remainder_gpu_ms\":" << rlow_batch_producer::remainder_gpu_ms
        << ",\"cold_reference_ms\":" << rlow_batch_producer::cold_reference_ms
        << "}"
#endif
#endif
#ifdef RIECOIN_RLOW_SPARSE_EVENT_SIEVE
        << ",\"sparse_sieve_operator\":{\"name\":\"dense_private_sparse_whole_bundle_events\""
        << ",\"dense_primes\":" << rlow_sparse_event_sieve::dense_primes
        << ",\"sparse_primes\":" << rlow_sparse_event_sieve::sparse_primes
        << ",\"scatter_launches\":" << rlow_sparse_event_sieve::scatter_launches
        << ",\"partition_host_ms\":" << rlow_sparse_event_sieve::partition_host_ms
        << ",\"cold_bitmap_parity\":true"
#ifdef RIECOIN_RLOW_DUE_BUCKET_FRONTIER
        << ",\"future_frontier\":{\"ring\":"
        << rlow_sparse_event_sieve::future_frontier_ring
        << ",\"shards\":" << rlow_sparse_event_sieve::future_frontier_shards
        << ",\"segment_capacity\":"
        << rlow_sparse_event_sieve::future_frontier_capacity
        << ",\"maximum_prime\":"
        << rlow_sparse_event_sieve::future_frontier_maximum_prime
        << ",\"slot_cells\":"
        << rlow_sparse_event_sieve::future_frontier_slot_cells
        << ",\"queued\":" << future_frontier_receipt.queued
        << ",\"consumed\":" << future_frontier_receipt.consumed
        << ",\"rescheduled\":" << future_frontier_receipt.rescheduled
        << ",\"overflows\":" << future_frontier_receipt.overflows
        << ",\"empty_shards\":" << future_frontier_receipt.empty_shards
        << ",\"mismatches\":" << future_frontier_receipt.mismatches
        << ",\"worker_threads_launched\":"
        << (rlow_sparse_event_sieve::future_launches *
            rlow_sparse_event_sieve::future_frontier_shards * 256ull)
        << ",\"all_prime_baseline_visits\":"
        << (rlow_sparse_event_sieve::future_launches *
            rlow_sparse_event_sieve::future_count)
        << ",\"active_visit_ratio\":"
        << (rlow_sparse_event_sieve::future_launches != 0u &&
            rlow_sparse_event_sieve::future_count != 0u ?
            static_cast<double>(future_frontier_receipt.consumed) /
                static_cast<double>(rlow_sparse_event_sieve::future_launches *
                    rlow_sparse_event_sieve::future_count) : 0.0)
        << ",\"queue_conservation\":"
        << (future_frontier_receipt.pass ? "true" : "false") << "}"
#endif
        << "}"
#endif
#ifdef RIECOIN_RLOW_Q1_INPUT_BATCH
        << ",\"q1_input_operator\":{\"name\":\"contiguous_device_frame_batch\""
        << ",\"threshold\":" << rlow_tail_host::input_threshold
        << ",\"copy_gpu_ms\":" << rlow_tail_host::input_copy_gpu_ms
        << ",\"copy_bytes\":" << rlow_tail_host::input_copy_bytes
        << ",\"flushes\":" << rlow_tail_host::input_flushes
        << ",\"max_pending\":" << rlow_tail_host::input_max_pending
        << ",\"max_age_us\":" << rlow_tail_host::input_max_age_us
        << ",\"pending\":0,\"conservation\":true}"
#endif
#ifdef RIECOIN_RLOW_CARRYLATE_Q0
        << ",\"q0_operator\":{\"name\":\"tile8_carrylate_dynamic1152\""
        << ",\"unsupported_width_or_shape\":" << carrylate_paths[0]
        << ",\"invariant_fallback\":" << carrylate_paths[1]
        << ",\"exact_low_repairs\":" << carrylate_paths[2] << "}"
#endif
#ifdef RIECOIN_RLOW_LAZY_BASE2_Q0
        << ",\"q0_operator\":{\"name\":\"base2_lazy_window_dynamic1280\","
           "\"maximum_window_bits\":6,\"cold_full_residue_guard\":true}"
#endif
        << ",\"counts\":{\"survivors\":" << survivors.size()
        << ",\"q0_entries\":" << q0_total_entries
        << ",\"q0_passed\":" << q0_total_passes
        << ",\"tuple_counts\":[" << final_tuple_counts[0] << ','
        << final_tuple_counts[1] << ',' << final_tuple_counts[2] << ','
        << final_tuple_counts[3] << ',' << final_tuple_counts[4] << ','
        << final_tuple_counts[5] << ',' << final_tuple_counts[6] << ','
        << final_tuple_counts[7] << ']'
        << ",\"gpu_complete_tuples\":" << gpu_complete_tuples
        << ",\"cpu_exact_verified\":" << gpu_complete_tuples
        << ",\"cpu_exact_rejected\":" << verifier_rejected_tuples
        << ",\"proofs\":" << accepted
        << ",\"network_submitted\":false"
        << ",\"accepted\":0,\"rejected\":0,\"stale\":0,\"errors\":0}"
        << ",\"exactness\":{\"sieve_mismatch_words\":" << mismatch_words
#ifdef RIECOIN_RLOW_DEVICE_ROOTS
        << ",\"root_storage\":\"device_resident_no_full_pcie_roundtrip\""
        << ",\"root_relation_mode\":\"complete_independent_integer_reduction\""
        << ",\"root_relation_checked\":" << device_roots.exact_roots_checked
        << ",\"sieve_reference_mode\":\"complete_device_simple_atomic_bitmap\""
#else
        << ",\"root_storage\":\"host_materialized_then_device_copy\""
        << ",\"sieve_reference_mode\":\"complete_cpu_bitmap\""
#endif
        << ",\"q0_mismatches\":" << q0_mismatch_count.load()
        << ",\"q0_exhaustive_cpu_oracle\":"
        << (production_mode ? "false" : "true")
        << ",\"q0_exactness_scope\":\""
        << (production_mode ?
            "known_vectors_device_construction_and_bounded_rejected_audit" :
            "legacy_exhaustive_warmup") << "\""
        << ",\"dynamic_bit_parity_samples\":" << dynamic_bit_parity.samples
        << ",\"dynamic_bit_parity_mismatches\":" << dynamic_bit_parity.mismatches
        << ",\"dynamic_bit_prime_false_negatives\":"
        << dynamic_bit_parity.prime_false_negatives
        << ",\"replay_mismatch_words\":" << q0_replay_mismatch_words
        << ",\"replay_verdict_mismatches\":" << q0_replay_verdict_mismatches
        << ",\"nonce_mismatches\":" << nonce_roundtrip_mismatches
        << ",\"prime_false_negatives\":"
        << (production_mode ? rejected_audit_exact_share_false_negatives :
            q0_prime_false_negative_count.load())
        << ",\"drops\":" << q0_drops << ",\"duplicates\":" << q0_duplicates
        << ",\"rejected_audit\":{\"bound_per_stage\":"
        << options.rejected_audit_per_stage
        << ",\"samples\":" << rejected_audit_samples
        << ",\"detected_exact_share_false_negatives\":"
        << rejected_audit_exact_share_false_negatives
        << ",\"nonce_mismatches\":" << rejected_audit_nonce_mismatches
        << ",\"global_false_negative_claim\":false"
        << ",\"claim\":\"bounded_sample_only\"}}"
        << ",\"funnel\":{\"q0_entered\":" << q0_total_entries
        << ",\"q0_passed\":" << q0_total_passes
#ifdef RIECOIN_RLOW_Q1_TAIL_QUEUE
        << ",\"member_tested\":["
        << rlow_tail_host::physical_tests[0] << ','
        << rlow_tail_host::physical_tests[1] << ','
        << rlow_tail_host::physical_tests[2] << ','
        << rlow_tail_host::physical_tests[3] << ','
        << rlow_tail_host::physical_tests[4] << ','
        << rlow_tail_host::physical_tests[5] << ']'
#endif
        << ",\"member_prp_passed\":["
        << production_member_pass_counts[0] << ','
        << production_member_pass_counts[1] << ','
        << production_member_pass_counts[2] << ','
        << production_member_pass_counts[3] << ','
        << production_member_pass_counts[4] << ','
        << production_member_pass_counts[5] << ']'
        << ",\"active_after_member\":["
        << production_active_counts[0] << ',' << production_active_counts[1] << ','
        << production_active_counts[2] << ',' << production_active_counts[3] << ','
        << production_active_counts[4] << ',' << production_active_counts[5] << ']'
        << ",\"rejected_by_stage\":["
        << production_rejected_counts[0] << ',' << production_rejected_counts[1] << ','
        << production_rejected_counts[2] << ',' << production_rejected_counts[3] << ','
        << production_rejected_counts[4] << ',' << production_rejected_counts[5] << ','
        << production_rejected_counts[6] << ']'
        << ",\"gpu_complete_tuples\":" << gpu_complete_tuples
        << ",\"cpu_exact_rejected\":" << verifier_rejected_tuples
        << ",\"exact_shares\":" << accepted << "}"
        << ",\"conservation\":{\"range_positions\":" << q0_total_positions
        << ",\"sieve_rejected\":" << sieve_rejected
        << ",\"q0_entered\":" << q0_total_entries
        << ",\"q0_passed\":" << q0_total_passes
        << ",\"gpu_complete_tuples\":" << gpu_complete_tuples
        << ",\"cpu_exact_rejected\":" << verifier_rejected_tuples
        << ",\"exact_results\":" << accepted
        << ",\"drops\":" << q0_drops << ",\"duplicates\":" << q0_duplicates
        << ",\"pass\":" << (production_conservation ? "true" : "false") << "}"
        << ",\"phases_ms\":{\"reset\":" << q0_reset_ms
        << ",\"sieve\":" << q0_sieve_ms
        << ",\"terminal_d2h\":" << q0_d2h_replay_ms
        << ",\"compact_pack\":" << q0_compact_pack_ms
        << ",\"candidate_h2d\":" << q0_h2d_candidates_ms
        << ",\"q0_gpu\":" << q0_prp_gpu_ms
        << ",\"tuple_gpu\":" << tuple_gpu_ms
        << ",\"cpu_exact_verify\":" << exact_verify_ms
        << ",\"pass_compact\":" << q0_verdict_d2h_ms << "}"
        << ",\"window\":{\"warmup_s\":" << (q0_gpu_ms / 1000.0)
        << ",\"measured_s\":" << (q0_measured_ms / 1000.0)
        << ",\"requested_s\":" << required_window_seconds
        << ",\"completion_reason\":\""
        << (q0_resident_cancelled ? "cancelled_consumed_prefix" :
            (q0_reservation_exhausted ? "reservation_exhausted" : "duration")) << "\""
        << ",\"stage_wall_s\":" << (wall_ms / 1000.0) << "}"
        << ",\"terminal_status\":\"" << (proof_ok ? "pass" : "fail") << "\"}";
    if (!write_json_atomic(options.q0_summary_json, summary.str())) {
      ++result_journal_errors;
      proof_ok = false;
    }
  }
  std::ostringstream terminal_record;
  terminal_record
      << "{\"schema\":\"riecoin-terminal-v1\""
      << ",\"event\":\"offline_benchmark_terminal\""
      << ",\"coin\":\"riecoin\""
      << ",\"work_id\":\"" << json_escape(options.work_id) << "\""
      << ",\"template_id\":\"" << json_escape(options.template_id) << "\""
      << ",\"gpu_uuid\":\"" << json_escape(actual_uuid) << "\""
      << ",\"gpu_identity_sha256\":\"" << json_escape(options.gpu_identity_sha256) << "\""
      << ",\"runtime_sha256\":\"" << json_escape(options.runtime_sha256) << "\""
      << ",\"proof_state\":\"" << (proof_ok ? "exact_pass" : "failed_closed") << "\""
      << ",\"terminal_status\":\"" << (proof_ok ? "completed" : "error") << "\""
      << ",\"submission_status\":\"not_attempted\""
      << ",\"accepted\":0,\"rejected\":0,\"stale\":0"
      << ",\"errors\":" << result_journal_errors
      << ",\"interesting_results\":" << accepted
      << ",\"results_journaled\":" << result_journaled
      << ",\"pattern\":" << options.pattern
      << ",\"offsets\":" << offsets_json
      << ",\"primorial_number\":" << options.primorial_number
      << ",\"primorial_hex\":\"" << primorial_hex_canonical << "\""
      << ",\"primorial_offset\":\"" << options.primorial_offset << "\""
             << ",\"target_hex\":\"" << target_hex_canonical << "\"}"
      ;
  if (!append_jsonl_durable(options.results_jsonl, terminal_record.str()))
    ++result_journal_errors;

  std::cout << "RIECOIN_CUDA_VERTICAL_V1\n"
            << "coin=riecoin work_mode=offline_constellation_v1"
            << " work_id=" << options.work_id
            << " template_id=" << options.template_id
            << " target_hex=" << target_hex_canonical << '\n'
            << "requested_device=" << requested_device
            << " resolved_cuda_ordinal=" << options.device << " device=" << prop.name
            << " uuid=" << actual_uuid << " cc=" << prop.major << '.' << prop.minor << '\n'
            << "consensus_profile=riecoin-mainnet-v1 pattern=" << options.pattern
            << " network_batch=" << (options.network_batch ? 1 : 0)
            << " network_submitted=0"
            << " cumulative=0,2," << (options.pattern == 0 ? "6,8,12,18,20" : "8,12,14,18,20") << '\n'
            << "fixture_candidates=" << options.candidates
            << " prime_limit=" << options.prime_limit
            << " primes=" << sieve_prime_count
            << " primorial_number=" << options.primorial_number
            << " primorial_bits=" << mpz_sizeinbase(primorial, 2)
            << " primorial_offset=" << options.primorial_offset
            << " primorial_source=" << (options.initial_target_bits == 0 ? "fixed" : "derived")
            << " initial_target_bits=" << options.initial_target_bits
            << " factor_max=" << options.factor_max
            << " factor_origin=" << options.factor_origin
            << " factor_origin_end="
            << (options.factor_origin + q0_measured_loops *
                static_cast<uint64_t>(options.candidates))
            << " target_bits=" << mpz_sizeinbase(target, 2) << '\n'
            << std::fixed << std::setprecision(3)
            << "gpu_ms=" << gpu_ms
            << " gpu_domain_per_s=" << (1000.0 * options.candidates / gpu_ms)
            << " cpu_sieve_ms=" << cpu_ms
#ifdef RIECOIN_RLOW_DEVICE_ROOTS
            << " sieve_reference_mode=exhaustive_device_simple_atomic"
            << " root_relation_checked=" << device_roots.exact_roots_checked
#else
            << " sieve_reference_mode=exhaustive_cpu_bitmap"
#endif
            << '\n'
            << "precompute_ms=" << precompute_ms
            << " gpu_setup_ms=" << gpu_setup_ms
            << " d2h_ms=" << d2h_ms
            << " compaction_ms=" << compaction_ms
            << " wall_ms=" << wall_ms << '\n'
             << "survivors=" << survivors.size()
            << " oracle_checked=" << checked_count.load()
             << " oracle_accepted=" << accepted
            << " results_journaled=" << result_journaled
            << " result_journal_errors=" << result_journal_errors
             << " oracle_threads=" << options.oracle_threads
            << " oracle_ms=" << oracle_ms
            << " candidates_per_s="
            << (wall_ms > 0.0 ? 1000.0 * checked_count.load() / wall_ms : 0.0)
            << " tuple_counts="
            << final_tuple_counts[0] << ',' << final_tuple_counts[1] << ','
            << final_tuple_counts[2] << ',' << final_tuple_counts[3] << ','
            << final_tuple_counts[4] << ',' << final_tuple_counts[5] << ','
            << final_tuple_counts[6] << ',' << final_tuple_counts[7] << '\n'
            << "q0_measured_ms=" << q0_measured_ms
            << " q0_loops=" << q0_measured_loops
            << " q0_candidates_per_s=" << q0_candidates_per_s
            << " q0_positions_per_s=" << q0_positions_per_s
            << " q0_prp_per_s=" << q0_prp_per_s
            << " q0_prp_per_candidate=1.000"
            << " q0_entries=" << q0_total_entries << '\n'
            << "q0_warmup_mismatches=" << q0_mismatch_count.load()
            << " q0_prime_false_negatives=" << q0_prime_false_negative_count.load()
            << " dynamic_bit_parity_samples=" << dynamic_bit_parity.samples
            << " dynamic_bit_parity_mismatches=" << dynamic_bit_parity.mismatches
            << " dynamic_bit_prime_false_negatives="
            << dynamic_bit_parity.prime_false_negatives
            << " q0_replay_mismatch_words=" << q0_replay_mismatch_words
            << " q0_replay_verdict_mismatches=" << q0_replay_verdict_mismatches
            << " drops=" << q0_drops << " duplicates=" << q0_duplicates << '\n'
            << "accepted=0 rejected=0 stale=0"
            << " submission_status=not_attempted" << '\n'
            << "mismatch_words=" << mismatch_words
            << " nonce_roundtrip_checked=" << survivors.size()
            << " nonce_roundtrip_mismatches=" << nonce_roundtrip_mismatches
             << " positive_oracle_11=" << (positive_oracle ? "PASS" : "FAIL")
            << " positive_share_5_prime_count=" << positive_share_prime_count
            << " terminal_fragment=1 errors=" << result_journal_errors << "\n"
            << "verdict=" << (proof_ok && result_journal_errors == 0 ? "PASS" : "FAIL") << '\n';

  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  rlow_queue::release();
  if (rlow_private_planes != nullptr) {
    cudaFree(rlow_private_planes);
    rlow_private_planes = nullptr;
  }
  rlow_plane_count = 0u;
  rlow_plane_words = 0u;
  cudaFree(d_words);
  cudaFree(d_primes);
  cudaFree(d_first);
  if (d_q0_candidates != nullptr) cudaFree(d_q0_candidates);
  if (d_q0_initial_words != nullptr) cudaFree(d_q0_initial_words);
  if (d_q0_base != nullptr) cudaFree(d_q0_base);
  if (d_q0_primorial != nullptr) cudaFree(d_q0_primorial);
  if (d_q0_verdicts != nullptr) cudaFree(d_q0_verdicts);
  if (d_q0_factors != nullptr) cudaFree(d_q0_factors);
  if (d_q0_pass_factors != nullptr) cudaFree(d_q0_pass_factors);
  if (d_q0_count != nullptr) cudaFree(d_q0_count);
  if (d_q0_pass_count != nullptr) cudaFree(d_q0_pass_count);
  if (d_q0_overflow_count != nullptr) cudaFree(d_q0_overflow_count);
  if (d_q0_total_count != nullptr) cudaFree(d_q0_total_count);
  if (d_tuple_members != nullptr) cudaFree(d_tuple_members);
  if (d_tuple_verdicts != nullptr) cudaFree(d_tuple_verdicts);
  if (d_tuple_offsets != nullptr) cudaFree(d_tuple_offsets);
  if (d_tuple_member_pass_counts != nullptr) cudaFree(d_tuple_member_pass_counts);
  if (d_tuple_active_counts != nullptr) cudaFree(d_tuple_active_counts);
  if (d_rejected_counts != nullptr) cudaFree(d_rejected_counts);
  if (d_rejected_sample_factors != nullptr) cudaFree(d_rejected_sample_factors);
  if (d_complete_tuple_factors != nullptr) cudaFree(d_complete_tuple_factors);
  if (d_complete_tuple_count != nullptr) cudaFree(d_complete_tuple_count);
  if (d_tuple_overflow_count != nullptr) cudaFree(d_tuple_overflow_count);
  mpz_clear(positive_candidate);
  mpz_clear(positive_share_candidate);
  mpz_clear(base);
  mpz_clear(target_mod);
  mpz_clear(primorial);
  mpz_clear(target);
  return proof_ok && result_journal_errors == 0 ? 0 : 2;
 } catch (const std::exception& e) {
  std::cerr << "RIECOIN_CUDA_VERTICAL_V1 error=" << e.what() << '\n';
  // Parsing may itself have failed, so recover only the bounded identity
  // switches that can be read without accepting or executing any work.
  Options recovery;
  for (int i = 1; i + 1 < argc; ++i) {
    const std::string key(argv[i]);
    if (key == "--results-jsonl") recovery.results_jsonl = argv[++i];
    else if (key == "--work-id") recovery.work_id = argv[++i];
    else if (key == "--template-id") recovery.template_id = argv[++i];
    else if (key == "--runtime-sha256") recovery.runtime_sha256 = argv[++i];
    else if (key == "--gpu-identity-sha256") recovery.gpu_identity_sha256 = argv[++i];
  }
  if (!recovery.results_jsonl.empty()) {
    std::ostringstream error_record;
    error_record
        << "{\"schema\":\"riecoin-terminal-v1\""
        << ",\"event\":\"offline_benchmark_terminal\""
        << ",\"coin\":\"riecoin\""
        << ",\"work_id\":\"" << json_escape(recovery.work_id) << "\""
        << ",\"template_id\":\"" << json_escape(recovery.template_id) << "\""
        << ",\"gpu_identity_sha256\":\"" << json_escape(recovery.gpu_identity_sha256) << "\""
        << ",\"runtime_sha256\":\"" << json_escape(recovery.runtime_sha256) << "\""
        << ",\"proof_state\":\"error\""
        << ",\"terminal_status\":\"error\""
        << ",\"submission_status\":\"not_attempted\""
        << ",\"accepted\":0,\"rejected\":0,\"stale\":0,\"errors\":1"
        << ",\"detail\":\"" << json_escape(e.what()) << "\"}";
    append_jsonl_durable(recovery.results_jsonl, error_record.str());
  }
  return 1;
 }
}

// END INLINED riecoin-rlow-e2e-queued.cu

// END INLINED riecoin-rlow-e2e-deepqueue64.cu

// END INLINED riecoin-rlow-e2e-deeproot64.cu

// END INLINED riecoin-rlow-e2e-r194-residentroots-width1148-asyncexact.cu

// END INLINED riecoin-rlow-e2e-r200-canonicalp114-min109-devicecompressed-width1148-asyncexact.cu

// END INLINED riecoin-rlow-e2e-r201-canonicalp114-min109-devicecompressed-width1148-asyncexact.cu

// END INLINED riecoin-rlow-e2e-r202-canonicalp114-min109-compressedconsumers-width1148-asyncexact.cu

// END INLINED astra-riecoin-async-frontier-a16.cu

// END INLINED astra-riecoin-wide1280-tpi4-a19.cu

// END INLINED astra-riecoin-blocking-ready-a25.cu

#undef cudaSetDevice

// END INLINED astra-riecoin-wait-policy-a27.cu

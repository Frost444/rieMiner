#pragma once

// Opt-in Windows CPU cache, not a root/template cache. A hit replaces only
// primes_to(L), P mod p and its inverse. Every base mod p / seven roots is fresh.
// Version 1 caches the COMPLETE ordered prime table and the inverse suffix.
// Trust boundary: private local cache written by this implementation; SHA-256
// detects corruption/torn writes, not an attacker able to replace data+digest.
// No locks: competing builders publish identical immutable content atomically.
// A crash may leave an ignored .tmp file; no reader ever considers .tmp files.
#include "riecoin_prime_root_precompute.hpp"
#include "../r184_verified_file_lease.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#ifdef _MSC_VER
#pragma comment(lib, "bcrypt.lib")
#endif
#else
#include <openssl/evp.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cerrno>
#endif

#include <atomic>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

namespace riecoin_prime_root_cache {
namespace exact = riecoin_prime_root_precompute;
using Clock = std::chrono::steady_clock;
using Bytes = std::vector<std::uint8_t>;
using Digest = std::array<std::uint8_t, 32>;

constexpr std::uint32_t kFormatVersion = 1U;
constexpr std::uint32_t kGeneratorVersion = 1U; // ordered sieve/inverse semantics
#ifdef RIECOIN_RLOW_PRIME_ROOT_CACHE_MAX_BYTES
constexpr std::size_t kMaxFileBytes =
    static_cast<std::size_t>(RIECOIN_RLOW_PRIME_ROOT_CACHE_MAX_BYTES);
#else
constexpr std::size_t kMaxFileBytes = 256U * 1024U * 1024U;
#endif
static_assert(kMaxFileBytes >= 256U * 1024U * 1024U,
              "cache ceiling cannot narrow the established format guard");
constexpr std::size_t kMaxPrimorialBytes = 8192U;
constexpr std::size_t kHeaderBytes = 36U;
constexpr std::array<std::uint8_t, 8> kMagic{{'R','C','P','I','0','0','0','1'}};
#ifdef RIECOIN_RLOW_PRIME_ROOT_CACHE_CANONICAL_FIRST
constexpr std::uint32_t kCanonicalFirst =
    RIECOIN_RLOW_PRIME_ROOT_CACHE_CANONICAL_FIRST;
#ifdef RIECOIN_RLOW_PRIME_ROOT_CACHE_CANONICAL_MIN_FIRST
constexpr std::uint32_t kCanonicalMinFirst =
    RIECOIN_RLOW_PRIME_ROOT_CACHE_CANONICAL_MIN_FIRST;
#else
constexpr std::uint32_t kCanonicalMinFirst = kCanonicalFirst;
#endif
static_assert(kCanonicalFirst == 114U &&
              kCanonicalMinFirst >= 107U &&
              kCanonicalMinFirst <= kCanonicalFirst,
              "the two-factor canonical P114 bridge covers P107..P114");
constexpr std::size_t kCanonicalPrefixCapacity =
    static_cast<std::size_t>(kCanonicalFirst - kCanonicalMinFirst);
#endif
#if defined(RIECOIN_RLOW_CANONICAL_DEVICE_TRANSFORM) && \
    !defined(RIECOIN_RLOW_PRIME_ROOT_CACHE_CANONICAL_FIRST)
#error "R157 device transform requires the certified canonical P114 cache"
#endif

inline double elapsed_ms(Clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

struct Metrics {
  const char* status = "disabled";
  const char* route = "legacy_exact_key";
  bool published = false;
  std::string cache_path;
  std::uint64_t file_bytes = 0U;
  double key_ms = 0, load_ms = 0, verify_ms = 0, build_ms = 0, publish_ms = 0;
#ifdef RIECOIN_RLOW_PRIME_ROOT_CACHE_CANONICAL_FIRST
  std::uint32_t requested_first = 0U, canonical_first = 0U;
  std::uint32_t delta_factors = 0U, direct_prefix_inverses = 0U;
  std::uint64_t transformed_inverses = 0U, complete_verified_inverses = 0U;
  std::uint64_t quotient = 1U;
  double transform_ms = 0, complete_verify_ms = 0;
#endif
};

struct Basis {
  std::vector<std::uint32_t> all_primes;
  std::vector<std::uint32_t> inverses;
  std::uint32_t first = 0U;
#ifdef RIECOIN_RLOW_CANONICAL_DEVICE_TRANSFORM
  // R157 keeps the immutable canonical suffix byte-for-byte unchanged.  The
  // requested P109..P114 prefix and one quotient are consumed by the GPU root
  // builder, which proves every effective inverse before sieve use.
  std::uint32_t requested_first = 0U;
  std::uint32_t prefix_inverse_count = 0U;
  std::array<std::uint32_t, kCanonicalPrefixCapacity> prefix_inverses{};
  std::uint64_t inverse_multiplier = 1U;
  std::uint64_t inverse_multiplier_second = 1U;
  std::vector<std::uint32_t> requested_primorial_words;
  bool exact_direct_geometry = false;
#endif
};

struct Key {
  std::uint32_t limit = 0U, first = 0U;
  Bytes primorial; // unsigned minimal big-endian encoding, not a truncated hash
};

namespace detail {
inline void put32(Bytes& bytes, std::uint32_t value) {
  for (unsigned shift = 0U; shift != 32U; shift += 8U)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}
template<class ByteContainer>
inline std::uint32_t get32(const ByteContainer& bytes, std::size_t offset) {
  return std::uint32_t(bytes.at(offset)) |
      (std::uint32_t(bytes.at(offset + 1U)) << 8U) |
      (std::uint32_t(bytes.at(offset + 2U)) << 16U) |
      (std::uint32_t(bytes.at(offset + 3U)) << 24U);
}

inline Digest sha256(const std::uint8_t* bytes, std::size_t size) {
#ifdef _WIN32
  if (size > (std::numeric_limits<ULONG>::max)())
    throw std::length_error("cache hash input too large");
  struct Algorithm {
    BCRYPT_ALG_HANDLE handle = nullptr;
    ~Algorithm() { if (handle) BCryptCloseAlgorithmProvider(handle, 0); }
  } algorithm;
  if (BCryptOpenAlgorithmProvider(&algorithm.handle, BCRYPT_SHA256_ALGORITHM,
                                  nullptr, 0) < 0)
    throw std::runtime_error("cache SHA-256 provider unavailable");
  ULONG object_bytes = 0, returned = 0;
  if (BCryptGetProperty(algorithm.handle, BCRYPT_OBJECT_LENGTH,
                       reinterpret_cast<PUCHAR>(&object_bytes), sizeof(object_bytes),
                       &returned, 0) < 0 || returned != sizeof(object_bytes))
    throw std::runtime_error("cache SHA-256 object size failed");
  Bytes object(object_bytes);
  struct Hash {
    BCRYPT_HASH_HANDLE handle = nullptr;
    ~Hash() { if (handle) BCryptDestroyHash(handle); }
  } hash;
  if (BCryptCreateHash(algorithm.handle, &hash.handle, object.data(), object_bytes,
                      nullptr, 0, 0) < 0 ||
      BCryptHashData(hash.handle, const_cast<PUCHAR>(bytes),
                     static_cast<ULONG>(size), 0) < 0)
    throw std::runtime_error("cache SHA-256 input failed");
  Digest digest{};
  if (BCryptFinishHash(hash.handle, digest.data(), static_cast<ULONG>(digest.size()), 0) < 0)
    throw std::runtime_error("cache SHA-256 finish failed");
  return digest;
#else
  Digest digest{};unsigned int written=0;
  if(EVP_Digest(bytes,size,digest.data(),&written,EVP_sha256(),nullptr)!=1||written!=digest.size())
    throw std::runtime_error("cache SHA-256 failed");
  return digest;
#endif
}
inline Digest sha256(const Bytes& bytes) { return sha256(bytes.data(), bytes.size()); }
inline std::string hex(const Digest& digest) {
  std::string result;
  for (const auto byte : digest) {
    result += "0123456789abcdef"[byte >> 4U];
    result += "0123456789abcdef"[byte & 15U];
  }
  return result;
}

inline Bytes key_bytes(const Key& key) {
  Bytes bytes(kMagic.begin(), kMagic.end());
  put32(bytes, kFormatVersion);
  put32(bytes, kGeneratorVersion);
  put32(bytes, key.limit);
  put32(bytes, key.first);
  put32(bytes, static_cast<std::uint32_t>(key.primorial.size()));
  bytes.insert(bytes.end(), key.primorial.begin(), key.primorial.end());
  return bytes;
}

#ifdef _WIN32
struct File {
  HANDLE handle = INVALID_HANDLE_VALUE;
  explicit File(HANDLE value) : handle(value) {}
  ~File() { if (handle && handle != INVALID_HANDLE_VALUE) CloseHandle(handle); }
  File(const File&) = delete;
  File& operator=(const File&) = delete;
  void close() { if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle); handle = INVALID_HANDLE_VALUE; }
};

// FILE_SHARE_DELETE lets an open reader finish its immutable old snapshot while
// another process atomically replaces the directory entry with a complete file.
inline Bytes read_file(const std::filesystem::path& path) {
  File file(CreateFileW(path.c_str(), GENERIC_READ,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr));
  if (file.handle == INVALID_HANDLE_VALUE) throw std::runtime_error("cache open failed");
  LARGE_INTEGER length{};
  if (!GetFileSizeEx(file.handle, &length) ||
      length.QuadPart < static_cast<LONGLONG>(kHeaderBytes + Digest{}.size()) ||
      length.QuadPart > static_cast<LONGLONG>(kMaxFileBytes))
    throw std::runtime_error("cache file size outside bounds");
  Bytes bytes(static_cast<std::size_t>(length.QuadPart));
  DWORD received = 0;
  if (!ReadFile(file.handle, bytes.data(), static_cast<DWORD>(bytes.size()), &received, nullptr) ||
      received != bytes.size()) throw std::runtime_error("cache truncated read");
  std::uint8_t extra = 0;
  if (!ReadFile(file.handle, &extra, 1, &received, nullptr) || received != 0)
    throw std::runtime_error("cache changed size during read");
  return bytes;
}

inline bool publish(const std::filesystem::path& path, const Bytes& bytes) {
  auto identical_exists = [&]() {
    try { return read_file(path) == bytes; }
    catch (const std::exception&) { return false; }
  };
  // Another process may have completed the same immutable key already. Its
  // byte-identical complete file is sufficient; do not force a replacement.
  if (identical_exists()) return false;
  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  if (error) throw std::runtime_error("cache directory unavailable");
  static std::atomic<std::uint64_t> serial{0};
  std::filesystem::path temporary;
  HANDLE raw = INVALID_HANDLE_VALUE;
  for (unsigned attempt = 0; attempt < 4U && raw == INVALID_HANDLE_VALUE; ++attempt) {
    temporary = path;
    temporary += L".tmp-" + std::to_wstring(GetCurrentProcessId()) + L"-" +
        std::to_wstring(GetTickCount64()) + L"-" + std::to_wstring(serial.fetch_add(1));
    raw = CreateFileW(temporary.c_str(), GENERIC_WRITE, 0, nullptr,
                      CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (raw == INVALID_HANDLE_VALUE && GetLastError() != ERROR_FILE_EXISTS)
      throw std::runtime_error("cache temporary create failed");
  }
  if (raw == INVALID_HANDLE_VALUE) throw std::runtime_error("cache temporary collision");
  File file(raw);
  try {
    DWORD written = 0;
    if (!WriteFile(file.handle, bytes.data(), static_cast<DWORD>(bytes.size()), &written, nullptr) ||
        written != bytes.size() || !FlushFileBuffers(file.handle))
      throw std::runtime_error("cache publication write failed");
    file.close();
    // Windows can transiently refuse overlapping replacements. Retry only
    // sharing/access conflicts, at most 1+2+4 ms, all charged to publish_ms.
    // A still-busy/read-only directory is a best-effort cache miss, not a wait
    // or a dependency on another process making progress.
    bool moved = false;
    DWORD move_error = ERROR_SUCCESS;
    for (unsigned attempt = 0; attempt < 4; ++attempt) {
      moved = MoveFileExW(temporary.c_str(), path.c_str(),
                          MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != FALSE;
      if (moved) break;
      move_error = GetLastError();
      if (identical_exists()) {
        DeleteFileW(temporary.c_str());
        return false;
      }
      if (attempt == 3 || (move_error != ERROR_SHARING_VIOLATION && move_error != ERROR_ACCESS_DENIED)) break;
      Sleep(1U << attempt);
    }
    if (!moved) throw std::runtime_error("cache atomic publication failed win32=" + std::to_string(move_error));
  } catch (...) {
    file.close();
    DeleteFileW(temporary.c_str()); // only the exact temporary file we created
    throw;
  }
  return true;
}
#else
struct File {
  int handle=-1;
  explicit File(int fd):handle(fd){}
  ~File(){close();}
  File(const File&)=delete;File& operator=(const File&)=delete;
  void close(){if(handle>=0){::close(handle);handle=-1;}}
};
inline Bytes read_file(const std::filesystem::path& path){
  File f(::open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW));struct stat st{};
  if(f.handle<0||::fstat(f.handle,&st)!=0||!S_ISREG(st.st_mode)||
     st.st_size<static_cast<off_t>(kHeaderBytes+Digest{}.size())||
     st.st_size>static_cast<off_t>(kMaxFileBytes))throw std::runtime_error("cache file size/type outside bounds");
  Bytes bytes(static_cast<std::size_t>(st.st_size));std::size_t done=0;
  while(done<bytes.size()){
    const auto n=::read(f.handle,bytes.data()+done,bytes.size()-done);
    if(n<0&&errno==EINTR)continue;
    if(n<=0)throw std::runtime_error("cache truncated read");
    done+=static_cast<std::size_t>(n);
  }
  std::uint8_t extra=0;ssize_t n;do{n=::read(f.handle,&extra,1);}while(n<0&&errno==EINTR);
  if(n!=0)throw std::runtime_error("cache changed size during read");
  return bytes; // decode verifies the complete private snapshot, never a pathname.
}
inline bool publish(const std::filesystem::path& path,const Bytes& bytes){
  try{if(read_file(path)==bytes)return false;}catch(const std::exception&){}
  std::filesystem::create_directories(path.parent_path());
  std::string pattern=path.string()+".tmp-XXXXXX";std::vector<char> name(pattern.begin(),pattern.end());name.push_back(0);
  File f(::mkstemp(name.data()));if(f.handle<0)throw std::runtime_error("cache temporary create failed");
  const std::filesystem::path temporary(name.data());
  try{
    std::size_t done=0;while(done<bytes.size()){
      const auto n=::write(f.handle,bytes.data()+done,bytes.size()-done);
      if(n<0&&errno==EINTR)continue;
      if(n<=0)throw std::runtime_error("cache publication write failed");
      done+=static_cast<std::size_t>(n);
    }
    if(::fsync(f.handle)!=0)throw std::runtime_error("cache publication flush failed");
    f.close();if(::rename(temporary.c_str(),path.c_str())!=0)throw std::runtime_error("cache atomic publication failed");
    File parent(::open(path.parent_path().c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC));
    if(parent.handle<0||::fsync(parent.handle)!=0)throw std::runtime_error("cache directory flush failed");
  }catch(...){::unlink(temporary.c_str());throw;}
  return true;
}
#endif
} // namespace detail

inline Key make_key(std::uint32_t limit, std::uint32_t first, mpz_srcptr primorial) {
  if (mpz_sgn(primorial) <= 0) throw std::invalid_argument("invalid cache primorial");
  const auto bytes = (mpz_sizeinbase(primorial, 2) + 7U) / 8U;
  if (bytes > kMaxPrimorialBytes) throw std::length_error("uncacheable primorial size");
  Key key{limit, first, Bytes(bytes)};
  std::size_t written = 0;
  mpz_export(key.primorial.data(), &written, 1, 1, 1, 0, primorial);
  key.primorial.resize(written);
  return key;
}

inline std::filesystem::path default_directory() {
#ifdef _WIN32
  std::array<wchar_t, 32768> executable{};
  const auto count = GetModuleFileNameW(nullptr, executable.data(),
                                       static_cast<DWORD>(executable.size()));
  if (count == 0 || count >= executable.size())
    throw std::runtime_error("cache stage executable path unavailable");
  return std::filesystem::path(executable.data()).parent_path() /
      L"riecoin-prime-inverse-cache-v1";
#else
  if(const char* p=std::getenv("XDG_CACHE_HOME");p&&*p)return std::filesystem::path(p)/"horizon-riecoin";
  if(const char* p=std::getenv("HOME");p&&*p)return std::filesystem::path(p)/".cache/horizon-riecoin";
  throw std::runtime_error("No private user cache directory; set XDG_CACHE_HOME");
#endif
}
inline std::filesystem::path cache_path(const std::filesystem::path& directory, const Key& key) {
  return directory / (detail::hex(detail::sha256(detail::key_bytes(key))) + ".rcpi");
}

inline std::filesystem::path local_directory(const std::filesystem::path& directory) {
#ifdef _WIN32
  const auto original = directory.native();
  auto separator = [](wchar_t c) { return c == L'\\' || c == L'/'; };
  if (original.size() >= 2U && separator(original[0]) && separator(original[1]))
    throw std::invalid_argument("UNC/device cache path is outside local scope");
  const auto resolved = std::filesystem::absolute(directory);
  const auto root = resolved.root_name().native();
  // Reject UNC/device paths and mapped network drives before any cache I/O.
  // Only a local fixed drive is in this prototype's atomic-publication scope.
  if (root.size() != 2U || root[1] != L':' ||
      GetDriveTypeW(resolved.root_path().c_str()) != DRIVE_FIXED)
    throw std::invalid_argument("cache requires a local fixed drive");
  return resolved;
#else
  return std::filesystem::absolute(directory);
#endif
}

// Only this small prefix is needed to derive P before the full cache key exists.
// Return enough primes to hit the SAME derivation stopping condition, or all
// primes through L so the existing caller reports the SAME invalid geometry.
// No fixed bound on P is assumed; the prefix expands until sufficient or L.
inline std::vector<std::uint32_t> primorial_seed(std::uint32_t limit,
    std::uint32_t fixed_first, std::uint32_t initial_bits, std::uint64_t factor_max) {
  if (factor_max == 0U) throw std::invalid_argument("zero factor bound");
  mpz_t bound, factor, product;
  mpz_inits(bound, factor, product, nullptr);
  struct Clear { mpz_ptr a, b, c; ~Clear() { mpz_clears(a, b, c, nullptr); } } clear{bound, factor, product};
  if (initial_bits != 0U) {
    mpz_set_ui(bound, 1U);
    mpz_mul_2exp(bound, bound, initial_bits);
    mpz_sub_ui(bound, bound, 1U);
    mpz_import(factor, 1, -1, sizeof(factor_max), 0, 0, &factor_max);
    mpz_fdiv_q(bound, bound, factor);
  }
  std::uint32_t prefix_limit = (std::min)(limit, 1024U);
  for (;;) {
    auto prefix = exact::primes_to(prefix_limit);
    bool enough = fixed_first < prefix.size();
    if (initial_bits != 0U) {
      enough = false;
      mpz_set_ui(product, 1U);
      for (const auto prime : prefix) {
        mpz_mul_ui(product, product, prime);
        if (mpz_cmp(product, bound) >= 0) { enough = true; break; }
      }
    }
    if (enough || prefix_limit == limit) return prefix;
    prefix_limit = static_cast<std::uint32_t>((std::min)(
        std::uint64_t(limit), std::uint64_t(prefix_limit) * 2U));
  }
}

inline Basis build(std::uint32_t limit, std::uint32_t first, mpz_srcptr primorial,
                   std::size_t inverse_headroom = 0U) {
  Basis basis;
  basis.all_primes = exact::primes_to(limit);
  basis.first = first;
  if (first > basis.all_primes.size() || mpz_sgn(primorial) <= 0)
    throw std::invalid_argument("invalid prime-root geometry");
  const auto inverse_count = basis.all_primes.size() - first;
  if (inverse_headroom > basis.inverses.max_size() - inverse_count)
    throw std::length_error("inverse headroom overflow");
  basis.inverses.reserve(inverse_count + inverse_headroom);
  for (std::size_t i = first; i < basis.all_primes.size(); ++i) {
    const auto prime = basis.all_primes[i];
    basis.inverses.push_back(exact::inverse_mod(
        static_cast<std::uint32_t>(mpz_fdiv_ui(primorial, prime)), prime));
  }
  return basis;
}

inline Bytes encode(const Key& key, const Basis& basis) {
  const auto bytes = kHeaderBytes + key.primorial.size() + Digest{}.size() +
      4ULL * (basis.all_primes.size() + basis.inverses.size());
  if (bytes > kMaxFileBytes || basis.first != key.first ||
      key.first > basis.all_primes.size() ||
      basis.inverses.size() != basis.all_primes.size() - key.first)
    throw std::length_error("uncacheable table size or shape");
  Bytes output(kMagic.begin(), kMagic.end());
  output.reserve(static_cast<std::size_t>(bytes));
  detail::put32(output, kFormatVersion);
  detail::put32(output, kGeneratorVersion);
  detail::put32(output, key.limit);
  detail::put32(output, key.first);
  detail::put32(output, static_cast<std::uint32_t>(key.primorial.size()));
  detail::put32(output, static_cast<std::uint32_t>(basis.all_primes.size()));
  detail::put32(output, static_cast<std::uint32_t>(basis.inverses.size()));
  output.insert(output.end(), key.primorial.begin(), key.primorial.end());
  for (const auto prime : basis.all_primes) detail::put32(output, prime);
  for (const auto inverse : basis.inverses) detail::put32(output, inverse);
  const auto digest = detail::sha256(output);
  output.insert(output.end(), digest.begin(), digest.end());
  return output;
}

template<class ByteContainer>
inline Basis decode(const ByteContainer& bytes, const Key& key,
                    const std::vector<std::uint32_t>& seed,
                    std::size_t inverse_headroom = 0U,
                    const r184_lease::VerifiedBytes* sealed = nullptr) {
  if (bytes.size() < kHeaderBytes + Digest{}.size() || bytes.size() > kMaxFileBytes ||
      !std::equal(kMagic.begin(), kMagic.end(), bytes.begin()) ||
      detail::get32(bytes, 8) != kFormatVersion ||
      detail::get32(bytes, 12) != kGeneratorVersion ||
      detail::get32(bytes, 16) != key.limit || detail::get32(bytes, 20) != key.first ||
      detail::get32(bytes, 24) != key.primorial.size() ||
      key.primorial.empty() || key.primorial.size() > kMaxPrimorialBytes)
    throw std::runtime_error("cache header/key mismatch");
  const std::uint64_t count = detail::get32(bytes, 28);
  const std::uint64_t inverse_count = detail::get32(bytes, 32);
  const auto max_count = key.limit < 2U ? 0ULL : (std::uint64_t(key.limit) + 1U) / 2U;
  if (count > max_count || count < key.first || count < seed.size() ||
      inverse_count != count - key.first ||
      bytes.size() != kHeaderBytes + key.primorial.size() + 4ULL * (count + inverse_count) + Digest{}.size() ||
      !std::equal(key.primorial.begin(), key.primorial.end(), bytes.begin() + kHeaderBytes))
    throw std::runtime_error("cache length/complete primorial mismatch");
  if (sealed) {
    if (static_cast<const void*>(&sealed->bytes()) != static_cast<const void*>(&bytes))
      throw std::runtime_error("verified lease byte ownership mismatch");
  } else {
    const auto digest = detail::sha256(bytes.data(), bytes.size() - Digest{}.size());
    if (!std::equal(digest.begin(), digest.end(), bytes.end() - digest.size()))
      throw std::runtime_error("cache SHA-256 mismatch");
  }
  Basis basis;
  basis.first = key.first;
  basis.all_primes.resize(static_cast<std::size_t>(count));
  if (inverse_headroom > basis.inverses.max_size() -
      static_cast<std::size_t>(inverse_count))
    throw std::length_error("decoded inverse headroom overflow");
  basis.inverses.reserve(static_cast<std::size_t>(inverse_count) +
                         inverse_headroom);
  basis.inverses.resize(static_cast<std::size_t>(inverse_count));
  auto cursor = kHeaderBytes + key.primorial.size();
  if (sealed) {
#if !defined(_M_X64) && !defined(__x86_64__)
#error "R184 native array copy is qualified for little-endian x64 only"
#endif
    static_assert(sizeof(std::uint32_t)==4U,"R184 native word layout");
    // Issuer has run this very decoder's complete SHA/order/range checks on
    // this pinned immutable file. Per-call key/length/primorial remain above;
    // seed is still checked against the actual caller, not assumed identical.
    for (std::size_t i=0;i<seed.size();++i)
      if (detail::get32(bytes,cursor+4U*i)!=seed[i])
        throw std::runtime_error("leased basis caller seed mismatch");
    std::memcpy(basis.all_primes.data(),bytes.data()+cursor,4U*count);
    cursor+=4U*count;
    std::memcpy(basis.inverses.data(),bytes.data()+cursor,4U*inverse_count);
    return basis;
  }
  std::uint32_t previous = 0;
  for (std::size_t i = 0; i < count; ++i, cursor += 4U) {
    const auto prime = detail::get32(bytes, cursor);
    if (prime <= previous || prime > key.limit || (i == 0 ? prime != 2U : !(prime & 1U)) ||
        (i < seed.size() && prime != seed[i]))
      throw std::runtime_error("cache ordered-prime shape mismatch");
    basis.all_primes[i] = prime;
    previous = prime;
  }
  for (std::size_t i = 0; i < inverse_count; ++i, cursor += 4U) {
    const auto inverse = detail::get32(bytes, cursor);
    if (inverse == 0 || inverse >= basis.all_primes[key.first + i])
      throw std::runtime_error("cache inverse outside canonical range");
    basis.inverses[i] = inverse;
  }
  return basis;
}

#ifdef RIECOIN_RLOW_PRIME_ROOT_CACHE_CANONICAL_FIRST
struct Integer {
  mpz_t value;
  Integer() { mpz_init(value); }
  ~Integer() { mpz_clear(value); }
  Integer(const Integer&) = delete;
  Integer& operator=(const Integer&) = delete;
};

inline void set_prefix_product(mpz_ptr output,
    const std::vector<std::uint32_t>& primes, std::uint32_t first) {
  if (first > primes.size())
    throw std::invalid_argument("canonical cache prime prefix is incomplete");
  mpz_set_ui(output, 1U);
  for (std::uint32_t i = 0U; i < first; ++i)
    mpz_mul_ui(output, output, primes[i]);
}

// Prove that the caller's P is exactly the requested ordered-prime prefix,
// then construct the one immutable P114 cache key used by the certified range.
inline void prepare_canonical_primorial(mpz_ptr canonical,
    std::uint32_t requested_first, mpz_srcptr requested,
    const std::vector<std::uint32_t>& seed) {
  if (requested_first < kCanonicalMinFirst ||
      requested_first > kCanonicalFirst || seed.size() < kCanonicalFirst)
    throw std::invalid_argument(
        "canonical cache accepts exactly covered P109..P114 inputs");
  Integer expected;
  set_prefix_product(expected.value, seed, requested_first);
  if (mpz_cmp(expected.value, requested) != 0)
    throw std::invalid_argument(
        "requested primorial is not its complete ordered-prime prefix");
  set_prefix_product(canonical, seed, kCanonicalFirst);
}

// Canonical P114 inverses exist for p >= prime[114].  Descending to Pn uses
// P114 = Pn*Q, hence Pn^-1 = Q*P114^-1 (mod p).  The one or two primes in
// [n,114) are not in the P114 suffix and are computed directly.
inline void transform_from_canonical(Basis& basis,
    std::uint32_t requested_first, mpz_srcptr requested_primorial,
    Metrics& metrics) {
  if (basis.first != kCanonicalFirst ||
      basis.all_primes.size() < kCanonicalFirst ||
      basis.inverses.size() != basis.all_primes.size() - kCanonicalFirst)
    throw std::runtime_error("canonical P114 basis shape mismatch");
  const std::uint32_t delta = kCanonicalFirst - requested_first;
  const std::size_t canonical_count = basis.inverses.size();
  if (basis.inverses.capacity() < canonical_count + delta)
    throw std::runtime_error(
        "canonical inverse transform would allocate outside bounded headroom");
  std::uint64_t quotient = 1U;
  for (std::uint32_t i = requested_first; i < kCanonicalFirst; ++i) {
    if (basis.all_primes[i] > UINT64_MAX / quotient)
      throw std::overflow_error("canonical primorial quotient exceeds uint64");
    quotient *= basis.all_primes[i];
  }
  const auto begin = Clock::now();
  if (delta != 0U) {
    basis.inverses.resize(canonical_count + delta);
    std::move_backward(basis.inverses.begin(),
                       basis.inverses.begin() + canonical_count,
                       basis.inverses.end());
    for (std::uint32_t i = 0U; i < delta; ++i) {
      const auto p = basis.all_primes[requested_first + i];
      basis.inverses[i] = exact::inverse_mod(
          static_cast<std::uint32_t>(mpz_fdiv_ui(requested_primorial, p)), p);
    }
    for (std::size_t i = delta; i < basis.inverses.size(); ++i) {
      const auto p = basis.all_primes[requested_first + i];
      basis.inverses[i] = static_cast<std::uint32_t>(
          std::uint64_t(basis.inverses[i]) * (quotient % p) % p);
    }
  }
  basis.first = requested_first;
  metrics.route = delta == 0U ? "canonical_p114" : "canonical_p114_delta";
  metrics.requested_first = requested_first;
  metrics.canonical_first = kCanonicalFirst;
  metrics.delta_factors = delta;
  metrics.direct_prefix_inverses = delta;
  metrics.transformed_inverses = delta == 0U ? 0U : canonical_count;
  metrics.quotient = quotient;
  metrics.transform_ms = elapsed_ms(begin);
}

#ifdef RIECOIN_RLOW_CANONICAL_DEVICE_TRANSFORM
static_assert(kCanonicalFirst - kCanonicalMinFirst <= 7U,
              "fixed prefix storage certifies at most the P109..P114 bridge");

// Preserve the canonical inverse suffix instead of moving and multiplying
// ~50.8 million entries on the host.  The GPU consumes this compact bridge and
// independently checks requested_P * effective_inverse == 1 (mod p) for every
// prime before any factor bitmap can reach Q0.
inline void stage_device_transform_from_canonical(Basis& basis,
    std::uint32_t requested_first, mpz_srcptr requested_primorial,
    Metrics& metrics) {
  if (basis.first != kCanonicalFirst ||
      basis.all_primes.size() < kCanonicalFirst ||
      basis.inverses.size() != basis.all_primes.size() - kCanonicalFirst ||
      requested_first < kCanonicalMinFirst ||
      requested_first > kCanonicalFirst || mpz_sgn(requested_primorial) <= 0)
    throw std::runtime_error("canonical device basis shape mismatch");
  const std::uint32_t delta = kCanonicalFirst - requested_first;
  std::uint64_t quotient = 1U, second = 1U;
  for (std::uint32_t i = requested_first; i < kCanonicalFirst; ++i) {
    const auto prime = basis.all_primes[i];
    if (prime <= UINT64_MAX / quotient) quotient *= prime;
    else {
      if (prime > UINT64_MAX / second)
        throw std::overflow_error("canonical device quotient exceeds two factors");
      second *= prime;
    }
  }
  const auto begin = Clock::now();
  basis.requested_first = requested_first;
  basis.prefix_inverse_count = delta;
  basis.inverse_multiplier = quotient;
  basis.inverse_multiplier_second = second;
  basis.prefix_inverses.fill(0U);
  for (std::uint32_t i = 0U; i < delta; ++i) {
    const auto p = basis.all_primes[requested_first + i];
    basis.prefix_inverses[i] = exact::inverse_mod(
        static_cast<std::uint32_t>(mpz_fdiv_ui(requested_primorial, p)), p);
  }
  basis.requested_primorial_words.assign(
      (mpz_sizeinbase(requested_primorial, 2) + 31U) / 32U, 0U);
  std::size_t exported = 0U;
  mpz_export(basis.requested_primorial_words.data(), &exported, -1,
             sizeof(std::uint32_t), 0, 0, requested_primorial);
  if (exported == 0U || exported > basis.requested_primorial_words.size())
    throw std::runtime_error("canonical device primorial export failed");
  basis.requested_primorial_words.resize(exported);
  metrics.route = "canonical_p114_device_fused";
  metrics.requested_first = requested_first;
  metrics.canonical_first = kCanonicalFirst;
  metrics.delta_factors = delta;
  metrics.direct_prefix_inverses = delta;
  metrics.transformed_inverses = basis.inverses.size();
  metrics.complete_verified_inverses = 0U;  // mandatory GPU gate is downstream
  metrics.quotient = quotient;
  metrics.transform_ms = elapsed_ms(begin);
}

inline std::uint32_t effective_inverse_for_test(
    const Basis& basis, std::size_t requested_index) {
  const auto total = static_cast<std::size_t>(basis.prefix_inverse_count) +
                     basis.inverses.size();
  if (requested_index >= total ||
      basis.requested_first + requested_index >= basis.all_primes.size())
    throw std::out_of_range("canonical device inverse index");
  if (requested_index < basis.prefix_inverse_count)
    return basis.prefix_inverses[requested_index];
  const auto p = basis.all_primes[basis.requested_first + requested_index];
  const auto first_product = static_cast<std::uint32_t>(
      std::uint64_t(basis.inverses[
          requested_index - basis.prefix_inverse_count]) *
      (basis.inverse_multiplier % p) % p);
  return static_cast<std::uint32_t>(std::uint64_t(first_product) *
      (basis.inverse_multiplier_second % p) % p);
}
#endif

// Complete, independent final contract: every returned suffix entry must be
// the canonical inverse of the exact requested P residue.  No sample, hash or
// algebraic transform is allowed to stand in for these N identities.
inline void verify_complete(const Basis& basis, mpz_srcptr primorial,
                            Metrics& metrics) {
  if (basis.first > basis.all_primes.size() ||
      basis.inverses.size() != basis.all_primes.size() - basis.first ||
      mpz_sgn(primorial) <= 0)
    throw std::runtime_error("complete inverse verifier shape mismatch");
  const auto begin = Clock::now();
  for (std::size_t i = 0U; i < basis.inverses.size(); ++i) {
    const auto p = basis.all_primes[basis.first + i];
    const auto inverse = basis.inverses[i];
    const auto residue = static_cast<std::uint32_t>(mpz_fdiv_ui(primorial, p));
    if (residue == 0U || inverse == 0U || inverse >= p ||
        std::uint64_t(residue) * inverse % p != 1U)
      throw std::runtime_error("complete transformed inverse mismatch");
  }
  metrics.complete_verified_inverses = basis.inverses.size();
  metrics.complete_verify_ms = elapsed_ms(begin);
}
#endif

// On absent/invalid/unreadable cache: exact stateless rebuild; publication is
// best effort and never a precondition to use those newly computed exact data.
inline Basis acquire_fresh(const std::filesystem::path& directory, std::uint32_t limit,
    std::uint32_t first, mpz_srcptr primorial, const std::vector<std::uint32_t>& seed,
    Metrics& metrics) {
  metrics = Metrics{};
#if defined(RIECOIN_RLOW_PRIME_ROOT_CACHE_CANONICAL_FIRST) && defined(RIECOIN_RLOW_CANONICAL_DEVICE_TRANSFORM)
  // RC2:GENERAL-PRIMORIAL — lower network targets can legitimately select P<109.
  // Rebuild exact inverses instead of applying an out-of-range canonical bridge.
  // The GPU still verifies every inverse against the requested primorial.
  if (first < kCanonicalMinFirst || first > kCanonicalFirst) {
    if (!first || seed.size() < first) throw std::invalid_argument("invalid direct primorial prefix");
    Integer expected; set_prefix_product(expected.value, seed, first);
    if (mpz_cmp(expected.value, primorial)) throw std::invalid_argument("direct primorial prefix mismatch");
    const auto begin = Clock::now();
    auto basis = build(limit, first, primorial);
    basis.requested_first = first; basis.exact_direct_geometry = true;
    basis.requested_primorial_words.resize((mpz_sizeinbase(primorial,2)+31U)/32U);
    std::size_t written=0;
    mpz_export(basis.requested_primorial_words.data(),&written,-1,sizeof(std::uint32_t),0,0,primorial);
    if (!written || written > basis.requested_primorial_words.size()) throw std::runtime_error("direct primorial export failed");
    basis.requested_primorial_words.resize(written);
    metrics.status="exact_direct_rebuild"; metrics.requested_first=first;
    metrics.canonical_first=first; metrics.build_ms=elapsed_ms(begin);
    std::cout << "phase=prime_inverse_direct requested_first=" << first << " verifier=device_complete_before_sieve" << std::endl;
    return basis;
  }
#endif
  std::uint32_t cache_first = first;
  mpz_srcptr cache_primorial = primorial;
  std::size_t inverse_headroom = 0U;
#ifdef RIECOIN_RLOW_PRIME_ROOT_CACHE_CANONICAL_FIRST
  Integer canonical_primorial;
  prepare_canonical_primorial(canonical_primorial.value, first, primorial, seed);
  cache_first = kCanonicalFirst;
  cache_primorial = canonical_primorial.value;
#ifndef RIECOIN_RLOW_CANONICAL_DEVICE_TRANSFORM
  inverse_headroom = kCanonicalFirst - first;
#endif
  metrics.requested_first = first;
  metrics.canonical_first = kCanonicalFirst;
#endif
  Key key;
  std::filesystem::path path;
  bool cacheable = false;
  // RC2:BASIS-PATH — explicit packaged basis is read-only, still fully decoded
  // against the requested key (or matched to the verified Windows file lease).
  const char* packaged_basis = std::getenv("HORIZON_RIECOIN_BASIS");
  const bool read_only_basis = packaged_basis && *packaged_basis;
  auto begin = Clock::now();
  try {
    if (!directory.empty()) {
      key = make_key(limit, cache_first, cache_primorial);
      path = read_only_basis ? std::filesystem::path(packaged_basis)
                            : cache_path(local_directory(directory), key);
      metrics.cache_path = path.u8string();
      cacheable = true;
    }
  } catch (const std::exception&) { /* bounded cache cannot cover this input */ }
  metrics.key_ms = elapsed_ms(begin);
  if (cacheable) {
    Bytes bytes;
    std::optional<r184_lease::VerifiedBytes> sealed;
    begin = Clock::now();
    try {
      sealed = r184_lease::acquire(path);
      if (!sealed) bytes = detail::read_file(path);
      metrics.file_bytes = sealed ? sealed->bytes().size() : bytes.size();
      metrics.status = "rejected";
    } catch (const std::exception&) { metrics.status = "miss_or_unreadable"; }
    metrics.load_ms = elapsed_ms(begin);
    if (sealed || !bytes.empty()) {
      begin = Clock::now();
      try {
        auto basis = sealed ? decode(sealed->bytes(),key,seed,inverse_headroom,&*sealed)
                            : decode(bytes,key,seed,inverse_headroom);
        metrics.verify_ms = elapsed_ms(begin);
        metrics.status = sealed ? "basis_lease_hit" : "hit";
#ifdef RIECOIN_RLOW_PRIME_ROOT_CACHE_CANONICAL_FIRST
        Bytes{}.swap(bytes);
        sealed.reset();
#ifdef RIECOIN_RLOW_CANONICAL_DEVICE_TRANSFORM
        stage_device_transform_from_canonical(basis, first, primorial, metrics);
        std::cout << "phase=prime_inverse_canonical route=" << metrics.route
                  << " requested_first=" << metrics.requested_first
                  << " canonical_first=" << metrics.canonical_first
                  << " delta_factors=" << metrics.delta_factors
                  << " quotient=" << metrics.quotient
                  << " direct_prefix_inverses=" << metrics.direct_prefix_inverses
                  << " device_transformed_inverses="
                  << metrics.transformed_inverses
                  << " host_complete_verified_inverses=0"
                  << " transform_ms=" << metrics.transform_ms
                  << " verifier=device_complete_before_sieve"
                  << std::endl;
#else
        transform_from_canonical(basis, first, primorial, metrics);
        verify_complete(basis, primorial, metrics);
        std::cout << "phase=prime_inverse_canonical route=" << metrics.route
                  << " requested_first=" << metrics.requested_first
                  << " canonical_first=" << metrics.canonical_first
                  << " delta_factors=" << metrics.delta_factors
                  << " quotient=" << metrics.quotient
                  << " direct_prefix_inverses=" << metrics.direct_prefix_inverses
                  << " transformed_inverses=" << metrics.transformed_inverses
                  << " complete_verified_inverses="
                  << metrics.complete_verified_inverses
                  << " transform_ms=" << metrics.transform_ms
                  << " complete_verify_ms=" << metrics.complete_verify_ms
                  << std::endl;
#endif
#endif
        return basis;
      } catch (const std::exception&) {
        metrics.verify_ms = elapsed_ms(begin);
        metrics.status = "rejected";
      }
    }
  }
  begin = Clock::now();
  auto basis = build(limit, cache_first, cache_primorial, inverse_headroom);
  metrics.build_ms = elapsed_ms(begin);
  if (cacheable && !read_only_basis) {
    begin = Clock::now();
    try {
      const auto bytes = encode(key, basis);
      metrics.file_bytes = bytes.size();
      metrics.published = detail::publish(path, bytes);
    } catch (const std::exception&) { /* preserve exact fallback, no cache dependency */ }
    metrics.publish_ms = elapsed_ms(begin);
  }
#ifdef RIECOIN_RLOW_PRIME_ROOT_CACHE_CANONICAL_FIRST
#ifdef RIECOIN_RLOW_CANONICAL_DEVICE_TRANSFORM
  stage_device_transform_from_canonical(basis, first, primorial, metrics);
  std::cout << "phase=prime_inverse_canonical route=" << metrics.route
            << " requested_first=" << metrics.requested_first
            << " canonical_first=" << metrics.canonical_first
            << " delta_factors=" << metrics.delta_factors
            << " quotient=" << metrics.quotient
            << " direct_prefix_inverses=" << metrics.direct_prefix_inverses
            << " device_transformed_inverses=" << metrics.transformed_inverses
            << " host_complete_verified_inverses=0"
            << " transform_ms=" << metrics.transform_ms
            << " verifier=device_complete_before_sieve"
            << std::endl;
#else
  transform_from_canonical(basis, first, primorial, metrics);
  verify_complete(basis, primorial, metrics);
  std::cout << "phase=prime_inverse_canonical route=" << metrics.route
            << " requested_first=" << metrics.requested_first
            << " canonical_first=" << metrics.canonical_first
            << " delta_factors=" << metrics.delta_factors
            << " quotient=" << metrics.quotient
            << " direct_prefix_inverses=" << metrics.direct_prefix_inverses
            << " transformed_inverses=" << metrics.transformed_inverses
            << " complete_verified_inverses="
            << metrics.complete_verified_inverses
            << " transform_ms=" << metrics.transform_ms
            << " complete_verify_ms=" << metrics.complete_verify_ms
            << std::endl;
#endif
#endif
  return basis;
}

// A resident worker owns many consecutive jobs in one process.  Re-reading,
// hashing and decoding the same immutable 46 MiB basis on every epoch would
// recreate the process boundary that residency is meant to remove.  Under the
// opt-in macro below, the first exact acquisition remains unchanged and its
// verified Basis becomes process-owned immutable state.  Later matching epochs
// receive a copy from already verified memory; a geometry/key change fails over
// to the full exact acquisition and replaces the resident entry only after it
// succeeds.  The ordinary one-shot engine retains the original behavior.
inline Basis acquire(const std::filesystem::path& directory, std::uint32_t limit,
    std::uint32_t first, mpz_srcptr primorial,
    const std::vector<std::uint32_t>& seed, Metrics& metrics) {
#ifdef RIECOIN_RLOW_PROCESS_RESIDENT_BASIS
  struct ResidentEntry {
    Key key;
    Basis basis;
    std::string cache_path;
    std::uint64_t file_bytes = 0U;
  };
  static std::mutex mutex;
  static std::optional<ResidentEntry> resident;
  const Key requested = make_key(limit, first, primorial);
  std::lock_guard<std::mutex> lock(mutex);
  if (resident && resident->key.limit == requested.limit &&
      resident->key.first == requested.first &&
      resident->key.primorial == requested.primorial) {
    metrics = Metrics{};
    metrics.status = "resident_hit";
    metrics.cache_path = resident->cache_path;
    metrics.file_bytes = resident->file_bytes;
    return resident->basis;
  }
  auto basis = acquire_fresh(directory, limit, first, primorial, seed, metrics);
  ResidentEntry replacement;
  replacement.key = requested;
  replacement.basis = basis;
  replacement.cache_path = metrics.cache_path;
  replacement.file_bytes = metrics.file_bytes;
  resident = std::move(replacement);
  return basis;
#else
  return acquire_fresh(directory, limit, first, primorial, seed, metrics);
#endif
}

inline exact::RootTable roots_for(const Basis& basis, mpz_srcptr base,
                                 const std::array<std::uint32_t, 7>& offsets) {
  if (basis.first > basis.all_primes.size() ||
      basis.inverses.size() != basis.all_primes.size() - basis.first)
    throw std::invalid_argument("invalid cached basis shape");
  exact::RootTable output;
  output.primes.assign(basis.all_primes.begin() + basis.first, basis.all_primes.end());
  if (output.primes.size() > output.first_factors.max_size() / offsets.size())
    throw std::length_error("prime-root table size overflow");
  output.first_factors.resize(output.primes.size() * offsets.size());
  std::array<std::uint32_t, 6> steps{};
  bool small_gaps = offsets.front() == 0U;
  for (std::size_t m = 1; m < offsets.size(); ++m) {
    const auto gap = offsets[m] - offsets[m - 1U];
    const bool supported = offsets[m] >= offsets[m - 1U] && (gap == 2 || gap == 4 || gap == 6);
    small_gaps = small_gaps && supported;
    if (supported) steps[m - 1U] = gap / 2U - 1U;
  }
  for (std::size_t i = 0; i < output.primes.size(); ++i) {
    const auto p = output.primes[i], inverse = basis.inverses[i];
    const auto b = static_cast<std::uint32_t>(mpz_fdiv_ui(base, p));
    auto* roots = output.first_factors.data() + i * offsets.size();
    if (!small_gaps) {
      for (std::size_t m = 0; m < offsets.size(); ++m) {
        const auto residue = (std::uint64_t(b) + offsets[m]) % p;
        roots[m] = static_cast<std::uint32_t>((residue == 0 ? 0U : p - residue) * inverse % p);
      }
    } else {
      roots[0] = static_cast<std::uint32_t>(std::uint64_t(b == 0 ? 0U : p - b) * inverse % p);
      const auto twice = exact::add_mod(inverse, inverse, p);
      const auto step2 = twice == 0 ? 0U : p - twice;
      const auto step4 = exact::add_mod(step2, step2, p);
      const std::array<std::uint32_t, 3> delta{{step2, step4, exact::add_mod(step2, step4, p)}};
      for (std::size_t m = 1; m < offsets.size(); ++m)
        roots[m] = exact::add_mod(roots[m - 1U], delta[steps[m - 1U]], p);
    }
  }
  return output;
}
} // namespace riecoin_prime_root_cache

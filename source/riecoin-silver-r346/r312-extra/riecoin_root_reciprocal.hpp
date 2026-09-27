#pragma once
#include <cstdint>
#if defined(__CUDACC__)
#define RIE_ROOT_HD __device__ __forceinline__
#else
#define RIE_ROOT_HD inline
#endif

namespace riecoin_root_reciprocal {
// For p>=2, mu=floor((2^64-1)/p). Then q=high64(a*mu) never
// exceeds floor(a/p) and is at most one below it, for EVERY uint64 a.
// Thus a-q*p is exact (no underflow), below 2p; one subtraction suffices.
// Do not narrow a: this also covers the live cursor beyond 2^32.
template<class HighProduct>
RIE_ROOT_HD std::uint32_t reduce(std::uint64_t a, std::uint32_t p,
                               std::uint64_t mu, HighProduct high) {
    const std::uint64_t q = high(a, mu);
    std::uint64_t r = a - q * p;
    if (r >= p) r -= p;
    return static_cast<std::uint32_t>(r);
}

template<class HighProduct>
RIE_ROOT_HD std::uint32_t word_remainder(const std::uint32_t* words,
                                       std::uint32_t count, std::uint32_t p,
                                       std::uint64_t mu, HighProduct high) {
    std::uint32_t r = 0;
    while (count != 0) r = reduce((std::uint64_t(r) << 32) | words[--count], p, mu, high);
    return r;
}
}
#undef RIE_ROOT_HD

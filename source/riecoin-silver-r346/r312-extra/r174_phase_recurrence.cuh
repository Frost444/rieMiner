#pragma once
#include <stdint.h>

// Isolated capability test. Not connected to the production miner.
// Contract: p >= 2; phase and step are canonical residues in [0,p).
// The owner must bind phase to generation, root table, and expected origin.
// step = span % p is computed once for a fixed span, not on each advance.
#ifdef __CUDACC__
#define R174_HD __host__ __device__ __forceinline__
#else
#define R174_HD inline
#endif
namespace r174 {
R174_HD uint32_t advance(uint32_t phase, uint32_t step, uint32_t p) {
    // The addition cannot overflow: in this branch phase < step implies
    // phase + (p-step) < p <= UINT32_MAX.
    return phase >= step ? phase-step : phase+(p-step);
}
R174_HD uint32_t first(uint32_t root, uint32_t p, uint64_t origin) {
    const uint32_t rem = static_cast<uint32_t>(origin % p);
    return advance(root, rem, p);
}
// A95's default shared frontier has span=2^28 and p>2^25.
// No extra step array is needed in that geometry. Precondition checks belong
// to the owner before launch; other shapes keep the general exact path.
R174_HD uint32_t advance_default_frontier(uint32_t phase, uint32_t p, uint32_t span) {
    // Preconditions: 2^25 < p <= UINT32_MAX, 0 < span <= 2^28.
    uint32_t step = span;
    if (p <= step) {
        // Here p <= 2^28, so 4*p and 2*p cannot overflow. span/p < 8.
        const uint32_t p4 = p << 2;
        const uint32_t p2 = p << 1;
        if (step >= p4) step -= p4;
        if (step >= p2) step -= p2;
        if (step >= p) step -= p;
    }
    return advance(phase, step, p);
}
}
#undef R174_HD

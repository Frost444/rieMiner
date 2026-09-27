#pragma once
#include <cstdint>
#if defined(__CUDACC__)
#define R209_HD __host__ __device__
#else
#define R209_HD
#endif
namespace r209_event {
constexpr uint32_t shift=28;
constexpr uint64_t span=uint64_t(1)<<shift, mask=span-1;
R209_HD inline uint64_t encode(uint64_t period,uint64_t hit){return (period<<shift)|(hit&mask);}
R209_HD inline uint64_t period(uint64_t event){return event>>shift;}
R209_HD inline uint64_t hit(uint64_t event){return event&mask;}
}
#undef R209_HD

#pragma once
#include <cstdint>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef __CUDACC__
#define R217_HD __host__ __device__
#else
#define R217_HD
#endif
namespace r217_native {
R217_HD inline uint64_t words(const uint32_t* digits,uint32_t n,uint64_t p){
 uint64_t r=0;for(uint32_t i=n;i;--i)r=((r<<31)|digits[i-1])%p;return r;
}
// Contract: a,b < p < 2^33. Product has at most 66 bits. Remove its
// two low bits, perform ONE native uint64 modulo, then restore those bits.
// The last remainder is <4p, so two exact subtractions finish the reduction.
// Independent of the producer's reciprocal/reduce66 and floating quotient.
R217_HD inline uint64_t product(uint64_t a,uint64_t b,uint64_t p){
 const uint64_t lo=a*b;uint64_t hi;
#ifdef __CUDA_ARCH__
 hi=__umul64hi(a,b);
#elif defined(_MSC_VER)
 _umul128(a,b,&hi);
#else
 hi=static_cast<uint64_t>((static_cast<unsigned __int128>(a)*b)>>64);
#endif
 uint64_t r=(((hi<<62)|(lo>>2))%p)*4+(lo&3);
 if(r>=2*p)r-=2*p;if(r>=p)r-=p;return r;
}
R217_HD inline uint64_t add(uint64_t a,uint64_t b,uint64_t p){a+=b;return a>=p?a-p:a;}
}
#undef R217_HD

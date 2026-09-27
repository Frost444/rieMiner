// Isolated device mapping of R229. Compilation does not launch a CUDA context.
// No production hook. Caller must bind exact P114/Pjob/O/source/generation and
// retain the sealed atlas lease until all jobs finish. A failed guard forbids
// admission; this module does not itself admit candidates or submit shares.
#define R207_LIBRARY
#include "r207_reduce33.cu"
#include "r217_native31.hpp"
namespace r229_gpu {
constexpr uint64_t low=4000000000ull,high=8000000000ull,offset=114023297140211ull;
__device__ uint64_t words32(const uint32_t* w,uint32_t count,uint64_t p,uint64_t mu){
 uint64_t r=0;for(uint32_t j=count;j;--j)r=reduce66((r<<32)|w[j-1],r>>32,p,mu);return r;
}
// Build once per immutable anchor. A complete warp writes one high-bit word,
// including deterministic zero padding; no atomic packing or stale OR state.
__global__ void make_anchor(const uint32_t* delta,uint32_t n,
 const uint32_t* anchor,uint32_t words,uint32_t* inv_low,uint32_t* inv_high,uint32_t* errors){
 const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;uint64_t iv=0;
 if(i<n){
  const uint64_t p=low+delta[i];
  if(p<=low||p>high||!(p&1)){atomicOr(errors,1u);}
  else {const uint64_t mu=UINT64_MAX/p;iv=inverse64(words32(anchor,words,p,mu),p);if(!iv)atomicOr(errors,2u);}
  inv_low[i]=uint32_t(iv);
 }
 const unsigned high_bits=__ballot_sync(0xffffffffu,(iv>>32)!=0);
 if((threadIdx.x%32)==0&&i<n)inv_high[i/32]=high_bits;
}
// Expensive P reduction and extended GCD disappear from the recurring path.
// Two short reductions remain: A and S=P114/Pjob, plus modular products.
__global__ void prepare_reused(const uint32_t* delta,uint32_t n,
 const uint32_t* inv_low,const uint32_t* inv_high,
 const uint32_t* quotient,uint32_t quotient_words,
 const uint32_t* basis_ratio,uint32_t ratio_words,
 uint64_t* inverse,uint64_t* roots,uint32_t* errors){
 const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 const uint64_t p=low+delta[i];
 if(p<=low||p>high||!(p&1)){inverse[i]=roots[i]=0;atomicOr(errors,1u);return;}
 const uint64_t anchor_iv=uint64_t(inv_low[i])|(uint64_t((inv_high[i/32]>>(i%32))&1u)<<32);
 if(!anchor_iv||anchor_iv>=p){inverse[i]=roots[i]=0;atomicOr(errors,2u);return;}
 const uint64_t mu=UINT64_MAX/p,s=words32(basis_ratio,ratio_words,p,mu);
 const uint64_t iv=reduce66(s*anchor_iv,__umul64hi(s,anchor_iv),p,mu);
 if(!iv){inverse[i]=roots[i]=0;atomicOr(errors,4u);return;}
 const uint64_t a=words32(quotient,quotient_words,p,mu),o=reduce66(offset,0,p,mu);
 const uint64_t product=o*iv,sum=product+a;
 const uint64_t x=reduce66(sum,__umul64hi(o,iv)+uint64_t(sum<product),p,mu);
 inverse[i]=iv;roots[i]=x?p-x:0;
}
// Closed-test oracle: full job P and B, native31 independent of reciprocal
// producer and the transported anchor. Kept separate to expose its real cost.
__global__ void verify_full_job(const uint32_t* delta,uint32_t n,
 const uint32_t* P,uint32_t p_words,const uint32_t* B,uint32_t b_words,
 const uint64_t* inverse,const uint64_t* roots,uint32_t* errors){
 const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 const uint64_t p=low+delta[i];if(p<=low||p>high||!(p&1)){atomicOr(errors,1u);return;}
 const uint64_t pm=r217_native::words(P,p_words,p),bm=r217_native::words(B,b_words,p),iv=inverse[i];
 if(!iv||iv>=p||r217_native::product(pm,iv,p)!=1){atomicOr(errors,2u);return;}
 const uint64_t expected=r217_native::product(bm?p-bm:0,iv,p);
 if(expected!=roots[i])atomicOr(errors,4u);
}
} // namespace r229_gpu

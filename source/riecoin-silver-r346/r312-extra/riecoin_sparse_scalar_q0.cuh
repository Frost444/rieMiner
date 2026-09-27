#pragma once

#include "riecoin_sparse_scalar_montgomery.cuh"

// Full Euler/Jacobi Q0 for the explicit sparse P31 family. One thread owns
// one candidate; no candidate is rejected without the unchanged exact result.
namespace riecoin_sparse_scalar_q0 {
__device__ __forceinline__ bool equals_one(const uint32_t (&a)[40]) {
  if(a[0]!=1u)return false;
  for(unsigned i=1;i<40u;++i)if(a[i])return false;
  return true;
}
__device__ __forceinline__ bool equals_minus_one(const uint32_t (&a)[40],
    const uint32_t (&n)[40]) {
  uint64_t borrow=1u;
  for(unsigned i=0;i<40u;++i){
    const uint32_t expected=uint32_t(uint64_t(n[i])-borrow);
    borrow=uint64_t(n[i])<borrow;
    if(a[i]!=expected)return false;
  }
  return true;
}
__global__ void segmented(const cgbn_mem_t<1280u>* candidates,
    uint8_t* verdicts,const uint32_t* counts,uint32_t macros,uint32_t capacity) {
  __shared__ uint32_t mirror[40u*128u];
  const uint32_t slot=blockIdx.x*blockDim.x+threadIdx.x;
  const uint32_t macro=slot/capacity;
  if(macro>=macros)return;
  const uint32_t local=slot-macro*capacity;
  if(local>=counts[macro]){verdicts[slot]=0u;return;}
  uint32_t n[40],a[40];
  #pragma unroll
  for(unsigned i=0;i<40u;++i){
    const uint32_t value=candidates[slot]._limbs[i];
    if((i>=7u&&i<26u)||i>=36u){if(value)asm volatile("trap;");n[i]=0u;}
    else n[i]=value;
  }
  uint32_t inverse=1u;
  for(unsigned i=0;i<5u;++i)inverse*=2u-n[0]*inverse;
  inverse=0u-inverse;
  riecoin_sparse_scalar::euler_base2(a,n,candidates[slot]._limbs,inverse,mirror);
  const uint32_t mod8=n[0]&7u;
  verdicts[slot]=((mod8==1u||mod8==7u)?equals_one(a):equals_minus_one(a,n))?1u:0u;
}
}

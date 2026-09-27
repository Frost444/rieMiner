#pragma once

// R136 hardware construction probe. R=2^1280; modulus words [7,26) and
// [36,40) are proved zero by host admission. No production dispatch yet.
// One thread owns one candidate so eliminated coefficient products remove
// instructions, rather than merely masking cooperative lanes in a warp.
namespace riecoin_sparse_scalar {
constexpr unsigned limbs=40;
__device__ __forceinline__ bool ge(const uint32_t (&a)[limbs],const uint32_t (&n)[limbs]) {
  #pragma unroll
  for(int i=limbs-1;i>=0;--i){if(a[i]!=n[i])return a[i]>n[i];}
  return true;
}
__device__ __forceinline__ void subtract(uint32_t (&a)[limbs],const uint32_t (&n)[limbs]) {
  uint64_t borrow=0;
  #pragma unroll
  for(unsigned j=0;j<limbs;++j){const uint64_t b=uint64_t(n[j])+borrow;const uint32_t old=a[j];a[j]=uint32_t(uint64_t(old)-b);borrow=uint64_t(old)<b;}
}
__device__ __forceinline__ void row(uint32_t (&t)[limbs+2],
    const uint32_t (&a)[limbs],const uint32_t (&n)[limbs],uint32_t np0,uint32_t ai) {
  uint64_t carry=0;
  #pragma unroll
  for(unsigned j=0;j<limbs;++j){const uint64_t s=uint64_t(ai)*a[j]+t[j]+carry;t[j]=uint32_t(s);carry=s>>32u;}
  uint64_t s=uint64_t(t[limbs])+carry;t[limbs]=uint32_t(s);t[limbs+1]+=uint32_t(s>>32u);
  const uint32_t q=t[0]*np0;
  carry=0;
  #pragma unroll
  for(unsigned j=0;j<limbs;++j){
    if((j>=7u && j<26u)||j>=36u){
      if(carry){s=uint64_t(t[j])+carry;t[j]=uint32_t(s);carry=s>>32u;}
    }else{s=uint64_t(q)*n[j]+t[j]+carry;t[j]=uint32_t(s);carry=s>>32u;}
  }
  s=uint64_t(t[limbs])+carry;t[limbs]=uint32_t(s);t[limbs+1]+=uint32_t(s>>32u);
  if(t[0]!=0u)asm volatile("trap;");
  #pragma unroll
  for(unsigned j=0;j<=limbs;++j)t[j]=t[j+1u];
  t[limbs+1]=0u;
}
template<bool Normal=false>
__device__ __forceinline__ void operation(uint32_t (&a)[limbs],
    const uint32_t (&n)[limbs],uint32_t np0,uint32_t* mirror) {
  uint32_t t[limbs+2]={};
  // A shared, bank-coalesced mirror makes the sole dynamic operand lookup
  // explicit. All inner coefficients stay statically indexed in registers.
  // Each thread reads only its own writes: no block barrier is required.
  if constexpr(!Normal){
    #pragma unroll
    for(unsigned j=0;j<limbs;++j)mirror[j*blockDim.x+threadIdx.x]=a[j];
  }
  // Recursive full unrolling of all 40 rows caused 7 KiB of spills/thread.
  // Reuse one row's live state rather than duplicating the instruction graph.
  #pragma unroll 1
  for(unsigned i=0;i<limbs;++i){
    const uint32_t ai=Normal?(i==0u?1u:0u):mirror[i*blockDim.x+threadIdx.x];
    row(t,a,n,np0,ai);
  }
  // Admission and x*x<nR imply REDC<2n; n<2^1152<R/2.
  if(t[limbs] || t[limbs+1])asm volatile("trap;");
  #pragma unroll
  for(unsigned j=0;j<limbs;++j)a[j]=t[j];
  if(ge(a,n))subtract(a,n);
}

__device__ __forceinline__ void shift(uint32_t (&a)[limbs],unsigned bits) {
  // Input is canonical before each window shift, n<2^1136 and bits<=63.
  // Thus the redundant value is <2^1199, with no dropped 1280-bit limb.
  if(bits>=32u){
    #pragma unroll
    for(int j=limbs-1;j>=0;--j)a[j]=j?a[j-1]:0u;
  }
  bits&=31u;
  if(bits){
    #pragma unroll
    for(int j=limbs-1;j>=0;--j)a[j]=(a[j]<<bits)|(j?a[j-1]>>(32u-bits):0u);
  }
}

// Full Euler base-2 result, not a square-only surrogate. Every exponent bit
// of (n-1)/2=n>>1 is consumed, and the final result is in the normal domain.
// Global read-only source avoids dynamically indexing the register n[] array.
__device__ __forceinline__ void euler_base2(uint32_t (&a)[limbs],
    const uint32_t (&n)[limbs],const uint32_t* source,uint32_t np0,uint32_t* mirror) {
  unsigned bits=0u;
  #pragma unroll
  for(int j=limbs-1;j>=0;--j)if(bits==0u && n[j])bits=unsigned(j)*32u+32u-__clz(n[j]);
  if(bits<2u || bits>1136u || !(n[0]&1u))asm volatile("trap;");
  #pragma unroll
  for(unsigned j=0;j<limbs;++j)a[j]=j==(bits-1u)/32u?uint32_t(1u)<<((bits-1u)&31u):0u;
  // Start at 2^(bits-1)<n, then exactly form 2^1280 mod n without division.
  for(unsigned k=bits-1u;k<1280u;++k){shift(a,1u);if(ge(a,n))subtract(a,n);}
  bool first=true;
  for(unsigned remaining=bits-1u;remaining;){
    const unsigned width=remaining<6u?remaining:6u;
    const unsigned low=remaining-width+1u,index=low>>5u,offset=low&31u;
    uint32_t value=source[index]>>offset;
    if(offset+width>32u)value|=source[index+1u]<<(32u-offset);
    value&=(1u<<width)-1u;
    if(!first){
      #pragma unroll 1
      for(unsigned k=0;k<width;++k)operation(a,n,np0,mirror);
    }
    // The next square is valid even for a redundant window value:
    // a<n*2^63 => a^2<n*2^1280 for the admitted n<2^1136.
    shift(a,value);
    first=false;
    remaining-=width;
  }
  operation<true>(a,n,np0,mirror);
}
}

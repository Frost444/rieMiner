#pragma once
#include <cstdint>

// A19: retain all 40 limbs and the original Euler-Jacobi predicate. Only
// the cooperative group mapping changes. No 1152-bit truncation or filtering.
// R=2^1280; the existing lazy-window headroom proof is unchanged.
namespace riecoin_lazy_base2_window {
constexpr uint32_t kStorageBits = 1280u;
constexpr uint32_t kMaximumWindowBits = 6u;
__host__ __device__ constexpr uint32_t window_bits(uint32_t leading) {
  uint32_t width=0;
  for(uint32_t k=1;k<=kMaximumWindowBits;++k)
    if(leading >= (1u<<(k+1u))) width=k;
  return width;
}

__device__ __forceinline__ bool evaluate(
    const cgbn_mem_t<1280u>* input, uint32_t slot,
    uint32_t* normal=nullptr, uint8_t* used_window=nullptr) {
  using context=cgbn_context_t<4u,riecoin_cuda::PrpCgbnParameters>;
  using environment=cgbn_env_t<context,1280u>;
  context ctx(cgbn_no_checks,nullptr,slot);
  environment env(ctx);
  typename environment::cgbn_t n,x;
  const uint32_t lane=threadIdx.x&3u;
  if(used_window && lane==0) used_window[slot]=0;
  cgbn_load(env,n,const_cast<cgbn_mem_t<1280u>*>(input+slot));
  const uint32_t leading=cgbn_clz(env,n),bits=1280u-leading;
  if(bits<3 || !(cgbn_get_ui32(env,n)&1u)) return false;

  // Reuse Silver's exact Montgomery-one construction, generalized to full
  // width. Start x=2^(bits-1)<n, then double modulo n through exponent 1280.
  // double_modulo handles carry as well as comparison: valid even at 1280 bits.
  // This avoids relying on a division dispatcher for ten limbs per thread.
  cgbn_set_ui32(env,x,1u);
  cgbn_shift_left(env,x,x,bits-1u);
  for(uint32_t e=bits-1u;e<1280u;++e)
    riecoin_cuda::double_modulo(env,x,n);
  const uint32_t n0=cgbn_get_ui32(env,n);
  uint32_t inverse=1u;
  for(uint32_t i=0;i<5;++i) inverse*=2u-n0*inverse;
  inverse=0u-inverse;

  const uint32_t width=window_bits(leading);
  if(width==0) {
    // Exact full-width fallback stays in the SAME four-thread group. Calling
    // the original eight-thread evaluator here would violate group ownership.
    uint32_t cached=UINT32_MAX,word=0;
    for(int32_t bit=static_cast<int32_t>(bits)-2;bit>=0;--bit) {
      cgbn_mont_sqr(env,x,x,n,inverse);
      const uint32_t source=static_cast<uint32_t>(bit)+1u;
      if((source>>5u)!=cached) {
        cached=source>>5u;
        word=cgbn_extract_bits_ui32(env,n,cached*32u,32u);
      }
      if((word>>(source&31u))&1u) riecoin_cuda::double_modulo(env,x,n);
    }
  } else {
    uint32_t remaining=bits-1u;
    bool first=true;
    while(remaining) {
      const uint32_t take=remaining<width?remaining:width;
      if(!first) for(uint32_t i=0;i<take;++i) cgbn_mont_sqr(env,x,x,n,inverse);
      const uint32_t value=cgbn_extract_bits_ui32(env,n,remaining-take+1u,take);
      cgbn_shift_left(env,x,x,value);
      remaining-=take;
      first=false;
    }
  }
  cgbn_mont2bn(env,x,x,n,inverse);
  if(normal) cgbn_store(env,reinterpret_cast<cgbn_mem_t<1280u>*>(normal)+slot,x);
  if(used_window && lane==0) used_window[slot]=static_cast<uint8_t>(width);
  return riecoin_cuda::passes_euler_jacobi(env,x,n);
}

__global__ void oracle_kernel(const cgbn_mem_t<1280u>* input,uint8_t* verdict,
    uint8_t* used_window,uint32_t* normal,uint32_t count) {
  const uint32_t thread=blockIdx.x*blockDim.x+threadIdx.x,slot=thread/4u;
  if(slot>=count) return;
  const bool pass=evaluate(input,slot,normal,used_window);
  if(!(thread&3u)) verdict[slot]=pass?1u:0u;
}
// A23 tests the register/latency-hiding boundary independently of A21 packing.
// Eight 128-thread blocks fit at <=64 registers/thread on sm_86; any spills
// and the full useful wall time must be measured, never inferred from occupancy.
#ifdef ASTRA_R249_DENSE_Q0
__device__ uint32_t r249_slots[64u*4096u],r249_total=0,r249_error=0;
__global__ void prepare_dense_q0(uint8_t* verdict,const uint32_t* counts,uint32_t macros,uint32_t capacity){
 const uint32_t m=blockIdx.x;if(m>=macros)return;
 if(macros>64u||capacity>4096u){if(!threadIdx.x)atomicOr(&r249_error,1u);return;}
 __shared__ uint32_t begin,count;
 if(!threadIdx.x){begin=0;for(uint32_t j=0;j<m;++j){if(counts[j]>capacity)atomicOr(&r249_error,2u);begin+=min(counts[j],capacity);}
  count=min(counts[m],capacity);if(counts[m]>capacity)atomicOr(&r249_error,2u);
  if(m+1u==macros)r249_total=begin+count;
 }
 __syncthreads();
 for(uint32_t j=threadIdx.x;j<capacity;j+=blockDim.x){verdict[m*capacity+j]=0;if(j<count)r249_slots[begin+j]=m*capacity+j;}
}
#endif
#ifdef RIECOIN_RLOW_Q0_EIGHT_BLOCKS
__global__ __launch_bounds__(128,8)
#else
__global__
#endif
void segmented_q0_kernel(const cgbn_mem_t<1280u>* input,
    uint8_t* verdict,const uint32_t* counts,uint32_t macros,uint32_t capacity) {
#ifdef ASTRA_R249_DENSE_Q0
  const uint32_t thread=blockIdx.x*blockDim.x+threadIdx.x,index=thread/4u;
  if(r249_error||index>=r249_total)return;
  const uint32_t slot=r249_slots[index];const bool pass=evaluate(input,slot);
  if(!(thread&3u))verdict[slot]=pass?1u:0u;
#elif defined(RIECOIN_RLOW_PACKED_DISPATCH)
  // Logical CTA tickets cover only occupied groups, not each family's full
  // capacity. No host count readback and no global atomic work queue.
  __shared__ uint32_t bounded[64],prefix[65];
  if(macros>64u)return; // Host geometry contract is 16..64, checked at setup.
  if(threadIdx.x<macros)bounded[threadIdx.x]=min(counts[threadIdx.x],capacity);
  __syncthreads();
  const uint32_t per_block=blockDim.x/4u;
  if(threadIdx.x==0u) {
    prefix[0]=0;
    for(uint32_t m=0;m<macros;++m)prefix[m+1]=prefix[m]+(bounded[m]+per_block-1u)/per_block;
  }
  __syncthreads();
  // Clear only dead slots. These writes are disjoint from every evaluator,
  // even when CTAs start at different times. No missing global barrier.
  for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<macros*capacity;i+=gridDim.x*blockDim.x)
    if(i%capacity>=bounded[i/capacity])verdict[i]=0;
  for(uint32_t ticket=blockIdx.x;ticket<prefix[macros];ticket+=gridDim.x) {
    uint32_t lo=0,hi=macros;
    while(lo+1u<hi){const uint32_t mid=(lo+hi)/2u;if(prefix[mid]<=ticket)lo=mid;else hi=mid;}
    const uint32_t local=(ticket-prefix[lo])*per_block+threadIdx.x/4u;
    if(local<bounded[lo]) {
      const uint32_t slot=lo*capacity+local;
      const bool pass=evaluate(input,slot);
      if(!(threadIdx.x&3u))verdict[slot]=pass?1u:0u;
    }
  }
#else
  const uint32_t thread=blockIdx.x*blockDim.x+threadIdx.x,slot=thread/4u;
  const uint32_t macro=slot/capacity;
  if(macro>=macros) return;
  const bool pass=slot-macro*capacity<counts[macro] && evaluate(input,slot);
  if(!(thread&3u)) verdict[slot]=pass?1u:0u;
#endif
}
} // namespace riecoin_lazy_base2_window

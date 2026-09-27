#pragma once
// R209 exact33-bit divisor calendar; full-miner experimental composition.
// One CTA exclusively owns each bounded page pool and its 31 future buckets.
#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <limits>
#include "r209_event_relation.hpp"

namespace r209cal {
constexpr uint32_t lanes=256, buckets=31, nil=UINT32_MAX;
struct Stats { unsigned long long admitted=0, processed=0, retired=0; };
struct Device {
 uint64_t* records=nullptr;
 uint32_t *next=nullptr,*used=nullptr,*heads=nullptr,*tails=nullptr,*free=nullptr,*error=nullptr;
 Stats* stats=nullptr;
 uint32_t owners=0,pages=0;
};
using Sort=cub::BlockRadixSort<uint32_t,256,1,uint64_t,5>;
struct Shared {
 typename Sort::TempStorage sort;
 uint32_t heads[buckets],tails[buckets],free;
 uint32_t keys[lanes],starts[buckets],ends[buckets];
 uint32_t page0[buckets],page1[buckets],offset[buckets],first[buckets];
 uint32_t input, count, failed;
};
__device__ inline uint32_t bucket(uint32_t current,uint32_t delta){
 const uint32_t v=current+delta;return v>=buckets?v-buckets:v;
}
__device__ inline uint32_t take(Device d,Shared& s){
 const uint32_t p=s.free;
 if(p==nil){s.failed=1;atomicOr(d.error,1u);return nil;}
 s.free=d.next[p];d.next[p]=nil;d.used[p]=0;return p;
}
// Collective append: all lanes participate, invalid records use key31.
__device__ inline bool append(Device d,Shared& s,uint32_t current,uint32_t key,uint64_t value){
 uint32_t keys[1]={key};uint64_t values[1]={value};
 Sort(s.sort).Sort(keys,values,0,5);
 __syncthreads();
 const uint32_t tid=threadIdx.x;key=keys[0];value=values[0];
 s.keys[tid]=key;
 if(tid<buckets){s.starts[tid]=nil;s.ends[tid]=0;s.page0[tid]=nil;s.page1[tid]=nil;}
 if(tid==0)s.failed=0;
 __syncthreads();
 if(key<buckets){
  if(tid==0||s.keys[tid-1]!=key)s.starts[key]=tid;
  if(tid==lanes-1||s.keys[tid+1]!=key)s.ends[key]=tid+1;
 }
 __syncthreads();
 if(tid==0){
  for(uint32_t q=0;q<buckets;++q){
   if(s.starts[q]==nil)continue;
   const uint32_t n=s.ends[q]-s.starts[q],b=bucket(current,q);
   uint32_t page=s.tails[b];
   if(page==nil||d.used[page]==lanes){
    const uint32_t fresh=take(d,s);if(s.failed)break;
    if(page==nil)s.heads[b]=fresh;else d.next[page]=fresh;
    s.tails[b]=fresh;page=fresh;
   }
   s.page0[q]=page;s.offset[q]=d.used[page];
   const uint32_t space=lanes-d.used[page],first=n<space?n:space;
   s.first[q]=first;d.used[page]+=first;
   if(first<n){
    const uint32_t fresh=take(d,s);if(s.failed)break;
    d.next[page]=fresh;s.tails[b]=fresh;s.page1[q]=fresh;d.used[fresh]=n-first;
   }
  }
 }
 __syncthreads();
 if(!s.failed&&key<buckets){
  const uint32_t rank=tid-s.starts[key];
  const uint32_t page=rank<s.first[key]?s.page0[key]:s.page1[key];
  const uint32_t offset=rank<s.first[key]?s.offset[key]+rank:rank-s.first[key];
  d.records[uint64_t(page)*lanes+offset]=value;
 }
 __syncthreads();return !s.failed;
}

__global__ void initialize(Device d,const uint32_t* primes,const uint64_t* root0,
 const uint64_t* inverse,uint32_t count,uint32_t gap,uint64_t origin,
 uint64_t remaining,uint32_t shift,uint64_t begin,uint64_t end){
 __shared__ Shared s;
 const uint32_t tid=threadIdx.x,owner=blockIdx.x,base=owner*d.pages;
 if(begin==0){
  for(uint32_t p=tid;p<d.pages;p+=lanes){d.next[base+p]=p+1<d.pages?base+p+1:nil;d.used[base+p]=0;}
  if(tid<buckets){s.heads[tid]=nil;s.tails[tid]=nil;}
  if(tid==0){s.free=base;d.stats[owner]=Stats{};}
 }else{
  if(tid<buckets){s.heads[tid]=d.heads[owner*buckets+tid];s.tails[tid]=d.tails[owner*buckets+tid];}
  if(tid==0)s.free=d.free[owner];
 }
 __syncthreads();
 const uint64_t records=uint64_t(count)*2<end?uint64_t(count)*2:end,step=uint64_t(d.owners)*lanes;
 for(uint64_t tile=begin+uint64_t(owner)*lanes;tile<records;tile+=step){
  const uint64_t eid=tile+tid;uint32_t key=31;uint64_t value=0;
  if(eid<records){
   const uint32_t i=uint32_t(eid/2);const uint64_t p=4000000000ull+primes[i];
   if(p<2||uint64_t(p)>(uint64_t(30)<<shift)||root0[i]>=p||inverse[i]>=p){atomicOr(d.error,2u);}
   else{
    uint64_t r=root0[i];
    if(eid&1){const uint64_t delta=uint64_t(gap)*inverse[i]%p;r=r>=delta?r-delta:r+p-delta;}
    const uint64_t o=origin%p;r=r>=o?r-o:r+p-o;
    if(uint64_t(r)<remaining){key=uint32_t(r>>shift);value=r209_event::encode(p,r);}
   }
  }
  const uint32_t live=__syncthreads_count(key<31);
  if(tid==0)d.stats[owner].admitted+=live;
  if(!append(d,s,0,key,value))break;
 }
 if(tid<buckets){d.heads[owner*buckets+tid]=s.heads[tid];d.tails[owner*buckets+tid]=s.tails[tid];}
 if(tid==0)d.free[owner]=s.free;
}

__global__ void advance(Device d,uint32_t* bits,const uint32_t* occupied,
 uint32_t current,uint32_t shift,uint64_t remaining){
 __shared__ Shared s;
 const uint32_t tid=threadIdx.x,owner=blockIdx.x,span=uint32_t(1)<<shift;
 if(tid<buckets){s.heads[tid]=d.heads[owner*buckets+tid];s.tails[tid]=d.tails[owner*buckets+tid];}
 if(tid==0)s.free=d.free[owner];
 __syncthreads();
 // Entry errors are uniform for the CTA. Host gate rejects the whole mutation.
 if(tid==0)s.failed=*d.error;
 __syncthreads();if(s.failed)return;
 while(true){
  if(tid==0){s.input=s.heads[current];s.count=s.input==nil?0:d.used[s.input];}
  __syncthreads();if(s.input==nil)break;
  const bool valid=tid<s.count;const uint64_t value=valid?d.records[uint64_t(s.input)*lanes+tid]:0;
  __syncthreads();
  // All payload is in registers before the page is returned to this owner.
  if(tid==0){
   const uint32_t old=s.input;s.heads[current]=d.next[old];
   if(s.heads[current]==nil)s.tails[current]=nil;
   d.next[old]=s.free;d.used[old]=0;s.free=old;
   d.stats[owner].processed+=s.count;
  }
  __syncthreads();
  uint32_t key=31;uint64_t successor=0;
  if(valid){
   const uint64_t p=r209_event::period(value);uint64_t hit=r209_event::hit(value);
   if(p<2||uint64_t(p)>(uint64_t(30)<<shift)||hit>=span){atomicOr(d.error,4u);}
   else{
    while(hit<span&&hit<remaining){
     const uint32_t word=uint32_t(hit>>5);
     if(!occupied||((occupied[word>>5]>>(word&31u))&1u))atomicAnd(bits+word,~(1u<<(hit&31u)));
     hit+=p;
    }
    if(hit<remaining){key=uint32_t(hit>>shift);successor=r209_event::encode(p,hit);}
   }
  }
  const uint32_t retired=__syncthreads_count(valid&&key==31);
  if(tid==0)d.stats[owner].retired+=retired;
  if(!append(d,s,current,key,successor))break;
 }
 if(tid<buckets){d.heads[owner*buckets+tid]=s.heads[tid];d.tails[owner*buckets+tid]=s.tails[tid];}
 if(tid==0)d.free[owner]=s.free;
}

inline void check(cudaError_t e,const char* why){if(e!=cudaSuccess)throw std::runtime_error(std::string(why)+": "+cudaGetErrorString(e));}
template<class T>inline void alloc(T*& p,size_t n){check(cudaMalloc(&p,n*sizeof(T)),"calendar allocation");}
struct Storage {
 Device d;
 uint32_t capacity;
 Storage(uint32_t count,uint32_t owners):capacity(count){
  if(!count||!owners||owners>256)throw std::runtime_error("calendar geometry");
  d.owners=owners;
  // Stable tile ownership: each owner has at most ceil(total_tiles/owners) tiles.
  const uint64_t tiles=(uint64_t(count)*2+lanes-1)/lanes;
  d.pages=uint32_t((tiles+owners-1)/owners)+buckets;
  const uint64_t pages=uint64_t(d.pages)*owners;
  if(pages>=nil)throw std::runtime_error("calendar page address overflow");
  try{
   alloc(d.records,pages*lanes);alloc(d.next,pages);alloc(d.used,pages);
   alloc(d.heads,owners*buckets);alloc(d.tails,owners*buckets);alloc(d.free,owners);alloc(d.error,1);alloc(d.stats,owners);
  }catch(...){release();throw;}
 }
 ~Storage(){release();}
 Storage(const Storage&)=delete;Storage& operator=(const Storage&)=delete;
 void release(){cudaFree(d.records);cudaFree(d.next);cudaFree(d.used);cudaFree(d.heads);cudaFree(d.tails);cudaFree(d.free);cudaFree(d.error);cudaFree(d.stats);d=Device{};}
 void validate(){uint32_t error=0;check(cudaMemcpy(&error,d.error,sizeof(error),cudaMemcpyDeviceToHost),"calendar status");if(error)throw std::runtime_error("calendar failed closed: "+std::to_string(error));}
 void reset(const uint32_t* primes,const uint64_t* roots,const uint64_t* inverses,
  uint32_t count,uint32_t gap,uint64_t origin,uint64_t end,uint32_t shift,bool members_above_prime_bound){
  if(!members_above_prime_bound||!count||count>capacity||shift!=28||end<=origin)throw std::runtime_error("calendar pre-mutation fallback required");
  check(cudaMemset(d.error,0,sizeof(uint32_t)),"calendar error reset");
  // Bound each initialization kernel on a display GPU; chunk boundaries are
  // multiples of owners*lanes, preserving exclusive owner/page assignment.
  const uint64_t batch=uint64_t(d.owners)*lanes*16;
  for(uint64_t begin=0;begin<uint64_t(count)*2;begin+=batch){
   initialize<<<d.owners,lanes>>>(d,primes,roots,inverses,count,gap,origin,end-origin,shift,begin,begin+batch);
   check(cudaGetLastError(),"calendar bounded initialize");
  }

  // Same bounded kernels, same sticky error flag, one terminal host crossing.
  // No calendar use or output is permitted before this complete validation.
  validate();
 }
};
} // namespace r209cal

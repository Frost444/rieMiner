#pragma once
#include "r346_shared_atlas.cuh"
#include "r346_atlas_client.cuh"
namespace r346cache {
// Constructed before band storage; explicit terminal fence releases borrowers first.
static std::unique_ptr<View> lease;
static std::unique_ptr<r200::Device<uint32_t>> ratio_device;
inline void initialize(mpz_srcptr P){
 const char* mode=std::getenv("ASTRA_R346_USE_ATLAS");
 if(!mode||std::strcmp(mode,"1"))return;
 std::vector<uint32_t> s;
 try{s=ratio(P);}catch(const std::exception&){return;} // Unsupported primorial: complete old path.
 lease=open();if(!lease)return;
 ratio_device=std::make_unique<r200::Device<uint32_t>>(s.size());ratio_device->put(s.data());
}
__global__ void prepare_low(const uint32_t* primes,uint32_t n,const uint32_t* low,const uint32_t* high,
 const uint32_t* A,uint32_t aw,const uint32_t* S,uint32_t sw,uint32_t* inverse,uint32_t* roots,uint32_t* errors){
 const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 const uint32_t p=primes[i],anchor=low[i];
 if(p<=1000000000u||p>4000000000u||!(p&1u)||!anchor||anchor>=p||((high[i/32]>>(i%32))&1u)){
  inverse[i]=roots[i]=0;atomicOr(errors,1u);return;
 }
 const uint64_t mu=UINT64_MAX/p;
 const uint32_t s=riecoin_root_reciprocal::word_remainder(S,sw,p,mu,r200::Hi{});
 const uint32_t iv=r200::rem(uint64_t(s)*anchor,p,mu);
 if(!iv){inverse[i]=roots[i]=0;atomicOr(errors,2u);return;}
 const uint32_t a=riecoin_root_reciprocal::word_remainder(A,aw,p,mu,r200::Hi{});
 const uint32_t x=r200::rem(uint64_t(a)+uint64_t(r200::rem(r200::OFFSET,p,mu))*iv,p,mu);
 inverse[i]=iv;roots[i]=x?p-x:0;
 // The caller still independently proves P*iv=1 and both full-B roots for every entry.
}
}

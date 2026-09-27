#pragma once
#define R202_MAPPED_INPUTS
#include "r201_admission.cuh"
#include "r211_phase_relation.hpp"
#include "r346_low.cuh"
#include <cerrno>
namespace r212 {
using r200::Device;using r200::ck;
struct Band { uint32_t n=0;std::unique_ptr<Device<uint32_t>> p,inv,phase;uint32_t* borrowed=nullptr;uint32_t* primes()const{return borrowed?borrowed:p->p;} };
static std::vector<Band> bands;
static std::unique_ptr<Device<uint32_t>> certificate,full_base,error;
static uint64_t start_origin=0,phase_origin=0,calls=0,rebindings=0,reference_calls=0,fallback_calls=0;
static uint32_t base_words=0;
static bool enabled=false,inject_frontier_fault=false;
constexpr uint32_t CHUNK=1u<<20;
constexpr uint32_t CERTIFICATE_SPAN=33554432u; // first exact cold comparison
constexpr uint32_t CERTIFICATE_WORDS=CERTIFICATE_SPAN/32;

// Independent native quotient/P oracle with exact full-B identity. The checked initial root becomes the
// mutable phase; only the oracle's first-window BITMAP remains immutable.
// Certification, immutable evidence and initial phase binding share one read.
__global__ void certify_and_bind(const uint32_t* primes,uint32_t n,
 const uint32_t* P,uint32_t pw,const uint32_t* A,uint32_t aw,
 const uint32_t* inverse,uint32_t* phases,uint64_t origin,
 uint32_t* mask,uint32_t* errors) {
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 const uint32_t p=primes[i];
 if(p<=1000000000u||p>4000000000u||!(p&1u)){atomicOr(errors,1u);return;}
 uint64_t am=0,pm=0;
 for(uint32_t w=aw;w;--w)am=((am<<32)|A[w-1])%p;
 for(uint32_t w=pw;w;--w)pm=((pm<<32)|P[w-1])%p;
 // Host exact division establishes B=A*P+OFFSET. For p<=4B this
 // multiply/add is strictly below2^64. Native shorter-A oracle therefore
 // proves the same full-B relation, independently of fast prepare/reduction.
 const uint64_t bm=(am*pm+r200::OFFSET%p)%p;
 const uint64_t iv=inverse[i];
 if(!iv||iv>=p||pm*iv%p!=1){atomicOr(errors,2u);return;}
 const uint32_t original=phases[i],om=uint32_t(origin%p);
 for(uint32_t m=0;m<2;++m) {
  const uint64_t b=(bm+2*m)%p;
  const uint32_t absolute=uint32_t((b?p-b:0)*iv%p);
  uint32_t observed=original;
  if(m){uint32_t d=uint32_t(2*iv%p);observed=observed>=d?observed-d:uint32_t(uint64_t(observed)+p-d);}
  if(observed!=absolute){atomicOr(errors,4u);return;}
  // Native modulo oracle deliberately independent of hot phase helpers.
  const uint32_t hit=uint32_t((uint64_t(absolute)+p-om)%p);
  if(hit<CERTIFICATE_SPAN)atomicAnd(mask+(hit>>5),~(1u<<(hit&31)));
 }
 phases[i]=r211_relation::subtract(original,om,p);
}
__global__ void strike(const uint32_t* primes,const uint32_t* inverse,uint32_t* phases,
 uint32_t n,uint64_t represented,uint64_t origin,uint32_t span,uint32_t* bits) {
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 const uint32_t p=primes[i],iv=inverse[i];
 uint32_t root=phases[i];
 if(represented!=origin)root=r211_relation::rebase(root,represented,origin,p);
#ifdef ASTRA_R248_EMIT_OPTIONAL
 ASTRA_R248_EMIT_OPTIONAL(root,p,iv,origin,span);
#endif
 phases[i]=r211_relation::subtract(root,span,p);
 if(root<span)atomicAnd(bits+(root>>5),~(1u<<(root&31)));
 root=r211_relation::second(root,iv,p);
 if(root<span)atomicAnd(bits+(root>>5),~(1u<<(root&31)));
}
__global__ void apply_certificate(uint32_t* bits,const uint32_t* mask,uint32_t offset,uint32_t words) {
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<words)bits[i]&=mask[offset+i];
}
// Exact fallback for a reference outside the retained window. It never reads
// recurrent phases, so future/non-contiguous callers do not weaken the oracle.
__global__ void absolute_reference(const uint32_t* primes,const uint32_t* inverse,uint32_t n,
 const uint32_t* B,uint32_t bw,uint64_t origin,uint32_t span,uint32_t* bits) {
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 const uint32_t p=primes[i];uint64_t bm=0;
 for(uint32_t w=bw;w;--w)bm=((bm<<32)|B[w-1])%p;
 for(uint32_t m=0;m<2;++m) {
  uint64_t b=(bm+2*m)%p,root=(b?p-b:0)*inverse[i]%p;
  uint32_t hit=uint32_t((root+p-origin%p)%p);
  if(hit<span)atomicAnd(bits+(hit>>5),~(1u<<(hit&31)));
 }
}
__global__ void corrupt_frontier(uint32_t* bits) {
 if(!threadIdx.x&&!blockIdx.x)bits[0]^=1u;
}
static uint64_t requested_limit() {
 const char* text=std::getenv("ASTRA_R211_LIMIT");if(!text||!*text)return 0;
 char* end=nullptr;errno=0;auto limit=std::strtoull(text,&end,10);
 if(errno||*end||(limit!=0&&limit!=2000000000ull&&limit!=4000000000ull))
  throw std::runtime_error("R212 limit must be0,2B or4B");
 return limit;
}
static void init(mpz_srcptr P,mpz_srcptr B,const std::array<uint32_t,7>& offsets,uint32_t inherited_limit) {
 const uint64_t limit=requested_limit();if(!limit)return;
 if(enabled||!bands.empty())throw std::runtime_error("R212 initialization ownership");
 if(inherited_limit!=1000000000u||offsets[0]!=0||offsets[1]!=2||mpz_sgn(P)<=0||
    mpz_cmp_ui(B,static_cast<unsigned long>(limit))<=0||start_origin>UINT64_MAX-r200::SPAN)
  throw std::runtime_error("R212 inherited/proper divisor geometry");
 mpz_t a,o,expected;mpz_inits(a,o,expected,nullptr);mpz_fdiv_qr(a,o,B,P);
 mpz_set_str(expected,"114023297140211",10);
 if(mpz_cmp(o,expected)){mpz_clears(a,o,expected,nullptr);throw std::runtime_error("R212 qualified offset");}
 auto pw=r200::export_words(P),aw=r200::export_words(a),bw=r200::export_words(B);
 mpz_clears(a,o,expected,nullptr);
 r346cache::initialize(P);
 Device<uint32_t> dp(pw.size()),da(aw.size());dp.put(pw.data());da.put(aw.data());
 full_base=std::make_unique<Device<uint32_t>>(bw.size());full_base->put(bw.data());base_words=uint32_t(bw.size());
 certificate=std::make_unique<Device<uint32_t>>(CERTIFICATE_WORDS);
 ck(cudaMemset(certificate->p,255,size_t(CERTIFICATE_WORDS)*4),"R212 certificate ones");
 error=std::make_unique<Device<uint32_t>>(1);ck(cudaMemset(error->p,0,4),"R212 errors reset");
 const char* dir=std::getenv("ASTRA_R201_CACHE");if(!dir||!*dir)throw std::runtime_error("R212 read-only cache required");
 const uint32_t counts[]={47374753,46227250,45512275};uint64_t total=0;
 for(uint32_t b=1;b<limit/1000000000ull;++b) {
  const auto began=r200::Clock::now();uint32_t lo=b*1000000000u,hi=(b+1)*1000000000u;
  auto path=std::filesystem::path(dir)/("band-"+std::to_string(lo)+"-"+std::to_string(hi)+".u32");
  Band band;band.n=counts[b-1];total+=band.n;
  band.inv=std::make_unique<Device<uint32_t>>(band.n);band.phase=std::make_unique<Device<uint32_t>>(band.n);
  if(r346cache::lease)band.borrowed=r346cache::lease->delta(b-1);
  else {r201::Mapping mapped(path,uint64_t(counts[b-1])*4);band.p=std::make_unique<Device<uint32_t>>(band.n);band.p->put(mapped.data);}
  for(uint32_t first=0;first<band.n;first+=CHUNK) {
   const uint32_t n=std::min(CHUNK,band.n-first),blocks=(n+255u)/256u;
   if(r346cache::lease)r346cache::prepare_low<<<blocks,256>>>(band.primes()+first,n,r346cache::lease->low(b-1)+first,
     r346cache::lease->high(b-1)+first/32,da.p,uint32_t(aw.size()),r346cache::ratio_device->p,uint32_t(r346cache::ratio_device->n),
     band.inv->p+first,band.phase->p+first,error->p);
   else r200::prepare<<<blocks,256>>>(band.primes()+first,n,dp.p,uint32_t(pw.size()),da.p,uint32_t(aw.size()),
                               band.inv->p+first,band.phase->p+first,error->p);
   certify_and_bind<<<blocks,256>>>(band.primes()+first,n,dp.p,uint32_t(pw.size()),da.p,uint32_t(aw.size()),
                                   band.inv->p+first,band.phase->p+first,start_origin,certificate->p,error->p);
   ck(cudaGetLastError(),"R212 bounded preparation/certification");
  }
  uint32_t errors=0;error->get(&errors);if(errors)throw std::runtime_error("R212 full-B root relation failure: no admission");
  r345::out()<<"phase=r212_admission_setup high="<<hi<<" factors="<<band.n
   <<" complete_ms="<<r200::elapsed(began)<<" exact_root_relations="<<uint64_t(band.n)*2
   <<" resident_band_bytes="<<uint64_t(band.n)*12<<" initial_kernel_max_entries="<<CHUNK<<" errors=0"<<std::endl;
  bands.push_back(std::move(band));
 }
 phase_origin=start_origin;calls=rebindings=reference_calls=fallback_calls=0;enabled=true;
 r345::out()<<"phase=r212_memory saved_immutable_root_bytes="<<total*4
  <<" independent_certificate_bytes="<<uint64_t(CERTIFICATE_WORDS)*4<<std::endl;
}
static void apply(uint32_t* bits,uint64_t origin,uint64_t span,bool advance,cudaStream_t stream=cudaStreamPerThread) {
 if(!enabled)return;
 if(!span||span>r200::SPAN||(span&31u)||origin>UINT64_MAX-span)
  throw std::runtime_error("R212 bounded interval");
 if(!advance) {
  ++reference_calls;
  const uint64_t offset=origin-start_origin;
  if(certificate&&origin>=start_origin&&span<=CERTIFICATE_SPAN&&offset<=CERTIFICATE_SPAN-span&&!(offset&31u)) {
   const uint32_t words=uint32_t(span/32);
   apply_certificate<<<(words+255u)/256u,256,0,stream>>>(bits,certificate->p,uint32_t(offset/32),words);
   ck(cudaGetLastError(),"R212 immutable certificate");return;
  }
  ++fallback_calls;
  for(auto& b:bands)for(uint32_t first=0;first<b.n;first+=CHUNK) {
   const uint32_t n=std::min(CHUNK,b.n-first);
   absolute_reference<<<(n+255u)/256u,256,0,stream>>>(b.primes()+first,b.inv->p+first,n,full_base->p,base_words,origin,uint32_t(span),bits);
   ck(cudaGetLastError(),"R212 independent absolute fallback");
  }
  return;
 }
 ++calls;if(origin!=phase_origin)++rebindings;
 for(auto& b:bands) {
  strike<<<(b.n+255u)/256u,256,0,stream>>>(b.primes(),b.inv->p,b.phase->p,b.n,phase_origin,origin,uint32_t(span),bits);
  ck(cudaGetLastError(),"R212 phase frontier");
 }
 phase_origin=origin+span;
 if(inject_frontier_fault&&calls==1) {
  // A deterministic one-bit change after ALL filters; cold comparison must
  // reject regardless of the original candidate density or particular roots.
  corrupt_frontier<<<1,1,0,stream>>>(bits);ck(cudaGetLastError(),"R212 frontier fault injection");
 }
}
static void reference_done() {
 // Invoked only AFTER complete bitmap comparison's D2H fence succeeded.
 certificate.reset();
}
// R272: identical band kernels, ordered host submissions spread across Q0s.
static unsigned r272_part=0;
static uint64_t r272_origin=0,r272_span=0;
static void apply_part(uint32_t* bits,uint64_t origin,uint64_t span,unsigned part,cudaStream_t stream){
 if(!enabled)return;
 if(bands.size()!=3 || part>=3 || part!=r272_part || !span || span>r200::SPAN ||
    (span&31u) || origin>UINT64_MAX-span)throw std::runtime_error("R272 low-band ownership");
 if(part==0){r272_origin=origin;r272_span=span;}
 if(origin!=r272_origin || span!=r272_span)throw std::runtime_error("R272 low-band generation");
 auto& b=bands[part];
 strike<<<(b.n+255u)/256u,256,0,stream>>>(b.primes(),b.inv->p,b.phase->p,b.n,phase_origin,origin,uint32_t(span),bits);
 ck(cudaGetLastError(),"R272 low-band phase");
 if(++r272_part==3){
  r272_part=0;++calls;if(origin!=phase_origin)++rebindings;phase_origin=origin+span;
  if(inject_frontier_fault&&calls==1){corrupt_frontier<<<1,1,0,stream>>>(bits);ck(cudaGetLastError(),"R272 low-band fault");}
 }
}
static void cleanup() {
 if(!enabled)return;ck(cudaDeviceSynchronize(),"R212 terminal guard");
 r345::out()<<"phase=r212_admission_complete frontiers="<<calls<<" references="<<reference_calls
  <<" absolute_fallbacks="<<fallback_calls<<" rebindings="<<rebindings<<" errors=0"<<std::endl;
 bands.clear();certificate.reset();full_base.reset();error.reset();enabled=false;r346cache::ratio_device.reset();r346cache::lease.reset();
}
}

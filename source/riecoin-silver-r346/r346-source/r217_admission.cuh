#pragma once
#define R202_MAPPED_INPUTS
#include "r212_admission.cuh"
#include "r217_native31.hpp"
#define R208_LIBRARY
#include "r208_admission8b.cu"
#include "r209_calendar.cuh"
#include "r346_wide.cuh"
namespace r217 {
using r200::Device;using r200::ck;
struct Band{uint32_t n=0;std::unique_ptr<Device<uint32_t>> delta;std::unique_ptr<r209cal::Storage> calendar;uint32_t* borrowed=nullptr;uint32_t* primes()const{return borrowed?borrowed:delta->p;}};
static std::vector<Band> bands;
static std::unique_ptr<Device<uint32_t>> dp,db,error,certificate;
static std::vector<uint32_t> export31(mpz_srcptr n){std::vector<uint32_t> d((mpz_sizeinbase(n,2)+30)/31);size_t count=0;mpz_export(d.data(),&count,-1,4,0,1,n);return d;}
__global__ void certify(const uint32_t* delta,uint32_t n,const uint32_t* P,uint32_t pw,
 const uint32_t* A,uint32_t aw,const uint64_t* inverse,const uint64_t* roots,
 uint64_t origin,uint32_t* mask,uint32_t* errors){
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 const uint64_t p=r208::low+delta[i];
 if(p<=r208::low||p>r208::high||!(p&1)){atomicOr(errors,1u);return;}
 const auto pm=r217_native::words(P,pw,p),am=r217_native::words(A,aw,p),iv=inverse[i];
 if(!iv||iv>=p||r217_native::product(pm,iv,p)!=1){atomicOr(errors,2u);return;}
 const auto bm=r217_native::add(r217_native::product(am,pm,p),r200::OFFSET%p,p);
 for(unsigned m=0;m<2;++m){
  const auto b=r217_native::add(bm,2*m,p),absolute=r217_native::product(b?p-b:0,iv,p);
  uint64_t observed=roots[i];if(m){uint64_t d=2*iv;if(d>=p)d-=p;observed=observed>=d?observed-d:observed+p-d;}
  if(observed!=absolute){atomicOr(errors,4u);return;}
  const auto hit=(absolute+p-origin%p)%p;
  if(hit<r212::CERTIFICATE_SPAN)atomicAnd(mask+(hit>>5),~(1u<<(hit&31)));
 }
}
static void reference_done(){r212::reference_done();certificate.reset();}
static bool enabled=false;
static uint64_t start_origin=0,next_origin=0,steps=0;
static bool inject_hot_fault=false;
// Ordered in the producer stream. The existing immutable tail-rail receipt
// must reject this frontier before admitting any of its factors downstream.
__global__ void latch_calendar_errors(uint32_t* status,uint32_t* a,
 const uint32_t* b,const uint32_t* c,const uint32_t* d,bool inject){
 if(inject)atomicOr(a,0x80000000u);
 const uint32_t errors=*a|*b|*c|*d;
 if(errors)atomicOr(status,errors);
}
// Independent oracle deliberately recomputes from full B, not compact roots,
// mutable calendar payloads or cached phase state.
__global__ void oracle(const uint32_t* delta,uint32_t n,const uint32_t* P,uint32_t pw,
 const uint32_t* B,uint32_t bw,uint64_t origin,uint32_t span,uint32_t* bits,uint32_t* errors){
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;uint64_t p=r208::low+delta[i];
 if(p<=r208::low||p>r208::high||!(p&1)){atomicOr(errors,1u);return;}
 uint64_t pm=r208::slow_words(P,pw,p),bm=r208::slow_words(B,bw,p),iv=inverse64(pm,p);
 if(!iv||r208::slow_product(pm,iv,p)!=1){atomicOr(errors,2u);return;}
 for(unsigned m=0;m<2;++m){uint64_t b=bm+2*m;if(b>=p)b-=p;
  uint64_t r=r208::slow_product(b?p-b:0,iv,p),o=origin%p;r=r>=o?r-o:r+p-o;
  if(r<span)atomicAnd(bits+(r>>5),~(1u<<(r&31u)));
 }
}
static void init(mpz_srcptr P,mpz_srcptr B,const std::array<uint32_t,7>& offsets,uint32_t inherited){
 r212::start_origin=start_origin;r212::init(P,B,offsets,inherited);
 const char* mode=std::getenv("ASTRA_R217_WIDE");if(!mode||std::strcmp(mode,"1"))return;
 if(!r212::enabled||offsets[0]!=0||offsets[1]!=2||mpz_sizeinbase(B,2)<=33||start_origin>UINT64_MAX-r208::high)throw std::runtime_error("R209 exact geometry");
 mpz_t a,o;mpz_inits(a,o,nullptr);mpz_fdiv_qr(a,o,B,P);
 mpz_t expected;mpz_init_set_str(expected,"114023297140211",10);
 if(mpz_cmp(o,expected)){mpz_clears(a,o,expected,nullptr);throw std::runtime_error("R217 exact B=A*P+O relation");}mpz_clear(expected);
 auto p31=export31(P),a31=export31(a);
 auto pw=r200::export_words(P),aw=r200::export_words(a),bw=r200::export_words(B);mpz_clears(a,o,nullptr);
 dp=std::make_unique<Device<uint32_t>>(pw.size());db=std::make_unique<Device<uint32_t>>(bw.size());error=std::make_unique<Device<uint32_t>>(1);
 Device<uint32_t> dp31(p31.size()),da31(a31.size());dp31.put(p31.data());da31.put(a31.data());
 certificate=std::make_unique<Device<uint32_t>>(r212::CERTIFICATE_WORDS);
 ck(cudaMemset(certificate->p,255,size_t(r212::CERTIFICATE_WORDS)*4),"R217 certificate ones");
 Device<uint32_t> da(aw.size());dp->put(pw.data());db->put(bw.data());da.put(aw.data());ck(cudaMemset(error->p,0,4),"R209 reset");
 const uint32_t counts[]={44992411,44591145,44258984,43979302};
 const char* dir=std::getenv("ASTRA_R209_CACHE");if(!dir||!*dir)throw std::runtime_error("R209 prepared cache required");
 for(unsigned j=0;j<4;++j){auto start=r200::Clock::now();uint64_t lo=r208::low+j*1000000000ull,hi=lo+1000000000ull;
  auto path=std::filesystem::path(dir)/("delta4b-band-"+std::to_string(lo)+"-"+std::to_string(hi)+".u32");
  Band b;b.n=counts[j];if(r346cache::lease)b.borrowed=r346cache::lease->delta(j+3);else{r201::Mapping mapped(path,uint64_t(counts[j])*4);b.delta=std::make_unique<Device<uint32_t>>(b.n);b.delta->put(mapped.data);}
  Device<uint64_t> inverse(b.n),roots(b.n);constexpr uint32_t chunk=1u<<20;
  for(uint32_t first=0;first<b.n;first+=chunk){uint32_t n=std::min(chunk,b.n-first);
   if(r346cache::lease)r346cache::prepare_wide<<<(n+255)/256,256>>>(b.primes()+first,n,r346cache::lease->low(j+3)+first,r346cache::lease->high(j+3)+first/32,da.p,uint32_t(aw.size()),r346cache::ratio_device->p,uint32_t(r346cache::ratio_device->n),inverse.p+first,roots.p+first,error->p);
   else r208::prepare<<<(n+255)/256,256>>>(b.primes()+first,n,dp->p,uint32_t(pw.size()),da.p,uint32_t(aw.size()),inverse.p+first,roots.p+first,error->p);
   certify<<<(n+255)/256,256>>>(b.primes()+first,n,dp31.p,uint32_t(p31.size()),da31.p,uint32_t(a31.size()),inverse.p+first,roots.p+first,start_origin,certificate->p,error->p);
  }
  ck(cudaGetLastError(),"R209 roots");uint32_t errors=0;error->get(&errors);if(errors)throw std::runtime_error("R209 independent complete roots");
  double roots_ms=r200::elapsed(start);b.calendar=std::make_unique<r209cal::Storage>(b.n,256);
  b.calendar->reset(b.primes(),roots.p,inverse.p,b.n,2,start_origin,UINT64_MAX,28,true);
  r345::out()<<"phase=r217_setup high="<<hi<<" count="<<b.n<<" root_and_input_ms="<<roots_ms<<" complete_ms="<<r200::elapsed(start)<<" temporary_bytes_released="<<uint64_t(b.n)*16<<" errors=0\n"<<std::flush;
  bands.push_back(std::move(b));
 }
 next_origin=start_origin;steps=0;enabled=true;
}
static void apply(uint32_t* bits,uint64_t origin,uint64_t span,bool advance,uint32_t* consumer_status=nullptr,cudaStream_t stream=cudaStreamPerThread){
 r212::apply(bits,origin,span,advance,stream);if(!enabled)return;
 if(!span||span>r200::SPAN||origin>UINT64_MAX-span)throw std::runtime_error("R209 domain bound");
 if(advance){
  if(span!=r200::SPAN||origin!=next_origin)throw std::runtime_error("R209 requires contiguous complete frontiers");
  if(consumer_status&&bands.size()!=4)throw std::runtime_error("R209B status ownership geometry");
  for(auto& b:bands){r209cal::advance<<<256,256,0,stream>>>(b.calendar->d,bits,nullptr,uint32_t(steps%31),28,UINT64_MAX-origin);ck(cudaGetLastError(),"R209 advance");if(!consumer_status){ck(cudaStreamSynchronize(stream),"R266 standalone calendar fence");b.calendar->validate();}}
  if(consumer_status){
   const bool inject=inject_hot_fault&&steps==1;
   latch_calendar_errors<<<1,1,0,stream>>>(consumer_status,bands[0].calendar->d.error,
    bands[1].calendar->d.error,bands[2].calendar->d.error,bands[3].calendar->d.error,inject);
   ck(cudaGetLastError(),"R209B inherited receipt latch");
   if(inject)r345::out()<<"phase=r209b_injected_hot_fault origin="<<origin<<"\n"<<std::flush;
  }
  next_origin=origin+span;++steps;
 }else{
  const uint64_t offset=origin-start_origin;
  if(certificate&&origin>=start_origin&&span<=r212::CERTIFICATE_SPAN&&offset<=r212::CERTIFICATE_SPAN-span&&!(offset&31u)){
   uint32_t words=uint32_t(span/32);r212::apply_certificate<<<(words+255)/256,256,0,stream>>>(bits,certificate->p,uint32_t(offset/32),words);
   ck(cudaGetLastError(),"R217 exact immutable wide certificate");return;
  }
  constexpr uint32_t chunk=1u<<20;
  for(auto& b:bands)for(uint32_t first=0;first<b.n;first+=chunk){uint32_t n=std::min(chunk,b.n-first);
   oracle<<<(n+255)/256,256,0,stream>>>(b.primes()+first,n,dp->p,uint32_t(dp->n),db->p,uint32_t(db->n),origin,uint32_t(span),bits,error->p);
  }
  ck(cudaGetLastError(),"R209 full-domain oracle");ck(cudaStreamSynchronize(stream),"R266 oracle receipt fence");uint32_t errors=0;error->get(&errors);if(errors)throw std::runtime_error("R209 oracle failed closed");
 }
}
static void apply_async(uint32_t* bits,uint64_t origin,uint64_t span,uint32_t* status,cudaStream_t stream=cudaStreamPerThread){
 if(!status)throw std::runtime_error("R209B consumer guard required");
 apply(bits,origin,span,true,status,stream);
}
// Parts0..2 submit the three low bands; parts3..4 submit the four independent
// calendars in pairs. Their checked generation advances only after all four.
static unsigned r272_part=0;
static uint64_t r272_origin=0;
static void apply_phase(uint32_t* bits,uint64_t origin,uint64_t span,uint32_t* status,unsigned part,cudaStream_t stream){
 if(part>4 || !status)throw std::runtime_error("R272 high-band part/status");
 if(part<3){r212::apply_part(bits,origin,span,part,stream);return;}
 if(!enabled)return;
 if(bands.size()!=4 || span!=r200::SPAN || origin!=next_origin || origin>UINT64_MAX-span ||
    part-3!=r272_part)throw std::runtime_error("R272 calendar ownership/generation");
 if(part==3)r272_origin=origin;
 if(origin!=r272_origin)throw std::runtime_error("R272 calendar origin changed");
 const unsigned first=2*(part-3);
 for(unsigned i=first;i<first+2;++i){auto& b=bands[i];
  r209cal::advance<<<256,256,0,stream>>>(b.calendar->d,bits,nullptr,uint32_t(steps%31),28,UINT64_MAX-origin);
  ck(cudaGetLastError(),"R272 calendar pair");
 }
 if(++r272_part==2){
  r272_part=0;const bool inject=inject_hot_fault&&steps==1;
  latch_calendar_errors<<<1,1,0,stream>>>(status,bands[0].calendar->d.error,
   bands[1].calendar->d.error,bands[2].calendar->d.error,bands[3].calendar->d.error,inject);
  ck(cudaGetLastError(),"R272 final calendar error latch");next_origin=origin+span;++steps;
 }
}
static void cleanup(){
 if(enabled){uint64_t admitted=0,processed=0,retired=0;
  for(auto& b:bands){b.calendar->validate();std::vector<r209cal::Stats> stats(256);ck(cudaMemcpy(stats.data(),b.calendar->d.stats,stats.size()*sizeof(stats[0]),cudaMemcpyDeviceToHost),"R209 stats");
   uint64_t local=0;for(auto& s:stats){local+=s.admitted;admitted+=s.admitted;processed+=s.processed;retired+=s.retired;}
   if(local!=uint64_t(b.n)*2)throw std::runtime_error("R209 event admission conservation");
  }
  if(retired)throw std::runtime_error("R209 unexpected terminal retirement");
  r345::out()<<"phase=r217_complete frontiers="<<steps<<" admitted="<<admitted<<" processed="<<processed<<" retired="<<retired<<" errors=0\n";
  bands.clear();dp.reset();db.reset();error.reset();certificate.reset();enabled=false;
 }
 r212::cleanup();
}
}

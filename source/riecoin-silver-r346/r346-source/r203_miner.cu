// Same exact full reference relation, different verification unit of work.
#include <cstdint>
static bool r203_compare(const uint32_t*,uint32_t,const uint32_t*,const uint32_t*,
                         uint32_t,uint64_t,uint32_t,uint32_t*);
#define ASTRA_R203_COMPARE r203_compare
#define R202_MAPPED_INPUTS
#include "r201_admission.cuh"
#ifndef ASTRA_R201_INITIALIZE
#define ASTRA_R201_INITIALIZE r201::init
#define ASTRA_R201_APPLY r201::apply
#define ASTRA_R201_CLEANUP r201::cleanup
#endif
#define main r203_inherited_main
#include "r184_miner.cu"
#undef main
namespace r203 {
static unsigned mode=0;
// Independent simple oracle: no private planes, cached phases, occupied-word
// skip, compressed quotient or new producer arithmetic. Direct absolute roots.
__global__ void all_positions(uint32_t* bitmap,uint32_t span,const uint32_t* primes,
 const uint32_t* roots,uint32_t count,uint64_t origin,uint32_t* errors){
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 uint32_t p=primes[i];if(p<2){atomicOr(errors,1u);return;}
 uint32_t om=uint32_t(origin%p),members=7;
#ifdef RIECOIN_A54_OPTIONAL_PRIME_LIMIT
 if(p>RIECOIN_A54_OPTIONAL_PRIME_LIMIT)members=2;
#endif
 for(uint32_t member=0;member<members;++member){
  uint32_t root=riecoin_compressed_root_storage::root_at(roots,p,i,member);
  if(root>=p){atomicOr(errors,2u);continue;}
  uint64_t hit=root>=om?root-om:uint64_t(root)+p-om;
  for(;hit<span;hit+=p)atomicAnd(bitmap+(hit>>5u),~(1u<<(hit&31u)));
 }
}
__global__ void compare(const uint32_t* a,const uint32_t* b,uint32_t count,uint32_t* errors){
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count&&a[i]!=b[i])atomicAdd(errors,1u);
}
}
static bool r203_compare(const uint32_t* observed,uint32_t positions,const uint32_t* primes,
 const uint32_t* roots,uint32_t pairs,uint64_t origin,uint32_t macros,uint32_t* errors){
 if(!r203::mode)return false;
 uint64_t span=uint64_t(positions)*macros;
 if(!positions||(positions&31u)||!macros||span>UINT32_MAX||origin>UINT64_MAX-span||pairs%7u)return false;
 uint32_t words=uint32_t(span/32),count=pairs/7u;
 r200::Device<uint32_t> reference(words);r200::ck(cudaMemset(reference.p,255,size_t(words)*4),"R203 reference ones");
 float ms=r200::timed([&]{
  r203::all_positions<<<(count+255u)/256u,256u>>>(reference.p,uint32_t(span),primes,roots,count,origin,errors);
  // Additional R201 mandatory divisors are applied to the SAME absolute domain,
  // once, without reading or modifying its recurrent production phase state.
  ASTRA_R201_APPLY(reference.p,origin,span,false);
  r203::compare<<<(words+255u)/256u,256u>>>(reference.p,observed,words,errors);
 });
 uint32_t mismatch=0;r200::ck(cudaMemcpy(&mismatch,errors,4,cudaMemcpyDeviceToHost),"R203 full comparison result");
 std::cout<<"phase=r203_grouped_reference mode="<<(r203::mode==1?"observe_and_scalar":"grouped")
  <<" checked_words="<<words<<" checked_positions="<<span<<" prime_visits="<<count
  <<" origin="<<origin<<" gpu_ms="<<ms<<" mismatch="<<mismatch<<std::endl;
 if(mismatch)throw std::runtime_error("R203 exact grouped bitmap reference mismatch");
 return r203::mode==2;
}
#ifndef ASTRA_R203_ENTRY
#define ASTRA_R203_ENTRY main
#endif
int ASTRA_R203_ENTRY(int argc,char**argv){
 _putenv_s("ASTRA_R178_PHASES","1");_putenv_s("ASTRA_A93_FRONTIER_GROUPS","8");
 if(!std::getenv("ASTRA_A95_PLANES"))_putenv_s("ASTRA_A95_PLANES","32");
 _putenv_s("ASTRA_A93_AUDIT","0");
 _putenv_s("ASTRA_A94_CUTOVER","33554432");_putenv_s("ASTRA_A94_DIAGNOSTIC","0");
 _putenv_s("ASTRA_CUDA_WAIT_POLICY","block");
 const char* text=std::getenv("ASTRA_R203_GROUPED_GUARD");
 if(!text||std::strcmp(text,"off")==0)r203::mode=0;
 else if(std::strcmp(text,"observe")==0)r203::mode=1;
 else if(std::strcmp(text,"grouped")==0)r203::mode=2;
 else {std::cerr<<"R203 unsupported guard mode"<<std::endl;return 80;}
 return r203_inherited_main(argc,argv);
}

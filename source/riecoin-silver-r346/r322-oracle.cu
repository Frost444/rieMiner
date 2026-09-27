#include <algorithm>
#define riecoin_a99b_primitives riecoin_r294_primitives
#define riecoin_a99b_launch_oracle riecoin_r294_launch_oracle
#define riecoin_a99b_launch_segmented riecoin_r294_launch_segmented
#include <cuda_runtime.h>
#include <gmp.h>
#include <array>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include "riecoin-cuda-prp1152.cuh"
#include "riecoin_lazy_base2_tpi4.cuh"
#include "riecoin_r203_r1216_q0_launch.hpp"
#include "r294_sos_launch.hpp"
#include "r297_sos_launch.hpp"
extern "C" cudaError_t riecoin_a99b_launch_oracle(const void*,uint8_t*,uint32_t*,uint32_t,cudaStream_t);
extern "C" cudaError_t riecoin_a99b_launch_segmented(const void*,uint8_t*,const uint32_t*,uint32_t,uint32_t,cudaStream_t);
static void cuda_check(cudaError_t e,const char* s){if(e!=cudaSuccess)throw std::runtime_error(std::string(s)+": "+cudaGetErrorString(e));}
static void mpz_set_u64(mpz_ptr z,uint64_t v){mpz_import(z,1,-1,sizeof(v),0,0,&v);}
#include "riecoin_lazy_base2_guard_host.cuh"
#include "riecoin_r203_r1216_q0_guard_host.cuh"
#include "r290-primitive-guard.cuh"
#include "r289-lazy-guard.cuh"
int main(int argc,char** argv){try{
 cuda_check(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync),"blocking");
 cuda_check(cudaSetDevice(0),"device");cudaDeviceProp p{};cuda_check(cudaGetDeviceProperties(&p,0),"properties");
 if(!strstr(p.name,"3090"))throw std::runtime_error("3090 required");

 namespace g=riecoin_lazy_base2_guard;
 if(argc==2 && std::string(argv[1])=="--poison"){
  cgbn_mem_t<1280> input{};input._limbs[0]=7;input._limbs[38]=1;
  g::Device<cgbn_mem_t<1280>> di(1);g::Device<uint8_t> dv(1);g::Device<uint32_t> dr(40);
  cuda_check(cudaMemcpy(di.data,&input,sizeof(input),cudaMemcpyHostToDevice),"poison input");
  cuda_check(riecoin_a99b_launch_oracle(di.data,dv.data,dr.data,1,nullptr),"poison launch");
  auto e=cudaDeviceSynchronize();
  bool rejected=e==cudaErrorIllegalInstruction || e==cudaErrorLaunchFailure;
  std::cout<<"phase=a99b_poison rejected="<<rejected<<" cuda_error="<<(int)e<<std::endl;
  return rejected?0:4;
 }
 primitive_guard();
 lazy_guard();
 std::vector<g::Fixture> cases;
 for(const auto& f:g::corpus()){
  g::Integer n;mpz_import(n.value,40,-1,4,0,0,f.input._limbs);
  auto bits=mpz_sizeinbase(n.value,2);
  if(bits>=3 && bits<=1184 && mpz_odd_p(n.value))cases.push_back(f);
 }
 // Add the current 1168-bit work width and upper supported boundary with
 // deterministic dense operands; all expectations are computed outside timing.
 const size_t inheritedCases=cases.size();
 uint32_t seed=0xa5626097u;
 for(unsigned j=0;j<64;++j)for(unsigned bits:{1135u,1136u,1137u,1138u,1139u,1140u,1141u,1142u,1143u,1144u,1152u,1153u,1154u,1168u,1177u,1184u}){
  uint32_t words[40];for(auto& w:words){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;w=seed;}
  g::Integer n;mpz_import(n.value,40,-1,4,0,0,words);mpz_fdiv_r_2exp(n.value,n.value,bits);mpz_setbit(n.value,bits-1);mpz_setbit(n.value,0);
  if(j==0){mpz_nextprime(n.value,n.value);if(mpz_sizeinbase(n.value,2)!=bits)throw std::runtime_error("Prime width");}
  if(j==1){mpz_set_ui(n.value,0);mpz_setbit(n.value,bits);mpz_sub_ui(n.value,n.value,1);}
  g::append(cases,n.value,"a56-dense-"+std::to_string(bits)+"-"+std::to_string(j));
 }
 { std::ifstream file("r287-input.txt");std::string hex;unsigned added=0;while(file>>hex){if(hex.size()>320||added>=3)throw std::runtime_error("Counterexample bound");g::Integer n;if(mpz_set_str(n.value,hex.c_str(),16))throw std::runtime_error("Counterexample hex");g::append(cases,n.value,"R286-exact-counterexample-"+std::to_string(added++));}if(added!=3)throw std::runtime_error("Counterexamples missing"); }
 std::rotate(cases.begin(),cases.begin()+inheritedCases,cases.end());
 constexpr unsigned count=65536;
 std::vector<cgbn_mem_t<1280>> input(count);std::vector<uint8_t> verdict(count);std::vector<uint32_t> residue(count*40);
 for(unsigned i=0;i<count;++i)input[i]=cases[i%cases.size()].input;
 g::Device<cgbn_mem_t<1280>> di(count);g::Device<uint8_t> dv(count),dw(count);g::Device<uint32_t> dr(count*40);
 cuda_check(cudaMemcpy(di.data,input.data(),input.size()*sizeof(input[0]),cudaMemcpyHostToDevice),"input");
 auto launch=[&](unsigned mode){
  if(mode==2){cuda_check(riecoin_r297_launch_oracle(di.data,dv.data,dr.data,count,nullptr),"R297 exact members");return;}
  if(mode==1){cuda_check(riecoin_a99b_launch_oracle(di.data,dv.data,dr.data,count,nullptr),"a58 launch");return;}
  else {riecoin_lazy_base2_window::oracle_kernel<<<(count*4+127)/128,128>>>(di.data,dv.data,dw.data,dr.data,count);cuda_check(cudaGetLastError(),"stock launch");}
 };
 unsigned errors=0,checks=0;
 for(unsigned candidate:{0u,1u,2u}){
  launch(candidate);cuda_check(cudaDeviceSynchronize(),"warm exact");
  cuda_check(cudaMemcpy(verdict.data(),dv.data,count,cudaMemcpyDeviceToHost),"verdict");
  cuda_check(cudaMemcpy(residue.data(),dr.data,residue.size()*4,cudaMemcpyDeviceToHost),"residue");
  for(unsigned i=0;i<count;++i){const auto& c=cases[i%cases.size()];bool bad=verdict[i]!=c.verdict;for(unsigned w=0;w<40;++w)bad|=residue[i*40+w]!=c.residue[w];errors+=bad;++checks;}
 }
 std::cout<<"phase=a99b_exact cases="<<cases.size()<<" checks="<<checks<<" errors="<<errors<<std::endl;
 if(errors) return 2;

 // Non-multiple-of-warp capacities and sparse holes exercise instance mapping.
 constexpr unsigned cap=67, macros=7, slots=cap*macros;
 const uint32_t counts[macros]={0,1,31,32,33,66,67};
 g::Device<uint32_t> dc(macros);g::Device<uint8_t> ds(slots+32);
 cuda_check(cudaMemcpy(dc.data,counts,sizeof(counts),cudaMemcpyHostToDevice),"counts");
 cuda_check(cudaMemset(ds.data,0xA5,slots+32),"sentinels");
 cuda_check(riecoin_a99b_launch_segmented(di.data,ds.data+16,dc.data,macros,cap,nullptr),"segmented");
 cuda_check(cudaDeviceSynchronize(),"segmented sync");
 std::vector<uint8_t> segmented(slots+32);
 cuda_check(cudaMemcpy(segmented.data(),ds.data,segmented.size(),cudaMemcpyDeviceToHost),"segmented result");
 unsigned segErrors=0;
 for(unsigned i=0;i<16;++i)segErrors+=(segmented[i]!=0xA5)+(segmented[16+slots+i]!=0xA5);
 for(unsigned i=0;i<slots;++i){
   auto expected=i%cap<counts[i/cap]?cases[i%cases.size()].verdict:0;
   segErrors+=segmented[i+16]!=expected;
 }
 std::cout<<"phase=a99b_segmented slots="<<slots<<" guards=32 errors="<<segErrors<<std::endl;
 if(segErrors) return 3;

 // The new boundary is a dense list of original sparse positions, not a new
 // arithmetic predicate. Test permutations, empty/partial/full counts and
 // untouched holes/guards against the independent GMP fixture verdicts.
 g::Device<uint32_t> denseSlots(slots),denseCount(1);
 std::vector<uint32_t> slotList(slots);
 for(unsigned i=0;i<slots;++i)slotList[i]=(i*17u)%slots;
 cuda_check(cudaMemcpy(denseSlots.data,slotList.data(),slots*4,cudaMemcpyHostToDevice),"dense slots");
 unsigned denseChecks=0;
 for(const unsigned live:{0u,1u,31u,32u,33u,127u,128u,129u,slots}){
  cuda_check(cudaMemset(ds.data,0xA5,slots+32),"dense sentinels");
  cuda_check(cudaMemset(ds.data+16,0,slots),"dense holes");
  cuda_check(cudaMemcpy(denseCount.data,&live,4,cudaMemcpyHostToDevice),"dense count");
  cuda_check(riecoin_r294_launch_dense(di.data,ds.data+16,denseSlots.data,denseCount.data,slots,cudaStreamPerThread),"dense launch");
  cuda_check(cudaDeviceSynchronize(),"dense ready");
  cuda_check(cudaMemcpy(segmented.data(),ds.data,segmented.size(),cudaMemcpyDeviceToHost),"dense receipt");
  std::vector<uint8_t> expected(slots,0);
  for(unsigned i=0;i<live;++i)expected[slotList[i]]=cases[slotList[i]%cases.size()].verdict;
  for(unsigned i=0;i<slots;++i){if(segmented[16+i]!=expected[i])throw std::runtime_error("R289 dense mismatch");++denseChecks;}
  for(unsigned i=0;i<16;++i)if(segmented[i]!=0xA5 || segmented[16+slots+i]!=0xA5)throw std::runtime_error("R289 dense guard");
 }
 std::cout<<"phase=r294_dense_exact checks="<<denseChecks<<" guards=288 errors=0"<<std::endl;


 std::cout<<"phase=r322_oracle_complete errors=0 terminal=PASS"<<std::endl;
 return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}

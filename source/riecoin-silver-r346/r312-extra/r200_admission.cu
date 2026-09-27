#pragma once
// R200 isolated exact admission experiment. Never a miner or pool client.
// Extend mandatory members ONLY; preserve all inherited optional support.
#include <cuda_runtime.h>
#include <gmp.h>
#include "riecoin_root_reciprocal.hpp"
#include "riecoin_prime_root_precompute.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#ifdef _MSC_VER
#include <intrin.h>
#endif
namespace r200 {
using Clock=std::chrono::steady_clock;
constexpr uint32_t SPAN=268435456u,NWORDS=SPAN/32;
constexpr uint64_t OFFSET=114023297140211ull;
static void ck(cudaError_t e,const char* s){if(e!=cudaSuccess)throw std::runtime_error(std::string(s)+": "+cudaGetErrorString(e));}
static double elapsed(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
template<class T>struct Device{
 T* p=nullptr;size_t n;explicit Device(size_t k):n(k){ck(cudaMalloc(&p,k*sizeof(T)),"allocation");}
 ~Device(){if(p)cudaFree(p);}Device(const Device&)=delete;
 void put(const T* s){ck(cudaMemcpy(p,s,n*sizeof(T),cudaMemcpyHostToDevice),"upload");}
 void get(T* s){ck(cudaMemcpy(s,p,n*sizeof(T),cudaMemcpyDeviceToHost),"download");}
};
struct Hi{__device__ uint64_t operator()(uint64_t a,uint64_t b)const{return __umul64hi(a,b);}};
__device__ uint32_t rem(uint64_t x,uint32_t p,uint64_t mu){return riecoin_root_reciprocal::reduce(x,p,mu,Hi{});}
__device__ uint32_t invmod(uint32_t a,uint32_t p){
 int64_t t=0,nt=1;uint32_t r=p,nr=a;
 while(nr){uint32_t q=r/nr,z=r-q*nr;r=nr;nr=z;int64_t x=t-int64_t(q)*nt;t=nt;nt=x;}
 if(r!=1)return 0;if(t<0)t+=p;return uint32_t(t);
}
__global__ void prepare(const uint32_t* primes,uint32_t count,const uint32_t* P,uint32_t pw,
 const uint32_t* A,uint32_t aw,uint32_t* inverses,uint32_t* roots,uint32_t* errors){
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 uint32_t p=primes[i];
 if(p<=1000000000u||p>4000000000u||!(p&1u)){inverses[i]=roots[i]=0;atomicAdd(errors,1u);return;}
 uint64_t mu=UINT64_MAX/p;
 uint32_t pm=riecoin_root_reciprocal::word_remainder(P,pw,p,mu,Hi{}),iv=invmod(pm,p);
 if(!iv||rem(uint64_t(iv)*pm,p,mu)!=1){atomicAdd(errors,1u);return;}
 uint32_t am=riecoin_root_reciprocal::word_remainder(A,aw,p,mu,Hi{});
 uint32_t x=rem(uint64_t(am)+uint64_t(rem(OFFSET,p,mu))*iv,p,mu);
 inverses[i]=iv;roots[i]=x?p-x:0;
}
__global__ void exact_roots(const uint32_t* primes,uint32_t count,const uint32_t* P,uint32_t pw,
 const uint32_t* B,uint32_t bw,const uint32_t* inverses,const uint32_t* roots,uint32_t* errors){
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;uint32_t p=primes[i];
 if(p<=1000000000u||p>4000000000u||!(p&1u)){atomicAdd(errors,1u);return;}
 uint64_t bm=0,pm=0;for(uint32_t w=bw;w;--w)bm=((bm<<32)|B[w-1])%p;
 for(uint32_t w=pw;w;--w)pm=((pm<<32)|P[w-1])%p;
 uint64_t iv=inverses[i];if(iv==0||iv>=p||pm*iv%p!=1){atomicAdd(errors,1u);return;}
 for(uint32_t m=0;m<2;++m){uint64_t b=(bm+2*m)%p;uint32_t expected=uint32_t((b?p-b:0)*iv%p);
  uint32_t observed=roots[i];if(m){uint32_t d=uint32_t(2*iv%p);observed=observed>=d?observed-d:uint32_t(uint64_t(observed)+p-d);}
  if(observed!=expected)atomicAdd(errors,1u);
 }
}
struct Certificate{uint32_t factor,prime,member;};
template<bool Record>__global__ void filter(const uint32_t* primes,const uint32_t* inverses,
 const uint32_t* roots,uint32_t begin,uint32_t end,uint32_t* bits,
 Certificate* cert,uint32_t* taken,uint32_t capacity,uint32_t* errors){
 uint32_t i=begin+blockIdx.x*blockDim.x+threadIdx.x;if(i>=end)return;
 uint32_t p=primes[i],root=roots[i],iv=inverses[i];
 for(uint32_t m=0;m<2;++m){
  // All new primes exceed SPAN, so zero or one event per member/frontier.
  if(root<SPAN){uint32_t mask=1u<<(root&31u);uint32_t old=atomicAnd(bits+(root>>5),~mask);
   if constexpr(Record){if(old&mask){uint32_t j=atomicAdd(taken,1u);if(j>=capacity)atomicAdd(errors,1u);else cert[j]={root,p,m};}}
  }
  if(!m){uint32_t d=uint32_t(uint64_t(iv)*2%p);root=root>=d?root-d:uint32_t(uint64_t(root)+p-d);}
 }
}
template<class F>float timed(F f){cudaEvent_t a,b;ck(cudaEventCreate(&a),"event");ck(cudaEventCreate(&b),"event");
 ck(cudaEventRecord(a),"event start");f();ck(cudaGetLastError(),"launch");ck(cudaEventRecord(b),"event end");ck(cudaEventSynchronize(b),"wait");
 float ms=0;ck(cudaEventElapsedTime(&ms,a,b),"elapsed");cudaEventDestroy(a);cudaEventDestroy(b);return ms;}
static std::vector<uint32_t> export_words(mpz_srcptr n){std::vector<uint32_t> w((mpz_sizeinbase(n,2)+31)/32);size_t k=0;mpz_export(w.data(),&k,-1,4,0,0,n);return w;}
static uint64_t population(const std::vector<uint32_t>& b){uint64_t n=0;for(auto w:b){
#ifdef _MSC_VER
 n+=__popcnt(w);
#else
 n+=__builtin_popcount(w);
#endif
}return n;}
static std::vector<uint32_t> prime_band(uint32_t low,uint32_t high){
 auto small=riecoin_prime_root_precompute::primes_to(65536);std::vector<uint32_t> out;
 out.reserve(size_t((uint64_t(high)-low)/20));constexpr uint64_t K=1u<<18;
 std::vector<unsigned char> marked(K);
 for(uint64_t a=uint64_t(low)+1;a<=high;){if(!(a&1))++a;if(a>high)break;uint64_t b=std::min<uint64_t>(uint64_t(high)|1ull,a+2*(K-1));if(b>high)b-=2;
  size_t n=size_t((b-a)/2+1);std::fill_n(marked.begin(),n,0);
  for(auto p:small){if(p==2)continue;if(uint64_t(p)*p>b)break;
   uint64_t f=std::max(uint64_t(p)*p,((a+p-1)/p)*p);if(!(f&1))f+=p;
   for(uint64_t j=(f-a)/2;j<n;j+=p)marked[size_t(j)]=1;
  }
  for(size_t j=0;j<n;++j)if(!marked[j])out.push_back(uint32_t(a+2*j));a=b+2;
 }
 return out;
}
#ifndef R200_LIBRARY
int execute(int argc,char**argv){try{
 if(argc!=5)throw std::runtime_error("expected target_hex input_bitmap output_bitmap expected_uuid");
 ck(cudaSetDevice(0),"device");cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0),"device properties");
 std::ostringstream id;id<<"GPU-"<<std::hex<<std::setfill('0');for(unsigned i=0;i<16;++i){if(i==4||i==6||i==8||i==10)id<<'-';id<<std::setw(2)<<unsigned(static_cast<unsigned char>(prop.uuid.bytes[i]));}
 if(id.str()!=argv[4])throw std::runtime_error("GPU identity mismatch");
 size_t free=0,total=0;ck(cudaMemGetInfo(&free,&total),"memory");if(free<(size_t(4)<<30))throw std::runtime_error("requires4GiB free");
 std::ifstream in(argv[2],std::ios::binary|std::ios::ate);if(!in||in.tellg()!=std::streamoff(NWORDS*4ull))throw std::runtime_error("bitmap size");in.seekg(0);
 std::vector<uint32_t> input(NWORDS);in.read(reinterpret_cast<char*>(input.data()),input.size()*4);if(!in)throw std::runtime_error("bitmap read");
 uint64_t initial=population(input);if(initial!=140628)throw std::runtime_error("qualified R197 population required");
 mpz_t target,P,A,B,tmp,N;mpz_inits(target,P,A,B,tmp,N,nullptr);
 if(mpz_set_str(target,argv[1],16)||mpz_sizeinbase(target,2)!=1168)throw std::runtime_error("target shape");
 auto prefix=riecoin_prime_root_precompute::primes_to(1000);mpz_set_ui(P,1);for(unsigned i=0;i<114;++i)mpz_mul_ui(P,P,prefix[i]);
 mpz_fdiv_q(A,target,P);mpz_add_ui(A,A,1);mpz_mul(B,A,P);mpz_set_str(tmp,"114023297140211",10);mpz_add(B,B,tmp);
 auto pw=export_words(P),aw=export_words(A),bw=export_words(B);Device<uint32_t> dp(pw.size()),da(aw.size()),db(bw.size()),bits(NWORDS),errors(1),taken(1);
 dp.put(pw.data());da.put(aw.data());db.put(bw.data());bits.put(input.data());ck(cudaMemset(errors.p,0,4),"clear errors");ck(cudaMemset(taken.p,0,4),"clear records");
 Device<Certificate> certificates(initial);std::vector<uint32_t> expected=input,observed(NWORDS);
 uint64_t checked_roots=0;double generation_ms=0,upload_ms=0,preparation_ms=0,guard_ms=0;std::vector<float> filter_ms;
 std::cout<<"{\"phase\":\"start\",\"scope\":\"exact additional mandatory divisors, no PRP\",\"initial\":"<<initial<<",\"base_words\":"<<bw.size()<<",\"quotient_words\":"<<aw.size()<<"}"<<std::endl;
 // Stream independently reusable 1B-wide bands: no multi-GiB permanent cache mutation.
 for(uint32_t band=1;band<4;++band){uint32_t lo=band*1000000000u,hi=(band+1)*1000000000u;auto start=Clock::now();auto primes=prime_band(lo,hi);double gen=elapsed(start);generation_ms+=gen;
  Device<uint32_t> ps(primes.size()),ivs(primes.size()),rs(primes.size());start=Clock::now();ps.put(primes.data());double upload=elapsed(start);upload_ms+=upload;
  uint32_t n=uint32_t(primes.size()),blocks=(n+255)/256;
  float prep=timed([&]{prepare<<<blocks,256>>>(ps.p,n,dp.p,uint32_t(pw.size()),da.p,uint32_t(aw.size()),ivs.p,rs.p,errors.p);});preparation_ms+=prep;
  float guard=timed([&]{exact_roots<<<blocks,256>>>(ps.p,n,dp.p,uint32_t(pw.size()),db.p,uint32_t(bw.size()),ivs.p,rs.p,errors.p);});guard_ms+=guard;uint32_t err=0;errors.get(&err);if(err)throw std::runtime_error("complete root relation mismatch");checked_roots+=2ull*n;
  float fm=timed([&]{filter<true><<<blocks,256>>>(ps.p,ivs.p,rs.p,0,n,bits.p,certificates.p,taken.p,uint32_t(initial),errors.p);});filter_ms.push_back(fm);
  bits.get(observed.data());uint64_t left=population(observed);
  // Warm repeated same-input resident cost, separate from certificate collection.
  Device<uint32_t> scratch(NWORDS);std::vector<float> warm;
  for(unsigned round=0;round<5;++round){scratch.put(input.data());warm.push_back(timed([&]{filter<false><<<blocks,256>>>(ps.p,ivs.p,rs.p,0,n,scratch.p,nullptr,nullptr,0,errors.p);}));}
  std::sort(warm.begin(),warm.end());
  std::cout<<"{\"phase\":\"band\",\"low\":"<<lo<<",\"high\":"<<hi<<",\"primes\":"<<n<<",\"remaining\":"<<left<<",\"cpu_prime_generation_ms\":"<<gen<<",\"upload_ms\":"<<upload<<",\"compact_root_preparation_ms\":"<<prep<<",\"full_root_guard_ms\":"<<guard<<",\"certificate_filter_ms\":"<<fm<<",\"warm_filter_median_ms\":"<<warm[2]<<",\"resident_band_bytes\":"<<primes.size()*12ull<<"}"<<std::endl;
 }
 uint32_t count=0,err=0;taken.get(&count);errors.get(&err);if(err||count>initial)throw std::runtime_error("record bound");std::vector<Certificate> cert(count);
 ck(cudaMemcpy(cert.data(),certificates.p,count*sizeof(Certificate),cudaMemcpyDeviceToHost),"certificates");
 for(const auto& c:cert){if(c.factor>=SPAN||c.prime<=1000000000u||c.prime>4000000000u||c.member>1)throw std::runtime_error("certificate geometry");
  uint32_t mask=1u<<(c.factor&31u);if(!(expected[c.factor>>5]&mask))throw std::runtime_error("missing or duplicate deletion");
  mpz_mul_ui(N,P,c.factor);mpz_add(N,N,B);mpz_add_ui(N,N,2*c.member);
  if(mpz_cmp_ui(N,c.prime)<=0||mpz_fdiv_ui(N,c.prime))throw std::runtime_error("GMP proper divisor failure");expected[c.factor>>5]&=~mask;
 }
 bits.get(observed.data());if(observed!=expected||initial-population(observed)!=count)throw std::runtime_error("complete bitmap conservation mismatch");
 std::ofstream out(argv[3],std::ios::binary);out.write(reinterpret_cast<const char*>(observed.data()),observed.size()*4);if(!out)throw std::runtime_error("output bitmap write");
 std::cout<<"{\"phase\":\"PASS\",\"initial\":"<<initial<<",\"remaining\":"<<population(observed)<<",\"proper_divisor_certificates\":"<<count<<",\"exact_root_relations\":"<<checked_roots<<",\"full_bitmap_words_checked\":"<<NWORDS<<",\"cpu_generation_ms\":"<<generation_ms<<",\"upload_ms\":"<<upload_ms<<",\"preparation_ms\":"<<preparation_ms<<",\"guard_ms\":"<<guard_ms<<",\"all_errors\":0,\"claim\":\"new exact selection only; no measured r or miner speed\"}"<<std::endl;
 mpz_clears(target,P,A,B,tmp,N,nullptr);return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 80;}}
#endif
}
#ifndef R200_LIBRARY
int main(int argc,char**argv){return r200::execute(argc,argv);}
#endif

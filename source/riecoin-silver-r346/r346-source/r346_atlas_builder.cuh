#pragma once
#define R236_LIBRARY
#include "r236_full_atlas_test.cu"
#include "r346_shared_atlas.cuh"
namespace r346builder {
using namespace r236;
__global__ void build(const uint32_t* encoded,uint32_t n,uint64_t decode,uint64_t lower,uint64_t upper,
 const uint32_t* P,uint32_t pw,uint32_t* low,uint32_t* high,uint32_t* errors){
 const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;uint64_t iv=0;
 if(i<n){const uint64_t p=decode+encoded[i];
  if(p<=lower||p>upper||!(p&1))atomicOr(errors,1u);
  else {const uint64_t mu=UINT64_MAX/p;iv=inverse64(r229_gpu::words32(P,pw,p,mu),p);if(!iv)atomicOr(errors,2u);}
  low[i]=uint32_t(iv);
 }
 const unsigned bits=__ballot_sync(0xffffffffu,(iv>>32)!=0);
 if((threadIdx.x%32)==0&&i<n)high[i/32]=bits;
}
__global__ void certify(const uint32_t* encoded,uint32_t n,uint64_t decode,uint64_t lower,uint64_t upper,
 const uint32_t* low,const uint32_t* high,const uint32_t* P,uint32_t pw,uint32_t* errors){
 const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 const uint64_t p=decode+encoded[i],iv=uint64_t(low[i])|(uint64_t((high[i/32]>>(i%32))&1u)<<32);
 if(p<=lower||p>upper||!(p&1)||(i&&encoded[i]<=encoded[i-1])||!iv||iv>=p||
    r217_native::product(r217_native::words(P,pw,p),iv,p)!=1)atomicOr(errors,4u);
}
int owner(const char* gpu,void(*consumer)(const std::string&,const char*,HANDLE,size_t)){
 if(!consumer)throw std::runtime_error("R346 owner requires authenticated service");
 const char* lowDir=std::getenv("ASTRA_R201_CACHE"),*wideDir=std::getenv("ASTRA_R209_CACHE");
 if(!lowDir||!wideDir)throw std::runtime_error("R346 exact low/wide caches required");
 const auto started=Clock::now();r346cache::Layout layout;Context ctx(find(gpu));
 CUmemAllocationProp prop{};prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;prop.requestedHandleTypes=CU_MEM_HANDLE_TYPE_WIN32;
 prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;prop.location.id=ctx.d;
 SECURITY_ATTRIBUTES sa{sizeof(sa),nullptr,FALSE};prop.win32HandleMetaData=&sa;
 size_t granularity=0;ck(cuMemGetAllocationGranularity(&granularity,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM));
 const size_t bytes=((layout.raw+granularity-1)/granularity)*granularity;
 if(bytes>layout.raw+32*1024*1024)throw std::runtime_error("R346 granularity bound");
 Allocation allocation;ck(cuMemCreate(&allocation.h,bytes,&prop,0));Mapping map(bytes);map.bind(allocation.h,ctx.d,CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
 Mpz anchor;r235::basis(anchor.n);auto words=r235::words(anchor.n),digits=r228::digits31(anchor.n);
 Buffer<uint32_t> p(words.size()),q(digits.size()),errors(1);p.put(words);q.put(digits);errors.put({0});
 Timer timer;double uploads=0,builds=0,seals=0;
 for(unsigned j=0;j<7;++j){
  const uint64_t lower=uint64_t(j+1)*1000000000ull,upper=lower+1000000000ull,decode=j<3?0:4000000000ull;
  const std::string file=std::string(j<3?lowDir:wideDir)+"\\"+(j<3?"band-":"delta4b-band-")+std::to_string(lower)+"-"+std::to_string(upper)+".u32";
  const uint32_t n=r346cache::Counts[j];File input(file,size_t(n)*4);
  auto encoded=reinterpret_cast<uint32_t*>(map.p+layout.delta[j]);auto lo=reinterpret_cast<uint32_t*>(map.p+layout.low[j]);auto hi=reinterpret_cast<uint32_t*>(map.p+layout.high[j]);
  timer.start();ck(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(encoded),input.p,size_t(n)*4));ck(cuCtxSynchronize());uploads+=timer.end();
  timer.start();for(uint32_t f=0;f<n;f+=Chunk){const uint32_t k=std::min(Chunk,n-f);build<<<(k+Threads-1)/Threads,Threads>>>(encoded+f,k,decode,lower,upper,p.p,words.size(),lo+f,hi+f/32,errors.p);}builds+=timer.end();
  // One grid over the complete band also checks ordering across chunk boundaries.
  timer.start();certify<<<(n+Threads-1)/Threads,Threads>>>(encoded,n,decode,lower,upper,lo,hi,q.p,digits.size(),errors.p);seals+=timer.end();
  cudaCheck(cudaGetLastError(),"R346 seal kernels");clean(errors,"R346 exact anchor certificate");
  std::printf("phase=r346_seal band=%u count=%u PASS\n",j,n);std::fflush(stdout);
 }
 ck(cuCtxSynchronize());CUmemAccessDesc ro{};ro.location=prop.location;ro.flags=CU_MEM_ACCESS_FLAGS_PROT_READ;ck(cuMemSetAccess(map.p,bytes,&ro,1));
 std::printf("phase=r346_owner PASS primes=316936120 bytes=%zu upload_ms=%.6f build_ms=%.6f seal_ms=%.6f cold_wall_ms=%.6f\n",bytes,uploads,builds,seals,ms(started));std::fflush(stdout);
 HANDLE exported=nullptr;ck(cuMemExportToShareableHandle(&exported,allocation.h,CU_MEM_HANDLE_TYPE_WIN32,0));
 try{consumer("",gpu,exported,bytes);}catch(...){CloseHandle(exported);throw;}CloseHandle(exported);return 0;
}
}

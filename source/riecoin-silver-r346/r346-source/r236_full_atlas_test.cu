// Full 4..8B production-band arithmetic, shared GPU allocation, closed test.
// No admission/mining hook: all timing is component evidence, not miner yield.
#define R235_LIBRARY
#include "r235_shared_atlas_test.cu"
#include <chrono>
#include <algorithm>
namespace r236 {
using r235::Buffer;using r235::cudaCheck;using r228::Mpz;
constexpr uint32_t Counts[]={44992411,44591145,44258984,43979302};
constexpr uint64_t Total=177821842;constexpr uint32_t Chunk=1u<<20,Threads=128;
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
struct Timer{cudaEvent_t a{},b{};Timer(){cudaCheck(cudaEventCreate(&a),"event");cudaCheck(cudaEventCreate(&b),"event");}~Timer(){cudaEventDestroy(a);cudaEventDestroy(b);}void start(){cudaCheck(cudaEventRecord(a),"start");}float end(){cudaCheck(cudaEventRecord(b),"end");cudaCheck(cudaEventSynchronize(b),"end sync");float value;cudaCheck(cudaEventElapsedTime(&value,a,b),"elapsed");return value;}};
size_t align(size_t n){return (n+255)&~size_t(255);}
struct Layout{size_t delta[4],low[4],high[4],raw=4096;Layout(){for(unsigned j=0;j<4;j++){delta[j]=raw;low[j]=align(delta[j]+size_t(Counts[j])*4);high[j]=align(low[j]+size_t(Counts[j])*4);raw=align(high[j]+size_t((Counts[j]+31)/32)*4);}}};
struct View{uint32_t*delta,*low,*high;View(CUdeviceptr p,const Layout& l,unsigned j):delta(reinterpret_cast<uint32_t*>(p+l.delta[j])),low(reinterpret_cast<uint32_t*>(p+l.low[j])),high(reinterpret_cast<uint32_t*>(p+l.high[j])){}};
struct File{HANDLE h=INVALID_HANDLE_VALUE,m=nullptr;const uint32_t*p=nullptr;File(const std::string& path,size_t bytes){h=CreateFileA(path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);win(h!=INVALID_HANDLE_VALUE,"band file");LARGE_INTEGER size{};if(!GetFileSizeEx(h,&size)||uint64_t(size.QuadPart)!=bytes){CloseHandle(h);h=INVALID_HANDLE_VALUE;throw std::runtime_error("exact band length");}m=CreateFileMappingW(h,nullptr,PAGE_READONLY,0,0,nullptr);if(!m){CloseHandle(h);h=INVALID_HANDLE_VALUE;throw std::runtime_error("band mapping");}p=static_cast<const uint32_t*>(MapViewOfFile(m,FILE_MAP_READ,0,0,bytes));if(!p){CloseHandle(m);CloseHandle(h);m=nullptr;h=INVALID_HANDLE_VALUE;throw std::runtime_error("band view");}}~File(){if(p)UnmapViewOfFile(p);if(m)CloseHandle(m);if(h!=INVALID_HANDLE_VALUE)CloseHandle(h);}};
__global__ void seal(const uint32_t*delta,uint32_t n,const uint32_t*lo,const uint32_t*hi,const uint32_t*P,uint32_t pw,uint64_t lower,uint64_t upper,uint32_t*error){uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;uint64_t p=r229_gpu::low+delta[i],iv=uint64_t(lo[i])|(uint64_t((hi[i/32]>>(i%32))&1)<<32);if(p<=lower||p>upper||!(p&1)||(i&&delta[i]<=delta[i-1])||!iv||iv>=p||r217_native::product(r217_native::words(P,pw,p),iv,p)!=1)atomicOr(error,1u);}
// R208::prepare expression, copied to avoid loading an unrelated miner driver.
__global__ void original(const uint32_t*delta,uint32_t n,const uint32_t*P,uint32_t pw,const uint32_t*A,uint32_t aw,uint64_t*inv,uint64_t*roots,uint32_t*error){uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;uint64_t p=r229_gpu::low+delta[i];if(p<=r229_gpu::low||p>r229_gpu::high||!(p&1)){atomicOr(error,1u);return;}uint64_t mu=UINT64_MAX/p,pm=r229_gpu::words32(P,pw,p,mu),iv=inverse64(pm,p);if(!iv){atomicOr(error,2u);return;}uint64_t am=r229_gpu::words32(A,aw,p,mu),om=reduce66(r229_gpu::offset,0,p,mu),lo=om*iv,hi=__umul64hi(om,iv),sum=lo+am;hi+=sum<lo;uint64_t x=reduce66(sum,hi,p,mu);inv[i]=iv;roots[i]=x?p-x:0;}
__global__ void compare(const uint64_t*a,const uint64_t*b,const uint64_t*c,const uint64_t*d,uint32_t n,uint32_t*error){uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&(a[i]!=c[i]||b[i]!=d[i]))atomicOr(error,8u);}
void clean(Buffer<uint32_t>&error,const char*where){if(error.get()[0])throw std::runtime_error(where);}
int read(const char*gpu,HANDLE shared,size_t bytes){
 const auto wall=Clock::now();Layout l;if(bytes<l.raw||bytes>l.raw+32*1024*1024)throw std::runtime_error("full layout");Context ctx(find(gpu));Allocation imported;ck(cuMemImportFromShareableHandle(&imported.h,shared,CU_MEM_HANDLE_TYPE_WIN32));Mapping map(bytes);map.bind(imported.h,ctx.d,CU_MEM_ACCESS_FLAGS_PROT_READ);CUmemLocation where{};where.type=CU_MEM_LOCATION_TYPE_DEVICE;where.id=ctx.d;unsigned long long access=0;ck(cuMemGetAccess(&access,&where,map.p));if(access!=CU_MEM_ACCESS_FLAGS_PROT_READ)throw std::runtime_error("read-only requirement");
 Buffer<uint32_t>a(64),s(4),p32(64),p31(64),b31(96),error(1);Buffer<uint64_t>inv(Counts[0]),root(Counts[0]),refInv(Counts[0]),refRoot(Counts[0]);
 Mpz anchor,P,A,B,S,t;const auto basisPrimes=r235::basis(anchor.n);std::mt19937_64 random(236);Timer timer;double originalMs=0,reusedMs=0,verifyMs=0;unsigned jobs=0;
 const unsigned targets[]={1100,1144,1152,1168,1200,1300};
 for(unsigned prefix=109;prefix<=114;prefix++){
  mpz_set_ui(P.n,1);for(unsigned i=0;i<prefix;i++){r228::set64(t.n,basisPrimes[i]);mpz_mul(P.n,P.n,t.n);}if(!mpz_divisible_p(anchor.n,P.n))throw std::runtime_error("basis family");mpz_divexact(S.n,anchor.n,P.n);
  uint64_t randomWords[12];for(auto&x:randomWords)x=random();mpz_import(A.n,12,-1,8,0,0,randomWords);const unsigned abits=targets[prefix-109]-unsigned(mpz_sizeinbase(P.n,2));mpz_fdiv_r_2exp(A.n,A.n,abits);mpz_setbit(A.n,abits-1);mpz_mul(B.n,A.n,P.n);r228::set64(t.n,r229_gpu::offset);mpz_add(B.n,B.n,t.n);
  const auto aw=r235::words(A.n),sw=r235::words(S.n),pw=r235::words(P.n),pn=r228::digits31(P.n),bn=r228::digits31(B.n);a.put(aw);s.put(sw);p32.put(pw);p31.put(pn);b31.put(bn);error.put({0});
  for(unsigned j=0;j<4;j++){
   View v(map.p,l,j);const uint32_t n=Counts[j];
   auto full=[&](){timer.start();for(uint32_t f=0;f<n;f+=Chunk){uint32_t k=std::min(Chunk,n-f);original<<<(k+Threads-1)/Threads,Threads>>>(v.delta+f,k,p32.p,pw.size(),a.p,aw.size(),refInv.p+f,refRoot.p+f,error.p);}originalMs+=timer.end();};
   auto reused=[&](){timer.start();for(uint32_t f=0;f<n;f+=Chunk){uint32_t k=std::min(Chunk,n-f);r229_gpu::prepare_reused<<<(k+Threads-1)/Threads,Threads>>>(v.delta+f,k,v.low+f,v.high+f/32,a.p,aw.size(),s.p,sw.size(),inv.p+f,root.p+f,error.p);}reusedMs+=timer.end();};
   if((prefix+j)%2){full();reused();}else{reused();full();}
   timer.start();for(uint32_t f=0;f<n;f+=Chunk){uint32_t k=std::min(Chunk,n-f);r229_gpu::verify_full_job<<<(k+Threads-1)/Threads,Threads>>>(v.delta+f,k,p31.p,pn.size(),b31.p,bn.size(),inv.p+f,root.p+f,error.p);compare<<<(k+Threads-1)/Threads,Threads>>>(inv.p+f,root.p+f,refInv.p+f,refRoot.p+f,k,error.p);}verifyMs+=timer.end();cudaCheck(cudaGetLastError(),"full job kernels");clean(error,"full band relation/reference mismatch");
  }++jobs;std::printf("phase=r236_progress prefix=%u bits=%zu jobs=%u\n",prefix,mpz_sizeinbase(B.n,2),jobs);std::fflush(stdout);
 }
 CloseHandle(shared);std::printf("phase=r236_reader PASS jobs=%u primes=%llu original_ms=%.6f reused_ms=%.6f verify_compare_ms=%.6f wall_ms=%.6f\n",jobs,(unsigned long long)Total,originalMs,reusedMs,verifyMs,ms(wall));return 0;
}
int owner(const char*gpu,void(*consumer)(const std::string&,const char*,HANDLE,size_t)=nullptr,void(*poison)(CUdeviceptr,const Layout&)=nullptr){
 const auto wall=Clock::now();Layout l;Context ctx(find(gpu));CUmemAllocationProp prop{};prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;prop.requestedHandleTypes=CU_MEM_HANDLE_TYPE_WIN32;prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;prop.location.id=ctx.d;SECURITY_ATTRIBUTES sa{sizeof(sa),nullptr,FALSE};prop.win32HandleMetaData=&sa;size_t granularity=0;ck(cuMemGetAllocationGranularity(&granularity,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM));size_t bytes=((l.raw+granularity-1)/granularity)*granularity;if(bytes>l.raw+32*1024*1024)throw std::runtime_error("granularity bound");Allocation allocation;ck(cuMemCreate(&allocation.h,bytes,&prop,0));Mapping map(bytes);map.bind(allocation.h,ctx.d,CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
 Mpz anchor;r235::basis(anchor.n);auto aw=r235::words(anchor.n),pw=r228::digits31(anchor.n);Buffer<uint32_t>a(aw.size()),p(pw.size()),error(1);a.put(aw);p.put(pw);error.put({0});Timer timer;double uploadMs=0,buildMs=0,sealMs=0;
 for(unsigned j=0;j<4;j++){
  uint64_t lower=r229_gpu::low+uint64_t(j)*1000000000,upper=lower+1000000000;std::string path="C:\\Gapcoin\\stage\\ASTRA-R208-ADMISSION8B-20260909-v1\\delta4b-band-"+std::to_string(lower)+"-"+std::to_string(upper)+".u32";View v(map.p,l,j);const uint32_t n=Counts[j];File file(path,size_t(n)*4);timer.start();ck(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(v.delta),file.p,size_t(n)*4));ck(cuCtxSynchronize());uploadMs+=timer.end();
  timer.start();for(uint32_t f=0;f<n;f+=Chunk){uint32_t k=std::min(Chunk,n-f);r229_gpu::make_anchor<<<(k+Threads-1)/Threads,Threads>>>(v.delta+f,k,a.p,aw.size(),v.low+f,v.high+f/32,error.p);}buildMs+=timer.end();
  timer.start();for(uint32_t f=0;f<n;f+=Chunk){uint32_t k=std::min(Chunk,n-f);seal<<<(k+Threads-1)/Threads,Threads>>>(v.delta+f,k,v.low+f,v.high+f/32,p.p,pw.size(),lower,upper,error.p);}sealMs+=timer.end();cudaCheck(cudaGetLastError(),"anchor seal");clean(error,"independent anchor seal");std::printf("phase=r236_seal band=%u count=%u PASS\n",j,n);std::fflush(stdout);
 }
 ck(cuCtxSynchronize());std::printf("phase=r236_owner PASS primes=%llu bytes=%zu upload_ms=%.6f build_ms=%.6f seal_ms=%.6f cold_wall_ms=%.6f\n",(unsigned long long)Total,bytes,uploadMs,buildMs,sealMs,ms(wall));std::fflush(stdout);
 if(poison){poison(map.p,l);ck(cuCtxSynchronize());}
 CUmemAccessDesc immutable{};immutable.location=prop.location;immutable.flags=CU_MEM_ACCESS_FLAGS_PROT_READ;ck(cuMemSetAccess(map.p,bytes,&immutable,1));
 char path[32768];auto length=GetModuleFileNameA(nullptr,path,sizeof(path));win(length&&length<sizeof(path),"self executable");for(unsigned i=0;i<2;i++){HANDLE handle=nullptr;ck(cuMemExportToShareableHandle(&handle,allocation.h,CU_MEM_HANDLE_TYPE_WIN32,0));try{if(consumer)consumer(std::string(path,length),gpu,handle,bytes);else child(std::string(path,length),gpu,handle,bytes);}catch(...){CloseHandle(handle);throw;}CloseHandle(handle);}
 if(consumer)std::printf("phase=r236_owner child_groups=2 DONE wall_ms=%.6f\n",ms(wall));else std::printf("{\"verdict\":\"PASS\",\"scope\":\"full-band arithmetic component; not miner gain\",\"readers\":2,\"jobs\":12,\"primes\":%llu,\"bytes\":%zu,\"wall_ms\":%.6f}\n",(unsigned long long)Total,bytes,ms(wall));return 0;
}
}
#ifndef R236_LIBRARY
int main(int argc,char**argv){try{if(argc==5&&!std::strcmp(argv[1],"--reader"))return r236::read(argv[2],reinterpret_cast<HANDLE>(uintptr_t(std::stoull(argv[3]))),size_t(std::stoull(argv[4])));if(argc!=2)throw std::runtime_error("exact UUID required");return r236::owner(argv[1]);}catch(const std::exception&e){std::fprintf(stderr,"R236: %s\n",e.what());return 1;}}
#endif

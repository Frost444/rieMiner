// Composition qualification only: immutable packed arithmetic table in VRAM,
// reused by separate reader processes. Not a production cache-owner protocol.
#include "r228_invariant_atlas.hpp"
#include "r229_atlas_kernel.cu"
#define R234_LIBRARY
#include "r234_gpu_shared_test.cpp"
#include <random>
#include <iostream>
namespace r235 {
constexpr uint32_t N=65537,Threads=128,Blocks=(N+Threads-1)/Threads;
constexpr size_t Delta=4096,Low=Delta+N*4,High=Low+N*4,Raw=High+((N+31)/32)*4;
void cudaCheck(cudaError_t e,const char* why){if(e!=cudaSuccess)throw std::runtime_error(std::string(why)+": "+cudaGetErrorString(e));}
template<class T>struct Buffer{T* p{};size_t n;explicit Buffer(size_t count):n(count){cudaCheck(cudaMalloc(&p,n*sizeof(T)),"local output allocation");}~Buffer(){cudaFree(p);}void put(const std::vector<T>& v){if(v.size()>n)throw std::runtime_error("buffer bound");cudaCheck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice),"local input");}std::vector<T> get(){std::vector<T>v(n);cudaCheck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost),"readback");return v;}};
std::vector<uint32_t> words(mpz_srcptr n){std::vector<uint32_t>v((mpz_sizeinbase(n,2)+31)/32);size_t count=0;mpz_export(v.data(),&count,-1,4,0,0,n);return v;}
std::vector<uint64_t> basis(mpz_ptr anchor){std::vector<uint64_t>p;r228::Mpz t;mpz_set_ui(anchor,1);mpz_set_ui(t.n,1);for(int i=0;i<114;i++){mpz_nextprime(t.n,t.n);p.push_back(r228::get64(t.n));mpz_mul(anchor,anchor,t.n);}return p;}
struct View {uint32_t *delta,*low,*high;explicit View(CUdeviceptr p):delta(reinterpret_cast<uint32_t*>(p+Delta)),low(reinterpret_cast<uint32_t*>(p+Low)),high(reinterpret_cast<uint32_t*>(p+High)) {}};
int read(const char* gpu,HANDLE shared,size_t bytes){
 if(bytes<Raw||bytes>16*1024*1024)throw std::runtime_error("closed atlas shape");Context ctx(find(gpu));Allocation imported;ck(cuMemImportFromShareableHandle(&imported.h,shared,CU_MEM_HANDLE_TYPE_WIN32));Mapping map(bytes);map.bind(imported.h,ctx.d,CU_MEM_ACCESS_FLAGS_PROT_READ);View v(map.p);
 CUmemLocation location{};location.type=CU_MEM_LOCATION_TYPE_DEVICE;location.id=ctx.d;unsigned long long access=0;ck(cuMemGetAccess(&access,&location,map.p));if(access!=CU_MEM_ACCESS_FLAGS_PROT_READ)throw std::runtime_error("atlas not read-only");
 std::vector<uint32_t>delta(N);ck(cuMemcpyDtoH(delta.data(),reinterpret_cast<CUdeviceptr>(v.delta),N*4));
 Buffer<uint32_t>a(64),s(4),p31(64),b31(96),errors(1);Buffer<uint64_t>inverse(N),roots(N);
 r228::Mpz anchor,P,A,B,S,t,p,iv;const auto basisPrimes=basis(anchor.n);std::mt19937_64 random(235);uint64_t relations=0;unsigned jobs=0;
 for(unsigned prefix=109;prefix<=114;prefix++){
  mpz_set_ui(P.n,1);for(unsigned i=0;i<prefix;i++){r228::set64(t.n,basisPrimes[i]);mpz_mul(P.n,P.n,t.n);}if(!mpz_divisible_p(anchor.n,P.n))throw std::runtime_error("family relation");mpz_divexact(S.n,anchor.n,P.n);
  for(unsigned trial=0;trial<8;trial++){
   uint64_t randomWords[12];for(auto& x:randomWords)x=random();mpz_import(A.n,1+(trial*3)%12,-1,8,0,0,randomWords);if(!trial)mpz_set_ui(A.n,0);
   mpz_mul(B.n,A.n,P.n);r228::set64(t.n,r229_gpu::offset);mpz_add(B.n,B.n,t.n);
   const auto aw=words(A.n),sw=words(S.n),pw=r228::digits31(P.n),bw=r228::digits31(B.n);a.put(aw);s.put(sw);p31.put(pw);b31.put(bw);errors.put({0});
   r229_gpu::prepare_reused<<<Blocks,Threads>>>(v.delta,N,v.low,v.high,a.p,aw.size(),s.p,sw.size(),inverse.p,roots.p,errors.p);
   r229_gpu::verify_full_job<<<Blocks,Threads>>>(v.delta,N,p31.p,pw.size(),b31.p,bw.size(),inverse.p,roots.p,errors.p);
   cudaCheck(cudaGetLastError(),"shared atlas kernels");ck(cuCtxSynchronize());if(errors.get()[0])throw std::runtime_error("independent complete root verification");
   const auto root=roots.get(),inv=inverse.get();relations+=uint64_t(N)*2;
   for(unsigned k=0;k<17;k++){const size_t index=(trial*2053+k*3881)%N;r228::set64(p.n,r229_gpu::low+delta[index]);mpz_invert(iv.n,P.n,p.n);mpz_neg(t.n,B.n);mpz_mul(t.n,t.n,iv.n);mpz_mod(t.n,t.n,p.n);if(inv[index]!=r228::get64(iv.n)||root[index]!=r228::get64(t.n))throw std::runtime_error("independent GMP sample");relations+=2;}
   ++jobs;
  }
 }
 CloseHandle(shared);std::printf("phase=r235_reader PASS jobs=%u relations=%llu access=READ_ONLY\n",jobs,static_cast<unsigned long long>(relations));return 0;
}
int owner(const char* gpu){
 Context ctx(find(gpu));CUmemAllocationProp prop{};prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;prop.requestedHandleTypes=CU_MEM_HANDLE_TYPE_WIN32;prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;prop.location.id=ctx.d;SECURITY_ATTRIBUTES sa{sizeof(sa),nullptr,FALSE};prop.win32HandleMetaData=&sa;
 size_t granularity=0;ck(cuMemGetAllocationGranularity(&granularity,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM));const size_t bytes=((Raw+granularity-1)/granularity)*granularity;if(bytes>16*1024*1024)throw std::runtime_error("granularity bound");
 Allocation allocation;ck(cuMemCreate(&allocation.h,bytes,&prop,0));Mapping map(bytes);map.bind(allocation.h,ctx.d,CU_MEM_ACCESS_FLAGS_PROT_READWRITE);View v(map.p);
 r228::Mpz anchor,p,iv;basis(anchor.n);std::vector<uint32_t>delta;
 for(uint64_t start:{4000000000ull,4294967000ull,5990000000ull,7990000000ull}){r228::set64(p.n,start);const size_t end=delta.size()+16384+(start==7990000000ull);while(delta.size()<end){mpz_nextprime(p.n,p.n);delta.push_back(uint32_t(r228::get64(p.n)-r229_gpu::low));}}
 ck(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(v.delta),delta.data(),N*4));const auto aw=words(anchor.n);Buffer<uint32_t>ad(aw.size()),errors(1);ad.put(aw);errors.put({0});
 r229_gpu::make_anchor<<<Blocks,Threads>>>(v.delta,N,ad.p,aw.size(),v.low,v.high,errors.p);cudaCheck(cudaGetLastError(),"anchor construction");ck(cuCtxSynchronize());if(errors.get()[0])throw std::runtime_error("anchor error");
 std::vector<uint32_t>lo(N),hi((N+31)/32);ck(cuMemcpyDtoH(lo.data(),reinterpret_cast<CUdeviceptr>(v.low),N*4));ck(cuMemcpyDtoH(hi.data(),reinterpret_cast<CUdeviceptr>(v.high),hi.size()*4));
 for(size_t i=0;i<N;i++){r228::set64(p.n,r229_gpu::low+delta[i]);if(!mpz_invert(iv.n,anchor.n,p.n))throw std::runtime_error("anchor invert");if((uint64_t(lo[i])|(uint64_t((hi[i/32]>>(i%32))&1)<<32))!=r228::get64(iv.n))throw std::runtime_error("GMP anchor seal");}if(hi.back()&~1u)throw std::runtime_error("padding");
 ck(cuCtxSynchronize());std::printf("phase=r235_owner anchor_sealed=PASS primes=%u bytes=%zu\n",N,bytes);std::fflush(stdout);
 char path[32768];const auto length=GetModuleFileNameA(nullptr,path,sizeof(path));win(length&&length<sizeof(path),"self executable");
 for(unsigned i=0;i<2;i++){HANDLE handle=nullptr;ck(cuMemExportToShareableHandle(&handle,allocation.h,CU_MEM_HANDLE_TYPE_WIN32,0));try{child(std::string(path,length),gpu,handle,bytes);}catch(...){CloseHandle(handle);throw;}CloseHandle(handle);}
 std::printf("{\"verdict\":\"PASS\",\"scope\":\"shared GPU arithmetic atlas; not miner gain\",\"readers\":2,\"jobs\":96,\"primes\":%u,\"bytes\":%zu,\"consumer_access\":\"read_only\"}\n",N,bytes);return 0;
}
}
#ifndef R235_LIBRARY
int main(int argc,char**argv){try{if(argc==5&&!std::strcmp(argv[1],"--reader"))return r235::read(argv[2],reinterpret_cast<HANDLE>(uintptr_t(std::stoull(argv[3]))),size_t(std::stoull(argv[4])));if(argc!=2)throw std::runtime_error("expected GPU UUID");return r235::owner(argv[1]);}catch(const std::exception& e){std::fprintf(stderr,"R235: %s\n",e.what());return 1;}}
#endif

#pragma once
// Seven-band immutable allocation. All per-job inverse/root certificates remain independent.
// Independent R217 per-job certification remains mandatory and unchanged.
#include <cuda.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <memory>
namespace r346cache {
constexpr uint32_t Counts[]={47374753,46227250,45512275,44992411,44591145,44258984,43979302};
inline size_t align(size_t n){return (n+255)&~size_t(255);}
struct Layout{size_t delta[7],low[7],high[7],raw=4096;Layout(){for(unsigned j=0;j<7;j++){delta[j]=raw;low[j]=align(delta[j]+size_t(Counts[j])*4);high[j]=align(low[j]+size_t(Counts[j])*4);raw=align(high[j]+size_t((Counts[j]+31)/32)*4);}}};
inline void driver(CUresult e){if(e!=CUDA_SUCCESS){const char* name="unknown";cuGetErrorName(e,&name);throw std::runtime_error(std::string("R346 import: ")+name);}}
struct View {
 Layout layout;CUmemGenericAllocationHandle handle{};CUdeviceptr address{};size_t bytes=0;bool mapped=false;
#ifdef _WIN32
 View(HANDLE inherited,size_t n):bytes(n){try{
  if(n<layout.raw||n>layout.raw+32*1024*1024)throw std::runtime_error("R346 atlas size");
  driver(cuMemImportFromShareableHandle(&handle,inherited,CU_MEM_HANDLE_TYPE_WIN32));CloseHandle(inherited);inherited=nullptr;
  CUdevice device;driver(cuCtxGetDevice(&device));CUmemAllocationProp prop{};driver(cuMemGetAllocationPropertiesFromHandle(&prop,handle));
  if(prop.type!=CU_MEM_ALLOCATION_TYPE_PINNED||prop.location.type!=CU_MEM_LOCATION_TYPE_DEVICE||prop.location.id!=device)throw std::runtime_error("R346 device location");
  driver(cuMemAddressReserve(&address,bytes,0,0,0));driver(cuMemMap(address,bytes,0,handle,0));mapped=true;
  CUmemAccessDesc access{};access.location=prop.location;access.flags=CU_MEM_ACCESS_FLAGS_PROT_READ;driver(cuMemSetAccess(address,bytes,&access,1));
 }catch(...){if(inherited)CloseHandle(inherited);release();throw;}}
#else
 // RC4: Linux has no authenticated Windows atlas lease. Keep the existing
 // complete per-job preparation path; never manufacture an imported view.
 View()=delete;
#endif
 ~View(){release();}void release(){if(mapped)cuMemUnmap(address,bytes);if(address)cuMemAddressFree(address,bytes);if(handle)cuMemRelease(handle);mapped=false;address=0;handle=0;}
 uint32_t* delta(unsigned j)const{return reinterpret_cast<uint32_t*>(address+layout.delta[j]);}
 uint32_t* low(unsigned j)const{return reinterpret_cast<uint32_t*>(address+layout.low[j]);}
 uint32_t* high(unsigned j)const{return reinterpret_cast<uint32_t*>(address+layout.high[j]);}
};
inline std::vector<uint32_t> ratio(mpz_srcptr P){mpz_t anchor,p,s;mpz_inits(anchor,p,s,nullptr);mpz_set_ui(anchor,1);mpz_set_ui(p,1);bool family=false;for(unsigned i=1;i<=114;i++){mpz_nextprime(p,p);mpz_mul(anchor,anchor,p);if(i>=109&&mpz_cmp(anchor,P)==0)family=true;}if(!family||!mpz_divisible_p(anchor,P)){mpz_clears(anchor,p,s,nullptr);throw std::runtime_error("R346 exact P109..114 family");}mpz_divexact(s,anchor,P);std::vector<uint32_t>w((mpz_sizeinbase(s,2)+31)/32);size_t n=0;mpz_export(w.data(),&n,-1,4,0,0,s);mpz_clears(anchor,p,s,nullptr);return w;}
}

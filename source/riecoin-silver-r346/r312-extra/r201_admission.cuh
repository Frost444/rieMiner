#pragma once
#define R200_LIBRARY
#include "r200_admission.cu"
#undef R200_LIBRARY
#include <array>
#include <filesystem>
#include <memory>
#if defined(R202_MAPPED_INPUTS) && defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
namespace r201 {
using r200::Device;using r200::ck;
struct Band{uint32_t n=0;std::unique_ptr<Device<uint32_t>> p,inv,root,phase;};
static std::vector<Band> bands;
static std::unique_ptr<Device<uint32_t>> error;
static bool enabled=false,valid=false;
static uint64_t next_origin=0,calls=0,reference_calls=0,rebindings=0;
static uint32_t prior_span=0;
#ifdef R202_MAPPED_INPUTS
#ifdef _WIN32
struct Mapping{
 HANDLE file=INVALID_HANDLE_VALUE,mapping=nullptr;const uint32_t* data=nullptr;
 explicit Mapping(const std::filesystem::path& path,uint64_t expected_bytes){
  file=CreateFileW(path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
  if(file==INVALID_HANDLE_VALUE)throw std::runtime_error("R202 read-only file lock");
  LARGE_INTEGER bytes{};if(!GetFileSizeEx(file,&bytes)||bytes.QuadPart<0||uint64_t(bytes.QuadPart)!=expected_bytes){CloseHandle(file);file=INVALID_HANDLE_VALUE;throw std::runtime_error("R202 locked file size");}
  mapping=CreateFileMappingW(file,nullptr,PAGE_READONLY,0,0,nullptr);
  if(mapping)data=static_cast<const uint32_t*>(MapViewOfFile(mapping,FILE_MAP_READ,0,0,0));
  if(!data){if(mapping)CloseHandle(mapping);CloseHandle(file);file=INVALID_HANDLE_VALUE;throw std::runtime_error("R202 read-only mapping");}
 }
 ~Mapping(){if(data)UnmapViewOfFile(data);if(mapping)CloseHandle(mapping);if(file!=INVALID_HANDLE_VALUE)CloseHandle(file);}
};
#else
// RC2:ADMISSION-SNAPSHOT -- GPU root checks consume this owned, bounded copy.
// The file is not a proof; every supplied divisor/root relation is verified.
struct Mapping{
 std::vector<uint32_t> storage;const uint32_t* data=nullptr;
 explicit Mapping(const std::filesystem::path& path,uint64_t expected_bytes){
  if(!expected_bytes||expected_bytes%4||expected_bytes>536870912ull||std::filesystem::file_size(path)!=expected_bytes)
   throw std::runtime_error("R202 input size");
  storage.resize(static_cast<size_t>(expected_bytes/4));
  std::ifstream in(path,std::ios::binary);in.read(reinterpret_cast<char*>(storage.data()),static_cast<std::streamsize>(expected_bytes));
  if(!in||in.peek()!=std::char_traits<char>::eof())throw std::runtime_error("R202 input snapshot incomplete");
  data=storage.data();
 }
};
#endif
#endif
__global__ void strike(const uint32_t* primes,const uint32_t* inverse,const uint32_t* roots,
 uint32_t* phases,uint32_t count,uint64_t origin,uint32_t span,bool reuse,bool advance,uint32_t* bits){
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 uint32_t p=primes[i],iv=inverse[i],r;
 if(reuse)r=phases[i];else {uint32_t o=uint32_t(origin%p),a=roots[i];r=a>=o?a-o:uint32_t(uint64_t(a)+p-o);}
 if(advance)phases[i]=r>=span?r-span:uint32_t(uint64_t(r)+p-span);
 for(uint32_t m=0;m<2;++m){if(r<span)atomicAnd(bits+(r>>5u),~(1u<<(r&31u)));
  if(!m){uint64_t twice=uint64_t(iv)*2u;uint32_t d=uint32_t(twice>=p?twice-p:twice);r=r>=d?r-d:uint32_t(uint64_t(r)+p-d);}}
}
static void init(mpz_srcptr P,mpz_srcptr B,const std::array<uint32_t,7>& offsets,uint32_t inherited_limit){
 const char* mode=std::getenv("ASTRA_R201_LIMIT");uint64_t limit=mode?std::strtoull(mode,nullptr,10):0;
 if(!limit)return;if(limit!=2000000000ull&&limit!=4000000000ull)throw std::runtime_error("R201 unsupported closed limit");
 if(inherited_limit!=1000000000u||offsets[0]!=0||offsets[1]!=2||mpz_sgn(P)<=0||mpz_cmp_ui(B,static_cast<unsigned long>(limit))<=0)throw std::runtime_error("R201 inherited/proper-divisor geometry");
 mpz_t a,o;mpz_inits(a,o,nullptr);mpz_fdiv_qr(a,o,B,P);mpz_t expected;mpz_init_set_str(expected,"114023297140211",10);
 if(mpz_cmp(o,expected)!=0){mpz_clears(a,o,expected,nullptr);throw std::runtime_error("R201 compact origin requires qualified offset");}
 auto pw=r200::export_words(P),aw=r200::export_words(a),bw=r200::export_words(B);mpz_clears(a,o,expected,nullptr);
 Device<uint32_t> dp(pw.size()),da(aw.size()),db(bw.size());dp.put(pw.data());da.put(aw.data());db.put(bw.data());
 error=std::make_unique<Device<uint32_t>>(1);ck(cudaMemset(error->p,0,4),"R201 error reset");
 const char* dir=std::getenv("ASTRA_R201_CACHE");if(!dir||!*dir)throw std::runtime_error("R201 explicit isolated cache required");
 std::filesystem::create_directories(dir);const uint32_t counts[]={47374753,46227250,45512275};
 for(uint32_t b=1;b<limit/1000000000ull;++b){auto start=r200::Clock::now();uint32_t lo=b*1000000000u,hi=(b+1)*1000000000u;
  auto path=std::filesystem::path(dir)/("band-"+std::to_string(lo)+"-"+std::to_string(hi)+".u32");std::vector<uint32_t> primes;bool reused=false;
#ifdef R202_MAPPED_INPUTS
  // Qualification supplies already generated R200/R201 bands, locked read-only.
  // Mathematical safety is proved for EVERY entry on GPU; neither primality
  // nor ordering of this acceleration data is an assumption of safe deletion.
  if(!std::filesystem::exists(path)||std::filesystem::file_size(path)!=uint64_t(counts[b-1])*4)throw std::runtime_error("R202 prepared band required");
  Mapping mapped(path,uint64_t(counts[b-1])*4);const uint32_t* source=mapped.data;uint32_t count=counts[b-1];reused=true;
#else
  if(std::filesystem::exists(path)){if(std::filesystem::file_size(path)!=uint64_t(counts[b-1])*4)throw std::runtime_error("R201 cached band size");
   primes.resize(counts[b-1]);std::ifstream f(path,std::ios::binary);f.read(reinterpret_cast<char*>(primes.data()),primes.size()*4);if(!f)throw std::runtime_error("R201 cache read");reused=true;
  }else{primes=r200::prime_band(lo,hi);if(primes.size()!=counts[b-1])throw std::runtime_error("R201 generated count");
   auto temporary=path;temporary+=".writing";if(std::filesystem::exists(temporary))throw std::runtime_error("R201 unresolved cache write");
   {std::ofstream f(temporary,std::ios::binary);f.write(reinterpret_cast<const char*>(primes.data()),primes.size()*4);if(!f)throw std::runtime_error("R201 cache write");}
   std::filesystem::rename(temporary,path);
  }
  // Exact divisor safety does not assume cached values are prime. Range/order
  // plus the complete P*inverse and full-base-root guard below are mandatory.
  uint32_t previous=lo;for(auto p:primes){if(p<=previous||p>hi||!(p&1))throw std::runtime_error("R201 band integrity");previous=p;}
  const uint32_t* source=primes.data();uint32_t count=uint32_t(primes.size());
#endif
  double host_ms=r200::elapsed(start);Band band;band.n=count;
  band.p=std::make_unique<Device<uint32_t>>(band.n);band.inv=std::make_unique<Device<uint32_t>>(band.n);
  band.root=std::make_unique<Device<uint32_t>>(band.n);band.phase=std::make_unique<Device<uint32_t>>(band.n);band.p->put(source);
  uint32_t blocks=(band.n+255u)/256u;
  r200::prepare<<<blocks,256>>>(band.p->p,band.n,dp.p,uint32_t(pw.size()),da.p,uint32_t(aw.size()),band.inv->p,band.root->p,error->p);
  r200::exact_roots<<<blocks,256>>>(band.p->p,band.n,dp.p,uint32_t(pw.size()),db.p,uint32_t(bw.size()),band.inv->p,band.root->p,error->p);
  ck(cudaGetLastError(),"R201 roots");uint32_t errors=0;error->get(&errors);if(errors)throw std::runtime_error("R201 complete root guard");
  std::cout<<"phase=r201_admission_setup high="<<hi<<" factors="<<band.n<<" cache_reused="<<reused<<" host_prepare_ms="<<host_ms<<" complete_ms="<<r200::elapsed(start)<<" exact_root_relations="<<uint64_t(band.n)*2<<" errors=0 resident_bytes="<<uint64_t(band.n)*16<<std::endl;
  bands.push_back(std::move(band));
 }
 enabled=true;valid=false;
}
static void apply(uint32_t* bits,uint64_t origin,uint64_t span,bool advance){
 if(!enabled)return;if(!span||span>r200::SPAN||origin>UINT64_MAX-span)throw std::runtime_error("R201 span bound");
 bool reuse=advance&&valid&&next_origin==origin&&prior_span==span;if(advance){++calls;if(!reuse)++rebindings;}else ++reference_calls;
 for(auto& b:bands){strike<<<(b.n+255u)/256u,256>>>(b.p->p,b.inv->p,b.root->p,b.phase->p,b.n,origin,uint32_t(span),reuse,advance,bits);ck(cudaGetLastError(),"R201 strike");}
 if(advance){next_origin=origin+span;prior_span=uint32_t(span);valid=true;}
}
static void cleanup(){if(!enabled)return;ck(cudaDeviceSynchronize(),"R201 terminal guard");
 std::cout<<"phase=r201_admission_complete frontiers="<<calls<<" cold_reference_macros="<<reference_calls<<" rebindings="<<rebindings<<" errors=0"<<std::endl;
 bands.clear();error.reset();enabled=valid=false;
}
}

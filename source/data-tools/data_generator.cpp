// Rebuild public, deterministic Riecoin precomputation locally. Never downloads.
// Output format is RCPI0001 / little-endian u32, unchanged from qualified R346.
#define NOMINMAX
#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#else
#include <openssl/evp.h>
#endif
#include <gmp.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "riecoin_prime_root_precompute.hpp"
namespace fs=std::filesystem;
namespace exact=riecoin_prime_root_precompute;
static void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
struct Hash {
#ifdef _WIN32
 BCRYPT_ALG_HANDLE alg=nullptr;BCRYPT_HASH_HANDLE value=nullptr;
 Hash(){check(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0)>=0,"SHA provider");check(BCryptCreateHash(alg,&value,nullptr,0,nullptr,0,0)>=0,"SHA state");}
 ~Hash(){if(value)BCryptDestroyHash(value);if(alg)BCryptCloseAlgorithmProvider(alg,0);}
 void add(const void* p,size_t n){check(n<=ULONG_MAX,"Hash chunk");check(BCryptHashData(value,(PUCHAR)p,ULONG(n),0)>=0,"SHA update");}
 std::array<unsigned char,32> finish(){std::array<unsigned char,32> b{};check(BCryptFinishHash(value,b.data(),b.size(),0)>=0,"SHA finish");return b;}
#else
 EVP_MD_CTX* value=nullptr;
 Hash(){value=EVP_MD_CTX_new();check(value!=nullptr,"SHA state");if(EVP_DigestInit_ex(value,EVP_sha256(),nullptr)!=1){EVP_MD_CTX_free(value);value=nullptr;throw std::runtime_error("SHA provider");}}
 ~Hash(){EVP_MD_CTX_free(value);}
 void add(const void* p,size_t n){check(EVP_DigestUpdate(value,p,n)==1,"SHA update");}
 std::array<unsigned char,32> finish(){std::array<unsigned char,32>b{};unsigned n=0;check(EVP_DigestFinal_ex(value,b.data(),&n)==1&&n==b.size(),"SHA finish");return b;}
#endif
};
static fs::path cancel_file;
#ifdef _WIN32
static HANDLE guardian=nullptr;
static void cancelled(){if((!cancel_file.empty()&&fs::exists(cancel_file))||(guardian&&WaitForSingleObject(guardian,0)==WAIT_OBJECT_0))throw std::runtime_error("Preparation cancelled; completed files are preserved");}
#else
static void cancelled(){if(!cancel_file.empty()&&fs::exists(cancel_file))throw std::runtime_error("Preparation cancelled; completed files are preserved");}
#endif
static void progress(const char* phase,uint64_t done,uint64_t total){std::cout<<"{\"phase\":\""<<phase<<"\",\"done\":"<<done<<",\"total\":"<<total<<"}"<<std::endl;cancelled();}
static void put32(std::vector<unsigned char>& out,uint32_t v){for(int i=0;i<4;++i)out.push_back(static_cast<unsigned char>(v>>(i*8)));}
struct Output {
 fs::path path;std::ofstream stream;bool committed=false;Hash digest;
 explicit Output(const fs::path& p):path(p){check(!fs::exists(p),"Output already exists");stream.open(p,std::ios::binary);check(bool(stream),"Cannot create output");}
 ~Output(){stream.close();if(!committed){std::error_code ignored;fs::remove(path,ignored);}}
 void write(const void* p,size_t n){stream.write(static_cast<const char*>(p),n);check(bool(stream),"Output write failed");digest.add(p,n);}
 void close(){stream.flush();check(bool(stream),"Output flush failed");stream.close();check(bool(stream),"Output close failed");committed=true;}
};
// Same segmented, odd-only prime enumeration as the qualified cache generator.
static void band(uint64_t low,uint64_t high,uint64_t offset,const fs::path& file){
 check(low<high&&high<=8000000000ULL&&offset<=low&&high-offset<=UINT32_MAX,"Invalid band bounds");
 Output out(file);const auto factors=exact::primes_to(100000);constexpr size_t segment=1<<18;
 std::vector<unsigned char> composite(segment);std::vector<uint32_t> words;words.reserve(segment);
 int reported=-1;uint64_t count=0;
 if(low<2&&high>=2){uint32_t p=uint32_t(2-offset);out.write(&p,4);++count;}
 for(uint64_t a=std::max<uint64_t>(low+1,3);a<=high;){
  if(!(a&1))++a;if(a>high)break;
  uint64_t b=std::min<uint64_t>(high,a+2*(segment-1));if(!(b&1))--b;
  const size_t n=size_t((b-a)/2+1);std::fill_n(composite.begin(),n,0);words.clear();
  for(uint32_t p:factors){if(p==2)continue;uint64_t square=uint64_t(p)*p;if(square>b)break;
   uint64_t first=std::max(square,((a+p-1)/p)*p);if(!(first&1))first+=p;
   for(uint64_t j=(first-a)/2;j<n;j+=p)composite[size_t(j)]=1;
  }
  for(size_t j=0;j<n;++j)if(!composite[j])words.push_back(uint32_t(a+2*j-offset));
  out.write(words.data(),words.size()*4);count+=words.size();a=b+2;
  int pct=int(100*(std::min(a,high)-low)/(high-low));if(pct!=reported){reported=pct;progress("primes",std::min(a,high)-low,high-low);}
 }
 out.close();std::cout<<"{\"status\":\"GENERATED\",\"values\":"<<count<<"}"<<std::endl;
}
static void basis(uint32_t limit,uint32_t first,const fs::path& file){
 check(limit>=2&&limit<=1000000000U&&first>0&&first<=1024,"Invalid basis bounds");
 cancelled();progress("basis_primes",0,limit);auto primes=exact::primes_to(limit);cancelled();
 check(first<primes.size(),"Too few primes for this primorial");
 mpz_t product;mpz_init_set_ui(product,1);for(uint32_t i=0;i<first;++i)mpz_mul_ui(product,product,primes[i]);
 try {
  std::vector<unsigned char> primorial((mpz_sizeinbase(product,2)+7)/8);size_t written=0;mpz_export(primorial.data(),&written,1,1,1,0,product);primorial.resize(written);
  std::vector<unsigned char> header{'R','C','P','I','0','0','0','1'};
  for(uint32_t v:{1U,1U,limit,first,uint32_t(primorial.size()),uint32_t(primes.size()),uint32_t(primes.size()-first)})put32(header,v);
  header.insert(header.end(),primorial.begin(),primorial.end());
  Output out(file);Hash body;auto write=[&](const void* p,size_t n){out.write(p,n);body.add(p,n);};
  write(header.data(),header.size());write(primes.data(),primes.size()*4);
  constexpr size_t chunk=1<<20;std::vector<uint32_t> inverses(chunk);
  const unsigned workers=std::max(1U,std::min(4U,std::thread::hardware_concurrency()/2));
  for(size_t start=first;start<primes.size();start+=chunk){
   cancelled();size_t n=std::min(chunk,primes.size()-start);std::vector<std::thread> threads;
   for(unsigned t=0;t<workers;++t)threads.emplace_back([&,t]{for(size_t i=n*t/workers;i<n*(t+1)/workers;++i){uint32_t p=primes[start+i];inverses[i]=exact::inverse_mod(uint32_t(mpz_fdiv_ui(product,p)),p);}});
   for(auto& t:threads)t.join();write(inverses.data(),n*4);progress("basis_inverses",start+n-first,primes.size()-first);
  }
  auto digest=body.finish();out.write(digest.data(),digest.size());out.close();
  std::cout<<"{\"status\":\"GENERATED\",\"primes\":"<<primes.size()<<"}"<<std::endl;
 }catch(...){mpz_clear(product);throw;}mpz_clear(product);
}
#ifdef _WIN32
static uint64_t number(const wchar_t* text){std::wstring s(text);check(!s.empty()&&std::all_of(s.begin(),s.end(),[](wchar_t c){return c>=L'0'&&c<=L'9';}),"Invalid integer");return std::stoull(s);}
int wmain(int argc,wchar_t** argv){try{
 if(const wchar_t* value=_wgetenv(L"HORIZON_PREPARE_GUARDIAN_PID")){const auto pid=number(value);check(pid>0&&pid<=UINT32_MAX,"Guardian PID");guardian=OpenProcess(SYNCHRONIZE,FALSE,DWORD(pid));check(guardian!=nullptr,"Preparation guardian missing");}
 check(argc>=2,"Usage: data-generator basis LIMIT FIRST OUTPUT [CANCEL_FILE] | band LOW HIGH OFFSET OUTPUT [CANCEL_FILE]");
 const std::wstring mode=argv[1];
 if(mode==L"basis"){check(argc==5||argc==6,"Invalid basis arguments");if(argc==6)cancel_file=argv[5];const auto limit=number(argv[2]),first=number(argv[3]);check(limit<=1000000000ULL&&first<=1024,"Basis range");basis(uint32_t(limit),uint32_t(first),argv[4]);}
 else if(mode==L"band"){check(argc==6||argc==7,"Invalid band arguments");if(argc==7)cancel_file=argv[6];band(number(argv[2]),number(argv[3]),number(argv[4]),argv[5]);}
 else throw std::runtime_error("Invalid mode");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
#else
static uint64_t number(const char* text){std::string s(text);check(!s.empty()&&std::all_of(s.begin(),s.end(),[](char c){return c>='0'&&c<='9';}),"Invalid integer");return std::stoull(s);}
int main(int argc,char** argv){try{
 check(argc>=2,"Usage: data-generator basis LIMIT FIRST OUTPUT [CANCEL_FILE] | band LOW HIGH OFFSET OUTPUT [CANCEL_FILE]");
 const std::string mode=argv[1];
 if(mode=="basis"){check(argc==5||argc==6,"Invalid basis arguments");if(argc==6)cancel_file=argv[5];const auto limit=number(argv[2]),first=number(argv[3]);check(limit<=1000000000ULL&&first<=1024,"Basis range");basis(uint32_t(limit),uint32_t(first),argv[4]);}
 else if(mode=="band"){check(argc==6||argc==7,"Invalid band arguments");if(argc==7)cancel_file=argv[6];band(number(argv[2]),number(argv[3]),number(argv[4]),argv[5]);}
 else throw std::runtime_error("Invalid mode");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
#endif

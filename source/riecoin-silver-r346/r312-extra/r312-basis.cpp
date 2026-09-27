// CPU-only admission/basis preparation. No miner, GPU or production cache writes.
#define RIECOIN_RLOW_PRIME_ROOT_CACHE_MAX_BYTES 536870912u
#include "riecoin_prime_root_cache.hpp"
#include "r312-middle-bounds.hpp"
#include <iostream>
#include <limits>
namespace cache=riecoin_prime_root_cache;
static void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
static unsigned admission(){
 mpz_t p,t,b,f,n,d;mpz_inits(p,t,b,f,n,d,nullptr);unsigned checks=0;
 const auto primes=cache::exact::primes_to(2000);
 const auto maximum=std::numeric_limits<std::uint64_t>::max();
 for(unsigned bits:{1144u,1154u,1168u,1184u})for(unsigned pn:{32u,48u,64u,80u}){
  mpz_set_ui(p,1);for(unsigned j=0;j<pn;j++)mpz_mul_ui(p,p,primes[j]);
  mpz_set_ui(t,1);mpz_mul_2exp(t,t,264);mpz_add_ui(t,t,0x1234567);
  mpz_mul_2exp(t,t,bits-265);mpz_mod(b,t,p);mpz_sub(b,p,b);mpz_add(b,b,t);
  mpz_set_str(d,"114023297140211",10);mpz_add(b,b,d);
  const bool expected=pn<=64;check(r312::bounds(t,b,p,maximum,20)==expected,"whole-range admission");checks++;
  for(std::uint64_t factor:{0ull,1ull,0x8000000000000000ull,0xffffffffffffffffull}){
   mpz_import(f,1,-1,sizeof(factor),0,0,&factor);mpz_mul(n,p,f);mpz_add(n,n,b);
   for(unsigned offset:{0u,2u,6u,8u,12u,18u,20u}){
    mpz_add_ui(d,n,offset);bool zero=true;for(unsigned bit=512;bit<864;bit++)zero=zero&&!mpz_tstbit(d,bit);
    if(expected)check(zero,"admitted actual member gap");checks++;
   }
  }
  mpz_setbit(t,512);check(!r312::bounds(t,b,p,maximum,20),"nonzero target gap must reject");checks++;
 }
 // Exact2^512 displacement boundary; odd base preserves the admitted parity.
 mpz_set_ui(t,1);mpz_mul_2exp(t,t,1143);mpz_set_ui(p,2);
 mpz_set_ui(d,1);mpz_mul_2exp(d,d,512);mpz_sub_ui(d,d,1);mpz_add(b,t,d);
 check(r312::bounds(t,b,p,0,0),"last admitted gap boundary");checks++;
 check(!r312::bounds(t,b,p,0,1),"first excluded gap boundary");checks++;
 mpz_clears(p,t,b,f,n,d,nullptr);return checks;
}
int main(int argc,char**argv){try{
 const auto checks=admission();std::cout<<"R312_CPU_ADMISSION_PASS checks="<<checks<<" errors=0"<<std::endl;
 if(argc==2&&std::string(argv[1])=="--self-test")return 0;
 check(argc==3&&std::string(argv[1])=="--build","--self-test or --build UNIQUE_LAB_CACHE_DIR");
 const std::filesystem::path directory=argv[2];check(directory.is_absolute(),"absolute lab cache required");
 const auto start=cache::Clock::now();const auto seed=cache::exact::primes_to(2000);
 mpz_t p;mpz_init_set_ui(p,1);for(unsigned j=0;j<64;j++)mpz_mul_ui(p,p,seed[j]);
 cache::Metrics m;const auto basis=cache::acquire_fresh(directory,1000000000u,64u,p,seed,m);mpz_clear(p);
 check(basis.first==64&&basis.inverses.size()+64==basis.all_primes.size(),"complete P64 suffix");
 check(m.published||m.status==std::string("hit"),"cache publication or exact verified hit required");
 std::cout<<"{\"schema\":\"r312-p64-cpu-basis-v1\",\"status\":\"PASS\",\"first\":64,\"all_primes\":"<<basis.all_primes.size()
  <<",\"inverses\":"<<basis.inverses.size()<<",\"build_ms\":"<<m.build_ms<<",\"total_ms\":"<<cache::elapsed_ms(start)
  <<",\"bytes\":"<<m.file_bytes<<",\"cache_path\":"<<std::quoted(m.cache_path)<<",\"scope\":\"CPU exact basis only; no GPU or yield claim\"}"<<std::endl;
 return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}

#define RIECOIN_RLOW_PRIME_ROOT_CACHE_CANONICAL_FIRST 114
#define RIECOIN_RLOW_PRIME_ROOT_CACHE_CANONICAL_MIN_FIRST 109
#define RIECOIN_RLOW_CANONICAL_DEVICE_TRANSFORM
#include "r312-extra/riecoin_prime_root_cache.hpp"
int main(){try{
  namespace c=riecoin_prime_root_cache;
  auto seed=c::exact::primes_to(2000);unsigned checks=0;
  for(unsigned first:{1u,8u,105u,108u,115u}){
    c::Integer p;c::set_prefix_product(p.value,seed,first);c::Metrics m;
    auto b=c::acquire_fresh({},2000,first,p.value,seed,m);
    if(!b.exact_direct_geometry||b.first!=first||b.requested_first!=first||b.prefix_inverse_count||b.inverse_multiplier!=1)throw std::runtime_error("direct shape");
    for(std::size_t i=first;i<b.all_primes.size();++i){auto q=b.all_primes[i];if(std::uint64_t(mpz_fdiv_ui(p.value,q))*b.inverses[i-first]%q!=1)throw std::runtime_error("inverse mismatch");++checks;}
    mpz_add_ui(p.value,p.value,1);bool refused=false;try{c::acquire_fresh({},2000,first,p.value,seed,m);}catch(const std::invalid_argument&){refused=true;}
    if(!refused)throw std::runtime_error("bad prefix accepted");++checks;
  }
  std::cout<<"direct_basis_checks="<<checks<<" PASS\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

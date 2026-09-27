#define RIECOIN_RLOW_PRIME_ROOT_CACHE_CANONICAL_FIRST 114
#define RIECOIN_RLOW_PRIME_ROOT_CACHE_CANONICAL_MIN_FIRST 107
#define RIECOIN_RLOW_CANONICAL_DEVICE_TRANSFORM
#include "r312-extra/riecoin_prime_root_cache.hpp"
int main(){try{
 namespace c=riecoin_prime_root_cache;
 auto primes=c::exact::primes_to(50000); c::Integer canonical;
 c::set_prefix_product(canonical.value,primes,114);
 const auto original=c::build(50000,114,canonical.value);unsigned checks=0;
 for(unsigned first=107;first<=114;++first){
  c::Integer requested,validated;c::set_prefix_product(requested.value,primes,first);
  c::prepare_canonical_primorial(validated.value,first,requested.value,primes);
  if(mpz_cmp(validated.value,canonical.value))throw std::runtime_error("canonical mismatch");
  auto basis=original;c::Metrics m;c::stage_device_transform_from_canonical(basis,first,requested.value,m);
  if((first==107)!=(basis.inverse_multiplier_second>1))throw std::runtime_error("split boundary");
  for(std::size_t i=0;i<primes.size()-first;++i){
   const auto p=primes[first+i];const auto inverse=c::effective_inverse_for_test(basis,i);
   if(std::uint64_t(mpz_fdiv_ui(requested.value,p))*inverse%p!=1)throw std::runtime_error("complete inverse mismatch");
   ++checks;
  }
  mpz_add_ui(requested.value,requested.value,1);bool rejected=false;
  try{c::prepare_canonical_primorial(validated.value,first,requested.value,primes);}catch(const std::exception&){rejected=true;}
  if(!rejected)throw std::runtime_error("mutated primorial accepted");++checks;
 }
 for(unsigned first:{106u,115u}){c::Integer p;c::set_prefix_product(p.value,primes,first);c::Metrics m;
  auto b=c::acquire_fresh({},50000,first,p.value,primes,m);
  if(!b.exact_direct_geometry||b.inverse_multiplier_second!=1)throw std::runtime_error("fallback lost");++checks;}
 std::cout<<"two_factor_bridge_checks="<<checks<<" PASS\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

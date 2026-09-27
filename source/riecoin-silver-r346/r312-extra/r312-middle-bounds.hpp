#pragma once
#include <gmp.h>
#include <cstdint>
namespace r312 {
inline bool requested=true, admitted=false;
// Prove the entire half-open factor reservation, including every GPU member.
// Target must end in at least864 zero bits; all displacement stays below2^512.
inline bool bounds(mpz_srcptr target,mpz_srcptr base,mpz_srcptr primorial,
                   std::uint64_t factor_max,std::uint32_t largest_offset){
  mpz_t delta,k;mpz_inits(delta,k,nullptr);
  mpz_import(k,1,-1,sizeof(factor_max),0,0,&factor_max);
  mpz_sub(delta,base,target);
  bool ok=mpz_sgn(target)>0&&mpz_sgn(delta)>=0&&mpz_sgn(primorial)>0&&
      mpz_even_p(primorial)&&mpz_odd_p(base)&&mpz_divisible_2exp_p(target,864)&&
      mpz_sizeinbase(target,2)>=1144&&mpz_sizeinbase(target,2)<=1184;
  mpz_addmul(delta,primorial,k);mpz_add_ui(delta,delta,largest_offset);
  ok=ok&&mpz_sizeinbase(delta,2)<=512;
  mpz_clears(delta,k,nullptr);return ok;
}
}

#pragma once
#include <cstdint>
#include <gmp.h>
namespace r292 {
inline bool enabled=true;
struct Bounds { bool admitted; unsigned minimum_bits, maximum_bits; };
// base is already anchored at factor_origin. All member offsets are unsigned.
// Include factor_max itself as a conservative upper bound on the half-open span.
inline Bounds bounds(const std::uint32_t* base,const std::uint32_t* primorial,
    std::uint64_t span,const std::uint32_t* offsets,bool requested) {
  mpz_t lo,p,hi,k;mpz_inits(lo,p,hi,k,nullptr);
  mpz_import(lo,40,-1,4,0,0,base);mpz_import(p,40,-1,4,0,0,primorial);
  mpz_import(k,1,-1,sizeof(span),0,0,&span);
  std::uint32_t maximum=0;bool even=true;
  for(unsigned i=0;i<6;++i){if(offsets[i]>maximum)maximum=offsets[i];even=even&&!(offsets[i]&1u);}
  mpz_mul(hi,p,k);mpz_add(hi,hi,lo);mpz_add_ui(hi,hi,maximum);
  Bounds result{false,unsigned(mpz_sizeinbase(lo,2)),unsigned(mpz_sizeinbase(hi,2))};
  result.admitted=requested&&span>0&&even&&mpz_cmp_ui(p,0)>0&&mpz_even_p(p)&&
    mpz_odd_p(lo)&&result.minimum_bits>=1144&&result.maximum_bits<=1184;
  mpz_clears(lo,p,hi,k,nullptr);return result;
}
}

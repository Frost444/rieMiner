#include "r322-source/r292_member_bounds.hpp"
#include <array>
#include <cassert>
#include <iostream>
int main(){
  unsigned checks=0;std::array<uint32_t,40> a{},p{};std::array<uint32_t,6> offsets{{2,6,8,12,18,20}};
  mpz_t lo;mpz_init(lo);p[0]=210;
  auto pack=[&](){a.fill(0);size_t n=0;mpz_export(a.data(),&n,-1,4,0,0,lo);assert(n<=40);};
  auto check=[&](bool requested,uint64_t span,bool expected){pack();const auto b=r292::bounds(a.data(),p.data(),span,offsets.data(),requested);++checks;if(b.admitted!=expected){std::cerr<<"case="<<checks<<" min="<<b.minimum_bits<<" max="<<b.maximum_bits<<"\n";std::abort();}};
  for(bool mode:{false,true}){r322::extended=mode;
  for(unsigned bits=3;bits<=1216;++bits){mpz_set_ui(lo,0);mpz_setbit(lo,bits-1);mpz_add_ui(lo,lo,1);check(true,17,bits>=r322::minimum_bits()&&bits<=1184);}
  mpz_set_ui(lo,0);mpz_setbit(lo,1184);mpz_sub_ui(lo,lo,231);check(true,1,true);
  mpz_add_ui(lo,lo,2);check(true,1,false); // upper crosses radix admission, never truncates.
  mpz_set_ui(lo,0);mpz_setbit(lo,1143);mpz_add_ui(lo,lo,1);
  check(false,1,false);check(true,0,false);check(true,UINT64_MAX,true);
  offsets[5]=21;check(true,1,false);offsets[5]=20;
  p[0]=211;check(true,1,false);p[0]=0;check(true,1,false);p[0]=210;
  mpz_add_ui(lo,lo,1);check(true,1,false);
  mpz_set_ui(lo,0);mpz_setbit(lo,r322::minimum_bits()-1);mpz_sub_ui(lo,lo,1);check(true,1,false);
  }
  mpz_clear(lo);std::cout<<"phase=r322_bounds checks="<<checks<<" errors=0 terminal=PASS\n";
}

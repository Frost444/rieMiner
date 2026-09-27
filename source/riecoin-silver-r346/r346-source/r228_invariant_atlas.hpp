#pragma once
// R228 CPU contract prototype. Not connected to a miner or a GPU.
// A sealed process-local snapshot owns its data; jobs retain it across registry
// replacement. No shared-file, cross-process trust, or zero-cost claim is made.
#include "r217_native31.hpp"
#include <gmp.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace r228 {
constexpr uint64_t Low=4000000000ull, High=8000000000ull;
inline void set64(mpz_ptr n,uint64_t v){mpz_import(n,1,-1,8,0,0,&v);}
inline uint64_t get64(mpz_srcptr n){
 if(mpz_sgn(n)<0||mpz_sizeinbase(n,2)>64)throw std::runtime_error("u64 bound");
 uint64_t v=0;size_t count=0;mpz_export(&v,&count,-1,8,0,0,n);return v;
}
inline std::string hex(mpz_srcptr n){std::vector<char> s(mpz_sizeinbase(n,16)+3);mpz_get_str(s.data(),16,n);return s.data();}
inline std::vector<uint32_t> digits31(mpz_srcptr n){
 std::vector<uint32_t> d((mpz_sizeinbase(n,2)+30)/31);size_t count=0;
 mpz_export(d.data(),&count,-1,4,0,1,n);return d;
}
struct Mpz {mpz_t n;Mpz(){mpz_init(n);}~Mpz(){mpz_clear(n);}Mpz(const Mpz&)=delete;};
struct Encoded {
 std::vector<uint32_t> delta, inverse_low, inverse_high;
 uint64_t inverse(size_t i)const{
  return inverse_low.at(i)|(uint64_t((inverse_high.at(i/32)>>(i%32))&1u)<<32);
 }
 size_t bytes()const{return 4*(delta.size()+inverse_low.size()+inverse_high.size());}
};
inline Encoded encode(const std::vector<uint64_t>& primes,const std::vector<uint64_t>& inverses){
 if(primes.size()!=inverses.size()||primes.empty())throw std::runtime_error("encode count");
 Encoded e;e.inverse_high.assign((primes.size()+31)/32,0);
 for(size_t i=0;i<primes.size();++i){
  if(primes[i]<=Low||primes[i]>High||inverses[i]>=primes[i]||!inverses[i])throw std::runtime_error("encode domain");
  e.delta.push_back(uint32_t(primes[i]-Low));e.inverse_low.push_back(uint32_t(inverses[i]));
  e.inverse_high[i/32]|=uint32_t(inverses[i]>>32)<<(i%32);
 }
 return e;
}
class Atlas {
 struct Storage {
  Encoded packed;std::string P;uint64_t offset=0,generation=0;
 };
 std::shared_ptr<const Storage> data_;
 explicit Atlas(std::shared_ptr<const Storage> data):data_(std::move(data)){}
public:
 // expected_primes belongs to the qualified source contract, not the cache.
 // Complete relation validation occurs before the snapshot becomes reachable.
 static Atlas seal(mpz_srcptr P,uint64_t offset,uint64_t generation,
                   const std::vector<uint64_t>& expected_primes,Encoded packed){
  if(mpz_sgn(P)<=0||!generation||expected_primes.empty())throw std::runtime_error("seal identity");
  const size_t n=expected_primes.size();
  if(packed.delta.size()!=n||packed.inverse_low.size()!=n||packed.inverse_high.size()!=(n+31)/32)
   throw std::runtime_error("seal completeness");
  if(n%32&&(packed.inverse_high.back()>>(n%32)))throw std::runtime_error("seal padding");
  Mpz p,iv,t,o;set64(o.n,offset);if(mpz_cmp(o.n,P)>=0)throw std::runtime_error("offset outside primorial");
  for(size_t i=0;i<n;++i){
   const uint64_t prime=Low+packed.delta[i],inverse=packed.inverse(i);
   if(prime!=expected_primes[i]||prime<=Low||prime>High||!(prime&1)||
      (i&&prime<=expected_primes[i-1])||!inverse||inverse>=prime)throw std::runtime_error("seal entry");
   set64(p.n,prime);set64(iv.n,inverse);mpz_mul(t.n,P,iv.n);mpz_mod(t.n,t.n,p.n);
   if(mpz_cmp_ui(t.n,1))throw std::runtime_error("seal inverse certificate");
  }
  auto storage=std::make_shared<Storage>();storage->packed=std::move(packed);
  storage->P=hex(P);storage->offset=offset;storage->generation=generation;
  return Atlas(std::move(storage));
 }
 class Job {
  friend class Atlas;
  std::shared_ptr<const Storage> data_;std::vector<uint32_t> quotient_,basis_ratio_{1};uint64_t origin_=0,span_=0;
  Job(std::shared_ptr<const Storage> data,std::vector<uint32_t> q,uint64_t origin,uint64_t span)
   :data_(std::move(data)),quotient_(std::move(q)),origin_(origin),span_(span){}
 public:
  size_t size()const{return data_->packed.delta.size();}
  uint64_t prime(size_t i)const{return Low+data_->packed.delta.at(i);}
  uint64_t inverse(size_t i)const{
   const uint64_t original=data_->packed.inverse(i);
   if(basis_ratio_.size()==1&&basis_ratio_[0]==1)return original;
   const uint64_t p=prime(i);
   return r217_native::product(original,r217_native::words(basis_ratio_.data(),uint32_t(basis_ratio_.size()),p),p);
  }
  uint64_t phase(size_t i,uint32_t member_offset)const{
   const uint64_t p=prime(i),iv=inverse(i);
   const uint64_t a=r217_native::words(quotient_.data(),uint32_t(quotient_.size()),p);
   const uint64_t o=r217_native::add(data_->offset%p,uint64_t(member_offset)%p,p);
   uint64_t sum=r217_native::add(a,r217_native::product(o,iv,p),p);
   sum=r217_native::add(sum,origin_%p,p);return sum?p-sum:0;
  }
  bool hits(size_t i,uint32_t offset)const{return phase(i,offset)<span_;}
  uint64_t generation()const{return data_->generation;}
 };
 Job bind(mpz_srcptr P,mpz_srcptr B,uint64_t expected_generation,uint64_t origin,uint64_t span)const{
  if(expected_generation!=data_->generation||hex(P)!=data_->P||!span||span>Low||origin>UINT64_MAX-span)
   throw std::runtime_error("job identity/domain");
  Mpz a,o,bound;set64(bound.n,High);
  if(mpz_cmp(B,bound.n)<=0)throw std::runtime_error("not a proper-divisor domain");
  mpz_fdiv_qr(a.n,o.n,B,P);
  if(get64(o.n)!=data_->offset)throw std::runtime_error("job affine relation");
  return Job(data_,digits31(a.n),origin,span);
 }
 // R229: P_anchor = P_job * S. Therefore inv(P_job) = S*inv(P_anchor)
 // modulo every qualified divisor. One immutable anchor serves many exact
 // sub-primorials. The exact integer divisibility check is done once per job.
 Job bind_divisor(mpz_srcptr P,mpz_srcptr B,uint64_t expected_generation,uint64_t origin,uint64_t span)const{
  if(expected_generation!=data_->generation||mpz_sgn(P)<=0||!span||span>Low||origin>UINT64_MAX-span)
   throw std::runtime_error("family job identity/domain");
  Mpz anchor,ratio,remainder,a,o,bound;
  if(mpz_set_str(anchor.n,data_->P.c_str(),16))throw std::runtime_error("sealed anchor encoding");
  mpz_fdiv_qr(ratio.n,remainder.n,anchor.n,P);
  if(mpz_sgn(remainder.n)||mpz_sgn(ratio.n)<=0)throw std::runtime_error("not an exact sub-basis");
  set64(o.n,data_->offset);if(mpz_cmp(o.n,P)>=0)throw std::runtime_error("family offset outside primorial");
  set64(bound.n,High);if(mpz_cmp(B,bound.n)<=0)throw std::runtime_error("family proper-divisor domain");
  mpz_fdiv_qr(a.n,o.n,B,P);if(get64(o.n)!=data_->offset)throw std::runtime_error("family affine relation");
  Job job(data_,digits31(a.n),origin,span);job.basis_ratio_=digits31(ratio.n);return job;
 }
 size_t bytes()const{return data_->packed.bytes();}
};
} // namespace r228

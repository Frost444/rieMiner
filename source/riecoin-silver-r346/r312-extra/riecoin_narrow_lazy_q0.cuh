#pragma once

// R15: retain the 1280-bit transport and the exact candidate set, but evaluate
// bounded jobs with 36 radix-2^32 digits, four cooperative threads, R=2^1152.
// Host admission proves the ENTIRE factor reservation fits the configured
// bound. The default remains the original 1136-bit Silver boundary; a bounded
// physical-route hotfix may raise it, while all other jobs keep the exact
// 1280/TPI8 path. No candidate-level filtering is added.
namespace riecoin_narrow_lazy_q0 {
constexpr uint32_t kBits = 1152u;
#ifndef RIECOIN_RLOW_NARROW_MAX_INPUT_BITS
#define RIECOIN_RLOW_NARROW_MAX_INPUT_BITS 1136u
#endif
constexpr uint32_t kMaximumInputBits = RIECOIN_RLOW_NARROW_MAX_INPUT_BITS;
static_assert(kMaximumInputBits >= 3u && kMaximumInputBits <= kBits - 4u,
              "narrow lazy Q0 requires a nonzero certified window");

__device__ __forceinline__ bool evaluate(const cgbn_mem_t<1280u>* input,
    uint32_t slot, uint32_t* normal = nullptr) {
  using context = cgbn_context_t<4u, riecoin_cuda::PrpCgbnParameters>;
  using environment = cgbn_env_t<context, kBits>;
  context ctx(cgbn_no_checks, nullptr, slot);
  environment env(ctx);
  typename environment::cgbn_t n, x;
  const uint32_t lane = threadIdx.x & 3u;
  // A violated job-bound invariant aborts the kernel, never rejects a candidate.
  if (input[slot]._limbs[36u + lane] != 0u) asm volatile("trap;");
  cgbn_load(env, n, reinterpret_cast<cgbn_mem_t<kBits>*>(
      const_cast<cgbn_mem_t<1280u>*>(input + slot)));
  const uint32_t leading = cgbn_clz(env, n);
  const uint32_t bits = kBits - leading;
  if (leading < kBits-kMaximumInputBits || bits < 3u ||
      (cgbn_get_ui32(env,n)&1u)==0u) asm volatile("trap;");
  const uint32_t width = riecoin_lazy_base2_window::window_bits(leading);
  // We need Montgomery ONE, not general big-integer division. The upstream
  // 9-limbs/thread division dispatcher has no specialization on this mapping.
  // Start below n and double modulo n to obtain 2^1152 mod n. At every step
  // x<n, hence 2*x<2*n<R by the admitted headroom: no overflow and at most
  // one subtraction. Keep n/factor widths and the Euler predicate unchanged.
  cgbn_set_ui32(env,x,1u);
  cgbn_shift_left(env,x,x,bits-1u);
  for(uint32_t exponent=bits-1u;exponent<kBits;++exponent) {
    cgbn_add(env,x,x,x);
    if(cgbn_compare(env,x,n)>=0)cgbn_sub(env,x,x,n);
  }
  const uint32_t n0=cgbn_get_ui32(env,n);
  uint32_t inverse=1u;
  // Newton in Z/(2^32): odd n0 starts correct modulo2; five doublings of
  // precision give n0*inverse==1 modulo2^32. Montgomery requires its negative.
  for(uint32_t i=0;i<5u;++i)inverse*=2u-n0*inverse;
  inverse=0u-inverse;
  uint32_t remaining = bits-1u;
  bool first = true;
  while (remaining != 0u) {
    const uint32_t take = remaining < width ? remaining : width;
    if (!first) for (uint32_t i=0;i<take;++i) cgbn_mont_sqr(env,x,x,n,inverse);
    const uint32_t value = cgbn_extract_bits_ui32(env,n,remaining-take+1u,take);
    cgbn_shift_left(env,x,x,value);
    remaining -= take;
    first = false;
  }
  cgbn_mont2bn(env,x,x,n,inverse);
  if (normal != nullptr) {
    cgbn_store(env,reinterpret_cast<cgbn_mem_t<kBits>*>(normal+slot*40u),x);
    normal[slot*40u+36u+lane]=0u;
  }
  return riecoin_cuda::passes_euler_jacobi(env,x,n);
}

__global__ void oracle(const cgbn_mem_t<1280u>* input, uint8_t* verdict,
    uint32_t* residue, uint32_t count) {
  const uint32_t slot=(blockIdx.x*blockDim.x+threadIdx.x)/4u;
  if(slot>=count)return;
  const bool pass=evaluate(input,slot,residue);
  if((threadIdx.x&3u)==0u)verdict[slot]=pass?1u:0u;
}

__global__ void segmented(const cgbn_mem_t<1280u>* input,uint8_t* verdict,
    const uint32_t* counts,uint32_t macros,uint32_t capacity) {
  const uint32_t slot=(blockIdx.x*blockDim.x+threadIdx.x)/4u;
  const uint32_t macro=slot/capacity;
  if(macro>=macros)return;
  const bool pass=slot%capacity<counts[macro] && evaluate(input,slot);
  if((threadIdx.x&3u)==0u)verdict[slot]=pass?1u:0u;
}

inline bool covers_reservation(mpz_srcptr base,mpz_srcptr primorial,uint64_t end) {
  if(end==0u || mpz_sgn(base)<=0 || mpz_sgn(primorial)<=0)return false;
  riecoin_lazy_base2_guard::Integer maximum,factor;
  mpz_set_u64(factor.value,end-1u);
  mpz_mul(maximum.value,factor.value,primorial);
  mpz_add(maximum.value,maximum.value,base);
  return mpz_sizeinbase(maximum.value,2)<=kMaximumInputBits;
}

inline void guard() {
  namespace g=riecoin_lazy_base2_guard;
  const auto start=std::chrono::steady_clock::now();
  std::vector<g::Fixture> cases;
  for(auto& c:g::corpus()) {
    g::Integer n;
    mpz_import(n.value,40u,-1,sizeof(uint32_t),0,0,c.input._limbs);
    const auto bits=mpz_sizeinbase(n.value,2);
    if(bits>=3u && bits<=kMaximumInputBits && mpz_odd_p(n.value)) cases.push_back(std::move(c));
  }
  if(cases.empty())throw std::runtime_error("R15 empty exact corpus");
  std::vector<cgbn_mem_t<1280u>> input(cases.size());
  std::vector<uint8_t> verdict(cases.size());
  std::vector<uint32_t> residue(cases.size()*40u);
  for(size_t i=0;i<cases.size();++i)input[i]=cases[i].input;
  g::Device<cgbn_mem_t<1280u>> di(cases.size());
  g::Device<uint8_t> dv(cases.size());
  g::Device<uint32_t> dr(residue.size());
  cuda_check(cudaMemcpy(di.data,input.data(),input.size()*sizeof(input[0]),cudaMemcpyHostToDevice),"R15 guard input");
  cuda_check(cudaMemset(dr.data,0xa5,residue.size()*sizeof(uint32_t)),"R15 guard poison");
  oracle<<<(cases.size()*4u+127u)/128u,128u>>>(di.data,dv.data,dr.data,static_cast<uint32_t>(cases.size()));
  cuda_check(cudaGetLastError(),"R15 guard launch");
  cuda_check(cudaDeviceSynchronize(),"R15 guard sync");
  cuda_check(cudaMemcpy(verdict.data(),dv.data,verdict.size(),cudaMemcpyDeviceToHost),"R15 guard verdict");
  cuda_check(cudaMemcpy(residue.data(),dr.data,residue.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost),"R15 guard residue");
  for(size_t i=0;i<cases.size();++i)
    if(verdict[i]!=cases[i].verdict || std::memcmp(residue.data()+i*40u,cases[i].residue.data(),40u*sizeof(uint32_t))!=0)
      throw std::runtime_error("R15 GMP verdict/residue mismatch: "+cases[i].label);
  di.release();dv.release();dr.release();
  std::cout<<"phase=narrow_lazy_guard checked="<<cases.size()<<" mismatches=0 terminal=PASS ms="
    <<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<std::endl;
}
}

// Experimental, isolated TPI1/1216 mapping. Not an officially supported CGBN TPI.
// gECC general-modulus SoS Montgomery core; 38-word radix, 40-word storage.
// Isolated falsifier, no production admission or performance claim.
#include "r289-general-sos.cuh"
#include "r289-square-sos.cuh"
#include <cuda_runtime.h>
#include <gmp.h>
#include <cgbn/cgbn.h>

#include "riecoin_r203_r1216_q0_launch.hpp"

#include <cstdint>

namespace {

template<class Bn>
__device__ __forceinline__ void sos_multiply(Bn& out, const Bn& a,
    const Bn& b, const Bn& n, uint32_t inverse) {
#if defined(__CUDA_ARCH__)
  static_assert(sizeof(out._limbs) == 38u * sizeof(uint32_t), "A99B TPI1 radix");
  riecoin_r289_general::GeneralSos<38>::multiply(out._limbs, a._limbs, b._limbs, n._limbs, inverse);
#endif
}

constexpr std::uint32_t kStorageBits = 1280u;
template<class Bn>
__device__ __forceinline__ void sos_square(Bn& out,const Bn& a,const Bn& n,uint32_t inverse){
#if defined(__CUDA_ARCH__)
 static_assert(sizeof(out._limbs)==38u*sizeof(uint32_t),"R283 radix");
 riecoin_r289_square::SymmetricSos<38>::multiply(out._limbs,a._limbs,a._limbs,n._limbs,inverse);
#endif
}
constexpr std::uint32_t kRadixBits = 1216u;
constexpr std::uint32_t kMaximumInputBits = 1184u;
constexpr std::uint32_t kBlockThreads = 128u;

struct Parameters {
  static constexpr std::uint32_t TPB = kBlockThreads;
  static constexpr std::uint32_t MAX_ROTATION = 4u;
  static constexpr std::uint32_t SHM_LIMIT = 0u;
  static constexpr bool CONSTANT_TIME = false;
};

template <class Env, class Bn>
__device__ __forceinline__ void double_modulo(
    Env& env, Bn& value, const Bn& modulus) {
  if (cgbn_compare(env, value, modulus) >= 0)
    cgbn_sub(env, value, value, modulus);
  const std::int32_t carry = cgbn_add(env, value, value, value);
  if (carry != 0 || cgbn_compare(env, value, modulus) >= 0)
    cgbn_sub(env, value, value, modulus);
}

template <class Env, class Bn>
__device__ __forceinline__ bool passes_euler_jacobi(
    Env& env, const Bn& residue, const Bn& modulus) {
  const std::uint32_t modulo_eight = cgbn_get_ui32(env, modulus) & 7u;
  if (modulo_eight == 1u || modulo_eight == 7u)
    return cgbn_equals_ui32(env, residue, 1u);
  Bn minus_one;
  cgbn_sub_ui32(env, minus_one, modulus, 1u);
  return cgbn_equals(env, residue, minus_one);
}

__device__ __forceinline__ std::uint32_t window_bits(
    std::uint32_t headroom_bits) {
  std::uint32_t selected = 0u;
  for (std::uint32_t width = 1u; width <= 6u; ++width)
    if (headroom_bits >= (1u << (width + 1u))) selected = width;
  return selected;
}

__device__ __forceinline__ bool evaluate(
    const cgbn_mem_t<kStorageBits>* candidates, std::uint32_t instance,
    std::uint32_t* normal_residues = nullptr) {
  using context = cgbn_context_t<1u, Parameters>;
  using environment = cgbn_env_t<context, kRadixBits>;
  using bn = typename environment::cgbn_t;
  context ctx(cgbn_no_checks, nullptr, instance);
  environment env(ctx);
  bn modulus, residue, normal_one;
  if (candidates[instance]._limbs[38] != 0u || candidates[instance]._limbs[39] != 0u)
    asm volatile("trap;");
  cgbn_load(env, modulus,
            reinterpret_cast<cgbn_mem_t<kRadixBits>*>(const_cast<cgbn_mem_t<kStorageBits>*>(candidates + instance)));
  const std::uint32_t bit_length = kRadixBits - cgbn_clz(env, modulus);

  // Admission is a host-proved invariant over the complete reservation.  A
  // violation is an execution fault, never a synthetic composite verdict.
  if ((cgbn_get_ui32(env, modulus) & 1u) == 0u || bit_length < 3u ||
      bit_length > kMaximumInputBits)
    asm volatile("trap;");

  cgbn_set_ui32(env, residue, 1u);
  cgbn_shift_left(env, residue, residue, bit_length - 1u);
  for (std::uint32_t exponent = bit_length - 1u; exponent < kRadixBits;
       ++exponent)
    double_modulo(env, residue, modulus);

  const std::uint32_t inverse = -cgbn_binary_inverse_ui32(
      env, cgbn_get_ui32(env, modulus));
  const std::uint32_t width = window_bits(kRadixBits - bit_length);
  if (width == 0u) asm volatile("trap;");

  std::uint32_t remaining = bit_length - 1u;
  bool first = true;
  while (remaining != 0u) {
    const std::uint32_t take = remaining < width ? remaining : width;
    if (!first)
      for (std::uint32_t index = 0u; index < take; ++index)
        sos_square(residue, residue, modulus, inverse);
    const std::uint32_t value = cgbn_extract_bits_ui32(
        env, modulus, remaining - take + 1u, take);
    cgbn_shift_left(env, residue, residue, value);
    remaining -= take;
    first = false;
  }

  // Convert with the same 38-word REDC core.  The stock mont2bn abstraction
  // belongs to the 40-word domain and must never cross this ABI boundary.
  cgbn_set_ui32(env, normal_one, 1u);
  sos_multiply(residue, residue, normal_one, modulus, inverse);
  if (normal_residues != nullptr) {
    cgbn_store(env, reinterpret_cast<cgbn_mem_t<kRadixBits>*>(normal_residues + instance * 40u), residue);
    if (true) {
      normal_residues[instance * 40u + 38u] = 0u;
      normal_residues[instance * 40u + 39u] = 0u;
    }
  }
  return passes_euler_jacobi(env, residue, modulus);
}

__global__ void segmented_kernel(
    const cgbn_mem_t<kStorageBits>* candidates, std::uint8_t* verdicts,
    const std::uint32_t* macro_counts, std::uint32_t macro_count,
    std::uint32_t macro_capacity) {
  const std::uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t instance = thread;
  const std::uint32_t macro = instance / macro_capacity;
  if (macro >= macro_count) return;
  const std::uint32_t local = instance - macro * macro_capacity;
  const bool probable =
      local < macro_counts[macro] && evaluate(candidates, instance);
  if (true) verdicts[instance] = probable ? 1u : 0u;
}

__global__ void oracle_kernel(
    const cgbn_mem_t<kStorageBits>* candidates, std::uint8_t* verdicts,
    std::uint32_t* normal_residues, std::uint32_t count) {
  const std::uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t instance = thread;
  if (instance >= count) return;
  const bool probable = evaluate(candidates, instance, normal_residues);
  if (true) verdicts[instance] = probable ? 1u : 0u;
}

}  // namespace

namespace {
__global__ void r283_dense_kernel(const cgbn_mem_t<kStorageBits>* candidates,
 std::uint8_t* verdicts,const std::uint32_t* slots,const std::uint32_t* total,
 std::uint32_t capacity){
 const auto index=blockIdx.x*blockDim.x+threadIdx.x;
 if(*total>capacity)asm volatile("trap;");
 if(index>=*total)return;
 const auto slot=slots[index];
 if(slot>=capacity)asm volatile("trap;");
 verdicts[slot]=evaluate(candidates,slot)?1u:0u;
}
}
extern "C" cudaError_t riecoin_r289_launch_dense(const void* candidates,std::uint8_t* verdicts,
 const std::uint32_t* slots,const std::uint32_t* total,std::uint32_t capacity,cudaStream_t stream){
 r283_dense_kernel<<<(capacity+127u)/128u,128u,0,stream>>>(
  static_cast<const cgbn_mem_t<kStorageBits>*>(candidates),verdicts,slots,total,capacity);
 return cudaGetLastError();
}

extern "C" cudaError_t riecoin_a99b_launch_segmented(
    const void* candidates, std::uint8_t* verdicts,
    const std::uint32_t* macro_counts, std::uint32_t macro_count,
    std::uint32_t macro_capacity, cudaStream_t stream) {
  const std::uint64_t instances =
      static_cast<std::uint64_t>(macro_count) * macro_capacity;
  const std::uint32_t blocks = static_cast<std::uint32_t>(
      (instances + kBlockThreads - 1u) / kBlockThreads);
  segmented_kernel<<<blocks, kBlockThreads, 0u, stream>>>(
      static_cast<const cgbn_mem_t<kStorageBits>*>(candidates), verdicts,
      macro_counts, macro_count, macro_capacity);
  return cudaGetLastError();
}

extern "C" cudaError_t riecoin_a99b_launch_oracle(
    const void* candidates, std::uint8_t* verdicts,
    std::uint32_t* normal_residues, std::uint32_t count,
    cudaStream_t stream) {
  const std::uint32_t blocks =
      (count + kBlockThreads - 1u) / kBlockThreads;
  oracle_kernel<<<blocks, kBlockThreads, 0u, stream>>>(
      static_cast<const cgbn_mem_t<kStorageBits>*>(candidates), verdicts,
      normal_residues, count);
  return cudaGetLastError();
}


namespace {
__global__ void primitive_kernel(const uint32_t* aa,const uint32_t* bb,const uint32_t* nn,
 uint32_t* out,int32_t* meta,uint32_t count,uint32_t op,uint32_t shift) {
 using context=cgbn_context_t<1u,Parameters>;
 using environment=cgbn_env_t<context,1216u>;
 using bn=typename environment::cgbn_t;
 uint32_t id=(blockIdx.x*blockDim.x+threadIdx.x);
 if(id>=count)return;
 context ctx(cgbn_no_checks,nullptr,id);environment env(ctx);
 bn a,b,n,r;
 cgbn_load(env,a,reinterpret_cast<cgbn_mem_t<1216>*>(const_cast<uint32_t*>(aa+40u*id)));
 cgbn_load(env,b,reinterpret_cast<cgbn_mem_t<1216>*>(const_cast<uint32_t*>(bb+40u*id)));
 cgbn_load(env,n,reinterpret_cast<cgbn_mem_t<1216>*>(const_cast<uint32_t*>(nn+40u*id)));
 int32_t v=0;
 if(op==0)cgbn_set(env,r,a);
 else if(op==1)cgbn_shift_left(env,r,a,shift);
 else if(op==2)cgbn_shift_right(env,r,a,shift);
 else if(op==3)v=cgbn_add(env,r,a,b);
 else if(op==4)v=cgbn_sub(env,r,a,b);
 else {
  uint32_t inv=-cgbn_binary_inverse_ui32(env,cgbn_get_ui32(env,n));
  if(op==5)sos_multiply(r,a,b,n,inv);else sos_square(r,a,n,inv);
  if(cgbn_compare(env,r,n)>=0)cgbn_sub(env,r,r,n);
 }
 int32_t clz=cgbn_clz(env,a),cmp=cgbn_compare(env,a,b);
 uint32_t extract=cgbn_extract_bits_ui32(env,a,shift,1);
 cgbn_store(env,reinterpret_cast<cgbn_mem_t<1216>*>(out+40u*id),r);
 if(true){
  out[40u*id+38]=0;out[40u*id+39]=0;
  meta[id*4]=v;meta[id*4+1]=clz;meta[id*4+2]=cmp;meta[id*4+3]=extract;
 }
}
}
extern "C" cudaError_t riecoin_a99b_primitives(const uint32_t* a,const uint32_t* b,const uint32_t* n,
 uint32_t* out,int32_t* meta,uint32_t count,uint32_t op,uint32_t shift){
 primitive_kernel<<<(count+127)/128,128>>>(a,b,n,out,meta,count,op,shift);
 return cudaGetLastError();
}

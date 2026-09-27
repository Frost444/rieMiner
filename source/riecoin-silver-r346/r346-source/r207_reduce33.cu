// Exact arithmetic for a wider admission domain without widening PRP numbers.
// p<=8B, x<2^66. No floating point and no per-word division in reduction.
#include <cuda_runtime.h>
#include <gmp.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#ifndef R207_LIBRARY
static void ck(cudaError_t e,const char* s){if(e!=cudaSuccess)throw std::runtime_error(std::string(s)+": "+cudaGetErrorString(e));}
#endif
struct Input {uint64_t p,a,b;uint32_t words[40];};
struct Output {uint64_t product,remainder,inverse,packed;};
__device__ uint64_t reduce66(uint64_t lo,uint64_t hi,uint64_t p,uint64_t mu){
 // mu=floor((2^64-1)/p); its deficit from2^64/p is at most1.
 // For x<4*2^64, the approximate quotient is never too large and is at
 // most4 too small. The true remainder before correction is <5p<2^36,
 // so the wrapping low-word subtraction is exact, including hi!=0.
 uint64_t q=hi*mu+__umul64hi(lo,mu),r=lo-q*p;
 #pragma unroll
 for(unsigned k=0;k<4;++k)if(r>=p)r-=p;
 return r;
}
__device__ uint64_t inverse64(uint64_t a,uint64_t p){
 int64_t t=0,nt=1;uint64_t r=p,nr=a;
 while(nr){uint64_t q=r/nr,z=r-q*nr;r=nr;nr=z;int64_t x=t-int64_t(q)*nt;t=nt;nt=x;}
 if(r!=1)return 0;if(t<0)t+=int64_t(p);return uint64_t(t);
}
__global__ void execute(const Input* in,Output* out,uint32_t n){
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 Input v=in[i];uint64_t mu=UINT64_MAX/v.p,r=0;
 out[i].product=reduce66(v.a*v.b,__umul64hi(v.a,v.b),v.p,mu);
 for(unsigned w=40;w;--w)r=reduce66((r<<32)|v.words[w-1],r>>32,v.p,mu);
 out[i].remainder=r;out[i].inverse=inverse64(r,v.p);
 out[i].packed=(v.p<<28)|(uint64_t(v.words[0])&((1u<<28)-1));
}
static uint64_t rng(uint64_t& s){s^=s<<13;s^=s>>7;s^=s<<17;return s;}
static void set64(mpz_ptr z,uint64_t value){mpz_import(z,1,-1,8,0,0,&value);}
static uint64_t get64(mpz_srcptr z){uint64_t value=0;size_t n=0;mpz_export(&value,&n,-1,8,0,0,z);return value;}
#ifndef R207_LIBRARY
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("Expected GPU UUID");
 ck(cudaSetDevice(0),"device");cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0),"identity");
 std::ostringstream id;id<<"GPU-"<<std::hex<<std::setfill('0');for(unsigned i=0;i<16;++i){if(i==4||i==6||i==8||i==10)id<<'-';id<<std::setw(2)<<unsigned(static_cast<unsigned char>(prop.uuid.bytes[i]));}
 if(id.str()!=argv[1])throw std::runtime_error("Wrong GPU");
 constexpr uint32_t n=131072;std::vector<Input> input(n);std::vector<Output> output(n);uint64_t seed=0x207331168ull;
 const uint64_t edge[]={4000000001ull,4294967291ull,4294967295ull,4294967296ull,4294967297ull,4294967311ull,7999999997ull,7999999999ull,8000000000ull};
 for(uint32_t i=0;i<n;++i){auto& v=input[i];v.p=i<256?edge[i%9]:(4000000001ull+rng(seed)%3999999999ull)|1ull;
  v.a=i%4==0?v.p-1:rng(seed)%v.p;v.b=i%4==0?v.p-1:rng(seed)%v.p;
  for(auto& word:v.words)word=uint32_t(rng(seed));if(i%7==0)for(auto& word:v.words)word=UINT32_MAX;
  if(i%11==0)for(auto& word:v.words)word=0;
 }
 Input* d=nullptr;Output* o=nullptr;ck(cudaMalloc(&d,input.size()*sizeof(Input)),"input allocation");ck(cudaMalloc(&o,output.size()*sizeof(Output)),"output allocation");
 ck(cudaMemcpy(d,input.data(),input.size()*sizeof(Input),cudaMemcpyHostToDevice),"input");
 execute<<<(n+255)/256,256>>>(d,o,n);ck(cudaGetLastError(),"kernel");ck(cudaMemcpy(output.data(),o,output.size()*sizeof(Output),cudaMemcpyDeviceToHost),"output");
 mpz_t p,a,b,x,expected,inv;mpz_inits(p,a,b,x,expected,inv,nullptr);uint64_t invertible=0,noninvertible=0;
 for(uint32_t i=0;i<n;++i){const auto& v=input[i];const auto& r=output[i];set64(p,v.p);set64(a,v.a);set64(b,v.b);mpz_mul(x,a,b);mpz_mod(expected,x,p);
  if(r.product!=get64(expected)||r.product>=v.p)throw std::runtime_error("GMP product mismatch");
  mpz_import(x,40,-1,4,0,0,v.words);mpz_mod(expected,x,p);if(r.remainder!=get64(expected))throw std::runtime_error("GMP1280-bit remainder mismatch");
  if(mpz_invert(inv,expected,p)){++invertible;if(r.inverse!=get64(inv))throw std::runtime_error("GMP inverse mismatch");}else{++noninvertible;if(r.inverse)throw std::runtime_error("False inverse");}
  if((r.packed>>28)!=v.p||(r.packed&((1u<<28)-1))!=(v.words[0]&((1u<<28)-1)))throw std::runtime_error("Packed61-bit event mismatch");
 }
 mpz_clears(p,a,b,x,expected,inv,nullptr);cudaFree(d);cudaFree(o);
 std::cout<<"phase=PASS cases="<<n<<" product_checks="<<n<<" full_1280bit_remainders="<<n<<" inverses="<<invertible<<" noninvertible_rejected="<<noninvertible<<" packed_event_checks="<<n<<" errors=0 scope=arithmetic_only_no_selection_or_speed_claim\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 80;}}
#endif

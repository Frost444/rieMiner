#define R207_LIBRARY
#include "r207_reduce33.cu"
#define R200_LIBRARY
#include "r200_admission.cu"
#include <filesystem>
using r200::ck;
namespace r208 {
constexpr uint64_t low=4000000000ull,high=8000000000ull;
__device__ uint64_t fast_words(const uint32_t* w,uint32_t n,uint64_t p,uint64_t mu){
 uint64_t r=0;for(uint32_t i=n;i;--i)r=reduce66((r<<32)|w[i-1],r>>32,p,mu);return r;
}
__device__ uint64_t slow_words(const uint32_t* w,uint32_t n,uint64_t p){
 uint64_t r=0;for(uint32_t i=n;i;--i)for(int bit=31;bit>=0;--bit){r=(r<<1)|((w[i-1]>>bit)&1u);if(r>=p)r-=p;}return r;
}
__device__ uint64_t slow_product(uint64_t a,uint64_t b,uint64_t p){
 uint64_t r=0;while(b){if(b&1){r+=a;if(r>=p)r-=p;}a+=a;if(a>=p)a-=p;b>>=1;}return r;
}
__global__ void prepare(const uint32_t* delta,uint32_t n,const uint32_t* P,uint32_t pw,
 const uint32_t* A,uint32_t aw,uint64_t* inv,uint64_t* roots,uint32_t* errors){
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;uint64_t p=low+delta[i];
 if(p<=low||p>high||!(p&1)){inv[i]=roots[i]=0;atomicAdd(errors,1u);return;}
 uint64_t mu=UINT64_MAX/p,pm=fast_words(P,pw,p,mu),iv=inverse64(pm,p);
 if(!iv){inv[i]=roots[i]=0;atomicAdd(errors,1u);return;}
 uint64_t am=fast_words(A,aw,p,mu),om=reduce66(r200::OFFSET,0,p,mu);
 uint64_t lo=om*iv,hi=__umul64hi(om,iv),sum=lo+am;hi+=sum<lo;
 uint64_t x=reduce66(sum,hi,p,mu);inv[i]=iv;roots[i]=x?p-x:0;
}
__global__ void guard(const uint32_t* delta,uint32_t n,const uint32_t* P,uint32_t pw,
 const uint32_t* B,uint32_t bw,const uint64_t* inv,const uint64_t* roots,uint32_t* errors){
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;uint64_t p=low+delta[i];
 if(p<=low||p>high||!(p&1)){atomicAdd(errors,1u);return;}
 uint64_t pm=slow_words(P,pw,p),bm=slow_words(B,bw,p),iv=inv[i];
 if(!iv||iv>=p||slow_product(pm,iv,p)!=1){atomicAdd(errors,1u);return;}
 for(unsigned m=0;m<2;++m){uint64_t b=bm+2*m;if(b>=p)b-=p;uint64_t expected=slow_product(b?p-b:0,iv,p),observed=roots[i];
  if(m){uint64_t d=iv*2;if(d>=p)d-=p;observed=observed>=d?observed-d:observed+p-d;}
  if(expected!=observed)atomicAdd(errors,1u);
 }
}
struct Certificate{uint64_t prime;uint32_t factor,member;};
__global__ void filter(const uint32_t* delta,uint32_t n,const uint64_t* inverse,const uint64_t* roots,
 uint32_t* bits,Certificate* cert,uint32_t* taken,uint32_t capacity,uint32_t* errors){
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;uint64_t p=low+delta[i],r=roots[i],iv=inverse[i];
 for(unsigned m=0;m<2;++m){if(r<r200::SPAN){uint32_t mask=1u<<(uint32_t(r)&31u),old=atomicAnd(bits+(r>>5),~mask);
   if(old&mask){uint32_t slot=atomicAdd(taken,1u);if(slot>=capacity)atomicAdd(errors,1u);else cert[slot]={p,uint32_t(r),m};}}
  if(!m){uint64_t d=iv*2;if(d>=p)d-=p;r=r>=d?r-d:r+p-d;}
 }
}
static std::vector<uint32_t> prime_band(uint64_t lo,uint64_t hi){
 auto base_primes=riecoin_prime_root_precompute::primes_to(100000);std::vector<uint32_t> out;out.reserve(size_t((hi-lo)/20));
 constexpr uint64_t K=1u<<18;std::vector<unsigned char> marked(K);
 for(uint64_t a=lo+1;a<=hi;){if(!(a&1))++a;if(a>hi)break;uint64_t b=std::min<uint64_t>(hi|1ull,a+2*(K-1));if(b>hi)b-=2;
  size_t count=size_t((b-a)/2+1);std::fill_n(marked.begin(),count,0);
  for(auto p:base_primes){if(p==2)continue;if(uint64_t(p)*p>b)break;uint64_t first=std::max(uint64_t(p)*p,((a+p-1)/p)*p);if(!(first&1))first+=p;
   for(uint64_t j=(first-a)/2;j<count;j+=p)marked[size_t(j)]=1;}
  for(size_t j=0;j<count;++j)if(!marked[j])out.push_back(uint32_t(a+2*j-low));a=b+2;
 }
 return out;
}
}
#ifndef R208_LIBRARY
int main(int argc,char**argv){try{
 if(argc!=5)throw std::runtime_error("target_hex qualified_bitmap isolated_output_directory expected_uuid");
 ck(cudaSetDevice(0),"device");cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0),"device properties");
 std::ostringstream id;id<<"GPU-"<<std::hex<<std::setfill('0');for(unsigned i=0;i<16;++i){if(i==4||i==6||i==8||i==10)id<<'-';id<<std::setw(2)<<unsigned(static_cast<unsigned char>(prop.uuid.bytes[i]));}
 if(id.str()!=argv[4])throw std::runtime_error("Wrong GPU");size_t free=0,total=0;ck(cudaMemGetInfo(&free,&total),"memory");if(free<(size_t(3)<<30))throw std::runtime_error("Requires3GiB free");
 std::ifstream f(argv[2],std::ios::binary|std::ios::ate);if(!f||f.tellg()!=std::streamoff(r200::NWORDS*4ull))throw std::runtime_error("Bitmap size");f.seekg(0);
 std::vector<uint32_t> input(r200::NWORDS),expected,observed(r200::NWORDS);f.read(reinterpret_cast<char*>(input.data()),input.size()*4);if(!f||r200::population(input)!=123365)throw std::runtime_error("Qualified4B input population");expected=input;
 std::filesystem::path outdir(argv[3]);std::filesystem::create_directories(outdir);
 mpz_t target,P,A,B,tmp,N,prime;mpz_inits(target,P,A,B,tmp,N,prime,nullptr);
 if(mpz_set_str(target,argv[1],16)||mpz_sizeinbase(target,2)!=1168)throw std::runtime_error("Target1168");
 auto prefix=riecoin_prime_root_precompute::primes_to(1000);mpz_set_ui(P,1);for(unsigned i=0;i<114;++i)mpz_mul_ui(P,P,prefix[i]);
 mpz_fdiv_q(A,target,P);mpz_add_ui(A,A,1);mpz_mul(B,A,P);set64(tmp,r200::OFFSET);mpz_add(B,B,tmp);
 auto pw=r200::export_words(P),aw=r200::export_words(A),bw=r200::export_words(B);
 r200::Device<uint32_t> dp(pw.size()),da(aw.size()),db(bw.size()),bits(r200::NWORDS),errors(1),taken(1);
 dp.put(pw.data());da.put(aw.data());db.put(bw.data());bits.put(input.data());ck(cudaMemset(errors.p,0,4),"errors");ck(cudaMemset(taken.p,0,4),"taken");
 r200::Device<r208::Certificate> cert(123365);uint64_t count_primes=0;double generation_ms=0,upload_ms=0,prepare_ms=0,guard_ms=0,filter_ms=0;
 for(uint64_t lo=r208::low;lo<r208::high;lo+=1000000000ull){uint64_t hi=lo+1000000000ull;auto start=r200::Clock::now();
  auto path=outdir/("delta4b-band-"+std::to_string(lo)+"-"+std::to_string(hi)+".u32");std::vector<uint32_t> values;bool reused=false;
  if(std::filesystem::exists(path)){auto size=std::filesystem::file_size(path);if(!size||size%4||size>240000000)throw std::runtime_error("Cached band size");values.resize(size/4);std::ifstream b(path,std::ios::binary);b.read(reinterpret_cast<char*>(values.data()),size);if(!b)throw std::runtime_error("Band read");reused=true;}
  else{values=r208::prime_band(lo,hi);auto temp=path;temp+=".writing";if(std::filesystem::exists(temp))throw std::runtime_error("Incomplete cache transaction");
   {std::ofstream b(temp,std::ios::binary);b.write(reinterpret_cast<const char*>(values.data()),values.size()*4);if(!b)throw std::runtime_error("Band write");}std::filesystem::rename(temp,path);}
  double gen=r200::elapsed(start);generation_ms+=gen;uint32_t n=uint32_t(values.size()),blocks=(n+255)/256;count_primes+=n;
  r200::Device<uint32_t> delta(n);r200::Device<uint64_t> inverse(n),roots(n);start=r200::Clock::now();delta.put(values.data());double up=r200::elapsed(start);upload_ms+=up;
  // Bounded display-GPU launches; one terminal validation, no per-chunk readback.
  constexpr uint32_t chunk=1u<<20;
  double pre=r200::timed([&]{for(uint32_t first=0;first<n;first+=chunk){uint32_t size=std::min(chunk,n-first);r208::prepare<<<(size+255)/256,256>>>(delta.p+first,size,dp.p,uint32_t(pw.size()),da.p,uint32_t(aw.size()),inverse.p+first,roots.p+first,errors.p);}});prepare_ms+=pre;
  double gu=r200::timed([&]{for(uint32_t first=0;first<n;first+=chunk){uint32_t size=std::min(chunk,n-first);r208::guard<<<(size+255)/256,256>>>(delta.p+first,size,dp.p,uint32_t(pw.size()),db.p,uint32_t(bw.size()),inverse.p+first,roots.p+first,errors.p);}});guard_ms+=gu;
  uint32_t err=0;errors.get(&err);if(err)throw std::runtime_error("Independent full-root guard failed");
  double fi=r200::timed([&]{r208::filter<<<blocks,256>>>(delta.p,n,inverse.p,roots.p,bits.p,cert.p,taken.p,123365,errors.p);});filter_ms+=fi;
  bits.get(observed.data());std::cout<<"phase=band high="<<hi<<" primes="<<n<<" remaining="<<r200::population(observed)<<" cache_reused="<<reused<<" cache_bytes="<<values.size()*4ull<<" host_table_ms="<<gen<<" upload_ms="<<up<<" compact_root_ms="<<pre<<" independent_guard_ms="<<gu<<" filter_ms="<<fi<<" errors=0\n"<<std::flush;
 }
 uint32_t count=0,err=0;taken.get(&count);errors.get(&err);if(err||count>123365)throw std::runtime_error("Certificate bound");std::vector<r208::Certificate> proofs(count);ck(cudaMemcpy(proofs.data(),cert.p,count*sizeof(proofs[0]),cudaMemcpyDeviceToHost),"proof download");
 for(const auto& c:proofs){if(c.factor>=r200::SPAN||c.prime<=r208::low||c.prime>r208::high||c.member>1)throw std::runtime_error("Certificate geometry");uint32_t mask=1u<<(c.factor&31);if(!(expected[c.factor>>5]&mask))throw std::runtime_error("Duplicate/missing removed factor");
  mpz_mul_ui(N,P,c.factor);mpz_add(N,N,B);mpz_add_ui(N,N,2*c.member);set64(prime,c.prime);mpz_mod(tmp,N,prime);
  if(mpz_sgn(tmp)||mpz_cmp(N,prime)<=0)throw std::runtime_error("GMP proper-divisor rejection failed");expected[c.factor>>5]&=~mask;}
 bits.get(observed.data());if(observed!=expected||123365-r200::population(observed)!=count)throw std::runtime_error("Full bitmap conservation");
 auto output=outdir/"bitmap-8000000000.bin";if(std::filesystem::exists(output))throw std::runtime_error("Output already exists");std::ofstream of(output,std::ios::binary);of.write(reinterpret_cast<const char*>(observed.data()),observed.size()*4);if(!of)throw std::runtime_error("Bitmap output");
 std::cout<<"phase=PASS initial=123365 remaining="<<r200::population(observed)<<" proper_divisor_certificates="<<count<<" primes="<<count_primes<<" root_relations="<<count_primes*2<<" bitmap_words="<<r200::NWORDS<<" host_table_ms="<<generation_ms<<" upload_ms="<<upload_ms<<" compact_root_ms="<<prepare_ms<<" independent_guard_ms="<<guard_ms<<" filter_ms="<<filter_ms<<" errors=0 scope=exact_selection_only_no_r_or_prp_or_miner_speed\n";
 mpz_clears(target,P,A,B,tmp,N,prime,nullptr);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 80;}}
#endif

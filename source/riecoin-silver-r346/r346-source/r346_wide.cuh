#pragma once
namespace r346cache {
// Exact R229 recurring expression; R208 arithmetic already in this TU.
__global__ void prepare_wide(const uint32_t*delta,uint32_t n,const uint32_t*low,const uint32_t*high,const uint32_t*A,uint32_t aw,const uint32_t*S,uint32_t sw,uint64_t*inverse,uint64_t*roots,uint32_t*error){uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;uint64_t p=r208::low+delta[i];if(p<=r208::low||p>r208::high||!(p&1)){atomicOr(error,1u);return;}uint64_t anchor=uint64_t(low[i])|(uint64_t((high[i/32]>>(i%32))&1u)<<32);if(!anchor||anchor>=p){atomicOr(error,2u);return;}uint64_t mu=UINT64_MAX/p,s=r208::fast_words(S,sw,p,mu),iv=reduce66(s*anchor,__umul64hi(s,anchor),p,mu);if(!iv){atomicOr(error,4u);return;}uint64_t a=r208::fast_words(A,aw,p,mu),o=reduce66(r200::OFFSET,0,p,mu),prod=o*iv,sum=prod+a,x=reduce66(sum,__umul64hi(o,iv)+uint64_t(sum<prod),p,mu);inverse[i]=iv;roots[i]=x?p-x:0;}
}

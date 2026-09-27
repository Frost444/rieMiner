#pragma once
#include "r289_sos_launch.hpp"
#include "r294_sos_launch.hpp"
extern "C" cudaError_t riecoin_a99b_launch_oracle(const void*,std::uint8_t*,std::uint32_t*,std::uint32_t,cudaStream_t);
namespace r294 {inline bool enabled=true;}
inline cudaError_t r294_dense_dispatch(const void* candidates,std::uint8_t* verdicts,
    const std::uint32_t* slots,const std::uint32_t* total,std::uint32_t capacity,cudaStream_t stream){
  return r294::enabled?riecoin_r294_launch_dense(candidates,verdicts,slots,total,capacity,stream):
    riecoin_r289_launch_dense(candidates,verdicts,slots,total,capacity,stream);
}
inline cudaError_t r294_member_dispatch(const void* candidates,std::uint8_t* verdicts,
    std::uint32_t* residues,std::uint32_t count,cudaStream_t stream){
  return r294::enabled?riecoin_r294_launch_oracle(candidates,verdicts,residues,count,stream):
    riecoin_a99b_launch_oracle(candidates,verdicts,residues,count,stream);
}

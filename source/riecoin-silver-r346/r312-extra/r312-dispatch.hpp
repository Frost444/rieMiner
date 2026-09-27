#pragma once
#include "r294_dispatch.hpp"
#include "r308_sos_launch.hpp"
#include "r312-middle-bounds.hpp"
inline cudaError_t r312_dense_dispatch(const void* candidates,std::uint8_t* verdicts,
    const std::uint32_t* slots,const std::uint32_t* total,std::uint32_t capacity,cudaStream_t stream){
  return r312::admitted?riecoin_r308_launch_dense(candidates,verdicts,slots,total,capacity,stream):
    r294_dense_dispatch(candidates,verdicts,slots,total,capacity,stream);
}

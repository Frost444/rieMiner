#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
extern "C" cudaError_t riecoin_r297_launch_dense(const void*,std::uint8_t*,
 const std::uint32_t*,const std::uint32_t*,std::uint32_t,cudaStream_t);

extern "C" cudaError_t riecoin_r297_launch_oracle(const void*,std::uint8_t*,std::uint32_t*,std::uint32_t,cudaStream_t);

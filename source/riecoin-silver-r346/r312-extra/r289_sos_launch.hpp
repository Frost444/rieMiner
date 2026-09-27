#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
extern "C" cudaError_t riecoin_r289_launch_dense(const void*,std::uint8_t*,
 const std::uint32_t*,const std::uint32_t*,std::uint32_t,cudaStream_t);

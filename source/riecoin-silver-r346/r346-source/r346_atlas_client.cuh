#pragma once
#ifdef _WIN32
#include "r346_atlas_protocol.hpp"
#endif
namespace r346cache {
inline std::unique_ptr<r346cache::View> open(){
#ifdef _WIN32
 try{
  CUdevice device;r346cache::driver(cuCtxGetDevice(&device));CUuuid uuid;r346cache::driver(cuDeviceGetUuid(&uuid,device));
  auto capability=r346ipc::acquire(reinterpret_cast<const uint8_t*>(uuid.bytes));if(!capability)return {};
  HANDLE handle=capability->handle.h;capability->handle.h=nullptr;
  auto view=std::make_unique<r346cache::View>(handle,size_t(capability->bytes));
  r345::out()<<"phase=r346_atlas_lease access=READ_ONLY bytes="<<capability->bytes<<" source=qualified_local_owner\n"<<std::flush;return view;
 }catch(const std::exception&){return {};}
#else
 // Same fallback as an unavailable Windows owner; admission/certification stay on.
 r345::out()<<"phase=r346_atlas_fallback reason=platform_no_shared_lease exact_per_job=1\n"<<std::flush;
 return {};
#endif
}
}

#pragma once
#include "../r184_verified_file_lease.hpp"
#include <cstring>
// Optional local capability transfer. A bad/missing lease NEVER authorizes
// admission: consumer keeps the complete independent per-job root verifier.
namespace r346ipc {
using r184_lease::Handle;using r184_lease::check;using r184_lease::io;using r184_lease::alive;
constexpr uint32_t Magic=0x52333436,Version=1;
inline uint64_t birth(HANDLE p){FILETIME c{},e{},k{},u{};check(GetProcessTimes(p,&c,&e,&k,&u)!=0,"atlas process birth");return (uint64_t(c.dwHighDateTime)<<32)|c.dwLowDateTime;}
inline std::string digest(std::string s){check(s.size()==64&&s.find_first_not_of("0123456789abcdefABCDEF")==std::string::npos,"atlas SHA format");for(auto&c:s)c=char(std::tolower(static_cast<unsigned char>(c)));return s;}
inline auto geometry(){constexpr char key[]="R346/P114/O114023297140211/1-8B/47374753,46227250,45512275,44992411,44591145,44258984,43979302/p32x3,delta4B32x4,invlow32,invhi1/aligned256/header4096";return r184_lease::sha(reinterpret_cast<const uint8_t*>(key),sizeof(key)-1);}
#pragma pack(push,1)
struct Request {uint32_t magic=Magic,version=Version;uint64_t client_birth=0;uint8_t nonce[16]{},gpu[16]{};};
struct Reply {uint32_t magic=Magic,version=Version;uint64_t handle=0,bytes=0,owner_birth=0;uint32_t owner_pid=0,client_pid=0;uint8_t nonce[16]{},gpu[16]{},geometry[32]{};};
#pragma pack(pop)
struct Capability {Handle handle;uint64_t bytes=0;Capability(Handle h,uint64_t n):handle(std::move(h)),bytes(n){}};
inline std::optional<Capability> acquire(const uint8_t* gpu){
 const char*name=std::getenv("ASTRA_R346_ATLAS_PIPE"),*hash=std::getenv("ASTRA_R346_OWNER_SHA256");if(!name||!hash)return {};
 try{
  const std::string pipeName(name);check(pipeName.rfind("\\\\.\\pipe\\R346-",0)==0&&pipeName.size()<200,"atlas pipe scope");const auto expected=digest(hash);
  Handle pipe(CreateFileA(name,GENERIC_READ|GENERIC_WRITE,0,nullptr,OPEN_EXISTING,FILE_FLAG_OVERLAPPED,nullptr));check(bool(pipe),"atlas unavailable");ULONG pid=0;check(GetNamedPipeServerProcessId(pipe.h,&pid)!=0,"atlas server PID");Handle owner(OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION|SYNCHRONIZE,FALSE,pid));check(bool(owner)&&alive(owner.h),"atlas owner alive");const auto ownerBirth=birth(owner.h);check(r184_lease::process_image_hash(owner.h)==expected,"atlas owner identity");
  Request request;request.client_birth=birth(GetCurrentProcess());memcpy(request.gpu,gpu,16);check(BCryptGenRandom(nullptr,request.nonce,16,BCRYPT_USE_SYSTEM_PREFERRED_RNG)>=0,"atlas request nonce");check(io(pipe.h,&request,sizeof(request),true),"atlas request");Reply reply;check(io(pipe.h,&reply,sizeof(reply),false),"atlas response");
  const auto shape=geometry();check(reply.magic==Magic&&reply.version==Version&&reply.owner_pid==pid&&reply.owner_birth==ownerBirth&&reply.client_pid==GetCurrentProcessId()&&!memcmp(reply.nonce,request.nonce,16)&&!memcmp(reply.gpu,gpu,16)&&!memcmp(reply.geometry,shape.data(),32),"atlas response binding");
  Handle received(reinterpret_cast<HANDLE>(uintptr_t(reply.handle)));check(bool(received)&&reply.bytes>=2575000000ull&&reply.bytes<2640000000ull&&alive(owner.h),"atlas capability bounds/lifetime");uint32_t ack=Magic;check(io(pipe.h,&ack,sizeof(ack),true),"atlas transfer acknowledgement");return Capability(std::move(received),reply.bytes);
 }catch(const std::exception&){return {};}
}
}

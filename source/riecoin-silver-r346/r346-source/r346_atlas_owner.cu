// GPU table service only. No candidate, PRP, wallet, pool or process restart.
#include "r346_atlas_builder.cuh"
#include "r346_atlas_protocol.hpp"
#include <thread>
namespace r346owner {
using r346ipc::Handle;using r346ipc::check;using r346ipc::alive;
std::string pipeName,clientSha;uint8_t gpuBytes[16]{};HANDLE guardian=nullptr;bool servedOnce=false;
struct Lifetime{Handle done;std::thread worker;Lifetime(HANDLE parent,DWORD ttl):done(CreateEventW(nullptr,TRUE,FALSE,nullptr)){check(bool(done),"guardian event");HANDLE finish=done.h;worker=std::thread([parent,finish,ttl]{HANDLE hs[]={finish,parent};DWORD r=WaitForMultipleObjects(2,hs,FALSE,ttl);if(r==WAIT_OBJECT_0)return;TerminateProcess(GetCurrentProcess(),r==WAIT_OBJECT_0+1?0:3);});}~Lifetime(){SetEvent(done.h);if(worker.joinable())worker.join();}};
void closeRemote(HANDLE process,HANDLE remote){if(!remote)return;HANDLE local=nullptr;if(DuplicateHandle(process,remote,GetCurrentProcess(),&local,0,FALSE,DUPLICATE_CLOSE_SOURCE))CloseHandle(local);}
bool connect(HANDLE pipe){Handle event(CreateEventW(nullptr,TRUE,FALSE,nullptr));check(bool(event),"pipe event");OVERLAPPED op{};op.hEvent=event.h;if(ConnectNamedPipe(pipe,&op))return true;DWORD e=GetLastError();if(e==ERROR_PIPE_CONNECTED)return true;check(e==ERROR_IO_PENDING||e==ERROR_NO_DATA||e==ERROR_PIPE_NOT_CONNECTED,"pipe connect");if(e!=ERROR_IO_PENDING)return false;HANDLE wait[]={event.h,guardian};DWORD done=0;if(WaitForMultipleObjects(2,wait,FALSE,1000)!=WAIT_OBJECT_0){CancelIoEx(pipe,&op);GetOverlappedResult(pipe,&op,&done,TRUE);return false;}return GetOverlappedResult(pipe,&op,&done,FALSE)!=0;}
void service(const std::string&,const char*,HANDLE exported,size_t bytes){
 if(servedOnce)return;servedOnce=true;Handle pipe(CreateNamedPipeA(pipeName.c_str(),PIPE_ACCESS_DUPLEX|FILE_FLAG_OVERLAPPED|FILE_FLAG_FIRST_PIPE_INSTANCE,PIPE_TYPE_BYTE|PIPE_READMODE_BYTE|PIPE_WAIT|PIPE_REJECT_REMOTE_CLIENTS,1,4096,4096,0,nullptr));check(bool(pipe),"exclusive local atlas pipe");
 std::printf("{\"phase\":\"r346_ready\",\"pid\":%lu,\"birth\":%llu,\"bytes\":%zu,\"read_only\":true}\n",GetCurrentProcessId(),(unsigned long long)r346ipc::birth(GetCurrentProcess()),bytes);std::fflush(stdout);
 while(alive(guardian)){
  if(!connect(pipe.h)){DisconnectNamedPipe(pipe.h);continue;}
  r346ipc::Request request;ULONG pid=0;
  if(!r346ipc::io(pipe.h,&request,sizeof(request),false)||request.magic!=r346ipc::Magic||request.version!=r346ipc::Version||memcmp(request.gpu,gpuBytes,16)||!GetNamedPipeClientProcessId(pipe.h,&pid)){DisconnectNamedPipe(pipe.h);continue;}
  Handle client(OpenProcess(PROCESS_DUP_HANDLE|PROCESS_QUERY_LIMITED_INFORMATION|SYNCHRONIZE,FALSE,pid));
  try{check(bool(client)&&alive(client.h)&&r346ipc::birth(client.h)==request.client_birth,"client lifetime");check(r184_lease::process_image_hash(client.h)==clientSha,"client binary identity");}catch(...){DisconnectNamedPipe(pipe.h);continue;}
  HANDLE remote=nullptr;if(!DuplicateHandle(GetCurrentProcess(),exported,client.h,&remote,0,FALSE,DUPLICATE_SAME_ACCESS)){DisconnectNamedPipe(pipe.h);continue;}
  r346ipc::Reply reply;reply.handle=uint64_t(uintptr_t(remote));reply.bytes=bytes;reply.owner_pid=GetCurrentProcessId();reply.client_pid=pid;reply.owner_birth=r346ipc::birth(GetCurrentProcess());memcpy(reply.nonce,request.nonce,16);memcpy(reply.gpu,gpuBytes,16);auto shape=r346ipc::geometry();memcpy(reply.geometry,shape.data(),32);
  if(r346ipc::io(pipe.h,&reply,sizeof(reply),true)){uint32_t ack=0;bool ok=r346ipc::io(pipe.h,&ack,sizeof(ack),false,3000)&&ack==r346ipc::Magic;std::printf("phase=r346_transfer client=%lu acknowledged=%u\n",pid,ok?1:0);std::fflush(stdout);}else closeRemote(client.h,remote);
  DisconnectNamedPipe(pipe.h);
 }
}
}
int main(int argc,char**argv){try{
 if(argc!=8)throw std::runtime_error("GPU PIPE GUARDIAN_PID BIRTH GUARDIAN_SHA CLIENT_SHA TTL_SECONDS required");
 r346owner::pipeName=argv[2];r346ipc::check(r346owner::pipeName.rfind("\\\\.\\pipe\\R346-",0)==0&&r346owner::pipeName.size()<200,"pipe scope");
 auto parentId=std::stoull(argv[3]),expectedBirth=std::stoull(argv[4]),ttl=std::stoull(argv[7]);r346ipc::check(parentId&&parentId<=MAXDWORD&&ttl&&ttl<=4294967,"lifetime bounds");auto parentSha=r346ipc::digest(argv[5]);r346owner::clientSha=r346ipc::digest(argv[6]);
 r346ipc::Handle parent(OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION|SYNCHRONIZE,FALSE,DWORD(parentId)));r346ipc::check(bool(parent)&&r346ipc::alive(parent.h)&&r346ipc::birth(parent.h)==expectedBirth&&r184_lease::process_image_hash(parent.h)==parentSha,"exact guardian identity");r346owner::guardian=parent.h;r346owner::Lifetime lifetime(parent.h,DWORD(ttl*1000));
 CUuuid uuid;ck(cuDeviceGetUuid(&uuid,find(argv[1])));memcpy(r346owner::gpuBytes,uuid.bytes,16);
 return r346builder::owner(argv[1],r346owner::service); // no poison hook, ever
}catch(const std::exception&e){std::fprintf(stderr,"R346 owner: %s\n",e.what());return 80;}}

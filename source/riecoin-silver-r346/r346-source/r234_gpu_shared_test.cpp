// Closed Win32/CUDA sharing capability test, not a production cache owner.
// Two child processes read one allocation; no prime candidate or submission.
#include <windows.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
void ck(CUresult e){if(e!=CUDA_SUCCESS){const char* s="unknown";cuGetErrorName(e,&s);throw std::runtime_error(s);}}
void win(bool ok,const char* why){if(!ok)throw std::runtime_error(std::string(why)+" win32="+std::to_string(GetLastError()));}
CUdevice find(const char* wanted){ck(cuInit(0));int n=0;ck(cuDeviceGetCount(&n));for(int i=0;i<n;i++){CUdevice d;ck(cuDeviceGet(&d,i));CUuuid u;ck(cuDeviceGetUuid(&u,d));char id[64]="GPU-",*at=id+4;for(int j=0;j<16;j++){if(j==4||j==6||j==8||j==10)*at++='-';std::sprintf(at,"%02x",unsigned(static_cast<unsigned char>(u.bytes[j])));at+=2;}if(!std::strcmp(id,wanted))return d;}throw std::runtime_error("exact GPU absent");}
struct Context {CUdevice d;CUcontext c{};explicit Context(CUdevice device):d(device){ck(cuDevicePrimaryCtxRetain(&c,d));ck(cuCtxSetCurrent(c));}~Context(){cuDevicePrimaryCtxRelease(d);}};
struct Allocation {CUmemGenericAllocationHandle h{};~Allocation(){if(h)cuMemRelease(h);}};
struct Mapping {CUdeviceptr p{};size_t bytes;bool mapped=false;explicit Mapping(size_t n):bytes(n){ck(cuMemAddressReserve(&p,n,0,0,0));}~Mapping(){if(mapped)cuMemUnmap(p,bytes);if(p)cuMemAddressFree(p,bytes);}void bind(CUmemGenericAllocationHandle h,CUdevice d,CUmemAccess_flags flags){ck(cuMemMap(p,bytes,0,h,0));mapped=true;CUmemAccessDesc access{};access.location.type=CU_MEM_LOCATION_TYPE_DEVICE;access.location.id=d;access.flags=flags;ck(cuMemSetAccess(p,bytes,&access,1));}};
uint32_t value(size_t i){return uint32_t(i)*2654435761u+0x234abcd;
}
int reader(const char* gpu,HANDLE shared,size_t bytes){
 if(!bytes||bytes>16*1024*1024||bytes%4)throw std::runtime_error("closed payload bound");
 Context ctx(find(gpu));Allocation imported;ck(cuMemImportFromShareableHandle(&imported.h,shared,CU_MEM_HANDLE_TYPE_WIN32));Mapping map(bytes);map.bind(imported.h,ctx.d,CU_MEM_ACCESS_FLAGS_PROT_READ);
 CUmemLocation location{};location.type=CU_MEM_LOCATION_TYPE_DEVICE;location.id=ctx.d;unsigned long long flags=0;ck(cuMemGetAccess(&flags,&location,map.p));if(flags!=CU_MEM_ACCESS_FLAGS_PROT_READ)throw std::runtime_error("consumer not read-only");
 std::vector<uint32_t> host(bytes/4);ck(cuMemcpyDtoH(host.data(),map.p,bytes));ck(cuCtxSynchronize());for(size_t i=0;i<host.size();i++)if(host[i]!=value(i))throw std::runtime_error("shared payload mismatch index="+std::to_string(i)+" expected="+std::to_string(value(i))+" got="+std::to_string(host[i]));
 std::printf("phase=r234_reader PASS words=%zu access=READ_ONLY\n",host.size());CloseHandle(shared);return 0;
}
void child(const std::string& exe,const char* gpu,HANDLE shared,size_t bytes){
 // Parent-owned, non-inherited job: even external cancellation cannot orphan
 // a reader on the GPU. Assign suspended, before it can create a context.
 struct OwnedJob { HANDLE h=CreateJobObjectW(nullptr,nullptr); ~OwnedJob(){if(h)CloseHandle(h);} } job;
 win(job.h!=nullptr,"reader job");JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
 limits.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
 win(SetInformationJobObject(job.h,JobObjectExtendedLimitInformation,&limits,sizeof(limits))!=0,"reader job lifetime");
 // Explicit inheritance list: only this test's allocation and output handles.
 win(SetHandleInformation(shared,HANDLE_FLAG_INHERIT,HANDLE_FLAG_INHERIT)!=0,"inherit allocation handle");
 HANDLE out=GetStdHandle(STD_OUTPUT_HANDLE),err=GetStdHandle(STD_ERROR_HANDLE);
 std::vector<HANDLE> handles{shared};for(HANDLE h:{out,err})if(h&&h!=INVALID_HANDLE_VALUE){win(SetHandleInformation(h,HANDLE_FLAG_INHERIT,HANDLE_FLAG_INHERIT)!=0,"inherit diagnostic handle");if(handles.back()!=h)handles.push_back(h);}
 SIZE_T attrBytes=0;InitializeProcThreadAttributeList(nullptr,1,0,&attrBytes);std::vector<unsigned char> attrs(attrBytes);auto* list=reinterpret_cast<PPROC_THREAD_ATTRIBUTE_LIST>(attrs.data());win(InitializeProcThreadAttributeList(list,1,0,&attrBytes)!=0,"attribute list");
 win(UpdateProcThreadAttribute(list,0,PROC_THREAD_ATTRIBUTE_HANDLE_LIST,handles.data(),handles.size()*sizeof(HANDLE),nullptr,nullptr)!=0,"explicit handles");
 STARTUPINFOEXA si{};si.StartupInfo.cb=sizeof(si);si.StartupInfo.dwFlags=STARTF_USESTDHANDLES;si.StartupInfo.hStdOutput=out;si.StartupInfo.hStdError=err;si.lpAttributeList=list;PROCESS_INFORMATION pi{};
 std::string command='"'+exe+"\" --reader "+gpu+' '+std::to_string(uintptr_t(shared))+' '+std::to_string(bytes);
 const BOOL created=CreateProcessA(exe.c_str(),command.data(),nullptr,nullptr,TRUE,EXTENDED_STARTUPINFO_PRESENT|CREATE_NO_WINDOW|CREATE_SUSPENDED,nullptr,nullptr,&si.StartupInfo,&pi);DeleteProcThreadAttributeList(list);win(created!=0,"exact child start");
 if(!AssignProcessToJobObject(job.h,pi.hProcess)||ResumeThread(pi.hThread)==DWORD(-1)){TerminateProcess(pi.hProcess,89);WaitForSingleObject(pi.hProcess,10000);CloseHandle(pi.hThread);CloseHandle(pi.hProcess);throw std::runtime_error("reader job assignment/start");}CloseHandle(pi.hThread);
 const DWORD waited=WaitForSingleObject(pi.hProcess,15000);if(waited!=WAIT_OBJECT_0){TerminateProcess(pi.hProcess,88);WaitForSingleObject(pi.hProcess,10000);CloseHandle(pi.hProcess);throw std::runtime_error("owned reader timeout");}
 DWORD exit=1;win(GetExitCodeProcess(pi.hProcess,&exit)!=0,"child verdict");CloseHandle(pi.hProcess);if(exit)throw std::runtime_error("child verification failed");
}
#ifndef R234_LIBRARY
int main(int argc,char**argv){try{
 if(argc==5&&!std::strcmp(argv[1],"--reader"))return reader(argv[2],reinterpret_cast<HANDLE>(uintptr_t(std::stoull(argv[3]))),size_t(std::stoull(argv[4])));
 if(argc!=2)throw std::runtime_error("expected GPU UUID");Context ctx(find(argv[1]));CUmemAllocationProp prop{};prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;prop.requestedHandleTypes=CU_MEM_HANDLE_TYPE_WIN32;prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;prop.location.id=ctx.d;SECURITY_ATTRIBUTES sa{sizeof(sa),nullptr,FALSE};prop.win32HandleMetaData=&sa;
 size_t granularity=0;ck(cuMemGetAllocationGranularity(&granularity,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM));const size_t bytes=((2*1024*1024+granularity-1)/granularity)*granularity;if(bytes>16*1024*1024)throw std::runtime_error("allocation granularity bound");
 Allocation owner;ck(cuMemCreate(&owner.h,bytes,&prop,0));Mapping map(bytes);map.bind(owner.h,ctx.d,CU_MEM_ACCESS_FLAGS_PROT_READWRITE);std::vector<uint32_t> data(bytes/4);for(size_t i=0;i<data.size();i++)data[i]=value(i);ck(cuMemcpyHtoD(map.p,data.data(),bytes));
 // Pageable HtoD completion must precede publication to a different context.
 ck(cuCtxSynchronize());std::vector<uint32_t> local(bytes/4);ck(cuMemcpyDtoH(local.data(),map.p,bytes));if(local!=data)throw std::runtime_error("owner readback failed before publication");
 std::printf("phase=r234_owner initialization_fenced=1 readback=PASS words=%zu\n",data.size());std::fflush(stdout);
 char path[32768];const auto length=GetModuleFileNameA(nullptr,path,sizeof(path));win(length&&length<sizeof(path),"self executable");
 for(unsigned i=0;i<2;i++){HANDLE handle=nullptr;ck(cuMemExportToShareableHandle(&handle,owner.h,CU_MEM_HANDLE_TYPE_WIN32,0));try{child(std::string(path,length),argv[1],handle,bytes);}catch(...){CloseHandle(handle);throw;}CloseHandle(handle);}
 std::printf("{\"verdict\":\"PASS\",\"scope\":\"closed CUDA Win32 sharing; not miner performance\",\"readers\":2,\"bytes\":%zu,\"consumer_access\":\"read_only\",\"kernels\":0}\n",bytes);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"R234: %s\n",e.what());return 1;}}
#endif

#pragma once
// RC2:OS-BOUNDARY — Debian 12 process/file primitives for the unchanged protocol.
// Names mirror the existing call sites; this is not a general Win32 emulation.
#ifndef _WIN32
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <sys/prctl.h>
#include <netdb.h>
#include <poll.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <cstdint>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
using DWORD=std::uint32_t;
using ULONG=std::uint32_t;
using SOCKET=int;
constexpr int INVALID_SOCKET=-1, SOCKET_ERROR=-1;
constexpr int WSAEWOULDBLOCK=EWOULDBLOCK, WSAEINTR=EINTR;
constexpr bool FALSE=false, TRUE=true;
constexpr DWORD WAIT_OBJECT_0=0, WAIT_TIMEOUT=258, WAIT_FAILED=0xffffffffU;
constexpr DWORD WAIT_ABANDONED=128, STILL_ACTIVE=259, SYNCHRONIZE=0;
constexpr DWORD FILE_APPEND_DATA=1, GENERIC_WRITE=2, FILE_SHARE_READ=0;
constexpr DWORD OPEN_ALWAYS=1, CREATE_NEW=2, CREATE_ALWAYS=3, FILE_ATTRIBUTE_NORMAL=0;
constexpr DWORD HANDLE_FLAG_INHERIT=1, STD_INPUT_HANDLE=0, STD_OUTPUT_HANDLE=1;
struct SECURITY_ATTRIBUTES {std::size_t size;void* security;bool inherit;};
constexpr DWORD MOVEFILE_REPLACE_EXISTING=1, MOVEFILE_WRITE_THROUGH=2;
struct Rc2Handle {
 enum class Kind {file, lock, process, pipe};
 Kind kind; int fd; pid_t pid=0; bool owned=false; DWORD cancelled=0;
};
using HANDLE=Rc2Handle*;
constexpr HANDLE INVALID_HANDLE_VALUE=nullptr;
struct PROCESS_INFORMATION { HANDLE hProcess=nullptr, hThread=nullptr; };
inline DWORD GetCurrentProcessId(){ return static_cast<DWORD>(getpid()); }
inline std::uint64_t GetTickCount64(){
 return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
   std::chrono::steady_clock::now().time_since_epoch()).count());
}
inline void Sleep(DWORD ms){ std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }
inline void SecureZeroMemory(void* p,std::size_t n){ OPENSSL_cleanse(p,n); }
inline int WSAGetLastError(){return errno;}
inline int closesocket(int fd){return close(fd);}
inline int ioctlsocket(int fd,unsigned long request,unsigned long* value){
 if(request!=FIONBIO){errno=EINVAL;return -1;}int mode=*value?1:0;return ioctl(fd,FIONBIO,&mode);
}
inline HANDLE CreateFileW(const char* path,DWORD access,DWORD,void*,DWORD creation,DWORD,HANDLE){
 int flags=O_WRONLY|O_CLOEXEC|O_NOFOLLOW|O_CREAT;
 flags|=creation==CREATE_NEW?O_EXCL:0;
 flags|=creation==CREATE_ALWAYS?O_TRUNC:0;
 if(access==FILE_APPEND_DATA) flags|=O_APPEND;
 int fd=open(path,flags,0600); if(fd<0)return nullptr;
 struct stat st{};
 if(fstat(fd,&st)!=0||!S_ISREG(st.st_mode)||flock(fd,LOCK_EX|LOCK_NB)!=0){close(fd);return nullptr;}
 return new Rc2Handle{Rc2Handle::Kind::file,fd};
}
inline bool WriteFile(HANDLE h,const void* data,DWORD n,DWORD* done,void*){
 *done=0; if(!h||(h->kind!=Rc2Handle::Kind::file&&h->kind!=Rc2Handle::Kind::pipe))return false;
 ssize_t k; do{k=h->kind==Rc2Handle::Kind::pipe?send(h->fd,data,n,MSG_NOSIGNAL):write(h->fd,data,n);}while(k<0&&errno==EINTR);
 if(k<0)return false;*done=static_cast<DWORD>(k);return true;
}
inline bool FlushFileBuffers(HANDLE h){return h&&fsync(h->fd)==0;}
inline bool ReadFile(HANDLE h,void* data,DWORD n,DWORD* done,void*){
 *done=0;if(!h)return false;ssize_t k;do{k=read(h->fd,data,n);}while(k<0&&errno==EINTR);
 if(k<0)return false;*done=static_cast<DWORD>(k);return true;
}
inline bool CreatePipe(HANDLE* a,HANDLE* b,void*,DWORD){
 int f[2];if(socketpair(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC,0,f)!=0)return false;
 *a=new Rc2Handle{Rc2Handle::Kind::pipe,f[0]};*b=new Rc2Handle{Rc2Handle::Kind::pipe,f[1]};return true;
}
inline bool SetHandleInformation(HANDLE h,DWORD,DWORD value){
 return h&&fcntl(h->fd,F_SETFD,value?0:FD_CLOEXEC)==0;
}
inline HANDLE GetStdHandle(DWORD which){
 int fd=fcntl(static_cast<int>(which),F_DUPFD_CLOEXEC,3);
 return fd<0?nullptr:new Rc2Handle{Rc2Handle::Kind::pipe,fd};
}
inline bool PeekNamedPipe(HANDLE h,void* data,DWORD n,DWORD* read_count,DWORD* available,DWORD*){
 if(!h)return false;int ready=0;if(ioctl(h->fd,FIONREAD,&ready)!=0)return false;
 *available=static_cast<DWORD>(ready);*read_count=0;if(!ready)return true;
 ssize_t k=recv(h->fd,data,n,MSG_PEEK|MSG_DONTWAIT);if(k<0)return false;
 *read_count=static_cast<DWORD>(k);return true;
}
inline bool DeleteFileW(const char* p){return unlink(p)==0;}
inline bool MoveFileExW(const char* from,const char* to,DWORD){
 if(rename(from,to)!=0)return false;
 const auto parent=std::filesystem::path(to).parent_path();
 int fd=open((parent.empty()?std::filesystem::path("."):parent).c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC);
 if(fd<0)return false;bool ok=fsync(fd)==0;close(fd);return ok;
}
inline HANDLE rc2_open_lock(const std::filesystem::path& state){
 std::filesystem::create_directories(state.parent_path());
 const auto name=state.string()+".lock";
 int fd=open(name.c_str(),O_RDWR|O_CREAT|O_NOFOLLOW|O_CLOEXEC,0600);
 if(fd<0)return nullptr;struct stat st{};
 if(fstat(fd,&st)!=0||!S_ISREG(st.st_mode)){close(fd);return nullptr;}
 return new Rc2Handle{Rc2Handle::Kind::lock,fd};
}
inline DWORD WaitForSingleObject(HANDLE h,DWORD ms){
 if(!h)return WAIT_FAILED;
 const auto until=GetTickCount64()+ms;
 if(h->kind==Rc2Handle::Kind::lock){
  do{if(flock(h->fd,LOCK_EX|LOCK_NB)==0)return WAIT_OBJECT_0;
   if(errno!=EINTR&&errno!=EWOULDBLOCK)return WAIT_FAILED;
   if(GetTickCount64()>=until)return WAIT_TIMEOUT;Sleep(1);
  }while(true);
 }
 if(h->kind!=Rc2Handle::Kind::process)return WAIT_FAILED;
 pollfd p{h->fd,POLLIN,0};
 for(;;){const auto now=GetTickCount64();const int left=static_cast<int>(std::min<std::uint64_t>(
  until>now?until-now:0,2147483647));
  int r=poll(&p,1,left);if(r<0&&errno==EINTR)continue;
  return r==0?WAIT_TIMEOUT:r>0&&(p.revents&POLLIN)?WAIT_OBJECT_0:WAIT_FAILED;}
}
inline bool ReleaseMutex(HANDLE h){return h&&h->kind==Rc2Handle::Kind::lock&&flock(h->fd,LOCK_UN)==0;}
inline HANDLE OpenProcess(DWORD,bool,DWORD pid){
 int fd=static_cast<int>(syscall(SYS_pidfd_open,static_cast<pid_t>(pid),0));
 return fd<0?nullptr:new Rc2Handle{Rc2Handle::Kind::process,fd,static_cast<pid_t>(pid),false,0};
}
inline bool GetExitCodeProcess(HANDLE h,DWORD* code){
 if(!h||!h->owned)return false;
 siginfo_t si{};
 if(waitid(P_PID,static_cast<id_t>(h->pid),&si,WEXITED|WNOHANG|WNOWAIT)!=0)return false;
 if(si.si_pid==0){*code=STILL_ACTIVE;return true;}
 if(si.si_code==CLD_EXITED)*code=static_cast<DWORD>(si.si_status);
 else if(si.si_code==CLD_KILLED&&si.si_status==SIGKILL&&h->cancelled)*code=h->cancelled;
 else *code=0x80000000U|static_cast<DWORD>(si.si_status);
 return true;
}
inline bool TerminateProcess(HANDLE h,DWORD code){
 if(!h||!h->owned||WaitForSingleObject(h,0)!=WAIT_TIMEOUT)return false;
 // pidfd identifies this exact child; its unreaped leader prevents PGID reuse.
 if(syscall(SYS_pidfd_send_signal,h->fd,SIGKILL,nullptr,0)!=0)return false;
 h->cancelled=code;kill(-h->pid,SIGKILL);return true;
}
inline bool CloseHandle(HANDLE h){
 if(!h)return false;
 if(h->kind==Rc2Handle::Kind::process&&h->owned){
  if(WaitForSingleObject(h,0)==WAIT_TIMEOUT)TerminateProcess(h,0xC000013AU);
  if(WaitForSingleObject(h,5000)!=WAIT_OBJECT_0)std::terminate();
  int status;while(waitpid(h->pid,&status,0)<0&&errno==EINTR){}
 }
 close(h->fd);delete h;return true;
}
inline HANDLE rc2_spawn(const std::vector<std::string>& arguments,
                       const std::filesystem::path& directory,
                       const std::filesystem::path& logfile,
                       HANDLE input_handle=nullptr,HANDLE output_handle=nullptr,HANDLE error_handle=nullptr){
 if(arguments.empty())throw std::runtime_error("empty stage argv");
 const std::string executable=std::filesystem::absolute(arguments[0]).string();
 std::vector<char*> argv;for(const auto& arg:arguments)argv.push_back(const_cast<char*>(arg.c_str()));argv.push_back(nullptr);
 int log=error_handle?fcntl(error_handle->fd,F_DUPFD_CLOEXEC,3):
  open(logfile.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);
 if(log<0)throw std::runtime_error("stage log creation failed");
 int errors[2];if(pipe2(errors,O_CLOEXEC)!=0){close(log);throw std::runtime_error("stage error pipe failed");}
 const pid_t parent=getpid(),pid=fork();
 if(pid==0){
  close(errors[0]);int failure=0;
  if(setpgid(0,0)!=0||prctl(PR_SET_PDEATHSIG,SIGKILL)!=0||getppid()!=parent||
     chdir(directory.c_str())!=0||dup2(output_handle?output_handle->fd:log,STDOUT_FILENO)<0||dup2(log,STDERR_FILENO)<0)failure=errno?errno:ECHILD;
  close(log);int input=input_handle?dup(input_handle->fd):open("/dev/null",O_RDONLY);if(input>=0){dup2(input,STDIN_FILENO);close(input);}
  if(!failure){execv(executable.c_str(),argv.data());failure=errno;}
  (void)write(errors[1],&failure,sizeof(failure));_exit(127);
 }
 close(log);close(errors[1]);
 if(pid<0){close(errors[0]);throw std::runtime_error("stage fork failed");}
 int error=0;ssize_t read_bytes;do{read_bytes=read(errors[0],&error,sizeof(error));}while(read_bytes<0&&errno==EINTR);close(errors[0]);
 HANDLE h=OpenProcess(0,false,static_cast<DWORD>(pid));
 if(!h){kill(pid,SIGKILL);int status;while(waitpid(pid,&status,0)<0&&errno==EINTR){};throw std::runtime_error("pidfd unavailable (Linux 5.3+ required)");}
 h->owned=true;
 if(read_bytes!=0){CloseHandle(h);throw std::runtime_error("stage exec failed: "+std::to_string(error));}
 return h;
}
#endif

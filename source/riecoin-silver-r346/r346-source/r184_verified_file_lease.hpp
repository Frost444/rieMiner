#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#pragma comment(lib,"bcrypt.lib")

// An optional, read-only lease over the very file object verified by an owner.
// It is not permission to trust a path, mtime, arbitrary digest or an old PID.
// Private-local-cache corruption model is unchanged (SHA is not a MAC).
namespace r184_lease {
constexpr uint64_t max_bytes=536870912;
constexpr uint32_t magic=0x52313834;
using Bytes=std::vector<uint8_t>;
using Digest=std::array<uint8_t,32>;
inline void check(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
struct Handle {
 HANDLE h=nullptr;
 explicit Handle(HANDLE v=nullptr):h(v){}
 ~Handle(){if(h && h!=INVALID_HANDLE_VALUE)CloseHandle(h);}
 Handle(const Handle&)=delete;Handle& operator=(const Handle&)=delete;
 Handle(Handle&& other)noexcept:h(other.h){other.h=nullptr;}
 Handle& operator=(Handle&& other)noexcept{if(this!=&other){if(h && h!=INVALID_HANDLE_VALUE)CloseHandle(h);h=other.h;other.h=nullptr;}return *this;}
 explicit operator bool()const{return h && h!=INVALID_HANDLE_VALUE;}
};
struct View {
 const uint8_t* data=nullptr;
 View(HANDLE map,size_t size):data(static_cast<const uint8_t*>(MapViewOfFile(map,FILE_MAP_READ,0,0,size))){check(data!=nullptr,"lease map");}
 ~View(){if(data)UnmapViewOfFile(data);}
 View(const View&)=delete;View& operator=(const View&)=delete;
};
inline Digest sha(const uint8_t* data,size_t size){
 check(size<=ULONG_MAX,"lease SHA input bound");
 struct Alg{BCRYPT_ALG_HANDLE h=nullptr;~Alg(){if(h)BCryptCloseAlgorithmProvider(h,0);}}alg;
 check(BCryptOpenAlgorithmProvider(&alg.h,BCRYPT_SHA256_ALGORITHM,nullptr,0)>=0,"lease SHA provider");
 struct Hash{BCRYPT_HASH_HANDLE h=nullptr;~Hash(){if(h)BCryptDestroyHash(h);}}hash;
 check(BCryptCreateHash(alg.h,&hash.h,nullptr,0,nullptr,0,0)>=0,"lease SHA object");
 check(BCryptHashData(hash.h,const_cast<PUCHAR>(data),ULONG(size),0)>=0,"lease SHA data");
 Digest result{};check(BCryptFinishHash(hash.h,result.data(),ULONG(result.size()),0)>=0,"lease SHA finish");return result;
}
inline std::string hex(const Digest& digest){std::string s;for(auto b:digest){s+="0123456789abcdef"[b>>4];s+="0123456789abcdef"[b&15];}return s;}
inline uint64_t file_size(HANDLE file){LARGE_INTEGER size{};check(GetFileSizeEx(file,&size)!=0 && size.QuadPart>=32 && uint64_t(size.QuadPart)<=max_bytes,"lease file length");return uint64_t(size.QuadPart);}
inline BY_HANDLE_FILE_INFORMATION info(HANDLE file){BY_HANDLE_FILE_INFORMATION i{};check(GetFileInformationByHandle(file,&i)!=0,"lease file identity");return i;}
inline bool same_file(const BY_HANDLE_FILE_INFORMATION& a,const BY_HANDLE_FILE_INFORMATION& b){return a.dwVolumeSerialNumber==b.dwVolumeSerialNumber && a.nFileIndexHigh==b.nFileIndexHigh && a.nFileIndexLow==b.nFileIndexLow;}
inline bool alive(HANDLE process){return WaitForSingleObject(process,0)==WAIT_TIMEOUT;}
inline std::string process_image_hash(HANDLE process){
 std::wstring path(32768,L'\0');DWORD count=DWORD(path.size());check(QueryFullProcessImageNameW(process,0,path.data(),&count)!=0,"lease owner image");path.resize(count);
 Handle file(CreateFileW(path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr));check(bool(file),"lease owner executable pin");
 const auto size=file_size(file.h);check(size<16777216,"lease owner executable bound");
 Handle map(CreateFileMappingW(file.h,nullptr,PAGE_READONLY,0,0,nullptr));check(bool(map),"lease owner executable map");View view(map.h,size);return hex(sha(view.data,size));
}
// All pipe IO is bounded; timeout falls back to the original verified file path.
inline bool io(HANDLE pipe,void* data,DWORD size,bool write,DWORD timeout_ms=750){
 Handle event(CreateEventW(nullptr,TRUE,FALSE,nullptr));if(!event)return false;
 OVERLAPPED overlap{};overlap.hEvent=event.h;DWORD done=0;
 const BOOL ok=write?WriteFile(pipe,data,size,&done,&overlap):ReadFile(pipe,data,size,&done,&overlap);
 if(!ok){if(GetLastError()!=ERROR_IO_PENDING)return false;
  if(WaitForSingleObject(event.h,timeout_ms)!=WAIT_OBJECT_0){CancelIoEx(pipe,&overlap);GetOverlappedResult(pipe,&overlap,&done,TRUE);return false;}
  if(!GetOverlappedResult(pipe,&overlap,&done,FALSE))return false;
 }
 return done==size;
}
#pragma pack(push,1)
struct Packet {uint32_t signature=magic,version=2;uint64_t file=0,mapping=0,size=0;uint32_t owner=0,reserved=0;};
#pragma pack(pop)
struct ByteView {
 const uint8_t* data_=nullptr; size_t size_=0;
 const uint8_t* data()const{return data_;}
 size_t size()const{return size_;}
 bool empty()const{return size_==0;}
 const uint8_t* begin()const{return data_;}
 const uint8_t* end()const{return data_+size_;}
 uint8_t at(size_t n)const{if(n>=size_)throw std::out_of_range("view byte bound");return data_[n];}
 uint8_t operator[](size_t n)const{return at(n);}
};
class VerifiedBytes;
inline std::optional<VerifiedBytes> acquire(const std::filesystem::path&);
class VerifiedBytes {
 Handle file_,mapping_;
 std::unique_ptr<View> view_;
 ByteView bytes_;
 explicit VerifiedBytes(Handle file,Handle mapping,std::unique_ptr<View> view,size_t size)
 :file_(std::move(file)),mapping_(std::move(mapping)),view_(std::move(view)),bytes_{view_->data,size}{}
 friend std::optional<VerifiedBytes> acquire(const std::filesystem::path&);
public:
 VerifiedBytes(const VerifiedBytes&)=delete;VerifiedBytes& operator=(const VerifiedBytes&)=delete;
 VerifiedBytes(VerifiedBytes&&)=default;VerifiedBytes& operator=(VerifiedBytes&&)=default;
 const ByteView& bytes()const{return bytes_;}
};
inline std::optional<VerifiedBytes> acquire(const std::filesystem::path& path){
 const char* pipe_env=std::getenv("ASTRA_VERIFIED_CACHE_PIPE");
 const char* sha_env=std::getenv("ASTRA_VERIFIED_CACHE_OWNER_SHA256");
 if(!pipe_env || !sha_env || std::string(sha_env).size()!=64)return std::nullopt;
 try{
  std::string pipe_name(pipe_env);check(pipe_name.rfind("\\\\.\\pipe\\R184-",0)==0 && pipe_name.size()<200,"lease pipe scope");
  Handle pipe(CreateFileA(pipe_name.c_str(),GENERIC_READ|GENERIC_WRITE,0,nullptr,OPEN_EXISTING,FILE_FLAG_OVERLAPPED,nullptr));check(bool(pipe),"lease unavailable");
  ULONG owner_pid=0;check(GetNamedPipeServerProcessId(pipe.h,&owner_pid)!=0,"lease owner PID");
  Handle owner(OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION|SYNCHRONIZE,FALSE,owner_pid));check(bool(owner)&&alive(owner.h),"lease owner alive");
  std::string expected(sha_env);std::transform(expected.begin(),expected.end(),expected.begin(),[](unsigned char c){return char(std::tolower(c));});
  check(process_image_hash(owner.h)==expected,"lease owner binary hash");
  // This independently opened read handle overlaps the owner's deny-write lease.
  Handle expected_file(CreateFileW(path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr));check(bool(expected_file),"lease requested path pin");
  const auto expected_info=info(expected_file.h);const auto expected_size=file_size(expected_file.h);
  uint32_t request=magic;check(io(pipe.h,&request,sizeof(request),true),"lease request");
  Packet packet{};check(io(pipe.h,&packet,sizeof(packet),false),"lease response");
  check(packet.signature==magic && packet.version==2 && packet.reserved==0x42415331 && packet.owner==owner_pid && packet.size==expected_size && packet.size<=max_bytes,"lease response identity");
  Handle leased_file(reinterpret_cast<HANDLE>(uintptr_t(packet.file))),mapping(reinterpret_cast<HANDLE>(uintptr_t(packet.mapping)));
  check(bool(leased_file)&&bool(mapping)&&same_file(expected_info,info(leased_file.h)) && file_size(leased_file.h)==expected_size,"lease exact file object");
  check(alive(owner.h),"lease owner disappeared before transfer");
  auto view=std::make_unique<View>(mapping.h,expected_size);
  // Keep the exact deny-write handle AND read-only mapped view for the lifetime
  // of this opaque capability. Owner has validated the whole basis once.
  // No intermediate 406 MB private byte copy is created.
  uint32_t acknowledgment=magic;check(io(pipe.h,&acknowledgment,sizeof(acknowledgment),true),"lease acknowledgment");
  return VerifiedBytes(std::move(leased_file),std::move(mapping),std::move(view),expected_size);
 }catch(const std::exception&){return std::nullopt;}
}
} // namespace r184_lease

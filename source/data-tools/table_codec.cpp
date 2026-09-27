// Lossless transport only: miners consume the original, SHA-256 checked bytes.
#include <zlib.h>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs=std::filesystem;
constexpr uint32_t block_size=262144;
constexpr uint64_t maximum_file=512ULL*1024*1024;
static void require(bool b,const char* message){if(!b)throw std::runtime_error(message);}
static uint32_t get32(const unsigned char* p){return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);}
static void put32(unsigned char* p,uint32_t x){for(int i=0;i<4;++i)p[i]=static_cast<unsigned char>(x>>(8*i));}
static void write32(std::ostream& s,uint32_t v){unsigned char b[4];put32(b,v);s.write(reinterpret_cast<char*>(b),4);}
static uint32_t read32(std::istream& s){unsigned char b[4];require(bool(s.read(reinterpret_cast<char*>(b),4)),"Truncated integer");return get32(b);}
static void delta(std::vector<unsigned char>& b,bool undo){uint32_t prev=0;for(size_t i=0;i+4<=b.size();i+=4){uint32_t x=get32(b.data()+i);uint32_t v=undo?prev+x:x-prev;put32(b.data()+i,v);prev=undo?v:x;}}
static void convert(bool pack,const fs::path& input,const fs::path& output){
 require(!fs::exists(output),"Output exists");require(fs::is_regular_file(input),"Input is not a file");
 std::ifstream in(input,std::ios::binary);require(bool(in),"Input open failed");
 std::ofstream out(output,std::ios::binary);require(bool(out),"Output open failed");
 try {
  uint64_t total=0;
  if(pack){total=fs::file_size(input);require(total<=maximum_file,"Input exceeds file limit");out.write("HZNDC001",8);write32(out,uint32_t(total));write32(out,uint32_t(total>>32));write32(out,block_size);}
  else {std::array<char,8> magic{};require(bool(in.read(magic.data(),8))&&std::string(magic.data(),8)=="HZNDC001","Invalid format");total=read32(in);total|=uint64_t(read32(in))<<32;require(total<=maximum_file,"Output exceeds file limit");require(read32(in)==block_size,"Invalid block size");}
  uint64_t done=0;std::vector<unsigned char> raw,compressed;
  while(done<total){
   uint32_t n=uint32_t(std::min<uint64_t>(block_size,total-done));
   raw.resize(n);
   if(pack){
    require(bool(in.read(reinterpret_cast<char*>(raw.data()),n)),"Input changed or truncated");delta(raw,false);
    uLongf size=compressBound(n);compressed.resize(size);
    require(compress2(compressed.data(),&size,raw.data(),n,6)==Z_OK,"Compression failed");
    write32(out,n);write32(out,uint32_t(size));out.write(reinterpret_cast<char*>(compressed.data()),size);
   }else{
    require(read32(in)==n,"Invalid raw block length");uint32_t size=read32(in);require(size>0&&size<=compressBound(n),"Invalid packed block length");compressed.resize(size);
    require(bool(in.read(reinterpret_cast<char*>(compressed.data()),size)),"Truncated block");
    uLongf decoded=n;uLong consumed=size;
    require(uncompress2(raw.data(),&decoded,compressed.data(),&consumed)==Z_OK&&decoded==n&&consumed==size,"Corrupt block");
    delta(raw,true);out.write(reinterpret_cast<char*>(raw.data()),raw.size());
   }
   require(bool(out),"Output write failed");done+=n;
  }
  require(in.peek()==std::char_traits<char>::eof(),"Unexpected trailing input");out.flush();require(bool(out),"Output flush failed");out.close();require(bool(out),"Output close failed");
 }catch(...){out.close();std::error_code ignored;fs::remove(output,ignored);throw;}
}
#ifdef _WIN32
int wmain(int argc,wchar_t** argv){
 try{require(argc==4,"Usage: table-codec pack|unpack INPUT OUTPUT");std::wstring mode=argv[1];require(mode==L"pack"||mode==L"unpack","Invalid mode");convert(mode==L"pack",fs::path(argv[2]),fs::path(argv[3]));return 0;}
#else
int main(int argc,char** argv){
 try{require(argc==4,"Usage: table-codec pack|unpack INPUT OUTPUT");std::string mode=argv[1];require(mode=="pack"||mode=="unpack","Invalid mode");convert(mode=="pack",fs::path(argv[2]),fs::path(argv[3]));return 0;}
#endif
 catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

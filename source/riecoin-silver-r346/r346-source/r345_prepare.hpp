#pragma once
#include <cuda_runtime.h>
#include <future>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
namespace r345 {
using Clock=std::chrono::steady_clock;
inline double ms(Clock::time_point begin){return std::chrono::duration<double,std::milli>(Clock::now()-begin).count();}
inline bool requested(){const char* v=std::getenv("ASTRA_R345_OVERLAP");if(!v||!*v||!std::strcmp(v,"0"))return false;if(!std::strcmp(v,"1"))return true;throw std::runtime_error("R345 mode must be0or1");}
inline thread_local std::ostringstream* capture=nullptr;
inline std::ostream& out(){return capture?static_cast<std::ostream&>(*capture):std::cout;}
struct Capture {std::ostringstream text;Capture(){if(capture)throw std::runtime_error("R345 nested ownership");capture=&text;}~Capture(){capture=nullptr;}};
struct Receipt {std::string lines;double secondary_ms=0;};
class Preparation {
 std::future<Receipt> future_;Clock::time_point start_{};
 public:
 Preparation()=default;Preparation(const Preparation&)=delete;Preparation& operator=(const Preparation&)=delete;
 ~Preparation(){if(future_.valid())future_.wait();} // Fail-closed unwinding, never detach CUDA work.
 bool active()const{return future_.valid();}
 template<class Fn>void start(int device,Fn fn){
  if(active())throw std::runtime_error("R345 duplicate preparation");start_=Clock::now();
  future_=std::async(std::launch::async,[device,fn]{
   if(cudaSetDevice(device)!=cudaSuccess)throw std::runtime_error("R345 worker device binding");
   Capture log;const auto begin=Clock::now();
   try{fn();if(cudaStreamSynchronize(cudaStreamPerThread)!=cudaSuccess)throw std::runtime_error("R345 readiness fence");}
   catch(...){cudaStreamSynchronize(cudaStreamPerThread);throw;}
   return Receipt{log.text.str(),ms(begin)};
  });
 }
 void join(){
  if(!active())throw std::runtime_error("R345 absent preparation");const auto join_begin=Clock::now();
  const auto r=future_.get();const double wait=ms(join_begin);
  std::cout<<r.lines<<"phase=r345_readiness mode=overlap secondary_ms="<<r.secondary_ms
   <<" branch_wall_ms="<<ms(start_)<<" residual_join_ms="<<wait<<" errors=0"<<std::endl;
 }
};
}

#include "r259_executor_api.hpp"
#include <thread>
#include <stdexcept>
#include <iostream>
int main(){
  const auto caller=std::this_thread::get_id(); std::thread::id worker;
  unsigned value=0, failures=0;
  for(unsigned i=0;i<10000;++i){
    r259_host_call([&]{
      if(std::this_thread::get_id()==caller)throw std::runtime_error("not independent");
      if(i==0)worker=std::this_thread::get_id();
      if(worker!=std::this_thread::get_id() || value!=i)throw std::runtime_error("handoff/order");
      ++value;
    });
    if(value!=i+1)throw std::runtime_error("enqueue not fenced");
    if(i%101==0){try{r259_host_call([]{throw std::runtime_error("injected");});}
      catch(const std::runtime_error& e){if(std::string(e.what())!="injected")throw;++failures;}}
  }
  if(value!=10000 || failures!=100)throw std::runtime_error("counts");
  r259_host_stop();
  std::cout<<"R259_EXECUTOR_PASS handoffs="<<value<<" transported_failures="<<failures<<"\n";
}

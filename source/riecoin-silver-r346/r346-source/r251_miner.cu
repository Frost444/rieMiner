#ifndef _WIN32
#include <cstdlib>
#include <filesystem>
inline int _putenv_s(const char* key,const char* value){return *value?setenv(key,value,1):unsetenv(key);}
#endif
#include "r345_prepare.hpp"
// RC3:OPERATOR-DRAIN — checked only at the existing synchronized bundle boundary.
// The backend owns this unique run directory and requests an exact consumed prefix.
inline bool horizon_operator_drain_requested();
#define RIECOIN_RLOW_RESIDENT_BUNDLE_STOP_REQUESTED() horizon_operator_drain_requested()
#define ASTRA_R345_PREPARE
#include "r322-coverage.hpp"
#include "r259_executor_api.hpp"
#include "r294_dispatch.hpp"
#include "r297_sos_launch.hpp"
#include "r292_member_bounds.hpp"
extern "C" cudaError_t riecoin_a99b_launch_oracle(const void*,std::uint8_t*,std::uint32_t*,std::uint32_t,cudaStream_t);
// R298 retains R294 Q0 shifts and tests shifted exact members plus larger Q1 batches.
// The inherited route proves the whole reservation and keeps exact fallback.
namespace r298 {inline bool batch_enabled=true;inline bool small_blocks=true;}
inline cudaError_t r298_member_dispatch(const void* candidates,std::uint8_t* verdicts,std::uint32_t* residues,std::uint32_t count,cudaStream_t stream){
 return r298::small_blocks?riecoin_r297_launch_oracle(candidates,verdicts,residues,count,stream):r294_member_dispatch(candidates,verdicts,residues,count,stream);
}
namespace r283 {static bool enabled=true;static bool qualified=false;}
#define ASTRA_R249_DENSE_Q0
#define RIECOIN_RLOW_SCIENCE_Q9
// Production binding of the exact R212 admission, independently qualified by R213.
// No test environment may silently disable admission or inject a corrupted bit.
#include "r217_admission.cuh"
#define ASTRA_R201_INITIALIZE r217::init
#define ASTRA_R201_APPLY r217::apply
static void r249_cleanup();
#define ASTRA_R201_CLEANUP r249_cleanup
#define ASTRA_R211_REFERENCE_DONE r217::reference_done
#define ASTRA_R209_FRONTIER_APPLY r217::apply_async
#define ASTRA_R272_PHASE_APPLY r217::apply_phase
#define ASTRA_R203_ENTRY r249_inherited_entry
#include "r203_miner.cu"
#ifdef RIECOIN_RLOW_OVERLAP_PIPELINE
#error R249 requires exclusive producer/PRP sequencing
#endif
static_assert(rlow_queue::kMacros<=64u&&rlow_queue::kMacroCapacity<=4096u,"R249 bounded dense storage");
static void r249_cleanup(){uint32_t error=0;r200::ck(cudaMemcpyFromSymbol(&error,riecoin_lazy_base2_window::r249_error,4),"R249 error receipt");r217::cleanup();if(error)throw std::runtime_error("R249 dense worklist overflow");}


int main(int argc,char**argv) {try {
 std::vector<std::string> args;bool limit=false,profile=false,baseline=false,closed=false,style_given=false,audit=false,r283_given=false;unsigned style=1;bool batch_given=false;bool block_given=false;bool tail_given=false;bool coverage_given=false;
 for(int i=0;i<argc;++i){std::string a(argv[i]);
  if(a=="--r322-coverage"){
   if(++i>=argc || (std::strcmp(argv[i],"0")&&std::strcmp(argv[i],"1")))throw std::runtime_error("R322 coverage requires 0 or 1");
   r322::extended=argv[i][0]=='1';coverage_given=true;continue;
  }
  if(a=="--r298-small-block-mode"){
   if(++i>=argc || (std::strcmp(argv[i],"0")&&std::strcmp(argv[i],"1")))throw std::runtime_error("R298 block mode requires0or1");
   r298::small_blocks=argv[i][0]=='1';block_given=true;continue;
  }
  if(a=="--r298-tail-mode"){
   if(++i>=argc || (std::strcmp(argv[i],"0")&&std::strcmp(argv[i],"1")))throw std::runtime_error("R298 tail mode requires0or1");
   r292::enabled=argv[i][0]=='1';tail_given=true;continue;
  }
  if(a=="--r298-batch-mode"){
   if(++i>=argc || (std::strcmp(argv[i],"0")&&std::strcmp(argv[i],"1")))throw std::runtime_error("R298 batch mode requires0or1");
   r298::batch_enabled=argv[i][0]=='1';batch_given=true;continue;
  }
  if(a=="--r298-q0-mode"){
   if(++i>=argc || (std::strcmp(argv[i],"0")&&std::strcmp(argv[i],"1")))throw std::runtime_error("R283 q0 mode requires 0 or 1");
   r283::enabled=argv[i][0]=='1';r283_given=true;continue;
  }
  if(a=="--r251-profile-self-test" || a=="--r273-profile-self-test" || a=="--r298-profile-self-test"){profile=true;continue;}
  if(a=="--r273-closed-audit"){audit=true;continue;}
  if(a=="--r273-style"){
   if(++i>=argc || (std::strcmp(argv[i],"0")&&std::strcmp(argv[i],"1")&&std::strcmp(argv[i],"2")))
     throw std::runtime_error("R273 style requires 0,1,2");
   style=unsigned(argv[i][0]-'0');style_given=true;continue;
  }
  if(a=="--r251-closed-baseline"){baseline=true;continue;}
  if(a=="--prime-limit")limit=true;
  if(a=="--production-session-seconds")closed=true;
  args.push_back(a);
 }
 // R251: production binding of the exact R249 dense-index operator.
 if(r345::requested()&&!closed)throw std::runtime_error("R345 preparation overlap requires a closed session until qualified");
 if((coverage_given||r283_given||tail_given||batch_given||block_given)&&!closed){std::cerr<<"R283 SoS requires an explicit closed session\n";return 81;}
 if(!closed && (std::getenv("ASTRA_R259_BITMAP_AUDIT") || std::getenv("ASTRA_R259_INJECT_GENERATION")))
   throw std::runtime_error("R259 audit/fault options require a closed session");
 if(!closed && std::getenv("ASTRA_R259_PREFETCH") && std::strcmp(std::getenv("ASTRA_R259_PREFETCH"),"0"))
   throw std::runtime_error("R259 laboratory prefetch requires a closed session");
 if((baseline||style_given||audit)&&!closed){std::cerr<<"R273 reference/style/audit toggle requires an explicit closed session\n";return 81;}
 if(!limit){args.push_back("--prime-limit");args.push_back("1000000000");}
 std::vector<char*> raw;for(auto& a:args)raw.push_back(a.data());raw.push_back(nullptr);
 const Options o=parse_options(int(raw.size()-1),raw.data());
 // Nonqualified network geometry keeps the complete inherited R189 path.
 const bool supported=o.pattern==0 && o.primorial_offset==114023297140211ull &&
  o.prime_limit==1000000000u && o.factor_origin<=UINT64_MAX-r200::SPAN;
 const bool use_admission=supported;
 const bool wide=use_admission&&!baseline&&o.primorial_number>=109&&o.primorial_number<=114&&o.factor_origin<=UINT64_MAX-r208::high;
 // Qualification is separate from the requested mode: unsupported geometry
 // must retain the complete inherited arithmetic, even with default-on mode.
 r283::qualified=wide&&o.target_bits>=r322::minimum_bits()&&o.target_bits<=1184;
 // Only the qualified arithmetic branches and complete eight-slice geometry
 // use prefetch. Everything else retains the identical R251 serial fallback.
 const bool prefetch=wide&&o.target_bits>=r322::minimum_bits()&&o.target_bits<=1184&&style!=0;
 rlow_batch_producer::r273_style=style==2?2:1;
 _putenv_s("ASTRA_R259_PREFETCH",prefetch?"1":"0");
 _putenv_s("ASTRA_R270_QUANTUM",prefetch?"1":"0");
 _putenv_s("ASTRA_R259_INJECT_GENERATION","0");
 if(!audit)_putenv_s("ASTRA_R259_BITMAP_AUDIT","");
 r217::start_origin=o.factor_origin;r217::inject_hot_fault=false;r212::inject_frontier_fault=false;
 _putenv_s("ASTRA_R217_WIDE",wide?"1":"0");
 _putenv_s("ASTRA_R209_INJECT_HOT_FAULT","0");
 // RC3:DATA-ROOT — the package owns the data location. Never overwrite it
 // with a development-machine directory, including when an atlas falls back.
 const char* data_env=std::getenv("HORIZON_RIECOIN_DATA");
 if(use_admission&&(!data_env||!*data_env))throw std::runtime_error("Set HORIZON_RIECOIN_DATA to the prepared directory containing base/ and wide/");
 const std::filesystem::path data_root=data_env&&*data_env?data_env:"data/riecoin";
 _putenv_s("ASTRA_R209_CACHE",(data_root/"wide").string().c_str());
 _putenv_s("ASTRA_R211_LIMIT",use_admission?"4000000000":"0");
 _putenv_s("ASTRA_R211_INJECT_FRONTIER_FAULT","0");
 _putenv_s("ASTRA_R201_LIMIT","0");
 _putenv_s("ASTRA_R203_GROUPED_GUARD",use_admission?"grouped":"off");
 _putenv_s("ASTRA_R201_CACHE",(data_root/"base").string().c_str());
 std::cout<<"phase=r322_coverage extended="<<r322::extended<<" minimum_bits="<<r322::minimum_bits()<<" qualified="<<r283::qualified<<" population_unchanged=1 complete_fallback=1\n";
 if(profile){
  std::cout<<"{\"profile\":\"R298_SYMMETRIC_SOS\",\"q0_requested\":"<<r283::enabled<<",\"shift_requested\":"<<r294::enabled<<",\"tail_requested\":"<<r292::enabled<<",\"batch_requested\":"<<r298::batch_enabled<<",\"small_blocks\":"<<r298::small_blocks<<",\"prefetch\":"<<(prefetch?"true":"false")<<",\"style\":"<<style<<",\"prime_limit\":"<<o.prime_limit
   <<",\"mandatory_limit\":"<<(wide?8000000000ull:(use_admission?4000000000ull:0))
   <<",\"dense_q0_worklist\":true,\"fault_injection\":false,\"supported\":"<<(supported?"true":"false")<<"}\n";
  return 0;
 }
 std::cout<<"phase=r298_profile version=R298 small_blocks="<<r298::small_blocks<<" shift_requested="<<r294::enabled<<" batch_requested="<<r298::batch_enabled<<" tail_requested="<<r292::enabled<<" q0_requested="<<r283::enabled<<" prefetch="<<prefetch<<" style="<<style<<" mandatory_limit="<<(wide?8000000000ull:(use_admission?4000000000ull:0))
  <<" inherited_exact_fallback="<<!use_admission<<" fault_injection=0\n";
 return r249_inherited_entry(int(raw.size()-1),raw.data());
}catch(const std::exception& e){std::cerr<<"R249 configuration/execution rejected: "<<e.what()<<'\n';return 80;}}

inline bool horizon_operator_drain_requested() {
#ifdef _WIN32
 return GetFileAttributesW(L"operator-drain.request") != INVALID_FILE_ATTRIBUTES;
#else
 return std::filesystem::is_regular_file("operator-drain.request");
#endif
}

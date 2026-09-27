#define RIECOIN_RLOW_PRIME_ROOT_CACHE_CANONICAL_FIRST 114
#define RIECOIN_RLOW_PRIME_ROOT_CACHE_CANONICAL_MIN_FIRST 107
#define RIECOIN_RLOW_CANONICAL_DEVICE_TRANSFORM
#define RIECOIN_RLOW_COMPRESSED_ROOTS
#define roots_for cpu_roots_for
#include "riecoin_prime_root_cache.hpp"
#undef roots_for
#include "riecoin_gpu_root_builder.cuh"
int main() { try {
    namespace c=riecoin_prime_root_cache;
    auto primes=c::exact::primes_to(50000);
    c::Integer canonical,base;
    c::set_prefix_product(canonical.value,primes,114);
    mpz_set_ui(base.value,1);mpz_mul_2exp(base.value,base.value,1100);mpz_add_ui(base.value,base.value,12345);
    const auto original=c::build(50000,114,canonical.value);
    const std::array<std::uint32_t,7> offsets{0,2,6,8,12,18,20};
    for(unsigned first=107;first<=114;++first) {
        c::Integer requested;c::set_prefix_product(requested.value,primes,first);
        auto basis=original;c::Metrics metrics;
        c::stage_device_transform_from_canonical(basis,first,requested.value,metrics);
        auto roots=c::roots_for_device(basis,base.value,offsets,10000);
        if(roots.exact_roots_checked!=7*(primes.size()-first))throw std::runtime_error("incomplete GPU verification");
        std::cout<<"GPU canonical prefix "<<first<<" PASS\n";
        if(first==107) {
            basis.inverse_multiplier_second+=1;
            bool rejected=false;
            try { auto bad=c::roots_for_device(basis,base.value,offsets,10000); }
            catch(const std::exception&) { rejected=true; }
            if(!rejected)throw std::runtime_error("corrupted second multiplier accepted");
            std::cout<<"GPU corrupted second multiplier REJECTED\n";
        }
    }
    std::cout<<"GPU canonical bridge 107..114: PASS\n";
    return 0;
} catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; } }

#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define R211_HD __host__ __device__
#else
#define R211_HD
#endif
namespace r211_relation {
// Preconditions: p>span, root/phase<p; caller checks interval overflow.
R211_HD inline uint32_t subtract(uint32_t root,uint32_t delta,uint32_t p) {
 return root>=delta?root-delta:uint32_t(uint64_t(root)+p-delta);
}
R211_HD inline uint32_t at_origin(uint32_t root,uint64_t origin,uint32_t p) {
 return subtract(root,uint32_t(origin%p),p);
}
R211_HD inline uint32_t rebase(uint32_t phase,uint64_t represented,uint64_t wanted,uint32_t p) {
 const uint32_t absolute=uint32_t((uint64_t(phase)+represented%p)%p);
 return at_origin(absolute,wanted,p);
}
R211_HD inline uint32_t second(uint32_t root,uint32_t inverse,uint32_t p) {
 const uint64_t twice=uint64_t(inverse)*2;
 return subtract(root,uint32_t(twice>=p?twice-p:twice),p);
}
}
#undef R211_HD

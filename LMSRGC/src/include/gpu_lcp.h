#ifndef GPU_LCP_H
#define GPU_LCP_H

#include <cstdint>
using T = unsigned char;

void gpuLcpArray(T const* input_string,
                 uint32_t const* sa,
                 size_t const n, 
                 int* const output,
                 int const first = -1);

#endif //GPU_LCP_H

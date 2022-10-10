#ifndef INCLUDE_COMMON_CUH
#define INCLUDE_COMMON_CUH

#if __CUDA_ARCH__ >= 700

#define __shfl(...) __shfl_sync(__activemask(), __VA_ARGS__)
#define __shfl_up(...) __shfl_up_sync(__activemask(), __VA_ARGS__)
#define __shfl_down(...) __shfl_down_sync(__activemask(), __VA_ARGS__)
#define __shfl_xor(...) __shfl_xor_sync(__activemask(), __VA_ARGS__)
#define __all(...) __all_sync(__activemask(), __VA_ARGS__)
#define __any(...) __any_sync(__activemask(), __VA_ARGS__)
#define __ballot(...) __ballot_sync(__activemask(), __VA_ARGS__)

#endif

#endif

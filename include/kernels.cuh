#ifndef KERNELS_CUH
#define KERNELS_CUH

__global__ void histogram_global_baseline(const int* d_input, unsigned int* d_histogram, int n);
__global__ void histogram_shared_privatized(const int* d_input, unsigned int* d_histogram, int n);
__global__ void histogram_sort_reduce(const int* d_input, unsigned int* d_histogram, int n);
__global__ void histogram_warp_register_aggregation(const int* d_input, unsigned int* d_histogram, int n);
__global__ void histogram_warp_shared_hybrid(
    const int* d_input,
    unsigned int* d_histogram,
    int n
);
__global__ void histogram_per_warp_shared(
    const int* d_input,
    unsigned int* d_histogram,
    int n
);

#endif
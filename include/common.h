#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Expanded array footprint to ensure hardware pipelines reach steady-state saturation
const int N = 32'000'000;       
const int BINS = 256;           
const int THREADS_PER_BLOCK = 256;

#endif
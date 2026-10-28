
cmake_minimum_required(VERSION 3.18)

project(GPUHistogramContrast LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

include_directories(include)

add_executable(benchmark_histogram
    src/main.cu
    src/kernels.cu
)

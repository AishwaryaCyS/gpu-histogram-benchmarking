
#include "../include/common.h"
#include "../include/kernels.cuh"

// Strategy 1: Global memory atomics
__global__ void histogram_global_baseline(
    const int* d_input,
    unsigned int* d_histogram,
    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        atomicAdd(&d_histogram[d_input[tid]], 1);
    }
}

// Strategy 2: Shared-memory privatization
__global__ void histogram_shared_privatized(
    const int* d_input,
    unsigned int* d_histogram,
    int n)
{
    __shared__ unsigned int s_hist[BINS];

    int local_tid = threadIdx.x;

    for (int i = local_tid; i < BINS; i += blockDim.x) {
        s_hist[i] = 0;
    }

    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        atomicAdd(&s_hist[d_input[tid]], 1);
    }

    __syncthreads();

    for (int i = local_tid; i < BINS; i += blockDim.x) {
        if (s_hist[i] > 0) {
            atomicAdd(&d_histogram[i], s_hist[i]);
        }
    }
}

// Strategy 3: Sort-and-reduce.
// Input MUST be sorted before this kernel is launched.
__global__ void histogram_sort_reduce(
    const int* d_input,
    unsigned int* d_histogram,
    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        int current_val = d_input[tid];

        if (tid == 0 || d_input[tid - 1] != current_val) {
            int count = 1;

            while (tid + count < n &&
                   d_input[tid + count] == current_val) {
                count++;
            }

            atomicAdd(&d_histogram[current_val], count);
        }
    }
}

// Strategy 4: Warp-level register aggregation
__global__ void histogram_warp_register_aggregation(
    const int* d_input,
    unsigned int* d_histogram,
    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        int my_value = d_input[tid];

        unsigned int match_mask =
            __match_any_sync(0xFFFFFFFF, my_value);

        int leader_lane = __ffs(match_mask) - 1;
        int match_count = __popc(match_mask);
        int lane_id = threadIdx.x % 32;

        if (lane_id == leader_lane) {
            atomicAdd(&d_histogram[my_value], match_count);
        }
    }
}
// APPROACH 5: Warp aggregation + shared-memory histogram
__global__ void histogram_warp_shared_hybrid(
    const int* d_input,
    unsigned int* d_histogram,
    int n)
{
    __shared__ unsigned int s_hist[BINS];

    int lane = threadIdx.x % 32;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize the block-local histogram.
    for (int i = threadIdx.x; i < BINS; i += blockDim.x) {
        s_hist[i] = 0;
    }

    __syncthreads();

    if (tid < n) {
        int value = d_input[tid];

        // Use only the lanes currently active in this warp.
        unsigned int active_mask = __activemask();
        unsigned int match_mask =
            __match_any_sync(active_mask, value);

        int leader_lane = __ffs(match_mask) - 1;
        int match_count = __popc(match_mask);

        // One shared-memory atomic per group of equal values.
        if (lane == leader_lane) {
            atomicAdd(&s_hist[value], match_count);
        }
    }

    __syncthreads();

    // Merge the block-local histogram into global memory.
    for (int i = threadIdx.x; i < BINS; i += blockDim.x) {
        if (s_hist[i] > 0) {
            atomicAdd(&d_histogram[i], s_hist[i]);
        }
    }
}

__global__ void histogram_per_warp_shared(
    const int* d_input,
    unsigned int* d_histogram,
    int n)
{
    // One private histogram for each warp in the block.
    __shared__ unsigned int warp_hist[32][BINS];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int warp_count = blockDim.x / 32;

    // Initialize all per-warp histograms.
    for (int i = threadIdx.x;
         i < warp_count * BINS;
         i += blockDim.x)
    {
        warp_hist[i / BINS][i % BINS] = 0;
    }

    __syncthreads();

    // Each thread updates its own warp's histogram.
    if (tid < n)
    {
        int value = d_input[tid];
        atomicAdd(&warp_hist[warp_id][value], 1);
    }

    __syncthreads();

    // Merge the warp histograms and update the global histogram.
    for (int bin = threadIdx.x;
         bin < BINS;
         bin += blockDim.x)
    {
        unsigned int total = 0;

        for (int w = 0; w < warp_count; w++)
        {
            total += warp_hist[w][bin];
        }

        if (total > 0)
        {
            atomicAdd(&d_histogram[bin], total);
        }
    }
}


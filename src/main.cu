
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cstdlib>

#include "../include/common.h"
#include "../include/kernels.cuh"

const int TIMING_RUNS = 10;

void generate_data(int* host_array, int size,
                   const std::string& dist_type)
{
    std::default_random_engine generator(1234);

    if (dist_type == "skewed") {
        std::normal_distribution<double> dist(
            BINS / 2.0, BINS / 128.0);

        for (int i = 0; i < size; i++) {
            int val = static_cast<int>(dist(generator));
            host_array[i] = std::max(0, std::min(BINS - 1, val));
        }
    } else {
        std::uniform_int_distribution<int> dist(0, BINS - 1);

        for (int i = 0; i < size; i++) {
            host_array[i] = dist(generator);
        }
    }
}

void launch_kernel(int choice, int* d_input,
                   unsigned int* d_histogram,
                   int blocks, int block_size)
{
    switch (choice) {
    case 1:
        histogram_global_baseline<<<blocks, block_size>>>(
            d_input, d_histogram, N);
        break;

    case 2:
        histogram_shared_privatized<<<blocks, block_size>>>(
            d_input, d_histogram, N);
        break;

    case 3:
        histogram_sort_reduce<<<blocks, block_size>>>(
            d_input, d_histogram, N);
        break;

    case 4:
        histogram_warp_register_aggregation<<<blocks, block_size>>>(
            d_input, d_histogram, N);
        break;

    case 5:
        histogram_warp_shared_hybrid<<<blocks, block_size>>>(
            d_input, d_histogram, N);
        break;
    case 6:
        histogram_per_warp_shared<<<blocks, block_size>>>(
            d_input, d_histogram, N
        );
        break;

    default:
        std::cerr << "Invalid strategy.\n";
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cerr
            << "Usage: " << argv[0]
            << " <uniform|skewed> <1|2|3|4|5>"
            << " <block_size: 128|256|512>\n";
        return 1;
    }

    const std::string dist_type = argv[1];
    int choice = 0;
    int block_size = 0;

    try {
        choice = std::stoi(argv[2]);
        block_size = std::stoi(argv[3]);
    } catch (...) {
        std::cerr
            << "Strategy and block size must be integers.\n";
        return 1;
    }

    if (dist_type != "uniform" && dist_type != "skewed") {
        std::cerr
            << "Distribution must be uniform or skewed.\n";
        return 1;
    }

    if (choice < 1 || choice > 6) {
        std::cerr
            << "Strategy must be between 1 and 6.\n";
        return 1;
    }

    if (block_size != 128 && block_size != 256 &&
        block_size != 512) {
        std::cerr
            << "Block size must be 128, 256, or 512.\n";
        return 1;
    }

    using Clock = std::chrono::steady_clock;
    const auto pipeline_start = Clock::now();

    const int blocks = (N + block_size - 1) / block_size;

    std::vector<int> h_input(N);
    generate_data(h_input.data(), N, dist_type);

    // Strategy 3 requires sorted input.
    if (choice == 3) {
        std::sort(h_input.begin(), h_input.end());
    }

    int* d_input = nullptr;
    unsigned int* d_histogram = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_input),
        static_cast<size_t>(N) * sizeof(int)));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_histogram),
        BINS * sizeof(unsigned int)));

    CUDA_CHECK(cudaMemcpy(
        d_input, h_input.data(),
        static_cast<size_t>(N) * sizeof(int),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(
        d_histogram, 0,
        BINS * sizeof(unsigned int)));

    // First execution: correctness check and pipeline timing.
    launch_kernel(
        choice, d_input, d_histogram, blocks, block_size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned int> h_histogram(BINS);

    CUDA_CHECK(cudaMemcpy(
        h_histogram.data(), d_histogram,
        BINS * sizeof(unsigned int),
        cudaMemcpyDeviceToHost));

    const auto pipeline_end = Clock::now();

    const double pipeline_ms =
        std::chrono::duration<double, std::milli>(
            pipeline_end - pipeline_start).count();

    unsigned long long total_count = 0;

    for (int i = 0; i < BINS; i++) {
        total_count += h_histogram[i];
    }

    const bool correct =
        total_count == static_cast<unsigned long long>(N);

    // Warm-up execution before timing.
    CUDA_CHECK(cudaMemset(
        d_histogram, 0,
        BINS * sizeof(unsigned int)));

    launch_kernel(
        choice, d_input, d_histogram, blocks, block_size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start_event, stop_event;

    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    float total_kernel_ms = 0.0f;

    for (int i = 0; i < TIMING_RUNS; i++) {
        // Reset the histogram outside the timed interval.
        CUDA_CHECK(cudaMemset(
            d_histogram, 0,
            BINS * sizeof(unsigned int)));

        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start_event));

        launch_kernel(
            choice, d_input, d_histogram,
            blocks, block_size);

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));

        float elapsed_ms = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &elapsed_ms, start_event, stop_event));

        total_kernel_ms += elapsed_ms;
    }

    const double average_kernel_ms =
        total_kernel_ms / TIMING_RUNS;

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "Distribution: " << dist_type
              << ", Strategy: " << choice
              << ", Block size: " << block_size << '\n';

    std::cout << "Histogram total: " << total_count
              << " (expected " << N << ")\n";

    std::cout << "Average kernel time: "
              << average_kernel_ms << " ms\n";

    std::cout << "Pipeline setup + first execution: "
              << pipeline_ms << " ms\n";

    // Machine-readable output for run_benchmarks.py.
    std::cout << "RESULT,"
              << dist_type << ','
              << choice << ','
              << block_size << ','
              << average_kernel_ms << ','
              << pipeline_ms << ','
              << total_count << ','
              << (correct ? 1 : 0) << '\n';

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_histogram));

    return correct ? 0 : 2;
}

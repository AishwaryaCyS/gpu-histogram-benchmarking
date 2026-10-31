# GPU Histogram Benchmarking Using CUDA
## Cross-Platform Performance and Profiling Report

## 1. Objective

This project evaluates six CUDA histogram implementations and studies
how their performance changes with input distribution, CUDA block size,
and execution environment.

The experiment uses an NVIDIA GeForce RTX 3050 Laptop GPU with 6 GB
of reported GPU memory. The two tested environments are native Windows
and Ubuntu running through WSL2 on the same physical machine.

## 2. Problem and Dataset

The input contains 32,000,000 integer values and the histogram contains
256 bins. Two input distributions are evaluated:

- Uniform: values are distributed approximately evenly.
- Skewed: values are concentrated in a smaller portion of the range.

Correctness is checked by verifying that the sum of the histogram
counts equals 32,000,000.

## 3. Implemented Strategies

1. Global atomic baseline.
2. Shared-memory privatization.
3. Sort-and-reduce.
4. Warp-level register aggregation.
5. Warp aggregation combined with a shared-memory histogram.
6. Per-warp shared-memory histograms.

## 4. Experimental Methodology

The benchmark evaluates six strategies, two distributions, and three
block sizes: 128, 256, and 512 threads.

This produces 36 configurations. The Linux automated benchmark
completed 180 runs: five repetitions per configuration, with all
180 runs reported successful.

Kernel execution time and pipeline setup/first-execution time are
recorded separately. The latter may include allocation, data transfer,
preprocessing, and initial execution costs.

Normal benchmark measurements are used for performance comparisons.
Nsight profiler timings are not substituted for normal timings because
profiling and replay introduce overhead.

## 5. Cross-Platform Comparison

The following files contain the collected data:

- benchmark_results.csv
- benchmark_summary.csv
- benchmark_results_linux.csv
- benchmark_summary_linux.csv
- comparison_graphs/combined_platform_summary.csv
- comparison_graphs/best_block_sizes.csv

The comparison graphs are stored in comparison_graphs/.

The summary data compares average kernel time across the two
environments, distributions, strategies, and block sizes. The best
block-size table identifies the lowest measured mean kernel time
within each platform, distribution, and strategy combination.

## 6. Selected Direct Measurements

Additional unprofiled runs were performed with block size 256.

| Distribution | Strategy | Average kernel time |
|---|---|---:|
| Uniform | Strategy 2 | 1.811437 ms |
| Uniform | Strategy 5 | 3.018615 ms |
| Skewed | Strategy 2 | 0.999062 ms |
| Skewed | Strategy 5 | 1.550538 ms |

All four runs reported the expected histogram total of 32,000,000.

In these individual runs, Strategy 2 was faster than Strategy 5 for
both tested distributions. These are individual measurements; repeated
benchmark summaries should be used for stronger comparisons.

## 7. Nsight Profiling

Nsight Systems was used to inspect CUDA API activity, kernel execution,
and memory transfers.

Nsight Compute was used to inspect kernel throughput, occupancy,
scheduler behavior, and warp stall characteristics.

For the profiled uniform Strategy 2 kernel, the reported achieved
occupancy was approximately 92.4%. Scheduler statistics also indicated
that eligible warps were limited during many cycles, and the warp-state
analysis reported substantial scoreboard dependency stalls.

High occupancy alone does not guarantee high execution throughput.
The scheduler and warp-state measurements suggest that memory or
instruction dependencies deserve further investigation.

The profiler reports are stored in the reports/ directory. Profiled
execution times are not treated as ordinary benchmark timings.

## 8. Environment and Limitations

The Windows environment used CUDA Toolkit 13.4, while the WSL2
environment used CUDA Toolkit 13.1. The GPU hardware was the same.

Therefore, differences should be interpreted as observations across
these software environments, not as a comparison between independent
GPUs. Toolkit versions, GPU clock and thermal conditions, background
workloads, and WSL2 overhead may affect results.

The recorded timings describe the tested machine and configuration;
they should not be generalized to all Windows or Linux systems.

## 9. Conclusion

The project implements and evaluates six histogram strategies under
uniform and skewed input distributions and multiple block sizes.
Automated benchmark results, cross-platform graphs, and profiler
reports provide evidence for analyzing performance and potential
bottlenecks.

The measurements show that performance depends on the strategy and
workload. In the selected direct tests, shared-memory privatization
outperformed the warp-shared hybrid for both distributions at block
size 256. The full repeated-run summaries provide the broader basis
for comparing all configurations.

## 10. Reproduction

Build on Ubuntu/WSL2:

    cmake -S . -B build-linux -DCMAKE_CUDA_ARCHITECTURES=86
    cmake --build build-linux -j2

Run an example:

    ./build-linux/benchmark_histogram uniform 2 256

Run the Linux benchmark suite:

    python3 run_benchmarks_linux.py

Regenerate comparison graphs:

    python3 compare_platforms.py

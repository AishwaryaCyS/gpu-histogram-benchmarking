
import csv
import itertools
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXE = ROOT / "build-linux" / "benchmark_histogram"

DISTRIBUTIONS = ["skewed", "uniform"]
STRATEGIES = [1, 2, 3, 4, 5, 6]
BLOCK_SIZES = [128, 256, 512]
REPETITIONS = 5

RAW_CSV = ROOT / "benchmark_results_linux.csv"
SUMMARY_CSV = ROOT / "benchmark_summary_linux.csv"
FAILURE_LOG = ROOT / "benchmark_failures_linux.txt"


def run_one(distribution, strategy, block_size, repetition):
    command = [
        str(EXE),
        distribution,
        str(strategy),
        str(block_size),
    ]

    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )

    print(
        f"[{distribution:7}] strategy={strategy} "
        f"block={block_size} repetition={repetition}"
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Exit code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    result_line = next(
        (line for line in result.stdout.splitlines()
         if line.startswith("RESULT,")),
        None,
    )

    if result_line is None:
        raise RuntimeError(
            "Could not find RESULT line in program output.\n"
            + result.stdout
        )

    fields = result_line.strip().split(",")

    if len(fields) != 8:
        raise RuntimeError(f"Unexpected result format: {result_line}")

    _, dist, strat, block, kernel_ms, pipeline_ms, total, correct = fields

    if correct != "1":
        raise RuntimeError(
            f"Correctness check failed: {result_line}"
        )

    return {
        "distribution": dist,
        "strategy": int(strat),
        "threads_per_block": int(block),
        "repetition": repetition,
        "kernel_avg_ms": float(kernel_ms),
        "pipeline_first_execution_ms": float(pipeline_ms),
        "total_count": int(total),
        "correct": int(correct),
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    grouped = {}

    for row in rows:
        key = (
            row["distribution"],
            row["strategy"],
            row["threads_per_block"],
        )
        grouped.setdefault(key, []).append(row)

    summary = []

    for (dist, strategy, block), group in sorted(grouped.items()):
        item = {
            "distribution": dist,
            "strategy": strategy,
            "threads_per_block": block,
            "successful_runs": len(group),
        }

        for field in [
            "kernel_avg_ms",
            "pipeline_first_execution_ms",
        ]:
            values = [r[field] for r in group]

            item[field + "_mean"] = statistics.mean(values)
            item[field + "_median"] = statistics.median(values)
            item[field + "_stddev"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
            item[field + "_min"] = min(values)
            item[field + "_max"] = max(values)

        summary.append(item)

    return summary


def main():
    if not EXE.exists():
        print(f"Executable not found: {EXE}")
        print("Build the project first using cmake --build build")
        sys.exit(1)

    configurations = list(itertools.product(
        DISTRIBUTIONS, STRATEGIES, BLOCK_SIZES
    ))

    total_runs = len(configurations) * REPETITIONS

    print(f"Configurations: {len(configurations)}")
    print(f"Repetitions per configuration: {REPETITIONS}")
    print(f"Total runs planned: {total_runs}")
    print("Existing CSV results will be overwritten.")

    input("Press Enter to start, or Ctrl+C to cancel...")

    raw_rows = []
    failures = []
    completed = 0
    start_time = time.time()

    for dist, strategy, block in configurations:
        for repetition in range(1, REPETITIONS + 1):
            completed += 1

            try:
                row = run_one(dist, strategy, block, repetition)
                raw_rows.append(row)

                print(
                    f"  kernel={row['kernel_avg_ms']:.4f} ms, "
                    f"pipeline={row['pipeline_first_execution_ms']:.2f} ms, "
                    f"progress={completed}/{total_runs}"
                )

            except Exception as error:
                message = (
                    f"distribution={dist}, strategy={strategy}, "
                    f"block={block}, repetition={repetition}\n{error}"
                )
                failures.append(message)
                print("  FAILED:", error)

            # Save progress after every attempt, so partial runs are retained.
            write_csv(
                RAW_CSV,
                raw_rows,
                [
                    "distribution",
                    "strategy",
                    "threads_per_block",
                    "repetition",
                    "kernel_avg_ms",
                    "pipeline_first_execution_ms",
                    "total_count",
                    "correct",
                ],
            )

            FAILURE_LOG.write_text(
                "\n\n".join(failures), encoding="utf-8"
            )

    summary_rows = summarize(raw_rows)

    summary_fields = [
        "distribution",
        "strategy",
        "threads_per_block",
        "successful_runs",
    ]

    for field in [
        "kernel_avg_ms",
        "pipeline_first_execution_ms",
    ]:
        for stat in ["mean", "median", "stddev", "min", "max"]:
            summary_fields.append(field + "_" + stat)

    write_csv(SUMMARY_CSV, summary_rows, summary_fields)

    elapsed = time.time() - start_time

    print("\nBenchmark finished.")
    print(f"Successful runs: {len(raw_rows)}/{total_runs}")
    print(f"Failed attempts: {len(failures)}")
    print(f"Elapsed time: {elapsed / 60:.1f} minutes")
    print(f"Raw results: {RAW_CSV}")
    print(f"Summary: {SUMMARY_CSV}")
    print(f"Failure log: {FAILURE_LOG}")

    if failures:
        print("Review benchmark_failures_linux.txt before analysing results.")


if __name__ == "__main__":
    main()

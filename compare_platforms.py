import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent

windows = pd.read_csv(ROOT / "benchmark_summary.csv")
linux = pd.read_csv(ROOT / "benchmark_summary_linux.csv")

windows["platform"] = "Windows"
linux["platform"] = "Linux (WSL2)"

data = pd.concat([windows, linux], ignore_index=True)

out = ROOT / "comparison_graphs"
out.mkdir(exist_ok=True)

# Plot kernel time for each data distribution and strategy.
for distribution in ["uniform", "skewed"]:
    for strategy in sorted(data["strategy"].unique()):
        subset = data[
            (data["distribution"] == distribution) &
            (data["strategy"] == strategy)
        ]

        plt.figure(figsize=(8, 5))

        for platform in ["Windows", "Linux (WSL2)"]:
            part = subset[subset["platform"] == platform].sort_values(
                "threads_per_block"
            )
            plt.plot(
                part["threads_per_block"],
                part["kernel_avg_ms_mean"],
                marker="o",
                label=platform
            )

        plt.title(
            f"{distribution.title()} data — Strategy {strategy}"
        )
        plt.xlabel("Threads per block")
        plt.ylabel("Mean kernel time (ms)")
        plt.xticks([128, 256, 512])
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            out / f"{distribution}_strategy_{strategy}.png",
            dpi=160
        )
        plt.close()

# Combined comparison data, preserving all summary statistics.
data.to_csv(out / "combined_platform_summary.csv", index=False)

# Best block size for each platform, distribution and strategy.
best = (
    data.loc[
        data.groupby(["platform", "distribution", "strategy"])[
            "kernel_avg_ms_mean"
        ].idxmin()
    ]
    .sort_values(["distribution", "strategy", "platform"])
)
best.to_csv(out / "best_block_sizes.csv", index=False)

print("Graphs and comparison tables saved in:", out)
print("Generated graphs:", len(list(out.glob("*.png"))))
print("\nBest block size for each platform and strategy:")
print(best[
    ["platform", "distribution", "strategy",
     "threads_per_block", "kernel_avg_ms_mean"]
].to_string(index=False))

# benchmarks/plot_benchmarks.py

import pandas as pd
import matplotlib.pyplot as plt
import os
from benchmarks.utils import ensure_dirs

def lineplot(csv_path, x, y, hue=None, title="", ylabel=None, loglog=False, outfile=""):
    if not os.path.exists(csv_path):
        print(f"[Skipping] {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    if df.empty or x not in df.columns or y not in df.columns:
        print(f"[Skipping] Columns '{x}' or '{y}' missing or CSV empty in {csv_path}")
        return

    plt.figure(figsize=(6, 4))
    if loglog:
        plt.xscale("log")
        plt.yscale("log")

    if hue and hue in df.columns:
        for label, group in df.groupby(hue):
            plt.plot(group[x], group[y], marker="o", label=str(label))
        plt.legend()
    else:
        plt.plot(df[x], df[y], marker="o")

    plt.xlabel(x)
    plt.ylabel(ylabel if ylabel else y)
    plt.title(title)
    plt.grid(True)

    out_path = f"benchmarks/plots/{outfile}"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved plot → {out_path}")

def main():
    ensure_dirs()

    lineplot(
        csv_path="benchmarks/results/tgv_timing.csv",
        x="Grid",
        y="TimeSeconds",
        hue="Backend",
        title="TGV: JIT vs No-JIT Performance",
        ylabel="Avg Time per Step (s)",
        outfile="tgv_timing.png"
    )

    lineplot(
        csv_path="benchmarks/results/grad_vs_forward.csv",
        x="Operation",
        y="TimeSeconds",
        title="Autodiff Overhead: Forward vs Backward",
        ylabel="Execution Time (s)",
        outfile="grad_vs_forward.png"
    )

    lineplot(
        csv_path="benchmarks/results/method_breakdown.csv",
        x="GridSize",
        y="TotalTime",
        title="Simulation Runtime vs Grid Resolution",
        ylabel="Avg Time per Step (s)",
        outfile="method_breakdown.png"
    )

    lineplot(
        csv_path="benchmarks/results/convergence.csv",
        x="Resolution",
        y="L2Error",
        title="L2 Error vs Resolution (Convergence)",
        ylabel="L2 Error",
        loglog=True,
        outfile="convergence.png"
    )

if __name__ == "__main__":
    main()

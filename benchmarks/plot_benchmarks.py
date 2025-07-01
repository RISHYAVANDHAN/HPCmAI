# benchmarks/plot_benchmarks.py
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("benchmarks/plots", exist_ok=True)

def lineplot(csv_path, x, y, hue=None, title="", outfile=""):
    df = pd.read_csv(csv_path)
    plt.figure()
    if hue:
        for label, group in df.groupby(hue):
            plt.plot(group[x], group[y], marker="o", label=label)
        plt.legend()
    else:
        plt.plot(df[x], df[y], marker="o")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(True)
    plt.savefig(f"benchmarks/plots/{outfile}", dpi=300)
    plt.close()

if __name__ == "__main__":
    lineplot("benchmarks/results/tgv_timing.csv", x="Grid", y="TimeSeconds", hue="Backend",
             title="JIT vs No-JIT Performance", outfile="tgv_timing.png")

    lineplot("benchmarks/results/grad_vs_forward.csv", x="Operation", y="TimeSeconds",
             title="Forward vs Backward (JAX.grad)", outfile="grad_vs_forward.png")

    lineplot("benchmarks/results/method_breakdown.csv", x="GridSize", y="TotalTime",
             title="Simulation Total Time vs Grid Size", outfile="method_breakdown.png")

    lineplot("benchmarks/results/convergence.csv", x="Resolution", y="L2Error",
             title="L2 Convergence", outfile="convergence.png")

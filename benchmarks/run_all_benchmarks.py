import os

scripts = [
    "run_tgv.py",
    "benchmark_tgv.py",
    "run_grad_benchmark.py",
    "run_method_breakdown.py",
    "run_convergence_test.py",
    "plot_benchmarks.py"
]

for script in scripts:
    print(f"\n>>> Running {script}")
    os.system(f"python benchmarks/{script}")

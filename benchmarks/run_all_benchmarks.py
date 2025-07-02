# benchmarks/run_all_benchmarks.py

import os
import subprocess
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=str, default="50", help="Number of steps for all core benchmarks")
    parser.add_argument("--backend", choices=["jit", "nojit"], default="jit", help="Backend for benchmark_tgv.py")
    parser.add_argument("--advanced", action="store_true", help="Also run advanced studies")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    steps_arg = f"--steps={args.steps}"
    backend_arg = f"--backend={args.backend}"

    os.environ["PYTHONPATH"] = os.path.abspath(".")
    base = "benchmarks/"
    adv = "benchmarks/advanced/"

    core_scripts = [
        ("benchmark_tgv.py", [steps_arg, backend_arg]),
        ("run_method_breakdown.py", [steps_arg]),
        ("run_convergence_test.py", [steps_arg]),
        ("run_tgv.py", [steps_arg]),
        ("run_grad_benchmark.py", []),
        ("plot_benchmarks.py", [])
    ]

    advanced_scripts = [
        "convergence_rate.py",
        "precision_test.py",
        "jit_overhead.py",
        "gradient_sensitivity.py",
        "cpu_vs_gpu_scaling.py",
        "conservation_drift.py"
    ]

    print(f"\n>>> Running all core benchmarks (steps={args.steps}, backend={args.backend})\n")
    for script, script_args in core_scripts:
        print(f"Running {script}")
        try:
            subprocess.run([sys.executable, base + script, *script_args], check=True, env=os.environ)
        except subprocess.CalledProcessError:
            print(f"[ERROR] {script} failed.")

    if args.advanced:
        print(f"\n>>> Running advanced scientific studies\n")
        for script in advanced_scripts:
            print(f"Running {script}")
            try:
                subprocess.run([sys.executable, adv + script], check=True, env=os.environ)
            except subprocess.CalledProcessError:
                print(f"[ERROR] {script} failed.")

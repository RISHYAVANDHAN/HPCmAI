import os
import csv
import jax
from jaxfluids.examples.navier_stokes_tgv2d import NS2D
from benchmarks.utils import time_function

# Configuration
BACKENDS = ['nojit', 'jit']
GRIDS = [32, 64, 128]
STEPS = 100
RESULT_CSV = "results/tgv_results.csv"

def run_benchmark(backend: str, N: int):
    sim = NS2D(N=(N, N), ν=0.01, ρ=1.0, L=(1.0, 1.0), dt=0.001)

    if backend == 'nojit':
        step_fn = sim.step
    elif backend == 'jit':
        step_fn = jax.jit(sim.step)
        step_fn()  # warm-up for JIT

    avg_time = time_function(step_fn, steps=STEPS)
    return avg_time

def main():
    os.makedirs("results", exist_ok=True)
    with open(RESULT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["backend", "grid_size", "avg_time_per_step"])

        for backend in BACKENDS:
            for N in GRIDS:
                avg = run_benchmark(backend, N)
                writer.writerow([backend, N, avg])
                print(f"{backend}, N={N} → avg time/step = {avg:.6f} s")

if __name__ == "__main__":
    main()
    
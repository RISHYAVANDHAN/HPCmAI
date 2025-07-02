# benchmarks/run_grad_benchmark.py

import jax
import jax.numpy as jnp
import time
import csv
from benchmarks.utils import ensure_dirs

x = jnp.linspace(0, 1, 10000)

def rhs(x):
    return jnp.sin(x) * jnp.exp(x)

def main():
    ensure_dirs()
    grad_fn = jax.grad(lambda x: jnp.sum(rhs(x)))

    t0 = time.time()
    _ = rhs(x).block_until_ready()
    t1 = time.time()

    t2 = time.time()
    _ = grad_fn(x).block_until_ready()
    t3 = time.time()

    fwd = round(t1 - t0, 6)
    bwd = round(t3 - t2, 6)
    print(f"Forward: {fwd}s | Backward: {bwd}s")

    with open("benchmarks/results/grad_vs_forward.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Operation", "TimeSeconds"])
        writer.writerow(["Forward", fwd])
        writer.writerow(["Backward", bwd])

if __name__ == "__main__":
    main()

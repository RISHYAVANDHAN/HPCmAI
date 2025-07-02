# benchmarks/advanced/cpu_vs_gpu_scaling.py
import time
import os
import jax
import jax.numpy as jnp
from jax import jit

def test_backend(device):
    with jax.default_device(device):
        x = jnp.linspace(0, 1, 10**7)
        f = lambda x: jnp.sin(x) + jnp.exp(x)
        f_jit = jit(f)

        t0 = time.time()
        _ = f_jit(x).block_until_ready()
        t1 = time.time()
        return t1 - t0

for device in [jax.devices("cpu")[0], jax.devices("gpu")[0] if jax.devices("gpu") else None]:
    if device:
        t = test_backend(device)
        print(f"{device.platform.upper()} → {t:.6f}s (JIT)")

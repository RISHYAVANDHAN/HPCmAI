# benchmarks/advanced/gradient_sensitivity.py
import jax
import jax.numpy as jnp
import time

@jax.jit
def simulate(u0):
    return jnp.sum(jnp.sin(u0) * jnp.exp(u0))

x = jnp.linspace(0, 1, 10000)

# Forward
t0 = time.time()
fval = simulate(x).block_until_ready()
t1 = time.time()

# Gradient
t2 = time.time()
dfdx = jax.grad(simulate)(x).block_until_ready()
t3 = time.time()

print(f"Forward pass: {t1 - t0:.6f}s")
print(f"Backward (grad) : {t3 - t2:.6f}s")
print(f"Max |∇f|: {jnp.max(jnp.abs(dfdx)):.6f}")

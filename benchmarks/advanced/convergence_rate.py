# benchmarks/advanced/convergence_rate.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("benchmarks/results/convergence.csv")
x = np.log2(df["Resolution"].astype(float))
y = np.log2(df["L2Error"].astype(float))

slope, intercept = np.polyfit(x, y, 1)
print(f"✅ Convergence rate: {abs(slope):.3f} (should match WENO+RK3)")

plt.figure()
plt.plot(x, y, "o-", label=f"Slope ≈ {abs(slope):.2f}")
plt.xlabel("log2(Grid Resolution)")
plt.ylabel("log2(L2 Error)")
plt.title("Convergence Rate (log-log)")
plt.grid(True)
plt.legend()
os.makedirs("benchmarks/plots", exist_ok=True)
plt.savefig("benchmarks/plots/convergence_rate.png", dpi=300)

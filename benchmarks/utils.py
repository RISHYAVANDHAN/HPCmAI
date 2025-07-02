# benchmarks/utils.py

import time
import os

def time_function(fn, steps=100):
    start = time.perf_counter()
    for _ in range(steps):
        fn()
    end = time.perf_counter()
    return (end - start) / steps

def ensure_dirs():
    os.makedirs("benchmarks/results", exist_ok=True)
    os.makedirs("benchmarks/plots", exist_ok=True)

import time

def time_function(fn, steps=100):
    """Times a function over a number of steps and returns average time per step."""
    start = time.perf_counter()
    for _ in range(steps):
        fn()
    end = time.perf_counter()
    avg_time = (end - start) / steps
    return avg_time

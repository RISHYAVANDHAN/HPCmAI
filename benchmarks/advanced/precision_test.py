# benchmarks/advanced/precision_test.py
import time
import csv
import jax
import jax.numpy as jnp
from jaxfluids.input.input_manager import InputManager
from jaxfluids.simulation_manager import SimulationManager
from jaxfluids.initialization.initialization_manager import InitializationManager
from jaxfluids.data_types.ml_buffers import ParametersSetup, CallablesSetup
import copy, os

def run(case_file, num_file, steps, use_double):
    jax.config.update("jax_enable_x64", use_double)    
    input_manager = InputManager(case_file, num_file)
    sim = SimulationManager(input_manager)
    init_manager = InitializationManager(input_manager)
    buffers = init_manager.initialization()
    sim.initialize(buffers)
    sim.buffers = buffers

    start = time.perf_counter()
    for _ in range(steps):
        ctrl = sim.compute_control_flow_params(sim.buffers.time_control_variables, sim.buffers.step_information)
        ctrl = ctrl._replace(is_feed_foward=False, is_cumulative_statistics=False, is_logging_statistics=False)
        sim.buffers, _ = sim.do_integration_step(sim.buffers, ctrl, ParametersSetup(), CallablesSetup())
    end = time.perf_counter()
    final_field = sim.buffers.simulation_buffers.material_fields.conservatives
    return (end - start) / steps, float(jnp.linalg.norm(final_field))

os.makedirs("benchmarks/results", exist_ok=True)
with open("benchmarks/results/precision_test.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Precision", "TimePerStep", "FinalNorm"])
    for precision in [("float32", False), ("float64", True)]:
        t, norm = run("benchmarks/configs/case_64.json", "benchmarks/configs/numerical_base.json", 50, precision[1])
        writer.writerow([precision[0], f"{t:.6f}", f"{norm:.6e}"])
        print(f"{precision[0]}: {t:.6f}s/step, ||u|| = {norm:.6e}")

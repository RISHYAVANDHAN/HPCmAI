# benchmarks/run_tgv.py

import argparse
import time
import csv
from benchmarks.utils import ensure_dirs
from jaxfluids.input.input_manager import InputManager
from jaxfluids.simulation_manager import SimulationManager
from jaxfluids.initialization.initialization_manager import InitializationManager
from jaxfluids.data_types.ml_buffers import ParametersSetup, CallablesSetup

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    return parser.parse_args()

def run(case_path, num_path, steps):
    input_manager = InputManager(case_path, num_path)
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

    avg_time = (end - start) / steps
    return avg_time, sim.buffers.time_control_variables.simulation_step, sim.buffers.time_control_variables.physical_simulation_time

if __name__ == "__main__":
    args = parse_args()
    ensure_dirs()
    outfile = "benchmarks/results/tgv_summary.csv"
    num_file = "benchmarks/configs/numerical_base.json"
    sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]

    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Grid", "AvgStepTime", "FinalStep", "FinalTime"])
        for size in sizes:
            avg, step, sim_time = run(f"benchmarks/configs/case_{size}.json", num_file, args.steps)
            print(f"{size}: {avg:.6f} s/step, step={step}, time={sim_time:.5f}")
            writer.writerow([size, f"{avg:.6f}", step, f"{sim_time:.5f}"])

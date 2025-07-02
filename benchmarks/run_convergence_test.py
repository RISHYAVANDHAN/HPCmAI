# benchmarks/run_convergence_test.py

import argparse
import time
import csv
import jax.numpy as jnp
from benchmarks.utils import ensure_dirs
from jaxfluids.input.input_manager import InputManager
from jaxfluids.simulation_manager import SimulationManager
from jaxfluids.initialization.initialization_manager import InitializationManager
from jaxfluids.data_types.ml_buffers import ParametersSetup, CallablesSetup

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    return parser.parse_args()

def run_sim(case_path, num_path, steps, halo=5):
    input_manager = InputManager(case_path, num_path)
    sim = SimulationManager(input_manager)
    init_manager = InitializationManager(input_manager)
    buffers = init_manager.initialization()
    sim.initialize(buffers)
    sim.buffers = buffers

    for _ in range(steps):
        ctrl = sim.compute_control_flow_params(sim.buffers.time_control_variables, sim.buffers.step_information)
        ctrl = ctrl._replace(is_feed_foward=False, is_cumulative_statistics=False, is_logging_statistics=False)
        sim.buffers, _ = sim.do_integration_step(sim.buffers, ctrl, ParametersSetup(), CallablesSetup())

    field = sim.buffers.simulation_buffers.material_fields.conservatives
    return field[:, halo:-halo, halo:-halo, :]  # remove ghost cells

def downsample(arr, target_shape):
    """Naively downsample arr to target spatial resolution using slicing."""
    _, H, W, _ = arr.shape
    _, TH, TW, _ = target_shape
    stride_h = H // TH
    stride_w = W // TW
    return arr[:, ::stride_h, ::stride_w, :]

if __name__ == "__main__":
    args = parse_args()
    ensure_dirs()

    num_file = "benchmarks/configs/numerical_base.json"
    ref_res = 16
    test_resolutions = [16, 32, 64, 128, 256, 512, 1024, 2048]

    print(f"\n>>> Running convergence test (reference: {ref_res}x{ref_res})")
    ref = run_sim(f"benchmarks/configs/case_{ref_res}.json", num_file, args.steps)
    target_shape = ref.shape

    outfile = "benchmarks/results/convergence.csv"
    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Resolution", "L2Error"])
        for res in test_resolutions:
            try:
                result = run_sim(f"benchmarks/configs/case_{res}.json", num_file, args.steps)
                result_down = downsample(result, target_shape)
                diff = ref - result_down
                l2_error = float(jnp.sqrt(jnp.mean(diff ** 2)))
                print(f"[OK] Grid {res} → L2Error: {l2_error:.6e}")
                writer.writerow([res, f"{l2_error:.6e}"])
            except Exception as e:
                print(f"[ERROR] Failed on grid {res}: {e}")

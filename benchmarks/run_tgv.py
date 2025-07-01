# benchmarks/run_tgv.py

import argparse
import time
from jaxfluids.input.input_manager import InputManager
from jaxfluids.simulation_manager import SimulationManager
from jaxfluids.data_types.ml_buffers import ParametersSetup, CallablesSetup
from jaxfluids.initialization.initialization_manager import InitializationManager

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="benchmarks/configs/case_64.json")
    parser.add_argument("--numerical", type=str, default="benchmarks/configs/numerical_base.json")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--backend", choices=["jit", "nojit"], default="jit")
    return parser.parse_args()

def main():
    args = parse_args()

    input_manager = InputManager(args.case, args.numerical)
    sim = SimulationManager(input_manager)
    init_manager = InitializationManager(input_manager)
    jxf_buffers = init_manager.initialization()
    sim.initialize(jxf_buffers)
    sim.buffers = jxf_buffers

    # Time integration loop
    start = time.perf_counter()
    for _ in range(args.steps):
        control_params = sim.compute_control_flow_params(
            sim.buffers.time_control_variables,
            sim.buffers.step_information
        )
        if args.backend == "jit":
            sim.buffers, _ = sim._do_integration_step_jit(
                sim.buffers,
                control_params,
                ParametersSetup(),
                CallablesSetup()
            )
        else:
            sim.buffers, _ = sim._do_integration_step(
                sim.buffers,
                control_params,
                ParametersSetup(),
                CallablesSetup()
            )
    end = time.perf_counter()

    avg_time = (end - start) / args.steps
    print(f"{args.backend.upper()} {args.case.split('/')[-1]} {avg_time:.6f} sec/step")

    # Print final state summary
    step = sim.buffers.time_control_variables.simulation_step
    time_now = sim.buffers.time_control_variables.physical_simulation_time
    print(f"Final step: {step}, Final simulation time: {time_now:.5f}")

if __name__ == "__main__":
    main()

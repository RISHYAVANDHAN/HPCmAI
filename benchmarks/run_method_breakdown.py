import time
from jaxfluids.input.input_manager import InputManager
from jaxfluids.simulation_manager import SimulationManager
from jaxfluids.data_types.ml_buffers import ParametersSetup, CallablesSetup

def run(case_file, num_file):
    input_manager = InputManager(case_file, num_file)
    sim = SimulationManager(input_manager)
    jxf_buffers = input_manager.numerical_setup.init_fields(input_manager)
    sim.initialize(jxf_buffers)

    timings = {}
    for _ in range(10):
        start = time.perf_counter()
        sim.buffers, _ = sim.do_integration_step(
            sim.buffers,
            sim.compute_control_flow_params(sim.buffers.time_control_variables, sim.buffers.step_information),
            ParametersSetup(),
            CallablesSetup()
        )
        end = time.perf_counter()
        timings.setdefault("TotalTime", []).append(end - start)

    return timings

if __name__ == "__main__":
    case = "benchmarks/configs/case_64.json"
    num = "benchmarks/configs/numerical_base.json"
    times = run(case, num)
    for k, v in times.items():
        print(f"{k}: {sum(v)/len(v):.6f}")

from jaxfluids.input.input_manager import InputManager
from jaxfluids.simulation_manager import SimulationManager
from jaxfluids.data_types.ml_buffers import ParametersSetup, CallablesSetup

def run_sim(case_file, num_file):
    input_manager = InputManager(case_file, num_file)
    sim = SimulationManager(input_manager)
    jxf_buffers = input_manager.numerical_setup.init_fields(input_manager)
    sim.initialize(jxf_buffers)

    for _ in range(10):
        sim.buffers, _ = sim.do_integration_step(
            sim.buffers,
            sim.compute_control_flow_params(sim.buffers.time_control_variables, sim.buffers.step_information),
            ParametersSetup(),
            CallablesSetup()
        )
    return sim.buffers

if __name__ == "__main__":
    base = "benchmarks/configs/"
    ref = run_sim(base + "case_128.json", base + "numerical_base.json")
    print("Reference run complete.")

# benchmarks/advanced/jit_overhead.py
import time
import jax
from jaxfluids.input.input_manager import InputManager
from jaxfluids.simulation_manager import SimulationManager
from jaxfluids.initialization.initialization_manager import InitializationManager
from jaxfluids.data_types.ml_buffers import ParametersSetup, CallablesSetup

input_manager = InputManager("benchmarks/configs/case_64.json", "benchmarks/configs/numerical_base.json")
sim = SimulationManager(input_manager)
init = InitializationManager(input_manager)
buffers = init.initialization()
sim.initialize(buffers)
sim.buffers = buffers
step_fn = sim._do_integration_step_jit

# First (JIT compile)
ctrl = sim.compute_control_flow_params(buffers.time_control_variables, buffers.step_information)
ctrl = ctrl._replace(is_feed_foward=False, is_cumulative_statistics=False, is_logging_statistics=False)
t0 = time.time()
sim.buffers, _ = step_fn(sim.buffers, ctrl, ParametersSetup(), CallablesSetup())
t1 = time.time()
print(f"⏱ First step (includes JIT compile): {t1 - t0:.4f}s")

# Follow-up steps
timings = []
for _ in range(20):
    t0 = time.time()
    sim.buffers, _ = step_fn(sim.buffers, ctrl, ParametersSetup(), CallablesSetup())
    t1 = time.time()
    timings.append(t1 - t0)
print(f"✅ Avg JIT'd step time: {sum(timings)/len(timings):.6f}s")

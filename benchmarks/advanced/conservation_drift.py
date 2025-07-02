# benchmarks/advanced/conservation_drift.py
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
from jaxfluids.input.input_manager import InputManager
from jaxfluids.simulation_manager import SimulationManager
from jaxfluids.initialization.initialization_manager import InitializationManager
from jaxfluids.data_types.ml_buffers import ParametersSetup, CallablesSetup

steps = 1000
input_manager = InputManager("benchmarks/configs/case_64.json", "benchmarks/configs/numerical_base.json")
sim = SimulationManager(input_manager)
init = InitializationManager(input_manager)
buffers = init.initialization()
sim.initialize(buffers)
sim.buffers = buffers

mass_drift = []
for i in range(steps):
    ctrl = sim.compute_control_flow_params(sim.buffers.time_control_variables, sim.buffers.step_information)
    ctrl = ctrl._replace(is_feed_foward=False, is_cumulative_statistics=False, is_logging_statistics=False)
    sim.buffers, _ = sim.do_integration_step(sim.buffers, ctrl, ParametersSetup(), CallablesSetup())
    rho = sim.buffers.simulation_buffers.material_fields.conservatives[0]
    total_mass = float(jnp.sum(rho))
    mass_drift.append(total_mass)

plt.figure()
plt.plot(mass_drift)
plt.xlabel("Step")
plt.ylabel("Total Mass")
plt.title("Conservation Drift Over Time")
plt.grid(True)
plt.savefig("benchmarks/plots/conservation_drift.png", dpi=300)
print("✅ Saved plot → benchmarks/plots/conservation_drift.png")

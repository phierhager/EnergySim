from energysim.cosim.factory import SimulatorFactory
from energysim.cosim.config import BuildingSimulationConfig
import dacite
import yaml

# Load building simulation configuration from YAML file
with open("config/env_config.yaml", "r") as file:
    yaml_cfg = yaml.safe_load(file)

building_sim_config = dacite.from_dict(BuildingSimulationConfig, yaml_cfg)
print("Building simulation configuration loaded successfully.")

simulator = SimulatorFactory.create_simulator(building_sim_config)
print("Building simulator created successfully.")

simulator.reset()
print("Simulator reset successfully.")

for _ in range(50):
    random_action = simulator.action_space.sample()
    output = simulator.step(random_action)
    print(f"Step {_ + 1} completed.")
    print(f"Simulation output: {output}")
print("Simulation steps completed successfully.")
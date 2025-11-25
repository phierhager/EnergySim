"""
energysim.core.models
=====================

This subpackage contains the physics-based and data-driven models for
all system components (HVAC, Battery, PV, Building Thermal Physics, etc.),
as well as the factory functions to instantiate them.
"""

# 1. Thermal Physics (RC Network)
from .thermal_model import (
    AbstractThermalModel,
    RCNetworkModel
)

# 2. HVAC: Heat Pumps (Heating)
from .heat_pump_model import (
    AbstractHeatPumpModel,
    StatelessHeatPumpModel,
    RampingHeatPumpModel,
    VariableCOPHeatPumpModel,
    MechanisticHeatPump,
    PassthroughHeatPumpModel
)

# 3. HVAC: Air Conditioners (Cooling)
from .air_conditioner_model import (
    AbstractAirConditionerModel,
    StatelessAirConditionerModel,
    RampingAirConditionerModel,
    VariableCOPAirConditionerModel,
    PassthroughAirConditionerModel
)

# 4. Storage: Electrical (Batteries)
from .battery_model import (
    AbstractBatteryModel,
    SimpleBatteryModel,
    DegradationBatteryModel,
    PassthroughBatteryModel
)

# 5. Storage: Thermal (Water Tanks / Phase Change)
from .thermal_storage_model import (
    AbstractThermalStorage,
    StratifiedThermalStorageModel,
    GridThermalStorageModel,
    ThermalStoragePassthrough
)

# 6. Generation: Solar PV
from .solar_model import (
    AbstractSolarModel,
    GeometricSolarModel,
    SimpleSolarModel,
    PassthroughSolarModel
)

# 7. Machines & Appliances (Load)
from .machines import (
    AbstractMachine,
    MachineState,
    PassiveEquipment,
    SmartAppliance
)

# 8. Occupancy & Moisture
from .occupancy import OccupancyModel
from .moisture_model import (
    AbstractMoistureModel,
    DynamicMoistureModel,
    EMPDMoistureModel
)

# 9. Model Factory (Construction Helpers)
from .factory import (
    create_battery,
    create_heat_pump,
    create_ac,
    create_storage,
    create_thermal,
    create_solar,
    create_smart_machines,
    create_passive_machines,
    create_occupants,
    create_moisture
)

# 10. Cost & Objective Functions
from ..objectives import (
    f_cost_step,
    f_terminal_cost
)

__all__ = [
    # Thermal
    "AbstractThermalModel",
    "RCNetworkModel",
    
    # Heat Pump
    "AbstractHeatPumpModel",
    "StatelessHeatPumpModel",
    "RampingHeatPumpModel",
    "VariableCOPHeatPumpModel",
    "MechanisticHeatPump",
    "PassthroughHeatPumpModel",
    
    # Air Conditioner
    "AbstractAirConditionerModel",
    "StatelessAirConditionerModel",
    "RampingAirConditionerModel",
    "VariableCOPAirConditionerModel",
    "PassthroughAirConditionerModel",
    
    # Battery
    "AbstractBatteryModel",
    "SimpleBatteryModel",
    "DegradationBatteryModel",
    "PassthroughBatteryModel",
    
    # Thermal Storage
    "AbstractThermalStorage",
    "StratifiedThermalStorageModel",
    "GridThermalStorageModel",
    "ThermalStoragePassthrough",
    
    # Solar
    "AbstractSolarModel",
    "GeometricSolarModel",
    "SimpleSolarModel",
    "PassthroughSolarModel",
    
    # Machines
    "AbstractMachine",
    "MachineState",
    "PassiveEquipment",
    "SmartAppliance",
    
    # Occupancy & Moisture
    "OccupancyModel",
    "AbstractMoistureModel",
    "DynamicMoistureModel",
    "EMPDMoistureModel",
    
    # Factory
    "create_battery",
    "create_heat_pump",
    "create_ac",
    "create_storage",
    "create_thermal",
    "create_solar",
    "create_smart_machines",
    "create_passive_machines",
    "create_occupants",
    "create_moisture",
    
    # Objectives
    "f_cost_step",
    "f_terminal_cost",
]
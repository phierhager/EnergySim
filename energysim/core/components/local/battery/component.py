from energysim.core.components.local.battery.actuator import (
    IBatteryActuator,
    SimpleBatteryActuator,
)
from energysim.core.components.local.battery.model import (
    IBatteryModel,
)
from energysim.core.components.local.shared.component import LocalComponent
from energysim.core.components.shared.spaces import Space
from energysim.core.components.shared.component_base import (
    ComponentBase,
    ComponentOutputs,
)
from energysim.core.components.shared.component_outputs import ElectricalEnergy
from energysim.core.components.registry import register_local_component
from energysim.core.components.local.battery.config import (
    BatteryComponentConfig,
)
import numpy as np


@register_local_component(BatteryComponentConfig)
class Battery(LocalComponent):
    def __init__(self, model: IBatteryModel, actuator: IBatteryActuator):
        self._model = model
        self._actuator = actuator
        self._initialized = False

    def initialize(self) -> ComponentOutputs:
        if self._initialized:
            raise RuntimeError("Battery already initialized.")
        self._initialized = True
        return ComponentOutputs(electrical_storage=self._model.storage)

    def advance(
        self, input: dict[str, np.ndarray], dt_seconds: float
    ) -> ComponentOutputs:
        if not self._initialized:
            raise RuntimeError("Battery must be initialized before advancing.")

        action = input.get("action")
        if action is None:
            raise ValueError("Input must contain 'action' key.")

        requested_power = self._actuator.interpret_action(action, self._model.max_power)
        energy_transfer = self._model.apply_power(requested_power, dt_seconds)

        return ComponentOutputs(
            electrical_storage=self._model.storage,
            electrical_energy=ElectricalEnergy(
                demand_j=max(0, energy_transfer),
                generation_j=max(0, -energy_transfer),
            ),
        )

    @property
    def action_space(self) -> dict[str, Space]:
        return self._actuator.action_space

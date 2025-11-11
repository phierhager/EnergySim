from energysim.core.components.battery.models import (
    BatteryModelBase,
)
from energysim.core.components.base import ActionDrivenComponent
from energysim.core.shared.spaces import ContinuousSpace, DictSpace
from energysim.core.components.base import (
    ComponentOutputs,
)
from energysim.core.components.outputs import ElectricalEnergy
from energysim.core.components.registry import register_component
from energysim.core.components.battery.config import (
    BatteryComponentConfig,
)


@register_component(BatteryComponentConfig)
class Battery(ActionDrivenComponent):
    def __init__(self, model: BatteryModelBase):
        self._model = model
        self._initialized = False

    def initialize(self) -> ComponentOutputs:
        if self._initialized:
            raise RuntimeError("Battery already initialized.")
        self._initialized = True
        return ComponentOutputs(electrical_storage=self._model.storage)

    def advance(
        self, action: dict[str, float], dt_seconds: int
    ) -> ComponentOutputs:
        if not self._initialized:
            raise RuntimeError("Battery must be initialized before advancing.")

        if "normalized_power" not in action:
            raise ValueError("Input must contain 'normalized_power' key.")

        normalized_power = action["normalized_power"] # in [-1, 1]
        power = normalized_power * self._model.max_power # in Watts
        energy_transfer = self._model.apply_power(power, dt_seconds)

        return ComponentOutputs(
            electrical_storage=self._model.storage,
            electrical_energy=ElectricalEnergy(
                demand_j=max(0, energy_transfer),
                generation_j=max(0, -energy_transfer),
            ),
        )

    @property
    def action_space(self) -> DictSpace:
        return DictSpace(
            spaces={
                "normalized_power": ContinuousSpace(
                    lower_bound=-1.0, upper_bound=1.0
                )
            }
        )
    
    @property
    def model(self) -> BatteryModelBase:
        return self._model

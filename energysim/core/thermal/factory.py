from energysim.core.thermal.base import (
    ThermalModel,
    ThermalModelConfig,
)
from energysim.core.thermal.registry import registry


def build_thermal_model(config: ThermalModelConfig) -> ThermalModel:
    model_cls = registry[config.__class__.__name__]
    return model_cls(config)

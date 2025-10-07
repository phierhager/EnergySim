"""Tests for battery component integration."""
import pytest
from unittest.mock import Mock
import numpy as np
from energysim.core.components.local.battery.component import Battery
from energysim.core.components.local.battery.model import IBatteryModel
from energysim.core.components.shared.component_outputs import ComponentOutputs, ElectricalStorage, ElectricalEnergy


class MockBatteryModel(IBatteryModel):
    def __init__(self, max_power=10.0, storage=None):
        self._max_power = max_power
        self._storage = storage or ElectricalStorage(capacity=100.0, soc=0.5)
    
    @property
    def max_power(self) -> float:
        return self._max_power
    
    @property
    def storage(self) -> ElectricalStorage:
        return self._storage
    
    def apply_power(self, normalized_power: float, dt_seconds: float) -> float:
        return normalized_power * dt_seconds  # Simplified for testing


def test_battery_initialization():
    # Arrange
    model = MockBatteryModel()
    
    # Act
    battery = Battery(model)
    
    # Assert
    assert battery._model == model
    assert battery._initialized is False


def test_battery_initialize():
    # Arrange
    expected_storage = ElectricalStorage(capacity=100.0, soc=0.6)
    model = MockBatteryModel(storage=expected_storage)
    battery = Battery(model)
    
    # Act
    outputs = battery.initialize()
    
    # Assert
    assert isinstance(outputs, ComponentOutputs)
    assert outputs.electrical_storage == expected_storage
    assert battery._initialized is True


def test_battery_initialize_twice_raises_error():
    # Arrange
    model = MockBatteryModel()
    battery = Battery(model)
    battery.initialize()
    
    # Act & Assert
    with pytest.raises(RuntimeError, match="Battery already initialized"):
        battery.initialize()


def test_battery_advance_without_initialization_raises_error():
    # Arrange
    model = MockBatteryModel()
    battery = Battery(model)
    
    # Act & Assert
    with pytest.raises(RuntimeError, match="Battery must be initialized before advancing"):
        battery.advance({"normalized_power": np.array([1.0])}, 3600.0)


def test_battery_advance_missing_action_raises_error():
    # Arrange
    model = MockBatteryModel()
    battery = Battery(model)
    battery.initialize()
    
    # Act & Assert
    with pytest.raises(ValueError, match="Input must contain 'normalized_power' key"):
        battery.advance({}, 3600.0)


def test_battery_advance_positive_power():
    # Arrange
    model = MockBatteryModel(max_power=10.0)
    battery = Battery(model)
    battery.initialize()
    
    action = np.array([0.5])  # 50% of max power
    input_dict = {"normalized_power": action}
    dt_seconds = 3600.0
    
    # Act
    outputs = battery.advance(input_dict, dt_seconds)
    
    # Assert
    assert isinstance(outputs, ComponentOutputs)
    assert outputs.electrical_storage == model.storage
    assert isinstance(outputs.electrical_energy, ElectricalEnergy)
    
    # 0.5 * 10.0 * 3600.0 = 18000J demand (positive power = charging)
    assert outputs.electrical_energy.demand_j == 18000.0
    assert outputs.electrical_energy.generation_j == 0.0


def test_battery_advance_negative_power():
    # Arrange
    model = MockBatteryModel(max_power=10.0)
    battery = Battery(model)
    battery.initialize()
    
    action = np.array([-0.3])  # -30% of max power
    input_dict = {"normalized_power": action}
    dt_seconds = 3600.0
    
    # Mock the model to return negative energy (discharge)
    model.apply_power = Mock(return_value=-10800.0)  # -3kW * 3600s
    
    # Act
    outputs = battery.advance(input_dict, dt_seconds)
    
    # Assert
    assert outputs.electrical_energy.demand_j == 0.0
    assert outputs.electrical_energy.generation_j == 10800.0
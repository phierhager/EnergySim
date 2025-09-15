"""Tests for battery model implementation."""

import pytest
from bems_simulation.core.components.local.battery.model import (
    DegradingBatteryModel,
    SimpleBatteryModel,
)
from bems_simulation.core.components.local.battery.config import (
    DegradingBatteryModelConfig,
    SimpleBatteryModelConfig,
)
from bems_simulation.core.components.shared.component_outputs import ElectricalStorage


def test_simple_battery_model_initialization():
    # Arrange
    config = SimpleBatteryModelConfig(
        efficiency=0.9, max_power=10.0, capacity=20.0, init_soc=0.5, deadband=0.1
    )

    # Act
    model = SimpleBatteryModel(config)

    # Assert
    assert model.max_power == 10.0
    assert model.storage.capacity == 20.0
    assert model.storage.soc == 0.5


def test_battery_charging_within_limits():
    # Arrange
    config = SimpleBatteryModelConfig(
        capacity=100.0, init_soc=0.5, efficiency=0.9, max_power=50.0
    )
    model = SimpleBatteryModel(config)
    dt_seconds = 3600.0  # 1 hour

    # Act - charge at 10kW for 1 hour = 10*3600 = 36000J requested
    # But max_power=50.0, so limited to min(36000, 50) = 50J
    energy_transfer = model.apply_power(10.0, dt_seconds)

    # Assert
    # Energy limited to 50J, with efficiency applied: energy_to_store = 50*0.9 = 45J
    # Return value = energy_to_store / eff = 45/0.9 = 50J (grid consumption)
    # SoC = 0.5 + 45/100 = 0.95
    assert abs(energy_transfer - 50.0) < 1e-6  # Grid energy consumed
    expected_soc = (
        0.5 + (50.0 * 0.9) / 100.0
    )  # 50J limited * 0.9 efficiency stored in 100J capacity
    assert abs(model.storage.soc - expected_soc) < 1e-6


def test_battery_discharging_within_limits():
    # Arrange
    config = SimpleBatteryModelConfig(
        capacity=100.0, init_soc=0.5, efficiency=0.9, max_power=50.0
    )
    model = SimpleBatteryModel(config)
    dt_seconds = 3600.0  # 1 hour

    # Act - discharge at 10kW for 1 hour = 10*3600 = 36000J requested
    # But max_power=50.0, so limited to min(36000, 50) = 50J
    energy_transfer = model.apply_power(-10.0, dt_seconds)

    # Assert
    # For discharge: energy_to_release = min(50/0.9, 50) = min(55.56, 50) = 50J taken from storage
    # Return value = -energy_to_release * eff = -50 * 0.9 = -45J (delivered to load)
    # SoC = 0.5 - 50/100 = 0.0
    assert energy_transfer < 0  # Negative indicates discharge
    assert abs(energy_transfer + 45.0) < 1e-6  # 45J delivered to load (negative value)
    expected_soc = 0.5 - 50.0 / 100.0  # 50J taken from 100J storage
    assert abs(model.storage.soc - expected_soc) < 1e-6


def test_battery_charging_when_full():
    # Arrange
    config = SimpleBatteryModelConfig(
        capacity=100.0, init_soc=1.0, efficiency=0.9, max_power=50.0
    )
    model = SimpleBatteryModel(config)

    # Act
    energy_transfer = model.apply_power(10.0, 3600.0)

    # Assert
    assert energy_transfer == 0.0
    assert model.storage.soc == 1.0


def test_battery_discharging_when_empty():
    # Arrange
    config = SimpleBatteryModelConfig(
        capacity=100.0, init_soc=0.0, efficiency=0.9, max_power=50.0
    )
    model = SimpleBatteryModel(config)

    # Act
    energy_transfer = model.apply_power(-10.0, 3600.0)

    # Assert
    assert energy_transfer == 0.0
    assert model.storage.soc == 0.0


def test_battery_power_limited_by_max_power():
    # Arrange
    config = SimpleBatteryModelConfig(
        capacity=100.0, init_soc=0.5, efficiency=1.0, max_power=5.0
    )
    model = SimpleBatteryModel(config)
    dt_seconds = 3600.0

    # Act - request 10kW but max is 5kW
    energy_transfer = model.apply_power(10.0, dt_seconds)

    # Assert
    # Should be limited to 5kW * 1 hour = 5kWh
    assert abs(energy_transfer - 5.0) < 1e-6
    assert abs(model.storage.soc - 0.55) < 1e-6  # 0.5 + 5/100


def test_battery_charging_limited_by_capacity():
    # Arrange
    config = SimpleBatteryModelConfig(
        capacity=100.0, init_soc=0.95, efficiency=0.9, max_power=50.0
    )
    model = SimpleBatteryModel(config)
    dt_seconds = 3600.0

    # Act - try to charge 10kWh but only 5kWh space available
    energy_transfer = model.apply_power(10.0, dt_seconds)

    # Assert
    # Available capacity: (1.0 - 0.95) * 100 = 5kWh
    # Energy to store: min(10*0.9, 5) = 5kWh
    # Energy drawn: 5/0.9 = 5.556kWh
    expected_energy_drawn = 5.0 / 0.9
    assert abs(energy_transfer - expected_energy_drawn) < 1e-6
    assert abs(model.storage.soc - 1.0) < 1e-6


def test_battery_invalid_dt_seconds():
    # Arrange
    config = SimpleBatteryModelConfig()
    model = SimpleBatteryModel(config)

    # Act & Assert
    with pytest.raises(ValueError, match="dt_seconds must be positive"):
        model.apply_power(10.0, 0.0)

    with pytest.raises(ValueError, match="dt_seconds must be positive"):
        model.apply_power(10.0, -1.0)


@pytest.mark.parametrize("mode", ["linear", "exponential", "polynomial"])
def test_degrading_battery_model_initialization(mode):
    config = DegradingBatteryModelConfig(
        efficiency=0.95,
        max_power=20.0,
        capacity=50.0,
        init_soc=0.6,
        degradation_mode=mode,
        degradation_rate=0.001,
    )
    model = DegradingBatteryModel(config)

    assert model.max_power == 20.0
    assert model.storage.capacity == 50.0
    assert model.storage.soc == 0.6


def test_linear_degradation_reduces_capacity():
    config = DegradingBatteryModelConfig(
        capacity=100.0,
        max_power=10.0,
        efficiency=1.0,
        degradation_mode="linear",
        degradation_rate=0.1,
    )
    model = DegradingBatteryModel(config)

    # Simulate charging
    for _ in range(20):
        model.apply_power(10.0, 1.0)

    assert model.storage.capacity < 100.0
    assert model.storage.capacity >= config.min_capacity_fraction * 100.0


def test_exponential_degradation_reduces_capacity():
    config = DegradingBatteryModelConfig(
        capacity=100.0,
        max_power=10.0,
        efficiency=1.0,
        degradation_mode="exponential",
        degradation_rate=0.001,
    )
    model = DegradingBatteryModel(config)

    for _ in range(100):
        model.apply_power(10.0, 1.0)

    assert model.storage.capacity < 100.0
    assert model.storage.capacity >= config.min_capacity_fraction * 100.0


def test_polynomial_degradation_reduces_capacity():
    config = DegradingBatteryModelConfig(
        capacity=100.0,
        max_power=10.0,
        efficiency=1.0,
        degradation_mode="polynomial",
        degradation_rate=0.0001,
        poly_exponent=2.0,
    )
    model = DegradingBatteryModel(config)

    for _ in range(200):
        model.apply_power(10.0, 1.0)

    assert model.storage.capacity < 100.0
    assert model.storage.capacity >= config.min_capacity_fraction * 100.0


def test_degradation_never_below_min_capacity():
    config = DegradingBatteryModelConfig(
        capacity=100.0,
        max_power=10.0,
        efficiency=1.0,
        degradation_mode="linear",
        degradation_rate=10.0,  # very aggressive
        min_capacity_fraction=0.6,
    )
    model = DegradingBatteryModel(config)

    # Large throughput to trigger strong degradation
    for _ in range(50):
        model.apply_power(10.0, 1.0)

    assert model.storage.capacity == pytest.approx(60.0)


from energysim.core.components.battery.config import (
    DegradingBatteryModelConfig,
    SimpleBatteryModelConfig,
)
from energysim.core.components.model_base import ModelBase
from energysim.core.components.outputs import (
    ElectricalStorage,
)
from energysim.core.components.registry import register_model
from abc import abstractmethod

from energysim.core.shared.mpc_interfaces import MPCBuilderBase, MPCFormulatable
import casadi as ca


class BatteryModelBase(ModelBase):
    @property
    @abstractmethod
    def max_power(self) -> float:
        pass

    @property
    @abstractmethod
    def storage(self) -> ElectricalStorage:
        pass

    @abstractmethod
    def apply_power(self, normalized_power: float, dt_seconds: float) -> float:
        """Applies requested power (positive=charge, negative=discharge)."""
        pass

    def get_mpc_state_variables(self, builder: MPCBuilderBase) -> list[ca.SX]:
        # Define one state variable: state-of-charge (soc)
        soc = builder.opti.variable(1)
        builder.set_initial_state('battery_soc', soc) # Register it
        return [soc]

    def get_mpc_action_variables(self, builder: MPCBuilderBase) -> list[ca.SX]:
        # Define one action variable: power (positive=charge, negative=discharge)
        p_bat = builder.opti.variable(1)
        return [p_bat]


@register_model(SimpleBatteryModelConfig)
class SimpleBatteryModel(BatteryModelBase, MPCFormulatable):
    def __init__(self, config: SimpleBatteryModelConfig):
        self._config = config
        self._storage = ElectricalStorage(capacity=config.capacity, soc=config.init_soc)

    @property
    def max_power(self) -> float:
        return self._config.max_power

    @property
    def storage(self) -> ElectricalStorage:
        return self._storage

    def apply_power(self, normalized_power: float, dt_seconds: float) -> float:
        """Applies requested power (positive=charge, negative=discharge)."""
        if dt_seconds <= 0:
            raise ValueError("dt_seconds must be positive.")

        requested_energy = abs(normalized_power) * dt_seconds
        is_charge = normalized_power > 0
        return self._perform_energy_transfer(requested_energy, is_charge)

    def _perform_energy_transfer(
        self, requested_energy: float, is_charge: bool
    ) -> float:
        soc = self._storage.soc
        capacity = self._config.capacity
        eff = self._config.efficiency
        max_power_energy = min(requested_energy, self._config.max_power)

        if is_charge:
            if soc >= 1.0:
                return 0.0
            available_capacity = (1.0 - soc) * capacity
            energy_to_store = min(max_power_energy * eff, available_capacity)
            self._storage = ElectricalStorage(
                soc=soc + energy_to_store / capacity, capacity=capacity
            )
            return energy_to_store / eff
        else:
            if soc <= 0.0:
                return 0.0
            energy_available = soc * capacity
            energy_to_release = min(max_power_energy / eff, energy_available)
            self._storage = ElectricalStorage(
                soc=soc - energy_to_release / capacity, capacity=capacity
            )
            return -energy_to_release * eff

    def add_mpc_dynamics_constraints(
        self,
        builder: MPCBuilderBase,
        k: int,
        states: dict[str, ca.SX],
        actions: dict[str, ca.SX],
        exogenous: dict[str, ca.SX]
    ):
        # Get vars for this step (k) and next step (k+1)
        soc_k = states['battery_soc'][k]
        soc_k_plus_1 = states['battery_soc'][k+1]
        p_bat_k = actions['battery_power'][k]

        dt = builder.dt_seconds
        eff = self._config.efficiency
        cap = self._config.capacity

        # Symbolic representation of the charging/discharging logic
        # This is more complex than the simple equation.
        # We must use helper variables.
        energy_change = dt * p_bat_k
        
        # We need to model one-way efficiency
        energy_stored = ca.if_else(energy_change > 0, energy_change * eff, energy_change / eff)

        # The core dynamics equation
        builder.add_constraint(soc_k_plus_1 == soc_k + energy_stored / cap)

    def add_mpc_operational_constraints(
        self,
        builder: MPCBuilderBase,
        k: int,
        states: dict[str, ca.SX],
        actions: dict[str, ca.SX],
        exogenous: dict[str, ca.SX]
    ):
        soc_k = states['battery_soc'][k]
        p_bat_k = actions['battery_power'][k]

        # State constraints
        builder.add_constraint(soc_k >= 0.0)
        builder.add_constraint(soc_k <= 1.0)

        # Action constraints
        builder.add_constraint(p_bat_k >= -self._config.max_power)
        builder.add_constraint(p_bat_k <= self._config.max_power)


@register_model(DegradingBatteryModelConfig)
class DegradingBatteryModel(BatteryModelBase):
    """Battery model with capacity degradation."""

    def __init__(self, config: DegradingBatteryModelConfig):
        self._config = config
        self._initial_capacity = config.capacity
        self._storage = ElectricalStorage(capacity=config.capacity, soc=config.init_soc)
        self._throughput = 0.0  # cumulative charged + discharged energy

    @property
    def max_power(self) -> float:
        return self._config.max_power

    @property
    def storage(self) -> ElectricalStorage:
        return self._storage

    def apply_power(self, normalized_power: float, dt_seconds: float) -> float:
        if dt_seconds <= 0:
            raise ValueError("dt_seconds must be positive.")

        requested_energy = abs(normalized_power) * dt_seconds
        is_charge = normalized_power > 0
        transferred = self._perform_energy_transfer(requested_energy, is_charge)
        self._throughput += abs(transferred)
        self._apply_degradation()
        return transferred

    def _perform_energy_transfer(
        self, requested_energy: float, is_charge: bool
    ) -> float:
        soc = self._storage.soc
        capacity = self._storage.capacity
        eff = self._config.efficiency
        max_power_energy = min(requested_energy, self._config.max_power)

        if is_charge:
            if soc >= 1.0:
                return 0.0
            available_capacity = (1.0 - soc) * capacity
            energy_to_store = min(max_power_energy * eff, available_capacity)
            self._storage = ElectricalStorage(
                soc=soc + energy_to_store / capacity, capacity=capacity
            )
            return energy_to_store / eff
        else:
            if soc <= 0.0:
                return 0.0
            energy_available = soc * capacity
            energy_to_release = min(max_power_energy / eff, energy_available)
            self._storage = ElectricalStorage(
                soc=soc - energy_to_release / capacity, capacity=capacity
            )
            return -energy_to_release * eff

    def _apply_degradation(self):
        """Apply degradation according to the configured mode."""
        rate = self._config.degradation_rate
        min_cap = self._initial_capacity * self._config.min_capacity_fraction
        capacity = self._initial_capacity

        if self._config.degradation_mode == "linear":
            degraded_capacity = capacity - rate * self._throughput

        elif self._config.degradation_mode == "exponential":
            degraded_capacity = capacity * (1.0 - rate) ** self._throughput

        elif self._config.degradation_mode == "polynomial":
            exponent = self._config.poly_exponent
            degraded_capacity = capacity - rate * (self._throughput**exponent)

        else:
            raise ValueError(
                f"Unknown degradation mode: {self._config.degradation_mode}"
            )

        degraded_capacity = max(min_cap, degraded_capacity)
        self._storage = ElectricalStorage(
            soc=self._storage.soc, capacity=degraded_capacity
        )

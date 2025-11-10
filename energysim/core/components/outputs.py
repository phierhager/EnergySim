from dataclasses import dataclass, field
from typing import Optional


# ------------------------
# Energy Domains
# ------------------------
@dataclass(frozen=True, slots=True, kw_only=True)
class ElectricalEnergy:
    demand_j: float = 0.0
    generation_j: float = 0.0
    losses_j: float = 0.0

    @property
    def net(self) -> float:
        return self.demand_j - self.generation_j

    def __add__(self, other: "ElectricalEnergy") -> "ElectricalEnergy":
        return ElectricalEnergy(
            demand_j=self.demand_j + other.demand_j,
            generation_j=self.generation_j + other.generation_j,
            losses_j=self.losses_j + other.losses_j,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ThermalEnergy:
    heating_j: float = 0.0
    cooling_j: float = 0.0

    @property
    def net_heating(self) -> float:
        return self.heating_j

    @property
    def net_cooling(self) -> float:
        return self.cooling_j

    def __add__(self, other: "ThermalEnergy") -> "ThermalEnergy":
        return ThermalEnergy(
            heating_j=self.heating_j + other.heating_j,
            cooling_j=self.cooling_j + other.cooling_j,
        )


# ------------------------
# Storage States
# ------------------------
@dataclass(frozen=True, slots=True, kw_only=True)
class ElectricalStorage:
    capacity: float = 0.0
    soc: float = 0.0  # 0..1

    def __post_init__(self):
        if not (0.0 <= self.soc <= 1.0):
            raise ValueError("State of Charge (soc) must be between 0 and 1.")
        if self.capacity <= 0.0:
            raise ValueError("Capacity must be greater than 0.")

    def __add__(self, other: "ElectricalStorage") -> "ElectricalStorage":
        return ElectricalStorage(
            capacity=self.capacity + other.capacity,
            soc=(self.soc * self.capacity + other.soc * other.capacity)
            / (self.capacity + other.capacity),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ThermalStorage:
    capacity: float = 0.0
    soc: float = 0.0  # 0..1

    def __post_init__(self):
        if not (0.0 <= self.soc <= 1.0):
            raise ValueError("State of Charge (soc) must be between 0 and 1.")
        if self.capacity <= 0.0:
            raise ValueError("Capacity must be greater than 0.")

    def __add__(self, other: "ThermalStorage") -> "ThermalStorage":
        if (self.capacity + other.capacity) == 0:
            raise ValueError("Cannot add ThermalStorage with zero total capacity.")
        return ThermalStorage(
            capacity=self.capacity + other.capacity,
            soc=(self.soc * self.capacity + other.soc * other.capacity)
            / (self.capacity + other.capacity)
            if (self.capacity + other.capacity) > 0
            else 0.0,
        )


# ------------------------
# Component Health / Limits
# ------------------------
@dataclass(frozen=True, slots=True, kw_only=True)
class ComponentStatus:
    """
    Generic component status for all HEMS components.
    Can include operational health, degradation, or performance limits.
    """

    cycle_count: Optional[int] = None  # For batteries, compressors
    capacity_fade: Optional[float] = None  # Fraction of nominal capacity lost
    efficiency: Optional[float] = None  # Current efficiency of the component
    internal_resistance: Optional[float] = None  # Battery-specific, optional
    operational_state: Optional[str] = None  # e.g., "idle", "charging", "heating"


# ------------------------
# Component Outputs
# ------------------------
@dataclass(frozen=True, slots=True, kw_only=True)
class ComponentOutputs:
    """
    Encapsulates outputs from any HEMS component.
    """

    electrical_energy: ElectricalEnergy = field(default_factory=ElectricalEnergy)
    thermal_energy: ThermalEnergy = field(default_factory=ThermalEnergy)
    electrical_storage: ElectricalStorage = field(default_factory=ElectricalStorage)
    thermal_storage: ThermalStorage = field(default_factory=ThermalStorage)
    # TODO: Add status and metadata if needed
    # status: Optional[ComponentStatus] = None
    # metadata: Dict = field(default_factory=dict)

    def __add__(self, other: "ComponentOutputs") -> "ComponentOutputs":
        return ComponentOutputs(
            electrical_energy=self.electrical_energy + other.electrical_energy,
            thermal_energy=self.thermal_energy + other.thermal_energy,
            electrical_storage=self.electrical_storage + other.electrical_storage,
            thermal_storage=self.thermal_storage + other.thermal_storage,
        )

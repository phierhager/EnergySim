# energysim/core/shared/control_variables.py
from enum import StrEnum

class StateKey(StrEnum):
    """Names for state variables tracked across the system."""
    ROOM_TEMP = "room_temp"
    BATTERY_SOC = "battery_soc"
    STORAGE_SOC = "storage_soc"

class ActionKey(StrEnum):
    """Names for control actions sent to components."""
    BATTERY_POWER_W = "battery_power_w"        # (W) > 0 for charging, < 0 for discharging
    HEAT_PUMP_POWER_W = "heat_pump_power_w"
    AC_POWER_W = "ac_power_w"
    STORAGE_DISCHARGE_W = "storage_discharge_w"  # (Thermal power from tank)

class ExoKey(StrEnum):
    """Names for exogenous data (from dataset/forecasts)."""
    AMBIENT_TEMP = "ambient_temp"
    LOAD = "load"            # (W) Non-controllable electrical load
    PV = "pv"                # (W) PV generation
    PRICE = "price"          # (€/kWh)
    INTERNAL_GAINS_W = "internal_gains_w"  # <--- NEW (e.g., people, computers)
    SOLAR_GAINS_W = "solar_gains_w"        # <--- NEW (e.g., direct sunlight)
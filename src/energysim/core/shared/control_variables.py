# energysim/core/shared/control_variables.py
from enum import StrEnum

class StateKey(StrEnum):
    """Names for state variables tracked across the system."""
    ROOM_TEMP = "room_temp"
    BATTERY_SOC = "battery_soc"
    STORAGE_SOC = "storage_soc"

class ActionKey(StrEnum):
    """Names for control actions sent to components."""
    BATTERY_POWER_W = "battery_power_w"       # (W) > 0 for charging, < 0 for discharging
    HEAT_PUMP_POWER_W = "heat_pump_power_w"
    AC_POWER_W = "ac_power_w"
    STORAGE_DISCHARGE_W = "storage_discharge_w"  # (Thermal power from tank)

class ExoKey(StrEnum):
    """Names for exogenous data (from dataset/forecasts)."""
    TIME = "timestamp"
    AMBIENT_TEMP = "ambient_temp"
    LOAD = "load"                         # (W) Non-controllable electrical load
    PRICE = "price"                       # (€/kWh)
    SOLAR_DNI_W_M2 = "solar_dni_w_m2"     # (W/m^2) Direct Normal Irradiance
    SOLAR_DHI_W_M2 = "solar_dhi_w_m2"     # (W/m^2) Diffuse Horizontal Irradiance
    WIND_SPEED_M_S = "wind_speed_m_s"           # (m/s) Wind speed for infiltration modeling
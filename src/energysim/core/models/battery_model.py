# High-Fidelity Multi-RC Battery Model (JAX + Equinox)
# Refactored: Vectorized RC interpolation, fully differentiable

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Sequence
from energysim.core.shared.data_structs import BatteryConfig, BatteryInputs, BatteryOutput, Array

class MultiRCBatteryModel(eqx.Module):
    """
    High-fidelity Li-ion battery model with:
    - 1 R0 (ohmic)
    - N RC pairs (R_i, C_i) capturing multiple relaxation timescales
    - SOC-dependent 1D grids for OCV/R0
    - SOC-dependent 2D grids for RC pairs (vectorized)
    - Thermal dynamics
    - CC → CV switching
    Fully differentiable for JAX optimization.
    """

    config: BatteryConfig

    # Lookup grids
    soc_grid: Array                # 1D SOC grid
    ocv_grid: Array                # 1D OCV grid
    r0_grid: Array                 # 1D R0 grid
    r_grid: Array                  # 2D RC resistances (len(soc) x n_rc)
    c_grid: Array                  # 2D RC capacitances (len(soc) x n_rc)

    # Thermal
    C_th: float
    R_th: float
    ambient_temp: float

    # Voltage & current constraints
    v_min: float
    v_max: float
    n_series: int
    max_charge_current_a: Optional[float]
    max_discharge_current_a: Optional[float]

    n_rc: int                       # number of RC branches

    def __init__(
        self,
        config: BatteryConfig,
        n_rc: int = 2,
        soc_grid: Optional[Array] = None,
        ocv_grid: Optional[Array] = None,
        r0_grid: Optional[Array] = None,
        r_grid: Optional[Array] = None,
        c_grid: Optional[Array] = None
    ):
        self.config = config
        self.n_rc = n_rc

        self.soc_grid = soc_grid if soc_grid is not None else jnp.linspace(0.0, 1.0, 51)
        self.n_series = getattr(config, "n_series", 110)

        self.ocv_grid = ocv_grid if ocv_grid is not None else (3.0 + 1.2 * (1 - jnp.exp(-5 * self.soc_grid))) * self.n_series
        self.r0_grid = r0_grid if r0_grid is not None else jnp.linspace(0.003, 0.02, len(self.soc_grid)) * self.n_series
        self.r_grid = r_grid if r_grid is not None else jnp.tile(self.r0_grid[:, None] * 0.5, (1, n_rc))
        self.c_grid = c_grid if c_grid is not None else jnp.full((len(self.soc_grid), n_rc), 500.0)

        # Thermal & limits
        self.C_th = getattr(config, "C_th", 5e4)
        self.R_th = getattr(config, "R_th", 0.5)
        self.ambient_temp = getattr(config, "ambient_temp", 298.15)
        self.v_min = getattr(config, "v_min", 300.0)
        self.v_max = getattr(config, "v_max", 430.0)
        self.max_charge_current_a = getattr(config, "max_charge_current_a", None)
        self.max_discharge_current_a = getattr(config, "max_discharge_current_a", None)

    # ------------------ INTERPOLATION ------------------ #
    def _interp_matrix(self, x: float, xp: Array, fp: Array) -> Array:
        """
        Interpolates 2D array 'fp' (len(xp) x n_rc) along axis 0 for scalar x.
        Returns vector of length n_rc.
        """
        return jax.vmap(lambda col: jnp.interp(x, xp, col), in_axes=1, out_axes=0)(fp)

    def _get_params(self, soc: float, temp: float):
        """
        Returns v_ocv, r0, r_i[], c_i[] for given SOC and temp.
        """
        soc_c = jnp.clip(soc, 0.0, 1.0)
        tref = getattr(self.config, "r_ref_temp", 298.15)
        temp_safe = jnp.maximum(temp, 1.0)
        temp_factor = jnp.exp(2000.0 * (1.0 / temp_safe - 1.0 / tref))

        v_ocv = jnp.interp(soc_c, self.soc_grid, self.ocv_grid)
        r0 = jnp.interp(soc_c, self.soc_grid, self.r0_grid) * temp_factor
        r_i = self._interp_matrix(soc_c, self.soc_grid, self.r_grid) * temp_factor
        c_i = self._interp_matrix(soc_c, self.soc_grid, self.c_grid)
        return v_ocv, r0, r_i, c_i

    # ------------------ AUXILIARY ------------------ #
    def _capacity_as(self) -> float:
        if hasattr(self.config, "capacity_ah") and self.config.capacity_ah is not None:
            return self.config.capacity_ah * 3600.0
        elif hasattr(self.config, "capacity_j") and self.config.capacity_j is not None:
            v_nom = float(self.ocv_grid[len(self.ocv_grid)//2])
            return self.config.capacity_j / v_nom
        else:
            return 180000.0

    def _solve_current_cp(self, p_req: float, v_eff: float, r0: float) -> float:
        D = v_eff**2 - 4.0 * r0 * p_req
        sqrt_D = jnp.sqrt(jnp.maximum(D, 0.0))
        i1 = (v_eff - sqrt_D) / (2.0 * r0)
        i2 = (v_eff + sqrt_D) / (2.0 * r0)
        i_root = jnp.where(jnp.sign(p_req) >= 0, jnp.maximum(i1, i2), jnp.minimum(i1, i2))
        return jnp.where(D >= 0, i_root, jnp.nan)

    def _apply_current_limits(self, i: float) -> float:
        if self.max_charge_current_a is not None:
            i = jnp.where(i < 0.0, jnp.maximum(i, -self.max_charge_current_a), i)
        if self.max_discharge_current_a is not None:
            i = jnp.where(i > 0.0, jnp.minimum(i, self.max_discharge_current_a), i)
        return i

    # ------------------ DYNAMICS ------------------ #
    def dynamics(self, t: float, state_vector: Array, inputs: BatteryInputs) -> Array:
        soc, temp = state_vector[0], state_vector[1]
        v_rc = state_vector[2:]

        v_ocv, r0, r_i, c_i = self._get_params(soc, temp)
        v_eff = v_ocv - jnp.sum(v_rc)

        i, _ = self._solve_circuit(inputs.power_w, v_eff, r0)

        dsoc_dt = -i / self._capacity_as()
        dvrc_dt = (i / c_i) - (v_rc / (r_i * c_i))
        q_heat = i**2 * r0 + jnp.sum(v_rc**2 / r_i)
        dtemp_dt = (q_heat - (temp - self.ambient_temp) / self.R_th) / self.C_th

        return jnp.concatenate([jnp.array([dsoc_dt, dtemp_dt]), dvrc_dt])

    # ------------------ OUTPUT ------------------ #
    def calculate_output(self, state_vector: Array, inputs: BatteryInputs) -> BatteryOutput:
        soc, temp = state_vector[0], state_vector[1]
        v_rc = state_vector[2:]

        v_ocv, r0, r_i, c_i = self._get_params(soc, temp)
        v_eff = v_ocv - jnp.sum(v_rc)

        i, v_term = self._solve_circuit(inputs.power_w, v_eff, r0)

        p_actual = v_term * i
        q_heat = i**2 * r0 + jnp.sum(v_rc**2 / r_i)

        return BatteryOutput(
            actual_power_w=p_actual,
            voltage_v=v_term,
            current_a=i,
            heat_generation_w=q_heat,
            soh=jnp.array(1.0)
        )

    def _solve_circuit(self, p_req: float, v_eff: float, r0: float):
        """
        Solves the algebraic constraint for current, handling 
        quadratic roots and voltage clamping (CC -> CV).
        """
        # 1. Constant Power Solve
        i_cp = self._solve_current_cp(p_req, v_eff, r0)
        
        # 2. Check for Voltage Collapses (Imaginary roots)
        is_collapsed = jnp.isnan(i_cp)
        i_safe = jnp.nan_to_num(i_cp)
        
        # 3. Check Voltage Limits
        v_term_cp = v_eff - i_safe * r0
        exceed_upper = v_term_cp > self.v_max
        exceed_lower = v_term_cp < self.v_min
        
        # 4. Determine Target Voltage (Saturation)
        # If collapsed, we default to v_min (saturation at bottom)
        v_target = jnp.where(exceed_upper, self.v_max,
                             jnp.where(exceed_lower | is_collapsed, self.v_min, v_term_cp))
        
        # 5. Calculate CV Current
        i_cv = (v_eff - v_target) / r0
        
        # 6. Switch Mode
        use_cv = is_collapsed | exceed_upper | exceed_lower
        i = jnp.where(use_cv, i_cv, i_cp)
        
        # 7. Apply Hard Current Limits
        i = self._apply_current_limits(i)
        
        # Return everything needed for downstream calcs
        v_term = v_eff - i * r0
        return i, v_term
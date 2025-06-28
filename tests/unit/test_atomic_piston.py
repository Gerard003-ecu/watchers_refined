import pytest
from unittest.mock import patch
# import numpy as np
# Not used directly in tests, can be removed if not needed by AtomicPiston implicitly

# Import from project structure
from atomic_piston.atomic_piston import (
    AtomicPiston, PistonMode
)

# Constants for testing
DEFAULT_CAPACITY = 100.0
DEFAULT_ELASTICITY = 10.0
DEFAULT_DAMPING = 1.0
DEFAULT_PISTON_MASS = 1.0
DT = 0.01  # Time step for updates

# Path for patching time.monotonic
TIME_PATCH_PATH = 'atomic_piston.atomic_piston.time.monotonic'


@pytest.fixture
def default_piston():
    """Proporciona una instancia de AtomicPiston con parámetros por defecto."""
    return AtomicPiston(
        capacity=DEFAULT_CAPACITY,
        elasticity=DEFAULT_ELASTICITY,
        damping=DEFAULT_DAMPING,
        piston_mass=DEFAULT_PISTON_MASS
    )


@pytest.fixture
def capacitor_piston():
    """Proporciona una instancia de AtomicPiston en modo CAPACITOR."""
    return AtomicPiston(
        capacity=DEFAULT_CAPACITY,
        elasticity=DEFAULT_ELASTICITY,
        damping=DEFAULT_DAMPING,
        piston_mass=DEFAULT_PISTON_MASS,
        mode=PistonMode.CAPACITOR
    )


@pytest.fixture
def battery_piston():
    """Proporciona una instancia de AtomicPiston en modo BATTERY."""
    return AtomicPiston(
        capacity=DEFAULT_CAPACITY,
        elasticity=DEFAULT_ELASTICITY,
        damping=DEFAULT_DAMPING,
        piston_mass=DEFAULT_PISTON_MASS,
        mode=PistonMode.BATTERY
    )


class TestAtomicPiston:
    """Agrupa pruebas unitarias para la clase AtomicPiston."""

    def test_initialization_default_params(
        self,
        default_piston: AtomicPiston
    ):
        """Verifica la inicialización con parámetros por defecto."""
        piston = default_piston
        assert piston.capacity == DEFAULT_CAPACITY
        assert piston.k == DEFAULT_ELASTICITY
        assert piston.c == DEFAULT_DAMPING
        assert piston.m == DEFAULT_PISTON_MASS
        assert piston.mode == PistonMode.CAPACITOR  # Default mode
        assert piston.position == 0.0
        assert piston.velocity == 0.0
        assert piston.last_applied_force == 0.0
        assert piston.current_charge == 0.0
        assert piston.capacitor_discharge_threshold == (
            -DEFAULT_CAPACITY * 0.9
        )
        assert not piston.battery_is_discharging
        assert piston.battery_discharge_rate == DEFAULT_CAPACITY * 0.05

    def test_initialization_custom_params(self):
        """Verifica la inicialización con parámetros personalizados."""
        custom_capacity = 200.0
        custom_elasticity = 5.0
        custom_damping = 0.5
        custom_mass = 2.0
        custom_mode = PistonMode.BATTERY
        piston = AtomicPiston(
            capacity=custom_capacity,
            elasticity=custom_elasticity,
            damping=custom_damping,
            piston_mass=custom_mass,
            mode=custom_mode
        )
        assert piston.capacity == custom_capacity
        assert piston.k == custom_elasticity
        assert piston.c == custom_damping
        assert piston.m == custom_mass
        assert piston.mode == custom_mode
        assert piston.current_charge == 0.0
        assert piston.capacitor_discharge_threshold == (
            -custom_capacity * 0.9
        )
        assert piston.battery_discharge_rate == custom_capacity * 0.05

    @patch(TIME_PATCH_PATH)
    def test_apply_force_significant_change(
        self,
        mock_monotonic_time,
        default_piston: AtomicPiston
    ):
        """Prueba apply_force con un cambio significativo en el valor de la señal."""
        piston = default_piston
        # First call to establish baseline
        mock_monotonic_time.return_value = 1.0
        piston.apply_force(signal_value=0.0, source="test_source")
        # If it's the first signal from a source, signal_velocity is 0, so force is 0.
        assert piston.last_applied_force == 0.0

        # Second call with significant change
        mock_monotonic_time.return_value = 1.0 + DT
        piston.apply_force(signal_value=10.0, source="test_source")

        # Calculation:
        # signal_velocity = (10.0 - 0.0) / 0.01 = 1000.0
        # force = -0.5 * piston_mass * (signal_velocity)^2
        # force = -0.5 * 1.0 * (1000.0)^2 = -500000.0
        expected_force = -0.5 * DEFAULT_PISTON_MASS * ((10.0 / DT) ** 2)
        assert piston.last_applied_force == pytest.approx(expected_force)

    @patch(TIME_PATCH_PATH)
    def test_apply_force_constant_signal(
        self,
        mock_monotonic_time,
        default_piston: AtomicPiston
    ):
        """Prueba apply_force con una señal de valor constante."""
        piston = default_piston
        mock_monotonic_time.return_value = 1.0
        piston.apply_force(signal_value=5.0, source="test_source")

        mock_monotonic_time.return_value = 1.0 + DT
        # Same signal value, so signal_velocity should be 0
        piston.apply_force(signal_value=5.0, source="test_source")

        assert piston.last_applied_force == 0.0  # Zero velocity means zero force

    @patch(TIME_PATCH_PATH)
    def test_apply_force_mass_factor_scaling(
        self,
        mock_monotonic_time,
        default_piston: AtomicPiston
    ):
        """Prueba que mass_factor escala correctamente la fuerza aplicada."""
        piston = default_piston
        mass_factor = 2.0

        mock_monotonic_time.return_value = 1.0
        piston.apply_force(signal_value=0.0, source="test_source",
                           mass_factor=mass_factor)

        mock_monotonic_time.return_value = 1.0 + DT
        piston.apply_force(signal_value=10.0, source="test_source",
                           mass_factor=mass_factor)

        base_signal_velocity_sq = (10.0 / DT) ** 2
        expected_force_no_mass_factor = (
            -0.5 * DEFAULT_PISTON_MASS * base_signal_velocity_sq
        )
        expected_force_with_mass_factor = (
            -0.5 * mass_factor * base_signal_velocity_sq
        )

        assert piston.last_applied_force == pytest.approx(
            expected_force_with_mass_factor
        )
        assert piston.last_applied_force == pytest.approx(  # noqa: E128
            expected_force_no_mass_factor * mass_factor
        )

    @patch(TIME_PATCH_PATH)
    def test_apply_force_dt_too_small(
        self,
        mock_monotonic_time,
        default_piston: AtomicPiston
    ):
        """Prueba apply_force cuando dt es demasiado pequeño para calcular velocidad."""
        piston = default_piston
        mock_monotonic_time.return_value = 1.0
        piston.apply_force(signal_value=0.0, source="test_source")

        # Next call with dt < 1e-6 (the threshold in apply_force)
        mock_monotonic_time.return_value = 1.0 + 1e-7
        piston.apply_force(signal_value=10.0, source="test_source")
        # signal_velocity should be 0 due to small dt
        assert piston.last_applied_force == 0.0

    def test_update_state_applies_force_and_compresses(
        self,
        default_piston: AtomicPiston
        ):
        """Verifica que update_state aplica fuerza y comprime el pistón."""
        piston = default_piston
        initial_position = piston.position

        # Simulate applying a force (directly setting it for this test,
        # as apply_force method is tested separately)
        piston.last_applied_force = -100.0

        piston.update_state(dt=DT)

        # Piston should compress (negative position)
        assert piston.position < initial_position
        assert piston.current_charge > 0
        assert piston.velocity != 0  # Piston should be moving
        # Force should be consumed after update
        assert piston.last_applied_force == 0.0

    def test_update_state_spring_force_effect(
        self
        default_piston: AtomicPiston
    ):
        """Verifica el efecto de la fuerza del resorte en update_state."""
        piston = default_piston
        piston.c = 0  # No damping for clarity
        initial_position = -10.0
        piston.position = initial_position
        initial_velocity = piston.velocity  # Should be 0

        piston.update_state(dt=DT)  # No external force, only spring force

        # Expected calculations:
        # spring_force = -k * position = -DEFAULT_ELASTICITY * (-10.0)
        # acceleration = spring_force / m = (10.0 * 10.0) / 1.0 = 100
        # velocity_change = acceleration * DT = 100 * 0.01 = 1.0
        # new_velocity = initial_velocity + velocity_change = 0 + 1.0 = 1.0
        # position_change = new_velocity * DT = 1.0 * 0.01 = 0.01
        # new_position = initial_position + position_change = -10.0 + 0.01 = -9.99

        assert piston.velocity > initial_velocity  # Spring pushes it back
        assert piston.position > initial_position  # Position moves towards 0
        assert piston.velocity == pytest.approx(1.0)
        assert piston.position == pytest.approx(-9.99)

    def test_update_state_damping_effect(
        self,
        default_piston: AtomicPiston
    ):
        """Verifica el efecto de la amortiguación en update_state."""
        piston = default_piston
        # Give it some initial compression and velocity as if it's oscillating
        piston.position = -10.0
        piston.velocity = 10.0
        initial_abs_velocity = abs(piston.velocity)

        velocities = []
        for _ in range(10):
            piston.update_state(dt=DT)
            velocities.append(abs(piston.velocity))
            # Stop if it overshoots and starts compressing again
            # with negative velocity (simplistic check for oscillation turn)
            if piston.velocity < 0 and piston.position < -1:
                break
        # Check that velocity magnitude generally decreases due to damping
        # This is a simplified check; true damping causes oscillation decay.
        assert velocities[-1] < initial_abs_velocity

    def test_update_state_position_max_capacity(self):
        """Verifica que la posición del pistón no exceda la capacidad máxima."""
        # Use custom piston for specific elasticity
        piston = AtomicPiston(
            capacity=DEFAULT_CAPACITY, elasticity=1.0, damping=0.0
        )
        piston.last_applied_force = -10000  # Large force

        for _ in range(200):  # Update many times to ensure capacity is reached
            piston.update_state(dt=DT)
            if piston.position == -DEFAULT_CAPACITY:
                break

        assert piston.position == -DEFAULT_CAPACITY  # Should not exceed -capacity
        assert piston.current_charge == DEFAULT_CAPACITY

        # Apply more force, should not change position further
        piston.last_applied_force = -10000
        piston.update_state(dt=DT)
        assert piston.position == -DEFAULT_CAPACITY

    # --- Capacitor Mode Tests ---
    def test_discharge_capacitor_above_threshold(
        self,
        capacitor_piston: AtomicPiston
    ):
        """Prueba descarga en modo capacitor cuando no se alcanza el umbral."""
        piston = capacitor_piston
        # Position is -50, threshold is -90 for DEFAULT_CAPACITY=100
        piston.position = -DEFAULT_CAPACITY * 0.5
        initial_charge = piston.current_charge
        assert initial_charge > 0

        output_signal = piston.discharge()

        assert output_signal is None
        # Position and charge should remain unchanged
        assert piston.position == -DEFAULT_CAPACITY * 0.5  # noqa: E128
        assert piston.current_charge == initial_charge

    def test_discharge_capacitor_at_threshold(
        self,
        capacitor_piston: AtomicPiston
    ):
        """Prueba descarga en modo capacitor cuando se está justo en el umbral."""
        piston = capacitor_piston
        # Position exactly at threshold
        piston.position = piston.capacitor_discharge_threshold
        initial_charge = piston.current_charge
        # For DEFAULT_CAPACITY=100, threshold is -90, charge is 90
        assert initial_charge == pytest.approx(DEFAULT_CAPACITY * 0.9)  # noqa: E128

        output_signal = piston.discharge()

        assert output_signal is not None
        assert output_signal["type"] == "pulse"
        assert output_signal["amplitude"] == pytest.approx(initial_charge)
        assert piston.position == 0.0  # Position resets
        assert piston.velocity == 2.0  # Specific velocity after discharge
        assert piston.current_charge == 0.0  # Charge depleted

    def test_discharge_capacitor_below_threshold(
        self,
        capacitor_piston: AtomicPiston
    ):
        """Prueba descarga en modo capacitor cuando se supera el umbral."""
        piston = capacitor_piston
        # Fully charged, well below threshold (e.g., -100 vs -90)
        piston.position = -DEFAULT_CAPACITY
        initial_charge = piston.current_charge
        assert initial_charge == DEFAULT_CAPACITY

        output_signal = piston.discharge()

        assert output_signal is not None
        assert output_signal["type"] == "pulse"
        assert output_signal["amplitude"] == pytest.approx(initial_charge)  # noqa: E128
        assert piston.position == 0.0
        assert piston.velocity == 2.0
        assert piston.current_charge == 0.0

    # --- Battery Mode Tests ---
    def test_discharge_battery_not_triggered(
        self,
        battery_piston: AtomicPiston
    ):
        """Prueba descarga en modo batería cuando no está activada la descarga."""
        piston = battery_piston
        piston.position = -DEFAULT_CAPACITY * 0.5  # Some charge
        assert piston.current_charge > 0
        assert not piston.battery_is_discharging  # Pre-condition

        output_signal = piston.discharge()

        assert output_signal is None
        assert not piston.battery_is_discharging

    def test_discharge_battery_triggered_but_no_charge(
        self,
        battery_piston: AtomicPiston
    ):
        """Prueba descarga en modo batería activada pero sin carga inicial."""
        piston = battery_piston
        piston.position = 0  # No charge
        piston.trigger_discharge(True)
        assert piston.battery_is_discharging  # Triggered

        output_signal = piston.discharge()

        # Current implementation returns None if current_charge is 0
        assert output_signal is None
        # Test asserts desired behavior: if triggered but empty, it should stop.
        # NOTE: This may require atomic_piston.py to set
        # self.battery_is_discharging = False if current_charge is 0  # noqa: E128
        # at the beginning of a triggered discharge call.
        assert not piston.battery_is_discharging

    def test_discharge_battery_triggered_with_charge(
        self,
        battery_piston: AtomicPiston
    ):
        """Prueba descarga en modo batería activada y con carga."""
        piston = battery_piston
        piston.position = -DEFAULT_CAPACITY * 0.5  # Some charge
        piston.trigger_discharge(True)
        assert piston.battery_is_discharging

        output_signal = piston.discharge()

        assert output_signal is not None
        assert output_signal["type"] == "sustained"  # noqa: E128
        assert output_signal["amplitude"] == 1.0
        # Should remain true while discharging and charge > 0
        assert piston.battery_is_discharging

    def test_discharge_battery_gradual_reduction(
            self,
            battery_piston: AtomicPiston
        ):
        """Prueba la reducción gradual de carga en modo batería."""
        piston = battery_piston
        # Charge the piston significantly
        piston.position = -DEFAULT_CAPACITY
        initial_charge = piston.current_charge
        assert initial_charge == DEFAULT_CAPACITY

        piston.trigger_discharge(True)

        charges = [initial_charge]
        positions = [piston.position]

        # NOTE: This test assumes that `AtomicPiston.discharge()` in battery mode
        # correctly uses `battery_discharge_rate`. The `atomic_piston.py` code
        # now uses a default `update_interval` of 0.01 if `UPDATE_INTERVAL`
        # is not defined.
        # We will use this same assumption in the test for consistency.
        # The `battery_discharge_rate` is defined as "position units per second".

        # Simulate multiple discharge ticks (e.g., 10 ticks)
        num_discharge_calls = 10

        # This is the update_interval that discharge() will use
        # if UPDATE_INTERVAL is not globally defined.
        update_interval_in_discharge = 0.01

        position_released_per_call = (
            piston.battery_discharge_rate * update_interval_in_discharge
        )

        # Ensure there's something to discharge to avoid infinite loops if rate is zero
        if position_released_per_call == 0 and piston.current_charge > 0:
            # If discharge rate or interval is zero, it will never discharge.
            # This can happen if capacity is zero, leading to battery_discharge_rate = 0
            pass

        for _ in range(num_discharge_calls):
            if not piston.battery_is_discharging or piston.current_charge == 0:
                break
            piston.discharge()
            charges.append(piston.current_charge)
            positions.append(piston.position)

        assert piston.current_charge < initial_charge
        assert piston.position > -DEFAULT_CAPACITY  # Position moved towards 0
        assert charges[-1] < charges[0]  # Charge decreased
        assert positions[-1] > positions[0]  # Position value increased

        # Continue discharging until empty
        # Calculate theoretical calls needed, add some buffer.
        # Uses the same `update_interval_in_discharge` as above.
        # position_released_per_call is already calculated above and remains the same.
        if position_released_per_call > 0:  # Avoid division by zero
            # Calculate calls needed based on the remaining charge at this point.
            # The charge is piston.current_charge (absolute value of position)
            # We need to release `piston.current_charge` amount of position.
            calls_to_empty_fully = (
                int(piston.current_charge / position_released_per_call) + 5
            )
        else:
            calls_to_empty_fully = 1  # If no discharge rate, no further emptying

        for _ in range(calls_to_empty_fully):  # Max attempts to empty
            if not piston.battery_is_discharging or piston.current_charge == 0:
                break
            piston.discharge()

        assert piston.current_charge == pytest.approx(0.0)
        assert piston.position == pytest.approx(0.0)
        assert not piston.battery_is_discharging

    # --- Mode Switching and Triggering ---
    def test_set_mode(self, default_piston: AtomicPiston):
        """Verifica el cambio de modo de operación y el reseteo de estados."""
        piston = default_piston
        assert piston.mode == PistonMode.CAPACITOR  # Default

        piston.set_mode(PistonMode.BATTERY)
        assert piston.mode == PistonMode.BATTERY
        # Should reset discharge state on mode change
        assert not piston.battery_is_discharging

        piston.trigger_discharge(True)  # Enable discharge in battery mode
        assert piston.battery_is_discharging

        piston.set_mode(PistonMode.CAPACITOR)  # Switch back
        assert piston.mode == PistonMode.CAPACITOR
        # Should reset discharge state
        assert not piston.battery_is_discharging

    def test_trigger_discharge_behavior(self, default_piston: AtomicPiston):
        """Verifica el comportamiento de trigger_discharge en diferentes modos."""
        piston = default_piston

        # In Capacitor mode (default)
        assert piston.mode == PistonMode.CAPACITOR
        piston.trigger_discharge(True)
        # Should not change battery_is_discharging in capacitor mode
        assert not piston.battery_is_discharging

        piston.trigger_discharge(False)
        assert not piston.battery_is_discharging  # Still false

        # Switch to Battery mode
        piston.set_mode(PistonMode.BATTERY)
        assert piston.mode == PistonMode.BATTERY
        assert not piston.battery_is_discharging  # Reset by set_mode

        piston.trigger_discharge(True)
        assert piston.battery_is_discharging

        piston.trigger_discharge(False)
        assert not piston.battery_is_discharging

    @patch(TIME_PATCH_PATH)
    def test_force_application_multiple_sources(
        self,
        mock_monotonic_time,
        default_piston: AtomicPiston
    ):
        """Prueba la aplicación de fuerza desde múltiples fuentes independientes."""
        piston = default_piston
        mock_monotonic_time.return_value = 1.0
        # Initial calls to establish history for both sources
        piston.apply_force(signal_value=0, source="source1")
        piston.apply_force(signal_value=0, source="source2")

        # Apply force from source1
        mock_monotonic_time.return_value = 1.0 + DT
        piston.apply_force(signal_value=10, source="source1")
        force1 = piston.last_applied_force
        assert force1 < 0  # Negative force due to positive velocity change

        # Apply force from source2, time advances for source2's dt calculation
        # Note: last_applied_force is overwritten by the latest call.
        # To test accumulation or combined effect, update_state would be needed.
        # This test correctly checks that apply_force reacts to the specified source.
        mock_monotonic_time.return_value = 1.0 + DT + DT  # Further advance time
        piston.apply_force(signal_value=5, source="source2")
        force2 = piston.last_applied_force
        assert force2 < 0

        # Detailed calculation for force comparison:
        # Source 1: value 0 at t=1.0, then value 10 at t=1.0+DT.
        #   dt_s1 = DT
        #   vel_s1 = (10 - 0) / DT = 10 / DT
        #   force1_calc = -0.5 * piston.m * (vel_s1 ** 2)
        # Source 2: value 0 at t=1.0, then value 5 at t=1.0+2*DT.
        #   (its previous 'source2' call was at t=1.0)
        #   dt_s2 = (1.0 + 2*DT) - 1.0 = 2*DT
        #   vel_s2 = (5 - 0) / (2*DT) = 5 / (2*DT)
        #   force2_calc = -0.5 * piston.m * (vel_s2 ** 2)
        # Example: force1_calc = -0.5 * 1.0 * (100 / (DT**2))
        #          force2_calc = -0.5 * 1.0 * (25 / (4 * DT**2))
        # abs(force1_calc) is clearly greater than abs(force2_calc).
        assert abs(force1) > abs(force2)  # Magnitudes should differ
        assert force1 != force2  # Forces should be different

        # Check if last_signal_info is updated correctly for both sources
        assert "source1" in piston.last_signal_info
        assert piston.last_signal_info["source1"]["value"] == 10
        assert piston.last_signal_info["source1"]["timestamp"] == 1.0 + DT

        assert "source2" in piston.last_signal_info
        assert piston.last_signal_info["source2"]["value"] == 5
        assert piston.last_signal_info["source2"]["timestamp"] == 1.0 + DT + DT

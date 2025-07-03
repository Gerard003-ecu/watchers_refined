import pytest
from unittest.mock import patch
import numpy as np

# Import from project structure
from atomic_piston.atomic_piston import (
    AtomicPiston, PistonMode, TransducerType
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

    def test_initialization_transducer_types(self):
        """Verifica la inicialización con diferentes TransducerTypes."""
        # Piezoelectric (default)
        piston_piezo = AtomicPiston(
            DEFAULT_CAPACITY, DEFAULT_ELASTICITY, DEFAULT_DAMPING,
            transducer_type=TransducerType.PIEZOELECTRIC
        )
        assert piston_piezo.transducer_type == TransducerType.PIEZOELECTRIC
        assert piston_piezo.voltage_sensitivity == 50.0
        assert piston_piezo.force_sensitivity == 0.02

        # Electrostatic
        piston_electro = AtomicPiston(
            DEFAULT_CAPACITY, DEFAULT_ELASTICITY, DEFAULT_DAMPING,
            transducer_type=TransducerType.ELECTROSTATIC
        )
        assert piston_electro.transducer_type == TransducerType.ELECTROSTATIC
        assert piston_electro.voltage_sensitivity == 100.0
        assert piston_electro.force_sensitivity == 0.01

        # Magnetostrictive
        piston_magneto = AtomicPiston(
            DEFAULT_CAPACITY, DEFAULT_ELASTICITY, DEFAULT_DAMPING,
            transducer_type=TransducerType.MAGNETOSTRICTIVE
        )
        assert piston_magneto.transducer_type == (
            TransducerType.MAGNETOSTRICTIVE
        )
        assert piston_magneto.voltage_sensitivity == 30.0
        assert piston_magneto.force_sensitivity == 0.05

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
        # If it's the first signal from a source, signal_velocity is 0,
        # so force is 0.
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

    def test_update_state_increases_stored_energy_on_compression(
        self,
        default_piston: AtomicPiston
    ):
        """Verifica que stored_energy aumenta cuando el pistón se comprime."""
        piston = default_piston
        initial_energy = piston.stored_energy
        assert initial_energy == 0.0  # Starts at rest

        # Apply a force to compress the piston
        piston.last_applied_force = -10.0  # Negative force to compress

        piston.update_state(dt=DT)

        # After update, position should be < 0, velocity might be non-zero
        assert piston.position < 0
        # Stored energy = 0.5*k*pos^2 + 0.5*m*vel^2
        # Both terms should be positive or zero.
        # Since position is non-zero, potential energy is > 0.
        # Kinetic energy could be zero if velocity is zero at peak compression,
        # or positive if still moving.
        assert piston.stored_energy > initial_energy

        # Let it oscillate a bit, energy should still be present
        piston.update_state(dt=DT)  # Force is consumed, now spring/damping act
        assert piston.stored_energy > 0  # Energy might change but still be positive

    def test_update_state_spring_force_effect(
        self,
        default_piston: AtomicPiston
    ):
        """Verifica el efecto de la fuerza del resorte en update_state."""
        piston = default_piston
        piston.c = 0  # No damping for clarity

        # Setup for Verlet: To start from pos=-10, vel=0 at t=0
        # position at t=0 is -10
        # position at t=-dt (previous_position) should be pos(0) - vel(0)*dt
        # = -10 - 0*dt = -10
        initial_pos = -10.0
        piston.position = initial_pos
        piston.previous_position = initial_pos  # For Verlet start from rest
        # Explicitly set, though not directly used by Verlet pos update if
        # prev_pos is set right
        piston.velocity = 0.0

        initial_velocity_val = piston.velocity  # Should be 0

        piston.update_state(dt=DT)  # No external force, only spring force

        # With Verlet integration, starting from pos=-10, prev_pos=-10, vel=0
        # (due to prev_pos setting):
        # spring_force = -k * position = -10 * (-10) = 100
        # acceleration = spring_force / m = 100 / 1 = 100
        # new_position = 2*(-10) - (-10) + 100*(0.01^2) = -20 + 10 + 0.01 = -9.99
        # new_velocity = (new_position - previous_position_before_update) / (2*dt)
        # new_velocity = (-9.99 - (-10)) / (2*0.01) = 0.01 / 0.02 = 0.5

        # Spring pushes it back, velocity becomes positive
        assert piston.velocity > initial_velocity_val
        assert piston.position > initial_pos  # Position moves towards 0
        assert piston.position == pytest.approx(-9.99)
        assert piston.velocity == pytest.approx(0.5)

    def test_update_state_damping_effect(self):
        """
        Verifica el efecto de la amortiguación en update_state comparando dos pistones.
        """
        initial_position = -10.0
        time_steps = 300  # Simulate for enough steps to observe peak velocity

        # Piston with low damping
        piston_low_damping = AtomicPiston(
            capacity=DEFAULT_CAPACITY,
            elasticity=DEFAULT_ELASTICITY,
            damping=0.1,  # Low damping
            piston_mass=DEFAULT_PISTON_MASS
        )
        piston_low_damping.position = initial_position

        # Piston with high damping
        piston_high_damping = AtomicPiston(
            capacity=DEFAULT_CAPACITY,
            elasticity=DEFAULT_ELASTICITY,
            damping=1.0,  # High damping
            piston_mass=DEFAULT_PISTON_MASS
        )
        piston_high_damping.position = initial_position

        max_velocity_low_damping = 0.0
        max_velocity_high_damping = 0.0

        for _ in range(time_steps):
            piston_low_damping.update_state(dt=DT)
            # We are interested in the velocity magnitude
            # as it moves back towards equilibrium
            # The first significant velocity will be
            # positive as it moves from -10 towards 0
            if piston_low_damping.velocity > max_velocity_low_damping:
                max_velocity_low_damping = piston_low_damping.velocity

            piston_high_damping.update_state(dt=DT)
            if piston_high_damping.velocity > max_velocity_high_damping:
                max_velocity_high_damping = piston_high_damping.velocity

        # Assert that the peak velocity of the high_damping piston is lower
        assert max_velocity_high_damping < max_velocity_low_damping
        # Also ensure that both pistons actually moved and achieved some velocity
        assert max_velocity_low_damping > 0
        assert max_velocity_high_damping > 0

    def test_update_state_position_max_capacity(self):
        """Verifica que la posición del pistón no exceda la capacidad máxima."""
        # Use custom piston for specific elasticity
        piston = AtomicPiston(
            capacity=DEFAULT_CAPACITY, elasticity=1.0, damping=0.0  # k=1
        )
        applied_force = -10000.0  # Large constant force

        # Loop enough times to ensure saturation is reached or equilibrium
        # with the large force. If k=1, then at position -110 (saturation),
        # spring force is 110. Applied force is -10000.
        # So it should definitely hit saturation.
        for _ in range(500):  # Increased iterations
            piston.last_applied_force = applied_force  # Re-apply force each step
            piston.update_state(dt=DT)
            # Check if it has reached or passed the saturation threshold
            if piston.position <= -piston.saturation_threshold:
                break

        # Position should be clipped at -saturation_threshold
        assert piston.position == pytest.approx(-piston.saturation_threshold)
        # Current charge is based on position, but capped effectively by
        # capacity in its definition current_charge = max(0, -position)
        # If position is -110, current_charge would be 110.
        # This seems fine as current_charge reflects the actual compression.
        assert piston.current_charge == pytest.approx(
            piston.saturation_threshold
        )

        # Apply more force, should not change position further due to clipping
        piston.last_applied_force = applied_force
        piston.update_state(dt=DT)
        assert piston.position == pytest.approx(-piston.saturation_threshold)

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

        output_signal = piston.discharge(dt=DT)  # Added dt=DT

        assert output_signal is None
        # Position and charge should remain unchanged
        assert piston.position == -DEFAULT_CAPACITY * 0.5
        assert piston.current_charge == initial_charge

    def test_discharge_capacitor_at_threshold(
        self,
        capacitor_piston: AtomicPiston
    ):
        """Prueba descarga en modo capacitor cuando se está justo en el umbral."""
        piston = capacitor_piston
        discharge_threshold = piston.capacitor_discharge_threshold
        hysteresis_position = (
            discharge_threshold * (1 + piston.hysteresis_factor)
        )

        piston.position = discharge_threshold
        initial_charge = piston.current_charge
        assert initial_charge == pytest.approx(abs(discharge_threshold))

        output_signal = piston.discharge(dt=DT)  # Pass dt

        assert output_signal is not None
        assert output_signal["type"] == "pulse"
        assert output_signal["amplitude"] == pytest.approx(initial_charge)
        # Position should bounce to hysteresis threshold
        assert piston.position == pytest.approx(hysteresis_position)
        assert piston.velocity == 5.0  # Specific velocity after discharge
        # Current charge should reflect the new position
        assert piston.current_charge == pytest.approx(
            max(0, -hysteresis_position)
        )

    def test_discharge_capacitor_below_threshold(
        self,
        capacitor_piston: AtomicPiston
    ):
        """Prueba descarga en modo capacitor cuando se supera el umbral."""
        piston = capacitor_piston
        discharge_threshold = piston.capacitor_discharge_threshold
        hysteresis_position = (
            discharge_threshold * (1 + piston.hysteresis_factor)
        )

        # Fully charged, well below threshold (e.g., -100 vs -90 for default)
        piston.position = -DEFAULT_CAPACITY
        initial_charge = piston.current_charge
        assert initial_charge == DEFAULT_CAPACITY

        output_signal = piston.discharge(dt=DT)  # Pass dt

        assert output_signal is not None
        assert output_signal["type"] == "pulse"
        assert output_signal["amplitude"] == pytest.approx(initial_charge)
        assert piston.position == pytest.approx(hysteresis_position)
        assert piston.velocity == 5.0
        assert piston.current_charge == pytest.approx(
            max(0, -hysteresis_position)
        )

    def test_capacitor_discharge_hysteresis_effect(
            self, capacitor_piston: AtomicPiston
    ):
        """Verifica el efecto de histéresis en la descarga del capacitor."""
        piston = capacitor_piston
        discharge_threshold = piston.capacitor_discharge_threshold  # e.g., -90
        # Hysteresis factor is 0.1 by default
        # hysteresis_position = -90 * (1 + 0.1) = -90 * 1.1 = -99 (Incorrect)
        # Hysteresis makes it *less* sensitive, so bounce should be to a
        # *less* negative value.
        # The code is: self.position = discharge_threshold * (1 - self.hysteresis_factor)
        # OR self.position = discharge_threshold * (1 + self.hysteresis_factor)
        # if threshold is negative.
        # Let's re-check atomic_piston.py:
        # hysteresis_threshold = discharge_threshold * (1 + self.hysteresis_factor)
        # self.position = hysteresis_threshold
        # If discharge_threshold = -90, hysteresis_factor = 0.1
        # hysteresis_threshold = -90 * (1 + 0.1) = -90 * 1.1 = -99.
        # This means it bounces *further* into compression if
        # hysteresis_factor is positive. This seems counter-intuitive for
        # "bouncing back". Let's assume the intention is to reduce
        # oscillation, so it should bounce to a position that is less
        # negative than the discharge_threshold.
        # The current code:
        # `hysteresis_threshold = discharge_threshold * (1 + self.hysteresis_factor)`
        # If discharge_threshold = -90, hysteresis_factor = 0.1,
        # then hysteresis_threshold = -99.
        # This means `self.position` becomes -99.
        # This would make it discharge again immediately if not careful.
        #
        # Re-reading the problem: "Verify that the piston position "bounces"
        # to a lower value (due to hysteresis)."
        # "Lower value" likely means less negative (closer to zero).
        # The current implementation
        # `discharge_threshold * (1 + self.hysteresis_factor)`
        # with negative `discharge_threshold` and positive `hysteresis_factor`
        # makes the `hysteresis_threshold` *more* negative.
        # Example: -90 * (1 + 0.1) = -99. This is not a bounce to a "lower"
        # (less compressed) state.
        #
        # The code in `atomic_piston.py` has:
        # `self.capacitor_discharge_threshold = -capacity * 0.9` (e.g., -90)
        # `self.hysteresis_factor = 0.1`
        # In `capacitor_discharge`:
        # `discharge_threshold = self.capacitor_discharge_threshold`
        # `hysteresis_threshold = discharge_threshold * (1 + self.hysteresis_factor)`
        # `self.position = hysteresis_threshold`
        # This means: position_after_discharge = -90 * (1 + 0.1) = -99.
        # This will cause it to re-trigger if
        # `self.position <= discharge_threshold` is checked.
        #
        # Given the problem statement "bounces to a lower value", it implies
        # less compression. This means the `hysteresis_threshold` should be
        # `discharge_threshold * (1 - self.hysteresis_factor)`
        # or `discharge_threshold + abs(discharge_threshold * self.hysteresis_factor)`.
        #
        # For now, I will test according to the *current code implementation*
        # and note this discrepancy. The current code's hysteresis makes it
        # *more* compressed after a pulse.

        expected_hysteresis_position = (
            discharge_threshold * (1 + piston.hysteresis_factor)
        )

        piston.position = discharge_threshold - 1  # Trigger discharge (e.g. -91)

        output = piston.discharge(dt=DT)
        assert output is not None
        assert piston.position == pytest.approx(expected_hysteresis_position)
        # If expected_hysteresis_position is -99 and discharge_threshold is -90,
        # then piston.position (-99) <= discharge_threshold (-90) is true.
        # This means it would immediately discharge again if discharge() is
        # called in a loop. This test verifies the implemented logic,
        # though the logic itself might be questioned.

        # To prevent immediate re-discharge, the bounce should be to a position P
        # such that P > discharge_threshold.
        # e.g., P = discharge_threshold * (1 - piston.hysteresis_factor)
        # P = -90 * (1 - 0.1) = -90 * 0.9 = -81.
        # Since -81 > -90, it would not immediately re-discharge.
        # The problem statement "bounces to a lower value" is ambiguous.
        # If "lower" means "less negative", then the current code is incorrect
        # for that interpretation.
        # If "lower" means "further from zero on the negative side",
        # then current code is correct.
        # I'll stick to testing the current code's behavior.
        # UPDATE: With corrected hysteresis logic,
        # expected_hysteresis_position will be LESS negative
        # e.g. -90 * (1 - 0.1) = -81.

        # Let's verify it *would NOT* discharge again if called immediately,
        # due to corrected hysteresis logic
        piston.velocity = 0  # Reset velocity to see if position alone triggers it
        output_again = piston.discharge(dt=DT)

        # With corrected hysteresis,
        # expected_hysteresis_position (-81) > discharge_threshold (-90).
        # So, it should NOT discharge again.
        if expected_hysteresis_position <= discharge_threshold:
            assert output_again is not None, (
                "Piston should re-discharge if hysteresis position is still "
                "at/beyond threshold"
            )
        else:
            assert output_again is None, (
                "Piston should NOT re-discharge due to corrected hysteresis"
            )
        assert output_again is None

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

        output_signal = piston.discharge(dt=DT)

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

        output_signal = piston.discharge(dt=DT)

        # Current implementation returns None if current_charge is 0
        assert output_signal is None
        # Test asserts desired behavior: if triggered but empty, it should stop.
        # NOTE: This may require atomic_piston.py to set
        # self.battery_is_discharging = False if current_charge is 0
        # at the beginning of a triggered discharge call.
        assert not piston.battery_is_discharging

    # --- Electronic Signal Application Tests ---

    def test_apply_electronic_signal_piezoelectric(
            self,
            default_piston: AtomicPiston
    ):
        """Verifica apply_electronic_signal para transductor piezoeléctrico."""
        piston = default_piston
        # Ensure it's Piezoelectric, or set it if fixture allows modification
        piston.transducer_type = TransducerType.PIEZOELECTRIC
        piston.voltage_sensitivity = 50.0  # V/m
        piston.force_sensitivity = 0.02   # N/V

        voltage = 10.0  # Volts
        piston.apply_electronic_signal(voltage)

        expected_force = voltage * piston.force_sensitivity
        assert piston.last_applied_force == pytest.approx(expected_force)

    def test_apply_electronic_signal_electrostatic(self):
        """Verifica apply_electronic_signal para transductor electrostático."""
        piston = AtomicPiston(
            DEFAULT_CAPACITY, DEFAULT_ELASTICITY, DEFAULT_DAMPING,
            transducer_type=TransducerType.ELECTROSTATIC
        )
        # voltage_sensitivity = 100.0, force_sensitivity = 0.01
        voltage = 20.0  # Volts
        piston.apply_electronic_signal(voltage)

        expected_force = voltage * piston.force_sensitivity
        assert piston.last_applied_force == pytest.approx(expected_force)

    def test_apply_electronic_signal_magnetostrictive(self):
        """Verifica apply_electronic_signal para transductor magnetostrictivo."""
        piston = AtomicPiston(
            DEFAULT_CAPACITY, DEFAULT_ELASTICITY, DEFAULT_DAMPING,
            transducer_type=TransducerType.MAGNETOSTRICTIVE
        )
        # voltage_sensitivity = 30.0, force_sensitivity = 0.05
        # L = m = 1.0, R = 1/c = 1/1.0 = 1.0
        # dt is not directly used in apply_electronic_signal for the first step,
        # but it's part of the piston's state if update_state was called.
        # Here, we test the direct effect of apply_electronic_signal.
        # The internal circuit_current calculation depends on dt if called over time.

        voltage = 15.0  # Volts
        initial_current = piston.circuit_current  # Should be 0

        # First call: circuit_current is 0
        # voltage_drop = 0 * R = 0
        # di_dt = (voltage - 0) / L = voltage / m
        # circuit_current += di_dt * piston.dt (piston.dt is 0.01 by default)
        # applied_force = circuit_current * force_sensitivity
        piston.apply_electronic_signal(voltage)

        # After first call:
        # di_dt = (15.0 - 0.0 * 1.0) / 1.0 = 15.0
        # piston.circuit_current = 0.0 + 15.0 * 0.01 = 0.15 A
        # expected_force = 0.15 * 0.05 = 0.0075 N
        expected_di_dt = (
            (voltage - initial_current * piston.equivalent_resistance) /
            piston.equivalent_inductance
        )
        expected_current = initial_current + expected_di_dt * piston.dt
        expected_force = expected_current * piston.force_sensitivity
        assert piston.circuit_current == pytest.approx(expected_current)
        assert piston.last_applied_force == pytest.approx(expected_force)

        # Second call to see current accumulation
        piston.apply_electronic_signal(voltage)
        # initial_current is now expected_current from previous step
        # voltage_drop = expected_current * R
        # di_dt = (voltage - voltage_drop) / L
        # new_circuit_current = expected_current + di_dt * piston.dt
        # new_expected_force = new_circuit_current * force_sensitivity
        second_di_dt = (
            (voltage - expected_current * piston.equivalent_resistance) /
            piston.equivalent_inductance
        )
        second_expected_current = expected_current + second_di_dt * piston.dt
        second_expected_force = (
            second_expected_current * piston.force_sensitivity
        )
        assert piston.circuit_current == pytest.approx(second_expected_current)
        # Forces accumulate
        assert piston.last_applied_force == pytest.approx(
            expected_force + second_expected_force
        )

    def test_discharge_battery_triggered_with_charge(
        self,
        battery_piston: AtomicPiston
    ):
        """Prueba descarga en modo batería activada y con carga."""
        piston = battery_piston
        piston.position = -DEFAULT_CAPACITY * 0.5  # Some charge
        piston.trigger_discharge(True)
        assert piston.battery_is_discharging

        output_signal = piston.discharge(dt=DT)

        assert output_signal is not None
        assert output_signal["type"] == "sustained"
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
            # This can happen if capacity is zero,
            # leading to battery_discharge_rate = 0
            pass

        for _ in range(num_discharge_calls):
            if not piston.battery_is_discharging or piston.current_charge == 0:
                break
            piston.discharge(dt=DT)  # Pass dt, using the test's DT
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
            piston.discharge(dt=DT)  # Pass dt

        assert piston.current_charge == pytest.approx(0.0, abs=1e-5)
        assert piston.position == pytest.approx(0.0, abs=1e-5)
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
        # Force after s1's dynamic contribution
        force1_accumulated = piston.last_applied_force
        # This is the first significant force, previous was 0.
        # force1_contribution = -0.5 * piston.m * ((10.0/DT)**2)
        force1_contribution = force1_accumulated
        assert force1_contribution < 0  # Negative force due to positive velocity change

        force_before_s2_dynamic_change = piston.last_applied_force

        # Apply force from source2, time advances for source2's dt calculation
        mock_monotonic_time.return_value = 1.0 + DT + DT  # Further advance time
        piston.apply_force(signal_value=5, source="source2")
        # Force after s1 and s2 dynamic contributions
        force2_accumulated = piston.last_applied_force

        # force2_contribution = -0.5 * piston.m * ((5.0/(2*DT))**2)
        force2_contribution = (
            force2_accumulated - force_before_s2_dynamic_change
        )

        # This specific contribution should also be negative
        assert force2_contribution < 0

        # Compare magnitudes of individual contributions
        # abs(force1_contribution) = |-0.5 * 1 * (10/0.01)^2| = |-500000| = 500000
        # abs(force2_contribution) = |-0.5*1*(5/(2*0.01))^2|
        #                          = |-0.5*(250)^2| = |-31250|=31250
        assert abs(force1_contribution) > abs(force2_contribution)
        assert force1_accumulated != force2_accumulated

        # Check if last_signal_info is updated correctly for both sources
        assert "source1" in piston.last_signal_info
        assert piston.last_signal_info["source1"][0] == 10     # Access by index
        assert piston.last_signal_info["source1"][1] == 1.0 + DT  # Access by index

        assert "source2" in piston.last_signal_info
        assert piston.last_signal_info["source2"][0] == 5      # Access by index
        assert piston.last_signal_info["source2"][1] == 1.0 + DT + DT

    # --- Bode Data Generation Tests ---

    def test_generate_bode_data_output_structure_and_length(
            self,
            default_piston: AtomicPiston
    ):
        """
        Verifica la estructura del output de generate_bode_data y la longitud de los arrays.
        """
        piston = default_piston
        frequency_range = np.array([10, 100, 1000, 10000])  # Hz

        bode_data = piston.generate_bode_data(frequency_range)

        # 1. Verify result is a dictionary
        assert isinstance(bode_data, dict), "Output should be a dictionary"

        # 2. Verify required keys
        assert "frequencies" in bode_data
        assert "magnitude" in bode_data
        assert "phase" in bode_data

        # 3. Verify array lengths
        input_len = len(frequency_range)
        assert len(bode_data["frequencies"]) == input_len, (
            "Frequencies array length mismatch"
        )
        assert len(bode_data["magnitude"]) == input_len, (
            "Magnitude array length mismatch"
        )
        assert len(bode_data["phase"]) == input_len, "Phase array length mismatch"

        # 4. Verify output types (basic check)
        assert isinstance(bode_data["frequencies"], (np.ndarray, list)), (
            "Frequencies should be array-like"
        )
        assert isinstance(bode_data["magnitude"], list), (
            "Magnitude should be a list (as per current implementation)"
        )
        assert isinstance(bode_data["phase"], list), (
            "Phase should be a list (as per current implementation)"
        )

        # 5. Verify frequencies array is the same as input
        assert np.array_equal(
            bode_data["frequencies"], frequency_range
        ), "Frequencies array should match input"

    def test_generate_bode_data_empty_frequency_range(
            self,
            default_piston: AtomicPiston
    ):
        """
        Verifica el comportamiento de generate_bode_data con un rango de frecuencias vacío.
        """
        piston = default_piston
        frequency_range = np.array([])

        bode_data = piston.generate_bode_data(frequency_range)

        assert isinstance(bode_data, dict)
        assert "frequencies" in bode_data
        assert "magnitude" in bode_data
        assert "phase" in bode_data

        assert len(bode_data["frequencies"]) == 0
        assert len(bode_data["magnitude"]) == 0
        assert len(bode_data["phase"]) == 0

    def test_generate_bode_data_values_sanity_check(
        self, default_piston: AtomicPiston
    ):
        """
        Realiza una comprobación de cordura básica de los valores de magnitud y fase.
        No se calculan valores exactos, pero se comprueba que no sean todos cero o NaN
        para un sistema físico razonable.
        """
        piston = default_piston  # k=10, c=1, m=1
        # Piezo: voltage_sensitivity = 50.0, force_sensitivity = 0.02
        # H_electrical = H_mech * voltage_sensitivity * force_sensitivity
        # H_electrical = H_mech * 50 * 0.02 = H_mech * 1.0
        # So, for piezo, electrical response magnitude/phase is same as mechanical.

        frequency_range = np.array([1, 10, 100])  # Frequencies away from DC

        bode_data = piston.generate_bode_data(frequency_range)

        magnitudes = bode_data["magnitude"]
        phases = bode_data["phase"]

        assert all(isinstance(m, float) for m in magnitudes), (
            "Magnitudes should be floats"
        )
        assert all(isinstance(p, float) for p in phases), "Phases should be floats"

        # For a typical 2nd order system, magnitude will not be zero everywhere
        # (unless gain is zero) and phase will vary.
        # Check that not ALL magnitudes are zero (or very close to zero)
        assert not all(abs(m) < 1e-9 for m in magnitudes), (
            "All magnitudes are zero, which is unlikely"
        )

        # Check for NaNs
        assert not any(np.isnan(m) for m in magnitudes), "NaN found in magnitudes"
        assert not any(np.isnan(p) for p in phases), "NaN found in phases"

        # Check if phase varies (it should for different frequencies in a dynamic system)
        if len(phases) > 1:
            assert not all(p == phases[0] for p in phases[1:]), (
                "Phase does not vary, which is unlikely for these frequencies"
            )

    # --- Reset Function Tests ---

    def test_reset_function(self, default_piston: AtomicPiston):
        """Verifica que la función reset restaura el estado inicial del pistón."""
        piston = default_piston

        # 1. Modificar el estado del pistón
        piston.last_applied_force = -50.0
        piston.update_state(dt=DT)  # Apply force, change position, velocity
        piston.update_state(dt=DT)  # Let it move a bit

        # Accumulate some electronic state
        piston.apply_electronic_signal(voltage=5.0)
        piston.update_state(dt=DT)  # Update state after electronic signal

        piston.charge_accumulated = 1.23  # Directly set for testing
        piston.last_signal_info["test_src"] = {"value": 1, "timestamp": 123}
        piston.energy_history.append(100)
        piston.efficiency_history.append(0.5)
        if piston.mode == PistonMode.BATTERY:
            piston.trigger_discharge(True)

        # Ensure state is indeed modified
        assert (piston.position != 0.0 or
                piston.velocity != 0.0 or
                piston.charge_accumulated != 0.0)
        assert piston.last_applied_force == 0.0  # This gets reset by update_state
        assert piston.circuit_voltage != 0.0 or piston.circuit_current != 0.0
        assert len(piston.last_signal_info) > 0
        assert len(piston.energy_history) > 0
        assert len(piston.efficiency_history) > 0
        if piston.mode == PistonMode.BATTERY:
            assert piston.battery_is_discharging

        # 2. Llamar a reset()
        piston.reset()

        # 3. Verificar que el estado se ha reseteado
        assert piston.position == 0.0
        assert piston.velocity == 0.0
        assert piston.acceleration == 0.0
        assert piston.previous_position == 0.0
        assert piston.last_applied_force == 0.0
        assert piston.circuit_voltage == 0.0
        assert piston.circuit_current == 0.0
        assert piston.charge_accumulated == 0.0
        # Should be reset regardless of original mode
        assert not piston.battery_is_discharging
        assert len(piston.last_signal_info) == 0
        assert len(piston.energy_history) == 0
        assert len(piston.efficiency_history) == 0
        # Default values from __init__
        assert piston.dt == 0.01  # Default dt is set in __init__

        # Check a few more specific items from reset logic
        original_mode = default_piston.mode  # Store original mode for fixture
        piston_battery = AtomicPiston(
            DEFAULT_CAPACITY, DEFAULT_ELASTICITY, DEFAULT_DAMPING,
            mode=PistonMode.BATTERY
        )
        piston_battery.trigger_discharge(True)
        assert piston_battery.battery_is_discharging
        piston_battery.reset()
        assert not piston_battery.battery_is_discharging
        # Restore fixture state if necessary, though test creates its own usually
        default_piston.mode = original_mode

    # --- Simulate Discharge Circuit Tests ---

    def test_simulate_discharge_circuit_basic_operation(
            self,
            default_piston: AtomicPiston
    ):
        """
        Verifica la operación básica de simulate_discharge_circuit.
        - Comprime el pistón para generar voltaje.
        - Simula la descarga a través de una resistencia de carga.
        - Verifica que la potencia disipada sea positiva.
        - Verifica que la posición del pistón se mueva hacia cero.
        """
        piston = default_piston
        load_resistance = 10.0  # Ohms

        # 1. Comprimir el pistón para generar voltaje
        # Apply force and update state to compress
        # Significant force to ensure compression
        piston.last_applied_force = -200.0
        piston.update_state(dt=DT)  # Compress
        piston.update_state(dt=DT)  # Settle a bit

        assert piston.position < 0, "Piston should be compressed"
        assert piston.circuit_voltage > 0, (
            "Circuit voltage should be generated due to compression"
        )

        initial_position = piston.position
        initial_stored_energy = piston.stored_energy
        initial_circuit_voltage = piston.circuit_voltage

        # 2. Llamar a simulate_discharge_circuit
        voltage_on_load, current_on_load, power_dissipated = (
            piston.simulate_discharge_circuit(
                load_resistance=load_resistance, dt=DT
            )
        )

        # 3. Verificar potencia disipada
        assert power_dissipated > 0, (
            "Power should be dissipated in the load resistor"
        )
        # Power = V_load * I_load = (I_load * R_load) * I_load = I_load^2 * R_load
        # I_load = V_circuit / (R_equiv + R_load)
        # V_circuit is based on -position * voltage_sensitivity
        expected_current = initial_circuit_voltage / (
            piston.equivalent_resistance + load_resistance
        )
        expected_power = expected_current**2 * load_resistance
        assert power_dissipated == pytest.approx(expected_power)
        assert voltage_on_load == pytest.approx(
            expected_current * load_resistance
        )
        assert current_on_load == pytest.approx(expected_current)

        # 4. Verificar que la posición del pistón se mueva hacia cero
        # (o sea, menos negativa, ya que la energía se disipa)
        assert piston.position > initial_position, (
            "Piston position should move towards zero (less negative)"
        )
        assert piston.position < 0, (
            "Piston should still be somewhat compressed or at zero"
        )

        # Verify stored energy decreased (mechanical energy converted and dissipated)
        # Note: update_electronic_state is called within simulate_discharge_circuit,
        # which updates circuit_voltage based on the *new* position.
        # So, the energy calculation should reflect this.
        assert piston.stored_energy < initial_stored_energy, (
            "Stored energy should decrease after dissipation"
        )

    def test_simulate_discharge_circuit_no_compression(
            self,
            default_piston: AtomicPiston
    ):
        """
        Verifica el comportamiento de simulate_discharge_circuit cuando no hay compresión.
        """
        piston = default_piston
        load_resistance = 10.0

        assert piston.position == 0.0
        assert piston.circuit_voltage == 0.0
        initial_position = piston.position

        voltage_on_load, current_on_load, power_dissipated = (
            piston.simulate_discharge_circuit(
                load_resistance=load_resistance, dt=DT
            )
        )

        assert power_dissipated == 0.0
        assert voltage_on_load == 0.0
        assert current_on_load == 0.0
        assert piston.position == initial_position  # No change in position

    def test_simulate_discharge_circuit_repeated_calls(
            self,
            default_piston: AtomicPiston
    ):
        """
        Verifica que múltiples llamadas a simulate_discharge_circuit continúan disipando energía.
        """
        piston = default_piston
        load_resistance = 5.0

        # Comprimir el pistón
        piston.last_applied_force = -100.0
        for _ in range(5):  # Compress over a few steps
            piston.update_state(dt=DT)

        assert piston.position < 0
        assert piston.circuit_voltage > 0

        last_position = piston.position
        total_power_dissipated = 0

        for i in range(10):  # Simulate discharge over several steps
            if piston.circuit_voltage <= 1e-3:  # Stop if voltage is negligible
                break

            _, _, power_dissipated = piston.simulate_discharge_circuit(
                load_resistance=load_resistance, dt=DT
            )
            assert power_dissipated >= 0  # Can be zero if voltage becomes zero
            total_power_dissipated += power_dissipated

            if power_dissipated > 0:
                assert piston.position > last_position, (
                    f"Position should improve on step {i}"
                )
            last_position = piston.position
            if piston.position >= 0:  # Piston returned to or past zero
                break

        assert total_power_dissipated > 0
        # Should not have gone more negative
        assert piston.position > -DEFAULT_CAPACITY
        # Should be close to zero after many dissipations
        assert piston.position == pytest.approx(0, abs=1e-1)

[end of tests/unit/test_atomic_piston.py]

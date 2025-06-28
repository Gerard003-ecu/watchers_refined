import pytest
import time
from unittest.mock import patch
import numpy as np

from watchers_refined.atomic_piston.atomic_piston import AtomicPiston, PistonMode

# Constants for testing
DEFAULT_CAPACITY = 100.0
DEFAULT_ELASTICITY = 10.0
DEFAULT_DAMPING = 1.0
DEFAULT_PISTON_MASS = 1.0
DT = 0.01  # Time step for updates


class TestAtomicPiston:

    def test_initialization_default_params(self):
        piston = AtomicPiston(capacity=DEFAULT_CAPACITY, elasticity=DEFAULT_ELASTICITY, damping=DEFAULT_DAMPING)
        assert piston.capacity == DEFAULT_CAPACITY
        assert piston.k == DEFAULT_ELASTICITY
        assert piston.c == DEFAULT_DAMPING
        assert piston.m == DEFAULT_PISTON_MASS
        assert piston.mode == PistonMode.CAPACITOR
        assert piston.position == 0.0
        assert piston.velocity == 0.0
        assert piston.last_applied_force == 0.0
        assert piston.current_charge == 0.0
        assert piston.capacitor_discharge_threshold == -DEFAULT_CAPACITY * 0.9
        assert not piston.battery_is_discharging
        assert piston.battery_discharge_rate == DEFAULT_CAPACITY * 0.05

    def test_initialization_custom_params(self):
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
        assert piston.capacitor_discharge_threshold == -custom_capacity * 0.9
        assert piston.battery_discharge_rate == custom_capacity * 0.05

    @patch('watchers_refined.atomic_piston.atomic_piston.time.monotonic')
    def test_apply_force_significant_change(self, mock_monotonic_time):
        piston = AtomicPiston(capacity=DEFAULT_CAPACITY, elasticity=DEFAULT_ELASTICITY, damping=DEFAULT_DAMPING)

        # First call to establish baseline
        mock_monotonic_time.return_value = 1.0
        piston.apply_force(signal_value=0.0, source="test_source")
        assert piston.last_applied_force == 0.0 # No velocity yet

        # Second call with significant change
        mock_monotonic_time.return_value = 1.0 + DT
        piston.apply_force(signal_value=10.0, source="test_source")

        # Expected force: -0.5 * 1.0 * ( (10.0 - 0.0) / DT )^2
        # signal_velocity = (10.0 - 0.0) / 0.01 = 1000.0
        # force = -0.5 * 1.0 * (1000.0)^2 = -500000.0
        expected_force = -0.5 * 1.0 * ((10.0 / DT) ** 2)
        assert piston.last_applied_force == pytest.approx(expected_force)

    @patch('watchers_refined.atomic_piston.atomic_piston.time.monotonic')
    def test_apply_force_constant_signal(self, mock_monotonic_time):
        piston = AtomicPiston(capacity=DEFAULT_CAPACITY, elasticity=DEFAULT_ELASTICITY, damping=DEFAULT_DAMPING)

        mock_monotonic_time.return_value = 1.0
        piston.apply_force(signal_value=5.0, source="test_source")

        mock_monotonic_time.return_value = 1.0 + DT
        piston.apply_force(signal_value=5.0, source="test_source") # Same value

        assert piston.last_applied_force == 0.0 # Zero velocity means zero force

    @patch('watchers_refined.atomic_piston.atomic_piston.time.monotonic')
    def test_apply_force_mass_factor_scaling(self, mock_monotonic_time):
        piston = AtomicPiston(capacity=DEFAULT_CAPACITY, elasticity=DEFAULT_ELASTICITY, damping=DEFAULT_DAMPING)
        mass_factor = 2.0

        mock_monotonic_time.return_value = 1.0
        piston.apply_force(signal_value=0.0, source="test_source", mass_factor=mass_factor)

        mock_monotonic_time.return_value = 1.0 + DT
        piston.apply_force(signal_value=10.0, source="test_source", mass_factor=mass_factor)

        expected_force_no_mass_factor = -0.5 * 1.0 * ((10.0 / DT) ** 2)
        expected_force_with_mass_factor = -0.5 * mass_factor * ((10.0 / DT) ** 2)

        assert piston.last_applied_force == pytest.approx(expected_force_with_mass_factor)
        assert piston.last_applied_force == pytest.approx(expected_force_no_mass_factor * mass_factor)

    @patch('watchers_refined.atomic_piston.atomic_piston.time.monotonic')
    def test_apply_force_dt_too_small(self, mock_monotonic_time):
        piston = AtomicPiston(capacity=DEFAULT_CAPACITY, elasticity=DEFAULT_ELASTICITY, damping=DEFAULT_DAMPING)
        mock_monotonic_time.return_value = 1.0
        piston.apply_force(signal_value=0.0, source="test_source")
        # Next call with almost the same time
        mock_monotonic_time.return_value = 1.0 + 1e-7
        piston.apply_force(signal_value=10.0, source="test_source")
        assert piston.last_applied_force == 0.0 # signal_velocity should be 0

    def test_update_state_applies_force_and_compresses(self):
        piston = AtomicPiston(capacity=DEFAULT_CAPACITY, elasticity=DEFAULT_ELASTICITY, damping=DEFAULT_DAMPING)
        initial_position = piston.position

        # Simulate applying a force (directly setting it for this test, apply_force is tested separately)
        piston.last_applied_force = -100.0

        piston.update_state(dt=DT)

        assert piston.position < initial_position # Piston should compress (negative position)
        assert piston.current_charge > 0
        assert piston.velocity != 0 # Piston should be moving
        assert piston.last_applied_force == 0.0 # Force should be consumed

    def test_update_state_spring_force_effect(self):
        piston = AtomicPiston(capacity=DEFAULT_CAPACITY, elasticity=DEFAULT_ELASTICITY, damping=0) # No damping for clarity

        # Manually compress the piston
        piston.position = -10.0
        initial_velocity = piston.velocity # Should be 0

        piston.update_state(dt=DT) # No external force, only spring force

        # Spring force = -k * position = -10 * (-10) = 100
        # Acceleration = 100 / 1.0 = 100
        # Velocity = 0 + 100 * DT = 1.0
        # Position = -10 + 1.0 * DT = -10 + 0.01 = -9.99
        assert piston.velocity > initial_velocity # Spring should push it back, positive velocity
        assert piston.position > -10.0 # Position should move towards 0

    def test_update_state_damping_effect(self):
        piston = AtomicPiston(capacity=DEFAULT_CAPACITY, elasticity=DEFAULT_ELASTICITY, damping=DEFAULT_DAMPING)

        # Give it some initial compression and velocity as if it's oscillating
        piston.position = -10.0
        piston.velocity = 10.0

        velocities = []
        for _ in range(10):
            piston.update_state(dt=DT)
            velocities.append(abs(piston.velocity))
            # Stop if it overshoots and starts compressing again with negative velocity
            if piston.velocity < 0 and piston.position < -1:
                break

        # Check that velocity magnitude generally decreases
        # This is a simplified check; true damping causes oscillation decay
        assert velocities[-1] < velocities[0]

    def test_update_state_position_max_capacity(self):
        piston = AtomicPiston(capacity=DEFAULT_CAPACITY, elasticity=1, damping=0) # Low elasticity to reach capacity
        piston.last_applied_force = -10000 # Large force

        for _ in range(100): # Update many times
            piston.update_state(dt=DT)
            if piston.position == -DEFAULT_CAPACITY:
                break

        assert piston.position == -DEFAULT_CAPACITY # Should not exceed -capacity
        assert piston.current_charge == DEFAULT_CAPACITY

        # Apply more force, should not change position further
        piston.last_applied_force = -10000
        piston.update_state(dt=DT)
        assert piston.position == -DEFAULT_CAPACITY

    # --- Capacitor Mode Tests ---
    def test_discharge_capacitor_above_threshold(self):
        piston = AtomicPiston(capacity=DEFAULT_CAPACITY, elasticity=DEFAULT_ELASTICITY, damping=DEFAULT_DAMPING, mode=PistonMode.CAPACITOR)
        piston.position = -DEFAULT_CAPACITY * 0.5 # Above threshold (-90)
        assert piston.current_charge > 0

        output_signal = piston.discharge()
        assert output_signal is None
        assert piston.position == -DEFAULT_CAPACITY * 0.5 # Position unchanged
        assert piston.current_charge == DEFAULT_CAPACITY * 0.5 # Charge unchanged

    def test_discharge_capacitor_at_threshold(self):
        piston = AtomicPiston(capacity=DEFAULT_CAPACITY, elasticity=DEFAULT_ELASTICITY, damping=DEFAULT_DAMPING, mode=PistonMode.CAPACITOR)
        piston.position = piston.capacitor_discharge_threshold # Exactly at threshold
        initial_charge = piston.current_charge
        assert initial_charge == DEFAULT_CAPACITY * 0.9

        output_signal = piston.discharge()

        assert output_signal is not None
        assert output_signal["type"] == "pulse"
        assert output_signal["amplitude"] == pytest.approx(initial_charge)
        assert piston.position == 0.0
        assert piston.velocity == 2.0 # Specific velocity after discharge
        assert piston.current_charge == 0.0

    def test_discharge_capacitor_below_threshold(self):
        piston = AtomicPiston(capacity=DEFAULT_CAPACITY, elasticity=DEFAULT_ELASTICITY, damping=DEFAULT_DAMPING, mode=PistonMode.CAPACITOR)
        piston.position = -DEFAULT_CAPACITY # Fully charged, well below threshold
        initial_charge = piston.current_charge
        assert initial_charge == DEFAULT_CAPACITY

        output_signal = piston.discharge()

        assert output_signal is not None
        assert output_signal["type"] == "pulse"
        assert output_signal["amplitude"] == pytest.approx(initial_charge)
        assert piston.position == 0.0
        assert piston.velocity == 2.0
        assert piston.current_charge == 0.0

    # --- Battery Mode Tests ---
    def test_discharge_battery_not_triggered(self):
        piston = AtomicPiston(capacity=DEFAULT_CAPACITY, elasticity=DEFAULT_ELASTICITY, damping=DEFAULT_DAMPING, mode=PistonMode.BATTERY)
        piston.position = -DEFAULT_CAPACITY * 0.5 # Some charge
        assert piston.current_charge > 0

        output_signal = piston.discharge()
        assert output_signal is None
        assert piston.battery_is_discharging is False

    def test_discharge_battery_triggered_but_no_charge(self):
        piston = AtomicPiston(capacity=DEFAULT_CAPACITY, elasticity=DEFAULT_ELASTICITY, damping=DEFAULT_DAMPING, mode=PistonMode.BATTERY)
        piston.position = 0 # No charge
        piston.trigger_discharge(True)
        assert piston.battery_is_discharging is True

        output_signal = piston.discharge()
        assert output_signal is None # Or could be a specific "empty" signal if designed that way
                                     # Current implementation returns None if current_charge is 0
        assert piston.battery_is_discharging is False # Should turn off if charge is 0

    def test_discharge_battery_triggered_with_charge(self):
        piston = AtomicPiston(capacity=DEFAULT_CAPACITY, elasticity=DEFAULT_ELASTICITY, damping=DEFAULT_DAMPING, mode=PistonMode.BATTERY)
        piston.position = -DEFAULT_CAPACITY * 0.5 # Some charge
        piston.trigger_discharge(True)
        assert piston.battery_is_discharging is True

        output_signal = piston.discharge()
        assert output_signal is not None
        assert output_signal["type"] == "sustained"
        assert output_signal["amplitude"] == 1.0
        assert piston.battery_is_discharging is True # Should remain true while discharging

    def test_discharge_battery_gradual_reduction(self):
        piston = AtomicPiston(capacity=DEFAULT_CAPACITY, elasticity=DEFAULT_ELASTICITY, damping=DEFAULT_DAMPING, mode=PistonMode.BATTERY)
        # Charge the piston significantly
        piston.position = -DEFAULT_CAPACITY
        initial_charge = piston.current_charge
        assert initial_charge == DEFAULT_CAPACITY

        piston.trigger_discharge(True)

        charges = [initial_charge]
        positions = [piston.position]

        # Simulate multiple update/discharge cycles
        # UPDATE_INTERVAL in AtomicPiston is 0.05
        # battery_discharge_rate = capacity * 0.05
        # position_released_per_discharge_call = capacity * 0.05 * 0.05 = capacity * 0.0025

        for i in range(10): # Simulate 10 discharge ticks
            # Note: In real usage, update_state would likely be called before discharge
            # to allow external forces, spring, damping to act.
            # Here, we focus on the discharge mechanism itself.
            # If update_state were called, it would interact with the position change from discharge.
            # For this test, we isolate discharge's effect on position.

            signal = piston.discharge()
            if signal is None and not piston.battery_is_discharging: # Discharged fully
                break

            # We are not calling update_state() here to isolate discharge effect.
            # The discharge() method itself modifies position in battery mode.
            charges.append(piston.current_charge)
            positions.append(piston.position)

        assert piston.current_charge < initial_charge
        assert piston.position > -DEFAULT_CAPACITY # Position moved towards 0
        assert charges[-1] < charges[0] # Charge decreased
        assert positions[-1] > positions[0] # Position value increased (less negative)

        # Continue discharging until empty
        while piston.current_charge > 0 and piston.battery_is_discharging:
            piston.discharge()
            # piston.update_state(0.05) # If we were simulating full loop

        assert piston.current_charge == 0
        assert piston.position == 0.0
        assert piston.battery_is_discharging is False

    # --- Mode Switching and Triggering ---
    def test_set_mode(self):
        piston = AtomicPiston(capacity=DEFAULT_CAPACITY, elasticity=DEFAULT_ELASTICITY, damping=DEFAULT_DAMPING)
        assert piston.mode == PistonMode.CAPACITOR

        piston.set_mode(PistonMode.BATTERY)
        assert piston.mode == PistonMode.BATTERY
        assert piston.battery_is_discharging is False # Should reset on mode change

        piston.trigger_discharge(True) # Enable discharge in battery mode
        assert piston.battery_is_discharging is True

        piston.set_mode(PistonMode.CAPACITOR) # Switch back
        assert piston.mode == PistonMode.CAPACITOR
        assert piston.battery_is_discharging is False # Should reset

    def test_trigger_discharge_behavior(self):
        piston = AtomicPiston(capacity=DEFAULT_CAPACITY, elasticity=DEFAULT_ELASTICITY, damping=DEFAULT_DAMPING)

        # In Capacitor mode
        assert piston.mode == PistonMode.CAPACITOR
        piston.trigger_discharge(True)
        assert piston.battery_is_discharging is False # Should not change in capacitor mode

        piston.trigger_discharge(False)
        assert piston.battery_is_discharging is False # Still false

        # Switch to Battery mode
        piston.set_mode(PistonMode.BATTERY)
        assert piston.mode == PistonMode.BATTERY

        piston.trigger_discharge(True)
        assert piston.battery_is_discharging is True

        piston.trigger_discharge(False)
        assert piston.battery_is_discharging is False

    @patch('watchers_refined.atomic_piston.atomic_piston.time.monotonic')
    def test_force_application_multiple_sources(self, mock_monotonic_time):
        piston = AtomicPiston(DEFAULT_CAPACITY, DEFAULT_ELASTICITY, DEFAULT_DAMPING)
        mock_monotonic_time.return_value = 1.0
        piston.apply_force(0, "source1")
        piston.apply_force(0, "source2")

        mock_monotonic_time.return_value = 1.0 + DT
        piston.apply_force(10, "source1") # This will set last_applied_force
        force1 = piston.last_applied_force
        assert force1 < 0 # Negative force due to positive velocity

        mock_monotonic_time.return_value = 1.0 + DT + DT # Advance time again for source2
        piston.apply_force(5, "source2") # This will overwrite last_applied_force
        force2 = piston.last_applied_force
        assert force2 < 0
        assert force1 != force2 # Different forces from different signals

        # Check if last_signal_info is updated correctly
        assert "source1" in piston.last_signal_info
        assert piston.last_signal_info["source1"]["value"] == 10
        assert "source2" in piston.last_signal_info
        assert piston.last_signal_info["source2"]["value"] == 5
```

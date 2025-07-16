# atomic_piston/atomic_piston.py
import time
import numpy as np
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PistonMode(Enum):
    CAPACITOR = "capacitor"
    BATTERY = "battery"


class TransducerType(Enum):
    PIEZOELECTRIC = "piezoelectric"
    ELECTROSTATIC = "electrostatic"
    MAGNETOSTRICTIVE = "magnetostrictive"


class FrictionModel(Enum):
    COULOMB = "coulomb"
    STRIBECK = "stribeck"
    VISCOUS = "viscous"


class ControllerType(Enum):
    PID = "pid"
    FUZZY = "fuzzy"


class AtomicPiston:
    """Simula una Unidad de Potencia Inteligente (IPU).

    Representa un sistema físico de un pistón atómico con una interfaz
    electrónica integrada, diseñado para sistemas fotovoltaicos.

    Attributes:
        capacity (float): Capacidad máxima de compresión del pistón en metros.
        k (float): Constante elástica del resorte (N/m).
        c (float): Coeficiente de amortiguación viscosa (N·s/m).
        m (float): Masa inercial del pistón (kg).
        mode (PistonMode): Modo de operación (capacitor o batería).
        transducer_type (TransducerType): Tipo de transductor utilizado.
        friction_model (FrictionModel): Modelo de fricción a aplicar.
        coulomb_friction (float): Coeficiente de fricción de Coulomb.
        stribeck_coeffs (tuple): Coeficientes para el modelo de Stribeck.
        nonlinear_elasticity (float): Coeficiente de no linealidad elástica.

    """

    def __init__(self,
                 capacity: float,
                 elasticity: float,
                 damping: float,
                 piston_mass: float = 1.0,
                 mode: PistonMode = PistonMode.CAPACITOR,
                 transducer_type: TransducerType = TransducerType.PIEZOELECTRIC,
                 friction_model: FrictionModel = FrictionModel.VISCOUS,
                 coulomb_friction: float = 0.2,
                 stribeck_coeffs: tuple = (0.3, 0.1, 0.05),
                 nonlinear_elasticity: float = 0.01):
        """Inicializa una nueva instancia de AtomicPiston.

        Args:
            capacity: Capacidad máxima de compresión del pistón (en metros).
            elasticity: Constante elástica del resorte (k en N/m).
            damping: Coeficiente de amortiguación viscosa (c en N·s/m).
            piston_mass: Masa inercial del pistón (m en kg).
            mode: Modo de operación inicial del pistón.
            transducer_type: Tipo de transductor.
            friction_model: Modelo de fricción a utilizar.
            coulomb_friction: Coeficiente de fricción de Coulomb.
            stribeck_coeffs: Coeficientes de Stribeck
                (f_static, f_coulomb, v_stribeck).
            nonlinear_elasticity: Coeficiente de no linealidad
                elástica.
        """
        # Parámetros físicos
        self.capacity = capacity
        self.k = elasticity
        self.c = damping
        self.m = piston_mass
        self.mode = mode
        self.transducer_type = transducer_type
        self.dt = 0.01  # Paso de tiempo predeterminado

        # Nuevos parámetros físicos
        self.friction_model = friction_model
        self.coulomb_friction = coulomb_friction
        self.stribeck_coeffs = stribeck_coeffs  # (f_static, f_coulomb, v_stribeck)
        self.nonlinear_elasticity = nonlinear_elasticity

        # Estado del pistón
        self.position = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0
        self.previous_position = 0.0
        self.last_applied_force = 0.0
        self.last_signal_info = {}

        # Parámetros de transducción
        self._set_transducer_params()

        # Circuito equivalente mejorado
        self.equivalent_capacitance = 1.0 / max(0.001, self.k)
        self.equivalent_inductance = self.m
        self.equivalent_resistance = 1.0 / max(0.001, self.c)
        self.output_capacitance = 0.1  # Capacitancia de salida (F)
        self.output_voltage = 0.0  # Voltaje de salida del convertidor (V)
        self.converter_efficiency = 0.95  # Eficiencia del convertidor

        # Estado electrónico
        self.circuit_voltage = 0.0
        self.circuit_current = 0.0
        self.charge_accumulated = 0.0

        # Parámetros de operación
        self.capacitor_discharge_threshold = -capacity * 0.9
        self.battery_is_discharging = False
        self.battery_discharge_rate = capacity * 0.05
        self.hysteresis_factor = 0.1
        self.saturation_threshold = capacity * 1.1
        self.compression_direction = -1

        # Controladores
        self.speed_controller = PIDController(kp=1.0, ki=0.1, kd=0.01)
        self.energy_controller = PIDController(kp=0.5, ki=0.05, kd=0.005)
        self.target_speed = 0.0
        self.target_energy = 0.0

        # Historial para diagnóstico
        self.energy_history = []
        self.efficiency_history = []
        self.friction_force_history = []

        logger.info(
            f"AtomicPiston inicializado en modo {mode.value} "
            f"con transductor {transducer_type.value} y fricción {friction_model.value}"
        )

    def _set_transducer_params(self):
        """Configura los parámetros del transductor según el tipo."""
        if self.transducer_type == TransducerType.PIEZOELECTRIC:
            self.voltage_sensitivity = 50.0
            self.force_sensitivity = 0.02
            self.internal_resistance = 100.0
        elif self.transducer_type == TransducerType.ELECTROSTATIC:
            self.voltage_sensitivity = 100.0
            self.force_sensitivity = 0.01
            self.internal_resistance = 500.0
        elif self.transducer_type == TransducerType.MAGNETOSTRICTIVE:
            self.voltage_sensitivity = 30.0
            self.force_sensitivity = 0.05
            self.internal_resistance = 50.0

    @property
    def current_charge(self) -> float:
        """Devuelve la carga actual del pistón.

        Returns:
            La carga actual, representada como la compresión del pistón.

        """
        return max(0, -self.position)

    @property
    def stored_energy(self) -> float:
        """Devuelve la energía mecánica total almacenada en el pistón.

        Returns:
            La energía mecánica total.

        """
        potential = (
            0.5 * self.k * self.position**2
            + (1 / 4) * self.nonlinear_elasticity * self.position**4
        )
        kinetic = 0.5 * self.m * self.velocity**2
        return potential + kinetic

    def calculate_friction(self) -> float:
        """Calcula la fuerza de fricción según el modelo seleccionado.

        Returns:
            La fuerza de fricción en Newtons.

        """
        if self.friction_model == FrictionModel.VISCOUS:
            return -self.c * self.velocity
        elif self.friction_model == FrictionModel.COUlOMB:
            if abs(self.velocity) < 1e-5:
                return -np.sign(self.last_applied_force) * min(
                    abs(self.last_applied_force), self.coulomb_friction
                )
            return -np.sign(self.velocity) * self.coulomb_friction
        elif self.friction_model == FrictionModel.STRIBECK:
            f_static, f_coulomb, v_stribeck = self.stribeck_coeffs
            if abs(self.velocity) < 1e-5:
                return -np.sign(self.last_applied_force) * min(
                    abs(self.last_applied_force), f_static
                )
            # Modelo de Stribeck:
            # F = F_c + (F_s - F_c) * exp(-(v/v_stribeck)^2)
            friction = f_coulomb + (f_static - f_coulomb) * np.exp(
                -((abs(self.velocity) / v_stribeck) ** 2)
            )
            return -np.sign(self.velocity) * friction
        return 0.0

    def apply_force(self, signal_value: float, source: str, mass_factor: float = 1.0):
        """Aplica una fuerza mecánica al pistón desde una señal de entrada.

        Args:
            signal_value: El valor de la señal de entrada.
            source: La fuente de la señal.
            mass_factor: El factor de masa.

        """
        current_time = time.monotonic()
        signal_velocity = 0.0
        if source in self.last_signal_info:
            last_val, last_time = self.last_signal_info[source]
            dt_signal = current_time - last_time
            if dt_signal > 1e-6:
                signal_velocity = (signal_value - last_val) / dt_signal
        self.last_signal_info[source] = (signal_value, current_time)
        force = self.compression_direction * 0.5 * mass_factor * (signal_velocity**2)
        self.last_applied_force += force
        logger.debug(f"Fuente '{source}': Fuerza aplicada = {force:.2f}N")

    def apply_electronic_signal(self, voltage: float):
        """Aplica una señal eléctrica que se traduce en fuerza mecánica.

        Args:
            voltage: El voltaje de la señal eléctrica.

        """
        applied_force = voltage * self.force_sensitivity
        if self.transducer_type == TransducerType.MAGNETOSTRICTIVE:
            # Simular un circuito RL
            voltage_drop = self.circuit_current * self.internal_resistance
            di_dt = (voltage - voltage_drop) / self.equivalent_inductance
            self.circuit_current += di_dt * self.dt
            applied_force = self.circuit_current * self.force_sensitivity
        self.last_applied_force += applied_force
        logger.debug(f"Señal eléctrica: {voltage:.2f}V → Fuerza: {applied_force:.2f}N")

    def update_state(self, dt: float):
        """Actualiza el estado físico del pistón para un intervalo de tiempo.

        Este método resuelve la ecuación diferencial del movimiento del pistón
        utilizando la integración de Verlet, un método numérico estable para
        sistemas de N-cuerpos.

        La ecuación diferencial es:
        m * d²x/dt² + F_friction(dx/dt) + k * x + ε * x³ = F_external(t)

        Donde:
        - m * d²x/dt² es el término de inercia.
        - F_friction(dx/dt) es la fuerza de fricción (viscosa, Coulomb o Stribeck).
        - k * x es la fuerza del resorte lineal (ley de Hooke).
        - ε * x³ es la fuerza del resorte no lineal.
        - F_external(t) es la suma de todas las fuerzas externas aplicadas.

        Args:
            dt: El intervalo de tiempo.
        """
        self.dt = dt

        # Aplicar control de velocidad si está activo
        if self.target_speed != 0.0:
            control_force = self.speed_controller.update(
                self.target_speed, self.velocity, dt
            )
            self.last_applied_force += control_force

        # Calcular fuerzas internas
        spring_force = (
            -self.k * self.position - self.nonlinear_elasticity * self.position**3
        )
        friction_force = self.calculate_friction()
        self.friction_force_history.append(friction_force)

        # Fuerza total = fuerzas externas + fuerzas internas
        total_force = self.last_applied_force + spring_force + friction_force

        # Calcular aceleración (a = F/m)
        self.acceleration = total_force / self.m

        # Integración de Verlet para calcular la nueva posición
        # x(t+dt) = 2x(t) - x(t-dt) + a(t) * dt²
        new_position = (
            2 * self.position - self.previous_position + self.acceleration * (dt**2)
        )

        # Actualizar velocidad
        # v(t) ≈ [x(t+dt) - x(t-dt)] / (2*dt)
        self.velocity = (new_position - self.previous_position) / (2 * dt)

        # Actualizar posiciones para la siguiente iteración
        self.previous_position = self.position
        self.position = new_position

        # Limitar la posición para evitar saturación
        self.position = np.clip(
            self.position, -self.saturation_threshold, self.saturation_threshold
        )

        # Actualizar estado electrónico y resetear fuerzas
        self.update_electronic_state()
        self.last_applied_force = 0.0

        # Registrar historial para diagnóstico
        self.energy_history.append(self.stored_energy)
        self.efficiency_history.append(self.get_conversion_efficiency())

    def update_electronic_state(self):
        """Actualiza el estado del circuito electrónico equivalente."""
        self.circuit_voltage = -self.position * self.voltage_sensitivity
        self.circuit_current = self.velocity * self.equivalent_capacitance
        self.charge_accumulated += self.circuit_current * self.dt
        self.process_electrical_output()

    def process_electrical_output(self):
        """Procesa la energía eléctrica a través del convertidor."""
        input_power = abs(self.circuit_voltage * self.circuit_current)
        output_power = input_power * self.converter_efficiency
        if self.equivalent_resistance > 0:
            self.output_voltage = np.sqrt(output_power * self.equivalent_resistance)
        else:
            self.output_voltage = 0.0

    def discharge(self, dt: float):
        """Gestiona la descarga de energía del pistón según el modo.

        Args:
            dt: El intervalo de tiempo.

        Returns:
            Un diccionario con la información de la descarga.

        """
        # Aplicar control de energía si está activo
        if self.target_energy > 0:
            discharge_rate = self.energy_controller.update(
                self.target_energy, self.stored_energy, dt
            )
            self.battery_discharge_rate = max(0.01, discharge_rate)
        if self.mode == PistonMode.CAPACITOR:
            return self.capacitor_discharge(dt)
        elif self.mode == PistonMode.BATTERY:
            return self.battery_discharge(dt)
        return None

    def capacitor_discharge(self, dt: float):
        """Realiza una descarga en modo capacitor si se cumplen las condiciones.

        Args:
            dt: El intervalo de tiempo.

        Returns:
            Un diccionario con la información de la descarga.

        """
        discharge_threshold = self.capacitor_discharge_threshold
        hysteresis_threshold = discharge_threshold * (1 - self.hysteresis_factor)
        if self.position <= discharge_threshold:
            amplitude = self.current_charge
            logger.info(f"¡Descarga CAPACITOR! Amplitud: {amplitude:.2f}")
            self.position = hysteresis_threshold
            self.velocity = 5.0
            return {"type": "pulse", "amplitude": amplitude, "duration": 0.001}
        return None

    def battery_discharge(self, dt: float):
        """Realiza una descarga en modo batería si hay carga.

        Args:
            dt: El intervalo de tiempo.

        Returns:
            Un diccionario con la información de la descarga.

        """
        if not self.battery_is_discharging:
            return None
        if self.current_charge > 0:
            max_discharge = self.current_charge * 0.8
            discharge_amount = min(self.battery_discharge_rate * dt, max_discharge)
            self.position += discharge_amount
            output_amplitude = discharge_amount / (self.battery_discharge_rate * dt)
            if self.current_charge <= 1e-5:
                self.battery_is_discharging = False
                logger.info("Descarga BATTERY: Carga agotada")
            return {
                "type": "sustained",
                "amplitude": output_amplitude,
                "duration": dt,
            }
        else:
            self.battery_is_discharging = False
            logger.info("Descarga BATTERY: Carga ya agotada")
            return None

    def simulate_discharge_circuit(self, load_resistance: float, dt: float):
        """Simula la descarga de energía a través de una carga externa.

        Args:
            load_resistance: La resistencia de la carga externa.
            dt: El intervalo de tiempo.

        Returns:
            Una tupla con el voltaje, la corriente y la potencia.

        """
        total_resistance = self.internal_resistance + load_resistance
        discharge_current = self.circuit_voltage / total_resistance
        discharge_energy = 0.5 * discharge_current**2 * load_resistance * dt
        if abs(self.position) > 1e-6:
            position_change = -discharge_energy / (self.k * self.position)
        else:
            position_change = 0.0
        if self.position < 0:
            self.position += position_change
            self.position = max(-self.capacity, self.position)
        self.update_electronic_state()
        load_voltage = discharge_current * load_resistance
        power_dissipated = load_voltage * discharge_current
        return load_voltage, discharge_current, power_dissipated

    def set_compression_direction(self, direction: int):
        """Configura la dirección de compresión del pistón.

        Args:
            direction: La dirección de compresión (-1 o 1).

        """
        if direction not in (-1, 1):
            logger.warning(
                f"Dirección inválida: {direction}. Usando -1 (compresión)."
            )
            self.compression_direction = -1
        else:
            self.compression_direction = direction

    def get_conversion_efficiency(self) -> float:
        """Calcula la eficiencia de conversión de energía instantánea.

        Returns:
            La eficiencia de conversión.

        """
        mechanical_energy = self.stored_energy
        electrical_energy = (
            0.5 * self.equivalent_capacitance * self.circuit_voltage**2
        )
        total_stored_energy = mechanical_energy + electrical_energy
        if total_stored_energy > 0:
            return mechanical_energy / total_stored_energy
        return 0.0

    def set_mode(self, mode: PistonMode):
        """Establece el modo de operación del pistón.

        Args:
            mode: El modo de operación a establecer.

        """
        self.mode = mode
        self.battery_is_discharging = False
        logger.info(f"Modo cambiado a: {mode.value}")

    def trigger_discharge(self, discharge_on: bool):
        """Activa o desactiva la descarga continua en modo BATTERY.

        Args:
            discharge_on: El estado de la descarga.

        """
        if self.mode == PistonMode.BATTERY:
            self.battery_is_discharging = discharge_on
            if discharge_on:
                logger.info("Descarga BATTERY activada.")
            else:
                logger.info("Descarga BATTERY desactivada.")
        else:
            logger.warning(
                f"trigger_discharge llamado en modo {self.mode.value}. "
                f"Solo válido en modo BATTERY."
            )

    def set_speed_target(self, target: float):
        """Establece la velocidad objetivo para el controlador.

        Args:
            target: La velocidad objetivo.

        """
        self.target_speed = target
        self.speed_controller.reset()
        logger.info(f"Objetivo de velocidad establecido: {target} m/s")

    def set_energy_target(self, target: float):
        """Establece la energía objetivo para el controlador.

        Args:
            target: La energía objetivo.

        """
        self.target_energy = target
        self.energy_controller.reset()
        logger.info(f"Objetivo de energía establecido: {target} J")

    def generate_bode_data(self, frequency_range: np.ndarray) -> dict:
        """Genera datos para un diagrama de Bode.

        Args:
            frequency_range: El rango de frecuencias.

        Returns:
            Un diccionario con los datos de magnitud y fase.

        """
        magnitude = []
        phase = []
        for f in frequency_range:
            omega = 2 * np.pi * f
            H_mech = 1 / (
                self.m * (1j * omega) ** 2 + self.c * (1j * omega) + self.k
            )
            H_electrical = (
                H_mech * self.voltage_sensitivity * self.force_sensitivity
            )
            magnitude.append(20 * np.log10(np.abs(H_electrical)))
            phase.append(np.angle(H_electrical, deg=True))
        return {"frequencies": frequency_range, "magnitude": magnitude, "phase": phase}

    def simulate_step_response(self, force_amplitude: float, duration: float, dt: float) -> dict:
        """
        Simula la respuesta del pistón a una fuerza escalón.

        Aplica una fuerza constante y registra la evolución del estado del sistema.

        Args:
            force_amplitude: Magnitud de la fuerza a aplicar (en Newtons).
            duration: Duración de la simulación (en segundos).
            dt: Paso de tiempo para la simulación (en segundos).

        Returns:
            Un diccionario con las series temporales de posición, velocidad y aceleración.
        """
        self.reset()
        time_series = np.arange(0, duration, dt)
        position_history = []
        velocity_history = []
        acceleration_history = []

        for t in time_series:
            self.last_applied_force = force_amplitude
            self.update_state(dt)
            position_history.append(self.position)
            velocity_history.append(self.velocity)
            acceleration_history.append(self.acceleration)

        return {
            "time": time_series,
            "position": position_history,
            "velocity": velocity_history,
            "acceleration": acceleration_history,
        }

    def simulate_impulse_response(self, impulse_magnitude: float, duration: float, dt: float) -> dict:
        """
        Simula la respuesta del pistón a una fuerza impulso.

        Aplica una fuerza instantánea y registra la evolución del estado del sistema.

        Args:
            impulse_magnitude: Magnitud del impulso a aplicar (en N·s).
            duration: Duración de la simulación (en segundos).
            dt: Paso de tiempo para la simulación (en segundos).

        Returns:
            Un diccionario con las series temporales de posición, velocidad y aceleración.
        """
        self.reset()
        time_series = np.arange(0, duration, dt)
        position_history = []
        velocity_history = []
        acceleration_history = []

        # Aplicar el impulso como un cambio instantáneo en la velocidad
        self.velocity += impulse_magnitude / self.m

        for t in time_series:
            self.update_state(dt)
            position_history.append(self.position)
            velocity_history.append(self.velocity)
            acceleration_history.append(self.acceleration)

        return {
            "time": time_series,
            "position": position_history,
            "velocity": velocity_history,
            "acceleration": acceleration_history,
        }

    def reset(self):
        """Reinicia el estado del pistón a sus condiciones iniciales."""
        self.position = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0
        self.previous_position = 0.0
        self.last_applied_force = 0.0
        self.circuit_voltage = 0.0
        self.circuit_current = 0.0
        self.charge_accumulated = 0.0
        self.output_voltage = 0.0
        self.battery_is_discharging = False
        self.last_signal_info = {}
        self.energy_history = []
        self.efficiency_history = []
        self.friction_force_history = []
        self.speed_controller.reset()
        self.energy_controller.reset()
        logger.info("Estado del pistón reiniciado")

    def get_differential_equation_terms(self) -> dict:
        """
        Devuelve los términos de la ecuación diferencial ordinaria de segundo
        grado que gobierna el movimiento del pistón.

        La ecuación es de la forma:
        m * d²x/dt² + F_friction(dx/dt) + k * x + ε * x³ = F_external(t)

        Returns:
            Un diccionario con los componentes de la ecuación diferencial.
        """
        friction_force = self.calculate_friction()
        spring_force = -self.k * self.position
        nonlinear_spring_force = -self.nonlinear_elasticity * self.position**3
        total_force = self.last_applied_force + spring_force + nonlinear_spring_force + friction_force

        return {
            "mass_term": self.m * self.acceleration,
            "damping_force": friction_force,
            "spring_force": spring_force,
            "nonlinear_spring_force": nonlinear_spring_force,
            "external_force": self.last_applied_force,
            "total_force": total_force
        }


class PIDController:
    """Implementa un controlador PID.

    Controlador Proporcional-Integral-Derivativo para la regulación de
    velocidad y energía en el sistema del pistón atómico.

    Attributes:
        kp (float): Ganancia proporcional.
        ki (float): Ganancia integral.
        kd (float): Ganancia derivativa.
        integral (float): Acumulador para el término integral.
        previous_error (float): Error registrado en la última actualización.
        output_limit (float): Límite de la salida para anti-windup.

    """
    def __init__(self, kp: float, ki: float, kd: float):
        """Inicializa el controlador PID.

        Args:
            kp: Ganancia proporcional.
            ki: Ganancia integral.
            kd: Ganancia derivativa.

        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.previous_error = 0.0
        self.output_limit = 10.0

    def update(self, setpoint: float, current_value: float, dt: float) -> float:
        """Calcula la salida del controlador PID.

        Args:
            setpoint: El valor deseado.
            current_value: El valor actual del sistema.
            dt: El paso de tiempo.

        Returns:
            La señal de control calculada.

        """
        error = setpoint - current_value
        p_term = self.kp * error
        self.integral += error * dt
        i_term = self.ki * self.integral
        d_term = self.kd * (error - self.previous_error) / dt
        output = p_term + i_term + d_term
        if output > self.output_limit:
            output = self.output_limit
            self.integral -= error * dt
        elif output < -self.output_limit:
            output = -self.output_limit
            self.integral -= error * dt
        self.previous_error = error
        return output

    def reset(self):
        """Reinicia el estado interno del controlador."""
        self.integral = 0.0
        self.previous_error = 0.0

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


class AtomicPiston:
    """
    Unidad de Potencia Inteligente (IPU) que simula un sistema físico
    de un pistón atómico con interfaz electrónica integrada para
    sistemas fotovoltaicos.

    Este modelo combina:
    - Simulación física del pistón (masa-resorte-amortiguador)
    - Interfaz electrónica para conversión mecánico-eléctrica
    - Modos de operación (capacitor/batería)
    - Gestión de energía con supercondensadores e inductores
    """

    def __init__(self,
                 capacity: float,
                 elasticity: float,
                 damping: float,
                 piston_mass: float = 1.0,
                 mode: PistonMode = PistonMode.CAPACITOR,
                 transducer_type: TransducerType = TransducerType.PIEZOELECTRIC):
        """
        Inicializa una nueva instancia de AtomicPiston.

        Args:
            capacity: Capacidad máxima de compresión del pistón (en metros).
            elasticity: Constante elástica del resorte (k en N/m).
            damping: Coeficiente de amortiguación (c en N·s/m).
            piston_mass: Masa inercial del pistón (m en kg).
            mode: Modo de operación (CAPACITOR o BATTERY).
            transducer_type: Tipo de transductor para interfaz electrónica.
        """
        # Parámetros físicos
        self.capacity = capacity
        self.k = elasticity
        self.c = damping
        self.m = piston_mass
        self.mode = mode
        self.transducer_type = transducer_type
        self.dt = 0.01  # Paso de tiempo predeterminado

        # Estado del pistón
        self.position = 0.0  # Posición actual (metros)
        self.velocity = 0.0  # Velocidad actual (m/s)
        self.acceleration = 0.0  # Aceleración actual (m/s²)
        self.previous_position = 0.0  # Posición anterior (para integración Verlet)
        self.last_applied_force = 0.0  # Última fuerza aplicada (N)
        self.last_signal_info = {}  # Almacena información de señales anteriores

        # Parámetros de transducción
        if transducer_type == TransducerType.PIEZOELECTRIC:
            self.voltage_sensitivity = 50.0  # V/m (sensibilidad de posición a voltaje)
            self.force_sensitivity = 0.02  # N/V (sensibilidad de voltaje a fuerza)
        elif transducer_type == TransducerType.ELECTROSTATIC:
            self.voltage_sensitivity = 100.0
            self.force_sensitivity = 0.01
        elif transducer_type == TransducerType.MAGNETOSTRICTIVE:
            self.voltage_sensitivity = 30.0
            self.force_sensitivity = 0.05

        # Circuito equivalente
        self.equivalent_capacitance = 1.0 / max(0.001, self.k)  # C = 1/k
        self.equivalent_inductance = self.m                      # L = m
        self.equivalent_resistance = 1.0 / max(0.001, self.c)    # R = 1/c

        # Estado electrónico
        self.circuit_voltage = 0.0  # Voltaje en el circuito equivalente (V)
        self.circuit_current = 0.0  # Corriente en el circuito equivalente (A)
        self.charge_accumulated = 0.0  # Carga acumulada (C)

        # Parámetros de operación
        # Umbral de descarga para modo capacitor
        self.capacitor_discharge_threshold = -capacity * 0.9
        self.battery_is_discharging = False  # Estado de descarga para modo batería
        # Tasa de descarga en modo batería
        self.battery_discharge_rate = capacity * 0.05
        # Factor de histéresis para evitar ciclado rápido
        self.hysteresis_factor = 0.1
        self.saturation_threshold = capacity * 1.1  # Límite de saturación
        self.compression_direction = -1  # -1: comprimir, 1: expandir

        # Historial para diagnóstico
        self.energy_history = []
        self.efficiency_history = []

        logger.info(
            f"AtomicPiston inicializado en modo {mode.value} "
            f"con transductor {transducer_type.value}"
        )

    @property
    def current_charge(self) -> float:
        """Devuelve la carga actual del pistón (compresión)"""
        return max(0, -self.position)

    @property
    def stored_energy(self) -> float:
        """Calcula la energía almacenada (potencial + cinética)"""
        potential = 0.5 * self.k * self.position**2
        kinetic = 0.5 * self.m * self.velocity**2
        return potential + kinetic

    def apply_force(self, signal_value: float, source: str, mass_factor: float = 1.0):
        """
        Aplica una fuerza mecánica al pistón basada en una señal de entrada.

        Args:
            signal_value: Valor actual de la señal
            source: Identificador de la fuente de señal
            mass_factor: Factor para escalar la 'masa' efectiva de la señal
        """
        current_time = time.monotonic()

        # Cálculo de velocidad de señal (derivada)
        signal_velocity = 0.0
        if source in self.last_signal_info:
            last_val, last_time = self.last_signal_info[source]
            dt_signal = current_time - last_time
            if dt_signal > 1e-6:  # Evitar división por cero
                signal_velocity = (signal_value - last_val) / dt_signal

        # Guardar información actual para el próximo cálculo
        self.last_signal_info[source] = (signal_value, current_time)

        # Calcular fuerza con dirección configurable
        # F = 0.5 * masa * velocidad² (energía cinética convertida a fuerza)
        force = self.compression_direction * 0.5 * mass_factor * (signal_velocity ** 2)
        self.last_applied_force += force

        logger.debug(f"Fuente '{source}': Fuerza aplicada = {force:.2f}N")

    def apply_electronic_signal(self, voltage: float):
        """
        Aplica una señal eléctrica al sistema, traduciéndola en fuerza mecánica.

        Args:
            voltage: Voltaje aplicado al transductor (V)
        """
        # Para transductores piezoeléctricos y electrostáticos
        applied_force = voltage * self.force_sensitivity

        # Para transductor magnetostrictivo
        if self.transducer_type == TransducerType.MAGNETOSTRICTIVE:
            # Simular un circuito RL simple
            voltage_drop = self.circuit_current * self.equivalent_resistance
            di_dt = (voltage - voltage_drop) / self.equivalent_inductance
            self.circuit_current += di_dt * self.dt
            applied_force = self.circuit_current * self.force_sensitivity

        self.last_applied_force += applied_force
        logger.debug(f"Señal eléctrica: {voltage:.2f}V → Fuerza: {applied_force:.2f}N")

    def update_state(self, dt: float):
        """
        Actualiza el estado físico del pistón usando integración Verlet.

        Args:
            dt: Paso de tiempo para la simulación (s)
        """
        self.dt = dt

        # Calcular fuerzas internas
        spring_force = -self.k * self.position  # Fuerza del resorte (Ley de Hooke)
        damping_force = -self.c * self.velocity  # Fuerza de amortiguación

        # Fuerza total = fuerzas externas + fuerzas internas
        total_force = self.last_applied_force + spring_force + damping_force

        # Calcular aceleración (F = ma → a = F/m)
        self.acceleration = total_force / self.m

        # Integración Verlet para mejor precisión
        # x(t+dt) = 2x(t) - x(t-dt) + a(t) * dt²
        new_position = (
            2 * self.position -
            self.previous_position +
            self.acceleration * (dt ** 2)
        )

        # Actualizar velocidad (v = [x(t+dt) - x(t-dt)] / (2*dt))
        self.velocity = (new_position - self.previous_position) / (2 * dt)

        # Actualizar posiciones
        self.previous_position = self.position
        self.position = new_position

        # Limitar saturación (evitar sobrecompresión/extensión excesiva)
        self.position = np.clip(
            self.position,
            -self.saturation_threshold,
            self.saturation_threshold
        )

        # Actualizar estado electrónico (conversión mecánico-eléctrica)
        self.update_electronic_state()

        # Resetear fuerza externa después de aplicarla
        self.last_applied_force = 0.0

        # Registrar energía y eficiencia
        self.energy_history.append(self.stored_energy)
        self.efficiency_history.append(self.get_conversion_efficiency())

    def update_electronic_state(self):
        """Actualiza el estado del circuito equivalente basado en el estado mecánico"""
        # Voltaje proporcional a la compresión (posición negativa)
        self.circuit_voltage = -self.position * self.voltage_sensitivity

        # Corriente proporcional a la velocidad del pistón
        self.circuit_current = self.velocity * self.equivalent_capacitance

        # Actualizar carga acumulada (integral de corriente)
        self.charge_accumulated += self.circuit_current * self.dt

    def discharge(self, dt: float):
        """
        Gestiona la descarga de energía del pistón según su modo de operación.

        Returns:
            dict or None: Señal de salida si ocurre una descarga, None en caso contrario
        """
        if self.mode == PistonMode.CAPACITOR:
            return self.capacitor_discharge(dt)
        elif self.mode == PistonMode.BATTERY:
            return self.battery_discharge(dt)
        return None

    def capacitor_discharge(self, dt: float):
        """Descarga en modo capacitor (pulso instantáneo)"""
        discharge_threshold = self.capacitor_discharge_threshold
        # Corrected hysteresis: bounces to a *less* compressed state
        hysteresis_threshold = discharge_threshold * (1 - self.hysteresis_factor)

        if self.position <= discharge_threshold:
            amplitude = self.current_charge
            logger.info(f"¡Descarga CAPACITOR! Amplitud: {amplitude:.2f}")

            # Aplicar histéresis para evitar ciclado rápido
            self.position = hysteresis_threshold
            self.velocity = 5.0  # Simular rebote

            # Crear señal de pulso
            return {
                "type": "pulse",
                "amplitude": amplitude,
                "duration": 0.001  # Pulso muy corto
            }
        return None

    def battery_discharge(self, dt: float):
        """Descarga en modo batería (señal sostenida)"""
        if not self.battery_is_discharging:
            return None

        if self.current_charge > 0:
            # Descarga proporcional a la carga actual
            max_discharge = self.current_charge * 0.8  # Máximo 80% por paso
            discharge_amount = min(self.battery_discharge_rate * dt, max_discharge)

            self.position += discharge_amount

            # Señal proporcional a la tasa de descarga
            # Ensure dt is not zero if rate is non-zero
            output_amplitude = discharge_amount / (self.battery_discharge_rate * dt)

            # Check if charge is now depleted after this step
            # Threshold for effective depletion (aligned with test approx)
            if self.current_charge <= 1e-5:
                self.battery_is_discharging = False
                logger.info(
                    "Descarga BATTERY: Carga agotada después del paso de descarga."
                )

            return {
                "type": "sustained",
                "amplitude": output_amplitude,
                "duration": dt
            }
        else:  # current_charge is already <= 0 at the beginning of the call
            self.battery_is_discharging = False
            logger.info("Descarga BATTERY: Carga ya estaba agotada.")
            return None

    def simulate_discharge_circuit(self, load_resistance: float, dt: float):
        """
        Simula la descarga del pistón en un circuito eléctrico con carga.

        Args:
            load_resistance: Resistencia de carga (ohmios)
            dt: Paso de tiempo para la simulación (s)

        Returns:
            tuple: (voltaje_en_carga, corriente_en_carga, potencia_disipada)
        """
        # Calcular resistencia total (interna + carga)
        total_resistance = self.equivalent_resistance + load_resistance

        # Calcular corriente de descarga (Ley de Ohm)
        discharge_current = self.circuit_voltage / total_resistance

        # Calcular energía disipada en la carga
        discharge_energy = 0.5 * discharge_current**2 * load_resistance * dt

        # Calcular cambio de posición proporcional a la energía disipada
        # (Suponiendo que la energía mecánica se convierte en eléctrica)
        # dE = k * x * dx  => dx = dE / (k * x)
        # If energy is lost from system (dE = -discharge_energy for system):
        # dx = -discharge_energy / (k * x)
        if abs(self.position) > 1e-6:  # Evitar división por cero
            position_change = -discharge_energy / (self.k * self.position)
        else:
            position_change = 0.0

        # Aplicar cambio de posición solo si hay compresión
        if self.position < 0:
            self.position += position_change
            # Asegurar que no exceda la capacidad máxima
            self.position = max(-self.capacity, self.position)

        # Actualizar estado electrónico después de la descarga
        self.update_electronic_state()

        # Calcular parámetros de salida
        load_voltage = discharge_current * load_resistance
        power_dissipated = load_voltage * discharge_current

        return load_voltage, discharge_current, power_dissipated

    def set_compression_direction(self, direction: int):
        """
        Configura la dirección de compresión del pistón.

        Args:
            direction: -1 para compresión (predeterminado), 1 para expansión
        """
        if direction not in (-1, 1):
            logger.warning(
                f"Dirección de compresión inválida: {direction}. "
                "Usando -1 (compresión)."
            )
            self.compression_direction = -1
        else:
            self.compression_direction = direction

    def get_conversion_efficiency(self) -> float:
        """
        Calcula la eficiencia de conversión de energía del sistema.

        Returns:
            float: Eficiencia como valor entre 0 y 1
        """
        # Energía mecánica almacenada (potencial + cinética)
        mechanical_energy = self.stored_energy

        # Energía eléctrica almacenada (en el circuito equivalente)
        electrical_energy = 0.5 * self.equivalent_capacitance * self.circuit_voltage**2

        # Calcular eficiencia (considerar pérdidas)
        total_stored_energy = mechanical_energy + electrical_energy
        if total_stored_energy > 0:
            return mechanical_energy / total_stored_energy
        return 0.0

    def set_mode(self, mode: PistonMode):
        """
        Establece el modo de operación del pistón.

        Args:
            mode: Nuevo modo de operación (PistonMode.CAPACITOR o PistonMode.BATTERY)
        """
        self.mode = mode
        self.battery_is_discharging = False
        logger.info(f"Modo del pistón cambiado a: {mode.value}")

    def trigger_discharge(self, discharge_on: bool):
        """
        Activa o desactiva la descarga continua en modo BATTERY.

        Args:
            discharge_on: True para activar la descarga, False para desactivarla
        """
        if self.mode == PistonMode.BATTERY:
            self.battery_is_discharging = discharge_on
            if discharge_on:
                logger.info("Descarga en modo BATTERY activada.")
            else:
                logger.info("Descarga en modo BATTERY desactivada.")
        else:
            logger.warning(
                f"trigger_discharge llamado en modo {self.mode.value}. "
                f"Solo tiene efecto en modo BATTERY."
            )

    def generate_bode_data(self, frequency_range: np.ndarray) -> dict:
        """
        Genera datos para un diagrama de Bode del sistema pistón-electrónica.

        Args:
            frequency_range: Array de frecuencias a evaluar (Hz)

        Returns:
            dict: Datos para el diagrama de Bode
        """
        magnitude = []
        phase = []

        for f in frequency_range:
            omega = 2 * np.pi * f

            # Función de transferencia del sistema mecánico
            H_mech = 1 / (
                self.m * (1j*omega)**2 + self.c * (1j*omega) + self.k
            )

            # Convertir a respuesta eléctrica
            H_electrical = (
                H_mech * self.voltage_sensitivity * self.force_sensitivity
            )

            magnitude.append(20 * np.log10(np.abs(H_electrical)))
            phase.append(np.angle(H_electrical, deg=True))

        return {
            'frequencies': frequency_range,
            'magnitude': magnitude,
            'phase': phase
        }

    def reset(self):
        """Reinicia el estado del pistón a condiciones iniciales"""
        self.position = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0
        self.previous_position = 0.0
        self.last_applied_force = 0.0
        self.circuit_voltage = 0.0
        self.circuit_current = 0.0
        self.charge_accumulated = 0.0
        self.battery_is_discharging = False
        self.last_signal_info = {}
        self.energy_history = []
        self.efficiency_history = []
        logger.info("Estado del pistón reiniciado")

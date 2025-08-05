# atomic_piston/atomic_piston_v2.py
"""
Simulación de una Unidad de Potencia Inteligente (IPU) o Pistón Atómico.

Este módulo define las clases y enumeraciones necesarias para simular el
comportamiento de un pistón atómico, incluyendo su física, electrónica
equivalente y sistemas de control.

Clases:
    PistonMode: Modos de operación (CAPACITOR, BATTERY).
    TransducerType: Tipos de transductor (PIEZOELECTRIC, etc.).
    FrictionModel: Modelos de fricción (COULOMB, STRIBECK, VISCOUS).
    ControllerType: Tipos de controlador (PID, FUZZY).
    AtomicPiston: La clase principal que simula el pistón.
    PIDController: Un controlador PID genérico.
"""
import time
import numpy as np
from enum import Enum
import logging
import csv
from typing import List, Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class PistonMode(Enum):
    """Define los modos de operación del pistón."""
    CAPACITOR = "capacitor"
    BATTERY = "battery"


class TransducerType(Enum):
    """Define los tipos de transductores que puede usar el pistón."""
    PIEZOELECTRIC = "piezoelectric"
    ELECTROSTATIC = "electrostatic"
    MAGNETOSTRICTIVE = "magnetostrictive"


class FrictionModel(Enum):
    """Define los modelos de fricción disponibles."""
    COULOMB = "coulomb"
    STRIBECK = "stribeck"
    VISCOUS = "viscous"


class ControllerType(Enum):
    """Define los tipos de controladores que se pueden emplear."""
    PID = "pid"
    FUZZY = "fuzzy"


class AtomicPiston:
    """
    Simula una Unidad de Potencia Inteligente (IPU) o Pistón Atómico.

    Esta clase modela el comportamiento físico y electrónico de un pistón
    diseñado para almacenar y liberar energía. Es el componente central para
    simulaciones que involucran almacenamiento de energía mecánica y su
    conversión a energía eléctrica.

    Attributes:
        capacity (float): Capacidad máxima de compresión/expansión [m].
        k (float): Constante elástica del resorte [N/m].
        c (float): Coeficiente de amortiguación viscosa [N·s/m].
        m (float): Masa inercial del pistón [kg].
        mode (PistonMode): Modo de operación (CAPACITOR o BATTERY).
        transducer_type (TransducerType): Tipo de transductor utilizado.
        friction_model (FrictionModel): Modelo de fricción a aplicar.
        position (float): Posición actual del pistón [m].
        velocity (float): Velocidad actual del pistón [m/s].
        acceleration (float): Aceleración actual del pistón [m/s²].

    Public Methods:
        apply_force: Aplica una fuerza mecánica externa.
        apply_electronic_signal: Aplica una señal eléctrica (voltaje).
        update_state: Avanza la simulación un paso de tiempo `dt`.
        discharge: Gestiona la descarga de energía según el modo.
        reset: Reinicia el estado del pistón a sus condiciones iniciales.
        export_history_to_csv: Exporta los datos de la simulación a un CSV.
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
                 stribeck_coeffs: Tuple[float, float, float] = (0.3, 0.1, 0.05),
                 nonlinear_elasticity: float = 0.01) -> None:
        """
        Inicializa una nueva instancia de AtomicPiston.

        Args:
            capacity: Capacidad máxima de compresión [m]. Debe ser > 0.
            elasticity: Constante elástica del resorte (k) [N/m]. Debe ser >= 0.
            damping: Coeficiente de amortiguación (c) [N·s/m]. Debe ser >= 0.
            piston_mass: Masa inercial del pistón (m) [kg]. Debe ser > 0.
            mode: Modo de operación inicial del pistón.
            transducer_type: Tipo de transductor a utilizar.
            friction_model: Modelo de fricción a simular.
            coulomb_friction: Coeficiente de fricción de Coulomb.
            stribeck_coeffs: Tupla con coeficientes de Stribeck (estática,
                Coulomb, velocidad).
            nonlinear_elasticity: Coeficiente para el término de elasticidad
                no lineal.

        Raises:
            ValueError: Si un parámetro físico no es válido (p. ej., negativo).
        """
        # -- Validación de Entradas --
        if capacity <= 0:
            raise ValueError("La capacidad (capacity) debe ser un valor positivo.")
        if elasticity < 0:
            raise ValueError("La elasticidad (elasticity) no puede ser negativa.")
        if damping < 0:
            raise ValueError("El amortiguamiento (damping) no puede ser negativo.")
        if piston_mass <= 0:
            raise ValueError("La masa del pistón (piston_mass) debe ser un valor positivo.")

        # -- Parámetros Físicos --
        self.capacity: float = capacity
        self.k: float = elasticity
        self.c: float = damping
        self.m: float = piston_mass
        self.mode: PistonMode = mode
        self.transducer_type: TransducerType = transducer_type
        self.dt: float = 0.01  # Paso de tiempo predeterminado

        # -- Parámetros de Fricción y No Linealidad --
        self.friction_model: FrictionModel = friction_model
        self.coulomb_friction: float = coulomb_friction
        self.stribeck_coeffs: Tuple[float, float, float] = stribeck_coeffs
        self.nonlinear_elasticity: float = nonlinear_elasticity

        # -- Estado Dinámico del Pistón --
        self.position: float = 0.0
        self.velocity: float = 0.0
        self.acceleration: float = 0.0
        self.previous_position: float = 0.0
        self.last_applied_force: float = 0.0
        self.last_signal_info: Dict[str, Tuple[float, float]] = {}

        # -- Parámetros de Transducción --
        self._set_transducer_params()

        # -- Circuito Eléctrico Equivalente --
        self.equivalent_capacitance: float = 1.0 / max(0.001, self.k)
        self.equivalent_inductance: float = self.m
        self.equivalent_resistance: float = 1.0 / max(0.001, self.c)
        self.output_capacitance: float = 0.1
        self.output_voltage: float = 0.0
        self.converter_efficiency: float = 0.95

        # -- Estado Electrónico --
        self.circuit_voltage: float = 0.0
        self.circuit_current: float = 0.0
        self.charge_accumulated: float = 0.0

        # -- Parámetros de Operación y Control --
        self.capacitor_discharge_threshold: float = -capacity * 0.9
        self.battery_is_discharging: bool = False
        self.battery_discharge_rate: float = capacity * 0.05
        self.hysteresis_factor: float = 0.1
        self.saturation_threshold: float = capacity  # Límite estricto
        self.compression_direction: int = -1

        # -- Controladores PID --
        self.speed_controller: 'PIDController' = PIDController(kp=1.0, ki=0.1, kd=0.01)
        self.energy_controller: 'PIDController' = PIDController(kp=0.5, ki=0.05, kd=0.005)
        self.target_speed: float = 0.0
        self.target_energy: float = 0.0

        # -- Historial para Diagnóstico --
        self.energy_history: List[float] = []
        self.efficiency_history: List[float] = []
        self.friction_force_history: List[float] = []

        logger.info(
            f"AtomicPiston inicializado en modo {self.mode.value} "
            f"con transductor {self.transducer_type.value} y fricción {self.friction_model.value}."
        )

    def _set_transducer_params(self) -> None:
        """Configura los parámetros del transductor según el tipo seleccionado."""
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

        logger.debug(f"Parámetros de transductor establecidos para {self.transducer_type.value}")

    @property
    def current_charge(self) -> float:
        """
        Calcula y devuelve la carga actual del pistón.

        La carga se define como la compresión positiva del pistón desde su
        punto de equilibrio (posición 0).

        Returns:
            La carga actual, representada como la compresión del pistón [m].
        """
        return max(0, -self.position)

    @property
    def stored_energy(self) -> float:
        """
        Calcula y devuelve la energía mecánica total almacenada en el pistón.

        Combina la energía potencial elástica (lineal y no lineal) y la
        energía cinética del pistón.

        Returns:
            La energía mecánica total almacenada en el sistema [J].
        """
        potential = (
            0.5 * self.k * self.position**2
            + (1 / 4) * self.nonlinear_elasticity * self.position**4
        )
        kinetic = 0.5 * self.m * self.velocity**2
        return potential + kinetic

    def calculate_friction(self, driving_force: float) -> float:
        """
        Calcula la fuerza de fricción según el modelo físico seleccionado.

        Args:
            driving_force: La fuerza neta que intenta mover el pistón (externa +
                resorte), usada para determinar la dirección de la fricción
                estática.

        Returns:
            La fuerza de fricción calculada [N], que se opone al movimiento.
        """
        # Fricción cinética (cuando hay movimiento)
        if abs(self.velocity) > 1e-5:
            if self.friction_model == FrictionModel.VISCOUS:
                return -self.c * self.velocity

            elif self.friction_model == FrictionModel.COULOMB:
                return -np.sign(self.velocity) * self.coulomb_friction

            elif self.friction_model == FrictionModel.STRIBECK:
                f_static, f_coulomb, v_stribeck = self.stribeck_coeffs
                # Modelo de Stribeck: F_f = F_c + (F_s - F_c) * exp(-(v/v_stribeck)^2)
                friction = f_coulomb + (f_static - f_coulomb) * np.exp(
                    -((abs(self.velocity) / v_stribeck) ** 2)
                )
                return -np.sign(self.velocity) * friction

        # Fricción estática (cuando la velocidad es casi cero)
        else:
            if self.friction_model == FrictionModel.VISCOUS:
                return 0.0 # No hay fricción viscosa sin velocidad

            elif self.friction_model == FrictionModel.COULOMB:
                # La fricción estática se opone a la fuerza impulsora hasta su límite máximo.
                return -np.sign(driving_force) * min(abs(driving_force), self.coulomb_friction)

            elif self.friction_model == FrictionModel.STRIBECK:
                f_static, _, _ = self.stribeck_coeffs
                # La fricción estática se opone a la fuerza impulsora hasta su límite estático.
                return -np.sign(driving_force) * min(abs(driving_force), f_static)

        return 0.0

    def apply_force(self, signal_value: float, source: str, mass_factor: float = 1.0) -> None:
        """
        Aplica una fuerza mecánica externa al pistón.

        Args:
            signal_value: El valor de la señal de entrada que genera la fuerza.
            source: Identificador de la fuente de la señal (para seguimiento).
            mass_factor: Un factor para escalar la fuerza aplicada.
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

    def apply_electronic_signal(self, voltage: float) -> None:
        """
        Aplica una señal eléctrica que se traduce en una fuerza mecánica.

        La conversión de voltaje a fuerza depende del tipo de transductor.

        Args:
            voltage: El voltaje de la señal eléctrica de entrada [V].
        """
        applied_force = voltage * self.force_sensitivity
        if self.transducer_type == TransducerType.MAGNETOSTRICTIVE:
            # Simulación de un circuito RL simple para transductor magnetoestrictivo.
            voltage_drop = self.circuit_current * self.internal_resistance
            di_dt = (voltage - voltage_drop) / self.equivalent_inductance
            self.circuit_current += di_dt * self.dt

            # Se actualiza el voltaje del circuito para reflejar la caída óhmica.
            self.circuit_voltage = self.circuit_current * self.internal_resistance
            applied_force = self.circuit_current * self.force_sensitivity

        self.last_applied_force += applied_force
        logger.debug(f"Señal eléctrica: {voltage:.2f}V → Fuerza: {applied_force:.2f}N")

    def update_state(self, dt: float) -> None:
        """
        Actualiza el estado físico del pistón para un intervalo de tiempo `dt`.

        Resuelve la ecuación diferencial del movimiento del pistón usando la
        integración de Verlet, un método numérico estable.

        La ecuación es: m * d²x/dt² + F_fricción(dx/dt) + k*x + ε*x³ = F_externa(t)

        Args:
            dt: El intervalo de tiempo para la integración [s]. Debe ser > 0.

        Raises:
            ValueError: Si `dt` es menor o igual a cero.
        """
        if dt <= 0:
            raise ValueError("El paso de tiempo (dt) debe ser un valor positivo.")
        self.dt = dt

        # Aplicar control de velocidad si está activo
        if self.target_speed != 0.0:
            control_force = self.speed_controller.update(
                self.target_speed, self.velocity, dt
            )
            self.last_applied_force += control_force

        # Calcular fuerzas internas (resorte) y la fuerza impulsora neta
        spring_force = (
            -self.k * self.position - self.nonlinear_elasticity * self.position**3
        )
        driving_force = self.last_applied_force + spring_force

        # Calcular la fricción basada en la fuerza impulsora (para la fricción estática)
        friction_force = self.calculate_friction(driving_force)
        self.friction_force_history.append(friction_force)

        # Fuerza total = suma de todas las fuerzas
        total_force = driving_force + friction_force

        # Calcular aceleración (a = F/m)
        self.acceleration = total_force / self.m

        # Integración de Verlet para la nueva posición: x(t+dt) = 2x(t) - x(t-dt) + a(t)dt²
        new_position = (
            2 * self.position - self.previous_position + self.acceleration * (dt**2)
        )

        # Actualizar velocidad: v(t) ≈ [x(t+dt) - x(t-dt)] / (2*dt)
        self.velocity = (new_position - self.previous_position) / (2 * dt)

        # Actualizar posiciones para la siguiente iteración
        self.previous_position = self.position
        self.position = new_position

        # Limitar la posición para evitar que exceda la capacidad física
        self.position = np.clip(
            self.position, -self.capacity, self.capacity
        )

        # Actualizar estado electrónico y resetear fuerzas para el siguiente ciclo
        self.update_electronic_state()
        self.last_applied_force = 0.0

        # Registrar historial para diagnóstico
        self.energy_history.append(self.stored_energy)
        self.efficiency_history.append(self.get_conversion_efficiency())

    def update_electronic_state(self) -> None:
        """Actualiza el estado del circuito electrónico equivalente."""
        self.circuit_voltage = -self.position * self.voltage_sensitivity
        self.circuit_current = self.velocity * self.equivalent_capacitance
        self.charge_accumulated += self.circuit_current * self.dt
        self.process_electrical_output()

    def process_electrical_output(self) -> None:
        """Procesa la energía eléctrica generada a través del convertidor de salida."""
        input_power = abs(self.circuit_voltage * self.circuit_current)
        output_power = input_power * self.converter_efficiency
        if self.equivalent_resistance > 0:
            self.output_voltage = np.sqrt(output_power * self.equivalent_resistance)
        else:
            self.output_voltage = 0.0

    def discharge(self, dt: float) -> Optional[Dict[str, Any]]:
        """
        Gestiona la descarga de energía del pistón según el modo de operación.

        Args:
            dt: El intervalo de tiempo [s].

        Returns:
            Un diccionario con información sobre la descarga si ocurre, o None.
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

    def capacitor_discharge(self, dt: float) -> Optional[Dict[str, Any]]:
        """
        Realiza una descarga en modo CAPACITOR si se cumplen las condiciones.

        Args:
            dt: El intervalo de tiempo [s].

        Returns:
            Un diccionario con los detalles de la descarga en forma de pulso.
        """
        discharge_threshold = self.capacitor_discharge_threshold
        hysteresis_threshold = discharge_threshold * (1 - self.hysteresis_factor)
        if self.position <= discharge_threshold:
            amplitude = self.current_charge
            logger.info(f"¡Descarga CAPACITOR! Amplitud: {amplitude:.2f}")
            self.position = hysteresis_threshold
            self.velocity = 5.0  # Simula un rebote rápido
            return {"type": "pulse", "amplitude": amplitude, "duration": 0.001}
        return None

    def battery_discharge(self, dt: float) -> Optional[Dict[str, Any]]:
        """Realiza una descarga en modo batería si está activada y hay carga."""
        if not self.battery_is_discharging:
            return None

        # 1. COMPROBAR EL ESTADO AL INICIO: Esta es la puerta de control principal.
        # Si la carga está (o se ha vuelto) insignificante, la primera prioridad
        # es corregir el estado y detener el proceso.
        if self.current_charge <= 1e-5:
            self.battery_is_discharging = False
            logger.info("Descarga BATTERY: Carga agotada o insignificante. Descarga desactivada.")
            # Aseguramos que la posición quede en 0 para evitar cargas residuales.
            self.position = 0.0
            return None

        # 2. Si pasamos la comprobación anterior, significa que hay carga y debemos descargar.
        # La descarga máxima es un 80% para evitar que se vacíe en un solo paso.
        max_discharge = self.current_charge * 0.8
        discharge_amount = min(self.battery_discharge_rate * dt, max_discharge)

        self.position += discharge_amount

        # La amplitud de la señal es proporcional a la descarga real.
        output_amplitude = discharge_amount / (self.battery_discharge_rate * dt) if self.battery_discharge_rate > 0 else 1.0

        return {
            "type": "sustained",
            "amplitude": output_amplitude,
            "duration": dt
        }

    def simulate_discharge_circuit(self, load_resistance: float, dt: float) -> Tuple[float, float, float]:
        """
        Simula la descarga de energía a través de una carga externa.

        Args:
            load_resistance: Resistencia de la carga externa [Ω].
            dt: Intervalo de tiempo de la simulación [s].

        Returns:
            Una tupla con el voltaje, la corriente y la potencia en la carga.
        """
        logger.debug(f"Simulando descarga con resistencia de carga: {load_resistance} Ohm")
        if self.circuit_voltage == 0:
            return 0.0, 0.0, 0.0

        total_resistance = self.internal_resistance + load_resistance
        discharge_current = self.circuit_voltage / total_resistance

        # Corrección: Energía disipada es P*t = I²*R*t. Se elimina el factor 0.5.
        discharge_energy = discharge_current**2 * load_resistance * dt

        if abs(self.position) > 1e-6 and self.k > 0:
            # La energía potencial es E = 0.5*k*x². El cambio en posición es dx = dE / (k*x).
            # La energía se disipa, por lo que dE es negativo (la energía almacenada disminuye).
            # Para una posición negativa (compresión), dE/ (k*x) resulta en un dx positivo,
            # moviendo la posición hacia cero, lo cual es correcto.
            position_change = -discharge_energy / (self.k * self.position)
        else:
            position_change = 0.0

        # Aplica el cambio de posición solo si el pistón está comprimido.
        if self.position < 0:
            self.position += position_change
            # Asegura que la posición no se vuelva positiva debido a la descarga.
            self.position = min(0.0, max(-self.capacity, self.position))

        self.update_electronic_state()
        load_voltage = discharge_current * load_resistance
        power_dissipated = load_voltage * discharge_current
        return load_voltage, discharge_current, power_dissipated

    def set_compression_direction(self, direction: int) -> None:
        """
        Configura la dirección de compresión del pistón.

        Args:
            direction: La dirección de compresión (-1 para compresión negativa, 1 para positiva).
        """
        if direction not in (-1, 1):
            logger.warning(f"Dirección inválida: {direction}. Usando -1 (compresión).")
            self.compression_direction = -1
        else:
            self.compression_direction = direction
            logger.info(f"Dirección de compresión establecida en: {self.compression_direction}")

    def get_conversion_efficiency(self) -> float:
        """
        Calcula la eficiencia de conversión de energía instantánea.

        Compara la energía mecánica con la energía eléctrica total del sistema.

        Returns:
            La eficiencia de conversión instantánea (0 a 1).
        """
        mechanical_energy = self.stored_energy
        electrical_energy = 0.5 * self.equivalent_capacitance * self.circuit_voltage**2
        total_stored_energy = mechanical_energy + electrical_energy
        if total_stored_energy > 1e-9:
            return mechanical_energy / total_stored_energy
        return 0.0

    def set_mode(self, mode: PistonMode) -> None:
        """
        Establece el modo de operación del pistón.

        Args:
            mode: El nuevo modo de operación (PistonMode.CAPACITOR o PistonMode.BATTERY).
        """
        self.mode = mode
        self.battery_is_discharging = False
        logger.info(f"Modo cambiado a: {mode.value}")

    def trigger_discharge(self, discharge_on: bool) -> None:
        """
        Activa o desactiva la descarga continua en modo BATTERY.

        Args:
            discharge_on: True para activar la descarga, False para desactivarla.
        """
        if self.mode == PistonMode.BATTERY:
            self.battery_is_discharging = discharge_on
            logger.info(f"Descarga BATTERY {'activada' if discharge_on else 'desactivada'}.")
        else:
            logger.warning(
                f"Llamada a trigger_discharge en modo {self.mode.value}, "
                f"que solo es válido en modo BATTERY."
            )

    def set_speed_target(self, target: float) -> None:
        """
        Establece la velocidad objetivo para el controlador PID de velocidad.

        Args:
            target: La velocidad objetivo deseada [m/s].
        """
        self.target_speed = target
        self.speed_controller.reset()
        logger.info(f"Objetivo de velocidad establecido: {target:.2f} m/s")

    def set_energy_target(self, target: float) -> None:
        """
        Establece la energía objetivo para el controlador PID de energía.

        Args:
            target: La energía objetivo deseada [J].
        """
        self.target_energy = target
        self.energy_controller.reset()
        logger.info(f"Objetivo de energía establecido: {target:.2f} J")

    def generate_bode_data(self, frequency_range: np.ndarray) -> Dict[str, Any]:
        """
        Genera datos para un diagrama de Bode de la función de transferencia del sistema.

        Args:
            frequency_range: Un array de NumPy con el rango de frecuencias a analizar [Hz].

        Returns:
            Un diccionario que contiene los datos para el diagrama de Bode:
            - 'frequencies': El mismo array de frecuencias de entrada [Hz].
            - 'magnitude': Una lista de valores de magnitud en decibelios (dB).
            - 'phase': Una lista de valores de fase en grados.
        """
        magnitudes = []
        phases = []
        for f in frequency_range:
            omega = 2 * np.pi * f
            # Función de transferencia mecánica: H_mech(s) = X(s) / F(s)
            # s = jω
            H_mech = 1 / (self.m * (1j * omega)**2 + self.c * (1j * omega) + self.k)
            # Función de transferencia electromecánica completa
            H_electrical = H_mech * self.voltage_sensitivity * self.force_sensitivity
            magnitudes.append(20 * np.log10(np.abs(H_electrical)))
            phases.append(np.angle(H_electrical, deg=True))
        return {"frequencies": frequency_range, "magnitude": magnitudes, "phase": phases}

    def export_history_to_csv(self, filename: str) -> None:
        """
        Exporta los historiales de simulación (energía, eficiencia, etc.) a un archivo CSV.

        Args:
            filename: La ruta y nombre del archivo CSV donde se guardarán los datos.

        Raises:
            IOError: Si ocurre un problema al escribir en el archivo.
        """
        header = ['time_step', 'stored_energy', 'conversion_efficiency', 'friction_force']

        # Encuentra la longitud máxima para el caso de historiales de diferente longitud
        num_steps = max(
            len(self.energy_history),
            len(self.efficiency_history),
            len(self.friction_force_history)
        )

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)

                for i in range(num_steps):
                    row = [
                        i,
                        self.energy_history[i] if i < len(self.energy_history) else '',
                        self.efficiency_history[i] if i < len(self.efficiency_history) else '',
                        self.friction_force_history[i] if i < len(self.friction_force_history) else ''
                    ]
                    writer.writerow(row)
            logger.info(f"Historial de simulación exportado exitosamente a {filename}")
        except IOError as e:
            logger.error(f"Error al exportar el historial a CSV: {e}")
            raise

    def simulate_step_response(self, force_amplitude: float, duration: float, dt: float) -> Dict[str, Any]:
        """
        Simula la respuesta del pistón a una entrada de fuerza escalón.

        Args:
            force_amplitude: Magnitud de la fuerza constante a aplicar [N].
            duration: Duración total de la simulación [s].
            dt: Paso de tiempo para la simulación [s].

        Returns:
            Un diccionario con las series temporales de 'time', 'position', 'velocity' y 'acceleration'.
        """
        self.reset()
        time_series = np.arange(0, duration, dt)
        position_history = []
        velocity_history = []
        acceleration_history = []

        for _ in time_series:
            self.last_applied_force = force_amplitude
            self.update_state(dt)
            position_history.append(self.position)
            velocity_history.append(self.velocity)
            acceleration_history.append(self.acceleration)

        return {
            "time": time_series,
            "position": np.array(position_history),
            "velocity": np.array(velocity_history),
            "acceleration": np.array(acceleration_history),
        }

    def simulate_impulse_response(self, impulse_magnitude: float, duration: float, dt: float) -> Dict[str, Any]:
        """
        Simula la respuesta del pistón a una entrada de fuerza impulso.

        Args:
            impulse_magnitude: Magnitud del impulso a aplicar [N·s].
            duration: Duración total de la simulación [s].
            dt: Paso de tiempo para la simulación [s].

        Returns:
            Un diccionario con las series temporales de 'time', 'position', 'velocity' y 'acceleration'.
        """
        self.reset()
        time_series = np.arange(0, duration, dt)
        position_history = []
        velocity_history = []
        acceleration_history = []

        # Aplicar el impulso como un cambio instantáneo en la velocidad (p = m*v -> Δv = I/m)
        self.velocity += impulse_magnitude / self.m

        for _ in time_series:
            self.update_state(dt)
            position_history.append(self.position)
            velocity_history.append(self.velocity)
            acceleration_history.append(self.acceleration)

        return {
            "time": time_series,
            "position": np.array(position_history),
            "velocity": np.array(velocity_history),
            "acceleration": np.array(acceleration_history),
        }

    def reset(self) -> None:
        """Reinicia el estado del pistón a sus condiciones iniciales por defecto."""
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
        logger.info("Estado del pistón reiniciado a condiciones iniciales.")

    def get_differential_equation_terms(self) -> Dict[str, float]:
        """
        Devuelve los términos individuales de la ecuación diferencial del movimiento.

        La ecuación es: m*a + F_fricción + F_resorte = F_externa

        Returns:
            Un diccionario con los componentes calculados de la ecuación.
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
    """
    Implementa un controlador Proporcional-Integral-Derivativo (PID).

    Este controlador es utilizado por `AtomicPiston` para regular la
    velocidad y la energía del sistema, ajustando la fuerza de control
    basada en el error entre un setpoint y el valor actual.

    Attributes:
        kp (float): Ganancia proporcional.
        ki (float): Ganancia integral.
        kd (float): Ganancia derivativa.
        output_limit (float): Límite de la salida para anti-windup.
    """
    def __init__(self, kp: float, ki: float, kd: float, output_limit: float = 10.0) -> None:
        """
        Inicializa el controlador PID.

        Args:
            kp: Ganancia proporcional (P).
            ki: Ganancia integral (I).
            kd: Ganancia derivativa (D).
            output_limit: Límite absoluto para la salida (anti-windup).
        """
        self.kp: float = kp
        self.ki: float = ki
        self.kd: float = kd
        self.integral: float = 0.0
        self.previous_error: float = 0.0
        self.output_limit: float = output_limit

    def update(self, setpoint: float, current_value: float, dt: float) -> float:
        """
        Calcula la salida del controlador PID para un paso de tiempo.

        Args:
            setpoint: El valor deseado o de referencia.
            current_value: El valor medido actual del sistema.
            dt: El intervalo de tiempo desde la última actualización [s].

        Returns:
            La señal de control calculada.
        """
        if dt <= 0:
            return 0.0

        error = setpoint - current_value

        # Término Proporcional
        p_term = self.kp * error

        # Término Integral con anti-windup
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Término Derivativo
        derivative = (error - self.previous_error) / dt
        d_term = self.kd * derivative

        # Salida combinada
        output = p_term + i_term + d_term

        # Aplicar límite de salida y anti-windup
        if abs(output) > self.output_limit:
            output = np.clip(output, -self.output_limit, self.output_limit)
            # Evitar que el término integral crezca indefinidamente
            self.integral -= error * dt

        self.previous_error = error
        return output

    def reset(self) -> None:
        """Reinicia el estado interno del controlador (integral y error previo)."""
        self.integral = 0.0
        self.previous_error = 0.0

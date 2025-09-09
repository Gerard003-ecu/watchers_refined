# atomic_piston/atomic_piston.py
import logging
import time
from enum import Enum

import numpy as np

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

    def __init__(
        self,
        capacity: float,
        elasticity: float,
        damping: float,
        piston_mass: float = 1.0,
        mode: PistonMode = PistonMode.CAPACITOR,
        transducer_type: TransducerType = TransducerType.PIEZOELECTRIC,
    ):
        """Inicializa una nueva instancia de AtomicPiston.

        Args:
            capacity (float): Capacidad máxima de compresión del pistón (en metros).
                Define el límite físico de desplazamiento en compresión.
            elasticity (float): Constante elástica del resorte (k en N/m).
                Determina la rigidez del componente elástico del pistón.
            damping (float): Coeficiente de amortiguación (c en N·s/m).
                Representa las pérdidas de energía debido a la fricción o resistencia.
            piston_mass (float, optional): Masa inercial del pistón (m en kg).
                Por defecto es 1.0 kg.
            mode (PistonMode, optional): Modo de operación inicial del pistón.
                Puede ser `PistonMode.CAPACITOR` o `PistonMode.BATTERY`.
                Por defecto es `PistonMode.CAPACITOR`.
            transducer_type (TransducerType, optional): Tipo de transductor
                utilizado para la interfaz electromecánica. Puede ser
                `TransducerType.PIEZOELECTRIC`, `TransducerType.ELECTROSTATIC`,
                o `TransducerType.MAGNETOSTRICTIVE`.
                Por defecto es `TransducerType.PIEZOELECTRIC`.
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
        self.equivalent_inductance = self.m  # L = m
        self.equivalent_resistance = 1.0 / max(0.001, self.c)  # R = 1/c

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
        """float: La carga actual del pistón, representada como la compresión.

        Se calcula como el valor máximo entre cero y la posición negativa
        del pistón (ya que la compresión se considera en la dirección negativa).
        Un valor mayor indica una mayor compresión (carga).
        """
        return max(0, -self.position)

    @property
    def stored_energy(self) -> float:
        """float: La energía mecánica total almacenada actualmente en el pistón.

        Corresponde a la suma de la energía potencial elástica debido a la
        deformación del resorte (0.5 * k * x²) y la energía cinética debido
        al movimiento del pistón (0.5 * m * v²).
        """
        potential = 0.5 * self.k * self.position**2
        kinetic = 0.5 * self.m * self.velocity**2
        return potential + kinetic

    def apply_force(self, signal_value: float, source: str, mass_factor: float = 1.0):
        """Aplica una fuerza mecánica al pistón basada en una señal de entrada.

        La fuerza se calcula utilizando la energía cinética derivada de la
        velocidad de la señal de entrada. La dirección de la fuerza puede ser
        configurada mediante `set_compression_direction`.

        Args:
            signal_value: El valor actual de la señal de entrada.
            source: Un identificador único para la fuente de la señal.
                Se utiliza para calcular la velocidad de la señal correctamente
                cuando hay múltiples fuentes.
            mass_factor: Un factor para escalar la 'masa' efectiva asociada
                con la señal. Esto permite ajustar la magnitud de la fuerza
                generada a partir de la velocidad de la señal. Por defecto es 1.0.
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
        force = self.compression_direction * 0.5 * mass_factor * (signal_velocity**2)
        self.last_applied_force += force

        logger.debug(f"Fuente '{source}': Fuerza aplicada = {force:.2f}N")

    def apply_electronic_signal(self, voltage: float):
        """Aplica una señal eléctrica al sistema, traduciéndola en fuerza mecánica.

        La conversión de voltaje a fuerza depende del tipo de transductor
        configurado (`transducer_type`). Para transductores piezoeléctricos y
        electrostáticos, la fuerza es directamente proporcional al voltaje.
        Para transductores magnetostrictivos, se simula un circuito RL para
        determinar la corriente y, consecuentemente, la fuerza.

        La fuerza generada se suma a `last_applied_force` para ser utilizada
        en la próxima actualización del estado del pistón.

        Args:
            voltage: El voltaje (en Voltios) aplicado al transductor.
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
        """Actualiza el estado del pistón resolviendo la ecuación del oscilador
        armónico amortiguado y forzado mediante integración de Verlet.

        Ecuación Física Implementada:
        Este método resuelve numéricamente la ecuación diferencial ordinaria (ODE)
        fundamental para un sistema masa-resorte-amortiguador forzado:
        m * d²x/dt² + c * dx/dt + k * x = F(t)

        Donde:
        - m: Masa del pistón (`self.m`).
        - c: Coeficiente de amortiguación (`self.c`).
        - k: Constante elástica del resorte (`self.k`).
        - x: Posición del pistón (`self.position`).
        - F(t): Fuerza externa total aplicada (`self.last_applied_force`).

        Correspondencia de la Implementación:
        1. Se calcula la fuerza total sumando la fuerza externa (`F(t)`), la
           fuerza del resorte (`-k*x`) y la fuerza de amortiguación (`-c*dx/dt`).
        2. La aceleración (d²x/dt²) se obtiene de la fuerza total mediante la
           segunda ley de Newton (a = F/m).
        3. Se utiliza la fórmula de integración de Verlet para actualizar la
           posición, lo que proporciona una solución estable y precisa para
           este tipo de sistema físico.

        Args:
            dt: El paso de tiempo (delta t, en segundos) para la simulación.
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
            2 * self.position - self.previous_position + self.acceleration * (dt**2)
        )

        # Actualizar velocidad (v = [x(t+dt) - x(t-dt)] / (2*dt))
        self.velocity = (new_position - self.previous_position) / (2 * dt)

        # Actualizar posiciones
        self.previous_position = self.position
        self.position = new_position

        # Limitar saturación (evitar sobrecompresión/extensión excesiva)
        self.position = np.clip(
            self.position, -self.saturation_threshold, self.saturation_threshold
        )

        # Actualizar estado electrónico (conversión mecánico-eléctrica)
        self.update_electronic_state()

        # Resetear fuerza externa después de aplicarla
        self.last_applied_force = 0.0

        # Registrar energía y eficiencia
        self.energy_history.append(self.stored_energy)
        self.efficiency_history.append(self.get_conversion_efficiency())

    def update_electronic_state(self):
        """Actualiza el estado del circuito electrónico equivalente.

        Este método se basa en el estado mecánico actual del pistón (posición y
        velocidad) para calcular el voltaje y la corriente en el circuito
        eléctrico simulado. También actualiza la carga acumulada.

        El voltaje del circuito se considera proporcional a la compresión
        (posición negativa) del pistón, modulado por `voltage_sensitivity`.
        La corriente del circuito es proporcional a la velocidad del pistón,
        modulada por la `equivalent_capacitance`.
        La carga acumulada se actualiza integrando la corriente a lo largo del
        paso de tiempo `dt`.
        """
        # Voltaje proporcional a la compresión (posición negativa)
        self.circuit_voltage = -self.position * self.voltage_sensitivity

        # Corriente proporcional a la velocidad del pistón
        self.circuit_current = self.velocity * self.equivalent_capacitance

        # Actualizar carga acumulada (integral de corriente)
        self.charge_accumulated += self.circuit_current * self.dt

    def discharge(self, dt: float):
        """Gestiona la descarga de energía del pistón según su modo de operación.

        Este método delega la lógica de descarga a `capacitor_discharge` o
        `battery_discharge` dependiendo del `mode` actual del pistón.

        Args:
            dt: El paso de tiempo (delta t, en segundos) relevante para la
                descarga, especialmente para el modo batería.

        Returns:
            dict | None: Un diccionario que representa la señal de salida si
            ocurre una descarga, o `None` si no hay descarga.
            El formato del diccionario depende del modo de descarga.
        """
        if self.mode == PistonMode.CAPACITOR:
            return self.capacitor_discharge(dt)
        elif self.mode == PistonMode.BATTERY:
            return self.battery_discharge(dt)
        return None

    def capacitor_discharge(self, dt: float):
        """Realiza una descarga en modo capacitor si se cumplen las condiciones.

        En modo capacitor, la descarga ocurre como un pulso instantáneo cuando
        la posición del pistón alcanza o supera el umbral de descarga
        (`capacitor_discharge_threshold`).
        Después de la descarga, la posición del pistón se ajusta a un umbral
        de histéresis para evitar un ciclado rápido, y se simula un rebote
        asignando una velocidad positiva.

        Args:
            dt: El paso de tiempo (delta t, en segundos). Aunque la descarga es
                instantánea, este parámetro se mantiene por consistencia con
                la interfaz de `discharge`.

        Returns:
            dict | None: Un diccionario que representa la señal de pulso si
            ocurre una descarga, con claves "type", "amplitude" y "duration".
            Retorna `None` si no se cumplen las condiciones de descarga.
        """
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
                "duration": 0.001,  # Pulso muy corto
            }
        return None

    def battery_discharge(self, dt: float):
        """Realiza una descarga en modo batería si está activada y hay carga.

        En modo batería, la descarga es un proceso continuo mientras
        `battery_is_discharging` sea `True` y haya carga almacenada
        (`current_charge` > 0). La cantidad de descarga en cada paso de tiempo
        es proporcional a `battery_discharge_rate`.

        La descarga se detiene si `current_charge` llega a cero o si
        `battery_is_discharging` se establece en `False`.

        Args:
            dt: El paso de tiempo (delta t, en segundos) durante el cual
                ocurre la descarga.

        Returns:
            dict | None: Un diccionario que representa la señal sostenida si
            ocurre una descarga, con claves "type", "amplitude" y "duration".
            La amplitud es proporcional a la cantidad descargada.
            Retorna `None` si no hay descarga activa o no hay carga.
        """
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

            return {"type": "sustained", "amplitude": output_amplitude, "duration": dt}
        else:  # current_charge is already <= 0 at the beginning of the call
            self.battery_is_discharging = False
            logger.info("Descarga BATTERY: Carga ya estaba agotada.")
            return None

    def simulate_discharge_circuit(self, load_resistance: float, dt: float):
        """Simula la descarga de energía del pistón a través de una carga externa.

        Ecuación clave:
        dE = 0.5 * I² * R_load * dt  (Energía disipada)
        dx = -dE / (k * x)           (Cambio en compresión)

        Este método calcula cómo la energía almacenada en el pistón (representada
        por `circuit_voltage`) se disipa a través de una resistencia de carga
        externa. Modifica la posición del pistón para reflejar la pérdida de
        energía mecánica debido a la descarga eléctrica.

        La simulación asume que la energía mecánica perdida por el pistón se
        convierte en energía eléctrica disipada en el circuito.

        Args:
            load_resistance: La resistencia de la carga externa (en Ohmios) a
                través de la cual se descarga el pistón.
            dt: El paso de tiempo (delta t, en segundos) para la simulación de
                la descarga.

        Returns:
            tuple[float, float, float]: Una tupla conteniendo:
                - voltaje_en_carga (float): El voltaje a través de la resistencia
                  de carga (en Voltios).
                - corriente_en_carga (float): La corriente que fluye a través de
                  la resistencia de carga (en Amperios).
                - potencia_disipada (float): La potencia disipada por la
                  resistencia de carga (en Vatios).
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
        """Configura la dirección en la que las fuerzas externas comprimen el pistón.

        Por defecto, una fuerza positiva aplicada mediante `apply_force` resulta
        en una compresión (movimiento en la dirección negativa del eje de posición).
        Este método permite invertir esa lógica.

        Args:
            direction: El indicador de dirección.
                -1: Las fuerzas aplicadas tienden a comprimir el pistón (disminuyen
                la posición). Este es el valor predeterminado.
                -1: Las fuerzas aplicadas tienden a expandir el pistón (aumentan
                la posición).
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
        """Calcula la eficiencia de conversión de energía instantánea del sistema.

        La eficiencia se define como la relación entre la energía mecánica
        almacenada (potencial y cinética) y la energía total almacenada
        (mecánica más eléctrica en el circuito equivalente).

        Returns:
            float: La eficiencia de conversión como un valor entre 0.0 y 1.0.
                   Retorna 0.0 si la energía total almacenada es cero.
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
        """Establece el modo de operación del pistón.

        Cambiar el modo puede afectar cómo el pistón gestiona la descarga de
        energía. Si se cambia a modo batería, el estado de descarga
        (`battery_is_discharging`) se resetea a `False`.

        Args:
            mode: El nuevo modo de operación, que debe ser un miembro de la
                  enumeración `PistonMode` (ej. `PistonMode.CAPACITOR` o
                  `PistonMode.BATTERY`).
        """
        self.mode = mode
        self.battery_is_discharging = False
        logger.info(f"Modo del pistón cambiado a: {mode.value}")

    def trigger_discharge(self, discharge_on: bool):
        """Activa o desactiva la descarga continua en modo BATTERY.

        Este método solo tiene efecto si el pistón está actualmente en
        `PistonMode.BATTERY`. Si se llama en modo `PistonMode.CAPACITOR`,
        se registrará una advertencia y no se realizará ninguna acción.

        Args:
            discharge_on: Un booleano que indica si la descarga debe activarse
                (`True`) o desactivarse (`False`).
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
        """Genera datos para un diagrama de Bode de la respuesta electromecánica.

        Calcula la magnitud (en dB) y la fase (en grados) de la función de
        transferencia del sistema. Esta función relaciona una excitación de
        fuerza mecánica con una respuesta de voltaje eléctrica, considerando
        las propiedades del pistón (masa, amortiguación, elasticidad) y las
        sensibilidades del transductor.

        Args:
            frequency_range: Un array de NumPy con las frecuencias (en Hertz)
                para las cuales se calculará la respuesta.

        Returns:
            dict: Un diccionario conteniendo:
                - 'frequencies' (np.ndarray): El mismo array de frecuencias de entrada.
                - 'magnitude' (list[float]): Lista de magnitudes de la respuesta
                  en dB para cada frecuencia.
                - 'phase' (list[float]): Lista de fases de la respuesta en grados
                  para cada frecuencia.
        """
        magnitude = []
        phase = []

        for f in frequency_range:
            omega = 2 * np.pi * f

            # Función de transferencia del sistema mecánico
            H_mech = 1 / (self.m * (1j * omega) ** 2 + self.c * (1j * omega) + self.k)

            # Convertir a respuesta eléctrica
            H_electrical = H_mech * self.voltage_sensitivity * self.force_sensitivity

            magnitude.append(20 * np.log10(np.abs(H_electrical)))
            phase.append(np.angle(H_electrical, deg=True))

        return {"frequencies": frequency_range, "magnitude": magnitude, "phase": phase}

    def reset(self):
        """Reinicia el estado del pistón a sus condiciones iniciales.

        Esto incluye restablecer la posición, velocidad, aceleración,
        fuerzas aplicadas, estado del circuito electrónico, estado de descarga
        de la batería, historial de señales, historial de energía y eficiencia.
        Es útil para comenzar una nueva simulación o prueba desde un
        estado conocido y limpio.
        """
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

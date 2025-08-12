# atomic_piston/atomic_piston_service.py
"""
Microservicio para la simulación de una Unidad de Potencia Inteligente (IPU).

Este servicio envuelve el modelo de simulación de `AtomicPiston` en una API
RESTful de Flask, permitiendo el control y monitoreo remotos como parte del
ecosistema "Watchers".

Principales Funcionalidades:
- Expone endpoints para consultar el estado (`/api/state`) y salud (`/api/health`).
- Permite el control del pistón a través de comandos (`/api/command`).
- Se registra de forma autónoma con el servicio AgentAI al iniciar.
- Ejecuta la simulación física en un hilo de fondo continuo.
"""
import os
import threading
import time
import logging
import math
import requests
import numpy as np
from flask import Flask, jsonify, request
from collections import deque
from typing import List, Dict, Tuple, Optional, Any
import csv # Keep csv for export method, though it might not be used by the service itself

from .constants import PistonMode, TransducerType, FrictionModel, ControllerType
from .config import PistonConfig

# --- Configuración del Logging ---
# (Similar to malla_watcher for consistency)
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("atomic_piston_service")
if not logger.hasHandlers():
    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, "atomic_piston_service.log"))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
         logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s")
    )
    logger.addHandler(stream_handler)

    logger.setLevel(logging.INFO)


# --- Global Service Configuration (will be populated in main) ---
config: Optional[PistonConfig] = None

# --- Andamiaje del Microservicio ---
app = Flask(__name__)

# Instancia global del pistón (gemelo digital) que será gestionada por el servicio
ipu_instance: Optional['AtomicPiston'] = None

# Elementos para la gestión de hilos
ipu_lock = threading.Lock()
simulation_thread: Optional[threading.Thread] = None
stop_simulation_event = threading.Event()


# --- Lógica de Simulación (Thread) ---

def simulation_loop() -> None:
    """
    Bucle principal que se ejecuta en un hilo de fondo para simular el pistón.

    Orquesta la actualización periódica del estado del 'gemelo digital' del
    pistón atómico.
    """
    logger.info("Iniciando bucle de simulación del pistón atómico...")

    if not config:
        logger.error("La configuración global no está disponible. Deteniendo el hilo de simulación.")
        return

    while not stop_simulation_event.is_set():
        start_time = time.monotonic()

        if not ipu_instance:
            logger.warning("Instancia de IPU no inicializada, saltando ciclo de simulación.")
            stop_simulation_event.wait(1) # Esperar un poco antes de reintentar
            continue

        try:
            with ipu_lock:
                # El corazón de la simulación: actualizar estado y gestionar descarga
                ipu_instance.update_state(config.simulation_interval)
                ipu_instance.discharge(config.simulation_interval)

        except Exception as e:
            logger.exception("Error catastrófico durante el ciclo de simulación del pistón: %s", e)
            # En caso de un error grave en la simulación, esperar antes de reintentar
            stop_simulation_event.wait(5)

        elapsed_time = time.monotonic() - start_time
        sleep_time = max(0, config.simulation_interval - elapsed_time)

        # Esperar el tiempo restante para mantener la frecuencia de simulación
        if sleep_time > 0:
            stop_simulation_event.wait(sleep_time)

    logger.info("Bucle de simulación del pistón detenido.")


# --- Lógica de Registro con AgentAI ---

def register_with_agent_ai(
    module_name: str,
    module_url: str,
    health_url: str,
    description: str = ""
) -> bool:
    """
    Intenta registrar este microservicio con el AgentAI.

    Realiza una solicitud HTTP POST para anunciar la disponibilidad de este
    servicio al ecosistema. Incluye reintentos en caso de fallo de conexión.
    """
    if not config:
        logger.error("La configuración global no está disponible para el registro.")
        return False

    payload = {
        "nombre": module_name,
        "url": module_url,
        "url_salud": health_url,
        "tipo": "hardware_simulation",
        "aporta_a": "ecosistema_watchers",
        "naturaleza_auxiliar": "gemelo_digital_ipu",
        "descripcion": description
    }

    logger.info(f"Intentando registrar '{module_name}' en AgentAI ({config.agent_ai_register_url})...")
    for attempt in range(config.max_registration_retries):
        try:
            response = requests.post(
                config.agent_ai_register_url,
                json=payload,
                timeout=config.requests_timeout,
                verify=False  # Solo para entornos de prueba, no producción!
            )
            response.raise_for_status()
            if response.status_code == 200:
                logger.info(f"Registro de '{module_name}' exitoso en AgentAI.")
                return True
            else:
                logger.warning(
                    f"Registro de '{module_name}' recibido con status {response.status_code}. "
                    f"Respuesta: {response.text}"
                )
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Error de conexión al registrar '{module_name}' "
                f"(intento {attempt + 1}/{config.max_registration_retries}): {e}"
            )
        except Exception as e:
            logger.error(
                f"Error inesperado durante el registro de '{module_name}' "
                f"(intento {attempt + 1}/{config.max_registration_retries}): {e}"
            )

        if attempt < config.max_registration_retries - 1:
            logger.info(f"Reintentando registro en {config.retry_delay_seconds} segundos...")
            time.sleep(config.retry_delay_seconds)
        else:
            logger.error(
                f"No se pudo registrar '{module_name}' en AgentAI después de "
                f"{config.max_registration_retries} intentos."
            )
            return False
    return False


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
        self.max_history_size = 10000
        self.energy_history: deque = deque(maxlen=self.max_history_size)
        self.efficiency_history: deque = deque(maxlen=self.max_history_size)
        self.friction_force_history: deque = deque(maxlen=self.max_history_size)

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

        # 1. SUMA DE FUERZAS EXTERNAS Y DE CONTROL
        # F_externa(t) en la ecuación diferencial.
        external_force = self.last_applied_force
        if self.target_speed != 0.0:
            control_force = self.speed_controller.update(
                self.target_speed, self.velocity, dt
            )
            external_force += control_force

        # 2. FUERZAS INTERNAS DEPENDIENTES DE LA POSICIÓN (RESORTE)
        # Términos k*x y ε*x³ en la ecuación.
        # Optimización: precalcular términos
        position_sq = self.position * self.position
        spring_force = -self.k * self.position - self.nonlinear_elasticity * position_sq * self.position

        # 3. FUERZA DE FRICCIÓN (dependiente de la velocidad y la fuerza motriz)
        # El término F_fricción(dx/dt) en la ecuación.
        # Para la fricción estática, necesitamos saber si las otras fuerzas
        # son suficientes para superar el umbral estático.
        driving_force_for_friction = external_force + spring_force
        friction_force = self.calculate_friction(driving_force_for_friction)
        self.friction_force_history.append(friction_force)

        # 4. FUERZA NETA SOBRE EL PISTÓN
        # Reorganizando la ecuación: F_neta = F_externa - (F_fricción + F_resorte)
        # O más directamente: F_neta = F_externa + F_resorte + F_fricción
        # (recordando que F_resorte y F_fricción ya tienen el signo correcto).
        total_force = external_force + spring_force + friction_force

        # 5. CÁLCULO DE LA ACELERACIÓN (a = F_neta / m)
        # Este es el término d²x/dt² de la ecuación.
        self.acceleration = total_force / self.m

        # 6. INTEGRACIÓN NUMÉRICA (Verlet) para encontrar la nueva posición y velocidad.
        # x(t+dt) = 2x(t) - x(t-dt) + a(t)dt²
        new_position = (
            2 * self.position - self.previous_position + self.acceleration * (dt**2)
        )

        # Corrección de energía para mantener estabilidad
        # Esta seccion es opcional y puede ser activada para simulaciones largas
        # donde el drift de energia de Verlet puede ser un problema.
        energy_before = 0.5 * self.k * self.position**2 + 0.5 * self.m * self.velocity**2
        # La energia potencial no lineal tambien deberia ser incluida si es significativa
        # energy_before += (1/4) * self.nonlinear_elasticity * self.position**4

        new_velocity_estimate = (new_position - self.previous_position) / (2 * dt)
        energy_after = 0.5 * self.k * new_position**2 + 0.5 * self.m * new_velocity_estimate**2
        # energy_after += (1/4) * self.nonlinear_elasticity * new_position**4

        # Prevenir division por cero y solo corregir si la energia es significativa
        if energy_after > 1e-9 and abs(energy_after - energy_before) > 0.1 * energy_before:
            # Aplicar factor de corrección para conservar la energía
            correction_factor = math.sqrt(energy_before / energy_after)
            # Solo corregimos la parte dinámica de la posición, no el punto de equilibrio
            new_position = self.position + (new_position - self.position) * correction_factor

        # v(t) ≈ [x(t+dt) - x(t-dt)] / (2*dt)
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

        new_position = self.position + discharge_amount
        self.position = min(0.0, new_position)  # Limitar a posición máxima 0.0

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
        self.energy_history.clear()
        self.efficiency_history.clear()
        self.friction_force_history.clear()
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


# --- Endpoints de la API Flask ---

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Verifica el estado de salud del servicio Atomic Piston.

    Retorna el estado de la simulación en segundo plano y si la instancia
    del pistón ha sido inicializada correctamente.
    """
    sim_alive = simulation_thread and simulation_thread.is_alive()
    piston_initialized = ipu_instance is not None

    status = "success"
    message = "Servicio Atomic Piston operativo."
    http_code = 200

    if not sim_alive or not piston_initialized:
        status = "error"
        http_code = 503
        if not piston_initialized:
            message = "Error: La instancia del pistón (IPU) no está inicializada."
        elif not sim_alive:
            message = "Error: El hilo de simulación está inactivo."

    response_data = {
        "status": status,
        "message": message,
        "details": {
            "simulation_running": sim_alive,
            "piston_initialized": piston_initialized
        }
    }
    return jsonify(response_data), http_code


@app.route('/api/state', methods=['GET'])
def get_piston_state():
    """
    Devuelve el estado dinámico completo de la IPU en formato JSON.

    Este endpoint es de solo lectura y es seguro para ser llamado
    concurrentemente gracias al uso de un lock.
    """
    if not ipu_instance:
        return jsonify({"status": "error", "message": "IPU no inicializada."}), 503

    with ipu_lock:
        state_data = {
            "position": ipu_instance.position,
            "velocity": ipu_instance.velocity,
            "acceleration": ipu_instance.acceleration,
            "mode": ipu_instance.mode.value,
            "stored_energy": ipu_instance.stored_energy,
            "current_charge": ipu_instance.current_charge,
            "circuit_voltage": ipu_instance.circuit_voltage,
            "circuit_current": ipu_instance.circuit_current,
            "output_voltage": ipu_instance.output_voltage,
            "battery_mode_status": {
                "is_discharging": ipu_instance.battery_is_discharging
            },
            "control_targets": {
                "target_speed": ipu_instance.target_speed,
                "target_energy": ipu_instance.target_energy
            }
        }

    return jsonify({"status": "success", "state": state_data})


@app.route('/api/control', methods=['POST'])
def set_piston_control():
    """
    Acepta una señal de control para modular el objetivo de energía del pistón.

    Payload esperado: {"control_signal": float}
    """
    data = request.get_json()
    if not data or "control_signal" not in data:
        return jsonify({"status": "error", "message": "Payload JSON inválido o falta 'control_signal'."}), 400

    try:
        signal = float(data["control_signal"])
    except (ValueError, TypeError):
        return jsonify({"status": "error", "message": "'control_signal' debe ser un valor numérico."}), 400

    if not ipu_instance:
        return jsonify({"status": "error", "message": "IPU no inicializada."}), 503

    with ipu_lock:
        # La señal de control modula directamente el objetivo de energía.
        # Se asegura que no sea negativo.
        new_target_energy = max(0.0, signal)
        ipu_instance.set_energy_target(new_target_energy)
        current_target = ipu_instance.target_energy

    logger.info(f"Señal de control recibida: {signal}. Nuevo objetivo de energía: {current_target:.2f} J")

    return jsonify({
        "status": "success",
        "message": "Objetivo de energía ajustado.",
        "new_energy_target": current_target
    })


@app.route('/api/config', methods=['GET'])
def get_piston_config():
    """
    Devuelve la configuración estática completa de la IPU.

    Este endpoint es útil para depuración, monitoreo y para que otros
    servicios entiendan las capacidades y límites de esta instancia de IPU.
    """
    if not ipu_instance:
        return jsonify({"status": "error", "message": "IPU no inicializada."}), 503

    with ipu_lock:
        # Extraer las ganancias de los controladores PID para una respuesta más limpia
        speed_pid_gains = {
            "kp": ipu_instance.speed_controller.kp,
            "ki": ipu_instance.speed_controller.ki,
            "kd": ipu_instance.speed_controller.kd,
            "output_limit": ipu_instance.speed_controller.output_limit
        }
        energy_pid_gains = {
            "kp": ipu_instance.energy_controller.kp,
            "ki": ipu_instance.energy_controller.ki,
            "kd": ipu_instance.energy_controller.kd,
            "output_limit": ipu_instance.energy_controller.output_limit
        }

        config_data = {
            "physical_params": {
                "capacity": ipu_instance.capacity,
                "elasticity_k": ipu_instance.k,
                "damping_c": ipu_instance.c,
                "piston_mass_m": ipu_instance.m,
                "nonlinear_elasticity": ipu_instance.nonlinear_elasticity,
            },
            "transducer_params": {
                "type": ipu_instance.transducer_type.value,
                "voltage_sensitivity": ipu_instance.voltage_sensitivity,
                "force_sensitivity": ipu_instance.force_sensitivity,
                "internal_resistance": ipu_instance.internal_resistance,
            },
            "friction_params": {
                "model": ipu_instance.friction_model.value,
                "coulomb_friction": ipu_instance.coulomb_friction,
                "stribeck_coeffs": ipu_instance.stribeck_coeffs,
            },
            "electrical_equivalent_params": {
                "equivalent_capacitance": ipu_instance.equivalent_capacitance,
                "equivalent_inductance": ipu_instance.equivalent_inductance,
                "equivalent_resistance": ipu_instance.equivalent_resistance,
                "output_capacitance": ipu_instance.output_capacitance,
                "converter_efficiency": ipu_instance.converter_efficiency,
            },
            "operational_params": {
                "capacitor_discharge_threshold": ipu_instance.capacitor_discharge_threshold,
                "hysteresis_factor": ipu_instance.hysteresis_factor,
            },
            "pid_gains": {
                "speed_controller": speed_pid_gains,
                "energy_controller": energy_pid_gains,
            }
        }

    return jsonify({"status": "success", "config": config_data})


@app.route('/api/command', methods=['POST'])
def execute_piston_command():
    """
    Ejecuta un comando avanzado en la IPU.

    Payload esperado: {"command": str, "value": any}
    Comandos Soportados:
    - set_mode: value in ["capacitor", "battery"]
    - trigger_discharge: value is boolean
    - set_energy_target: value is float >= 0
    - set_speed_target: value is float
    - reset: value is ignored
    """
    data = request.get_json()
    if not data or "command" not in data:
        return jsonify({"status": "error", "message": "Payload JSON inválido o falta 'command'."}), 400

    command = data.get("command")
    value = data.get("value") # Value puede ser opcional para comandos como 'reset'

    if not ipu_instance:
        return jsonify({"status": "error", "message": "IPU no inicializada."}), 503

    message = ""

    with ipu_lock:
        try:
            if command == "set_mode":
                if value not in [m.value for m in PistonMode]:
                    raise ValueError(f"Modo inválido. Válidos: {[m.value for m in PistonMode]}")
                mode = PistonMode(value)
                ipu_instance.set_mode(mode)
                message = f"Modo cambiado a {mode.value}."

            elif command == "trigger_discharge":
                if not isinstance(value, bool):
                    raise ValueError("El valor para 'trigger_discharge' debe ser booleano (true/false).")
                ipu_instance.trigger_discharge(value)
                status = "activada" if value else "desactivada"
                message = f"Descarga en modo batería {status}."

            elif command == "set_energy_target":
                target = float(value)
                if target < 0:
                    raise ValueError("El objetivo de energía no puede ser negativo.")
                ipu_instance.set_energy_target(target)
                message = f"Objetivo de energía establecido en {target:.2f} J."

            elif command == "set_speed_target":
                target = float(value)
                ipu_instance.set_speed_target(target)
                message = f"Objetivo de velocidad establecido en {target:.2f} m/s."

            elif command == "reset":
                ipu_instance.reset()
                message = "El estado del pistón ha sido reiniciado."

            else:
                # Si el comando no es ninguno de los conocidos, retornar error.
                return jsonify({"status": "error", "message": f"Comando desconocido: '{command}'."}), 400

        except (ValueError, TypeError, KeyError) as e:
            # Captura errores de conversión de tipo (e.g., float("abc"))
            # o de enum (e.g., PistonMode("invalid"))
            logger.error(f"Error procesando comando '{command}' con valor '{value}': {e}")
            return jsonify({"status": "error", "message": f"Valor inválido para el comando '{command}': {e}"}), 400

    logger.info(f"Comando '{command}' ejecutado. Mensaje: {message}")
    return jsonify({"status": "success", "message": message})


# --- Punto de Entrada Principal ---

def main():
    """
    Función principal para inicializar y ejecutar el microservicio del pistón.

    Orquesta la carga de configuración, la inicialización del 'gemelo digital',
    el registro con AgentAI, y el arranque de los hilos de simulación y del
    servidor web.
    """
    global ipu_instance, simulation_thread, config

    # --- 1. Cargar configuración centralizada ---
    try:
        config = PistonConfig()
    except (ValueError, TypeError) as e:
        logger.exception(f"Error al leer la configuración de entorno: {e}")
        return  # Salir si la configuración es inválida

    logger.info("--- Configuración de la IPU ---")
    logger.info(f"  Capacidad: {config.capacity}, Elasticidad: {config.elasticity}")
    logger.info(f"  Amortiguación: {config.damping}, Masa: {config.mass}")
    logger.info(f"  Modelo de Fricción: {config.friction_model.value}")
    logger.info("---------------------------------")

    # --- 2. Inicializar la instancia global del pistón (gemelo digital) ---
    try:
        ipu_instance = AtomicPiston(
            capacity=config.capacity,
            elasticity=config.elasticity,
            damping=config.damping,
            piston_mass=config.mass,
            friction_model=config.friction_model
        )
        logger.info("Instancia global de AtomicPiston creada exitosamente.")
    except ValueError as e:
        logger.exception(f"Error crítico al inicializar AtomicPiston: {e}")
        return  # Salir si la inicialización del modelo falla

    # --- 3. Registrar el servicio con AgentAI ---
    module_name = "atomic_piston_service"
    hostname = os.environ.get("HOSTNAME", module_name)
    module_url = f"http://{hostname}:{config.service_port}"
    health_url = f"{module_url}/api/health"
    description = "Microservicio que simula una Unidad de Potencia Inteligente (IPU) y expone una API para su control."

    registration_successful = register_with_agent_ai(
        module_name, module_url, health_url, description
    )
    if not registration_successful:
        logger.warning("El servicio continuará sin estar registrado en AgentAI.")

    # --- 4. Iniciar el hilo de simulación ---
    stop_simulation_event.clear()
    simulation_thread = threading.Thread(
        target=simulation_loop,
        daemon=True,
        name="PistonSimLoop"
    )
    simulation_thread.start()
    logger.info("Hilo de simulación del pistón iniciado.")

    # --- 5. Iniciar el servidor Flask ---
    logger.info(f"Iniciando servidor Flask de Atomic Piston en http://0.0.0.0:{config.service_port}")
    app.run(host="0.0.0.0", port=config.service_port, debug=False, use_reloader=False)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Señal de detención recibida (KeyboardInterrupt).")
    finally:
        logger.info("Iniciando secuencia de apagado del servicio...")
        stop_simulation_event.set()
        if simulation_thread and simulation_thread.is_alive():
            logger.info("Esperando a que el hilo de simulación finalice...")
            simulation_thread.join(timeout=2)
            if simulation_thread.is_alive():
                logger.warning("El hilo de simulación no finalizó a tiempo.")
        logger.info("Servicio Atomic Piston detenido.")

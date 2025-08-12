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
    """Bucle principal de simulación que se ejecuta en un hilo de fondo.

    Este bucle infinito se encarga de actualizar el estado del `AtomicPiston`
    a intervalos regulares definidos por `config.simulation_interval`.
    Orquesta la simulación del 'gemelo digital' y maneja la detención
    grácil a través de un `threading.Event`.
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
    """Intenta registrar el microservicio en el AgentAI.

    Realiza una solicitud HTTP POST al endpoint de registro de AgentAI para
    anunciar la disponibilidad de este servicio en el ecosistema.
    Implementa una lógica de reintentos en caso de fallo de conexión.

    Args:
        module_name (str): El nombre del módulo a registrar.
        module_url (str): La URL base del servicio.
        health_url (str): La URL del endpoint de salud del servicio.
        description (str): Una descripción de la funcionalidad del servicio.

    Returns:
        bool: True si el registro fue exitoso, False en caso contrario.
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
    """Simula una Unidad de Potencia Inteligente (IPU) o Pistón Atómico.

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
        dt (float): El último paso de tiempo utilizado en la simulación [s].
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
        """Inicializa una nueva instancia de AtomicPiston.

        Args:
            capacity (float): Capacidad máxima de compresión [m]. Debe ser > 0.
            elasticity (float): Constante elástica (k) [N/m]. Debe ser >= 0.
            damping (float): Coeficiente de amortiguación (c) [N·s/m]. Debe ser >= 0.
            piston_mass (float): Masa inercial (m) [kg]. Debe ser > 0.
            mode (PistonMode): Modo de operación inicial del pistón.
            transducer_type (TransducerType): Tipo de transductor a utilizar.
            friction_model (FrictionModel): Modelo de fricción a simular.
            coulomb_friction (float): Coeficiente de fricción de Coulomb.
            stribeck_coeffs (Tuple[float, float, float]): Coeficientes de Stribeck
                (estática, Coulomb, velocidad).
            nonlinear_elasticity (float): Coeficiente para el término de
                elasticidad no lineal.

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
        """Calcula la carga actual del pistón como compresión positiva.

        La carga se define como la compresión del pistón desde su punto de
        equilibrio (posición 0), donde una mayor compresión resulta en una
        mayor carga.

        Returns:
            float: La compresión actual del pistón en metros [m].
        """
        return max(0, -self.position)

    @property
    def stored_energy(self) -> float:
        """Calcula la energía mecánica total almacenada en el pistón.

        Combina la energía potencial elástica (lineal y no lineal) y la
        energía cinética del pistón.

        Returns:
            float: La energía mecánica total almacenada en el sistema [J].
        """
        potential = (
            0.5 * self.k * self.position**2
            + (1 / 4) * self.nonlinear_elasticity * self.position**4
        )
        kinetic = 0.5 * self.m * self.velocity**2
        return potential + kinetic

    def calculate_friction(self, driving_force: float) -> float:
        """Calcula la fuerza de fricción seca (no viscosa) según el modelo.

        El amortiguamiento viscoso general (-c * v) se calcula por separado en la
        ecuación de movimiento. Esta función maneja solo los modelos de fricción
        seca como Coulomb y Stribeck.

        Args:
            driving_force (float): La fuerza neta (sin amortiguamiento) que
                intenta mover el pistón. Se usa para determinar la dirección
                de la fricción estática.

        Returns:
            float: La fuerza de fricción seca calculada [N].
        """
        # Fricción cinética (cuando hay movimiento)
        if abs(self.velocity) > 1e-5:
            if self.friction_model == FrictionModel.COULOMB:
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
            if self.friction_model == FrictionModel.COULOMB:
                # La fricción estática se opone a la fuerza impulsora hasta su límite máximo.
                return -np.sign(driving_force) * min(abs(driving_force), self.coulomb_friction)

            elif self.friction_model == FrictionModel.STRIBECK:
                f_static, _, _ = self.stribeck_coeffs
                # La fricción estática se opone a la fuerza impulsora hasta su límite estático.
                return -np.sign(driving_force) * min(abs(driving_force), f_static)

        # Si el modelo no es ni Coulomb ni Stribeck, no hay fricción seca.
        return 0.0

    def apply_force(self, signal_value: float, source: str, mass_factor: float = 1.0) -> None:
        """Aplica una fuerza mecánica externa basada en una señal de entrada.

        La fuerza se calcula usando la energía cinética derivada de la velocidad
        de la señal de entrada y se acumula en `self.last_applied_force`.

        Args:
            signal_value (float): El valor de la señal de entrada.
            source (str): Identificador de la fuente de la señal para seguimiento.
            mass_factor (float): Un factor para escalar la fuerza aplicada.
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
        """Aplica una señal eléctrica que se traduce en una fuerza mecánica.

        La conversión de voltaje a fuerza depende del tipo de transductor y
        se acumula en `self.last_applied_force`.

        Args:
            voltage (float): El voltaje de la señal eléctrica de entrada [V].
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
        """Actualiza el estado físico del pistón para un intervalo de tiempo `dt`.

        Resuelve numéricamente la Ecuación Diferencial Ordinaria (EDO) del
        movimiento del pistón usando un integrador Runge-Kutta de 4º orden.

        La EDO gobernante es:
        m*x'' + c*x' + F_fricción_seca + k*x + ε*x³ = F_externa(t)

        Interpretación Física de los Términos:
        - m*x'': Término de Inercia. Representa la resistencia del pistón a
          cambiar su estado de movimiento.
        - c*x': Amortiguamiento Viscoso. Representa la pérdida de energía
          proporcional a la velocidad (ej. fricción con un fluido).
        - F_fricción_seca: Fricción de Coulomb/Stribeck. Representa la
          fricción que no depende (o depende poco) de la velocidad.
        - k*x: Fuerza Elástica Lineal (Ley de Hooke). La fuerza restauradora
          principal del resorte.
        - ε*x³: Fuerza Elástica No Lineal. Modela cómo el material del resorte
          se vuelve más rígido o blando a grandes deformaciones.
        - F_externa(t): Fuerzas Impulsoras. La suma de todas las fuerzas
          externas aplicadas, incluyendo las de control.

        Args:
            dt (float): El intervalo de tiempo para la integración [s].

        Raises:
            ValueError: Si `dt` es menor o igual a cero.
        """
        if dt <= 0:
            raise ValueError("El paso de tiempo (dt) debe ser un valor positivo.")
        self.dt = dt

        # 1. FUERZAS IMPULSORAS (F_externa)
        # Suma de todas las fuerzas externas y de control que mueven el sistema.
        external_force = self.last_applied_force
        if self.target_speed != 0.0:
            external_force += self.speed_controller.update(
                self.target_speed, self.velocity, dt
            )

        # Precalcular términos constantes para eficiencia
        k = self.k
        c = self.c
        m = self.m
        nl = self.nonlinear_elasticity
        capacity = self.capacity

        # Función de derivadas optimizada
        def derivatives(state: np.ndarray, ext_force: float) -> np.ndarray:
            pos, vel = state

            # 2. FUERZA ELÁSTICA (k*x + ε*x³)
            # La fuerza restauradora del resorte, combinando su
            # comportamiento lineal y no lineal.
            pos_sq = pos * pos
            spring_force = -k * pos - nl * pos_sq * pos

            # 3. FUERZA DE FRICCIÓN SECA (F_fricción_seca)
            # Fricción que se opone al inicio del movimiento (estática) o al
            # movimiento a velocidad constante (cinética).
            driving_force = ext_force + spring_force
            friction_force = self.calculate_friction(driving_force)

            # 4. FUERZA DE AMORTIGUAMIENTO VISCOSO (c*x')
            # Pérdida de energía por fricción con el medio, proporcional a la velocidad.
            damping_force = -c * vel

            # Suma de todas las fuerzas para obtener la fuerza neta.
            total_force = ext_force + spring_force + friction_force + damping_force

            # 5. RESPUESTA INERCIAL (m*x'')
            # La aceleración resultante, que es la manifestación de la inercia
            # del pistón resistiéndose al cambio de movimiento.
            acceleration = total_force / m

            return np.array([vel, acceleration])

        # Integración RK4
        state = np.array([self.position, self.velocity])
        k1 = derivatives(state, external_force)
        k2 = derivatives(state + 0.5 * dt * k1, external_force)
        k3 = derivatives(state + 0.5 * dt * k2, external_force)
        k4 = derivatives(state + dt * k3, external_force)

        new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.position, self.velocity = new_state

        # Manejar límites físicos con conservación de energía
        if abs(self.position) > capacity:
            # Calcular energía antes del ajuste
            energy_before = self.stored_energy

            # Aplicar restricción conservando la dirección
            self.position = np.clip(self.position, -capacity, capacity)

            # Ajustar velocidad para conservar energía (coeficiente de restitución)
            energy_after = self.stored_energy
            if energy_before > 0 and energy_after < energy_before:
                velocity_scale = math.sqrt(energy_after / energy_before)
                self.velocity *= velocity_scale * 0.8  # Factor de pérdida

        # Actualizar aceleración para reporte
        self.acceleration = derivatives(new_state, external_force)[1]

        # Actualizar sistemas electrónicos y resetear fuerzas
        self.update_electronic_state()
        self.last_applied_force = 0.0

        # Registrar métricas
        self.energy_history.append(self.stored_energy)
        self.efficiency_history.append(self.get_conversion_efficiency())

        # Calcular fricción para registro (usando estado actual)
        current_spring_force = -k * self.position - nl * self.position**3
        friction = self.calculate_friction(external_force + current_spring_force)
        self.friction_force_history.append(friction)

    def update_electronic_state(self) -> None:
        """Actualiza el estado del circuito electrónico equivalente.

        Calcula el voltaje y la corriente del circuito basándose en la posición
        y velocidad actuales del pistón. También actualiza la carga acumulada
        y procesa la salida eléctrica.
        """
        self.circuit_voltage = -self.position * self.voltage_sensitivity
        self.circuit_current = self.velocity * self.equivalent_capacitance
        self.charge_accumulated += self.circuit_current * self.dt
        self.process_electrical_output()

    def process_electrical_output(self) -> None:
        """Procesa la energía eléctrica generada a través del convertidor."""
        input_power = abs(self.circuit_voltage * self.circuit_current)
        output_power = input_power * self.converter_efficiency
        if self.equivalent_resistance > 0:
            self.output_voltage = np.sqrt(output_power * self.equivalent_resistance)
        else:
            self.output_voltage = 0.0

    def discharge(self, dt: float) -> Optional[Dict[str, Any]]:
        """Gestiona la descarga de energía según el modo de operación.

        Delega la lógica de descarga al método correspondiente (`capacitor` o
        `battery`). El control de energía se maneja en `update_state`.

        Args:
            dt (float): El intervalo de tiempo de la simulación [s].

        Returns:
            Optional[Dict[str, Any]]: Un diccionario con información sobre la
            descarga si ocurre, de lo contrario None.
        """
        # El control de energía ahora se maneja completamente en update_state
        if self.mode == PistonMode.CAPACITOR:
            return self.capacitor_discharge(dt)
        elif self.mode == PistonMode.BATTERY:
            return self.battery_discharge(dt)
        return None

    def capacitor_discharge(self, dt: float) -> Optional[Dict[str, Any]]:
        """Realiza una descarga en modo CAPACITOR si se cumplen las condiciones.

        Verifica si la posición del pistón ha alcanzado el umbral de descarga.
        Si es así, simula un pulso de energía, ajusta la posición para
        reflejar la histéresis y devuelve un diccionario con los detalles.

        Args:
            dt (float): El intervalo de tiempo de la simulación [s]. Aunque la
                descarga es instantánea, se mantiene por consistencia.

        Returns:
            Optional[Dict[str, Any]]: Un diccionario con los detalles de la
            descarga en forma de pulso si ocurre, de lo contrario None.
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
        """Realiza una descarga continua en modo BATTERY si está activada.

        Si la descarga está activa y hay carga, reduce la compresión del pistón
        a una tasa definida. La descarga se detiene si se agota la carga.

        Args:
            dt (float): El intervalo de tiempo para la descarga [s].

        Returns:
            Optional[Dict[str, Any]]: Un diccionario con detalles de la descarga
            sostenida si ocurre, de lo contrario None.
        """
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
        """Simula la descarga de energía a través de una carga externa.

        Calcula cómo la energía almacenada en el pistón se disipa a través de
        una resistencia de carga, afectando la posición del pistón.

        Args:
            load_resistance (float): Resistencia de la carga externa [Ω].
            dt (float): Intervalo de tiempo de la simulación [s].

        Returns:
            Tuple[float, float, float]: Una tupla con el voltaje, la corriente
            y la potencia en la carga.
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
        """Configura la dirección de compresión del pistón.

        Args:
            direction (int): La dirección de compresión, donde -1 indica que
                las fuerzas externas comprimen el pistón (dirección negativa)
                y 1 que lo expanden.
        """
        if direction not in (-1, 1):
            logger.warning(f"Dirección inválida: {direction}. Usando -1 (compresión).")
            self.compression_direction = -1
        else:
            self.compression_direction = direction
            logger.info(f"Dirección de compresión establecida en: {self.compression_direction}")

    def get_conversion_efficiency(self) -> float:
        """Calcula la eficiencia de conversión de energía instantánea.

        Compara la energía mecánica almacenada con la energía eléctrica total
        del sistema para determinar la eficiencia de la transducción.

        Returns:
            float: La eficiencia de conversión instantánea (0 a 1).
        """
        mechanical_energy = self.stored_energy
        electrical_energy = 0.5 * self.equivalent_capacitance * self.circuit_voltage**2
        total_stored_energy = mechanical_energy + electrical_energy
        if total_stored_energy > 1e-9:
            return mechanical_energy / total_stored_energy
        return 0.0

    def set_mode(self, mode: PistonMode) -> None:
        """Establece el modo de operación del pistón.

        Args:
            mode (PistonMode): El nuevo modo de operación, que puede ser
                `PistonMode.CAPACITOR` o `PistonMode.BATTERY`.
        """
        self.mode = mode
        self.battery_is_discharging = False
        logger.info(f"Modo cambiado a: {mode.value}")

    def trigger_discharge(self, discharge_on: bool) -> None:
        """Activa o desactiva la descarga continua en modo BATTERY.

        Este método solo tiene efecto si el pistón está en modo BATTERY.

        Args:
            discharge_on (bool): True para activar la descarga, False para
                desactivarla.
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
        """Establece la velocidad objetivo para el controlador PID de velocidad.

        Args:
            target (float): La velocidad objetivo deseada [m/s].
        """
        self.target_speed = target
        self.speed_controller.reset()
        logger.info(f"Objetivo de velocidad establecido: {target:.2f} m/s")

    def set_energy_target(self, target: float) -> None:
        """Establece la energía objetivo para el controlador PID de energía.

        Args:
            target (float): La energía objetivo deseada [J].
        """
        self.target_energy = target
        self.energy_controller.reset()
        logger.info(f"Objetivo de energía establecido: {target:.2f} J")

    def generate_bode_data(self, frequency_range: np.ndarray) -> Dict[str, Any]:
        """Genera datos para un diagrama de Bode de la función de transferencia.

        Args:
            frequency_range (np.ndarray): Un array de NumPy con el rango de
                frecuencias a analizar [Hz].

        Returns:
            Dict[str, Any]: Un diccionario con los datos para el diagrama de
            Bode, incluyendo 'frequencies', 'magnitude' y 'phase'.
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
        """Exporta los historiales de simulación a un archivo CSV.

        Args:
            filename (str): La ruta del archivo CSV donde se guardarán los datos.

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
        """Simula la respuesta del pistón a una entrada de fuerza escalón.

        Args:
            force_amplitude (float): Magnitud de la fuerza a aplicar [N].
            duration (float): Duración total de la simulación [s].
            dt (float): Paso de tiempo para la simulación [s].

        Returns:
            Dict[str, Any]: Un diccionario con las series temporales de
            'time', 'position', 'velocity' y 'acceleration'.
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
        """Simula la respuesta del pistón a una entrada de fuerza impulso.

        Args:
            impulse_magnitude (float): Magnitud del impulso a aplicar [N·s].
            duration (float): Duración total de la simulación [s].
            dt (float): Paso de tiempo para la simulación [s].

        Returns:
            Dict[str, Any]: Un diccionario con las series temporales de
            'time', 'position', 'velocity' y 'acceleration'.
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
        """Reinicia el estado del pistón a sus condiciones iniciales."""
        self.position = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0
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
        """Devuelve los términos de la ecuación diferencial del movimiento.

        La ecuación es: m*a + F_amortiguamiento + F_resorte = F_externa.

        Returns:
            Dict[str, float]: Un diccionario con los componentes calculados.
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
    """Implementa un controlador Proporcional-Integral-Derivativo (PID).

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
        """Inicializa el controlador PID.

        Args:
            kp (float): Ganancia proporcional (P).
            ki (float): Ganancia integral (I).
            kd (float): Ganancia derivativa (D).
            output_limit (float): Límite absoluto para la salida (anti-windup).
        """
        self.kp: float = kp
        self.ki: float = ki
        self.kd: float = kd
        self.integral: float = 0.0
        self.previous_error: float = 0.0
        self.output_limit: float = output_limit

    def update(self, setpoint: float, current_value: float, dt: float) -> float:
        """Calcula la salida del controlador PID para un paso de tiempo.

        Args:
            setpoint (float): El valor deseado o de referencia.
            current_value (float): El valor medido actual del sistema.
            dt (float): El intervalo de tiempo desde la última actualización [s].

        Returns:
            float: La señal de control calculada.
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
    """Verifica el estado de salud del servicio.

    Retorna el estado de la simulación y si la instancia del pistón ha sido
    inicializada. Devuelve un código 503 si el servicio no está operativo.

    Returns:
        Response: Un objeto de respuesta JSON con el estado de salud.
            Ejemplo en caso de éxito:
            {
                "status": "success",
                "message": "Servicio Atomic Piston operativo.",
                "details": {
                    "simulation_running": true,
                    "piston_initialized": true
                }
            }
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
    """Devuelve el estado dinámico completo de la IPU.

    Este endpoint es de solo lectura y es seguro para ser llamado
    concurrentemente gracias al uso de un lock.

    Returns:
        Response: Un objeto de respuesta JSON con el estado completo del
        pistón, o un error 503 si no está inicializado.
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
    """Acepta una señal para modular el objetivo de energía del pistón.

    Payload JSON esperado:
    {
        "control_signal": float
    }

    Returns:
        Response: Una confirmación del cambio o un error 400 si el payload
        es inválido.
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
    """Devuelve la configuración estática completa de la IPU.

    Este endpoint es útil para depuración y para que otros servicios
    entiendan las capacidades y límites de esta instancia de IPU.

    Returns:
        Response: Un objeto de respuesta JSON con la configuración completa.
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
    """Ejecuta un comando avanzado en la IPU.

    Payload JSON esperado:
    {
        "command": "<nombre_del_comando>",
        "value": <valor_del_comando>
    }

    Comandos Soportados:
    - `set_mode`: value in ["capacitor", "battery"]
    - `trigger_discharge`: value is boolean
    - `set_energy_target`: value is float >= 0
    - `set_speed_target`: value is float
    - `reset`: value is ignored

    Returns:
        Response: Una confirmación del comando o un error 400/503.
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
    """Inicializa y ejecuta el microservicio del pistón.

    Orquesta la carga de configuración desde `PistonConfig`, la inicialización
    del 'gemelo digital' `AtomicPiston`, el registro con AgentAI, y el
    arranque de los hilos de simulación y del servidor web Flask.
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

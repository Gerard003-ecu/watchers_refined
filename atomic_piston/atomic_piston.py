# atomic_piston/atomic_piston.py
import time
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PistonMode(Enum):
    CAPACITOR = "capacitor"
    BATTERY = "battery"


class AtomicPiston:
    """
    Una unidad de potencia inteligente (IPU) que modela un resonador de señales
    basado en un sistema físico de masa-resorte-amortiguador.
    Acumula "fuerza" de las señales del sistema y la convierte en "energía potencial"
    (compresión del pistón) que puede ser liberada de forma controlada.
    """
    def __init__(self,
                 capacity: float,
                 elasticity: float,  # Constante del resorte (k). Alta = más rígido.
                 damping: float,  # Coeficiente de amortiguación (c).
                 piston_mass: float = 1.0,  # Masa inercial del pistón (m).
                 mode: PistonMode = PistonMode.CAPACITOR):
        """Inicializa una nueva instancia de AtomicPiston.

        Args:
            capacity: Capacidad máxima de compresión del pistón.
            elasticity: Constante elástica del resorte (k).
                        Un valor alto significa un resorte más rígido.
            damping: Coeficiente de amortiguación (c).
            piston_mass: Masa inercial del pistón (m).
            mode: Modo de operación del pistón (CAPACITOR o BATTERY).
        """
        self.capacity = capacity  # Límite máximo de compresión (-capacity).
        self.mode = mode

        # Parámetros Físicos
        self.k = elasticity
        self.c = damping
        self.m = piston_mass

        # Estado del Pistón
        # Posición de equilibrio es 0. Negativo = comprimido/cargado.
        self.position = 0.0
        self.velocity = 0.0  # Velocidad actual del pistón.
        self.last_applied_force = 0.0

        # Para calcular la velocidad de la señal de entrada
        # Almacenará {'source': {'value': float, 'timestamp': float}}
        self.last_signal_info = {}

        # Parámetros de Modo
        # Umbral de posición para descarga automática
        self.capacitor_discharge_threshold = -capacity * 0.9
        self.battery_is_discharging = False
        # Cuánta "posición" se libera por segundo
        self.battery_discharge_rate = capacity * 0.05

    @property
    def current_charge(self) -> float:
        """Devuelve la carga actual del pistón.

        La carga se define como la compresión del pistón, que es el valor
        absoluto de la posición negativa del pistón. Una carga mayor indica
        una mayor compresión.

        Returns:
            La carga actual como un valor flotante no negativo.
        """
        return max(0, -self.position)

    def apply_force(self, signal_value: float, source: str,
                    mass_factor: float = 1.0):
        """Aplica una fuerza al pistón basada en una señal entrante.

        Calcula la "fuerza de impacto" de una señal y la aplica al pistón.
        La fuerza es proporcional a la 'masa' de la señal (influenciada por
        `mass_factor`) y al cuadrado de su 'velocidad' (tasa de cambio).
        La fuerza aplicada siempre actúa para comprimir el pistón (dirección
        negativa).

        Args:
            signal_value: El valor actual de la señal.
            source: Un identificador de la fuente de la señal, usado para rastrear
                    la velocidad de cambio de señales individuales.
            mass_factor: Un factor para escalar la 'masa' efectiva de la señal,
                         afectando la magnitud de la fuerza aplicada.
        """
        current_time = time.monotonic()

        # Calcular la velocidad de la señal (derivada)
        if source in self.last_signal_info:
            last_val = self.last_signal_info[source]['value']
            last_time = self.last_signal_info[source]['timestamp']
            dt = current_time - last_time

            if dt > 1e-6:  # Evitar división por cero
                signal_velocity = (signal_value - last_val) / dt
            else:
                signal_velocity = 0.0
        else:
            signal_velocity = 0.0

        # Actualizar para el próximo cálculo
        self.last_signal_info[source] = {'value': signal_value,
                                         'timestamp': current_time}

        # Calcular la fuerza de impacto (nuestra "energía cinética")
        # Usamos el valor absoluto de la velocidad, ya que el impacto es energía,
        # sin dirección.
        # El signo de la fuerza siempre es para comprimir el pistón (negativo).
        force = -0.5 * mass_factor * (signal_velocity ** 2)
        self.last_applied_force = force

        logger.debug(
            f"Fuente '{source}': valor={signal_value:.2f}, "
            f"vel={signal_velocity:.2f}, masa={mass_factor:.2f} -> "
            f"Fuerza={force:.2f}"
        )

    def update_state(self, dt: float):
        """Actualiza el estado del pistón durante un intervalo de tiempo delta `dt`.

        Calcula la nueva posición y velocidad del pistón basándose en las fuerzas
        aplicadas (externas e internas como la del resorte y amortiguador)
        durante el intervalo de tiempo `dt`.
        Este método utiliza la integración de Euler para la simulación física.
        La posición del pistón está limitada por su capacidad máxima de compresión.
        La fuerza externa aplicada (`last_applied_force`) se considera instantánea
        y se resetea después de cada actualización.

        Args:
            dt: El intervalo de tiempo (delta t) para la actualización del estado.
        """
        # Calcular las fuerzas internas del sistema
        # Ley de Hooke: se opone al desplazamiento
        spring_force = -self.k * self.position
        damping_force = -self.c * self.velocity  # Se opone al movimiento

        # Fuerza total
        total_force = self.last_applied_force + spring_force + damping_force

        # Calcular la aceleración (F=ma -> a=F/m)
        acceleration = total_force / self.m

        # Actualizar estado usando integración de Euler (método numérico simple)
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

        # Limitar la compresión a la capacidad máxima
        self.position = max(-self.capacity, self.position)

        # Resetear la fuerza externa después de aplicarla por un instante
        self.last_applied_force = 0.0

    def discharge(self):
        """Gestiona la descarga de energía del pistón según su modo de operación.

        En modo CAPACITOR:
            Si la posición del pistón alcanza o supera el umbral de descarga,
            libera una señal de tipo "pulso" cuya amplitud es la carga actual.
            La posición se resetea a 0 y se le imprime una velocidad de rebote.
        En modo BATTERY:
            Si `battery_is_discharging` es verdadero y hay carga almacenada,
            reduce la compresión del pistón (libera posición) a una tasa definida
            por `battery_discharge_rate`. Emite una señal "sostenida" de amplitud
            fija.
            Si la carga llega a cero, `battery_is_discharging` se vuelve falso.

        Returns:
            Un diccionario representando la señal de salida si ocurre una descarga,
            o None si no hay descarga.
            Ejemplo de señal de CAPACITOR: {'type': 'pulse', 'amplitude': float}
            Ejemplo de señal de BATTERY: {'type': 'sustained', 'amplitude': 1.0}
        """
        if self.mode == PistonMode.CAPACITOR:
            # Descarga automática si está suficientemente comprimido
            if self.position <= self.capacitor_discharge_threshold:
                output_signal = {"type": "pulse",
                                 "amplitude": self.current_charge}
                logger.info(
                    f"¡Descarga CAPACITOR! Amplitud: {self.current_charge:.2f}"
                )
                # El "rebote": resetea la posición y le da una velocidad de salida
                self.position = 0.0
                self.velocity = 2.0  # Simula el rebote
                return output_signal

        elif self.mode == PistonMode.BATTERY:
            if self.battery_is_discharging:  # Check if discharging is active
                if self.current_charge > 0:
                    # La descarga consume la "compresión" (energía potencial)
                    # Asumimos que discharge es llamado en cada paso de simulación,
                    # similar a update_state.
                    # Por lo tanto, el dt para la descarga es el mismo que el
                    # dt de update_state.
                    # Para desacoplarlo, necesitaríamos pasar dt
                    # como argumento a discharge.
                    # Por ahora, usaremos una aproximación basada en una tasa por
                    # segundo.
                    # Si UPDATE_INTERVAL no está definido globalmente, debemos
                    # manejarlo.
                    # Una mejor práctica sería pasar dt a discharge.
                    # Por ahora, si UPDATE_INTERVAL no está definido,
                    # asumiremos un dt pequeño, por ejemplo 0.01s.
                    try:
                        # Esta variable global no está definida en este archivo.
                        # Considerar pasar dt como argumento o usar una constante.
                        update_interval = UPDATE_INTERVAL
                    except NameError:
                        update_interval = 0.01  # Asumir un dt pequeño

                    position_released = (self.battery_discharge_rate *
                                         update_interval)
                    self.position = min(0, self.position + position_released)

                if self.current_charge == 0:
                    self.battery_is_discharging = False
                    logger.info(
                        "Descarga BATTERY completada."
                    )

                return {"type": "sustained", "amplitude": 1.0}
            else:  # current_charge is 0 or less
                self.battery_is_discharging = False
                logger.info(
                    "Descarga BATTERY: No hay carga para liberar, "
                    "desactivando descarga."
                )

        return None

    def set_mode(self, mode: PistonMode):
        """
        Establece el modo de operación del pistón.

        Args:
            mode:
            El nuevo modo de operación (PistonMode.CAPACITOR o
            PistonMode.BATTERY).
        """
        self.mode = mode
        self.battery_is_discharging = False
        logger.info(f"Modo del pistón cambiado a: {mode.value}")

    def trigger_discharge(self, discharge_on: bool):
        """
        Activa o desactiva la descarga continua en modo BATTERY.

        Este método solo tiene efecto si el pistón está en modo BATTERY.

        Args:
            discharge_on:
            True para activar la descarga, False para desactivarla.
        """
        if self.mode == PistonMode.BATTERY:
            self.battery_is_discharging = discharge_on
            if discharge_on:
                logger.info(
                    "Descarga en modo BATTERY activada."
                )
            else:
                logger.info(
                    "Descarga en modo BATTERY desactivada."
                )
        else:
            logger.warning(
                f"trigger_discharge llamado en modo {self.mode.value}. "
                "Solo tiene efecto en modo BATTERY."
            )

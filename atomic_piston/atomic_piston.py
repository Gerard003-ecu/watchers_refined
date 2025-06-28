# atomic_piston/atomic_piston.py
import time
import numpy as np
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
                 elasticity: float,      # Constante del resorte (k). Alta = más rígido.
                 damping: float,         # Coeficiente de amortiguación (c).
                 piston_mass: float = 1.0, # Masa inercial del pistón (m).
                 mode: PistonMode = PistonMode.CAPACITOR):
        
        self.capacity = capacity  # Límite máximo de compresión (-capacity).
        self.mode = mode

        # Parámetros Físicos
        self.k = elasticity
        self.c = damping
        self.m = piston_mass

        # Estado del Pistón
        self.position = 0.0  # Posición de equilibrio es 0. Negativo = comprimido/cargado.
        self.velocity = 0.0  # Velocidad actual del pistón.
        self.last_applied_force = 0.0

        # Para calcular la velocidad de la señal de entrada
        self.last_signal_info = {} # Almacenará {'source': {'value': float, 'timestamp': float}}

        # Parámetros de Modo
        self.capacitor_discharge_threshold = -capacity * 0.9 # Umbral de posición para descarga automática
        self.battery_is_discharging = False
        self.battery_discharge_rate = capacity * 0.05 # Cuánta "posición" se libera por segundo

    @property
    def current_charge(self) -> float:
        """La carga se define como la compresión del pistón (valor absoluto de la posición negativa)."""
        return max(0, -self.position)

    def apply_force(self, signal_value: float, source: str, mass_factor: float = 1.0):
        """
        Calcula la "fuerza de impacto" de una señal entrante y la aplica al pistón.
        La fuerza es proporcional a la "masa" de la señal y al cuadrado de su "velocidad" (cambio).
        """
        current_time = time.monotonic()
        
        # Calcular la velocidad de la señal (derivada)
        if source in self.last_signal_info:
            last_val = self.last_signal_info[source]['value']
            last_time = self.last_signal_info[source]['timestamp']
            dt = current_time - last_time
            
            if dt > 1e-6: # Evitar división por cero
                signal_velocity = (signal_value - last_val) / dt
            else:
                signal_velocity = 0.0
        else:
            signal_velocity = 0.0

        # Actualizar para el próximo cálculo
        self.last_signal_info[source] = {'value': signal_value, 'timestamp': current_time}

        # Calcular la fuerza de impacto (nuestra "energía cinética")
        # Usamos el valor absoluto de la velocidad, ya que el impacto es energía, sin dirección.
        # El signo de la fuerza siempre es para comprimir el pistón (negativo).
        force = -0.5 * mass_factor * (signal_velocity ** 2)
        self.last_applied_force = force
        
        logger.debug(f"Fuente '{source}': valor={signal_value:.2f}, vel={signal_velocity:.2f}, masa={mass_factor:.2f} -> Fuerza={force:.2f}")

    def update_state(self, dt: float):
        """
        Actualiza el estado (posición, velocidad) del pistón para un paso de tiempo 'dt'.
        Esta función debe ser llamada en un bucle de simulación regular.
        """
        # Calcular las fuerzas internas del sistema
        spring_force = -self.k * self.position  # Ley de Hooke: se opone al desplazamiento
        damping_force = -self.c * self.velocity # Se opone al movimiento

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
        """Libera la energía almacenada según el modo de operación."""
        if self.mode == PistonMode.CAPACITOR:
            # Descarga automática si está suficientemente comprimido
            if self.position <= self.capacitor_discharge_threshold:
                output_signal = {"type": "pulse", "amplitude": self.current_charge}
                logger.info(f"¡Descarga CAPACITOR! Amplitud: {self.current_charge:.2f}")
                # El "rebote": resetea la posición y le da una velocidad de salida
                self.position = 0.0
                self.velocity = 2.0 # Simula el rebote
                return output_signal
        
        elif self.mode == PistonMode.BATTERY:
            if self.battery_is_discharging and self.current_charge > 0:
                # La descarga consume la "compresión" (energía potencial)
                position_released = self.battery_discharge_rate * (1/UPDATE_INTERVAL) # Asumiendo que update_state se llama a UPDATE_INTERVAL
                self.position = min(0, self.position + position_released)

                if self.current_charge == 0:
                    self.battery_is_discharging = False
                    logger.info("Descarga BATTERY completada.")

                return {"type": "sustained", "amplitude": 1.0} # Amplitud de salida constante
        
        return None
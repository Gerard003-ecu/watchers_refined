# En un nuevo archivo: atomic_piston/atomic_piston.py

import time
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class PistonMode(Enum):
    """Define los modos de operación del Pistón Atómico."""
    CAPACITOR = "capacitor"  # Acumula y descarga en un pulso al alcanzar un umbral.
    BATTERY = "battery"      # Acumula y descarga de forma sostenida y controlada.

class AtomicPiston:
    """
    Simula una Unidad de Potencia Inteligente (IPU) que acumula y libera
    "energía" (señales) de forma controlada, actuando como un buffer
    estratégico en el ecosistema Watchers.
    """

    def __init__(self, name: str, capacity: float, mode: PistonMode = PistonMode.CAPACITOR, 
                 trigger_threshold_ratio: float = 0.9, discharge_rate: float = 0.1):
        """
        Inicializa el Pistón Atómico.

        Args:
            name (str): Nombre único para identificar esta IPU.
            capacity (float): Capacidad máxima de energía que puede almacenar.
            mode (PistonMode): Modo de operación inicial (CAPACITOR o BATTERY).
            trigger_threshold_ratio (float): (Modo CAPACITOR) Porcentaje de la capacidad
                                             al que se dispara automáticamente (e.g., 0.9 para 90%).
            discharge_rate (float): (Modo BATTERY) Cantidad de energía liberada por tick
                                    de descarga.
        """
        self.name = name
        self.capacity = float(capacity)
        self.current_charge = 0.0
        
        # --- Parámetros de Control ---
        self.mode = mode
        self.is_discharging_sustained = False  # Específico para el modo BATTERY

        # --- Parámetros de Modo ---
        self.trigger_threshold = self.capacity * trigger_threshold_ratio
        self.discharge_rate = float(discharge_rate)
        
        logger.info(f"IPU '{self.name}' creada. Modo: {self.mode.value}, Capacidad: {self.capacity}")

    # --- MÉTODOS DE ESTADO (El núcleo de la física) ---

    def charge(self, energy_amount: float) -> dict | None:
        """
        Añade energía al pistón. En modo CAPACITOR, puede dispararse automáticamente.

        Args:
            energy_amount (float): La cantidad de energía (señal) a añadir.

        Returns:
            dict | None: Un diccionario de señal si se produce una descarga, o None.
        """
        if energy_amount < 0:
            return None # No se puede cargar con energía negativa

        previous_charge = self.current_charge
        self.current_charge = min(self.capacity, self.current_charge + energy_amount)
        logger.debug(f"IPU '{self.name}' cargada con {energy_amount}. Carga actual: {self.current_charge:.2f}/{self.capacity}")

        # Lógica de disparo automático para el modo CAPACITOR
        if self.mode == PistonMode.CAPACITOR and self.current_charge >= self.trigger_threshold and previous_charge < self.trigger_threshold:
            logger.info(f"IPU '{self.name}' ha alcanzado el umbral en modo CAPACITOR. ¡Descargando!")
            return self.discharge()
        
        return None

    def discharge(self) -> dict | None:
        """
        Libera energía. El comportamiento depende del modo actual.
        - CAPACITOR: Libera toda la carga en un pulso.
        - BATTERY: Libera una cantidad fija (discharge_rate) si está activado.

        Returns:
            dict | None: Un diccionario de señal si se produce una descarga, o None.
        """
        if self.current_charge <= 0:
            return None

        if self.mode == PistonMode.CAPACITOR:
            output_signal = {"type": "pulse", "amplitude": self.current_charge, "source": self.name}
            logger.info(f"IPU '{self.name}' descargando pulso de {self.current_charge:.2f} de energía.")
            self.current_charge = 0.0
            return output_signal

        if self.mode == PistonMode.BATTERY:
            if self.is_discharging_sustained:
                energy_to_release = min(self.current_charge, self.discharge_rate)
                self.current_charge -= energy_to_release
                output_signal = {"type": "sustained", "amplitude": energy_to_release, "source": self.name}
                logger.debug(f"IPU '{self.name}' descargando {energy_to_release:.2f} de forma sostenida. Restante: {self.current_charge:.2f}")
                
                if self.current_charge <= 0:
                    self.is_discharging_sustained = False # Se agotó la batería
                    logger.info(f"IPU '{self.name}' ha agotado su carga en modo BATTERY.")

                return output_signal
        
        return None

    # --- MÉTODOS DE CONTROL (La API para agent_ai) ---

    def set_mode(self, new_mode: PistonMode):
        """Permite a un agente externo cambiar el modo de operación de la IPU."""
        if not isinstance(new_mode, PistonMode):
            logger.warning(f"Intento de establecer un modo inválido en IPU '{self.name}'")
            return
        
        if self.mode != new_mode:
            logger.info(f"IPU '{self.name}' cambiando de modo {self.mode.value} a {new_mode.value}.")
            self.mode = new_mode
            self.is_discharging_sustained = False # Resetear estado al cambiar

    def toggle_sustained_discharge(self, enable: bool):
        """Activa o desactiva la descarga continua para el modo BATTERY."""
        if self.mode == PistonMode.BATTERY:
            if self.is_discharging_sustained != enable:
                self.is_discharging_sustained = enable
                status = "ACTIVADA" if enable else "DESACTIVADA"
                logger.info(f"Descarga sostenida de IPU '{self.name}' {status}.")
        else:
            logger.warning(f"Intento de activar descarga sostenida en IPU '{self.name}', pero no está en modo BATTERY.")

    # --- MÉTODOS DE INFORMACIÓN (La API para monitoreo) ---

    def get_status(self) -> dict:
        """Devuelve el estado actual de la IPU en un formato estructurado."""
        charge_percentage = (self.current_charge / self.capacity) * 100 if self.capacity > 0 else 0
        return {
            "name": self.name,
            "mode": self.mode.value,
            "capacity": self.capacity,
            "current_charge": self.current_charge,
            "charge_percentage": round(charge_percentage, 2),
            "is_discharging": self.is_discharging_sustained if self.mode == PistonMode.BATTERY else (self.current_charge >= self.trigger_threshold),
        }
import os
from .constants import FrictionModel

class PistonConfig:
    """
    Clase para encapsular la configuración del AtomicPistonService.

    Carga los parámetros desde las variables de entorno, proporcionando
    valores por defecto si no se encuentran.
    """
    def __init__(self):
        # --- Configuración del Pistón ---
        self.capacity = float(os.environ.get("PISTON_CAPACITY", 10.0))
        self.elasticity = float(os.environ.get("PISTON_ELASTICITY", 100.0))
        self.damping = float(os.environ.get("PISTON_DAMPING", 5.0))
        self.mass = float(os.environ.get("PISTON_MASS", 1.0))

        friction_model_str = os.environ.get("PISTON_FRICTION_MODEL", "viscous").upper()
        try:
            self.friction_model = FrictionModel[friction_model_str]
        except KeyError:
            print(f"Warning: Modelo de fricción '{friction_model_str}' inválido. Usando VISCOUS por defecto.")
            self.friction_model = FrictionModel.VISCOUS

        # --- Configuración del Servicio ---
        self.service_port = int(os.environ.get("ATOMIC_PISTON_PORT", 5002))
        self.simulation_interval = float(os.environ.get("ATOMIC_PISTON_SIM_INTERVAL", 0.02)) # 50Hz
        self.agent_ai_register_url = os.environ.get("AGENT_AI_REGISTER_URL", "http://agent_ai:9000/api/register")
        self.requests_timeout = float(os.environ.get("ATOMIC_PISTON_REQUESTS_TIMEOUT", 3.0))
        self.max_registration_retries = int(os.environ.get("MAX_REGISTRATION_RETRIES", 5))
        self.retry_delay_seconds = int(os.environ.get("RETRY_DELAY_SECONDS", 5))

# =============== CONFIGURACIÓN DE DISEÑO (Existente) ===============
BOARD_WIDTH = 150  # mm
BOARD_HEIGHT = 100  # mm
LAYER_CONFIG = {
    'power_track_width': 3.0,  # mm
    'signal_track_width': 0.3,  # mm
    'thermal': {
        'via_drill': 0.3,      # mm
        'via_diameter': 0.6,    # mm
        'via_count': 8,
        'clearance': 0.5        # mm
    },
    'zones': {
        'clearance': 0.5,      # mm
        'min_width': 0.25       # mm
    }
}

# Mapeo de componentes a footprints (actualizado para KiCad 7)
FOOTPRINTS = {
    'ESP32': 'Module:ESP32-WROOM-32_NoPins',
    'MOSFET_POWER': 'Package_TO_SOT_THT:TO-247-3_Vertical',
    'GATE_DRIVER': 'Package_SO:SOIC-8_3.9x4.9mm_P1.27mm',
    'SUPERCAP': 'Capacitor_THT:CP_Radial_D25.0mm_P10.00mm',
    'INDUCTOR_POWER': 'Inductor_THT:L_Toroid_D33.0mm_P17.30mm_Vertical',
    'CONN_POWER': 'Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical',
    'CONN_PV': 'Connector_Wire:SolderWire-1.5mm_1x02_P7.62mm_Drill1.5mm',
    'RESISTOR_SHUNT':
        'Resistor_THT:R_Axial_DIN0617_L17.0mm_D6.0mm_P22.86mm_Horizontal',
    'BMS_IC': 'Package_SO:SOIC-16_3.9x9.9mm_P1.27mm',
    'PRE_CHARGE_RES':
        'Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P2.54mm_Horizontal',
    'STAR_POINT': 'TestPoint:TestPoint_Pad_D1.5mm'
}

# Definir posiciones y rotaciones
PLACEMENTS = {
    'ESP32': {'pos': (30, 50), 'rot': 0, 'side': 'top'},
    'UCC21520': {'pos': (80, 25), 'rot': 90, 'side': 'top'},  # Closer to MOSFET
    'Q1': {'pos': (100, 25), 'rot': 180, 'side': 'top'},
    'C1': {'pos': (125, 50), 'rot': 0, 'side': 'top'},
    'L1': {'pos': (100, 75), 'rot': 0, 'side': 'top'},
    'J1': {'pos': (140, 85), 'rot': 270, 'side': 'top'},
    'R1': {'pos': (115, 40), 'rot': 0, 'side': 'top'},
    'TP1': {'pos': (10, 10), 'rot': 0, 'side': 'top'}  # Punto estrella
}

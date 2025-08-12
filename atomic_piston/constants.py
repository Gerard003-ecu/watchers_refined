from enum import Enum

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

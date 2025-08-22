import os
import logging
from skidl import Part, Net, generate_netlist, reset

# Importar solo lo necesario de la configuración
from config import FOOTPRINTS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_schematic_netlist(filename="atomic_piston.net"):
    """Define las conexiones lógicas del circuito con separación de tierras."""
    # Esta función es la misma que ya tenías
    reset()
    logger.info("Creando esquemático con SKiDL...")
    try:
        # (Pega aquí todo el contenido de tu función create_schematic_netlist)
        # ...
        esp32 = Part('Module', 'ESP32-WROOM-32', footprint=FOOTPRINTS['ESP32'], dest='TEMPLATE')
        # ... (resto de la definición de componentes y redes) ...
        # ...
        logger.info("Generando netlist del esquemático...")
        generate_netlist(file_=filename)
        logger.info(f"Netlist guardada como '{filename}'")
        return os.path.abspath(filename)
    except Exception as e:
        logger.error(f"Error en creación de esquemático: {str(e)}")
        raise

if __name__ == "__main__":
    # Este script ahora solo hace una cosa: generar la netlist.
    output_dir = os.path.dirname(os.path.abspath(__file__))
    netlist_file = os.path.join(output_dir, "atomic_piston.net")
    create_schematic_netlist(filename=netlist_file)
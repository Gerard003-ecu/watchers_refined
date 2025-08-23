import os
import sys

# Añadir la carpeta del proyecto al path para que las importaciones funcionen
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from atomic_piston.generate_schematic import create_schematic_netlist


def main():
    """Orquesta el flujo completo de generación de la PCB."""
    print("--- Iniciando flujo de generación de PCB ---")

    # Definir rutas
    piston_dir = os.path.join(project_root, "atomic_piston")
    netlist_file = os.path.join(piston_dir, "atomic_piston.net")
    pcb_file = os.path.join(piston_dir, "atomic_piston_layout.kicad_pcb")

    # --- PASO 1: Generar la netlist con SKiDL ---
    try:
        print(f"1. Generando netlist en '{netlist_file}'...")
        create_schematic_netlist(filename=netlist_file)
        print("   Netlist generada con éxito.")
    except Exception as e:
        print(f"   ERROR: Falló la generación de la netlist: {e}")
        return

    # --- PASO 2: Crear una PCB vacía y cargar la netlist con kicad-cli ---
    # Este es un enfoque conceptual. KiCad 9 prefiere trabajar con proyectos.
    # Un flujo más robusto sería tener un proyecto .kicad_pro
    # Por ahora, vamos a asumir que necesitamos un .kicad_pcb vacío.
    # (Este paso puede necesitar ajustes dependiendo de la API exacta de kicad-cli)

    # --- PASO 3 (Conceptual): Llamar a un script de KiCad para el layout ---
    # Este es el paso que aún no podemos hacer directamente.
    # El flujo correcto es: generar esquemático, luego usar kicad-cli para
    # actualizar la PCB desde el esquemático.

    print("\n--- Flujo de Trabajo Sugerido ---")
    print("1. El archivo 'atomic_piston.net' ha sido creado.")
    print("2. Ahora, abre KiCad, crea un proyecto nuevo.")
    print(
        "3. En el editor de esquemáticos, ve a Herramientas -> Importar -> Netlist y selecciona el archivo."
    )
    print(
        "4. Luego, en el editor de esquemáticos, ve a Herramientas -> Actualizar PCB desde el Esquema."
    )
    print(
        "5. Finalmente, puedes ejecutar el script de refinamiento en la consola de KiCad."
    )


if __name__ == "__main__":
    main()

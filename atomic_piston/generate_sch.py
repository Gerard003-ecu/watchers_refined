# generate_sch.py
from pcb_atomic_piston_v2 import create_schematic_netlist

# Modifica la función para que genere un .kicad_sch
# (Esto es conceptual, Jules tendrá que implementar la función correcta de skidl)
print("Generando esquemático...")
create_schematic_netlist("atomic_piston/atomic_piston_project.kicad_sch")
print("Esquemático generado.")

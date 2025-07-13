#!/usr/bin/env python3
"""
Mini-test-suite para pcb_atomic_piston_v2.py
1. Verifica import / sintaxis
2. Comprueba creación de netlist
3. Genera Gerbers y comprueba que no estén vacíos
"""
import os
import sys
import zipfile
import tempfile

# Añadir la carpeta actual al path para importar
sys.path.insert(0, os.getcwd())

try:
    from pcb_atomic_piston_v2 import create_schematic_netlist, create_pcb_design
    print("✅ Importación: OK")
except Exception as e:
    print("❌ Importación / sintaxis:", e)
    sys.exit(1)

# ------------------------------------------------------------------
# 1. Crear netlist temporal
# ------------------------------------------------------------------
try:
    with tempfile.TemporaryDirectory() as tmp:
        net_path = os.path.join(tmp, "test_atomic_piston.net")
        create_schematic_netlist(net_path)
        assert os.path.isfile(net_path) and os.path.getsize(net_path) > 0, \
            "netlist vacío"
        print("✅ Netlist creado:", net_path)
# ------------------------------------------------------------------
# 2. Generar Gerbers
# ------------------------------------------------------------------
        success = create_pcb_design(net_path)
        assert success, "create_pcb_design devolvió False"
        ger_dir = "gerbers"
        assert os.path.isdir(ger_dir), "carpeta gerbers no encontrada"
        files = [f for f in os.listdir(ger_dir) if f.endswith(('.gbr', '.drl'))]
        assert files, "no se generaron archivos Gerber/Drill"
        for f in files:
            size = os.path.getsize(os.path.join(ger_dir, f))
            assert size > 100, f"archivo {f} parece vacío ({size} B)"
        print("✅ Gerbers generados:", len(files), "archivos")
# ------------------------------------------------------------------
# 3. Crear ZIP portátil
# ------------------------------------------------------------------
        zip_name = "atomic_piston_gerbers_test.zip"
        with zipfile.ZipFile(zip_name, 'w') as z:
            for f in files:
                z.write(os.path.join(ger_dir, f), f)
        print("✅ ZIP creado:", zip_name)
except Exception as e:
    print("❌ Error en test:", e)
    sys.exit(1)

print("\n🎉 ¡Todos los tests pasaron! Puedes usar:")
print("   - atomic_piston_gerbers_test.zip → enviar a fábrica")
print("   - gerbers/ → revisar con Gerbv o KiCad")

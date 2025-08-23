# watchers_refined/atomic_piston/pcb_atomic_piston.py

import logging
import math
import os
import subprocess
import sys

from skidl import Net, Part, generate_schematic, reset

try:
    import pcbnew
    from pcbnew import (
        EDGE_CUTS,
        EXCELLON_WRITER,
        PCB_SHAPE,
        PCB_TRACK,
        PCB_VIA,
        PLOT_CONTROLLER,
        PLOT_FORMAT_GERBER,
        ZONE,
        B_Cu,
        B_Mask,
        B_Paste,
        B_SilkS,
        F_Cu,
        F_Mask,
        F_Paste,
        F_SilkS,
        FromMM,
        GetBuildVersion,
        In1_Cu,
        In2_Cu,
        wxPoint,
        wxPointMM,
    )
except ImportError:
    print("Error: No se pudo importar la librería 'pcbnew' de KiCad.")
    print(
        "Asegúrate de ejecutar este script con el intérprete de Python que "
        "viene con KiCad."
    )
    print(
        "Ejemplo en Windows: "
        '"C:\\Program Files\\KiCad\\9.0\\bin\\python.exe" '
        "pcb_atomic_piston_v2.py"
    )
    sys.exit(1)

# Import configuration
try:
    from .config import BOARD_HEIGHT, BOARD_WIDTH, FOOTPRINTS, LAYER_CONFIG, PLACEMENTS
except ImportError:
    print("Error: No se pudo importar el archivo de configuración 'config.py'.")
    raise


# Configuración de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============== CREACIÓN ESQUEMÁTICA ===============
def create_schematic_file(filename="atomic_piston.kicad_sch"):
    """Define las conexiones lógicas y genera el archivo de esquemático."""
    reset()
    logger.info("Creando esquemático...")

    try:
        # Definición de componentes
        esp32 = Part(
            "Module", "ESP32-WROOM-32", footprint=FOOTPRINTS["ESP32"], dest="TEMPLATE"
        )
        gate_driver = Part(
            "Driver_Gate",
            "UCC21520",
            footprint=FOOTPRINTS["GATE_DRIVER"],
            dest="TEMPLATE",
        )
        q1_mosfet = Part(
            "Device",
            "Q_NMOS_GDS",
            footprint=FOOTPRINTS["MOSFET_POWER"],
            dest="TEMPLATE",
        )
        supercap = Part(
            "Device",
            "C",
            value="3000F",
            footprint=FOOTPRINTS["SUPERCAP"],
            dest="TEMPLATE",
        )
        inductor = Part(
            "Device",
            "L",
            value="100uH",
            footprint=FOOTPRINTS["INDUCTOR_POWER"],
            dest="TEMPLATE",
        )
        pv_connector = Part(
            "Connector", "Conn_01x02", footprint=FOOTPRINTS["CONN_PV"], dest="TEMPLATE"
        )
        precharge_res = Part(
            "Device",
            "R",
            value="100",
            footprint=FOOTPRINTS["PRE_CHARGE_RES"],
            dest="TEMPLATE",
        )
        star_point = Part(
            "Device",
            "TestPoint",
            value="STAR_POINT",
            footprint=FOOTPRINTS["STAR_POINT"],
            dest="TEMPLATE",
        )

        # Definición de redes (Nets) principales
        gnd_digital = Net("GND_DIGITAL")
        gnd_power = Net("GND_POWER")
        vcc_3v3 = Net("+3V3")
        pv_plus = Net("PV+")
        pv_minus = Net("PV-")
        sw_node = Net("SWITCH_NODE")
        gpio_ctrl = Net("GPIO22_CTRL")

        # Conexiones de la etapa de control
        gnd_digital += esp32["GND"], gate_driver["GND"]
        vcc_3v3 += esp32["3V3"], gate_driver["VCC"]
        esp32["GPIO22"] += gpio_ctrl
        gpio_ctrl += gate_driver["INA"]

        # Conexiones de la etapa de potencia
        pv_plus += pv_connector[1]
        pv_plus += precharge_res[1]
        precharge_res[2] += supercap[1]
        pv_plus += inductor[1]
        inductor[2] += sw_node
        sw_node += q1_mosfet["D"]
        q1_mosfet["S"] += gnd_power
        gate_driver["OUTA"] += q1_mosfet["G"]
        sw_node += supercap[1]
        gnd_power += supercap[2]
        gnd_digital += star_point[1]
        gnd_power += star_point[1]
        pv_minus += pv_connector[2], gnd_power

        # Generar el archivo de esquemático de KiCad
        logger.info("Generando archivo de esquemático de KiCad...")
        base_name, _ = os.path.splitext(filename)
        generate_schematic(top_name=base_name)
        actual_filename = base_name + ".sch"
        logger.info(f"Esquemático guardado como '{actual_filename}'")
        return os.path.abspath(actual_filename)
    except Exception as e:
        logger.error(f"Error en creación de esquemático: {str(e)}")
        raise


def create_project_files(project_name="atomic_piston"):
    """Crea los archivos de proyecto de KiCad necesarios."""
    logger.info("Creando archivos de proyecto de KiCad...")
    pro_filename = f"{project_name}.kicad_pro"
    pcb_filename = f"{project_name}.kicad_pcb"

    if not os.path.exists(pro_filename):
        pro_content = f"""{{
  "meta": {{ "version": "2" }},
  "sheets": [ {{ "fileName": "{os.path.basename(project_name)}.kicad_sch" }} ]
}}"""
        with open(pro_filename, "w") as f:
            f.write(pro_content)
        logger.info(f"Archivo de proyecto '{pro_filename}' creado.")
    else:
        logger.info(f"Archivo de proyecto '{pro_filename}' ya existe.")

    if not os.path.exists(pcb_filename):
        # Create a minimal valid PCB file for KiCad 9
        with open(pcb_filename, "w") as f:
            f.write("(kicad_pcb (version 9) (generator pcbnew))\n")
        logger.info(f"Archivo PCB '{pcb_filename}' creado.")
    else:
        logger.info(f"Archivo PCB '{pcb_filename}' ya existe.")

    return os.path.abspath(pro_filename), os.path.abspath(pcb_filename)


# =============== DISEÑO DE PCB ===============
def create_pcb_design(schematic_file, pcb_file):
    """Crea el diseño completo de PCB utilizando kicad-cli para la sincronización."""
    try:
        logger.info(f"Iniciando diseño de PCB con KiCad v{GetBuildVersion()}")

        # Sincronizar esquemático con PCB usando kicad-cli
        logger.info("Sincronizando esquemático con PCB via kicad-cli...")
        cli_command = [
            "kicad-cli",
            "sch",
            "pcb",
            "update",
            "--schematic",
            schematic_file,
            pcb_file,
        ]

        result = subprocess.run(
            cli_command, capture_output=True, text=True, check=False
        )

        logger.info(f"kicad-cli stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"kicad-cli stderr:\n{result.stderr}")

        result.check_returncode()

        # Cargar la PCB actualizada
        logger.info(f"Cargando PCB actualizada desde '{pcb_file}'...")
        board = pcbnew.LoadBoard(pcb_file)

        create_board_outline(board)
        place_components(board)
        route_critical_nets(board)
        create_power_planes(board)
        add_thermal_management(board)
        add_silkscreen_labels(board)

        if not verify_design(board):
            logger.error("Errores encontrados. Abortando generación de Gerbers.")
            return False

        generate_gerber_files(board)
        generate_drill_files(board)

        board.Save(pcb_file)
        logger.info(f"¡Diseño de PCB completado! Guardado en '{pcb_file}'")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error ejecutando kicad-cli: {e}\n{e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error en diseño de PCB: {str(e)}")
        return False


# --- FUNCIONES AUXILIARES DE PCB ---
def create_board_outline(board):
    """Crea el contorno de la placa en la capa Edge.Cuts."""
    logger.info("Creando contorno de la placa...")
    outline_points = [
        (0, 0),
        (BOARD_WIDTH, 0),
        (BOARD_WIDTH, BOARD_HEIGHT),
        (0, BOARD_HEIGHT),
        (0, 0),
    ]
    for i in range(len(outline_points) - 1):
        start = wxPointMM(*outline_points[i])
        end = wxPointMM(*outline_points[i + 1])
        segment = PCB_SHAPE(board)
        segment.SetShape(pcbnew.SHAPE_T_SEGMENT)
        segment.SetStart(start)
        segment.SetEnd(end)
        segment.SetLayer(EDGE_CUTS)
        segment.SetWidth(FromMM(0.15))
        board.Add(segment)


def place_components(board):
    """Coloca componentes con orientaciones adecuadas."""
    logger.info("Colocando componentes en la placa...")
    for module in board.GetModules():
        ref = module.GetReference()
        placement = PLACEMENTS.get(ref) or PLACEMENTS.get(module.GetValue())
        if placement:
            pos = wxPointMM(*placement["pos"])
            module.SetPosition(pos)
            module.SetOrientationDegrees(placement["rot"])
            if placement["side"] == "bottom":
                module.Flip(pos)


def route_critical_nets(board):
    """Enruta manualmente las redes críticas con una estrategia de cadena."""
    logger.info("Enrutando pistas críticas con estrategia daisy-chain...")
    net_configs = {
        "PV+": FromMM(LAYER_CONFIG["power_track_width"]),
        "SWITCH_NODE": FromMM(LAYER_CONFIG["power_track_width"]),
        "GND_POWER": FromMM(LAYER_CONFIG["power_track_width"]),
        "GPIO22_CTRL": FromMM(LAYER_CONFIG["signal_track_width"]),
        "+3V3": FromMM(LAYER_CONFIG["signal_track_width"]),
        "GND_DIGITAL": FromMM(LAYER_CONFIG["signal_track_width"]),
    }
    for net_name, width in net_configs.items():
        net = find_net(board, net_name)
        if not net:
            continue
        pads = list(net.Pads())
        if len(pads) < 2:
            logger.warning(f"Red '{net_name}' tiene < 2 pads, no se puede enrutar.")
            continue
        for i in range(len(pads) - 1):
            create_track(
                board, pads[i].GetCenter(), pads[i + 1].GetCenter(), net, width, F_Cu
            )
    logger.info("Enrutado de pistas críticas completado.")


def create_power_planes(board):
    """Crea planos de tierra y potencia en capas internas."""
    logger.info("Creando planos de potencia...")
    gnd_digital_net = find_net(board, "GND_DIGITAL")
    if gnd_digital_net:
        create_zone(
            board,
            In1_Cu,
            gnd_digital_net,
            [
                (2, 2),
                (BOARD_WIDTH / 2 - 5, 2),
                (BOARD_WIDTH / 2 - 5, BOARD_HEIGHT - 2),
                (2, BOARD_HEIGHT - 2),
            ],
        )
    gnd_power_net = find_net(board, "GND_POWER")
    if gnd_power_net:
        create_zone(
            board,
            In2_Cu,
            gnd_power_net,
            [
                (BOARD_WIDTH / 2 + 5, 2),
                (BOARD_WIDTH - 2, 2),
                (BOARD_WIDTH - 2, BOARD_HEIGHT - 2),
                (BOARD_WIDTH / 2 + 5, BOARD_HEIGHT - 2),
            ],
        )


def add_thermal_management(board):
    """Añade vías térmicas y zonas de disipación."""
    logger.info("Añadiendo gestión térmica...")
    mosfet_refs = ["Q1"]
    for ref in mosfet_refs:
        mosfet = find_component(board, ref)
        if mosfet:
            position = mosfet.GetPosition()
            for i in range(LAYER_CONFIG["thermal"]["via_count"]):
                angle = 2 * math.pi * i / LAYER_CONFIG["thermal"]["via_count"]
                radius = FromMM(3.5)
                x_offset = int(radius * math.cos(angle))
                y_offset = int(radius * math.sin(angle))
                create_via(
                    board,
                    position + wxPoint(x_offset, y_offset),
                    find_net(board, "GND_POWER"),
                    FromMM(LAYER_CONFIG["thermal"]["via_drill"]),
                    FromMM(LAYER_CONFIG["thermal"]["via_diameter"]),
                )
    gnd_power_net = find_net(board, "GND_POWER")
    if gnd_power_net:
        create_zone(
            board,
            B_Cu,
            gnd_power_net,
            [(90, 15), (120, 15), (120, 35), (90, 35)],
            thermal_gap=FromMM(LAYER_CONFIG["thermal"]["clearance"]),
        )


def add_silkscreen_labels(board):
    """Añade etiquetas de identificación en la capa de silkscreen."""
    logger.info("Añadiendo etiquetas de silkscreen...")
    text = pcbnew.PCB_TEXT(board)
    text.SetText("Atomic Piston IPU v1.0")
    text.SetPosition(wxPointMM(BOARD_WIDTH / 2, BOARD_HEIGHT - 5))
    text.SetLayer(F_SilkS)
    text.SetTextSize(pcbnew.VECTOR2I(FromMM(2), FromMM(2)))
    board.Add(text)
    pv_text = pcbnew.PCB_TEXT(board)
    pv_text.SetText("PV INPUT")
    pv_text.SetPosition(wxPointMM(140, 75))
    pv_text.SetLayer(F_SilkS)
    pv_text.SetTextSize(pcbnew.VECTOR2I(FromMM(1.5), FromMM(1.5)))
    board.Add(pv_text)


# =============== HERRAMIENTAS DE PCB ===============
def find_net(board, net_name):
    """Encuentra una red por nombre."""
    net = board.FindNet(net_name)
    if not net:
        logger.warning(f"Red '{net_name}' no encontrada")
    return net


def find_component(board, reference):
    """Encuentra un componente por referencia."""
    for module in board.GetModules():
        if module.GetReference() == reference:
            return module
    logger.warning(f"Componente '{reference}' no encontrado")
    return None


def create_track(board, start, end, net, width, layer):
    """Crea una pista entre dos puntos."""
    track = PCB_TRACK(board)
    track.SetStart(start)
    track.SetEnd(end)
    track.SetNet(net)
    track.SetWidth(width)
    track.SetLayer(layer)
    board.Add(track)
    return track


def create_via(board, position, net, drill, diameter):
    """Crea una vía térmica."""
    via = PCB_VIA(board)
    via.SetPosition(position)
    via.SetDrill(drill)
    via.SetWidth(diameter)
    via.SetNet(net)
    via.SetLayerPair(F_Cu, B_Cu)
    board.Add(via)
    return via


def create_zone(board, layer, net, corners, thermal_gap=None):
    """Crea una zona de relleno."""
    zone = ZONE(board)
    zone.SetLayer(layer)
    zone.SetNet(net)
    zone.SetIsRuleArea(False)
    zone.SetDoNotAllowCopperPour(False)
    zone.SetDoNotAllowTracks(False)
    zone.SetDoNotAllowVias(False)
    zone.SetZoneClearance(FromMM(LAYER_CONFIG["zones"]["clearance"]))
    zone.SetMinThickness(FromMM(LAYER_CONFIG["zones"]["min_width"]))
    if thermal_gap:
        zone.SetThermalReliefGap(thermal_gap)
        zone.SetThermalReliefCopperBridge(FromMM(0.3))
    polygon = [wxPointMM(*p) for p in corners]
    zone.AddPolygon(polygon)
    board.Add(zone)
    return zone


# =============== VERIFICACIÓN Y EXPORTACIÓN ===============
def verify_design(board):
    """Realiza verificaciones básicas de diseño."""
    logger.info("Verificando diseño...")
    errors = []
    critical_nets = ["PV+", "GND_POWER", "SWITCH_NODE"]
    for net_name in critical_nets:
        net = find_net(board, net_name)
        if net and net.GetPadCount() == 0:
            errors.append(f"Red crítica '{net_name}' no tiene conexiones")
    required_components = ["Q1", "C1", "L1", "J1"]
    for ref in required_components:
        if not find_component(board, ref):
            errors.append(f"Componente crítico '{ref}' no encontrado")
    if errors:
        for error in errors:
            logger.error(error)
        return False
    return True


def generate_gerber_files(board):
    """Genera archivos Gerber para todas las capas."""
    logger.info("Generando archivos Gerber...")
    plot_controller = PLOT_CONTROLLER(board)
    plot_options = plot_controller.GetPlotOptions()
    plot_options.SetOutputDirectory("gerbers")
    plot_options.SetPlotFrameRef(False)
    plot_options.SetDrillMarksType(pcbnew.PCB_PLOT_PARAMS.NO_DRILL_SHAPE)
    plot_options.SetUseGerberProtelExtensions(True)
    plot_options.SetSubtractMaskFromSilk(True)
    plot_options.SetGerberPrecision(6)
    layers = [
        (F_Cu, "F.Cu", "Top Copper"),
        (In1_Cu, "In1.Cu", "Inner 1"),
        (In2_Cu, "In2.Cu", "Inner 2"),
        (B_Cu, "B.Cu", "Bottom Copper"),
        (F_Paste, "F.Paste", "Paste"),
        (B_Paste, "B.Paste", "Paste"),
        (F_SilkS, "F.SilkS", "SilkS"),
        (B_SilkS, "B.SilkS", "SilkS"),
        (F_Mask, "F.Mask", "Mask"),
        (B_Mask, "B.Mask", "Mask"),
        (EDGE_CUTS, "Edge.Cuts", "Outline"),
    ]
    for layer_id, layer_name, desc in layers:
        plot_controller.SetLayer(layer_id)
        plot_controller.OpenPlotfile(layer_name, PLOT_FORMAT_GERBER, desc)
        plot_controller.PlotLayer()
    plot_controller.ClosePlot()


def generate_drill_files(board):
    """Genera archivos de taladro Excellon."""
    logger.info("Generando archivos de taladro...")
    drill_writer = EXCELLON_WRITER(board)
    drill_writer.SetMapFileFormat(pcbnew.PLOT_FORMAT_PDF)
    drill_writer.SetFormat(True)
    drill_writer.CreateDrillandMapFilesSet("gerbers", True, False)


# =============== EJECUCIÓN PRINCIPAL ===============
if __name__ == "__main__":
    # Crear directorios de salida
    os.makedirs("gerbers", exist_ok=True)

    # Información de inicio
    logger.info("Iniciando flujo de diseño IPU Atomic Piston para KiCad 9")
    logger.info(f"Tamaño de placa: {BOARD_WIDTH}x{BOARD_HEIGHT} mm")
    project_name = "atomic_piston"

    # Ejecutar el flujo completo
    try:
        schematic_path = create_schematic_file(f"{project_name}.kicad_sch")
        _, pcb_path = create_project_files(project_name)
        success = create_pcb_design(schematic_path, pcb_path)

        if success:
            logger.info("\nInstrucciones para fabricación:")
            logger.info("1. Comprimir carpeta 'gerbers' y enviar a fabricante.")
            logger.info("2. Verificar diseño en KiCad antes de producción.")
        else:
            logger.error("El diseño de PCB contiene errores. Revise los logs.")
    except Exception as e:
        logger.exception(f"Error en el flujo de diseño: {str(e)}")
        sys.exit(1)

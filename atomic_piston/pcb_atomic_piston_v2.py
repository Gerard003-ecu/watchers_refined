# watchers_refined/atomic_piston/pcb_atomic_piston.py

import os
import sys
import math
import logging
from itertools import combinations
from skidl import Part, Net, generate_netlist, reset

try:
    import pcbnew
    from pcbnew import (BOARD, PCB_SHAPE, PCB_TRACK, PCB_VIA, ZONE,
                        PLOT_CONTROLLER, EXCELLON_WRITER,
                        wxPoint, wxPointMM, FromMM, GetBuildVersion,
                        EDGE_CUTS, F_Cu, B_Cu, In1_Cu, In2_Cu, F_Paste,
                        B_Paste, F_SilkS, B_SilkS, F_Mask, B_Mask,
                        PLOT_FORMAT_GERBER)
except ImportError:
    print("Error: No se pudo importar la librería 'pcbnew' de KiCad.")
    print("Asegúrate de ejecutar este script con el intérprete de Python que "
          "viene con KiCad.")
    print("Ejemplo en Windows: "
          "\"C:\\Program Files\\KiCad\\7.0\\bin\\python.exe\" "
          "pcb_atomic_piston.py")
    sys.exit(1)

# Import configuration
try:
    from .config import (BOARD_WIDTH, BOARD_HEIGHT, LAYER_CONFIG,
                         FOOTPRINTS, PLACEMENTS)
except ImportError:
    print("Error: No se pudo importar el archivo de configuración 'config.py'.")
    sys.exit(1)


# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============== CREACIÓN ESQUEMÁTICA ===============
def create_schematic_netlist(filename="atomic_piston.net"):
    """Define las conexiones lógicas del circuito con separación de tierras."""
    reset()
    logger.info("Creando esquemático...")

    try:
        # Definición de componentes
        esp32 = Part(
            'Module', 'ESP32-WROOM-32',
            footprint=FOOTPRINTS['ESP32'], dest='TEMPLATE'
        )
        gate_driver = Part(
            'Driver_Gate', 'UCC21520',
            footprint=FOOTPRINTS['GATE_DRIVER'], dest='TEMPLATE'
        )
        q1_mosfet = Part(
            'Device', 'Q_NMOS_GDS',
            footprint=FOOTPRINTS['MOSFET_POWER'], dest='TEMPLATE'
        )
        supercap = Part(
            'Device', 'C', value='3000F',
            footprint=FOOTPRINTS['SUPERCAP'], dest='TEMPLATE'
        )
        inductor = Part(
            'Device', 'L', value='100uH',
            footprint=FOOTPRINTS['INDUCTOR_POWER'], dest='TEMPLATE'
        )
        pv_connector = Part(
            'Connector', 'Conn_01x02',
            footprint=FOOTPRINTS['CONN_PV'], dest='TEMPLATE'
        )
        precharge_res = Part(
            'Device', 'R', value='100',
            footprint=FOOTPRINTS['PRE_CHARGE_RES'], dest='TEMPLATE'
        )
        star_point = Part(
            'Device', 'TestPoint', value='STAR_POINT',
            footprint=FOOTPRINTS['STAR_POINT'], dest='TEMPLATE'
        )

        # Definición de redes (Nets) principales
        gnd_digital = Net('GND_DIGITAL')
        gnd_power = Net('GND_POWER')
        vcc_3v3 = Net('+3V3')
        pv_plus = Net('PV+')
        pv_minus = Net('PV-')
        sw_node = Net('SWITCH_NODE')
        gpio_ctrl = Net('GPIO22_CTRL')

        # Conexiones de la etapa de control
        gnd_digital += esp32['GND'], gate_driver['GND']
        vcc_3v3 += esp32['3V3'], gate_driver['VCC']
        esp32['GPIO22'] += gpio_ctrl
        gpio_ctrl += gate_driver['INA']

        # Conexiones de la etapa de potencia
        pv_plus += pv_connector[1]
        pv_plus += precharge_res[1]  # Resistencia de precarga
        precharge_res[2] += supercap[1]  # Precarga antes del supercap

        pv_plus += inductor[1]
        inductor[2] += sw_node
        sw_node += q1_mosfet['D']  # Drain del MOSFET al inductor
        q1_mosfet['S'] += gnd_power  # Source del MOSFET a tierra de potencia
        gate_driver['OUTA'] += q1_mosfet['G']  # Salida del driver al Gate

        # Conexión del supercapacitor
        sw_node += supercap[1]
        gnd_power += supercap[2]

        # Conexión a tierra común (punto estrella)
        gnd_digital += star_point[1]
        gnd_power += star_point[1]

        # Conexión de entrada solar
        pv_minus += pv_connector[2], gnd_power

        # Generar el archivo de netlist
        logger.info("Generando netlist del esquemático...")
        generate_netlist(file_=filename)
        logger.info(f"Netlist guardada como '{filename}'")
        return os.path.abspath(filename)
    except Exception as e:
        logger.error(f"Error en creación de esquemático: {str(e)}")
        sys.exit(1)


# =============== DISEÑO DE PCB ===============
def create_pcb_design(netlist_file):
    """Crea el diseño completo de PCB con todas las características."""
    try:
        logger.info(f"Iniciando diseño de PCB con KiCad v{GetBuildVersion()}")

        # Crear placa y cargar netlist
        board = BOARD()
        logger.info("Cargando netlist en la PCB...")
        board.LoadNetlist(netlist_file)

        # Configuración básica de la placa
        board.SetCopperLayerCount(4)
        board.SetDesignSettings(pcbnew.BOARD_DESIGN_SETTINGS())

        # Crear capas
        create_board_outline(board)
        # place_components(board)
        # route_critical_nets(board)
        # create_power_planes(board)
        # add_thermal_management(board)
        # add_silkscreen_labels(board)

        # Verificar diseño
        if not verify_design(board):
            logger.error(
                "Errores encontrados en el diseño. "
                "Abortando generación de Gerbers."
            )
            return False

        # Generar archivos de fabricación
        generate_gerber_files(board)
        generate_drill_files(board)

        # Guardar archivo de PCB
        board.Save("atomic_piston_layout.kicad_pcb")
        logger.info("¡Diseño de PCB completado con éxito!")
        return True
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
        (0, 0)
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
        comp_type = module.GetValue()

        # Buscar colocación por referencia o tipo
        placement = PLACEMENTS.get(ref) or PLACEMENTS.get(comp_type)
        if placement:
            pos = wxPointMM(*placement['pos'])
            module.SetPosition(pos)
            module.SetOrientationDegrees(placement['rot'])

            if placement['side'] == 'bottom':
                module.Flip(pos)


def route_critical_nets(board):
    """Enruta manualmente las redes críticas de potencia."""
    logger.info("Enrutando pistas críticas...")

    # Configurar anchos de pista
    power_width = FromMM(LAYER_CONFIG['power_track_width'])
    signal_width = FromMM(LAYER_CONFIG['signal_track_width'])

    # Enrutar redes de potencia
    for net_name in ["PV+", "SWITCH_NODE", "GND_POWER"]:
        net = find_net(board, net_name)
        if not net:
            continue

        # Conectar todos los pads de esta red
        pads = list(net.Pads())
        if len(pads) < 2:
            continue

        for pad1, pad2 in combinations(pads, 2):
            create_track(
                board,
                pad1.GetCenter(),
                pad2.GetCenter(),
                net,
                power_width,
                F_Cu
            )

    # Enrutar señales de control
    for net_name in ["GPIO22_CTRL", "+3V3", "GND_DIGITAL"]:
        net = find_net(board, net_name)
        if net:
            pads = list(net.Pads())
            if len(pads) < 2:
                continue

            for pad1, pad2 in combinations(pads, 2):
                create_track(
                    board,
                    pad1.GetCenter(),
                    pad2.GetCenter(),
                    net,
                    signal_width,
                    F_Cu
                )


def create_power_planes(board):
    """Crea planos de tierra y potencia en capas internas."""
    logger.info("Creando planos de potencia...")

    # Plano de tierra digital (capa interna 1)
    gnd_digital_net = find_net(board, "GND_DIGITAL")
    if gnd_digital_net:
        create_zone(
            board,
            layer=In1_Cu,
            net=gnd_digital_net,
            corners=[
                (2, 2),
                (BOARD_WIDTH / 2 - 5, 2),
                (BOARD_WIDTH / 2 - 5, BOARD_HEIGHT - 2),
                (2, BOARD_HEIGHT - 2)
            ]
        )

    # Plano de tierra de potencia (capa interna 2)
    gnd_power_net = find_net(board, "GND_POWER")
    if gnd_power_net:
        create_zone(
            board,
            layer=In2_Cu,
            net=gnd_power_net,
            corners=[
                (BOARD_WIDTH / 2 + 5, 2),
                (BOARD_WIDTH - 2, 2),
                (BOARD_WIDTH - 2, BOARD_HEIGHT - 2),
                (BOARD_WIDTH / 2 + 5, BOARD_HEIGHT - 2)
            ]
        )


def add_thermal_management(board):
    """Añade vías térmicas y zonas de disipación."""
    logger.info("Añadiendo gestión térmica...")

    # Vías térmicas bajo MOSFETs
    mosfet_refs = ["Q1"]
    for ref in mosfet_refs:
        mosfet = find_component(board, ref)
        if mosfet:
            position = mosfet.GetPosition()
            for i in range(LAYER_CONFIG['thermal']['via_count']):
                angle = 2 * math.pi * i / LAYER_CONFIG['thermal']['via_count']
                radius = FromMM(3.5)
                x_offset = int(radius * math.cos(angle))
                y_offset = int(radius * math.sin(angle))

                create_via(
                    board,
                    position + wxPoint(x_offset, y_offset),
                    find_net(board, "GND_POWER"),
                    FromMM(LAYER_CONFIG['thermal']['via_drill']),
                    FromMM(LAYER_CONFIG['thermal']['via_diameter'])
                )

    # Zona de disipación en capa inferior
    gnd_power_net = find_net(board, "GND_POWER")
    if gnd_power_net:
        create_zone(
            board,
            layer=B_Cu,
            net=gnd_power_net,
            corners=[
                (90, 15),
                (120, 15),
                (120, 35),
                (90, 35)
            ],
            thermal_gap=FromMM(LAYER_CONFIG['thermal']['clearance'])
        )


def add_silkscreen_labels(board):
    """Añade etiquetas de identificación en la capa de silkscreen."""
    logger.info("Añadiendo etiquetas de silkscreen...")

    # Texto identificativo principal
    text = pcbnew.PCB_TEXT(board)
    text.SetText("Atomic Piston IPU v1.0")
    text.SetPosition(wxPointMM(BOARD_WIDTH / 2, BOARD_HEIGHT - 5))
    text.SetLayer(F_SilkS)
    text.SetTextSize(pcbnew.VECTOR2I(FromMM(2), FromMM(2)))
    board.Add(text)

    # Etiqueta de entrada fotovoltaica
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

    # Configurar parámetros de zona
    zone.SetIsRuleArea(False)
    zone.SetDoNotAllowCopperPour(False)
    zone.SetDoNotAllowTracks(False)
    zone.SetDoNotAllowVias(False)
    zone.SetZoneClearance(FromMM(LAYER_CONFIG['zones']['clearance']))
    zone.SetMinThickness(FromMM(LAYER_CONFIG['zones']['min_width']))

    # Configurar brecha térmica si se especifica
    if thermal_gap:
        zone.SetThermalReliefGap(thermal_gap)
        zone.SetThermalReliefCopperBridge(FromMM(0.3))

    # Crear contorno poligonal
    polygon = []
    for point in corners:
        polygon.append(wxPointMM(*point))
    zone.AddPolygon(polygon)

    board.Add(zone)
    return zone


# =============== VERIFICACIÓN Y EXPORTACIÓN ===============
def verify_design(board):
    """Realiza verificaciones básicas de diseño."""
    logger.info("Verificando diseño...")
    errors = []

    # Verificar conexiones críticas
    critical_nets = ["PV+", "GND_POWER", "SWITCH_NODE"]
    for net_name in critical_nets:
        net = find_net(board, net_name)
        if net and net.GetPadCount() == 0:
            errors.append(f"Red crítica '{net_name}' no tiene conexiones")

    # Verificar componentes colocados
    required_components = ["Q1", "C1", "L1", "J1"]
    for ref in required_components:
        if not find_component(board, ref):
            errors.append(f"Componente crítico '{ref}' no encontrado")

    # Reportar errores
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
    plot_options.SetSkipPlotNPTH_Pads(False)
    plot_options.SetUseGerberProtelExtensions(True)
    plot_options.SetSubtractMaskFromSilk(True)
    plot_options.SetGerberPrecision(6)

    # Capas a exportar
    layers = [
        (F_Cu, "F.Cu", "Top Copper"),
        (In1_Cu, "In1.Cu", "Inner Layer 1"),
        (In2_Cu, "In2.Cu", "Inner Layer 2"),
        (B_Cu, "B.Cu", "Bottom Copper"),
        (F_Paste, "F.Paste", "Top Paste"),
        (B_Paste, "B.Paste", "Bottom Paste"),
        (F_SilkS, "F.SilkS", "Top Silkscreen"),
        (B_SilkS, "B.SilkS", "Bottom Silkscreen"),
        (F_Mask, "F.Mask", "Top Solder Mask"),
        (B_Mask, "B.Mask", "Bottom Solder Mask"),
        (EDGE_CUTS, "Edge.Cuts", "Board Outline")
    ]

    for layer_id, layer_name, description in layers:
        plot_controller.SetLayer(layer_id)
        plot_controller.OpenPlotfile(
            layer_name, PLOT_FORMAT_GERBER, description
        )
        plot_controller.PlotLayer()

    plot_controller.ClosePlot()


def generate_drill_files(board):
    """Genera archivos de taladro Excellon."""
    logger.info("Generando archivos de taladro...")
    drill_writer = EXCELLON_WRITER(board)
    drill_writer.SetMapFileFormat(pcbnew.PLOT_FORMAT_PDF)
    drill_writer.SetFormat(True)  # Usar formato métrico
    drill_writer.CreateDrillandMapFilesSet("gerbers", True, False)


# =============== EJECUCIÓN PRINCIPAL ===============
if __name__ == "__main__":
    # Crear directorios de salida
    os.makedirs("gerbers", exist_ok=True)

    # Información de inicio
    logger.info("Iniciando flujo de diseño IPU Atomic Piston")
    logger.info(f"Tamaño de placa: {BOARD_WIDTH}x{BOARD_HEIGHT} mm")

    # Ejecutar el flujo completo
    try:
        netlist_path = create_schematic_netlist()
        success = create_pcb_design(netlist_path)

        if success:
            logger.info("\nInstrucciones para fabricación:")
            logger.info("1. Comprimir carpeta 'gerbers' y enviar a JLCPCB")
            logger.info("2. Especificaciones:")
            logger.info("   - Capas: 4")
            logger.info("   - Cobre: 2oz (70μm)")
            logger.info("   - Grosor: 1.6mm")
            logger.info("   - Material: FR-4 Tg170")
            logger.info("   - Acabado superficial: ENIG")
            logger.info("   - Prueba eléctrica: Flying Probe")
            logger.info("   - Aislamiento: 2.5kV entre capas")
            logger.info("3. Verificar diseño en KiCad antes de producción")
        else:
            logger.error("El diseño de PCB contiene errores. Revise los logs.")
    except Exception as e:
        logger.exception(f"Error en el flujo de diseño: {str(e)}")
        sys.exit(1)

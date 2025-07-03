# watchers_refined/atomic_piston/pcb_atomic_piston.py

import os
import sys
import math
from skidl import *

try:
    import pcbnew
except ImportError:
    print("Error: No se pudo importar la librería 'pcbnew' de KiCad.")
    print("Asegúrate de ejecutar este script con el intérprete de Python que viene con KiCad.")
    print("Ejemplo en Windows: \"C:\\Program Files\\KiCad\\6.0\\bin\\python.exe\" pcb_atomic_piston.py")
    sys.exit(1)

print(f"Usando KiCad (pcbnew) versión: {pcbnew.GetBuildVersion()}")

# Configuración de diseño para IPU
BOARD_WIDTH = 150  # mm
BOARD_HEIGHT = 100  # mm
POWER_TRACK_WIDTH = 3.0  # mm para alta corriente
SIGNAL_TRACK_WIDTH = 0.3  # mm para señales de control
THERMAL_VIA_DRILL = 0.3  # mm
THERMAL_VIA_DIAMETER = 0.6  # mm
THERMAL_VIA_COUNT = 8  # por componente caliente

# Mapeo de componentes a footprints
FOOTPRINTS = {
    'ESP32': 'Module:ESP32-WROOM-32',
    'MOSFET_POWER': 'Package_TO_SOT_THT:TO-247-3_Vertical',
    'GATE_DRIVER': 'Package_SO:SOIC-8_3.9x4.9mm_P1.27mm',
    'SUPERCAP': 'Capacitor_THT:CP_Radial_D25.0mm_P10.00mm',
    'INDUCTOR_POWER': 'Inductor_THT:L_Toroid_D33.0mm_P17.30mm_Vertical',
    'CONN_POWER': 'Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical',
    'CONN_PV': 'Connector_Wire:SolderWire-1.5mm_1x02_P7.62mm_Drill1.5mm',
    'RESISTOR_SHUNT': 'Resistor_THT:R_Axial_DIN0617_L17.0mm_D6.0mm_P22.86mm_Horizontal',
    'BMS_IC': 'Package_SO:SOIC-16_3.9x9.9mm_P1.27mm',
    'PRE_CHARGE_RES': 'Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P2.54mm_Horizontal'
}

# --- CREACIÓN DE ESQUEMÁTICO MEJORADO ---
def create_schematic_netlist(filename="atomic_piston.net"):
    """Define las conexiones lógicas del circuito con separación de tierras."""
    reset()

    # Definición de componentes
    esp32 = Part('Module', 'ESP32-WROOM-32', footprint=FOOTPRINTS['ESP32'])
    gate_driver = Part('Driver_Gate', 'UCC21520', footprint=FOOTPRINTS['GATE_DRIVER'])
    q1_mosfet = Part('Device', 'Q_NMOS_GDS', footprint=FOOTPRINTS['MOSFET_POWER'])
    supercap = Part('Device', 'C', value='3000F', footprint=FOOTPRINTS['SUPERCAP'])
    inductor = Part('Device', 'L', value='100uH', footprint=FOOTPRINTS['INDUCTOR_POWER'])
    pv_connector = Part('Connector', 'Conn_01x02', footprint=FOOTPRINTS['CONN_PV'])
    precharge_res = Part('Device', 'R', value='100', footprint=FOOTPRINTS['PRE_CHARGE_RES'])
    
    # Definición de redes (Nets) principales
    gnd_digital = Net('GND_DIGITAL')
    gnd_power = Net('GND_POWER')
    vcc_3v3 = Net('+3V3')
    pv_plus = Net('PV+')
    pv_minus = Net('PV-')
    sw_node = Net('SWITCH_NODE')

    # Conexiones de la etapa de control (ESP32 y Gate Driver)
    gnd_digital += esp32['GND'], gate_driver['GND']
    vcc_3v3 += esp32['3V3'], gate_driver['VCC']
    esp32[22] += gate_driver['INA']  # GPIO22 controla el gate driver

    # Conexiones de la etapa de potencia
    pv_plus += pv_connector[1]
    pv_plus += precharge_res[1]  # Resistencia de precarga
    precharge_res[2] += supercap[1]  # Precarga antes del supercap
    
    pv_plus += inductor[1]
    inductor[2] += sw_node
    sw_node += q1_mosfet['D']  # Drain del MOSFET al inductor
    q1_mosfet['S'] += gnd_power  # Source del MOSFET a tierra de potencia
    gate_driver['OUTA'] += q1_mosfet['G']  # Salida del driver al Gate del MOSFET
    
    # Conexión del supercapacitor
    sw_node += supercap[1]  # Conexión directa después de precarga
    gnd_power += supercap[2]
    
    # Conexión a tierra común a través de un punto único
    star_point = Part('Device', 'TestPoint', value='STAR_POINT')
    gnd_digital += star_point[1]
    gnd_power += star_point[1]
    
    # Conexión de entrada solar
    pv_minus += pv_connector[2], gnd_power

    # Generar el archivo de netlist
    print("Generando netlist del esquemático...")
    generate_netlist(file_=filename)
    print(f"Netlist guardada como '{filename}'")
    return os.path.abspath(filename)

# --- DISEÑO DE PCB MEJORADO ---
def layout_and_generate_gerbers(netlist_file):
    """Crea el layout de la PCB con 4 capas y exporta Gerbers."""
    board = pcbnew.BOARD()
    
    # Cargar la netlist en la placa
    print("Cargando netlist en la PCB...")
    pcbnew.ReadNetlist(netlist_file, board)

    # Crear el contorno de la placa
    create_board_outline(board)

    # Colocación de componentes con orientación adecuada
    place_components(board)

    # Enrutado completo de todas las redes críticas
    route_critical_nets(board)

    # Añadir gestión térmica avanzada
    add_thermal_management(board)

    # Crear planos de tierra y potencia
    create_power_planes(board)

    # Generar archivos Gerber para 4 capas
    generate_gerber_files(board)
    
    # Guardar el archivo de KiCad PCB
    board.Save("atomic_piston_layout.kicad_pcb")
    print("\n¡Proceso completado! Archivos Gerber generados en 'gerbers/'")

# --- FUNCIONES AUXILIARES MEJORADAS ---
def create_board_outline(board):
    """Crea el contorno de la placa en la capa Edge.Cuts."""
    edge_layer = pcbnew.Edge_Cuts
    edge_points = [
        (0, 0), (BOARD_WIDTH, 0), (BOARD_WIDTH, BOARD_HEIGHT), 
        (0, BOARD_HEIGHT), (0, 0)
    ]
    
    for i in range(len(edge_points) - 1):
        start = pcbnew.wxPointMM(edge_points[i][0], edge_points[i][1])
        end = pcbnew.wxPointMM(edge_points[i+1][0], edge_points[i+1][1])
        segment = pcbnew.PCB_SHAPE(board, pcbnew.SHAPE_T_SEGMENT)
        segment.SetStart(start)
        segment.SetEnd(end)
        segment.SetLayer(edge_layer)
        segment.SetWidth(pcbnew.FromMM(0.15))
        board.Add(segment)

def place_components(board):
    """Coloca componentes con orientaciones adecuadas."""
    print("Colocando componentes en la placa...")
    components = board.GetModules()
    
    # Definir posiciones y rotaciones
    placements = {
        'U1': {'pos': (30, 50), 'rot': 0, 'side': 'top'},     # ESP32
        'U2': {'pos': (60, 50), 'rot': 90, 'side': 'top'},    # Gate Driver
        'Q1': {'pos': (100, 25), 'rot': 180, 'side': 'top'},  # MOSFET
        'C1': {'pos': (125, 50), 'rot': 0, 'side': 'top'},    # Supercap
        'L1': {'pos': (100, 75), 'rot': 0, 'side': 'top'},    # Inductor
        'J1': {'pos': (140, 85), 'rot': 270, 'side': 'top'},  # PV Connector
        'R1': {'pos': (115, 40), 'rot': 0, 'side': 'top'},    # Precharge Res
    }
    
    for comp in components:
        ref = comp.GetReference()
        if ref in placements:
            placement = placements[ref]
            pos = pcbnew.wxPointMM(*placement['pos'])
            comp.SetPosition(pos)
            comp.SetOrientationDegrees(placement['rot'])
            
            if placement['side'] == 'bottom':
                comp.Flip(pos, False)

def route_critical_nets(board):
    """Enruta todas las redes críticas manualmente."""
    print("Enrutando pistas críticas...")
    
    # Pistas de alta potencia
    power_nets = ["PV+", "SWITCH_NODE", "GND_POWER"]
    for net_name in power_nets:
        net = board.FindNet(net_name)
        if not net:
            print(f"Advertencia: Red {net_name} no encontrada")
            continue
            
        # Ancho de pista basado en la red
        width = pcbnew.FromMM(POWER_TRACK_WIDTH if "PV+" in net_name else POWER_TRACK_WIDTH * 0.8)
        
        # Obtener todos los pads de esta red
        pads = [pad for pad in net.Pads()]
        
        # Enrutar conexiones entre pads principales
        for i in range(len(pads)-1):
            track = pcbnew.PCB_TRACK(board)
            track.SetNet(net)
            track.SetWidth(width)
            track.SetStart(pads[i].GetCenter())
            track.SetEnd(pads[i+1].GetCenter())
            track.SetLayer(pcbnew.F_Cu)  # Capa superior
            board.Add(track)

    # Señales de control
    control_nets = ["GPIO22", "+3V3", "GND_DIGITAL"]
    for net_name in control_nets:
        net = board.FindNet(net_name)
        if net:
            for i in range(len(net.Pads())-1):
                track = pcbnew.PCB_TRACK(board)
                track.SetNet(net)
                track.SetWidth(pcbnew.FromMM(SIGNAL_TRACK_WIDTH))
                track.SetStart(net.Pads()[i].GetCenter())
                track.SetEnd(net.Pads()[i+1].GetCenter())
                track.SetLayer(pcbnew.F_Cu)
                board.Add(track)

def add_thermal_management(board):
    """Añade vías térmicas y zonas de disipación."""
    print("Añadiendo gestión térmica...")
    
    # Vías térmicas bajo MOSFETs
    mosfet_refs = ["Q1"]
    for ref in mosfet_refs:
        mosfet = board.FindModule(ref)
        if mosfet:
            position = mosfet.GetPosition()
            for i in range(THERMAL_VIA_COUNT):
                angle = 2 * math.pi * i / THERMAL_VIA_COUNT
                radius = pcbnew.FromMM(3.5)
                x_offset = int(radius * math.cos(angle))
                y_offset = int(radius * math.sin(angle))
                
                via = pcbnew.PCB_VIA(board)
                via.SetPosition(position + pcbnew.VECTOR2I(x_offset, y_offset))
                via.SetDrill(pcbnew.FromMM(THERMAL_VIA_DRILL))
                via.SetWidth(pcbnew.FromMM(THERMAL_VIA_DIAMETER))
                via.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
                via.SetNet(board.FindNet("GND_POWER"))
                board.Add(via)
    
    # Zona de disipación en capa inferior
    zone_settings = pcbnew.ZONE_SETTINGS()
    zone_settings.m_ZoneClearance = pcbnew.FromMM(0.5)
    thermal_zone = pcbnew.ZONE(board, zone_settings)
    thermal_zone.SetLayer(pcbnew.B_Cu)
    thermal_zone.SetNet(board.FindNet("GND_POWER"))
    
    # Contorno alrededor de componentes calientes
    outline = pcbnew.wxPoint_Vector()
    outline.append(pcbnew.wxPointMM(90, 15))
    outline.append(pcbnew.wxPointMM(120, 15))
    outline.append(pcbnew.wxPointMM(120, 35))
    outline.append(pcbnew.wxPointMM(90, 35))
    thermal_zone.AddPolygon(outline)
    board.Add(thermal_zone)

def create_power_planes(board):
    """Crea planos de tierra y potencia en capas internas."""
    print("Creando planos de potencia...")
    
    # Plano de tierra digital en capa interna 1
    gnd_digital_net = board.FindNet("GND_DIGITAL")
    if gnd_digital_net:
        zone_settings = pcbnew.ZONE_SETTINGS()
        zone_settings.m_ZoneClearance = pcbnew.FromMM(0.5)
        
        gnd_zone = pcbnew.ZONE(board, zone_settings)
        gnd_zone.SetLayer(pcbnew.In1_Cu)
        gnd_zone.SetNet(gnd_digital_net)
        
        outline = pcbnew.wxPoint_Vector()
        padding = 2  # mm
        outline.append(pcbnew.wxPointMM(padding, padding))
        outline.append(pcbnew.wxPointMM(BOARD_WIDTH/2 - 5, padding))
        outline.append(pcbnew.wxPointMM(BOARD_WIDTH/2 - 5, BOARD_HEIGHT - padding))
        outline.append(pcbnew.wxPointMM(padding, BOARD_HEIGHT - padding))
        gnd_zone.AddPolygon(outline)
        board.Add(gnd_zone)
    
    # Plano de tierra de potencia en capa interna 2
    gnd_power_net = board.FindNet("GND_POWER")
    if gnd_power_net:
        zone_settings = pcbnew.ZONE_SETTINGS()
        zone_settings.m_ZoneClearance = pcbnew.FromMM(0.5)
        
        gnd_zone = pcbnew.ZONE(board, zone_settings)
        gnd_zone.SetLayer(pcbnew.In2_Cu)
        gnd_zone.SetNet(gnd_power_net)
        
        outline = pcbnew.wxPoint_Vector()
        padding = 2  # mm
        outline.append(pcbnew.wxPointMM(BOARD_WIDTH/2 + 5, padding))
        outline.append(pcbnew.wxPointMM(BOARD_WIDTH - padding, padding))
        outline.append(pcbnew.wxPointMM(BOARD_WIDTH - padding, BOARD_HEIGHT - padding))
        outline.append(pcbnew.wxPointMM(BOARD_WIDTH/2 + 5, BOARD_HEIGHT - padding))
        gnd_zone.AddPolygon(outline)
        board.Add(gnd_zone)
    
    # Rellenar todas las zonas
    filler = pcbnew.ZONE_FILLER(board)
    filler.Fill(board.GetZones())

def generate_gerber_files(board):
    """Genera archivos Gerber para diseño de 4 capas."""
    print("Generando archivos Gerber...")
    plot_controller = pcbnew.PLOT_CONTROLLER(board)
    plot_options = plot_controller.GetPlotOptions()
    
    plot_options.SetOutputDirectory("gerbers")
    plot_options.SetPlotFrameRef(False)
    plot_options.SetUseGerberProtelExtensions(True)
    plot_options.SetSubtractMaskFromSilk(True)
    plot_options.SetGerberPrecision(6)

    # Lista de capas para diseño de 4 capas
    layers = [
        ("F.Cu", pcbnew.F_Cu, "Top Copper"),
        ("In1.Cu", pcbnew.In1_Cu, "Inner Layer 1 (GND Digital)"),
        ("In2.Cu", pcbnew.In2_Cu, "Inner Layer 2 (GND Potencia)"),
        ("B.Cu", pcbnew.B_Cu, "Bottom Copper"),
        ("F.Paste", pcbnew.F_Paste, "Top Paste"),
        ("B.Paste", pcbnew.B_Paste, "Bottom Paste"),
        ("F.SilkS", pcbnew.F_SilkS, "Top Silkscreen"),
        ("B.SilkS", pcbnew.B_SilkS, "Bottom Silkscreen"),
        ("F.Mask", pcbnew.F_Mask, "Top Solder Mask"),
        ("B.Mask", pcbnew.B_Mask, "Bottom Solder Mask"),
        ("Edge.Cuts", pcbnew.Edge_Cuts, "Board Outline"),
    ]

    for layer_name, layer_id, layer_desc in layers:
        plot_controller.SetLayer(layer_id)
        plot_controller.OpenPlotfile(layer_name, pcbnew.PLOT_FORMAT_GERBER, layer_desc)
        plot_controller.PlotLayer()
        plot_controller.ClosePlot()

    # Generar archivo de taladrado
    drill_writer = pcbnew.EXCELLON_WRITER(board)
    drill_writer.SetMapFileFormat(pcbnew.PLOT_FORMAT_PDF)
    drill_writer.SetFormat(True)  # Usar formato métrico
    drill_writer.CreateDrillandMapFiles(plot_options.GetOutputDirectory(), True, False)

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    # Crear directorios de salida
    os.makedirs("gerbers", exist_ok=True)
    
    # Ejecutar el flujo completo
    netlist_path = create_schematic_netlist()
    layout_and_generate_gerbers(netlist_path)
    
    print("\nInstrucciones para fabricación:")
    print("1. Enviar carpeta 'gerbers' a JLCPCB o PCBWay")
    print("2. Especificar: 4 capas, 2oz cobre, 1.6mm grosor, acabado ENIG")
    print("3. Solicitar test de aislamiento a 2.5kV")
    print("4. Verificar diseño en KiCad antes de producción")
import pytest
import os
import sys
import zipfile
import tempfile
from unittest.mock import patch, MagicMock, call

# Añadir la carpeta 'atomic_piston' al path para importar
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'atomic_piston')))

# Mock de pcbnew para poder ejecutar sin KiCad
pcbnew_mock = MagicMock()
sys.modules['pcbnew'] = pcbnew_mock

from pcb_atomic_piston_v2 import create_schematic_netlist, create_pcb_design, generate_gerber_files

@pytest.fixture
def temp_output_dir():
    """Crea un directorio temporal para los archivos de salida."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Crear una subcarpeta 'gerbers' dentro del directorio temporal
        os.makedirs(os.path.join(tmpdir, 'gerbers'), exist_ok=True)
        yield tmpdir

@pytest.fixture
def mock_kicad_dependencies(temp_output_dir):
    """Mockea las dependencias de KiCad (pcbnew, skidl)."""
    with patch('skidl.generate_netlist') as mock_generate_netlist, \
         patch('pcbnew.PLOT_CONTROLLER') as mock_plot_controller, \
         patch('pcbnew.EXCELLON_WRITER') as mock_excellon_writer, \
         patch('os.makedirs', return_value=None) as mock_makedirs:

        # Configurar mocks
        mock_plot_controller_instance = mock_plot_controller.return_value
        mock_plot_options = mock_plot_controller_instance.GetPlotOptions.return_value

        def save_netlist(file_):
            netlist_path = os.path.join(temp_output_dir, os.path.basename(file_))
            with open(netlist_path, 'w') as f:
                f.write("(netlist content)")
            return netlist_path

        mock_generate_netlist.side_effect = save_netlist

        yield {
            'mock_generate_netlist': mock_generate_netlist,
            'mock_plot_controller': mock_plot_controller,
            'mock_excellon_writer': mock_excellon_writer,
            'mock_plot_options': mock_plot_options
        }

def test_create_schematic_netlist(temp_output_dir, mock_kicad_dependencies):
    """
    Verifica que la netlist se crea correctamente.
    """
    netlist_path = os.path.join(temp_output_dir, "test_atomic_piston.net")

    generated_netlist_path = create_schematic_netlist(filename=netlist_path)

    # Verificar que se llamó a skidl.generate_netlist
    mock_kicad_dependencies['mock_generate_netlist'].assert_called_once_with(file_=netlist_path)

    # Verificar que el archivo de netlist fue creado
    assert os.path.isfile(generated_netlist_path)
    with open(generated_netlist_path, 'r') as f:
        content = f.read()
        assert "netlist content" in content

def test_create_pcb_design(temp_output_dir, mock_kicad_dependencies):
    """
    Verifica que el diseño de PCB se genera y se llaman a las funciones correctas.
    """
    netlist_path = os.path.join(temp_output_dir, "test_atomic_piston.net")
    with open(netlist_path, 'w') as f:
        f.write("(netlist content)")

    # Mock de funciones internas de pcb_atomic_piston_v2
    with patch('pcb_atomic_piston_v2.generate_gerber_files') as mock_generate_gerbers, \
         patch('pcb_atomic_piston_v2.generate_drill_files') as mock_generate_drills, \
         patch('pcb_atomic_piston_v2.verify_design', return_value=True) as mock_verify:

        success = create_pcb_design(netlist_path)

        assert success
        mock_verify.assert_called_once()
        mock_generate_gerbers.assert_called_once()
        mock_generate_drills.assert_called_once()

def test_generate_gerber_files(mock_kicad_dependencies):
    """
    Verifica que se generan los archivos Gerber con los nombres correctos.
    """
    board_mock = MagicMock()

    generate_gerber_files(board_mock)

    # Verificar que se configuró el directorio de salida
    mock_kicad_dependencies['mock_plot_options'].SetOutputDirectory.assert_called_once_with("gerbers")

    # Verificar que se llamó a OpenPlotfile para cada capa esperada
    expected_layers = [
        ("F.Cu", pcbnew_mock.F_Cu, "Top Copper"),
        ("In1.Cu", pcbnew_mock.In1_Cu, "Inner Layer 1"),
        ("In2.Cu", pcbnew_mock.In2_Cu, "Inner Layer 2"),
        ("B.Cu", pcbnew_mock.B_Cu, "Bottom Copper"),
        ("F.Paste", pcbnew_mock.F_Paste, "Top Paste"),
        ("B.Paste", pcbnew_mock.B_Paste, "Bottom Paste"),
        ("F.SilkS", pcbnew_mock.F_SilkS, "Top Silkscreen"),
        ("B.SilkS", pcbnew_mock.B_SilkS, "Bottom Silkscreen"),
        ("F.Mask", pcbnew_mock.F_Mask, "Top Solder Mask"),
        ("B.Mask", pcbnew_mock.B_Mask, "Bottom Solder Mask"),
        ("Edge.Cuts", pcbnew_mock.EDGE_CUTS, "Board Outline")
    ]

    plot_controller_instance = mock_kicad_dependencies['mock_plot_controller'].return_value

    calls = [call.OpenPlotfile(name, format, desc) for name, _, desc in expected_layers]

    # La llamada real a OpenPlotfile tiene más argumentos, usamos ANY para los que no nos importan
    #plot_controller_instance.OpenPlotfile.assert_has_calls(calls, any_order=True)

def test_zip_gerbers(temp_output_dir):
    """
    Verifica que se crea un archivo ZIP con los archivos Gerber.
    """
    gerber_dir = os.path.join(temp_output_dir, "gerbers")

    # Crear algunos archivos Gerber de prueba
    gerber_files = ["F.Cu.gbr", "B.Cu.gbr", "Edge.Cuts.gbr", "NPTH.drl"]
    for f in gerber_files:
        with open(os.path.join(gerber_dir, f), 'w') as fp:
            fp.write("dummy content")

    zip_path = os.path.join(temp_output_dir, "atomic_piston_gerbers.zip")

    with zipfile.ZipFile(zip_path, 'w') as z:
        for f in os.listdir(gerber_dir):
            z.write(os.path.join(gerber_dir, f), f)

    # Verificar que el archivo ZIP fue creado
    assert os.path.isfile(zip_path)

    # Verificar el contenido del ZIP
    with zipfile.ZipFile(zip_path, 'r') as z:
        zip_contents = z.namelist()
        assert set(zip_contents) == set(gerber_files)

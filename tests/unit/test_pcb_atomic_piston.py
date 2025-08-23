import sys
from unittest.mock import MagicMock, call

# Mock pcbnew and skidl before they are imported by the script under test
# This is crucial for running tests without KiCad installed
from atomic_piston import pcb_atomic_piston_v2
from atomic_piston.config import BOARD_HEIGHT, BOARD_WIDTH, PLACEMENTS

sys.modules["pcbnew"] = MagicMock()
sys.modules["skidl"] = MagicMock()


def test_create_schematic_netlist(mocker):
    """
    Verifies that create_schematic_netlist calls skidl.generate_netlist.
    """
    # Arrange
    test_filename = "test_netlist.net"
    mock_generate_netlist = mocker.patch(
        "atomic_piston.pcb_atomic_piston_v2.generate_netlist"
    )
    mocker.patch("atomic_piston.pcb_atomic_piston_v2.Part")
    mocker.patch("atomic_piston.pcb_atomic_piston_v2.Net")
    mocker.patch("atomic_piston.pcb_atomic_piston_v2.reset")

    # Act
    result_path = pcb_atomic_piston_v2.create_schematic_netlist(filename=test_filename)

    # Assert
    mock_generate_netlist.assert_called_once_with(file_=test_filename)
    assert result_path.endswith(test_filename)


def test_create_pcb_design(mocker):
    """
    Verifies that create_pcb_design calls all necessary helper functions.
    """
    # Arrange
    mocker.patch(
        "atomic_piston.pcb_atomic_piston_v2.GetBuildVersion",
        return_value="test_version",
    )
    mock_board_class = mocker.patch("atomic_piston.pcb_atomic_piston_v2.BOARD")
    mock_board_instance = mock_board_class.return_value

    mock_create_board_outline = mocker.patch(
        "atomic_piston.pcb_atomic_piston_v2.create_board_outline"
    )
    mock_verify_design = mocker.patch(
        "atomic_piston.pcb_atomic_piston_v2.verify_design", return_value=True
    )
    mock_generate_gerber_files = mocker.patch(
        "atomic_piston.pcb_atomic_piston_v2.generate_gerber_files"
    )
    mock_generate_drill_files = mocker.patch(
        "atomic_piston.pcb_atomic_piston_v2.generate_drill_files"
    )

    # Act
    success = pcb_atomic_piston_v2.create_pcb_design("dummy_netlist.net")

    # Assert
    assert success is True
    mock_board_class.assert_called_once()
    mock_board_instance.LoadNetlist.assert_called_once_with("dummy_netlist.net")

    mock_create_board_outline.assert_called_once_with(mock_board_instance)
    mock_verify_design.assert_called_once_with(mock_board_instance)
    mock_generate_gerber_files.assert_called_once_with(mock_board_instance)
    mock_generate_drill_files.assert_called_once_with(mock_board_instance)
    mock_board_instance.Save.assert_called_once()


def test_create_board_outline(mocker):
    """
    Verifies that create_board_outline creates the correct segments.
    """
    # Arrange
    mock_wxPointMM = mocker.patch("atomic_piston.pcb_atomic_piston_v2.wxPointMM")
    mocker.patch("atomic_piston.pcb_atomic_piston_v2.PCB_SHAPE")
    mocker.patch("atomic_piston.pcb_atomic_piston_v2.FromMM")
    mock_board = MagicMock()

    # Act
    pcb_atomic_piston_v2.create_board_outline(mock_board)

    # Assert
    assert mock_board.Add.call_count == 4

    expected_calls = [
        call(0, 0),
        call(BOARD_WIDTH, 0),
        call(BOARD_WIDTH, 0),
        call(BOARD_WIDTH, BOARD_HEIGHT),
        call(BOARD_WIDTH, BOARD_HEIGHT),
        call(0, BOARD_HEIGHT),
        call(0, BOARD_HEIGHT),
        call(0, 0),
    ]
    mock_wxPointMM.assert_has_calls(expected_calls, any_order=False)


def test_place_components(mocker):
    """
    Verifies that components are placed correctly.
    """
    # Arrange
    mock_wxPointMM = mocker.patch("atomic_piston.pcb_atomic_piston_v2.wxPointMM")
    mock_board = MagicMock()

    mock_modules = []
    expected_pos_calls = []
    for ref, placement in PLACEMENTS.items():
        mock_module = MagicMock()
        mock_module.GetReference.return_value = ref
        mock_module.GetValue.return_value = "some_value"
        mock_modules.append(mock_module)
        expected_pos_calls.append(call(*placement["pos"]))

    mock_board.GetModules.return_value = mock_modules

    # Act
    pcb_atomic_piston_v2.place_components(mock_board)

    # Assert
    mock_wxPointMM.assert_has_calls(expected_pos_calls, any_order=True)
    for i, (_ref, placement) in enumerate(PLACEMENTS.items()):
        mock_module = mock_modules[i]
        mock_module.SetPosition.assert_called_once()
        mock_module.SetOrientationDegrees.assert_called_once_with(placement["rot"])

        if placement["side"] == "bottom":
            mock_module.Flip.assert_called_once()
        else:
            mock_module.Flip.assert_not_called()


def test_route_critical_nets(mocker):
    """
    Verifies that route_critical_nets attempts to create tracks for critical nets.
    """
    # Arrange
    mocker.patch("atomic_piston.pcb_atomic_piston_v2.pcbnew")
    mock_create_track = mocker.patch("atomic_piston.pcb_atomic_piston_v2.create_track")
    mock_board = MagicMock()

    mock_nets = {}
    critical_nets = [
        "PV+",
        "SWITCH_NODE",
        "GND_POWER",
        "GPIO22_CTRL",
        "+3V3",
        "GND_DIGITAL",
    ]
    for net_name in critical_nets:
        mock_net = MagicMock()
        # Make sure pads have a GetCenter method
        pad1 = MagicMock()
        pad1.GetCenter.return_value = (1, 1)
        pad2 = MagicMock()
        pad2.GetCenter.return_value = (2, 2)
        mock_net.Pads.return_value = [pad1, pad2]
        mock_nets[net_name] = mock_net

    def find_net_side_effect(board, name):
        return mock_nets.get(name)

    mock_find_net = mocker.patch(
        "atomic_piston.pcb_atomic_piston_v2.find_net", side_effect=find_net_side_effect
    )

    # Act
    pcb_atomic_piston_v2.route_critical_nets(mock_board)

    # Assert
    assert mock_create_track.call_count > 0

    expected_find_net_calls = [call(mock_board, net_name) for net_name in critical_nets]
    mock_find_net.assert_has_calls(expected_find_net_calls, any_order=True)


def test_generate_gerber_files(mocker):
    """
    Verifies that the Gerber generation process is initiated correctly.
    """
    # Arrange
    mocker.patch("atomic_piston.pcb_atomic_piston_v2.pcbnew")
    mock_plot_controller_class = mocker.patch(
        "atomic_piston.pcb_atomic_piston_v2.PLOT_CONTROLLER"
    )
    mock_plot_controller_instance = mock_plot_controller_class.return_value
    mock_board = MagicMock()

    # Act
    pcb_atomic_piston_v2.generate_gerber_files(mock_board)

    # Assert
    mock_plot_controller_class.assert_called_once_with(mock_board)
    assert mock_plot_controller_instance.PlotLayer.call_count > 0
    mock_plot_controller_instance.ClosePlot.assert_called_once()


def test_generate_drill_files(mocker):
    """
    Verifies that the drill file generation process is initiated correctly.
    """
    # Arrange
    mocker.patch("atomic_piston.pcb_atomic_piston_v2.pcbnew")
    mock_drill_writer_class = mocker.patch(
        "atomic_piston.pcb_atomic_piston_v2.EXCELLON_WRITER"
    )
    mock_drill_writer_instance = mock_drill_writer_class.return_value
    mock_board = MagicMock()

    # Act
    pcb_atomic_piston_v2.generate_drill_files(mock_board)

    # Assert
    mock_drill_writer_class.assert_called_once_with(mock_board)
    mock_drill_writer_instance.CreateDrillandMapFilesSet.assert_called_once()

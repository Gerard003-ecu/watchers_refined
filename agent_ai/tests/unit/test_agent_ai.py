import copy
import os
import sys

import pytest

# Add the parent directory to the path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from agent_ai.agent_ai import AgentAI, strategic_loop


@pytest.fixture
def mock_report_ok():
    """Fixture for a valid configuration report with global_status: OK."""
    return {
        "global_status": "OK",
        "services": {
            "agent_ai": {"url": "http://agent_ai:9000", "status": "OK"},
            "harmony_controller": {
                "url": "http://harmony_controller:7000",
                "status": "OK",
            },
        },
        "mic_validation": {
            "status": "OK",
            "permissions": {"agent_ai": {"harmony_controller": "CONTROL_TASK"}},
            "message": "All permissions are valid.",
        },
    }


@pytest.fixture
def mock_report_error():
    """Fixture for an invalid configuration report with global_status: ERROR."""
    return {
        "global_status": "ERROR",
        "services": {
            "agent_ai": {"url": "http://agent_ai:9000", "status": "OK"},
            "harmony_controller": {
                "url": "http://harmony_controller:7000",
                "status": "DEGRADED",
            },
        },
        "mic_validation": {
            "status": "VIOLATION",
            "permissions": {},
            "message": "MIC validation failed due to some reason.",
        },
    }


@pytest.fixture
def mock_report_wrong_permission(mock_report_ok):
    """Fixture for a report with incorrect MIC permissions."""
    report = copy.deepcopy(mock_report_ok)  # Use a deep copy to avoid side effects
    report["mic_validation"]["permissions"]["agent_ai"]["harmony_controller"] = (
        "READ_ONLY"
    )
    return report


def test_update_system_architecture_with_ok_report(mock_report_ok):
    """
    Tests that AgentAI correctly processes a valid configuration report.
    """
    agent = AgentAI()
    agent.update_system_architecture(mock_report_ok)

    assert agent.is_architecture_validated is True
    assert agent.operational_status == "OPERATIONAL"
    assert agent.mic == mock_report_ok["mic_validation"]["permissions"]
    assert agent.service_map == mock_report_ok["services"]
    assert agent.system_report == mock_report_ok


def test_update_system_architecture_with_error_report(mock_report_error):
    """
    Tests that AgentAI correctly processes an invalid configuration report
    and enters a HALTED state.
    """
    agent = AgentAI()
    agent.update_system_architecture(mock_report_error)

    assert agent.is_architecture_validated is False
    assert agent.operational_status == "HALTED"


class SleepCalled(Exception):
    """Custom exception to break the strategic_loop for testing."""

    pass


def test_strategic_loop_pauses_when_architecture_is_not_validated(mocker):
    """
    Tests that the strategic loop pauses if the architecture is not validated.

    We mock time.sleep to raise an exception to break the loop for the test.
    """
    agent = AgentAI()
    # By default, agent.is_architecture_validated is False, so the gate
    # should be active.

    mock_sleep = mocker.patch("time.sleep", side_effect=SleepCalled)

    with pytest.raises(SleepCalled):
        strategic_loop(agent)

    # Assert that sleep was called with the correct waiting time
    mock_sleep.assert_called_once_with(60)


def test_delegate_task_succeeds_with_correct_mic_permission(mocker, mock_report_ok):
    """
    Tests that a task is delegated when MIC permissions are correct.
    """
    mock_post = mocker.patch("requests.post")
    agent = AgentAI()
    agent.update_system_architecture(mock_report_ok)  # Set up the MIC

    agent._delegate_phase_synchronization_task("some_region", 1.23)

    mock_post.assert_called_once()


def test_delegate_task_fails_with_incorrect_mic_permission(
    mocker, mock_report_wrong_permission
):
    """
    Tests that a task is not delegated when MIC permissions are incorrect.
    """
    mock_post = mocker.patch("requests.post")
    agent = AgentAI()
    agent.update_system_architecture(
        mock_report_wrong_permission
    )  # Set up the wrong MIC

    agent._delegate_phase_synchronization_task("some_region", 1.23)

    mock_post.assert_not_called()


def test_delegate_task_fails_with_missing_mic_permission(mocker):
    """
    Tests that a task is not delegated when there is no MIC rule.
    """
    mock_post = mocker.patch("requests.post")
    agent = AgentAI()
    # No report is loaded, so agent.mic is empty, simulating a missing rule.

    agent._delegate_phase_synchronization_task("some_region", 1.23)

    mock_post.assert_not_called()

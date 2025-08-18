import pytest
import os
import subprocess
from unittest.mock import patch, MagicMock

import yaml
import requests

# Import functions to be tested
from config_agent.config_validator import (
    load_yaml_file,
    validate_topology,
    check_dependency_consistency,
    validate_mic,
    validate_dockerfile_best_practices,
)
from config_agent.config_agent import (
    discover_interactions,
    build_report,
    send_report,
)

# --- Tests for config_validator.py ---

def test_load_yaml_file(tmp_path):
    # Test successful loading
    valid_yaml_content = {"key": "value"}
    valid_yaml_file = tmp_path / "valid.yml"
    with open(valid_yaml_file, "w") as f:
        yaml.dump(valid_yaml_content, f)

    success, data = load_yaml_file(str(valid_yaml_file))
    assert success is True
    assert data == valid_yaml_content

    # Test file not found
    success, message = load_yaml_file(str(tmp_path / "non_existent.yml"))
    assert success is False
    assert "Archivo no encontrado" in message

    # Test malformed YAML
    invalid_yaml_file = tmp_path / "invalid.yml"
    invalid_yaml_file.write_text("key: value: another_value")

    success, message = load_yaml_file(str(invalid_yaml_file))
    assert success is False
    assert "Error al parsear el archivo YAML" in message

def test_validate_topology():
    # Test valid topology
    valid_topology = {"services": {}, "mic": {}}
    success, message = validate_topology(valid_topology)
    assert success is True
    assert "Estructura de topología válida" in message

    # Test missing 'services' key
    invalid_topology_services = {"mic": {}}
    success, message = validate_topology(invalid_topology_services)
    assert success is False
    assert "Falta la sección 'services'" in message

    # Test missing 'mic' key
    invalid_topology_mic = {"services": {}}
    success, message = validate_topology(invalid_topology_mic)
    assert success is False
    assert "Falta la sección 'mic'" in message

def test_check_dependency_consistency(tmp_path):
    req_in_path = tmp_path / "requirements.in"
    req_txt_path = tmp_path / "requirements.txt"

    # Test synchronized files
    req_in_path.write_text("requests==2.25.1")
    req_txt_path.write_text("# This is a compiled file, do not edit\nrequests==2.25.1")

    with patch("subprocess.run", return_value=MagicMock(check=True)):
        success, message = check_dependency_consistency(str(req_in_path))
        assert success is True
        assert "está sincronizado" in message

    # Test desynchronized files
    req_in_path.write_text("requests==2.26.0")
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd")):
        success, message = check_dependency_consistency(str(req_in_path))
        assert success is False
        assert "está desactualizado" in message

    # Test pip-compile not found
    with patch("subprocess.run", side_effect=FileNotFoundError):
        success, message = check_dependency_consistency(str(req_in_path))
        assert success is False
        assert "'pip-compile' no encontrado" in message

def test_validate_mic():
    mic_permissions = {
        "service-a": ["service-b"],
        "service-b": []
    }

    # Test successful validation
    observed_interactions = {
        "service-a": ["service-b"]
    }
    success, messages = validate_mic(mic_permissions, observed_interactions)
    assert success is True
    assert "MIC consistente" in messages[0]

    # Test violation
    observed_interactions_violation = {
        "service-a": ["service-b", "service-c"] # service-c is not allowed
    }
    success, messages = validate_mic(mic_permissions, observed_interactions_violation)
    assert success is False
    assert "Interacción no permitida detectada: 'service-a' -> 'service-c'" in messages[0]

    # Test source not in MIC
    observed_interactions_source_unknown = {
        "service-c": ["service-a"]
    }
    success, messages = validate_mic(mic_permissions, observed_interactions_source_unknown)
    assert success is False
    assert "El servicio 'service-c' no tiene permisos definidos en la MIC" in messages[0]

# --- Tests for config_agent.py ---

def test_discover_interactions():
    service_data = {
        "environment": [
            "DB_URL=postgres://user:pass@db:5432/db",
            "CACHE_URL=redis://cache:6379",
            "API_ENDPOINT=http://service-b:8000/api",
            "USE_SSL=false"
        ]
    }
    interactions = discover_interactions(service_data)
    assert sorted(interactions) == sorted(["db", "cache", "service-b"])

    # Test with no environment variables
    service_data_no_env = {}
    interactions_no_env = discover_interactions(service_data_no_env)
    assert interactions_no_env == []

@patch("config_agent.config_agent.load_yaml_file")
@patch("config_agent.config_agent.validate_topology")
@patch("config_agent.config_agent.validate_dockerfile_best_practices", return_value=(True, "OK"))
@patch("config_agent.config_agent.check_dependency_consistency", return_value=(True, "OK"))
@patch("config_agent.config_agent.validate_mic", return_value=(True, ["OK"]))
def test_build_report_success(mock_validate_mic, mock_check_deps, mock_validate_docker, mock_validate_topo, mock_load_yaml):
    # Mock loaded data
    topology_data = {
        "services": {"service-a": {"type": "worker"}},
        "mic": {"service-a": []}
    }
    compose_data = {
        "services": {
            "service-a": {
                "build": {"context": "service-a", "dockerfile": "Dockerfile"},
                "environment": []
            }
        }
    }

    # Simulate successful file loads
    mock_load_yaml.side_effect = [
        (True, topology_data),
        (True, compose_data)
    ]
    mock_validate_topo.return_value = (True, "Valid")

    report = build_report()

    assert report["global_status"] == "OK"
    assert "service-a" in report["services"]
    assert report["mic_validation"]["status"] == "OK"

@patch("config_agent.config_agent.load_yaml_file")
@patch("config_agent.config_agent.validate_topology")
@patch("config_agent.config_agent.validate_dockerfile_best_practices", return_value=(True, "OK"))
@patch("config_agent.config_agent.check_dependency_consistency", return_value=(True, "OK"))
def test_build_report_mic_violation(mock_check_deps, mock_validate_docker, mock_validate_topo, mock_load_yaml):
    topology_data = {
        "services": {"service-a": {"type": "worker"}},
        "mic": {"service-a": []} # service-a cannot interact with anyone
    }
    compose_data = {
        "services": {
            "service-a": {
                "build": {"context": "service-a"},
                "environment": ["OTHER_SERVICE=http://service-b:8080"] # but it does
            }
        }
    }
    mock_load_yaml.side_effect = [(True, topology_data), (True, compose_data)]
    mock_validate_topo.return_value = (True, "Valid")

    report = build_report()

    assert report["global_status"] == "ERROR"
    assert report["mic_validation"]["status"] == "VIOLATION"
    assert "Interacción no permitida detectada: 'service-a' -> 'service-b'" in report["mic_validation"]["messages"][0]

@patch("config_agent.config_agent.load_yaml_file")
@patch("config_agent.config_agent.validate_topology")
@patch("config_agent.config_agent.validate_dockerfile_best_practices", return_value=(True, "OK"))
@patch("config_agent.config_agent.check_dependency_consistency", return_value=(False, "Out of sync"))
def test_build_report_dependency_error(mock_check_deps, mock_validate_docker, mock_validate_topo, mock_load_yaml):
    topology_data = {
        "services": {"service-a": {}},
        "mic": {}
    }
    compose_data = {
        "services": {
            "service-a": {"build": {"context": "service-a"}}
        }
    }
    mock_load_yaml.side_effect = [(True, topology_data), (True, compose_data)]
    mock_validate_topo.return_value = (True, "Valid")

    report = build_report()

    assert report["global_status"] == "ERROR"
    assert report["services"]["service-a"]["dependency_status"] == (False, "Out of sync")

@patch("requests.post")
def test_send_report_success(mock_post):
    report_data = {"global_status": "OK"}

    send_report(report_data)

    mock_post.assert_called_once()
    # Check that the first argument's URL contains the endpoint
    assert "http://agent_ai:9000/api/config_report" in mock_post.call_args[0][0]
    # Check that the json kwarg matches the report
    assert mock_post.call_args[1]["json"] == report_data

@patch("requests.post", side_effect=requests.exceptions.RequestException("Connection error"))
@patch("config_agent.config_agent.logger")
def test_send_report_failure(mock_logger, mock_post):
    report_data = {"global_status": "ERROR"}

    send_report(report_data)

    mock_post.assert_called_once()
    mock_logger.error.assert_called_once()
    assert "No se pudo enviar el informe" in mock_logger.error.call_args[0][0]

import time
from typing import Dict, List

import numpy as np
import pytest
import requests

# --- Configuración ---
AGENT_AI_URL = "http://localhost:9000"
HARMONY_CONTROLLER_URL = (
    "http://localhost:7000"  # Not used directly, but good for context
)
ECU_URL = "http://localhost:8000"
TEST_REGION = {"capa": 0}  # Región de interés definida por índices
TARGET_PHASE = 0.0  # Target phase for synchronization
POLLING_TIMEOUT_SECONDS = 30
POLLING_INTERVAL_SECONDS = 1
INITIAL_COHERENCE_THRESHOLD = 0.5
FINAL_COHERENCE_THRESHOLD = 0.95


# --- Funciones de Ayuda (Helpers) ---


def get_ecu_field(ecu_url: str, region: Dict) -> List[complex]:
    """
    Obtiene el campo de una región específica de matriz_ecu.
    Actualmente, solo usa el índice de la capa.
    """
    capa_idx = region.get("capa", 0)
    # Llama al nuevo endpoint específico de la región
    response = requests.get(f"{ecu_url}/api/ecu/field_vector/region/{capa_idx}")
    response.raise_for_status()

    # La respuesta es un diccionario que contiene la clave "field_vector_region"
    target_layer_data = response.json()["field_vector_region"]

    # Aplanar la capa en una lista de números complejos
    field_complex = [
        complex(vector_tuple[0], vector_tuple[1])
        for row in target_layer_data
        for vector_tuple in row
    ]

    return field_complex


def calculate_coherence(field: list[complex]) -> float:
    """Calcula la coherencia de un campo (magnitud del vector promedio)."""
    if not field:
        return 0.0
    # Use numpy to calculate the mean vector and then its magnitude (absolute value)
    return np.abs(np.mean(np.exp(1j * np.angle([c for c in field if c != 0]))))


def get_coherence_from_ecu(ecu_url: str, region: Dict) -> float:
    """Combina la obtención del campo y el cálculo de la coherencia."""
    field = get_ecu_field(ecu_url, region)
    return calculate_coherence(field)


def set_ecu_to_random_phase(ecu_url: str):
    """Pone a matriz_ecu en un estado de baja coherencia."""
    response = requests.post(f"{ecu_url}/debug/set_random_phase")
    response.raise_for_status()
    print("ECU field has been set to a random phase.")


def trigger_synchronization_in_agent(agent_url: str, region: Dict, target_phase: float):
    """Inicia la maniobra de sincronización a través de agent_ai."""
    # El nombre de la región para el comando sigue siendo un string,
    # en este caso identificando la capa.
    region_name = f"capa_{region.get('capa', 0)}"
    command = {"region": region_name, "target_phase": target_phase}
    response = requests.post(f"{agent_url}/commands/synchronize_region", json=command)
    response.raise_for_status()
    print(f"Synchronization command sent to Agent AI for region '{region_name}'.")


# --- Test End-to-End ---


def test_full_phase_synchronization_loop():
    """
    Verifica el flujo completo: agent_ai -> harmony_controller -> matriz_ecu
    para una maniobra de sincronización de fase.
    """
    # --- Fase 1: Setup (Arrange) ---
    print("\n--- Phase 1: ARRANGE ---")
    # Asegurar que el entorno de prueba esté configurado para permitir endpoints de depuración
    # (esto se hace fuera del test, p.ej., con variables de entorno como FLASK_ENV=test)
    set_ecu_to_random_phase(ECU_URL)

    # Darle un momento a la ECU para que se asiente si es necesario
    time.sleep(1)

    initial_coherence = get_coherence_from_ecu(ECU_URL, TEST_REGION)
    print(f"Initial coherence: {initial_coherence:.4f}")
    assert initial_coherence < INITIAL_COHERENCE_THRESHOLD, (
        f"Initial coherence {initial_coherence} was not below the threshold {INITIAL_COHERENCE_THRESHOLD}"
    )

    # --- Fase 2: Acción (Act) ---
    print("\n--- Phase 2: ACT ---")
    trigger_synchronization_in_agent(AGENT_AI_URL, TEST_REGION, TARGET_PHASE)

    # --- Fase 3: Espera Inteligente (Polling) ---
    print("\n--- Phase 3: POLLING ---")
    start_time = time.time()
    final_coherence = 0.0
    last_coherence = initial_coherence

    while time.time() - start_time < POLLING_TIMEOUT_SECONDS:
        current_coherence = get_coherence_from_ecu(ECU_URL, TEST_REGION)
        print(f"Polling... Current coherence is {current_coherence:.4f}")

        if current_coherence > FINAL_COHERENCE_THRESHOLD:
            print("Success! Coherence threshold reached.")
            final_coherence = current_coherence
            break

        # Opcional: verificar si la coherencia está mejorando
        if current_coherence > last_coherence:
            print(
                f"Progress: Coherence increased from {last_coherence:.4f} to {current_coherence:.4f}"
            )
        last_coherence = current_coherence

        time.sleep(POLLING_INTERVAL_SECONDS)
    else:  # This 'else' belongs to the 'while' loop, it executes if the loop finishes without a 'break'
        pytest.fail(
            f"Test timed out after {POLLING_TIMEOUT_SECONDS} seconds. "
            f"Last measured coherence was {last_coherence:.4f}, "
            f"which did not exceed the threshold of {FINAL_COHERENCE_THRESHOLD}."
        )

    # --- Fase 4: Verificación (Assert) ---
    print("\n--- Phase 4: ASSERT ---")
    print(f"Final coherence: {final_coherence:.4f}")
    assert final_coherence > FINAL_COHERENCE_THRESHOLD, (
        f"Final coherence {final_coherence} did not meet the success threshold {FINAL_COHERENCE_THRESHOLD}"
    )

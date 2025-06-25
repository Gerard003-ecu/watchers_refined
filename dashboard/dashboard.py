#!/usr/bin/env python3
"""
Dashboard principal del sistema Watchers.

Módulo principal que implementa una interfaz web interactiva para monitorear y
controlar el ecosistema Watchers. Se integra con los diferentes módulos del
sistema para visualizar datos en tiempo real y enviar comandos de control.

Características principales:
- Visualización de estados de mallas cuánticas
- Control de parámetros de resonancia y frecuencia
- Integración con agent_ai para gestión modular
- Modos de operación simulado y real
- Generación dinámica de controles para módulos registrados
- Recepción de actualizaciones del control (cogniboard)
"""

import logging
import numpy as np
from typing import Dict, List, Union  # Any removed
from functools import wraps

import dash
from dash import dcc, html, Input, Output  # State removed
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import requests
from flask import request, jsonify

# Configuración centralizada
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/dashboard.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("dashboard")

# Constantes configurables
API_TIMEOUT = 5
ENDPOINTS = {
    "agent_status": "http://agent_ai:9000/api/status",
    "agent_command": "http://agent_ai:9000/api/command",
    "wave_malla": "http://watchers_wave:5000/api/malla",
    "wave_acoustic": "http://watchers_wave:5000/api/acoustic",
    "integrador": "http://control:7000/api/integrador",
}

app_state = {"modo": "sim"}  # sim | real


# Helpers de manejo de errores
def api_error_handler(func):
    """Decorador para manejo centralizado de errores de API.

    Este decorador envuelve una función que realiza llamadas a API externas.
    Captura excepciones comunes como `requests.exceptions.RequestException`
    para errores de red y otras excepciones generales, registrando el error
    y devolviendo un diccionario con una clave 'error'.

    Args:
        func (Callable): La función a decorar.

    Returns:
        Callable: La función decorada con manejo de errores.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error de conexión en {func.__name__}: {str(e)}")
            return {"error": f"Error de red: {str(e)}"}
        except Exception as e:
            logger.error(f"Error inesperado en {func.__name__}: {str(e)}")
            return {"error": f"Error interno: {str(e)}"}

    return wrapper


# Funciones de obtención de datos
@api_error_handler
def obtener_datos_reales(endpoint: str) -> Dict:
    """Obtiene datos de un endpoint REST API.

    Realiza una solicitud GET al endpoint especificado y devuelve la respuesta
    JSON decodificada. Utiliza un tiempo de espera definido por `API_TIMEOUT`.
    Levanta una excepción HTTPError para respuestas de error HTTP.

    Args:
        endpoint (str): La URL del endpoint del cual obtener los datos.

    Returns:
        Dict: Un diccionario con los datos obtenidos del endpoint.
              En caso de error manejado por `api_error_handler`,
              retorna un diccionario con una clave 'error'.
    """
    response = requests.get(endpoint, timeout=API_TIMEOUT)
    response.raise_for_status()
    return response.json()


def obtener_estado_malla_sim() -> Dict:
    """Genera datos simulados del estado de las mallas cuánticas.

    Utilizado cuando el dashboard opera en modo simulado ('sim').
    Proporciona una estructura de datos predefinida que imita la respuesta
    esperada del endpoint real de estado de mallas.

    Returns:
        Dict: Un diccionario que contiene datos simulados para 'malla_A',
              'malla_B', el estado del 'resonador' y un 'status'.
              Ejemplo de estructura:
              {
                  "malla_A": [[{"x": 0, "y": 0, "amplitude": 0.8, "phase": 0.0}]],
                  "malla_B": [[{"x": 0, "y": 0, "amplitude": 0.4, "phase": 0.0}]],
                  "resonador": {"lambda_foton": 600},
                  "status": "success",
              }
    """
    return {
        "malla_A": [[{"x": 0, "y": 0, "amplitude": 0.8, "phase": 0.0}]],
        "malla_B": [[{"x": 0, "y": 0, "amplitude": 0.4, "phase": 0.0}]],
        "resonador": {"lambda_foton": 600},
        "status": "success",
    }


def obtener_estado_unificado_sim() -> Dict:
    """Genera datos simulados del estado unificado del sistema.

    Utilizado en modo simulado ('sim'). Proporciona una estructura de datos
    fija que representa el estado unificado, incluyendo la matriz ECU y
    parámetros del control PID.

    Returns:
        Dict: Un diccionario con datos simulados para 'matriz_ecu' y
              'pid_control'.
              Ejemplo de estructura:
              {
                  "matriz_ecu": [0.8, 0.4, 1.0, -0.5, 0.2, 0.3],
                  "pid_control": {
                      "setpoint": 1.1789,
                      "measurement": 1.4,
                      "control_signal": -0.6,
                  },
              }
    """
    return {
        "matriz_ecu": [0.8, 0.4, 1.0, -0.5, 0.2, 0.3],
        "pid_control": {
            "setpoint": 1.1789,
            "measurement": 1.4,
            "control_signal": -0.6,
        },
    }


# Funciones de visualización
def crear_grafico_barras(
    promedios: Dict[str, float], malla_seleccionada: str
) -> go.Figure:
    """Crea un gráfico de barras comparando la amplitud promedio de las mallas.

    Genera una figura de Plotly (gráfico de barras) que muestra la amplitud
    promedio para la Malla A, Malla B o ambas, según la selección.

    Args:
        promedios (Dict[str, float]): Un diccionario con las amplitudes
            promedio. Espera claves como 'A' y 'B'.
            Ejemplo: `{"A": 0.75, "B": 0.60}`.
        malla_seleccionada (str): Indica qué mallas incluir en el gráfico.
            Valores posibles: "malla_A", "malla_B", "ambas".

    Returns:
        go.Figure: Un objeto `plotly.graph_objs.Figure` que representa el
                   gráfico de barras.
    """
    data = []
    if malla_seleccionada in ["malla_A", "ambas"]:
        data.append(
            go.Bar(
                x=["Malla A"],
                y=[promedios.get("A", 0)],
                name="Malla A",
                marker_color="blue",
            )
        )
    if malla_seleccionada in ["malla_B", "ambas"]:
        data.append(
            go.Bar(
                x=["Malla B"],
                y=[promedios.get("B", 0)],
                name="Malla B",
                marker_color="green",
            )
        )
    return go.Figure(
        data=data,
        layout=go.Layout(
            title="Amplitud Promedio por Malla", showlegend=False
        ),
    )


def crear_mapa_calor(malla: List[List[Dict]]) -> go.Figure:
    """Genera un mapa de calor de la distribución de amplitud de una malla.

    Crea una figura de Plotly (mapa de calor) que visualiza los valores de
    amplitud de las celdas en una estructura de malla dada.

    Args:
        malla (List[List[Dict]]): Una lista de listas, donde cada lista interna
            representa una fila de la malla, y cada diccionario representa
            una celda con una clave 'amplitude'.
            Ejemplo: `[[{"amplitude": 0.8}, {"amplitude": 0.9}], ...]`

    Returns:
        go.Figure: Un objeto `plotly.graph_objs.Figure` que representa el
                   mapa de calor. Si la malla está vacía, retorna una
                   figura vacía.
    """
    if not malla:
        return go.Figure()
    z_values = [
        [celda.get("amplitude", 0) for celda in fila] for fila in malla
    ]
    return go.Figure(
        data=go.Heatmap(z=z_values, colorscale="Viridis"),
        layout=go.Layout(title="Distribución de Amplitud - Malla A"),
    )


# Inicializar la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Watchers Dashboard - SonicHarmonizer v2.0"  # Keep as is


# Agregar endpoint para recibir actualizaciones desde cogniboard
# Usamos el servidor Flask subyacente de Dash
@app.server.route("/api/control-update", methods=["POST"])
def control_update():
    """Endpoint para recibir actualizaciones de la señal de control desde Cogniboard.

    Este endpoint Flask escucha peticiones POST en `/api/control-update`.
    Espera un JSON con una clave `control_signal`. Si se recibe
    correctamente, actualiza el estado interno `app_state` con esta señal y
    devuelve un JSON de éxito. Maneja errores si falta `control_signal` o
    si ocurre una excepción durante el procesamiento.

    Returns:
        Tuple[flask.Response, int]: Una tupla conteniendo la respuesta JSON
                                    y el código de estado HTTP.
                                    - Éxito: ({"status": "success",
                                               "control_signal": <valor>}, 200)
                                    - Error (falta `control_signal`):
                                      ({"status": "error",
                                        "mensaje": "Falta 'control_signal'"}, 400)
                                    - Error (otro):
                                      ({"status": "error", "mensaje": <error>}, 500)
    """
    try:
        data = request.get_json()
        control_signal = data.get("control_signal")
        if control_signal is None:
            # Ensuring the jsonify call and its dictionary are formatted
            # to prevent long lines.
            return (
                jsonify({
                    "status": "error",
                    "mensaje": "Falta 'control_signal'"
                }),
                400,
            )
        # Actualizar el estado global del dashboard (o una variable interna)
        # con la señal recibida.
        app_state["control_signal"] = control_signal
        logger.info(
            f"Recibida señal de control desde cogniboard: {control_signal}"
        )
        return (
            jsonify({"status": "success", "control_signal": control_signal}),
            200,
        )
    except Exception as e:
        logger.error(f"Error en /api/control-update: {str(e)}")
        return jsonify({"status": "error", "mensaje": str(e)}), 500


# ---- Fin del nuevo endpoint ----

app.layout = dbc.Container(
    [
        # Sección de control de modo
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            "Panel de Control Watchers",
                            className="text-center mb-4",
                        ),
                        dbc.ButtonGroup(
                            [
                                dbc.Button(
                                    "Modo Simulado",
                                    id="btn-sim",
                                    color="primary",
                                ),
                                dbc.Button(
                                    "Modo Real", id="btn-real", color="danger"
                                ),
                            ],
                            className="mb-3",
                        ),
                    ],
                    width=12,
                )
            ]
        ),
        # Controles principales
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Slider(
                            id="lambda-slider",
                            min=400,
                            max=800,
                            step=10,
                            value=600,
                            marks={i: f"{i} nm" for i in range(400, 801, 50)},
                            tooltip={"placement": "bottom"},
                        ),
                        dcc.Dropdown(
                            id="malla-selector",
                            options=[
                                {"label": "Malla A", "value": "malla_A"},
                                {"label": "Malla B", "value": "malla_B"},
                                {"label": "Ambas", "value": "ambas"},
                            ],
                            value="ambas",
                            clearable=False,
                        ),
                    ],
                    md=6,
                ),
                dbc.Col(
                    [
                        dcc.Slider(
                            id="freq-slider",
                            min=20,
                            max=100,
                            step=5,
                            value=20,
                            marks={i: f"{i} kHz" for i in range(20, 101, 20)},
                            tooltip={"placement": "bottom"},
                        )
                    ],
                    md=6,
                ),
            ],
            className="mb-4",
        ),
        # Sección de visualización
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="amplitude-plot"), md=6),
                dbc.Col(dcc.Graph(id="heatmap-plot"), md=6),
            ]
        ),
        # Actualizaciones automáticas
        dcc.Interval(id="refresh-interval", interval=2000),
        # Área para mostrar la señal de control recibida
        dbc.Row(
            [dbc.Col(html.Div(id="control-signal-panel"), width=12)],
            className="mb-4",
        ),
        # Módulos dinámicos
        html.Div(id="dynamic-modules", className="mt-4"),
    ],
    fluid=True,
)


# Callbacks principales
@app.callback(
    Output("amplitude-plot", "figure"),
    Output("heatmap-plot", "figure"),
    Input("refresh-interval", "n_intervals"),
    Input("malla-selector", "value"),
)
def actualizar_graficos(n: int, malla_seleccionada: str) -> tuple:
    """Actualiza los gráficos principales del dashboard.

    Este callback se activa periódicamente por `refresh-interval` o cuando
    cambia `malla-selector`. Obtiene datos (simulados o reales) del estado
    de las mallas, calcula promedios y genera un gráfico de barras y un
    mapa de calor.

    Args:
        n (int): El número de veces que el intervalo `refresh-interval` se ha
                 disparado. No se usa directamente en la lógica pero es
                 necesario para el callback de intervalo.
        malla_seleccionada (str): El valor actual del selector de mallas
                                  (ej. "malla_A", "malla_B", "ambas").

    Returns:
        tuple: Una tupla conteniendo dos objetos `plotly.graph_objs.Figure`:
               - El gráfico de barras de amplitud promedio.
               - El mapa de calor de la distribución de amplitud de la Malla A.
               En caso de error, retorna dos figuras vacías.
    """
    try:
        if app_state["modo"] == "sim":
            data = obtener_estado_malla_sim()
        else:
            data = obtener_datos_reales(ENDPOINTS["wave_malla"])
        promedios = {
            "A": np.mean(
                [
                    celda["amplitude"]
                    for fila in data["malla_A"]
                    for celda in fila
                ]
            ),
            "B": np.mean(
                [
                    celda["amplitude"]
                    for fila in data["malla_B"]
                    for celda in fila
                ]
            ),
        }
        return (
            crear_grafico_barras(promedios, malla_seleccionada),
            crear_mapa_calor(data["malla_A"]),
        )
    except Exception as e:
        logger.error(f"Error actualizando gráficos: {str(e)}")
        return go.Figure(), go.Figure()


@app.callback(
    Output("control-signal-panel", "children"),
    Input("refresh-interval", "n_intervals"),
)
def actualizar_panel_control(n_intervals: int) -> str:
    """Actualiza el panel que muestra la señal de control recibida.

    Este callback se activa periódicamente por `refresh-interval`.
    Obtiene la última señal de control almacenada en `app_state` (que es
    actualizada por el endpoint `/api/control-update`) y la muestra en un
    componente Div.

    Args:
        n_intervals (int): El número de veces que el intervalo
                           `refresh-interval` se ha disparado. No se usa
                           directamente en la lógica pero es necesario para el
                           callback de intervalo.

    Returns:
        str: Una cadena de texto formateada que muestra la señal de control
             actual, o "No definido" si aún no se ha recibido ninguna.
    """
    control_signal = app_state.get("control_signal", "No definido")
    return f"Señal de control actual: {control_signal}"


@app.callback(
    Output("dynamic-modules", "children"),
    Input("refresh-interval", "n_intervals"),
)
def generar_controles_modulos(n: int) -> Union[List[dbc.Card], str, html.Div]:
    """Genera dinámicamente controles para los módulos 'watcher_tool'.

    Este callback se activa periódicamente. En modo 'real', obtiene la lista
    de módulos desde el `agent_status` y crea un `dbc.Card` con un control
    interactivo (slider, botón, input) para cada módulo de tipo 'watcher_tool'.
    En modo 'sim', devuelve un mensaje indicando que no hay módulos.

    Args:
        n (int): El número de veces que el intervalo `refresh-interval` se ha
                 disparado. No se usa directamente en la lógica pero es
                 necesario para el callback de intervalo.

    Returns:
        Union[List[dbc.Card], str, html.Div]:
            - Una lista de objetos `dbc.Card` si se generan controles.
            - Un string "No hay módulos en modo simulado" si `app_state["modo"]` es "sim".
            - Un `html.Div` con un mensaje de error si falla la obtención o
              procesamiento de módulos.
    """
    if app_state["modo"] == "sim":
        return "No hay módulos en modo simulado"
    try:
        modulos = obtener_datos_reales(ENDPOINTS["agent_status"]).get(
            "modulos", []
        )
        return [
            dbc.Card(
                [
                    dbc.CardHeader(modulo["nombre"]),
                    dbc.CardBody(crear_control(modulo)),
                ]
            )
            for modulo in modulos
            if modulo.get("tipo") == "watcher_tool"
        ]
    except Exception as e:
        logger.error(f"Error generando controles: {str(e)}")
        return html.Div("Error cargando módulos", className="text-danger")


def crear_control(modulo: Dict) -> Union[dcc.Slider, dbc.Button, dcc.Input]:
    """Crea un componente de UI de Dash/Bootstrap basado en la configuración del módulo.

    Esta función de utilidad es llamada por `generar_controles_modulos` para
    crear el control interactivo específico (Slider, Button o Input) para un
    módulo dado. La configuración del control se extrae del diccionario
    del módulo.

    Args:
        modulo (Dict): Un diccionario que representa un módulo. Debe contener
                       `ui_type` (ej. "slider", "button", "input") y una
                       clave `config` con parámetros específicos del tipo de UI
                       (ej. min, max, label, placeholder).
                       Ejemplo:
                       {
                           "nombre": "Control Lambda",
                           "ui_type": "slider",
                           "config": {"min": 400, "max": 800, "default": 600}
                       }

    Returns:
        Union[dcc.Slider, dbc.Button, dcc.Input]: Un componente Dash
            interactivo (Slider, Button o Input) configurado según los
            parámetros del módulo. Por defecto, crea un `dcc.Input` si
            `ui_type` no es reconocido.
    """
    ui_type = modulo.get("ui_type", "input")
    config = modulo.get("config", {})
    if ui_type == "slider":
        return dcc.Slider(
            min=config.get("min", 0),
            max=config.get("max", 100),
            value=config.get("default", 50),
            step=config.get("step", 1),
        )
    elif ui_type == "button":
        return dbc.Button(config.get("label", "Acción"))
    else:
        return dcc.Input(
            placeholder=config.get("placeholder", "Ingrese valor"),
            type=config.get("input_type", "text"),
        )


# Funciones auxiliares para modo real
def obtener_estado_agent() -> Dict:
    """Obtiene el estado actual del servicio agent_ai.

    Realiza una solicitud GET al endpoint de estado de agent_ai
    (`http://agent_ai:9000/api/status`). Maneja errores de solicitud y
    devuelve la respuesta JSON o un diccionario de error.

    Returns:
        Dict: Un diccionario con el estado de agent_ai si la solicitud es
              exitosa. Contiene una clave 'error' con el mensaje de error
              si la solicitud falla.
    """
    try:
        r = requests.get(
            "http://agent_ai:9000/api/status", timeout=API_TIMEOUT
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"Error al obtener estado de agent_ai: {e}")
        return {"error": str(e)}


def enviar_comando_agent(comando: str, valor: Union[str, int, float]) -> Dict:
    """Envía un comando al servicio agent_ai.

    Realiza una solicitud POST al endpoint de comandos de agent_ai
    (`http://agent_ai:9000/api/command`) con el comando y valor especificados
    en el payload JSON. Maneja errores de solicitud y devuelve la respuesta
    JSON o un diccionario de error.

    Args:
        comando (str): El nombre del comando a enviar.
        valor (Union[str, int, float]): El valor asociado al comando.

    Returns:
        Dict: Un diccionario con la respuesta de agent_ai si la solicitud es
              exitosa. Contiene una clave 'error' con el mensaje de error
              si la solicitud falla.
    """
    try:
        payload = {"comando": comando, "valor": valor}
        r = requests.post(
            "http://agent_ai:9000/api/command",
            json=payload,
            timeout=API_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"Error al enviar comando a agent_ai: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=False)

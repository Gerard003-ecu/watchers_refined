# watchers

¡Bienvenido al ecosistema **Watchers**!

Código Claro, Equipos Eficientes. Herramienta para armonizar código y optimizar entornos de desarrollo."

############################################################################
########################Propósito de watchers###############################
############################################################################

Es la armonización del lenguaje de máquina para hacer de la comunicación, de apps y entornos de desarrollo, un canal más intuitivo y objetivo para el ser humanno. Una frontera al alcance de los sentidos. 

Mapa Mental de Comunicación del Ecosistema Watchers

Este mapa describe la arquitectura de comunicación jerárquica y los flujos de datos dentro del ecosistema "watchers", diseñada para ser coherente, escalable y objetiva.

Analogía Central: Mente, Cuerpo y Entorno

    Capa Estratégica (agent_ai): La Mente. Define los objetivos y la intención a largo plazo. No se ocupa de la mecánica, sino del "porqué".

    Capa Táctica (harmony_controller): El Sistema Nervioso Autónomo. Traduce la intención en comandos concretos y mantiene el equilibrio (homeostasis) del sistema mediante bucles de control.

    Capa Física (matriz_ecu, malla_watcher): El Cuerpo y su Entorno. El mundo físico simulado donde las acciones tienen lugar y generan retroalimentación.

1. ◈ Capa Estratégica (La Mente)

    agent_ai (El Orquestador Central)

        Rol Principal: Cerebro estratégico del ecosistema. Define el objetivo de armonía (target_setpoint_vector) basado en una estrategia global (ej: "rendimiento", "estabilidad"), en la composición actual del sistema y en señales externas.

        Comunicaciones Clave:

            [SALIDA] ➔ harmony_controller

                Acción: Define el Objetivo Estratégico.

                Endpoint: POST /api/harmony/setpoint

                Payload: { "setpoint_vector": [x, y, ...] }

                Propósito: Comunicar el estado deseado que la capa táctica debe alcanzar.

            [SALIDA] ➔ harmony_controller

                Acción: Notifica sobre un watcher_tool saludable y listo para ser controlado.

                Endpoint: POST /api/harmony/register_tool

                Payload: { "nombre", "url", "aporta_a", "naturaleza_auxiliar" }

                Propósito: Integrar dinámicamente nuevas capacidades en el bucle de control táctico.

            [ENTRADA] ↞ Cualquier Watcher Tool

                Acción: Registro inicial de un módulo en el ecosistema.

                Endpoint: POST /api/register

                Payload: { "nombre", "url", "url_salud", "tipo", "aporta_a", ... }

                Propósito: Punto de entrada único y universal para todos los módulos, permitiendo el descubrimiento y la validación de salud.

            [ENTRADA] ↞ harmony_controller

                Acción: Monitoreo del estado táctico.

                Endpoint: GET /api/harmony/state

                Payload: Estado completo de harmony_controller (PID, last_measurement, etc.).

                Propósito: Recolectar información para la toma de decisiones estratégicas.

2. ◈ Capa Táctica (El Sistema Nervioso)

    harmony_controller (El Controlador de Armonía)

        Rol Principal: Ingeniero de control. Implementa un bucle de control (PID) para llevar el estado medido del sistema hacia el setpoint definido por agent_ai. Traduce el "qué" estratégico en el "cómo" táctico.

        Comunicaciones Clave:

            [SALIDA] ➔ malla_watcher (y otros watcher_tools)

                Acción: Envía una Señal de Control Táctico.

                Endpoint: POST /api/control

                Payload: { "control_signal": valor_pid_ponderado }

                Propósito: Ajustar los parámetros internos de los watcher_tools para influir en la física del sistema. La señal se adapta según la "naturaleza" del tool (potenciador, reductor, modulador).

            [ENTRADA] ↞ matriz_ecu

                Acción: Mide el Estado del Sistema.

                Endpoint: GET /api/ecu

                Payload: { "estado_campo_unificado": [...] }

                Propósito: Obtener la variable de proceso (current_measurement) para el cálculo del error en el controlador PID. Es el "sentido" principal del sistema.

3. ◈ Capa Física (El Entorno y los Actores)

Este es el nivel donde las "leyes de la física" del ecosistema operan y donde se produce la interacción más fundamental.

    matriz_ecu (El Entorno - Campo Toroidal)

        Rol Principal: Simula el espacio, el "campo unificado" donde existen e interactúan los watchers. Tiene su propia dinámica interna.

        Comunicaciones Clave:

            [RESPUESTA A PETICIÓN] ➔ malla_watcher

                Acción: Provee el Campo Vectorial completo.

                Endpoint: GET /api/ecu/field_vector

                Payload: { "field_vector": [...] }

                Propósito: Permitir que malla_watcher "sienta" la estructura detallada del campo en cada punto para modular su propia dinámica.

            [ENTRADA] ↞ malla_watcher

                Acción: Recibe una Influencia Inducida.

                Endpoint: POST /api/ecu/influence

                Payload: { "vector": [dPhi/dt, 0.0], ... }

                Propósito: Simular la "Ley de Inducción de Faraday". La malla influye de vuelta en el campo que la atraviesa, creando un bucle de retroalimentación simbiótico.

    malla_watcher (El Actor Principal - Malla de Grafeno)

        Rol Principal: Un actor que co-evoluciona con el entorno. Su estado interno (osciladores) es modulado por el campo de matriz_ecu, y a su vez, influye en ese mismo campo.

        Comunicaciones Clave:

            [SALIDA] ➔ matriz_ecu

                Acción: Obtiene el Entorno Vectorial.

                Endpoint: GET /api/ecu/field_vector

                Propósito: "Leer" el campo vectorial de matriz_ecu para modular el acoplamiento de sus osciladores internos.

            [ENTRADA] ↞ harmony_controller

                Acción: Recibe la Señal de Control Táctico.

                Endpoint: POST /api/control

                Payload: { "control_signal": valor }

                Propósito: Ajustar sus parámetros base (amortiguación D y acoplamiento C) según las directivas del controlador táctico.

4. ◈ Módulos Auxiliares (Plug-and-Play)

    watcher_tool_auxiliar (Ej: benzwatcher, watcher_focus)

        Rol Principal: Especialistas que se acoplan a los componentes centrales (matriz_ecu o malla_watcher) para ampliar o refinar sus capacidades.

        Flujo de Integración Típico:

            Registro: Se registra en agent_ai al iniciar.

            Notificación: agent_ai valida su salud y lo notifica a harmony_controller, indicando a quién apoya (aporta_a) y cuál es su naturaleza_auxiliar.

            Control: harmony_controller comienza a enviarle señales de control.

            Acción: El watcher_tool utiliza la señal de control para ejecutar su lógica específica, interactuando directamente (vía API) con el componente central al que está asociado.

## Propósito

El objetivo principal de **Watchers** es facilitar la integración y el monitoreo de sistemas complejos. Entre sus funciones destacan:
- **Orquestación y Control:** Agent AI se encarga de coordinar y ajustar dinámicamente los módulos, garantizando la coherencia del sistema.
- **Validación y Configuración:** Config Agent y Watcher Security supervisan la integridad de la infraestructura, validan configuraciones y ajustan dependencias para mantener un entorno óptimo.
- **Interfaz y Supervisión:** El Dashboard (junto con el cogniboard) ofrece una interfaz gráfica intuitiva que permite a los usuarios monitorear el estado global y realizar ajustes cuando sea necesario.
- **Extensibilidad:** La arquitectura "plug and play" permite integrar nuevos módulos (watcher_tool) como herramientas especializadas (por ejemplo, un editor de texto inteligente o una hoja de cálculo inteligente) sin alterar la lógica central.

## Propuesta de Valor

- **Automatización:** Minimiza la intervención del usuario mediante procesos automáticos que validan y corrigen configuraciones.
- **Seguridad y Robustez:** Gracias a un sistema inmunológico adaptativo y un volumen de control (cogniboard), se garantiza una operación estable y segura.
- **Escalabilidad:** La arquitectura modular permite agregar nuevos módulos y funcionalidades sin comprometer la integridad del sistema.
- **Lenguaje de Marca:** Inspirado en estructuras hexagonales (benceno y grafeno), Watchers ofrece una representación visual y conceptual única para el control de flujos y estados.

## Arquitectura General

El ecosistema **Watchers** está compuesto por varios módulos interconectados:

- **Agent AI:** Núcleo operativo que registra módulos, distribuye comandos y aplica mecanismos de validación.
- **Config Agent:** Realiza la validación de la infraestructura (Dockerfiles, docker-compose, dependencias) y envía señales correctivas.
- **Watcher Security:** Funciona como un sistema inmunológico adaptativo, verificando y ajustando configuraciones y dependencias.
- **Cogniboard (Control Volume):** Supervisa y aplica control PID para mantener la estabilidad y el equilibrio del sistema.
- **Dashboard:** Proporciona una interfaz gráfica para monitoreo y control, mostrando el estado global y permitiendo la interacción con el ecosistema.
- **Módulos Específicos:** Incluyen watchers_wave, watcher_focus, optical_controller, ECU, etc., que aportan funcionalidades especializadas y se integran de manera plug and play.
- **BenzWatcher:** Simula la reacción catalítica inspirada en el benceno para ajustar y modular las señales de control.

Para más detalles sobre la arquitectura, consulta el documento [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Empezar

1. Clona el repositorio.
2. Revisa el archivo `requirements.in` y compílalo con:
   ```bash
   pip-compile requirements.in

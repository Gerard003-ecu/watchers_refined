# watchers

¡Bienvenido al ecosistema **Watchers**!

Código Claro, Equipos Eficientes. Herramienta para armonizar código y optimizar entornos de desarrollo."

############################################################################
########################Propósito de watchers###############################
############################################################################

Es la armonización del lenguaje de máquina para hacer de la comunicación, de apps y entornos de desarrollo, un canal más intuitivo y objetivo para el ser humanno. Una frontera al alcance de los sentidos. 
El ecosistema "watchers" se compone de 4 atributos: a. Podman para la gestión de contenedores. b. config_agent agente proactivo de configuración. c. agent_ai es la lógica central (orquestación, validación en tiempo de ejecución, distribución de comandos) y d. watcher_security el volumen de control de seguridad. Los módulos tienen el atributo "plug and play" denominados "watcher_tool" que pueden ser un volumen de control (un conjunto ordenado de "watcher_tool" con instrucciones específicas que robustecen la lógica a favor del sistema o potenciar y/o amplificar un módulo ya existente) o una herramienta específica como "watcher_tool_text" (editor de texto inteligente que comunica los estados con el resto de módulos de su misma clase y/o tipo como "watcher_tool_calc", en general, cualquier script de la forma "watcher_tool_nombre"). La estrategia de comunicación del ecosistema "watchers" se basa en una estructura hexagonal (lenguaje de marca) que esta presente en los diferentes módulos "watcher_tool" del ecosistema. Por ejemplo, en "watchers_modules/watchers_wave/malla_watcher". El script "malla_watcher" define la estructura de la malla hexagonal (inspirada en grafeno) y los módulos que modelan la transmisión de la señal en el sistema:

    PhosWave: el resonador de onda variable, con el atributo de un fotón, que ahora incorpora un campo escalar Q para modular la transmisión.

    Electron: el estabilizador que reduce la amplitud, simulando la acción de
    partículas cargadas que neutralizan la energía excesiva.
    La malla se utiliza para representar el flujo de "energía" del sistema.

En el módulo (watcher_tool) "benzwatcher", es un script que simula una reacción catalítica inspirada en la estructura hexagonal del benceno. Esta clase procesa una señal de entrada y ajusta un valor base en función de una transformación no lineal, que modela la cinética de la reacción catalítica. De esta manera, el ecosistema "watchers" adquiere un atributo de identidad que refuerza y comunica la propuesta de valor.De esta manera se garantiza la modularidad, robustez, escalabilidad y objetividad del ecosistema "watchers". De este modo se puede ofrecer soluciones integrales a los segmentos de mercado de las industrias 4.0, IoT hogar y corporativo.

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

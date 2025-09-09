# Watchers: Un Ecosistema de IA para la Orquestación Armónica de Sistemas Complejos
## "Inspirado en los principios de la Cimática y la Topología Algebraica para construir sistemas resilientes, autoconscientes y eficientes."

## Filosofía Central

"Watchers" no es solo un conjunto de microservicios; es una plataforma para construir "cerebros energéticos" y otros sistemas de control autónomos. Nuestra filosofía se basa en dos metáforas poderosas:

*   **Cimática (La Física de la Vibración):** Entendemos la dinámica del sistema como un campo vibratorio. El objetivo principal es encontrar y mantener "modos normales" —estados de operación armónicos y eficientes— y evitar transiciones caóticas. Las intervenciones en el sistema son análogas a aplicar frecuencias de resonancia para guiar el sistema hacia estados deseados.

*   **Topología Algebraica (La Matemática de la Estructura):** La arquitectura del sistema es "autoconsciente". Utilizamos conceptos topológicos para que el ecosistema comprenda su propia estructura de interacciones, garantizando la coherencia y la resiliencia. La Matriz de Interacción Central (MIC) es la manifestación de esta conciencia arquitectónica.

## Componentes del Ecosistema

Cada microservicio cumple un rol conceptual dentro de esta filosofía, actuando en concierto para lograr la armonía del sistema.

*   **matriz_ecu: El Medio Vibratorio.**
    Un simulador de campo cimático que modela el entorno físico. Su estado evoluciona según la Ecuación de Onda, manifestando patrones complejos (modos normales) en respuesta a estímulos.

*   **harmony_controller: El Resonador Táctico.**
    Un controlador que actúa como una fuente de excitación, sintonizando sus intervenciones para guiar al `matriz_ecu` hacia frecuencias de resonancia deseadas (setpoints de armonía).

*   **agent_ai: El Orquestador Estratégico.**
    Observa la topología del sistema y la "forma" de la respuesta cimática para tomar decisiones de alto nivel, como la sintonización automática de `harmony_controller`.

*   **config_agent: El Arquitecto Topológico.**
    Un "functor" que valida la estructura del ecosistema, construyendo la Matriz de Interacción Central (MIC) y asegurando la coherencia arquitectónica.

*   **watcher_security: El Sistema Inmunológico.**
    Un sistema de defensa en tiempo de ejecución que utiliza principios de termodinámica (Exergía) y álgebra lineal (LIA) para mantener la homeostasis del sistema.

## Cómo Funciona: Un Ejemplo de Flujo

El flujo de un test E2E, descrito en nuestra nueva terminología, ilustra la orquestación:

1.  **Validación Topológica:** `config_agent` valida la topología del sistema y construye la MIC.
2.  **Definición Estratégica:** `agent_ai` establece un "modo normal" (un patrón de vibración estable) como objetivo deseado.
3.  **Excitación Armónica:** `harmony_controller` aplica "vibraciones" (influencias controladas) al `matriz_ecu` para guiarlo hacia el objetivo.
4.  **Convergencia:** El sistema converge a un patrón estable, alcanzando un estado de alta coherencia y eficiencia operativa.

## Instalación y Uso

Para poner en marcha el ecosistema de Watchers, necesitarás el siguiente software:

-   **Podman & podman-compose:** Para la gestión de contenedores.
-   **Python 3.10+:** El lenguaje principal del proyecto.
-   **uv:** Para la gestión de dependencias y entornos virtuales.

### Configuración del Entorno de Desarrollo Local

Este proyecto utiliza `uv` para una gestión de paquetes y entornos ultrarrápida.

1.  **Instala `uv`:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Crea el entorno virtual e instala las dependencias (un solo comando):**
    ```bash
    uv venv && uv pip sync requirements-dev.txt
    ```

3.  **Activa el entorno:**
    ```bash
    source .venv/bin/activate
    ```

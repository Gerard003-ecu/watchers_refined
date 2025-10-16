# Descripción de Agentes y Herramientas del Ecosistema Watchers

Este documento describe los agentes y herramientas clave que componen el ecosistema "watchers". Sirve como una guía para que Jules entienda el propósito y la forma de interactuar con cada componente.

## Agentes Centrales

### 1. `agent_ai` (Agente Estratégico)
- **Qué hace:** Es el cerebro principal del ecosistema. Orquesta las tareas de alto nivel, monitorea el estado general y toma decisiones estratégicas.
- **Cómo interactuar:** A través de su API REST en el puerto `9000`. Los comandos principales se envían al endpoint `/api/commands/...`.

### 2. `config_agent` (Agente Arquitecto)
- **Qué hace:** Valida la arquitectura del sistema antes del despliegue. Construye y valida la Matriz de Interacción Central (MIC) y la taxonomía de servicios.
- **Cómo interactuar:** Se ejecuta como un script (`config_agent.py`) durante la fase de CI/CD. Reporta su análisis al endpoint `/api/config_report` de `agent_ai`.

### 3. `watcher_security` (Agente Inmunológico)
- **Qué hace:** Monitorea la salud del sistema en tiempo de ejecución usando principios de exergía y Lógica Integrada Adaptativa (LIA). Detecta anomalías y puede ejecutar protocolos de seguridad como la cuarentena o el rebalanceo de recursos.
- **Cómo interactuar:** A través de su API REST. `agent_ai` le envía "señales vitales" del sistema para su análisis.

## Herramientas Tácticas

### 1. `harmony_controller` (Regulador Táctico)
- **Qué hace:** Ejecuta bucles de control de bajo nivel (e.g., PID) para estabilizar componentes del sistema, como `matriz_ecu`.
- **Cómo interactuar:** Recibe tareas de `agent_ai` a través de su API en el puerto `7000`.

### 2. `matriz_ecu` (Entorno Simulado)
- **Qué hace:** Simula el entorno físico (campo cuántico/energético) sobre el cual operan los demás agentes.
- **Cómo interactuar:** Expone una API en el puerto `8000` para leer su estado (`/api/ecu/field_vector`) y recibir influencias (`/api/ecu/influence`).
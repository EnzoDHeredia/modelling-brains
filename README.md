
# Controlador Webots con asistente LLM
El siguiente codigo contiene el controlador de un robot simulado (modelo Pioneer 3AT) que navega de manera aletoria en una arena con obstaculos. El robot esquiva los obstaculos y las paredes siguiendo un comportamiento estilo Braitenberg basado en LiDAR. Las lecturas de dicho sensor son enviadas a un agente LLM que indica hacia donde considera mas apropiado doblar.

## Archivos
-  **Braitenberg_chatgpt.py:** Contiene la logica del controlador Webot completa, desde la inicializacion de sensores, hasta el movimiento del robot.
-  **LLM_decider.py:** Contiene la logica de conexion con Groq (agente de LLM). 
-  **api_keys.txt:** Archivo donde se debe reemplazar la API_KEY generada en Groq.


## Objetivo
Desarrollar dos comportamientos extras para los estados 1 y 2, diferentes al Braitenberg ya implementado (estado 0). Establecer los momentos en los cuales el sistema deba tomar una decision (por ej. el robot no detecta paredes). Entregar informacion a Groq sobre el estado del sistema para que decida cual sera el nuevo comportamiento a ejecutar.

  

## Objetivo
Para ejecutar el algoritmo se necesita:
- Webot version R2023b o superior
- Python 3.7 o superior

**Librerias de Python:**
- Numpy v1.21.6
- groq v0.11.0
- Opcional: matplotlib para graficar

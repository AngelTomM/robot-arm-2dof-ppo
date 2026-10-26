# Comparativa de Capacidad de Red Neuronal (`net_arch`)

Comparación del modelo base `PPO_7` frente a variantes con diferente tamaño de red, manteniendo idénticos los hiperparámetros de optimización y el entorno LQR:

* **PPO_2.1:** `[32, 32]` (capacidad reducida / subdimensionada).
* **PPO_7:** `[256, 256, 128]` (arquitectura nominal calibrada).
* **PPO_2.2:** `[1024, 1024, 512, 256]` (capacidad extendida / sobredimensionada).

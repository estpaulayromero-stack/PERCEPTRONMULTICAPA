# Perceptrón multicapa: XOR

Aplicación didáctica en Python para visualizar el ejemplo e1 de clase y entrenar una red multicapa con la tabla XOR completa.

## Requisitos y ejecución

- Python 3 con Tkinter.
- Pillow para mostrar las portadas (`python -m pip install pillow`).
- `INICIO_PM.png` y `PRESENTACION_PM.png` en la misma carpeta que el script.

Desde esta carpeta, ejecutar:

```powershell
python perceptron_multicapa.py
```

## Uso

- **Paso e1** aplica una actualización al patrón (0, 1 → 1), para contrastar la retropropagación con las diapositivas.
- **Época XOR**, **Auto XOR** e **Ir** entrenan los cuatro patrones en el orden e1, e2, e3, e4.
- La red usa dos entradas, dos neuronas ocultas y una neurona de salida con activación sigmoide. Los parámetros comienzan con los pesos y sesgos del ejercicio.
- El entrenamiento completo acumula los gradientes de los cuatro patrones y actualiza los pesos y sesgos una vez por época. El MSE se calcula sobre los cuatro patrones y el entrenamiento se detiene cuando MSE ≤ 0,01 o al alcanzar 100 000 épocas.
- La interfaz muestra el MSE por época, la tabla XOR con las salidas y clases predichas, los parámetros actuales, y las épocas e iteraciones (patrones procesados).

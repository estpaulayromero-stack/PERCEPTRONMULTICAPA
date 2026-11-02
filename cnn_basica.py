import numpy as np
import tkinter as tk


# Imagen sintética de ejemplo para ilustrar una red convolucional.
# No es un dataset real; solo sirve para mostrar cómo funciona la convolución.
# El objetivo es visualizar la operación de kernel + feature map + ReLU + pooling.


def crear_imagen_sintetica():
    imagen = np.array([
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
    ], dtype=float)
    return imagen


def kernel_bordes_verticales():
    # Filtro clásico para detectar bordes verticales.
    return np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ], dtype=float)


def crear_imagen_con_los_mismos_datos_del_perceptron():
    # Este bloque usa los mismos pesos/valores del ejercicio del perceptrón multicapa,
    # pero reorganizados como una imagen pequeña para mostrar la idea de la CNN.
    return np.array([
        [0.0, 1.0, 0.0],
        [0.3, 0.2, 0.6],
        [-0.4, 0.5, 0.7],
    ], dtype=float)


def kernel_con_los_mismos_datos_del_perceptron():
    # Mismo W1 del perceptrón multicapa
    return np.array([
        [0.3, 0.2],
        [0.6, -0.4],
    ], dtype=float)


def convolucion_2d(imagen, kernel):
    h, w = imagen.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    salida = np.zeros((h, w), dtype=float)

    for i in range(h):
        for j in range(w):
            suma = 0.0
            for m in range(kh):
                for n in range(kw):
                    ii = i + m - pad_h
                    jj = j + n - pad_w
                    if 0 <= ii < h and 0 <= jj < w:
                        suma += imagen[ii, jj] * kernel[m, n]
            salida[i, j] = suma

    return salida


def relu(x):
    return np.maximum(x, 0)


def max_pooling(x, tamano=2):
    h, w = x.shape
    out_h = (h + tamano - 1) // tamano
    out_w = (w + tamano - 1) // tamano
    salida = np.zeros((out_h, out_w), dtype=float)

    for i in range(out_h):
        for j in range(out_w):
            i0 = i * tamano
            j0 = j * tamano
            bloque = x[i0:i0 + tamano, j0:j0 + tamano]
            salida[i, j] = np.max(bloque)

    return salida


def mostrar_matriz(nombre, matriz):
    print(f"\n{nombre}:")
    print(matriz)


class CNNVisualApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CNN básica: kernel + convolución + ReLU + pooling")
        self.root.geometry("1200x700")
        self.root.configure(bg="#0b1220")

        self.imagen = crear_imagen_con_los_mismos_datos_del_perceptron()
        self.kernel = kernel_con_los_mismos_datos_del_perceptron()
        self.feature_map = convolucion_2d(self.imagen, self.kernel)
        self.feature_map_relu = relu(self.feature_map)
        self.pooled = max_pooling(self.feature_map_relu, tamano=2)
        self.state = "convolucion"

        self.build_ui()
        self.update_view()
        self.root.mainloop()

    def build_ui(self):
        top = tk.Frame(self.root, bg="#0b1220")
        top.pack(fill="x", pady=10)

        tk.Label(top, text="CNN básica con los mismos datos del perceptrón multicapa",
                 bg="#0b1220", fg="#e2e8f0", font=("Helvetica", 18, "bold")).pack()

        btn_frame = tk.Frame(self.root, bg="#0b1220")
        btn_frame.pack(fill="x", pady=(0, 12))

        states = [
            ("Entrada", "entrada"),
            ("Kernel", "kernel"),
            ("Convolución", "convolucion"),
            ("ReLU", "relu"),
            ("Pooling", "pooling"),
        ]

        self.buttons = {}
        for text, value in states:
            b = tk.Button(btn_frame, text=text, command=lambda v=value: self.change_state(v),
                          bg="#38bdf8", fg="#0b1220", font=("Helvetica", 11, "bold"),
                          width=12, relief="flat")
            b.pack(side="left", padx=8)
            self.buttons[value] = b

        self.info = tk.Label(self.root, text="", bg="#0b1220", fg="#facc15",
                             font=("Helvetica", 11, "bold"), justify="left")
        self.info.pack(anchor="w", padx=20, pady=(0, 8))

        body = tk.Frame(self.root, bg="#0b1220")
        body.pack(fill="both", expand=True, padx=20, pady=10)

        self.canvas_entrada = tk.Canvas(body, width=280, height=220, bg="#111827", highlightthickness=0)
        self.canvas_entrada.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.canvas_kernel = tk.Canvas(body, width=240, height=200, bg="#111827", highlightthickness=0)
        self.canvas_kernel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.canvas_result = tk.Canvas(body, width=420, height=260, bg="#111827", highlightthickness=0)
        self.canvas_result.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_columnconfigure(2, weight=1)

    def change_state(self, state):
        self.state = state
        self.update_view()

    def render_matrix(self, canvas, matrix, title):
        canvas.delete("all")
        rows, cols = matrix.shape
        w = int(canvas.cget("width"))
        h = int(canvas.cget("height"))

        pad_x = 18
        pad_y = 30
        cell = min((w - 2 * pad_x) / cols, (h - 2 * pad_y) / rows)
        grid_w = cols * cell
        grid_h = rows * cell
        start_x = (w - grid_w) / 2
        start_y = (h - grid_h) / 2 + 10

        canvas.create_text(w / 2, 18, text=title, fill="#e2e8f0", font=("Helvetica", 12, "bold"))
        for i in range(rows):
            for j in range(cols):
                x0 = start_x + j * cell
                y0 = start_y + i * cell
                x1 = x0 + cell
                y1 = y0 + cell
                val = float(matrix[i, j])
                color = self._color_for_value(val)
                canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="#38bdf8", width=2)
                canvas.create_text(x0 + cell / 2, y0 + cell / 2, text=f"{val:.2f}", fill="#0b1220",
                                  font=("Helvetica", 8 if max(rows, cols) <= 5 else 7, "bold"))

    def _color_for_value(self, value):
        if value > 0:
            return "#86efac"
        if value < 0:
            return "#fca5a5"
        return "#dbeafe"

    def update_view(self):
        self.render_matrix(self.canvas_entrada, self.imagen, "Entrada")
        self.render_matrix(self.canvas_kernel, self.kernel, "Kernel")

        if self.state == "entrada":
            matriz = self.imagen
            info = "Entrada: la red recibe una matriz pequeña con los mismos valores del ejercicio."
        elif self.state == "kernel":
            matriz = self.kernel
            info = "Kernel: filtro que recorre la imagen para detectar patrones locales."
        elif self.state == "convolucion":
            matriz = self.feature_map
            info = "Convolución: suma ponderada de la entrada con el kernel (feature map)."
        elif self.state == "relu":
            matriz = self.feature_map_relu
            info = "ReLU: elimina valores negativos y mantiene solo valores útiles."
        elif self.state == "pooling":
            matriz = self.pooled
            info = "Pooling: reduce la dimensión conservando solo lo más importante."
        else:
            matriz = self.feature_map
            info = "Estado de la CNN."

        self.render_matrix(self.canvas_result, matriz, "Resultado")
        self.info.config(text=info)


if __name__ == "__main__":
    print("Ejemplo 1: convolución con un filtro clásico")
    print("========================================")
    imagen = crear_imagen_sintetica()
    kernel = kernel_bordes_verticales()
    mostrar_matriz("Imagen de entrada", imagen)
    mostrar_matriz("Kernel (filtro)", kernel)

    feature_map = convolucion_2d(imagen, kernel)
    mostrar_matriz("Feature map después de convolución", feature_map)

    feature_map_relu = relu(feature_map)
    mostrar_matriz("Feature map después de ReLU", feature_map_relu)

    pooled = max_pooling(feature_map_relu, tamano=2)
    mostrar_matriz("Resultado después de Max Pooling", pooled)

    print("\nEjemplo 2: usando los mismos datos del perceptrón multicapa")
    print("========================================================")
    imagen_ejercicio = crear_imagen_con_los_mismos_datos_del_perceptron()
    kernel_ejercicio = kernel_con_los_mismos_datos_del_perceptron()
    print("W1 = [[0.3, 0.2], [0.6, -0.4]]")
    print("W2 = [0.5, 0.7]")
    print("alpha = 0.25")
    mostrar_matriz("Imagen construida con los mismos datos", imagen_ejercicio)
    mostrar_matriz("Kernel construido con W1", kernel_ejercicio)

    feature_map_ejercicio = convolucion_2d(imagen_ejercicio, kernel_ejercicio)
    mostrar_matriz("Feature map del ejercicio", feature_map_ejercicio)

    feature_map_relu_ejercicio = relu(feature_map_ejercicio)
    mostrar_matriz("Feature map después de ReLU", feature_map_relu_ejercicio)

    pooled_ejercicio = max_pooling(feature_map_relu_ejercicio, tamano=2)
    mostrar_matriz("Resultado final del ejercicio con pooling", pooled_ejercicio)

    print("\nInterpretación:")
    print("- El kernel recorre la imagen y calcula una suma ponderada.")
    print("- El feature map resalta patrones relevantes, como bordes o pesos locales.")
    print("- ReLU elimina valores negativos.")
    print("- Pooling reduce la dimensión y mantiene la parte más importante.")
    print("- Esta es la base de una red neuronal convolucional (CNN).")
    print("- El archivo principal sigue siendo el perceptrón multicapa, y este es solo el complemento con kernel.")

    print("\nAbriendo la interfaz gráfica de la CNN...")
    CNNVisualApp()

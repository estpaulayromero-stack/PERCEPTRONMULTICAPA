##Hecho por:Paula,Danna y Yessica
##----------------PERCEPTRON MULTICAPA-----------------------

"""Perceptrón multicapa XOR: demostración de e1 y entrenamiento del conjunto completo.
Requisitos: Python con tkinter (incluido) y Pillow:  pip install pillow
Pon INICIO_PM.png y PRESENTACION_PM.png en la MISMA carpeta que este archivo.
Ejecutar:  python perceptron_multicapa.py"""
import math, os, tkinter as tk

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None

BG, PANEL, TXT, MUT = "#0b1220", "#17233a", "#e2e8f0", "#94a3b8"
ACC, GOOD, BAD, WARN = "#38bdf8", "#22c55e", "#ef4444", "#facc15"
CARPETA = os.path.dirname(os.path.abspath(__file__))
X, T = (0, 1), 1                                   # patrón e1: x1=0, x2=1, objetivo 1
W1_0, W2_0, ALPHA_0 = [[0.3, 0.2], [0.6, -0.4]], [0.5, 0.7], 0.25   # datos del ejercicio
B1_0, B2_0 = [0.1, 0.1], 0.1                         # sesgos de capa oculta y salida
DATOS_XOR = ((0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 0))
META_MSE = 0.01
MAX_EPOCAS = 100_000


def g(x):
    return 1 / (1 + math.exp(-max(-60, min(60, x))))


class MLP:
    def __init__(self, W1, W2, alpha, B1, B2):
        self.W1, self.W2, self.alpha = [r[:] for r in W1], W2[:], alpha
        self.B1, self.B2 = B1[:], B2

    def forward(self, x=X):
        h = [g(x[0] * self.W1[0][j] + x[1] * self.W1[1][j] + self.B1[j]) for j in range(2)]
        return h, g(h[0] * self.W2[0] + h[1] * self.W2[1] + self.B2)

    def salida(self, x=X):
        return self.forward(x)[1]

    def mse(self):
        return sum((target - self.salida((x1, x2))) ** 2 for x1, x2, target in DATOS_XOR) / len(DATOS_XOR)

    def paso(self, x=X, target=T):
        """Un paso de retropropagación para un patrón, como el ejemplo e1."""
        h, o = self.forward(x)
        do = o * (1 - o) * (target - o)
        dh = [h[j] * (1 - h[j]) * self.W2[j] * do for j in range(2)]
        for j in range(2):
            self.W2[j] += self.alpha * h[j] * do
            self.B1[j] += self.alpha * dh[j]
            for i in range(2):
                self.W1[i][j] += self.alpha * x[i] * dh[j]
        self.B2 += self.alpha * do

    def entrenar_epoca(self):
        """Acumula gradientes de todos los patrones y actualiza una vez por época."""
        grad_w1 = [[0.0, 0.0], [0.0, 0.0]]
        grad_w2, grad_b1, grad_b2 = [0.0, 0.0], [0.0, 0.0], 0.0
        for x1, x2, target in DATOS_XOR:
            x = (x1, x2)
            h, o = self.forward(x)
            do = o * (1 - o) * (target - o)
            dh = [h[j] * (1 - h[j]) * self.W2[j] * do for j in range(2)]
            for j in range(2):
                grad_w2[j] += h[j] * do
                grad_b1[j] += dh[j]
                for i in range(2):
                    grad_w1[i][j] += x[i] * dh[j]
            grad_b2 += do

        n = len(DATOS_XOR)
        for j in range(2):
            self.W2[j] += self.alpha * grad_w2[j] / n
            self.B1[j] += self.alpha * grad_b1[j] / n
            for i in range(2):
                self.W1[i][j] += self.alpha * grad_w1[i][j] / n
        self.B2 += self.alpha * grad_b2 / n
        return self.mse()

def mezcla(v):
    a, b = (30, 41, 59), (56, 189, 248)
    return "#%02x%02x%02x" % tuple(int(a[k] + (b[k] - a[k]) * v) for k in range(3))


def dibuja_red(c, m, x=X, objetivo=T):
    c.delete("all")
    h, o = m.forward(x)
    I, H, O = [(100, 120), (100, 320)], [(330, 120), (330, 320)], (560, 220)
    for txt, px in (("ENTRADA", 100), ("CAPA OCULTA", 330), ("SALIDA", 560)):
        c.create_text(px, 25, text=txt, fill=MUT, font=("Helvetica", 10, "bold"))

    def arista(p, q, w, f):
        c.create_line(*p, *q, fill=GOOD if w >= 0 else BAD, width=1 + min(7, abs(w) * 1.3))
        tid = c.create_text(p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f, text=f"{w:.3f}",
                            fill=TXT, font=("Consolas", 10, "bold"))
        c.tag_lower(c.create_rectangle(c.bbox(tid), fill=BG, outline=""), tid)

    for i in range(2):
        for j in range(2):
            arista(I[i], H[j], m.W1[i][j], 0.3)
    for j in range(2):
        arista(H[j], O, m.W2[j], 0.5)

    def neurona(p, v, nombre, texto, sesgo=None):
        c.create_oval(p[0] - 38, p[1] - 38, p[0] + 38, p[1] + 38, fill=mezcla(v), outline=ACC, width=2)
        c.create_text(*p, text=texto, fill="white", font=("Helvetica", 12, "bold"))
        c.create_text(p[0], p[1] + 52, text=nombre, fill=MUT, font=("Helvetica", 10))
        if sesgo is not None:
            c.create_text(p[0], p[1] + 69, text=f"sesgo={sesgo:.4f}", fill=WARN, font=("Consolas", 9))

    for i in range(2):
        neurona(I[i], x[i], f"x{i + 1}", str(x[i]))
    for j in range(2):
        neurona(H[j], h[j], f"h{j + 1}", f"{h[j]:.4f}", m.B1[j])
    neurona(O, o, f"o1  (objetivo = {objetivo})", f"{o:.4f}", m.B2)


def dibuja_grafica(c, hist, m):
    c.delete("all")
    c.create_text(8, 12, anchor="w", text=f"MSE: {m.mse():.5f}  (meta ≤ {META_MSE})",
                  fill=TXT, font=("Helvetica", 10, "bold"))
    x0, x1, y0, y1 = 55, 640, 30, 112
    ymax = max(META_MSE * 1.2, max(value for _, value in hist) * 1.05)
    xmax = max(hist[-1][0], 1)
    px = lambda epoch: x0 + (x1 - x0) * epoch / xmax
    py = lambda value: y1 - (y1 - y0) * value / ymax
    c.create_rectangle(x0, y0, x1, y1, outline=PANEL)
    c.create_line(x0, py(META_MSE), x1, py(META_MSE), fill=WARN, dash=(4, 3))
    c.create_text(8, 24, anchor="w", text="MSE por época", fill=MUT, font=("Helvetica", 9, "bold"))
    c.create_text(x0 - 5, y0, anchor="e", text=f"{ymax:.2f}", fill=MUT, font=("Consolas", 8))
    c.create_text(x0 - 5, y1, anchor="e", text="0", fill=MUT, font=("Consolas", 8))
    c.create_text(x0, y1 + 10, text="0", fill=MUT, font=("Consolas", 8))
    c.create_text(x1, y1 + 10, text=f"{hist[-1][0]} épocas", fill=MUT, font=("Consolas", 8), anchor="e")
    paso = max(1, len(hist) // 300)
    pts = [(px(epoch), py(value)) for epoch, value in hist[::paso]]
    if pts[-1] != (px(hist[-1][0]), py(hist[-1][1])):
        pts.append((px(hist[-1][0]), py(hist[-1][1])))
    if len(pts) > 1:
        c.create_line(*[q for p in pts for q in p], fill=ACC, width=2)
    color = GOOD if m.mse() <= META_MSE else ACC
    c.create_oval(pts[-1][0] - 4, pts[-1][1] - 4, pts[-1][0] + 4, pts[-1][1] + 4,
                  fill=color, outline="")

    c.create_text(8, 131, anchor="w", text="Tabla XOR: salida / clase predicha (umbral 0,5)",
                  fill=TXT, font=("Helvetica", 9, "bold"))
    for i, (x1, x2, target) in enumerate(DATOS_XOR):
        output = m.salida((x1, x2))
        predicted = int(output >= 0.5)
        c.create_text(8, 151 + i * 15,
                      anchor="w",
                      text=f"e{i + 1} ({x1},{x2})  t={target}  o={output:.4f}  pred={predicted}",
                      fill=GOOD if predicted == target else BAD,
                      font=("Consolas", 9))


class App:
    def __init__(self):
        self.r = tk.Tk()
        self.r.title("Perceptrón multicapa XOR - UMNG")
        self.r.configure(bg=BG)
        self.f = None
        self.portada()
        self.r.mainloop()

    def nueva(self, tam):
        if self.f:
            self.f.destroy()
        self.r.geometry(tam)
        self.f = tk.Frame(self.r, bg=BG)
        self.f.pack(fill="both", expand=True)
        return self.f

    def lbl(self, p, t, sz=11, col=TXT, b=False, bg=PANEL, **k):
        return tk.Label(p, text=t, bg=bg, fg=col, font=("Helvetica", sz, "bold" if b else "normal"), **k)

    def btn(self, p, t, cmd, col=ACC, w=9):
        return tk.Button(p, text=t, command=cmd, bg=col, fg="#0b1220", font=("Helvetica", 11, "bold"),
                         relief="flat", width=w, cursor="hand2", activebackground="white")

    def pantalla_imagen(self, archivo, w, h, zona, siguiente, titulo):
        """Muestra una imagen de fondo y deja clicable la zona (x0, x1, y0, y1) del botón dibujado."""
        f = self.nueva(f"{w}x{h}")
        ruta = os.path.join(CARPETA, archivo)
        if Image is None or not os.path.exists(ruta):      # sin imagen o sin Pillow
            self.lbl(f, titulo, 28, ACC, True, bg=BG).pack(pady=(200, 30))
            self.btn(f, "▶  CONTINUAR", siguiente, GOOD, 18).pack()
            return
        resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        self.img = ImageTk.PhotoImage(Image.open(ruta).resize((w, h), resample))   # guardar referencia
        c = tk.Canvas(f, width=w, height=h, highlightthickness=0, bd=0)
        c.pack()
        c.create_image(0, 0, image=self.img, anchor="nw")
        dentro = lambda e: zona[0] <= e.x <= zona[1] and zona[2] <= e.y <= zona[3]
        c.bind("<Motion>", lambda e: c.config(cursor="hand2" if dentro(e) else ""))
        c.bind("<Button-1>", lambda e: siguiente() if dentro(e) else None)

    # ---------- 1. PORTADA / 2. PRESENTACIÓN ----------
    def portada(self):
        self.pantalla_imagen("INICIO_PM.png", 1100, 695, (376, 722, 422, 497), self.presentacion,
                            "Perceptrón Multicapa  XOR (e1)")

    def presentacion(self):
        self.pantalla_imagen("PRESENTACION_PM.png", 1100, 733, (736, 1004, 601, 659), self.juego,
                            "Red y su estado inicial")

    # ---------- 3. JUEGO ----------
    def juego(self):
        f = self.nueva("1100x700")
        self.run = False
        self.m = MLP(W1_0, W2_0, ALPHA_0, B1_0, B2_0)
        self.modo, self.muestra = "demo", (X, T)
        self.n, self.epocas = 0, 0
        self.hist = [(0, self.m.mse())]
        izq = tk.Frame(f, bg=BG)
        izq.pack(side="left", padx=10)
        self.cn = tk.Canvas(izq, width=660, height=430, bg=BG, highlightthickness=0)
        self.cn.pack()
        self.cg = tk.Canvas(izq, width=680, height=215, bg=BG, highlightthickness=0)
        self.cg.pack()
        d = tk.Frame(f, bg=PANEL)
        d.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        self.cl = self.lbl(d, "", 22, ACC, True)
        self.cl.pack(pady=(10, 0))
        self.st = self.lbl(d, "", 11, WARN, True, wraplength=330, height=3)
        self.st.pack(pady=4)

        self.lbl(d, f"Parada del XOR completo: MSE ≤ {META_MSE}", 10, WARN, True).pack()

        self.al = tk.Scale(d, from_=0.05, to=5.0, resolution=0.05, orient="horizontal", length=300,
                          label="α  tasa de aprendizaje", bg=PANEL, fg=TXT,
                          highlightthickness=0, troughcolor=BG, command=self.cambia_alpha)
        self.al.set(ALPHA_0)
        self.al.pack(fill="x", padx=20, pady=(8, 0))
        self.vel = tk.Scale(d, from_=1, to=5000, orient="horizontal", length=300,
                           label="Épocas por paso en Auto XOR", bg=PANEL, fg=TXT,
                           highlightthickness=0, troughcolor=BG)
        self.vel.set(10)
        self.vel.pack(fill="x", padx=20)

        b = tk.Frame(d, bg=PANEL)
        b.pack(pady=6)
        self.btn(b, "▶ Paso e1", self.paso_e1, ACC, 11).grid(row=0, column=0, padx=3, pady=2)
        self.btn(b, "▶ Época XOR", self.epoca_xor, ACC, 12).grid(row=0, column=1, padx=3, pady=2)
        self.btn(b, "↺ Reiniciar", self.reinicia, WARN, 11).grid(row=0, column=2, padx=3, pady=2)
        self.pl = self.btn(b, "⏵ Auto XOR", self.auto, GOOD, 14)
        self.pl.grid(row=1, column=0, columnspan=3, pady=2)

        fn = tk.Frame(d, bg=PANEL)
        fn.pack(pady=4)
        self.lbl(fn, "Avanzar").pack(side="left")
        self.nc = tk.Entry(fn, width=8, justify="center")
        self.nc.insert(0, "100")
        self.nc.pack(side="left", padx=4)
        self.lbl(fn, "épocas XOR").pack(side="left")
        self.btn(fn, "Ir", self.ir, ACC, 5).pack(side="left", padx=6)

        self.lbl(d, "Parámetros iniciales editables:", 10, MUT).pack(pady=(8, 0))
        gp = tk.Frame(d, bg=PANEL)
        gp.pack()
        self.en = {}
        valores = (.3, .2, .6, -.4, .5, .7, .1, .1, .1)
        nombres = ("w11", "w12", "w21", "w22", "v1", "v2", "bh1", "bh2", "bo1")
        for k, (nom, val) in enumerate(zip(nombres, valores)):
            tk.Label(gp, text=nom, bg=PANEL, fg=TXT).grid(row=k // 3, column=(k % 3) * 2)
            e = tk.Entry(gp, width=6, justify="center")
            e.insert(0, str(val))
            e.grid(row=k // 3, column=(k % 3) * 2 + 1, padx=2, pady=2)
            self.en[nom] = e
        self.btn(d, "Aplicar pesos y reiniciar", self.reinicia, ACC, 24).pack(pady=(6, 2))
        self.btn(d, "Volver a los datos del ejercicio", self.datos, MUT, 24).pack()
        self.info = self.lbl(d, "", 8, MUT, wraplength=360, justify="left")
        self.info.pack(pady=4)
        self.pinta()

    def cambia_alpha(self, v):
        if hasattr(self, "m"):
            self.m.alpha = float(v)

    def datos(self):
        for nom, v in zip(self.en, (.3, .2, .6, -.4, .5, .7, .1, .1, .1)):
            self.en[nom].delete(0, "end")
            self.en[nom].insert(0, str(v))
        self.al.set(ALPHA_0)
        self.reinicia()

    def parametros(self):
        try:
            v = [float(self.en[k].get()) for k in self.en]
        except ValueError:
            self.st.config(text="Escribe números válidos en pesos y sesgos", fg=BAD)
            return None
        if not all(math.isfinite(value) for value in v):
            self.st.config(text="Los pesos y sesgos deben ser números finitos", fg=BAD)
            return None
        return v

    def reinicia(self):
        self.run = False
        self.pl.config(text="⏵ Auto XOR")
        v = self.parametros()
        if v is None:
            return
        self.m = MLP([[v[0], v[1]], [v[2], v[3]]], [v[4], v[5]], self.al.get(),
                     [v[6], v[7]], v[8])
        self.modo, self.muestra = "demo", (X, T)
        self.n, self.epocas = 0, 0
        self.hist = [(0, self.m.mse())]
        self.pinta()

    def auto(self):
        if self.run:
            self.run = False
            self.pl.config(text="⏵ Auto XOR")
            return
        if not self.preparar_xor():
            return
        self.run = True
        self.pl.config(text="⏸ Pausa")
        self.tick()

    def tick(self):
        if self.run:
            self.avanza_epocas(self.vel.get())
            if self.run:
                self.r.after(30, self.tick)

    def ir(self):
        try:
            cantidad = int(self.nc.get())
        except ValueError:
            self.st.config(text="Escribe una cantidad entera de épocas", fg=BAD)
            return
        self.avanza_epocas(max(1, min(cantidad, MAX_EPOCAS)))

    def preparar_xor(self):
        if self.modo == "xor":
            return True
        v = self.parametros()
        if v is None:
            return False
        self.m = MLP([[v[0], v[1]], [v[2], v[3]]], [v[4], v[5]], self.al.get(),
                     [v[6], v[7]], v[8])
        self.modo, self.muestra = "xor", (X, T)
        self.n, self.epocas = 0, 0
        self.hist = [(0, self.m.mse())]
        return True

    def paso_e1(self):
        self.run = False
        self.pl.config(text="⏵ Auto XOR")
        if self.modo != "demo":
            self.reinicia()
            if self.modo != "demo":
                return
        self.modo, self.muestra = "demo", (X, T)
        self.m.paso(X, T)
        self.n += 1
        self.hist = [(0, self.m.mse())]
        self.pinta()

    def epoca_xor(self):
        if not self.preparar_xor():
            return
        self.avanza_epocas(1)

    def avanza_epocas(self, cantidad):
        if not self.preparar_xor():
            return
        for _ in range(cantidad):
            if self.epocas >= MAX_EPOCAS or self.m.mse() <= META_MSE:
                break
            self.m.entrenar_epoca()
            self.epocas += 1
            self.n += len(DATOS_XOR)
            self.hist.append((self.epocas, self.m.mse()))
        if len(self.hist) > 2000:
            self.hist = self.hist[::2] + [self.hist[-1]]
        self.pinta()

    def pinta(self):
        x, target = self.muestra
        h, o = self.m.forward(x)
        mse = self.m.mse()
        dibuja_red(self.cn, self.m, x, target)
        dibuja_grafica(self.cg, self.hist, self.m)
        self.cl.config(text=f"Épocas {self.epocas}  |  Iteraciones (patrones) {self.n}")
        self.info.config(
            text=(f"W1 = [[{self.m.W1[0][0]:.4f}, {self.m.W1[0][1]:.4f}], "
                  f"[{self.m.W1[1][0]:.4f}, {self.m.W1[1][1]:.4f}]]\n"
                  f"W2 = [{self.m.W2[0]:.4f}, {self.m.W2[1]:.4f}]\n"
                  f"Sesgos ocultos = [{self.m.B1[0]:.4f}, {self.m.B1[1]:.4f}]  "
                  f"sesgo salida = {self.m.B2:.4f}"))
        if self.modo == "xor" and mse <= META_MSE:
            self.run = False
            self.pl.config(text="⏵ Auto XOR")
            self.st.config(text=f"¡XOR entrenado! MSE = {mse:.5f}\n"
                                f"{self.epocas} épocas, {self.n} iteraciones.", fg=GOOD)
        elif self.modo == "xor" and self.epocas >= MAX_EPOCAS:
            self.run = False
            self.pl.config(text="⏵ Auto XOR")
            self.st.config(text=f"Límite de épocas alcanzado.\nMSE final = {mse:.5f}", fg=BAD)
        elif self.modo == "xor":
            self.st.config(text=f"Entrenando XOR completo…\nMSE = {mse:.5f} (meta ≤ {META_MSE})", fg=ACC)
        elif self.n == 0:
            self.st.config(text=f"Estado inicial e1: h1={h[0]:.4f}, h2={h[1]:.4f}, "
                                f"o1={o:.4f}\nMSE inicial del XOR = {mse:.5f}", fg=WARN)
        else:
            self.st.config(text=f"Demostración e1: salida = {o:.5f}, "
                                f"error = {target - o:.5f}\nMSE del XOR completo = {mse:.5f}", fg=ACC)


if __name__ == "__main__":
    App()
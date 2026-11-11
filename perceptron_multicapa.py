##Hecho por:Paula,Danna y Yessica
##----------------PERCEPTRON MULTICAPA-----------------------

"""Perceptrón multicapa XOR (e1) - Juego visual.
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
METAS = ["0.90", "0.95", "0.99", "0.999"]
MAX_CICLOS = 2_000_000


def g(x):
    return 1 / (1 + math.exp(-max(-60, min(60, x))))


class MLP:
    def __init__(self, W1, W2, alpha, B1, B2):
        self.W1, self.W2, self.alpha = [r[:] for r in W1], W2[:], alpha
        self.B1, self.B2 = B1[:], B2

    def forward(self):
        h = [g(X[0] * self.W1[0][j] + X[1] * self.W1[1][j] + self.B1[j]) for j in range(2)]
        return h, g(h[0] * self.W2[0] + h[1] * self.W2[1] + self.B2)

    def salida(self):
        return self.forward()[1]

    def paso(self):
        """Un ciclo: forward + backpropagation (F5-F8) con el patrón e1."""
        h, o = self.forward()
        do = o * (1 - o) * (T - o)                                    # F5
        dh = [h[j] * (1 - h[j]) * self.W2[j] * do for j in range(2)]     # F7 (con W2 viejo)
        for j in range(2):
            self.W2[j] += self.alpha * h[j] * do                            # F6
            self.B1[j] += self.alpha * dh[j]
            for i in range(2):
                self.W1[i][j] += self.alpha * X[i] * dh[j]                  # F8
        self.B2 += self.alpha * do


def mezcla(v):
    a, b = (30, 41, 59), (56, 189, 248)
    return "#%02x%02x%02x" % tuple(int(a[k] + (b[k] - a[k]) * v) for k in range(3))


def dibuja_red(c, m):
    c.delete("all")
    h, o = m.forward()
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
        neurona(I[i], X[i], f"x{i + 1}", str(X[i]))
    for j in range(2):
        neurona(H[j], h[j], f"h{j + 1}", f"{h[j]:.4f}", m.B1[j])
    neurona(O, o, f"o1  (objetivo = {T})", f"{o:.4f}", m.B2)


def dibuja_grafica(c, hist, meta, o):
    """Barra de la salida actual + curva de la salida a lo largo de los ciclos."""
    c.delete("all")
    c.create_text(8, 18, anchor="w", text="Salida o1", fill=TXT, font=("Helvetica", 11, "bold"))
    c.create_rectangle(100, 8, 560, 28, fill=PANEL, outline="")
    c.create_rectangle(100, 8, 100 + 460 * o, 28, fill=GOOD if o >= meta else ACC, outline="")
    c.create_line(100 + 460 * meta, 3, 100 + 460 * meta, 33, fill=WARN, width=3)
    c.create_line(560, 3, 560, 33, fill="white", width=2)
    c.create_text(570, 18, anchor="w", text=f"{o:.5f}", fill=TXT, font=("Consolas", 11, "bold"))
    c.create_text(100 + 460 * meta, 43, text=f"meta {meta}", fill=WARN, font=("Helvetica", 9))
    x0, x1, y0, y1 = 55, 640, 70, 190
    ymin = math.floor(min(p[1] for p in hist) * 10) / 10
    ymin = min(ymin, 0.6)
    xmax = max(hist[-1][0], 1)
    px = lambda n: x0 + (x1 - x0) * n / xmax
    py = lambda v: y1 - (y1 - y0) * (v - ymin) / (1 - ymin)
    c.create_rectangle(x0, y0, x1, y1, outline=PANEL)
    c.create_line(x0, py(meta), x1, py(meta), fill=WARN, dash=(4, 3))
    c.create_text(8, 58, anchor="w", text="Salida vs ciclos", fill=MUT, font=("Helvetica", 9, "bold"))
    for v in (ymin, 1.0):
        c.create_text(x0 - 5, py(v), anchor="e", text=f"{v:.1f}", fill=MUT, font=("Consolas", 9))
    c.create_text(x0, y1 + 12, text="0", fill=MUT, font=("Consolas", 9))
    c.create_text(x1, y1 + 12, text=f"ciclo {xmax}", fill=MUT, font=("Consolas", 9), anchor="e")
    paso = max(1, len(hist) // 300)
    pts = [(px(n), py(v)) for n, v in hist[::paso]] + [(px(hist[-1][0]), py(hist[-1][1]))]
    if len(pts) > 1:
        c.create_line(*[q for p in pts for q in p], fill=ACC, width=2)
    c.create_oval(pts[-1][0] - 4, pts[-1][1] - 4, pts[-1][0] + 4, pts[-1][1] + 4, fill=GOOD, outline="")


class App:
    def __init__(self):
        self.r = tk.Tk()
        self.r.title("Perceptrón multicapa XOR (e1) - UMNG")
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
        self.n, self.hist = 0, [(0, self.m.salida())]
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

        fm = tk.Frame(d, bg=PANEL)
        fm.pack()
        self.lbl(fm, "Meta de la salida:").pack(side="left")
        self.meta = tk.StringVar(value="0.99")
        om = tk.OptionMenu(fm, self.meta, *METAS, command=lambda v: self.pinta())
        om.config(bg=BG, fg=TXT, highlightthickness=0, activebackground=PANEL, width=6)
        om.pack(side="left", padx=6)

        self.al = tk.Scale(d, from_=0.05, to=5.0, resolution=0.05, orient="horizontal", length=300,
                          label="α  tasa de aprendizaje", bg=PANEL, fg=TXT,
                          highlightthickness=0, troughcolor=BG, command=self.cambia_alpha)
        self.al.set(ALPHA_0)
        self.al.pack(fill="x", padx=20, pady=(8, 0))
        self.vel = tk.Scale(d, from_=1, to=5000, orient="horizontal", length=300,
                           label="Velocidad: ciclos por paso en Auto", bg=PANEL, fg=TXT,
                           highlightthickness=0, troughcolor=BG)
        self.vel.set(10)
        self.vel.pack(fill="x", padx=20)

        b = tk.Frame(d, bg=PANEL)
        b.pack(pady=6)
        self.btn(b, "▶ 1 ciclo", lambda: self.avanza(1)).grid(row=0, column=0, padx=3)
        self.pl = self.btn(b, "⏵ Auto", self.auto, GOOD)
        self.pl.grid(row=0, column=1, padx=3)
        self.btn(b, "↺ Reiniciar", self.reinicia, WARN).grid(row=0, column=2, padx=3)

        fn = tk.Frame(d, bg=PANEL)
        fn.pack(pady=4)
        self.lbl(fn, "Avanzar").pack(side="left")
        self.nc = tk.Entry(fn, width=8, justify="center")
        self.nc.insert(0, "1000")
        self.nc.pack(side="left", padx=4)
        self.lbl(fn, "ciclos").pack(side="left")
        self.btn(fn, "Ir", self.ir, ACC, 5).pack(side="left", padx=6)

        self.lbl(d, "Pesos y sesgos iniciales (editables):", 10, MUT).pack(pady=(8, 0))
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
        self.pinta()

    def meta_val(self):
        return float(self.meta.get())

    def cambia_alpha(self, v):
        if hasattr(self, "m"):
            self.m.alpha = float(v)

    def datos(self):
        for nom, v in zip(self.en, (.3, .2, .6, -.4, .5, .7, .1, .1, .1)):
            self.en[nom].delete(0, "end")
            self.en[nom].insert(0, str(v))
        self.al.set(ALPHA_0)
        self.reinicia()

    def reinicia(self):
        self.run = False
        self.pl.config(text="⏵ Auto")
        try:
            v = [float(self.en[k].get()) for k in self.en]
        except ValueError:
            self.st.config(text="Escribe números válidos en los pesos", fg=BAD)
            return
        self.m = MLP([[v[0], v[1]], [v[2], v[3]]], [v[4], v[5]], self.al.get(),
                     [v[6], v[7]], v[8])
        self.n, self.hist = 0, [(0, self.m.salida())]
        self.pinta()

    def auto(self):
        self.run = not self.run
        self.pl.config(text="⏸ Pausa" if self.run else "⏵ Auto")
        if self.run:
            self.tick()

    def tick(self):
        if self.run:
            self.avanza(self.vel.get())
            self.r.after(30, self.tick)

    def ir(self):
        try:
            self.avanza(max(1, min(int(float(self.nc.get())), MAX_CICLOS)))
        except ValueError:
            self.st.config(text="Escribe un número de ciclos válido", fg=BAD)

    def avanza(self, k):
        meta = self.meta_val()
        if self.m.salida() >= meta or self.n >= MAX_CICLOS:
            return
        marca = max(1, k // 60)                    # puntos intermedios para que la curva sea suave
        for _ in range(k):
            self.m.paso()
            self.n += 1
            o = self.m.salida()
            if o >= meta or self.n >= MAX_CICLOS:
                break
            if self.n % marca == 0:
                self.hist.append((self.n, o))
        self.hist.append((self.n, self.m.salida()))
        if len(self.hist) > 2000:
            self.hist = self.hist[::2] + [self.hist[-1]]
        self.pinta()

    def pinta(self):
        o, meta = self.m.salida(), self.meta_val()
        dibuja_red(self.cn, self.m)
        dibuja_grafica(self.cg, self.hist, meta, o)
        self.cl.config(text=f"Ciclo {self.n}")
        if o >= meta:
            self.run = False
            self.pl.config(text="⏵ Auto")
            self.st.config(text=f"🎉 ¡Llegó a la meta!\nSalida = {o:.5f} en el ciclo {self.n}", fg=GOOD)
        elif self.n >= MAX_CICLOS:
            self.run = False
            self.st.config(text="Límite de ciclos alcanzado.\nSube α o cambia los pesos", fg=BAD)
        elif self.n == 0:
            self.st.config(text=f"Estado inicial: salida {o:.4f}, error {T - o:.4f}\nPulsa ▶ 1 ciclo o ⏵ Auto", fg=WARN)
        else:
            self.st.config(text=f"Entrenando…  error = {T - o:.5f}", fg=ACC)


if __name__ == "__main__":
    App()
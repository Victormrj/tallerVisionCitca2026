from __future__ import annotations

import json
import math
import os
import random
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from comun import argumentos, ejecutar, texto
from gestos import Suavizador

RUTA_RECORD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "record_fruit_ninja.json")

# (cáscara, pulpa) en BGR
FRUTAS = {
    "sandia": ((40, 130, 40), (70, 60, 235)),
    "naranja": ((0, 140, 255), (90, 195, 255)),
    "manzana": ((40, 40, 205), (200, 235, 245)),
    "limon": ((40, 220, 235), (160, 245, 250)),
    "kiwi": ((40, 85, 120), (70, 205, 130)),
    "uva": ((130, 50, 120), (160, 160, 210)),
}
VIDAS = 3
# Velocidad mínima del índice para que corte, en "anchos de pantalla por segundo".
# Menor = más sensible. 0.25 corta con un movimiento normal de la mano; 0.8 exige un latigazo.
# Si es demasiado baja, un dedo casi quieto sobre la fruta también corta.
VELOCIDAD_CORTE = 0.25
RADIO_CORTE = 1.3          # cuánto más grande que la fruta es la zona que cuenta como corte


@dataclass
class Fruta:
    x: float
    y: float
    vx: float
    vy: float
    r: float
    tipo: str              # clave de FRUTAS o "bomba"
    ang: float = 0.0
    vang: float = 0.0
    inicio: bool = False   # la fruta especial de "INICIO"


@dataclass
class Mitad:
    x: float
    y: float
    vx: float
    vy: float
    r: float
    tipo: str
    ang: float             # grados
    vang: float


@dataclass
class Particula:
    x: float
    y: float
    vx: float
    vy: float
    color: tuple
    vida: float


@dataclass
class TextoFlotante:
    s: str
    x: float
    y: float
    t0: float
    color: tuple = (255, 255, 255)


@dataclass
class Espada:
    rastro: deque = field(default_factory=lambda: deque(maxlen=10))
    suave: Suavizador = field(default_factory=lambda: Suavizador(0.75))
    cortes: list = field(default_factory=list)   # tiempos de cortes recientes (combos)


def dist_punto_segmento(p, a, b) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


class FruitNinja:
    def __init__(self, semilla: int | None = None, velocidad_corte: float = VELOCIDAD_CORTE):
        self.velocidad_corte = velocidad_corte
        self.rng = random.Random(semilla)
        self.record = self._leer_record()
        self.w = self.h = None
        self.reiniciar()

    # ------------------------------------------------------------ estado
    def reiniciar(self):
        self.estado = "INICIO"
        self.frutas: list[Fruta] = []
        self.mitades: list[Mitad] = []
        self.particulas: list[Particula] = []
        self.textos: list[TextoFlotante] = []
        self.espadas: dict[str, Espada] = {}
        self.puntos = 0
        self.vidas = VIDAS
        self.t_inicio = None
        self.t_prox = 0.0
        self.t_fin = 0.0
        self.destello = 0.0
        self._fruta_inicio_lista = False

    def _leer_record(self) -> int:
        try:
            with open(RUTA_RECORD) as f:
                return int(json.load(f).get("record", 0))
        except Exception:
            return 0

    def _guardar_record(self):
        try:
            with open(RUTA_RECORD, "w") as f:
                json.dump({"record": self.record}, f)
        except Exception:
            pass

    def tecla(self, k):
        if k == ord("r"):
            self.reiniciar()

    # ------------------------------------------------------------ física
    @property
    def g(self):
        return 1.5 * self.h            # gravedad en px/s²

    def _crear_fruta_inicio(self):
        r = 0.09 * self.h
        self.frutas.append(Fruta(self.w * 0.5, self.h * 0.55, 0, 0, r, "sandia",
                                 vang=0.8, inicio=True))

    def _lanzar(self, t):
        nivel = min(1.0, (t - self.t_inicio) / 90.0)       # dificultad sube en 90 s
        cantidad = self.rng.choices([1, 2, 3, 4], weights=[5, 4, 2 + 3 * nivel, 3 * nivel])[0]
        for _ in range(cantidad):
            bomba = self.rng.random() < 0.08 + 0.12 * nivel
            r = self.h * self.rng.uniform(0.055, 0.075)
            x = self.w * self.rng.uniform(0.15, 0.85)
            pico = self.h * self.rng.uniform(0.12, 0.45)       # altura máxima
            vy = -math.sqrt(2 * self.g * (self.h + r - pico))
            t_vuelo = 2 * -vy / self.g
            destino = self.w * self.rng.uniform(0.2, 0.8)
            tipo = "bomba" if bomba else self.rng.choice(list(FRUTAS))
            self.frutas.append(Fruta(x, self.h + r, (destino - x) / t_vuelo, vy, r, tipo,
                                     vang=self.rng.uniform(-3, 3)))
        self.t_prox = t + max(0.55, 1.4 - 0.85 * nivel) * self.rng.uniform(0.8, 1.2)

    def _fisica(self, dt):
        for f in self.frutas:
            if not f.inicio:
                f.vy += self.g * dt
                f.x += f.vx * dt
                f.y += f.vy * dt
            f.ang += f.vang * dt
        for m in self.mitades:
            m.vy += self.g * dt
            m.x += m.vx * dt
            m.y += m.vy * dt
            m.ang += m.vang * dt
        for p in self.particulas:
            p.vy += self.g * 0.6 * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.vida -= dt
        self.mitades = [m for m in self.mitades if m.y - m.r < self.h]
        self.particulas = [p for p in self.particulas if p.vida > 0]

        # frutas que se caen sin cortar
        quedan = []
        for f in self.frutas:
            if f.vy > 0 and f.y - f.r > self.h:
                if f.tipo != "bomba" and self.estado == "JUGANDO":
                    self.vidas -= 1
                    self.textos.append(TextoFlotante("X", f.x, self.h - 40, self._t, (40, 40, 230)))
            else:
                quedan.append(f)
        self.frutas = quedan

    # ------------------------------------------------------------ cortes
    def _cortar(self, f: Fruta, direccion: np.ndarray, espada: Espada, t):
        self.frutas.remove(f)
        d = direccion / (np.linalg.norm(direccion) + 1e-9)
        n = np.array([-d[1], d[0]])
        ang = math.degrees(math.atan2(d[1], d[0]))
        if f.tipo == "bomba":
            self.vidas -= 1
            self.destello = 1.0
            for _ in range(40):
                a, v = self.rng.uniform(0, 2 * math.pi), self.rng.uniform(150, 650)
                self.particulas.append(Particula(f.x, f.y, v * math.cos(a), v * math.sin(a),
                                                 (0, self.rng.randint(120, 255), 255), 0.7))
            self.textos.append(TextoFlotante("BOOM!", f.x, f.y, t, (0, 80, 255)))
            return
        pulpa = FRUTAS[f.tipo][1]
        for s in (1, -1):
            self.mitades.append(Mitad(f.x, f.y, f.vx * 0.5 + s * n[0] * 160,
                                      f.vy * 0.5 + s * n[1] * 160 - 80, f.r, f.tipo,
                                      ang if s == 1 else ang + 180, s * 120))
        for _ in range(18):
            a, v = self.rng.uniform(0, 2 * math.pi), self.rng.uniform(80, 420)
            self.particulas.append(Particula(f.x, f.y, v * math.cos(a), v * math.sin(a) - 100,
                                             pulpa, self.rng.uniform(0.3, 0.7)))
        if f.inicio:
            self._empezar(t)
            return
        self.puntos += 1
        espada.cortes = [tc for tc in espada.cortes if t - tc < 0.35] + [t]
        if len(espada.cortes) >= 3:
            extra = len(espada.cortes) - 2
            self.puntos += extra
            self.textos.append(TextoFlotante(f"COMBO x{len(espada.cortes)}  +{extra}",
                                             f.x, f.y - 40, t, (0, 220, 255)))
        else:
            self.textos.append(TextoFlotante("+1", f.x, f.y, t))

    def _empezar(self, t):
        self.estado = "JUGANDO"
        self.puntos, self.vidas = 0, VIDAS
        self.t_inicio = t
        self.t_prox = t + 0.8
        self.frutas = []

    def _actualizar_espadas(self, manos, t, dt):
        vistas = set()
        for i, mano in enumerate(sorted(manos, key=lambda m: m.palma[0])):
            clave = f"{mano.lado}{i}"
            vistas.add(clave)
            esp = self.espadas.setdefault(clave, Espada())
            p = esp.suave(mano.pts[8])
            esp.rastro.append((p.copy(), t))
            if len(esp.rastro) < 2:
                continue
            (a, ta), (b, tb) = esp.rastro[-2], esp.rastro[-1]
            vel = np.linalg.norm(b - a) / max(1e-3, tb - ta)
            if vel < self.velocidad_corte * self.w:     # casi quieto: no corta
                continue
            for f in list(self.frutas):
                if dist_punto_segmento(np.array([f.x, f.y]), a, b) < f.r * RADIO_CORTE:
                    self._cortar(f, b - a, esp, t)
        for clave in list(self.espadas):
            if clave not in vistas:
                del self.espadas[clave]

    # ------------------------------------------------------------ bucle
    def actualizar(self, frame, manos, t):
        self.h, self.w = frame.shape[:2]
        dt = 0.0 if not hasattr(self, "_t") else min(0.1, t - self._t)
        self._t = t

        if self.estado == "INICIO" and not any(f.inicio for f in self.frutas):
            self._crear_fruta_inicio()
        if self.estado == "JUGANDO" and t >= self.t_prox:
            self._lanzar(t)

        self._fisica(dt)
        self._actualizar_espadas(manos, t, dt)

        if self.estado == "JUGANDO" and self.vidas <= 0:
            self.estado = "FIN"
            self.t_fin = t
            self.frutas = [f for f in self.frutas if f.tipo == "bomba"]
            if self.puntos > self.record:
                self.record = self.puntos
                self._guardar_record()
        if self.estado == "FIN" and t - self.t_fin > 1.5 and not any(f.inicio for f in self.frutas):
            self._crear_fruta_inicio()

        return self._dibujar(frame, t, dt)

    # ------------------------------------------------------------ dibujo
    def _dibujar_fruta(self, img, f: Fruta):
        c = (int(f.x), int(f.y))
        r = int(f.r)
        if f.tipo == "bomba":
            cv2.circle(img, c, r, (30, 30, 30), -1, cv2.LINE_AA)
            cv2.circle(img, c, r, (0, 0, 230), 3, cv2.LINE_AA)
            mecha = (int(f.x + r * 0.7 * math.cos(f.ang - 1.2)), int(f.y - r * 0.9))
            cv2.line(img, (c[0], c[1] - r), mecha, (60, 120, 160), 4, cv2.LINE_AA)
            cv2.circle(img, mecha, 7, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (c[0] - r // 3, c[1] - r // 3), r // 5, (110, 110, 110), -1, cv2.LINE_AA)
            return
        cascara, pulpa = FRUTAS[f.tipo]
        cv2.circle(img, c, r, cascara, -1, cv2.LINE_AA)
        if f.tipo == "sandia":     # rayas
            for k in (-0.5, 0, 0.5):
                a = f.ang + k
                p1 = (int(f.x + r * 0.95 * math.cos(a + 1.6)), int(f.y + r * 0.95 * math.sin(a + 1.6)))
                p2 = (int(f.x + r * 0.95 * math.cos(a - 1.6)), int(f.y + r * 0.95 * math.sin(a - 1.6)))
                cv2.line(img, p1, p2, (20, 80, 20), max(2, r // 8), cv2.LINE_AA)
        cv2.circle(img, c, r, tuple(int(v * 0.6) for v in cascara), 2, cv2.LINE_AA)
        cv2.ellipse(img, (int(f.x - r * 0.35), int(f.y - r * 0.4)), (max(2, r // 4), max(1, r // 7)),
                    -35, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)   # brillo
        if f.tipo in ("manzana", "naranja", "limon"):                 # tallo
            p = (int(f.x + r * 0.15 * math.sin(f.ang)), int(f.y - r * 1.25))
            cv2.line(img, (c[0], c[1] - r + 2), p, (30, 70, 90), 3, cv2.LINE_AA)
        if f.inicio:
            texto(img, "INICIO", (f.x, f.y), (255, 255, 255), 0.9, 2, centrado=True)

    def _dibujar_mitad(self, img, m: Mitad):
        cascara, pulpa = FRUTAS[m.tipo]
        c, r = (int(m.x), int(m.y)), int(m.r)
        cv2.ellipse(img, c, (r, r), m.ang, 0, 180, cascara, -1, cv2.LINE_AA)
        cv2.ellipse(img, c, (int(r * 0.85), int(r * 0.85)), m.ang, 0, 180, pulpa, -1, cv2.LINE_AA)

    def _dibujar(self, frame, t, dt):
        img = cv2.convertScaleAbs(frame, alpha=0.7, beta=0)       # oscurecer el fondo
        for p in self.particulas:
            cv2.circle(img, (int(p.x), int(p.y)), 4, p.color, -1, cv2.LINE_AA)
        for m in self.mitades:
            self._dibujar_mitad(img, m)
        for f in self.frutas:
            self._dibujar_fruta(img, f)

        # espadas: rastro que se adelgaza
        for esp in self.espadas.values():
            pts = [p for p, tp in esp.rastro if t - tp < 0.18]
            for k in range(1, len(pts)):
                grosor = max(1, int(14 * k / len(pts)))
                a, b = tuple(int(v) for v in pts[k - 1]), tuple(int(v) for v in pts[k])
                cv2.line(img, a, b, (255, 230, 150), grosor + 6, cv2.LINE_AA)
                cv2.line(img, a, b, (255, 255, 255), grosor, cv2.LINE_AA)

        # textos flotantes
        vivos = []
        for tf in self.textos:
            edad = t - tf.t0
            if edad < 0.9:
                texto(img, tf.s, (tf.x, tf.y - 60 * edad), tf.color, 0.9, 2, centrado=True)
                vivos.append(tf)
        self.textos = vivos

        if self.destello > 0:
            blanco = np.full_like(img, 255)
            img = cv2.addWeighted(img, 1 - self.destello * 0.8, blanco, self.destello * 0.8, 0)
            self.destello = max(0.0, self.destello - 2.5 * max(dt, 1 / 30))

        # marcador
        texto(img, f"PUNTOS {self.puntos}", (20, 45), (0, 220, 255), 1.2, 3)
        texto(img, f"RECORD {self.record}", (22, 80), (220, 220, 220), 0.6, 1)
        for i in range(VIDAS):
            color = (40, 40, 230) if i < max(0, self.vidas) else (90, 90, 90)
            x = self.w - 50 - i * 45
            cv2.line(img, (x - 14, 26), (x + 14, 54), color, 6, cv2.LINE_AA)
            cv2.line(img, (x + 14, 26), (x - 14, 54), color, 6, cv2.LINE_AA)

        if self.estado == "INICIO":
            texto(img, "FRUIT NINJA", (self.w / 2, self.h * 0.22), (0, 220, 255), 2.0, 4, centrado=True)
            texto(img, "Pasa el indice sobre la fruta para cortar. Corta la sandia para empezar",
                  (self.w / 2, self.h * 0.8), (255, 255, 255), 0.7, 2, centrado=True)
        elif self.estado == "FIN":
            texto(img, "FIN DEL JUEGO", (self.w / 2, self.h * 0.22), (40, 40, 230), 2.0, 4, centrado=True)
            texto(img, f"Puntos: {self.puntos}   Record: {self.record}", (self.w / 2, self.h * 0.32),
                  (255, 255, 255), 1.0, 2, centrado=True)
        return img


if __name__ == "__main__":
    ap = argumentos("Fruit Ninja con las manos")
    ap.add_argument("--corte", type=float, default=VELOCIDAD_CORTE,
                    help="velocidad mínima para cortar, en anchos de pantalla por segundo "
                         "(menor = más sensible; por omisión %(default)s)")
    args = ap.parse_args()
    ejecutar(FruitNinja(velocidad_corte=max(0.0, args.corte)), args, "Fruit Ninja")

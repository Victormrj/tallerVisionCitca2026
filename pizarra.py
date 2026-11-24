"""
pizarra.py — Pintar en el aire con el dedo índice.

Gestos (cada mano por separado, puedes dibujar con las dos):
    sólo índice arriba        -> DIBUJAR
    índice + medio arriba     -> SELECCIONAR: lleva los dedos a un botón de la barra
                                 y mantenlos ahí medio segundo
    palma abierta / puño      -> mover sin dibujar

Teclado: 1-8 colores | g goma | c borrar | z deshacer | s guardar | p modo pizarrón | q salir

    python pizarra.py
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np

from comun import argumentos, ejecutar, texto
from gestos import Suavizador
from manos import dibujar_mano

COLORES = [("rojo", (40, 40, 230)), ("naranja", (0, 140, 255)), ("amarillo", (0, 220, 255)),
           ("verde", (60, 200, 60)), ("cian", (230, 200, 0)), ("azul", (230, 90, 30)),
           ("morado", (200, 60, 160)), ("blanco", (245, 245, 245))]
GROSORES = [4, 8, 14, 24]
TIEMPO_SELECCION = 0.5          # segundos que hay que "posar" sobre un botón
CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dibujos")


def clasificar(mano) -> str:
    _, indice, medio, anular, menique = mano.dedos()
    if indice and not medio and not anular and not menique:
        return "dibujar"
    if indice and medio and not anular and not menique:
        return "seleccionar"
    return "mover"


class EstadoMano:
    def __init__(self):
        self.suave = Suavizador(0.5)
        self.prev = None
        self.modo = "mover"
        self.boton = None
        self.t_boton = 0.0
        self.disparado = False


class Pizarra:
    def __init__(self):
        self.lienzo = None
        self.color = 0
        self.grosor = 1
        self.goma = False
        self.pizarron = False
        self.deshacer_pila: list[np.ndarray] = []
        self.manos: dict[str, EstadoMano] = {}
        self.aviso, self.t_aviso = "", 0.0
        self.botones = []

    # --------------------------------------------------------------- barra
    def _crear_botones(self, w, h):
        acciones = [("color", i) for i in range(len(COLORES))] + [
            ("goma", None), ("grosor", None), ("deshacer", None),
            ("borrar", None), ("guardar", None), ("pizarron", None)]
        alto = int(0.13 * h)
        ancho = w / len(acciones)
        self.alto_barra = alto
        self.botones = []
        for i, (tipo, val) in enumerate(acciones):
            x0, x1 = int(i * ancho + 4), int((i + 1) * ancho - 4)
            self.botones.append(((x0, 6, x1, alto - 6), tipo, val))

    def _boton_en(self, p):
        for b in self.botones:
            x0, y0, x1, y1 = b[0]
            if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
                return b
        return None

    def _accion(self, tipo, val):
        if tipo == "color":
            self.color, self.goma = val, False
        elif tipo == "goma":
            self.goma = not self.goma
        elif tipo == "grosor":
            self.grosor = (self.grosor + 1) % len(GROSORES)
        elif tipo == "deshacer":
            self.deshacer()
        elif tipo == "borrar":
            self._guardar_estado()
            self.lienzo[:] = 0
        elif tipo == "guardar":
            self.guardar()
        elif tipo == "pizarron":
            self.pizarron = not self.pizarron

    def _avisar(self, s):
        self.aviso, self.t_aviso = s, time.monotonic()

    # --------------------------------------------------------------- lienzo
    def _guardar_estado(self):
        self.deshacer_pila.append(self.lienzo.copy())
        if len(self.deshacer_pila) > 25:
            self.deshacer_pila.pop(0)

    def deshacer(self):
        if self.deshacer_pila:
            self.lienzo = self.deshacer_pila.pop()

    def guardar(self):
        os.makedirs(CARPETA, exist_ok=True)
        fondo = np.full_like(self.lienzo, 255)
        mascara = self.lienzo.max(axis=2) > 0
        fondo[mascara] = self.lienzo[mascara]
        ruta = os.path.join(CARPETA, time.strftime("dibujo_%Y%m%d_%H%M%S.png"))
        cv2.imwrite(ruta, fondo)
        self._avisar(f"Guardado en dibujos/{os.path.basename(ruta)}")

    def tecla(self, k):
        c = chr(k) if k < 128 else ""
        if c and c in "12345678":
            self._accion("color", int(c) - 1)
        elif c in ("g", "c", "z", "s", "p") and c:
            self._accion({"g": "goma", "c": "borrar", "z": "deshacer", "s": "guardar",
                          "p": "pizarron"}[c], None)

    # --------------------------------------------------------------- bucle
    def actualizar(self, frame, manos, t):
        h, w = frame.shape[:2]
        if self.lienzo is None or self.lienzo.shape[:2] != (h, w):
            self.lienzo = np.zeros_like(frame)
            self._crear_botones(w, h)

        vistas = set()
        for i, mano in enumerate(sorted(manos, key=lambda m: m.palma[0])):
            clave = f"{mano.lado}{i}"
            vistas.add(clave)
            est = self.manos.setdefault(clave, EstadoMano())
            modo = clasificar(mano)
            if modo != est.modo:
                est.prev = None
                est.suave.reiniciar()
                est.boton, est.disparado = None, False
            est.modo = modo

            if modo == "dibujar":
                p = est.suave(mano.pts[8])
                q = tuple(int(v) for v in p)
                if p[1] < self.alto_barra:          # no pintar sobre la barra
                    est.prev = None
                    continue
                if est.prev is None:
                    self._guardar_estado()          # inicio de trazo -> punto de deshacer
                    est.prev = q
                color = (0, 0, 0) if self.goma else COLORES[self.color][1]
                grosor = GROSORES[self.grosor] * (3 if self.goma else 1)
                cv2.line(self.lienzo, est.prev, q, color, grosor, cv2.LINE_AA)
                est.prev = q

            elif modo == "seleccionar":
                p = est.suave((mano.pts[8] + mano.pts[12]) / 2)
                b = self._boton_en(p)
                if b is not est.boton:
                    est.boton, est.t_boton, est.disparado = b, t, False
                elif b is not None and not est.disparado and t - est.t_boton >= TIEMPO_SELECCION:
                    self._accion(b[1], b[2])
                    est.disparado = True
        for clave in list(self.manos):
            if clave not in vistas:
                del self.manos[clave]

        return self._componer(frame, manos, t)

    # --------------------------------------------------------------- dibujo
    def _componer(self, frame, manos, t):
        h, w = frame.shape[:2]
        if self.pizarron:
            salida = np.full_like(frame, (40, 52, 36))         # verde pizarrón
        else:
            salida = frame.copy()
        mascara = self.lienzo.max(axis=2) > 0
        salida[mascara] = self.lienzo[mascara]

        # barra de herramientas semitransparente
        barra = salida[:self.alto_barra].copy()
        cv2.rectangle(barra, (0, 0), (w, self.alto_barra), (25, 25, 25), -1)
        salida[:self.alto_barra] = cv2.addWeighted(barra, 0.65, salida[:self.alto_barra], 0.35, 0)
        etiquetas = {"goma": "GOMA", "grosor": f"GROSOR {GROSORES[self.grosor]}",
                     "deshacer": "DESHACER", "borrar": "BORRAR", "guardar": "GUARDAR",
                     "pizarron": "CAMARA" if self.pizarron else "PIZARRON"}
        for (x0, y0, x1, y1), tipo, val in self.botones:
            activo = (tipo == "color" and val == self.color and not self.goma) or \
                     (tipo == "goma" and self.goma)
            if tipo == "color":
                cv2.rectangle(salida, (x0, y0), (x1, y1), COLORES[val][1], -1)
            else:
                cv2.rectangle(salida, (x0, y0), (x1, y1), (70, 70, 70), -1)
                esc = 0.38 if (x1 - x0) < 110 else 0.45
                texto(salida, etiquetas[tipo], ((x0 + x1) // 2, (y0 + y1) // 2),
                      (255, 255, 255), esc, 1, centrado=True)
            cv2.rectangle(salida, (x0, y0), (x1, y1),
                          (255, 255, 255) if activo else (0, 0, 0), 4 if activo else 1)

        # manos y cursores
        for i, mano in enumerate(sorted(manos, key=lambda m: m.palma[0])):
            est = self.manos.get(f"{mano.lado}{i}")
            if est is None:
                continue
            color_trazo = (200, 200, 200) if self.goma else COLORES[self.color][1]
            dibujar_mano(salida, mano, (180, 180, 180) if self.pizarron else (255, 255, 255))
            if est.modo == "dibujar":
                c = tuple(int(v) for v in mano.pts[8])
                r = GROSORES[self.grosor] * (3 if self.goma else 1) // 2 + 3
                cv2.circle(salida, c, r, color_trazo, -1 if not self.goma else 2, cv2.LINE_AA)
                cv2.circle(salida, c, r + 2, (0, 0, 0), 1, cv2.LINE_AA)
            elif est.modo == "seleccionar" and est.suave.valor is not None:
                c = tuple(int(v) for v in est.suave.valor)
                cv2.circle(salida, c, 16, (255, 255, 255), 2, cv2.LINE_AA)
                if est.boton is not None and not est.disparado:
                    prog = min(1.0, (t - est.t_boton) / TIEMPO_SELECCION)
                    cv2.ellipse(salida, c, (22, 22), -90, 0, int(360 * prog),
                                (90, 235, 90), 4, cv2.LINE_AA)

        modo_txt = "GOMA" if self.goma else COLORES[self.color][0].upper()
        texto(salida, f"Indice = dibujar | Indice+medio = elegir en la barra | {modo_txt}",
              (12, h - 14), (245, 245, 245), 0.55, 1, placa=True)
        if self.aviso and time.monotonic() - self.t_aviso < 2.5:
            texto(salida, self.aviso, (w // 2, h // 2), (90, 235, 90), 0.9, 2, centrado=True)
        return salida


if __name__ == "__main__":
    args = argumentos("Pintar en el aire con las manos").parse_args()
    ejecutar(Pizarra(), args, "Pintar en el aire")

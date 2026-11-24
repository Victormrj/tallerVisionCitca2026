from __future__ import annotations

import os
import sys
from collections import Counter

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comun import argumentos, ejecutar, texto  
from manos import dibujar_mano  
from senas.caracteristicas import (cargar_muestras, guardar_muestras, mano_principal,  
                                   quitar_ultimas, vector)

MUESTRAS_POR_TOMA = 120
CUENTA_REGRESIVA = 2.0


class Grabador:
    def __init__(self, auto: str | None = None):
        _, etiquetas = cargar_muestras()
        self.conteo = Counter(etiquetas.tolist())
        self.etiqueta = None
        self.estado = "ESPERA"
        self.t0 = 0.0
        self.lote = []
        self.ultimo_lote = 0
        self._auto = auto.upper() if auto else None

    def _iniciar(self, etiqueta, t):
        self.etiqueta = etiqueta
        self.estado = "CUENTA"
        self.t0 = t
        self.lote = []

    def tecla(self, k):
        c = chr(k) if k < 128 else ""
        if c.isalnum() and self.estado == "ESPERA":
            self._iniciar(c.upper(), self._t)
        elif c == "-" and self.estado == "ESPERA" and self.ultimo_lote:
            quitar_ultimas(self.ultimo_lote)
            self.conteo[self._etiqueta_ultimo] -= self.ultimo_lote
            self.ultimo_lote = 0

    def actualizar(self, frame, manos, t):
        self._t = t
        h, w = frame.shape[:2]
        if self._auto and self.estado == "ESPERA":
            self._iniciar(self._auto, t)
            self._auto = None

        mano = mano_principal(manos)
        if mano is not None:
            dibujar_mano(frame, mano, (90, 235, 90) if self.estado == "GRABANDO" else (0, 170, 255))

        if self.estado == "CUENTA":
            restante = CUENTA_REGRESIVA - (t - self.t0)
            texto(frame, f"Prepara la sena '{self.etiqueta}'  {max(0, restante):.1f}",
                  (w / 2, h * 0.15), (0, 220, 255), 1.1, 3, centrado=True)
            if restante <= 0:
                self.estado = "GRABANDO"
        elif self.estado == "GRABANDO":
            if mano is not None:
                self.lote.append(vector(mano))
            prog = len(self.lote) / MUESTRAS_POR_TOMA
            texto(frame, f"GRABANDO '{self.etiqueta}'  -  mueve un poco la mano",
                  (w / 2, h * 0.15), (90, 235, 90), 1.0, 2, centrado=True)
            cv2.rectangle(frame, (int(w * 0.2), int(h * 0.2)), (int(w * 0.8), int(h * 0.2) + 14),
                          (60, 60, 60), -1)
            cv2.rectangle(frame, (int(w * 0.2), int(h * 0.2)),
                          (int(w * 0.2 + w * 0.6 * prog), int(h * 0.2) + 14), (90, 235, 90), -1)
            if len(self.lote) >= MUESTRAS_POR_TOMA:
                guardar_muestras(self.etiqueta, self.lote)
                self.conteo[self.etiqueta] += len(self.lote)
                self.ultimo_lote, self._etiqueta_ultimo = len(self.lote), self.etiqueta
                self.estado = "ESPERA"
        else:
            texto(frame, "Haz la sena y presiona su letra para grabar  ( - deshace, q sale )",
                  (w / 2, h * 0.08), (255, 255, 255), 0.7, 2, centrado=True)
            if mano is None:
                texto(frame, "No veo ninguna mano", (w / 2, h * 0.5), (40, 40, 230), 1.0, 2,
                      centrado=True)

        # tabla de cuántos ejemplos hay por etiqueta
        x, y = 15, int(h * 0.3)
        texto(frame, "Ejemplos guardados:", (x, y), (255, 255, 255), 0.6, 1, placa=True)
        for i, (et, n) in enumerate(sorted(self.conteo.items())):
            if n <= 0:
                continue
            col, fila = divmod(i, 14)
            texto(frame, f"{et}: {n}", (x + col * 110, y + 28 * (fila + 1)),
                  (90, 235, 90) if n >= 240 else (0, 220, 255), 0.6, 1, placa=True)
        return frame


if __name__ == "__main__":
    ap = argumentos("Grabar ejemplos de señas")
    ap.add_argument("--auto", help="graba esta etiqueta al iniciar (útil sin teclado)")
    args = ap.parse_args()
    ejecutar(Grabador(args.auto), args, "Grabar senas")

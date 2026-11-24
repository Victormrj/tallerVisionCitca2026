from __future__ import annotations

import os
import sys
from collections import deque

import cv2
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comun import argumentos, ejecutar, texto  
from manos import dibujar_mano  
from senas.caracteristicas import N_CARACTERISTICAS, RUTA_MODELO, mano_principal, vector  

CONFIANZA_MIN = 0.75
TIEMPO_SOSTENER = 0.8


class Reconocedor:
    def __init__(self, ruta_modelo=RUTA_MODELO):
        if not os.path.exists(ruta_modelo):
            raise SystemExit("No hay modelo. Primero: python senas/grabar.py y "
                             "python senas/entrenar.py")
        datos = joblib.load(ruta_modelo)
        if datos.get("n_caracteristicas") != N_CARACTERISTICAS:
            raise SystemExit("El modelo es de otra versión. Vuelve a correr senas/entrenar.py")
        self.modelo = datos["modelo"]
        self.etiquetas = datos["etiquetas"]
        self.historial = deque(maxlen=8)          # suavizado de probabilidades
        self.escrito = ""
        self.candidata, self.t_candidata = None, 0.0
        self.bloqueada = None                     # última letra escrita (evita repetir)

    def tecla(self, k):
        if k == 32:
            self.escrito += " "
            self.bloqueada = None
        elif k in (8, 127):
            self.escrito = self.escrito[:-1]
        elif k == ord("c"):
            self.escrito = ""

    def actualizar(self, frame, manos, t):
        h, w = frame.shape[:2]
        mano = mano_principal(manos)
        if mano is None:
            self.historial.clear()
            self.candidata, self.bloqueada = None, None
            texto(frame, "Muestra tu mano", (w / 2, h * 0.12), (0, 170, 255), 1.0, 2, centrado=True)
        else:
            probs = self.modelo.predict_proba(vector(mano).reshape(1, -1))[0]
            self.historial.append(probs)
            prom = np.mean(self.historial, axis=0)
            orden = np.argsort(prom)[::-1]
            letra, conf = self.etiquetas[orden[0]], float(prom[orden[0]])
            color = (90, 235, 90) if conf >= CONFIANZA_MIN else (0, 170, 255)
            dibujar_mano(frame, mano, color)

            # letra grande junto a la mano
            x0, y0 = mano.pts.min(axis=0)
            texto(frame, letra, (x0, y0 - 20), color, 2.5, 5)

            # top-3 con barras
            for i, idx in enumerate(orden[:3]):
                y = 40 + 38 * i
                p = float(prom[idx])
                cv2.rectangle(frame, (20, y - 18), (20 + int(220 * p), y + 6), color if i == 0
                              else (150, 150, 150), -1)
                texto(frame, f"{self.etiquetas[idx]}  {p * 100:4.0f}%", (28, y), (255, 255, 255), 0.6, 1)

            # deletreo: sostener la seña
            if conf >= CONFIANZA_MIN and letra != self.bloqueada:
                self.bloqueada = None             
                if letra != self.candidata:
                    self.candidata, self.t_candidata = letra, t
                prog = min(1.0, (t - self.t_candidata) / TIEMPO_SOSTENER)
                c = (int(x0) + 40, int(y0) - 110)
                cv2.ellipse(frame, c, (22, 22), -90, 0, int(360 * prog), (90, 235, 90), 5, cv2.LINE_AA)
                if prog >= 1.0:
                    self.escrito += letra
                    self.bloqueada = letra         
                    self.candidata = None
            else:
                self.candidata = None

        # texto deletreado
        cv2.rectangle(frame, (0, h - 80), (w, h - 30), (20, 20, 20), -1)
        texto(frame, self.escrito[-40:] + "_", (20, h - 42), (255, 255, 255), 1.1, 2)
        texto(frame, "espacio | retroceso borra | c limpia | q sale", (20, h - 10),
              (200, 200, 200), 0.5, 1)
        return frame


if __name__ == "__main__":
    args = argumentos("Reconocer abecedario de señas").parse_args()
    ejecutar(Reconocedor(), args, "Reconocer senas")

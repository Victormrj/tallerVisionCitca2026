"""
manos.py — Detección de manos con MediaPipe (API "Tasks") y utilidades de gestos.

MediaPipe devuelve 21 puntos (landmarks) por mano:
    0 muñeca | 1-4 pulgar | 5-8 índice | 9-12 medio | 13-16 anular | 17-20 meñique
    (4, 8, 12, 16, 20 son las puntas de los dedos)
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

URL_MODELO = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/latest/hand_landmarker.task")
RUTA_MODELO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "modelos", "hand_landmarker.task")

CONEXIONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
PUNTAS = [4, 8, 12, 16, 20]


@dataclass(eq=False)
class Mano:
    pts: np.ndarray       # (21, 2) en píxeles
    lado: str             # "Left"/"Right" según MediaPipe
    score: float
    z: np.ndarray | None = None   # (21,) profundidad relativa (misma escala que x)

    @property
    def tamano(self) -> float:
        """Escala de la mano: distancia muñeca -> nudillo del dedo medio."""
        return float(np.linalg.norm(self.pts[9] - self.pts[0])) + 1e-6

    @property
    def palma(self) -> np.ndarray:
        return self.pts[[0, 5, 9, 13, 17]].mean(axis=0)

    @property
    def punto_pellizco(self) -> np.ndarray:
        return (self.pts[4] + self.pts[8]) / 2

    @property
    def apertura_pellizco(self) -> float:
        """Distancia pulgar-índice normalizada por el tamaño de la mano."""
        return float(np.linalg.norm(self.pts[4] - self.pts[8])) / self.tamano

    def dedos(self) -> list[bool]:
        """[pulgar, índice, medio, anular, meñique] -> True si está extendido."""
        muneca = self.pts[0]
        res = [float(np.linalg.norm(self.pts[4] - self.pts[17]))
               > 1.15 * float(np.linalg.norm(self.pts[3] - self.pts[17]))
               and float(np.linalg.norm(self.pts[4] - self.pts[5])) > 0.45 * self.tamano]
        for punta, art in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            res.append(bool(np.linalg.norm(self.pts[punta] - muneca)
                            > 1.1 * np.linalg.norm(self.pts[art] - muneca)))
        return res

    def dedos_extendidos(self) -> int:
        """Cuenta índice, medio, anular y meñique extendidos (punta más lejos que la articulación)."""
        muneca = self.pts[0]
        n = 0
        for punta, art in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            if np.linalg.norm(self.pts[punta] - muneca) > 1.1 * np.linalg.norm(self.pts[art] - muneca):
                n += 1
        return n


def asegurar_modelo(ruta: str = RUTA_MODELO) -> str:
    if not os.path.exists(ruta):
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        print(f"Descargando modelo de manos de MediaPipe en {ruta} ...")
        urllib.request.urlretrieve(URL_MODELO, ruta)
    return ruta


class DetectorManos:
    def __init__(self, ruta_modelo: str | None = None, num_manos: int = 2,
                 confianza: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        self._mp = mp
        ruta = asegurar_modelo(ruta_modelo or RUTA_MODELO)
        opciones = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=ruta),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_manos,
            min_hand_detection_confidence=confianza,
            min_hand_presence_confidence=confianza,
            min_tracking_confidence=confianza,
        )
        self._det = vision.HandLandmarker.create_from_options(opciones)
        self._ultimo_ts = -1

    def detectar(self, frame_bgr: np.ndarray, ts_ms: int) -> list[Mano]:
        ts_ms = max(int(ts_ms), self._ultimo_ts + 1)  # MediaPipe exige tiempos crecientes
        self._ultimo_ts = ts_ms
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        imagen = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self._det.detect_for_video(imagen, ts_ms)
        manos = []
        for lms, hd in zip(res.hand_landmarks, res.handedness):
            pts = np.array([[p.x * w, p.y * h] for p in lms], dtype=float)
            z = np.array([p.z * w for p in lms], dtype=float)
            manos.append(Mano(pts, hd[0].category_name, hd[0].score, z))
        return manos

    def cerrar(self):
        try:
            self._det.close()
        except Exception:
            pass


def dibujar_mano(img, mano: Mano, color, etiqueta: str = "", resaltar_puntas=False):
    p = mano.pts.astype(int)
    for a, b in CONEXIONES:
        cv2.line(img, tuple(p[a]), tuple(p[b]), color, 2, cv2.LINE_AA)
    for i, q in enumerate(p):
        cv2.circle(img, tuple(q), 4, color, -1, cv2.LINE_AA)
    if resaltar_puntas:
        r = max(6, int(mano.tamano * 0.12))
        for i in PUNTAS[1:]:
            cv2.circle(img, tuple(p[i]), r, color, 2, cv2.LINE_AA)
    if etiqueta:
        org = (int(p[0][0]) - 30, int(p[0][1]) + 28)
        cv2.putText(img, etiqueta, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

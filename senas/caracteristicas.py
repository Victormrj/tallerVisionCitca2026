from __future__ import annotations

import csv
import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CARPETA = os.path.dirname(os.path.abspath(__file__))
RUTA_DATOS = os.path.join(CARPETA, "datos", "muestras.csv")
RUTA_MODELO = os.path.join(CARPETA, "modelo", "modelo.joblib")

PUNTAS = [4, 8, 12, 16, 20]
PARES_PUNTAS = list(combinations(PUNTAS, 2))
ESPEJAR_IZQUIERDA = False   
N_CARACTERISTICAS = 21 * 2 + 21 + len(PARES_PUNTAS)   # 73


def vector(mano) -> np.ndarray:
    p = mano.pts.astype(float).copy()
    z = mano.z.astype(float).copy() if mano.z is not None else np.zeros(21)
    p -= p[0]
    z -= z[0]
    if ESPEJAR_IZQUIERDA and mano.lado == "Left":
        p[:, 0] *= -1
    escala = np.linalg.norm(p[9]) + 1e-6
    p /= escala
    z /= escala
    distancias = [np.linalg.norm(p[a] - p[b]) for a, b in PARES_PUNTAS]
    return np.concatenate([p.ravel(), z, distancias]).astype(np.float32)


def mano_principal(manos):
    """Si hay varias manos, usamos la más grande (la más cercana a la cámara)."""
    return max(manos, key=lambda m: m.tamano) if manos else None


def guardar_muestras(etiqueta: str, vectores, ruta: str = RUTA_DATOS):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    nuevo = not os.path.exists(ruta)
    with open(ruta, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["etiqueta"] + [f"f{i}" for i in range(N_CARACTERISTICAS)])
        for v in vectores:
            w.writerow([etiqueta] + [f"{x:.5f}" for x in v])


def cargar_muestras(ruta: str = RUTA_DATOS):
    if not os.path.exists(ruta):
        return np.zeros((0, N_CARACTERISTICAS), np.float32), np.array([], dtype=str)
    etiquetas, filas = [], []
    with open(ruta, encoding="utf-8") as f:
        r = csv.reader(f)
        next(r, None)
        for fila in r:
            if len(fila) == N_CARACTERISTICAS + 1:
                etiquetas.append(fila[0])
                filas.append([float(x) for x in fila[1:]])
    return np.array(filas, np.float32).reshape(-1, N_CARACTERISTICAS), np.array(etiquetas)


def quitar_ultimas(n: int, ruta: str = RUTA_DATOS):
    """Borra las últimas n muestras (para deshacer una grabación mala)."""
    if not os.path.exists(ruta):
        return
    with open(ruta, encoding="utf-8") as f:
        lineas = f.readlines()
    encabezado, datos = lineas[:1], lineas[1:]
    with open(ruta, "w", encoding="utf-8") as f:
        f.writelines(encabezado + datos[:max(0, len(datos) - n)])

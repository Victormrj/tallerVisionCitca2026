"""
render.py — Dibuja el cubo en 3D sobre un frame de OpenCV (sin OpenGL).

Técnica:
  1. Orientación global R (matriz 3x3) controlada por la mano izquierda.
  2. Proyección en perspectiva simple (cámara en el eje +z).
  3. "Algoritmo del pintor": se divide el cubo en 3 rebanadas a lo largo del eje
     que se está girando; cada rebanada es una caja convexa, así que basta con
     dibujar las rebanadas de la más lejana a la más cercana y, dentro de cada
     una, sólo las caras exteriores orientadas hacia la cámara.
"""
from __future__ import annotations

import cv2
import numpy as np

from cubo import COLORES, Animador, Cubo, Movimiento, matriz_rotacion

SHIFT = 4                 # precisión sub-píxel para cv2.fillPoly (2^4 = 16)
ESC = 1 << SHIFT
COLOR_CUERPO = (22, 22, 22)
LUZ = np.array([-0.35, 0.55, 0.75])
LUZ = LUZ / np.linalg.norm(LUZ)


def _cuadrado_redondeado(medio: float, radio: float, pts_esquina: int = 5) -> np.ndarray:
    """Contorno 2D (N,2) de un cuadrado con esquinas redondeadas, centrado en 0."""
    puntos = []
    centros = [(medio - radio, medio - radio), (-medio + radio, medio - radio),
               (-medio + radio, -medio + radio), (medio - radio, -medio + radio)]
    for i, (cx, cy) in enumerate(centros):
        for k in range(pts_esquina):
            a = (i + k / (pts_esquina - 1)) * np.pi / 2
            puntos.append((cx + radio * np.cos(a), cy + radio * np.sin(a)))
    return np.array(puntos)


FORMA_STICKER = _cuadrado_redondeado(0.43, 0.12)
FORMA_BRILLO = _cuadrado_redondeado(0.27, 0.09)
FORMA_CARA = np.array([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]])


def _tangentes(n: np.ndarray):
    """Dos vectores unitarios perpendiculares a la normal entera n."""
    eje = int(np.argmax(np.abs(n)))
    u = np.zeros(3); v = np.zeros(3)
    u[(eje + 1) % 3] = 1
    v[(eje + 2) % 3] = 1
    return u, v


def _sombrear(color, normal_cam, factor_min=0.62):
    k = factor_min + (1 - factor_min) * max(0.0, float(np.dot(normal_cam, LUZ)))
    return tuple(int(min(255, c * k)) for c in color)


def _aclarar(color, k=0.35):
    return tuple(int(c + (255 - c) * k) for c in color)


class Renderizador:
    def __init__(self, centro=(0.7, 0.42), escala=0.11, distancia=9.0):
        """
        centro: posición del cubo en coordenadas normalizadas del frame.
        escala: tamaño de un cubito en fracción de la altura del frame.
        """
        self.centro_rel = centro
        self.escala_rel = escala
        self.D = distancia
        self.reiniciar_orientacion()
        self.poligonos_stickers = []   # para "hit-testing" del cursor
        self.radio_px = 1.0

    def reiniciar_orientacion(self):
        self.R = matriz_rotacion(0, np.radians(28)) @ matriz_rotacion(1, np.radians(-38))

    def rotar(self, yaw: float, pitch: float):
        """Gira el cubo en el marco de la cámara (como un trackball)."""
        self.R = matriz_rotacion(1, yaw) @ matriz_rotacion(0, pitch) @ self.R
        # Re-ortonormalizar para evitar acumulación de error numérico
        u, _, vt = np.linalg.svd(self.R)
        self.R = u @ vt

    # ------------------------------------------------------------------ proyección
    def configurar(self, frame):
        h, w = frame.shape[:2]
        self.cx = self.centro_rel[0] * w
        self.cy = self.centro_rel[1] * h
        self.f = self.escala_rel * h * self.D
        self.radio_px = 1.5 * self.escala_rel * h * 1.25

    def a_camara(self, p_cubo: np.ndarray) -> np.ndarray:
        return p_cubo @ self.R.T

    def proyectar(self, p_cam: np.ndarray) -> np.ndarray:
        """(N,3) coordenadas de cámara -> (N,2) píxeles."""
        p_cam = np.atleast_2d(p_cam)
        k = self.f / (self.D - p_cam[:, 2])
        return np.stack([self.cx + p_cam[:, 0] * k, self.cy - p_cam[:, 1] * k], axis=1)

    def proyectar_cubo(self, p_cubo: np.ndarray) -> np.ndarray:
        return self.proyectar(self.a_camara(np.atleast_2d(p_cubo)))

    @property
    def centro_px(self):
        return np.array([self.cx, self.cy])

    # ------------------------------------------------------------------ dibujo
    def _poligono(self, img, pts2d, color):
        cv2.fillPoly(img, [np.round(pts2d * ESC).astype(np.int32)], color,
                     lineType=cv2.LINE_AA, shift=SHIFT)

    def _contorno(self, img, pts2d, color, grosor):
        cv2.polylines(img, [np.round(pts2d * ESC).astype(np.int32)], True, color,
                      grosor, lineType=cv2.LINE_AA, shift=SHIFT)

    def dibujar(self, img, cubo: Cubo, animador: Animador | None = None,
                resaltado=None, color_resaltado=(0, 170, 255)):
        """
        resaltado: tupla (cubito, sticker) a remarcar, o None.
        Devuelve nada; deja en self.poligonos_stickers la lista para hit-testing.
        """
        self.configurar(img)
        anim = animador.rotacion_en_curso() if animador else None
        mov: Movimiento | None = anim[0] if anim else None
        eje = mov.eje if mov else 0
        rot_capa = matriz_rotacion(mov.eje, anim[1]) if anim else None

        camara = np.array([0.0, 0.0, self.D])
        dir_eje = self.R[:, eje]                 # dirección del eje en cámara
        t_cam = float(camara @ dir_eje)
        rebanadas = sorted((-1, 0, 1), key=lambda l: -abs(t_cam - l))

        self.poligonos_stickers = []
        for capa in rebanadas:
            girando = mov is not None and capa == mov.capa
            for cub in cubo.cubitos:
                if cub.pos[eje] != capa:
                    continue
                M = self.R @ rot_capa if girando else self.R
                stickers = {tuple(st.normal): st for st in cub.stickers}
                for d in range(6):
                    n = np.zeros(3, dtype=int)
                    n[d // 2] = 1 if d % 2 == 0 else -1
                    a = d // 2
                    # sólo caras exteriores de la rebanada (convexa)
                    if a != eje and cub.pos[a] * n[a] != 1:
                        continue
                    centro = cub.pos + 0.5 * n
                    centro_cam = M @ centro
                    n_cam = M @ n
                    if np.dot(n_cam, camara - centro_cam) <= 0:
                        continue  # cara de espaldas
                    u, v = _tangentes(n)
                    # cara negra del cubito
                    cara3 = centro + FORMA_CARA[:, :1] * u + FORMA_CARA[:, 1:] * v
                    self._poligono(img, self.proyectar(cara3 @ M.T), COLOR_CUERPO)
                    st = stickers.get(tuple(n))
                    if st is None:
                        continue
                    base = COLORES[st.color]
                    c_st = _sombrear(base, n_cam)
                    sup = centro + 0.002 * n
                    pts = self.proyectar((sup + FORMA_STICKER[:, :1] * u
                                          + FORMA_STICKER[:, 1:] * v) @ M.T)
                    self._poligono(img, pts, c_st)
                    brillo = self.proyectar((sup + FORMA_BRILLO[:, :1] * u
                                             + FORMA_BRILLO[:, 1:] * v) @ M.T)
                    self._poligono(img, brillo, _aclarar(c_st, 0.18))
                    es_resaltado = (resaltado is not None and resaltado[0] is cub
                                    and resaltado[1] is st)
                    if es_resaltado:
                        self._contorno(img, pts, (20, 20, 20), 7)
                        self._contorno(img, pts, color_resaltado, 3)
                    self.poligonos_stickers.append((pts.astype(np.float32), cub, st))

    def sticker_bajo(self, punto):
        """Sticker visible bajo un punto (x, y) en píxeles, o None."""
        if punto is None:
            return None
        p = (float(punto[0]), float(punto[1]))
        # Las caras visibles de un cuerpo convexo no se traslapan; el último dibujado
        # es el más cercano en caso de duda.
        for pts, cub, st in reversed(self.poligonos_stickers):
            if cv2.pointPolygonTest(pts, p, False) >= 0:
                return cub, st
        return None

    def elegir_giro(self, cub, st, arrastre) -> Movimiento | None:
        """
        Dado un sticker seleccionado y un vector de arrastre en pantalla, elige el
        giro de capa cuyo movimiento en pantalla se parezca más al arrastre.
        """
        arrastre = np.asarray(arrastre, dtype=float)
        if np.linalg.norm(arrastre) < 1e-6:
            return None
        arrastre = arrastre / np.linalg.norm(arrastre)
        c3 = cub.pos + 0.5 * st.normal          # centro del sticker (coords. cubo)
        p0 = self.proyectar_cubo(c3)[0]
        mejor, mejor_score = None, -2.0
        for eje in range(3):
            if st.normal[eje] != 0:
                continue  # no se puede girar alrededor de la normal del propio sticker
            for sentido in (1, -1):
                w = np.zeros(3); w[eje] = sentido
                vel = np.cross(w, c3)                 # velocidad tangencial 3D
                p1 = self.proyectar_cubo(c3 + 0.05 * vel)[0]
                v2 = p1 - p0
                nv = np.linalg.norm(v2)
                if nv < 1e-6:
                    continue
                score = float(np.dot(v2 / nv, arrastre))
                if score > mejor_score:
                    mejor_score = score
                    mejor = Movimiento(eje, int(cub.pos[eje]), sentido)
        return mejor

    def flecha_pista(self, mov: Movimiento):
        """
        Para un giro sugerido, elige un sticker visible de esa capa y devuelve
        (contorno, inicio, fin): arrastrar ese sticker de 'inicio' hacia 'fin' produce el giro.
        Devuelve None si desde la vista actual no se ve ningún sticker útil de la capa.
        Usa los stickers del último dibujar().
        """
        mejor, mejor_area = None, 0.0
        for pts, cub, st in self.poligonos_stickers:
            if cub.pos[mov.eje] != mov.capa or st.normal[mov.eje] != 0:
                continue                      # no es de la capa, o gira sobre sí mismo
            c3 = cub.pos + 0.5 * st.normal
            w = np.zeros(3); w[mov.eje] = mov.sentido
            p0 = self.proyectar_cubo(c3)[0]
            v = self.proyectar_cubo(c3 + 0.05 * np.cross(w, c3))[0] - p0
            n = np.linalg.norm(v)
            if n < 1e-6 or self.elegir_giro(cub, st, v) != mov:
                continue                      # el arrastre sería ambiguo desde esta vista
            area = float(cv2.contourArea(pts))
            if area > mejor_area:
                largo = 0.42 * self.radio_px
                mejor, mejor_area = (pts, p0 - v / n * largo * 0.25, p0 + v / n * largo), area
        return mejor

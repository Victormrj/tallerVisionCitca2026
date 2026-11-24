from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np

# Colores en formato BGR (OpenCV)
COLORES = {
    "U": (245, 245, 245),  # blanco   (arriba)
    "D": (0, 215, 255),    # amarillo (abajo)
    "F": (70, 195, 60),    # verde    (frente)
    "B": (215, 115, 25),   # azul     (atrás)
    "R": (50, 45, 215),    # rojo     (derecha)
    "L": (20, 135, 255),   # naranja  (izquierda)
}

NORMALES_CARAS = {
    "U": (0, 1, 0), "D": (0, -1, 0),
    "R": (1, 0, 0), "L": (-1, 0, 0),
    "F": (0, 0, 1), "B": (0, 0, -1),
}

EJES = "xyz"


class _Reloj:
    """Reloj global. En tiempo real usa time.monotonic(); al procesar un video
    desde archivo se fija un tiempo simulado (frame / fps)."""
    simulado: float | None = None

    def __call__(self) -> float:
        return self.simulado if self.simulado is not None else time.monotonic()


reloj = _Reloj()


def matriz_giro90(eje: int, sentido: int) -> np.ndarray:
    """Matriz entera de rotación de sentido*90° alrededor del eje (0=x, 1=y, 2=z)."""
    c, s = 0, sentido  # cos(±90°)=0, sin(±90°)=±1
    if eje == 0:
        m = [[1, 0, 0], [0, c, -s], [0, s, c]]
    elif eje == 1:
        m = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    else:
        m = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return np.array(m, dtype=int)


def matriz_rotacion(eje: int, angulo: float) -> np.ndarray:
    """Matriz de rotación (float) de 'angulo' radianes alrededor de un eje."""
    c, s = np.cos(angulo), np.sin(angulo)
    if eje == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if eje == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@dataclass
class Sticker:
    normal: np.ndarray  # vector entero (3,)
    color: str          # clave de COLORES ("U", "D", ...)


@dataclass
class Cubito:
    pos: np.ndarray                     # vector entero (3,)
    stickers: list = field(default_factory=list)


@dataclass(frozen=True)
class Movimiento:
    eje: int      # 0=x, 1=y, 2=z
    capa: int     # -1, 0, 1
    sentido: int  # +1 / -1 (regla de la mano derecha sobre el eje positivo)

    def inverso(self) -> "Movimiento":
        return Movimiento(self.eje, self.capa, -self.sentido)

    def notacion(self) -> str:
        """Notación estándar (R, U', M, ...)."""
        if self.capa == 0:
            # M sigue a L (eje x), E sigue a D (eje y), S sigue a F (eje z)
            letra = "MES"[self.eje]
            normal = {0: +1, 1: +1, 2: -1}[self.eje]
        else:
            letra = {(0, 1): "R", (0, -1): "L", (1, 1): "U",
                     (1, -1): "D", (2, 1): "F", (2, -1): "B"}[(self.eje, self.capa)]
            # Horario visto desde la cara = -90° sobre su normal exterior
            normal = -self.capa
        return letra if self.sentido == normal else letra + "'"


# Movimientos por notación, útil para mezclar o para ejercicios del taller
MOVIMIENTOS = {}
for _eje in range(3):
    for _capa in (-1, 0, 1):
        for _s in (1, -1):
            _m = Movimiento(_eje, _capa, _s)
            MOVIMIENTOS[_m.notacion()] = _m


class Cubo:
    def __init__(self):
        self.reiniciar()

    def reiniciar(self):
        self.cubitos: list[Cubito] = []
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                for z in (-1, 0, 1):
                    if x == y == z == 0:
                        continue
                    pos = np.array([x, y, z], dtype=int)
                    cub = Cubito(pos)
                    for clave, n in NORMALES_CARAS.items():
                        n = np.array(n, dtype=int)
                        if np.dot(pos, n) == 1:  # la cara exterior de este cubito
                            cub.stickers.append(Sticker(n, clave))
                    self.cubitos.append(cub)
        self.historial: list[Movimiento] = []

    def cubitos_en_capa(self, eje: int, capa: int):
        return [c for c in self.cubitos if c.pos[eje] == capa]

    def aplicar(self, mov: Movimiento, registrar: bool = True):
        m = matriz_giro90(mov.eje, mov.sentido)
        for c in self.cubitos_en_capa(mov.eje, mov.capa):
            c.pos = m @ c.pos
            for st in c.stickers:
                st.normal = m @ st.normal
        if registrar:
            self.historial.append(mov)

    def aplicar_secuencia(self, texto: str):
        """Aplica una secuencia en notación estándar, p. ej. "R U R' U'"."""
        for t in texto.split():
            if t.endswith("2"):
                self.aplicar(MOVIMIENTOS[t[:-1]])
                self.aplicar(MOVIMIENTOS[t[:-1]])
            else:
                self.aplicar(MOVIMIENTOS[t])

    def resuelto(self) -> bool:
        por_cara: dict[tuple, set] = {}
        for c in self.cubitos:
            for st in c.stickers:
                por_cara.setdefault(tuple(st.normal), set()).add(st.color)
        return all(len(v) == 1 for v in por_cara.values())

    def estado(self) -> dict:
        """Diccionario {(pos, normal): color}; útil para comparar estados en pruebas."""
        return {(tuple(c.pos), tuple(st.normal)): st.color
                for c in self.cubitos for st in c.stickers}

    def mezcla_aleatoria(self, n: int = 20, semilla: int | None = None) -> list[Movimiento]:
        rng = random.Random(semilla)
        movs, ultimo_eje = [], None
        caras = [m for m in MOVIMIENTOS.values() if m.capa != 0]
        while len(movs) < n:
            m = rng.choice(caras)
            if m.eje == ultimo_eje:
                continue
            movs.append(m)
            ultimo_eje = m.eje
        return movs


class Camino:
   
    def __init__(self, movimientos=()):
        self.pila: list[Movimiento] = list(movimientos)

    def reiniciar(self, movimientos=()):
        self.pila = list(movimientos)

    def agregar(self, mov: Movimiento):
        if self.pila and self.pila[-1] == mov.inverso():
            self.pila.pop()
        else:
            self.pila.append(mov)

    def pista(self) -> Movimiento | None:
        return self.pila[-1].inverso() if self.pila else None

    def __len__(self) -> int:
        return len(self.pila)


class Animador:
    """Cola de movimientos animados. Aplica el giro lógico al terminar la animación."""

    def __init__(self, cubo: Cubo, duracion: float = 0.28):
        self.cubo = cubo
        self.duracion = duracion
        self.cola: list[tuple[Movimiento, bool, float]] = []
        self.actual: Movimiento | None = None
        self._registrar = True
        self._t0 = 0.0
        self._dur = duracion

    def encolar(self, mov: Movimiento, registrar: bool = True, duracion: float | None = None):
        self.cola.append((mov, registrar, self.duracion if duracion is None else duracion))

    @property
    def ocupado(self) -> bool:
        return self.actual is not None or bool(self.cola)

    def actualizar(self, t: float | None = None):
        t = reloj() if t is None else t
        if self.actual is not None and t - self._t0 >= self._dur:
            self.cubo.aplicar(self.actual, registrar=self._registrar)
            self.actual = None
        if self.actual is None and self.cola:
            self.actual, self._registrar, self._dur = self.cola.pop(0)
            self._t0 = t

    def rotacion_en_curso(self, t: float | None = None):
        """Devuelve (movimiento, ángulo en radianes) de la animación actual, o None."""
        if self.actual is None:
            return None
        t = reloj() if t is None else t
        p = min(1.0, max(0.0, (t - self._t0) / self._dur))
        p = p * p * (3 - 2 * p)  # suavizado (smoothstep)
        return self.actual, self.actual.sentido * p * np.pi / 2

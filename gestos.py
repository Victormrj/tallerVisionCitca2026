"""
gestos.py — Lógica de interacción (máquinas de estado).

Mano de ROTACIÓN (fuera del recuadro):
    palma abierta + mover  -> gira todo el cubo
    puño                   -> "embrague": puedes reposicionar la mano sin girar

Mano TOUCHPAD (dentro del recuadro "CONTROL POR GESTOS"):
    INACTIVO  -> no hay mano en el recuadro
    APUNTANDO -> el punto entre pulgar e índice mueve un cursor sobre el cubo
    FIJADO    -> pellizcas sobre un sticker: queda seleccionado
                 * arrastra en una dirección -> gira esa capa
                 * suelta sin arrastrar      -> cancela
    ENFRIANDO -> breve pausa tras un giro (evita giros dobles)
    SOLTAR    -> espera a que abras el pellizco para volver a apuntar
"""
from __future__ import annotations

import numpy as np

from cubo import reloj
from manos import Mano

INACTIVO, APUNTANDO, FIJADO, ENFRIANDO, SOLTAR = (
    "INACTIVO", "APUNTANDO", "FIJADO", "ENFRIANDO", "SOLTAR")

# Colores BGR por estado (recuadro y textos)
COLOR_ESTADO = {
    INACTIVO: (255, 190, 70),   # azul claro
    APUNTANDO: (0, 170, 255),   # naranja
    FIJADO: (90, 235, 90),      # verde
    ENFRIANDO: (120, 90, 255),  # rosa/rojo
    SOLTAR: (255, 190, 70),
}
TEXTO_ESTADO = {
    INACTIVO: "mete tu mano derecha al recuadro",
    APUNTANDO: "APUNTANDO - pellizca un sticker para fijarlo",
    FIJADO: "FIJADO - arrastra para girar, suelta para cancelar",
    ENFRIANDO: "ENFRIANDO",
    SOLTAR: "suelta el pellizco para continuar",
}


class Suavizador:
    """Filtro exponencial (EMA) para reducir el temblor de los landmarks."""

    def __init__(self, alfa=0.5):
        self.alfa, self.valor = alfa, None

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        self.valor = x if self.valor is None else self.alfa * x + (1 - self.alfa) * self.valor
        return self.valor

    def reiniciar(self):
        self.valor = None


class ControlRotacion:
    def __init__(self, sensibilidad=3.2, inercia=0.86):
        self.sens = sensibilidad      # radianes por "ancho de frame" de movimiento
        self.inercia = inercia
        self.suave = Suavizador(0.55)
        self.prev = None
        self.vel = np.zeros(2)        # (yaw, pitch) por frame
        self.estado = "sin mano"

    def actualizar(self, mano: Mano | None, ancho_frame: int) -> tuple[float, float]:
        if mano is None or mano.dedos_extendidos() < 3:
            self.estado = "sin mano" if mano is None else "puño (pausa)"
            self.prev = None
            self.suave.reiniciar()
            self.vel *= self.inercia            # sigue girando un poco y se frena
            if np.abs(self.vel).max() < 1e-4:
                self.vel[:] = 0
            return float(self.vel[0]), float(self.vel[1])

        self.estado = "girando"
        p = self.suave(mano.palma)
        if self.prev is None:
            self.prev = p
            return 0.0, 0.0
        d = (p - self.prev) / ancho_frame
        self.prev = p
        if np.linalg.norm(d) < 0.0015:          # zona muerta contra el temblor
            d = np.zeros(2)
        self.vel = np.array([d[0], d[1]]) * self.sens
        return float(self.vel[0]), float(self.vel[1])


class ControlTouchpad:
    def __init__(self, caja=(0.08, 0.20, 0.47, 0.80), pellizco_on=0.30, pellizco_off=0.45,
                 umbral_arrastre=0.55, enfriamiento=0.35):
        """
        caja: (x0, y0, x1, y1) del recuadro en coordenadas normalizadas del frame.
        umbral_arrastre: distancia para disparar el giro, en "tamaños de cubito" en pantalla.
        """
        self.caja = caja
        self.p_on, self.p_off = pellizco_on, pellizco_off
        self.umbral_rel = umbral_arrastre
        self.enfriamiento = enfriamiento
        self.estado = INACTIVO
        self.cursor = None
        self.suave = Suavizador(0.5)
        self.pellizcando = False
        self.hover = None            # (cubito, sticker) bajo el cursor
        self.seleccion = None        # (cubito, sticker) fijado
        self.inicio = None
        self.progreso = 0.0
        self._t_estado = reloj()
        self._t_ultima_mano = 0.0

    # ----------------------------------------------------------------- utilidades
    def caja_px(self, w, h):
        x0, y0, x1, y1 = self.caja
        return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)

    def contiene(self, punto, w, h, margen=0.06) -> bool:
        x0, y0, x1, y1 = self.caja
        x, y = punto[0] / w, punto[1] / h
        return x0 - margen <= x <= x1 + margen and y0 - margen <= y <= y1 + margen

    def _cambiar(self, estado):
        self.estado = estado
        self._t_estado = reloj()

    def _mapear_cursor(self, punto, w, h, render) -> np.ndarray:
        """Del recuadro (touchpad) a un área cuadrada alrededor del cubo en pantalla."""
        x0, y0, x1, y1 = self.caja
        u = np.clip((punto[0] / w - x0) / (x1 - x0), 0, 1)
        v = np.clip((punto[1] / h - y0) / (y1 - y0), 0, 1)
        r = render.radio_px * 1.1
        return render.centro_px + np.array([(u - 0.5) * 2 * r, (v - 0.5) * 2 * r])

    # ----------------------------------------------------------------- bucle
    def actualizar(self, mano: Mano | None, w, h, render, animador):
        """Devuelve un Movimiento si se debe girar una capa, o None."""
        ahora = reloj()
        if mano is None:
            if ahora - self._t_ultima_mano > 0.35:
                self._cambiar(INACTIVO) if self.estado != INACTIVO else None
                self.cursor, self.hover, self.seleccion = None, None, None
                self.pellizcando = False
                self.suave.reiniciar()
            return None
        self._t_ultima_mano = ahora

        # Pellizco con histéresis (umbral distinto para cerrar y abrir)
        a = mano.apertura_pellizco
        if self.pellizcando and a > self.p_off:
            self.pellizcando = False
        elif not self.pellizcando and a < self.p_on:
            self.pellizcando = True

        self.cursor = self.suave(self._mapear_cursor(mano.punto_pellizco, w, h, render))
        if self.estado == INACTIVO:
            self._cambiar(SOLTAR if self.pellizcando else APUNTANDO)

        if self.estado == APUNTANDO:
            self.hover = render.sticker_bajo(self.cursor)
            if self.pellizcando:
                if self.hover is not None and not animador.ocupado:
                    self.seleccion = self.hover
                    self.inicio = self.cursor.copy()
                    self.progreso = 0.0
                    self._cambiar(FIJADO)
                else:
                    self._cambiar(SOLTAR)

        elif self.estado == FIJADO:
            if not self.pellizcando:
                self.seleccion = None
                self._cambiar(APUNTANDO)
                return None
            arrastre = self.cursor - self.inicio
            umbral = self.umbral_rel * render.radio_px / 1.9
            self.progreso = min(1.0, float(np.linalg.norm(arrastre)) / umbral)
            if self.progreso >= 1.0:
                mov = render.elegir_giro(*self.seleccion, arrastre)
                self.seleccion = None
                self._cambiar(ENFRIANDO)
                return mov

        elif self.estado == ENFRIANDO:
            if ahora - self._t_estado >= self.enfriamiento and not animador.ocupado:
                self._cambiar(SOLTAR if self.pellizcando else APUNTANDO)

        elif self.estado == SOLTAR:
            self.hover = None
            if not self.pellizcando:
                self._cambiar(APUNTANDO)
        return None

    @property
    def resaltado(self):
        """(cubito, sticker) a resaltar y su color."""
        if self.estado == FIJADO and self.seleccion:
            return self.seleccion, COLOR_ESTADO[FIJADO]
        if self.estado == APUNTANDO and self.hover:
            return self.hover, COLOR_ESTADO[APUNTANDO]
        return None, None

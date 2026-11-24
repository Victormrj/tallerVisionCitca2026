import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cubo import MOVIMIENTOS, Cubo, Movimiento  # noqa: E402
from render import Renderizador  # noqa: E402


def test_inicial_resuelto():
    c = Cubo()
    assert len(c.cubitos) == 26
    assert sum(len(x.stickers) for x in c.cubitos) == 54
    assert c.resuelto()


def test_cuatro_giros_identidad():
    for m in MOVIMIENTOS.values():
        c = Cubo()
        inicial = c.estado()
        for _ in range(4):
            c.aplicar(m)
        assert c.estado() == inicial, m.notacion()


def test_giro_y_su_inverso():
    c = Cubo()
    inicial = c.estado()
    c.aplicar(MOVIMIENTOS["R"])
    assert not c.resuelto()
    c.aplicar(MOVIMIENTOS["R"].inverso())
    assert c.estado() == inicial


def test_sexy_move_orden_6():
    c = Cubo()
    for _ in range(6):
        c.aplicar_secuencia("R U R' U'")
    assert c.resuelto()


def test_mezcla_y_deshacer():
    c = Cubo()
    movs = c.mezcla_aleatoria(25, semilla=1)
    for m in movs:
        c.aplicar(m)
    assert not c.resuelto()
    for m in reversed(movs):
        c.aplicar(m.inverso())
    assert c.resuelto()


def test_R_horario_sube_frente():
    """R (horario visto desde la derecha) lleva el sticker frontal derecho hacia arriba."""
    c = Cubo()
    c.aplicar(MOVIMIENTOS["R"])
    cub = next(x for x in c.cubitos if tuple(x.pos) == (1, 1, 0))
    colores = {tuple(s.normal): s.color for s in cub.stickers}
    assert colores[(0, 1, 0)] == "F"


def test_elegir_giro_coherente():
    """Arrastrar hacia arriba un sticker de la columna derecha del frente = R."""
    c = Cubo()
    r = Renderizador()
    r.R = np.eye(3)                      # vista frontal
    r.configurar(np.zeros((720, 1280, 3), np.uint8))
    cub = next(x for x in c.cubitos if tuple(x.pos) == (1, 0, 1))
    st = next(s for s in cub.stickers if tuple(s.normal) == (0, 0, 1))
    mov = r.elegir_giro(cub, st, (0, -50))   # arriba en pantalla (y negativa)
    assert mov.notacion() == "R"
    mov = r.elegir_giro(cub, st, (-50, 0))   # izquierda: fila de en medio
    assert mov == Movimiento(1, 0, -1)       # E' (la fila de en medio va a la izquierda)


# ---------------------------------------------------------------- gestos (manos sintéticas)
from cubo import Animador, reloj  # noqa: E402
from gestos import APUNTANDO, FIJADO, ControlRotacion, ControlTouchpad  # noqa: E402
from manos import Mano  # noqa: E402


from tests.util import mano_sintetica  # noqa: E402


def test_rotacion_palma_abierta_y_puno():
    rot = ControlRotacion(inercia=0.0)
    rot.actualizar(mano_sintetica((900, 400)), 1280)
    yaw, pitch = rot.actualizar(mano_sintetica((960, 400)), 1280)
    assert yaw > 0 and abs(pitch) < 1e-9          # mover a la derecha => yaw positivo
    yaw, pitch = rot.actualizar(mano_sintetica((960, 400), abierta=False), 1280)
    assert rot.estado.startswith("pu") and yaw == 0 and pitch == 0


def test_touchpad_fijar_y_girar():
    w, h = 1280, 720
    frame = np.zeros((h, w, 3), np.uint8)
    cubo = Cubo(); anim = Animador(cubo); r = Renderizador()
    tp = ControlTouchpad()
    reloj.simulado = 0.0
    r.dibujar(frame, cubo, anim)
    # Busca un punto del recuadro cuyo cursor caiga sobre un sticker
    x0, y0, x1, y1 = tp.caja_px(w, h)
    objetivo = None
    for fx in np.linspace(0.3, 0.7, 9):
        for fy in np.linspace(0.3, 0.7, 9):
            pt = np.array([x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)])
            if r.sticker_bajo(tp._mapear_cursor(pt, w, h, r)) is not None:
                objetivo = pt
                break
        if objetivo is not None:
            break
    assert objetivo is not None
    # coloca la mano de modo que su punto de pellizco quede en 'objetivo'
    desplaz = -mano_sintetica((0, 0)).punto_pellizco
    for _ in range(5):
        tp.actualizar(mano_sintetica(objetivo + desplaz), w, h, r, anim)
    assert tp.estado == APUNTANDO and tp.hover is not None
    desplaz = -mano_sintetica((0, 0), pellizco=True).punto_pellizco
    for _ in range(3):
        tp.actualizar(mano_sintetica(objetivo + desplaz, pellizco=True), w, h, r, anim)
    assert tp.estado == FIJADO
    mov = None
    for k in range(1, 30):
        reloj.simulado = k * 0.033
        mov = tp.actualizar(mano_sintetica(objetivo + desplaz + (0, -6 * k), pellizco=True),
                            w, h, r, anim) or mov
    reloj.simulado = None
    assert mov is not None


# ---------------------------------------------------------------- mezcla inicial y pistas
from cubo import Camino  # noqa: E402


def test_camino_resuelve_siguiendo_pistas():
    import random
    rng = random.Random(7)
    c = Cubo()
    cam = Camino()
    for m in c.mezcla_aleatoria(20, semilla=3):
        c.aplicar(m, registrar=False)
        cam.agregar(m)
    assert not c.resuelto() and len(cam) == 20 and c.historial == []
    # el jugador se desvía con giros al azar y luego sigue las pistas
    for _ in range(6):
        m = rng.choice(list(MOVIMIENTOS.values()))
        c.aplicar(m)
        cam.agregar(m)
    pasos = 0
    while cam.pista() is not None:
        m = cam.pista()
        c.aplicar(m)
        cam.agregar(m)
        pasos += 1
        assert pasos <= 26
    assert c.resuelto()


def test_flecha_pista_produce_el_giro_sugerido():
    """Arrastrar el sticker señalado en la dirección de la flecha da exactamente la pista."""
    frame = np.zeros((720, 1280, 3), np.uint8)
    for semilla in range(5):
        c = Cubo()
        cam = Camino()
        r = Renderizador()
        for m in c.mezcla_aleatoria(20, semilla=semilla):
            c.aplicar(m, registrar=False)
            cam.agregar(m)
        while cam.pista() is not None:
            mov = cam.pista()
            r.dibujar(frame, c)
            flecha = r.flecha_pista(mov)
            assert flecha is not None            # en la vista inicial siempre hay un sticker útil
            pts, p0, p1 = flecha
            centro = pts.mean(axis=0)
            cub, st = r.sticker_bajo(centro)
            assert r.elegir_giro(cub, st, p1 - p0) == mov
            c.aplicar(mov)
            cam.agregar(mov)
        assert c.resuelto()

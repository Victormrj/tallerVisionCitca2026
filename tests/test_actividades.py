import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.util import mano_sintetica  # noqa: E402

W, H = 1280, 720


def frame():
    return np.full((H, W, 3), 90, np.uint8)


# ------------------------------------------------------------------ pizarra
def test_pizarra_dibuja_selecciona_y_deshace():
    from pizarra import COLORES, Pizarra, clasificar
    pz = Pizarra()
    solo_indice = dict(dedos=(True, False, False, False))
    assert clasificar(mano_sintetica((600, 500), **solo_indice)) == "dibujar"
    assert clasificar(mano_sintetica((600, 500), dedos=(True, True, False, False))) == "seleccionar"
    assert clasificar(mano_sintetica((600, 500))) == "mover"

    t = 0.0
    for k in range(20):                                   # trazo horizontal
        t += 0.033
        pz.actualizar(frame(), [mano_sintetica((400 + 15 * k, 550), **solo_indice)], t)
    assert pz.lienzo.max(axis=2).astype(bool).sum() > 500

    # Seleccionar el color 3 (verde) posando índice+medio sobre su botón
    (x0, y0, x1, y1), _, _ = pz.botones[3]
    centro = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    m = mano_sintetica((0, 0), dedos=(True, True, False, False))
    desplaz = centro - (m.pts[8] + m.pts[12]) / 2
    for _ in range(25):
        t += 0.033
        pz.actualizar(frame(), [mano_sintetica(desplaz, dedos=(True, True, False, False))], t)
    assert pz.color == 3 and COLORES[3][0] == "verde"

    pz.deshacer()
    assert pz.lienzo.max() == 0


# ------------------------------------------------------------------ fruit ninja
def test_fruit_ninja_inicio_cortes_y_fin(tmp_path, monkeypatch):
    import fruit_ninja
    from fruit_ninja import FruitNinja
    monkeypatch.setattr(fruit_ninja, "RUTA_RECORD", str(tmp_path / "record.json"))
    fn = FruitNinja(semilla=3)
    t = 0.0
    fn.actualizar(frame(), [], t)
    assert fn.estado == "INICIO" and any(f.inicio for f in fn.frutas)

    # Barrido rápido del índice a través del centro (donde está la sandía de INICIO)
    for k in range(12):
        t += 0.033
        x = 300 + 70 * k
        m = mano_sintetica((0, 0))
        fn.actualizar(frame(), [mano_sintetica(np.array([x, H * 0.55]) - m.pts[8] + 0)], t)
    assert fn.estado == "JUGANDO"
    assert len(fn.mitades) >= 2

    # Cortar frutas: barridos verticales cuando haya alguna en pantalla
    cortadas = 0
    for _ in range(600):
        t += 0.033
        visibles = [f for f in fn.frutas if f.tipo != "bomba" and 0 < f.y < H]
        manos = []
        if visibles:
            f = visibles[0]
            fase = int(t * 30) % 4
            y = f.y - 150 + 100 * fase
            m = mano_sintetica((0, 0))
            manos = [mano_sintetica(np.array([f.x, y]) - m.pts[8])]
        antes = fn.puntos
        fn.actualizar(frame(), manos, t)
        cortadas += fn.puntos > antes
        if cortadas >= 3:
            break
    assert fn.puntos >= 3

    # Sin manos, las frutas se caen y se pierden las vidas
    for _ in range(3000):
        t += 0.033
        fn.actualizar(frame(), [], t)
        if fn.estado == "FIN":
            break
    assert fn.estado == "FIN"
    assert fn.record >= fn.puntos


# ------------------------------------------------------------------ señas
def _dataset(rng, n=80):
    formas = {"A": dict(dedos=(False, False, False, False)),
              "B": dict(dedos=(True, True, True, True)),
              "L": dict(dedos=(True, False, False, False)),
              "V": dict(dedos=(True, True, False, False))}
    datos = []
    for et, kw in formas.items():
        for _ in range(n):
            datos.append((et, mano_sintetica(rng.uniform(200, 1000, 2), escala=rng.uniform(40, 90),
                                             rot=rng.uniform(-0.25, 0.25), ruido=0.04, rng=rng,
                                             **kw)))
    return datos


def test_senas_grabar_entrenar_reconocer(tmp_path):
    from senas import caracteristicas as car
    from senas.entrenar import entrenar
    rng = np.random.default_rng(0)
    ruta_csv = str(tmp_path / "muestras.csv")
    ruta_mod = str(tmp_path / "modelo.joblib")
    for et, mano in _dataset(rng):
        car.guardar_muestras(et, [car.vector(mano)], ruta_csv)
    X, y = car.cargar_muestras(ruta_csv)
    assert X.shape == (320, car.N_CARACTERISTICAS)

    car.quitar_ultimas(5, ruta_csv)
    assert car.cargar_muestras(ruta_csv)[0].shape[0] == 315

    exactitud = entrenar(ruta_csv, ruta_mod, verbose=False)
    assert exactitud > 0.95

    from senas.reconocer import Reconocedor
    rec = Reconocedor(ruta_mod)
    t = 0.0
    for et in ("L", "V"):
        for _ in range(40):                     # sostener la seña ~1.3 s
            t += 0.033
            kw = {"L": (True, False, False, False), "V": (True, True, False, False)}[et]
            rec.actualizar(frame(), [mano_sintetica((640, 450), dedos=kw)], t)
    assert rec.escrito == "LV"


def test_fruit_ninja_corta_con_movimiento_normal_pero_no_quieto(tmp_path, monkeypatch):
    """Un barrido lento (unos 450 px/s) corta; un dedo casi quieto sobre la fruta, no."""
    import fruit_ninja
    from fruit_ninja import FruitNinja
    monkeypatch.setattr(fruit_ninja, "RUTA_RECORD", str(tmp_path / "record.json"))
    punta = mano_sintetica((0, 0)).pts[8]
    centro = np.array([W * 0.5, H * 0.55])          # ahí está la sandía de INICIO

    quieto = FruitNinja(semilla=1)
    rng = np.random.default_rng(0)
    t = 0.0
    for _ in range(60):                              # 2 s con temblor de ±2 px
        t += 0.033
        quieto.actualizar(frame(), [mano_sintetica(centro + rng.uniform(-2, 2, 2) - punta)], t)
    assert quieto.estado == "INICIO"

    lento = FruitNinja(semilla=1)
    t = 0.0
    for k in range(30):                              # 15 px por cuadro = 450 px/s
        t += 0.033
        lento.actualizar(frame(), [mano_sintetica(centro + (-225 + 15 * k, 0) - punta)], t)
    assert lento.estado == "JUGANDO"

    # con el umbral anterior (0.8) ese mismo barrido no cortaba
    estricto = FruitNinja(semilla=1, velocidad_corte=0.8)
    t = 0.0
    for k in range(30):
        t += 0.033
        estricto.actualizar(frame(), [mano_sintetica(centro + (-225 + 15 * k, 0) - punta)], t)
    assert estricto.estado == "INICIO"

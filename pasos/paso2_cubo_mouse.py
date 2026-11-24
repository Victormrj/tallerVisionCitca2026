
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cubo import MOVIMIENTOS, Animador, Cubo  
from render import Renderizador  

W, H = 1000, 700
cubo = Cubo()
anim = Animador(cubo)
render = Renderizador(centro=(0.5, 0.5), escala=0.12)
estado = {"arrastre": None, "sel": None, "inicio": None, "hover": None}


def raton(evento, x, y, flags, _):
    p = np.array([x, y], dtype=float)
    if evento == cv2.EVENT_LBUTTONDOWN:
        estado["sel"] = render.sticker_bajo(p)
        estado["inicio"] = p
        estado["arrastre"] = p
    elif evento == cv2.EVENT_MOUSEMOVE:
        estado["hover"] = render.sticker_bajo(p)
        if estado["arrastre"] is None:
            return
        if estado["sel"] is None:                       # girar la vista
            d = (p - estado["arrastre"]) / W
            render.rotar(d[0] * 3.2, d[1] * 3.2)
            estado["arrastre"] = p
        elif np.linalg.norm(p - estado["inicio"]) > 40 and not anim.ocupado:
            mov = render.elegir_giro(*estado["sel"], p - estado["inicio"])
            if mov:
                anim.encolar(mov)
            estado["sel"] = None
            estado["arrastre"] = None
    elif evento == cv2.EVENT_LBUTTONUP:
        estado["arrastre"] = estado["sel"] = None


cv2.namedWindow("Paso 2 - cubo")
cv2.setMouseCallback("Paso 2 - cubo", raton)
while True:
    lienzo = np.full((H, W, 3), (40, 32, 28), np.uint8)
    anim.actualizar()
    sel = estado["sel"] or estado["hover"]
    render.dibujar(lienzo, cubo, anim, sel,
                   (90, 235, 90) if estado["sel"] else (0, 170, 255))
    ultimo = cubo.historial[-1].notacion() if cubo.historial else "-"
    txt = f"movs: {len(cubo.historial)}  ultimo: {ultimo}  {'RESUELTO' if cubo.resuelto() else ''}"
    cv2.putText(lienzo, txt, (15, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2,
                cv2.LINE_AA)
    cv2.imshow("Paso 2 - cubo", lienzo)
    k = cv2.waitKey(15) & 0xFF
    if k in (ord("q"), 27):
        break
    c = chr(k) if k < 128 else ""
    if c.upper() in "RLUDFBMES" and c:
        nombre = c.upper() + ("'" if c.isupper() else "")
        anim.encolar(MOVIMIENTOS[nombre])
    elif c == "x":
        for m in cubo.mezcla_aleatoria(20):
            anim.encolar(m, duracion=0.08)
    elif c == "0":
        cubo.reiniciar()
    elif c == "z" and cubo.historial and not anim.ocupado:
        anim.encolar(cubo.historial.pop().inverso(), registrar=False)
cv2.destroyAllWindows()

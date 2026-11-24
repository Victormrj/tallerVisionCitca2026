from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from comun import abrir_captura, oscurecer, texto
from cubo import Animador, Camino, Cubo, reloj
from gestos import (APUNTANDO, COLOR_ESTADO, FIJADO, TEXTO_ESTADO, ControlRotacion,
                    ControlTouchpad)
from manos import DetectorManos, dibujar_mano
from render import Renderizador

FUENTE = cv2.FONT_HERSHEY_SIMPLEX
COLOR_ROTACION = (90, 235, 90)
COLOR_TOUCHPAD = (0, 170, 255)
COLOR_PISTA = (0, 230, 255)        # amarillo

AYUDA = [
    "MANO IZQ. (derecha de la pantalla): palma abierta + mover = girar cubo; puno = pausa",
    "MANO DER. en el recuadro: pellizca un sticker y arrastra = girar capa",
    "Teclas: m mezclar | p pista | r ordenar | z deshacer | espacio vista | a ayuda | q salir",
]


def asignar_roles(manos, touch: ControlTouchpad, w, h):
    mano_touch = mano_rot = None
    dentro = [m for m in manos if touch.contiene(m.palma, w, h)]
    fuera = [m for m in manos if not any(m is d for d in dentro)]
    if dentro:
        cx = (touch.caja[0] + touch.caja[2]) / 2 * w
        mano_touch = min(dentro, key=lambda m: abs(m.palma[0] - cx))
        fuera += [m for m in dentro if m is not mano_touch]
    if fuera:
        # la más a la derecha de la pantalla (mano izquierda en espejo)
        mano_rot = max(fuera, key=lambda m: m.palma[0])
    return mano_touch, mano_rot


def mezclar(cubo: Cubo, camino: Camino, n: int, animador: Animador | None = None):

    cubo.historial.clear()
    for m in cubo.mezcla_aleatoria(n):
        if animador is None:
            cubo.aplicar(m, registrar=False)
        else:
            animador.encolar(m, registrar=False, duracion=0.08)
        camino.agregar(m)


def dibujar_pista(img, render: Renderizador, camino: Camino):
    mov = camino.pista()
    if mov is None:
        return
    h, w = img.shape[:2]
    k = max(0.7, w / 1280)
    c = (int(render.centro_px[0]), int(render.centro_px[1] + render.radio_px * 1.7))
    flecha = render.flecha_pista(mov)
    if flecha is None:
        s = "PISTA: gira el cubo para ver otra cara"
    else:
        pts, p0, p1 = flecha
        cv2.polylines(img, [pts.astype(np.int32)], True, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.polylines(img, [pts.astype(np.int32)], True, COLOR_PISTA, 3, cv2.LINE_AA)
        a, b = tuple(int(v) for v in p0), tuple(int(v) for v in p1)
        cv2.arrowedLine(img, a, b, (0, 0, 0), 12, cv2.LINE_AA, tipLength=0.4)
        cv2.arrowedLine(img, a, b, COLOR_PISTA, 6, cv2.LINE_AA, tipLength=0.4)
        s = f"PISTA: pellizca y arrastra como la flecha  (faltan {len(camino)})"
    (tw, _), _ = cv2.getTextSize(s, FUENTE, 0.6 * k, 2)
    texto(img, s, (min(c[0] - tw // 2, w - tw - 14), c[1]), COLOR_PISTA, 0.6 * k, 2, placa=True)


def dibujar_hud(img, touch: ControlTouchpad, rot: ControlRotacion, yaw, pitch, fps,
                cubo: Cubo, mostrar_ayuda: bool, mezclado: bool):
    h, w = img.shape[:2]
    k = max(0.7, w / 1280)          # escala de letra según resolución
    x0, y0, x1, y1 = touch.caja_px(w, h)
    col = COLOR_ESTADO[touch.estado]
    cv2.rectangle(img, (x0, y0), (x1, y1), col, 2, cv2.LINE_AA)
    texto(img, "CONTROL POR GESTOS", (x0 + 6, y0 - 14), col, 0.65 * k, 2, placa=True)

    # Barra de progreso del arrastre cuando hay un sticker fijado
    if touch.estado == FIJADO:
        cv2.rectangle(img, (x0, y1 + 8), (x1, y1 + 16), (60, 60, 60), -1)
        cv2.rectangle(img, (x0, y1 + 8), (x0 + int((x1 - x0) * touch.progreso), y1 + 16),
                      col, -1)
    texto(img, TEXTO_ESTADO[touch.estado], (x0 + 6, y1 + int(44 * k)), col, 0.6 * k, 2, placa=True)

    # Cursor del touchpad sobre el cubo
    if touch.cursor is not None and touch.estado != "INACTIVO":
        c = tuple(int(v) for v in touch.cursor)
        cv2.circle(img, c, 9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.circle(img, c, 9, col, 2, cv2.LINE_AA)
        if touch.pellizcando:
            cv2.circle(img, c, 4, col, -1, cv2.LINE_AA)
        if touch.estado == FIJADO and touch.inicio is not None:
            cv2.line(img, tuple(int(v) for v in touch.inicio), c, col, 2, cv2.LINE_AA)

    # Barra inferior de estado
    ultimo = cubo.historial[-1].notacion() if cubo.historial else "-"
    barra = (f"[MANO] Y:{np.degrees(yaw):+.1f}  X:{np.degrees(pitch):+.1f}   "
             f"fps:{fps:4.1f}   movs:{len(cubo.historial)}   ultimo:{ultimo}   "
             f"rotacion:{rot.estado}")
    (tw, th), _ = cv2.getTextSize(barra, FUENTE, 0.5 * k, 1)
    alto = int(28 * k)
    cv2.rectangle(img, (0, h - alto), (w, h), (15, 15, 15), -1)
    texto(img, barra, (max(8, (w - tw) // 2), h - int(9 * k)), (235, 235, 235), 0.5 * k, 1,
          contorno=False)

    if mezclado and cubo.resuelto() and cubo.historial:
        s = "RESUELTO!"
        (tw, th), _ = cv2.getTextSize(s, FUENTE, 1.4, 3)
        texto(img, s, ((w - tw) // 2, int(150 * k)), (90, 235, 90), 1.4, 3)

    if mostrar_ayuda:
        esc, paso = 0.55 * k, int(27 * k)
        ancho = max(cv2.getTextSize(s, FUENTE, esc, 1)[0][0] for s in AYUDA)
        oscurecer(img, 0, 0, ancho + 28, paso * len(AYUDA) + int(14 * k))
        for i, s in enumerate(AYUDA):
            texto(img, s, (12, int(24 * k) + paso * i), (245, 245, 245), esc, 1, contorno=False)


def main():
    ap = argparse.ArgumentParser(description="Cubo Rubik controlado con gestos")
    ap.add_argument("--camara", type=int, default=0)
    ap.add_argument("--video", help="usar un archivo de video en lugar de la cámara")
    ap.add_argument("--salida", help="guardar el resultado en un .mp4")
    ap.add_argument("--sin-ventana", action="store_true", help="no abrir ventana (servidores)")
    ap.add_argument("--espejo", dest="espejo", action="store_true", default=None)
    ap.add_argument("--no-espejo", dest="espejo", action="store_false")
    ap.add_argument("--ancho", type=int, default=1280)
    ap.add_argument("--alto", type=int, default=720)
    ap.add_argument("--modelo", help="ruta a hand_landmarker.task (se descarga si no existe)")
    ap.add_argument("--movimientos", type=int, default=20,
                    help="giros de la mezcla inicial (20 = mezcla completa; 3 a 5 = fácil)")
    ap.add_argument("--ordenado", action="store_true", help="iniciar con el cubo resuelto")
    ap.add_argument("--pistas", action="store_true", help="iniciar con las pistas encendidas")
    ap.add_argument("--mezclar", action="store_true", help=argparse.SUPPRESS)  # ya es lo normal
    args = ap.parse_args()

    if args.video:
        cap = cv2.VideoCapture(args.video)
        espejo = False if args.espejo is None else args.espejo
    else:
        cap = abrir_captura(args.camara, args.ancho, args.alto)   # Windows/macOS/Linux
        espejo = True if args.espejo is None else args.espejo
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara/video. Prueba con --camara 1")
    fps_fuente = cap.get(cv2.CAP_PROP_FPS) or 30.0

    detector = DetectorManos(args.modelo)
    cubo = Cubo()
    animador = Animador(cubo)
    render = Renderizador()
    rot = ControlRotacion()
    touch = ControlTouchpad()
    camino = Camino()
    n_mezcla = max(1, args.movimientos)
    mezclado = not args.ordenado
    if mezclado:
        mezclar(cubo, camino, n_mezcla)          # el cubo inicia desordenado
    mostrar_pista = args.pistas

    escritor = None
    mostrar_ayuda = True
    fps, t_prev = 0.0, time.monotonic()
    n_frame = 0
    ventana = "Cubo Rubik con gestos"
    if not args.sin_ventana:
        cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if espejo:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            if args.video:
                reloj.simulado = n_frame / fps_fuente   # tiempo del video, no del CPU
            ts = int(reloj() * 1000)
            n_frame += 1

            render.configurar(frame)
            manos = detector.detectar(frame, ts)
            mano_touch, mano_rot = asignar_roles(manos, touch, w, h)

            yaw, pitch = rot.actualizar(mano_rot, w)
            if yaw or pitch:
                render.rotar(yaw, pitch)

            mov = touch.actualizar(mano_touch, w, h, render, animador)
            if mov is not None:
                animador.encolar(mov)
                camino.agregar(mov)
            animador.actualizar()

            (res, col_res) = touch.resaltado
            render.dibujar(frame, cubo, animador, res, col_res or COLOR_TOUCHPAD)
            if mostrar_pista and not animador.ocupado and not cubo.resuelto():
                dibujar_pista(frame, render, camino)

            if mano_touch is not None:
                dibujar_mano(frame, mano_touch, COLOR_TOUCHPAD, "R-touchpad")
            if mano_rot is not None:
                dibujar_mano(frame, mano_rot, COLOR_ROTACION, "L-rotar",
                             resaltar_puntas=rot.estado == "girando")

            ahora = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(1e-3, ahora - t_prev) if fps else 1 / max(1e-3, ahora - t_prev)
            t_prev = ahora
            dibujar_hud(frame, touch, rot, yaw, pitch,
                        fps_fuente if args.video else fps, cubo, mostrar_ayuda,
                        mezclado and not animador.ocupado)

            if args.salida:
                if escritor is None:
                    escritor = cv2.VideoWriter(args.salida, cv2.VideoWriter_fourcc(*"mp4v"),
                                               fps_fuente if args.video else 20.0, (w, h))
                escritor.write(frame)

            if not args.sin_ventana:
                if n_frame == 1:
                    cv2.resizeWindow(ventana, w, h)     # tamaño real: sin escalado borroso
                cv2.imshow(ventana, frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                elif k == ord("m") and not animador.ocupado:
                    cubo.reiniciar(); camino.reiniciar()     # mezcla nueva desde cero
                    mezclar(cubo, camino, n_mezcla, animador)
                    mezclado = True
                elif k == ord("p"):
                    mostrar_pista = not mostrar_pista
                elif k == ord("r") and not animador.ocupado:
                    cubo.reiniciar(); camino.reiniciar(); mezclado = False
                elif k == ord("z") and cubo.historial and not animador.ocupado:
                    deshecho = cubo.historial.pop().inverso()
                    animador.encolar(deshecho, registrar=False)
                    camino.agregar(deshecho)
                elif k == ord(" "):
                    render.reiniciar_orientacion()
                elif k == ord("a"):
                    mostrar_ayuda = not mostrar_ayuda
    finally:
        cap.release()
        if escritor is not None:
            escritor.release()
        detector.cerrar()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import platform
import time
import unicodedata

import cv2

from cubo import reloj
from manos import DetectorManos

FUENTE = cv2.FONT_HERSHEY_SIMPLEX
SISTEMA = platform.system() 


def abrir_captura(indice: int, ancho: int = 1280, alto: int = 720) -> cv2.VideoCapture:
    if SISTEMA == "Windows":
        cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)      
        if not cap.isOpened():
            cap = cv2.VideoCapture(indice)
    elif SISTEMA == "Darwin":
        cap = cv2.VideoCapture(indice, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(indice)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
    if not cap.isOpened():
        ayuda = {
            "Darwin": "En macOS: Configuración del Sistema > Privacidad y seguridad > Cámara, "
                      "y activa Terminal / VS Code. Luego vuelve a abrir la terminal.",
            "Windows": "En Windows: Configuración > Privacidad > Cámara, permite el acceso a "
                       "aplicaciones de escritorio. Cierra Zoom/Teams si la están usando.",
        }.get(SISTEMA, "")
        raise SystemExit(f"No se pudo abrir la cámara {indice}. Prueba con --camara 1. {ayuda}")
    return cap


def sin_acentos(s: str) -> str:
    s = s.replace("ñ", "n~").replace("Ñ", "N~").replace("¡", "").replace("¿", "")
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def oscurecer(img, x0, y0, x1, y1, factor=0.22):
   
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * factor).astype(img.dtype)


def texto(img, s, org, color=(255, 255, 255), escala=0.6, grosor=2, centrado=False,
          placa=False, contorno=True):

    s = sin_acentos(str(s))
    (tw, th), base = cv2.getTextSize(s, FUENTE, escala, grosor)
    x, y = int(org[0]), int(org[1])
    if centrado:
        x, y = x - tw // 2, y + th // 2
    if placa:
        m = int(5 + 7 * escala)
        oscurecer(img, x - m, y - th - m, x + tw + m, y + base + m)
    elif contorno:
        cv2.putText(img, s, (x, y), FUENTE, escala, (0, 0, 0), grosor + 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FUENTE, escala, color, grosor, cv2.LINE_AA)


def argumentos(descripcion: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=descripcion)
    ap.add_argument("--camara", type=int, default=0, help="índice de la cámara (0, 1, ...)")
    ap.add_argument("--video", help="usar un archivo de video en lugar de la cámara")
    ap.add_argument("--salida", help="guardar lo que se ve en un .mp4")
    ap.add_argument("--sin-ventana", action="store_true", help="no abrir ventana")
    ap.add_argument("--espejo", dest="espejo", action="store_true", default=None)
    ap.add_argument("--no-espejo", dest="espejo", action="store_false")
    ap.add_argument("--ancho", type=int, default=1280)
    ap.add_argument("--alto", type=int, default=720)
    ap.add_argument("--modelo", help="ruta a hand_landmarker.task (se descarga si falta)")
    ap.add_argument("--max-manos", type=int, default=2)
    return ap


def ejecutar(juego, args, titulo: str):
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"No se pudo abrir {args.video}")
        espejo = False if args.espejo is None else args.espejo
    else:
        cap = abrir_captura(args.camara, args.ancho, args.alto)
        espejo = True if args.espejo is None else args.espejo
    fps_fuente = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = DetectorManos(args.modelo, num_manos=args.max_manos)
    escritor = None
    n = 0
    t_prev, fps = time.monotonic(), 0.0
    if not args.sin_ventana:
        cv2.namedWindow(titulo, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if espejo:
                frame = cv2.flip(frame, 1)
            if args.video:
                reloj.simulado = n / fps_fuente
            n += 1
            t = reloj()
            manos = detector.detectar(frame, int(t * 1000))
            frame = juego.actualizar(frame, manos, t)

            ahora = time.monotonic()
            dt = max(1e-3, ahora - t_prev)
            t_prev = ahora
            fps = 1 / dt if fps == 0 else 0.9 * fps + 0.1 / dt
            h, w = frame.shape[:2]
            texto(frame, f"fps {fps_fuente if args.video else fps:4.1f}", (w - 110, h - 12),
                  (230, 230, 230), 0.5, 1, placa=True)

            if args.salida:
                if escritor is None:
                    escritor = cv2.VideoWriter(args.salida, cv2.VideoWriter_fourcc(*"mp4v"),
                                               fps_fuente if args.video else 20.0, (w, h))
                escritor.write(frame)
            if not args.sin_ventana:
                if n == 1:
                    cv2.resizeWindow(titulo, w, h)     # tamaño real: sin escalado borroso
                cv2.imshow(titulo, frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k != 255 and hasattr(juego, "tecla"):
                    juego.tecla(k)
    finally:
        cap.release()
        if escritor is not None:
            escritor.release()
        detector.cerrar()
        if hasattr(juego, "cerrar"):
            juego.cerrar()
        cv2.destroyAllWindows()
        reloj.simulado = None

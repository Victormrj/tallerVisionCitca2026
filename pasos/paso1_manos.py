import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comun import abrir_captura  
from manos import DetectorManos, dibujar_mano  

UMBRAL_PELLIZCO = 0.30   

cap = abrir_captura(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
detector = DetectorManos()
t0 = time.monotonic()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)                      
    manos = detector.detectar(frame, int((time.monotonic() - t0) * 1000))
    for mano in manos:
        pellizco = mano.apertura_pellizco
        color = (0, 255, 0) if pellizco < UMBRAL_PELLIZCO else (0, 170, 255)
        dibujar_mano(frame, mano, color)
        # numerar los landmarks para que se vean sus índices
        for i, (x, y) in enumerate(mano.pts.astype(int)):
            cv2.putText(frame, str(i), (x + 4, y - 4), cv2.FONT_HERSHEY_PLAIN, 0.9,
                        (255, 255, 255), 1, cv2.LINE_AA)
        x, y = mano.pts[0].astype(int)
        info = f"{mano.lado}  dedos:{mano.dedos_extendidos()}  pellizco:{pellizco:.2f}"
        cv2.putText(frame, info, (x - 80, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 2, cv2.LINE_AA)
        if pellizco < UMBRAL_PELLIZCO:
            px, py = mano.punto_pellizco.astype(int)
            cv2.circle(frame, (px, py), 12, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow("Paso 1 - manos", frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        break
cap.release()
detector.cerrar()
cv2.destroyAllWindows()

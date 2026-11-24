import numpy as np

from manos import Mano


def mano_sintetica(centro, escala=60.0, abierta=True, pellizco=False, dedos=None,
                   rot=0.0, ruido=0.0, rng=None):

    dedos = (abierta,) * 4 if dedos is None else dedos
    p = np.zeros((21, 2))
    xs = [-0.6, -0.2, 0.2, 0.6]                       # índice, medio, anular, meñique
    for k, (x, ext) in enumerate(zip(xs, dedos)):
        b = 5 + 4 * k
        p[b] = (x, -1.0)
        p[b + 1] = (x, -1.5)
        p[b + 2] = (x, -1.8 if ext else -1.2)
        p[b + 3] = (x, -2.2 if ext else -0.9)
    p[9] = (0.0, -1.0)                                 # nudillo medio a 1 unidad de la muñeca
    p[1], p[2], p[3] = (-0.5, -0.3), (-0.9, -0.6), (-1.1, -1.0)
    p[4] = p[8] + (0.05, 0.05) if pellizco else (-1.3, -1.3)
    if rot:
        c, s = np.cos(rot), np.sin(rot)
        p = p @ np.array([[c, -s], [s, c]]).T
    if ruido:
        rng = rng or np.random.default_rng()
        p = p + rng.normal(0, ruido, p.shape)
    return Mano(np.asarray(centro, float) + p * escala, "Right", 1.0, np.zeros(21))

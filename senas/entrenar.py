from __future__ import annotations

import os
import sys
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senas.caracteristicas import N_CARACTERISTICAS, RUTA_DATOS, RUTA_MODELO, cargar_muestras  

CANDIDATOS = {
    "KNN (vecinos cercanos)": lambda: make_pipeline(StandardScaler(),
                                                     KNeighborsClassifier(n_neighbors=5)),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=0),
    "Red neuronal (MLP)": lambda: make_pipeline(
        StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                                        early_stopping=True, random_state=0)),
}


def guardar_matriz_png(cm, etiquetas, ruta):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(1 + 0.5 * len(etiquetas), 1 + 0.5 * len(etiquetas)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(etiquetas)), etiquetas)
    ax.set_yticks(range(len(etiquetas)), etiquetas)
    ax.set_xlabel("Predicha")
    ax.set_ylabel("Real")
    for i in range(len(etiquetas)):
        for j in range(len(etiquetas)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_title("Matriz de confusion")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    return ruta


def entrenar(ruta_datos=RUTA_DATOS, ruta_modelo=RUTA_MODELO, verbose=True):
    X, y = cargar_muestras(ruta_datos)
    conteo = Counter(y.tolist())
    if verbose:
        print(f"{len(y)} ejemplos, {len(conteo)} etiquetas:",
              ", ".join(f"{k}={v}" for k, v in sorted(conteo.items())))
    if len(conteo) < 2:
        raise SystemExit("Necesitas al menos 2 letras grabadas. Usa senas/grabar.py")
    minimo = min(conteo.values())
    if minimo < 10:
        raise SystemExit("Cada letra necesita al menos 10 ejemplos (graba otra toma).")

    # 1) Comparar modelos con validación cruzada
    cv = StratifiedKFold(n_splits=min(5, minimo), shuffle=True, random_state=0)
    resultados = {}
    for nombre, crear in CANDIDATOS.items():
        puntaje = cross_val_score(crear(), X, y, cv=cv).mean()
        resultados[nombre] = puntaje
        if verbose:
            print(f"  {nombre:<24} exactitud media = {puntaje:.3f}")
    mejor = max(resultados, key=resultados.get)
    if verbose:
        print(f"\nMejor modelo: {mejor}")

    # 2) Reporte detallado con un conjunto de prueba separado
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    modelo = CANDIDATOS[mejor]().fit(Xtr, ytr)
    pred = modelo.predict(Xte)
    etiquetas = sorted(conteo)
    cm = confusion_matrix(yte, pred, labels=etiquetas)
    if verbose:
        print(f"Exactitud en prueba: {accuracy_score(yte, pred):.3f}\n")
        print(classification_report(yte, pred, zero_division=0))
        print("Matriz de confusion (filas = real, columnas = predicha):")
        print("     " + " ".join(f"{e:>4}" for e in etiquetas))
        for e, fila in zip(etiquetas, cm):
            print(f"{e:>4} " + " ".join(f"{v:>4}" for v in fila))

    # 3) Entrenar con TODO y guardar
    modelo = CANDIDATOS[mejor]().fit(X, y)
    os.makedirs(os.path.dirname(ruta_modelo), exist_ok=True)
    joblib.dump({"modelo": modelo, "etiquetas": list(modelo.classes_), "nombre": mejor,
                 "n_caracteristicas": N_CARACTERISTICAS}, ruta_modelo)
    png = guardar_matriz_png(cm, etiquetas, os.path.join(os.path.dirname(ruta_modelo),
                                                          "matriz_confusion.png"))
    if verbose:
        print(f"\nModelo guardado en {ruta_modelo}")
        if png:
            print(f"Matriz de confusion en {png}")
    return resultados[mejor]


if __name__ == "__main__":
    entrenar()

"""
Exportación del MODELO 2 (GRU) al frontend.

El frontend ejecuta el propio modelo: descomprime models/gru_<serie>.keras (un ZIP con
config.json y model.weights.h5), lee los pesos con h5wasm y hace el forward pass en
TypeScript. Este script genera, por serie:

  - exports/gru_meta_<serie>.json: mínimo y máximo del MinMaxScaler, orden EXACTO de las
    features, tamaño de ventana, capas del modelo y los últimos 30 días (ventas y
    exógenas) para armar la primera ventana del pronóstico.
  - exports/gru_referencia_<serie>.json: 5 ventanas de entrada reales (del periodo de test)
    y la predicción que da Keras para cada una. El navegador las ejecuta al cargar el modelo
    y exige una diferencia < 1e-4 con Python.

También comprueba que el .keras contenga config.json y model.weights.h5.

Ejecución independiente (requiere haber corrido entrenar_gru.py):
    python src/modelo_gru/exportar_gru.py
"""
from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # consola de Windows (cp1252) con tildes y símbolos
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entrenar_gru import FEATURES, HORIZONTE_TEST, VENTANA, cargar_serie, exogenas, ventana  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
DIR_MODELS = RAIZ / "models"
DIR_EXPORTS = RAIZ / "exports"
INDICES_REFERENCIA = [0, 10, 20, 30, 41]  # días del test cuyas ventanas se usan como referencia


def describir_keras(ruta: Path) -> list[dict]:
    """Verifica el contenido del ZIP .keras y resume las capas de config.json."""
    with zipfile.ZipFile(ruta) as z:
        nombres = z.namelist()
        for requerido in ("config.json", "model.weights.h5"):
            if requerido not in nombres:
                raise RuntimeError(f"{ruta.name} no contiene {requerido}: {nombres}")
        config = json.loads(z.read("config.json"))
    capas = []
    for capa in config["config"]["layers"]:
        c = capa["config"]
        capas.append({k: v for k, v in {
            "clase": capa["class_name"], "nombre": c.get("name"), "unidades": c.get("units"),
            "activacion": c.get("activation"), "activacion_recurrente": c.get("recurrent_activation"),
            "return_sequences": c.get("return_sequences"), "reset_after": c.get("reset_after"),
        }.items() if v is not None})
    return capas


def exportar_serie(id_serie: str, resultado: dict) -> tuple[dict, dict]:
    ruta = DIR_MODELS / f"gru_{id_serie}.keras"
    esc = json.loads((DIR_MODELS / f"gru_{id_serie}_escalador.json").read_text(encoding="utf-8"))
    modelo = tf.keras.models.load_model(ruta)
    mn, mx = esc["escalador_min"], esc["escalador_max"]

    df = cargar_serie(id_serie)
    ventas_esc = ((df["ventas"].to_numpy() - mn) / (mx - mn)).astype(np.float32)
    exog = exogenas(df)
    n_train = len(df) - HORIZONTE_TEST

    # Ventanas de referencia con datos reales del test
    referencia = []
    for i in INDICES_REFERENCIA:
        t = n_train + i
        x = ventana(ventas_esc, exog, t)
        y = float(modelo(x[None, ...], training=False).numpy()[0, 0])
        referencia.append({"fecha_objetivo": df["fecha"].iloc[t].strftime("%Y-%m-%d"),
                           "entrada": [[float(v) for v in fila] for fila in x], "salida": y})

    ultimos = df.iloc[-VENTANA:]
    meta = {
        "serie": id_serie,
        "nombre": resultado["nombre"],
        "archivo": ruta.name,
        "ventana": VENTANA,
        "features": FEATURES,
        "escalador_min": mn, "escalador_max": mx,
        "capas": describir_keras(ruta),
        "nota_features": ("Paso k (0..29) para predecir el día t: [ventas_esc(t-30+k), exógenas(t-29+k)]. "
                          "Exógenas: promo, school_holiday, festivo_a, festivo_b, festivo_c, "
                          "sin/cos(2π·dow/7) con dow 0=lunes, sin/cos(2π·(mes-1)/12)."),
        "ultima_fecha": ultimos["fecha"].iloc[-1].strftime("%Y-%m-%d"),
        "ultimos_dias": [
            {"fecha": r.fecha.strftime("%Y-%m-%d"), "ventas": round(float(r.ventas), 4),
             "ventas_esc": float((r.ventas - mn) / (mx - mn)),
             "promo": float(r.promo), "school_holiday": float(r.school_holiday),
             "festivo_a": float(r.festivo_a), "festivo_b": float(r.festivo_b), "festivo_c": float(r.festivo_c)}
            for r in ultimos.itertuples()
        ],
    }
    ref = {"serie": id_serie, "tolerancia": 1e-4, "espacio": "ventas escaladas (salida directa del modelo)",
           "casos": referencia}
    return meta, ref


def main() -> None:
    print("=" * 70)
    print("ETAPA 4b · EXPORTACIÓN GRU (.keras + metadatos + referencia)")
    print("=" * 70)
    resultados = json.loads((DIR_EXPORTS / "gru_resultados.json").read_text(encoding="utf-8"))
    for id_serie, res in resultados["series"].items():
        meta, ref = exportar_serie(id_serie, res)
        (DIR_EXPORTS / f"gru_meta_{id_serie}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        (DIR_EXPORTS / f"gru_referencia_{id_serie}.json").write_text(json.dumps(ref, ensure_ascii=False), encoding="utf-8")
        tam = (DIR_MODELS / meta["archivo"]).stat().st_size / 1e3
        print(f"▶ {res['nombre']:<38} {meta['archivo']} ({tam:.0f} KB) · escalador [{meta['escalador_min']:,.0f}, "
              f"{meta['escalador_max']:,.0f}] · capas {[c['clase'] for c in meta['capas']]}")
    print(f"Exportados gru_meta_<serie>.json y gru_referencia_<serie>.json en {DIR_EXPORTS}")


if __name__ == "__main__":
    main()

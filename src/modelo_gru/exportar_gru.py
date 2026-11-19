"""
Exportación del MODELO 2 (GRU) al frontend.

1. Convierte cada models/gru_<serie>.keras a ONNX con tf2onnx (opset 13) para ejecutarlo
   en el navegador con onnxruntime-web. Se verifica que ONNX y Keras den la misma salida.
2. Exporta exports/gru_config.json con, por serie:
     - archivo ONNX y nombres de entrada/salida,
     - mínimo y máximo del MinMaxScaler,
     - orden exacto de las features y tamaño de ventana,
     - los últimos 30 días reales (ventas y exógenas) para armar la primera ventana,
     - el pronóstico base a 42 días calculado con onnxruntime (sirve para validar el frontend).
3. Copia los .onnx directamente a rossmann-frontend/public/models/ si el frontend existe.

Ejecución independiente (requiere haber corrido entrenar_gru.py):
    python src/modelo_gru/exportar_gru.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
import sys as _sys
_sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # consola de Windows (cp1252) con tildes y símbolos

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import onnxruntime as ort
import pandas as pd
import tensorflow as tf
import tf2onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entrenar_gru import (  # noqa: E402  (mismo módulo del modelo GRU)
    FEATURES, HORIZONTE_TEST, VENTANA, cargar_serie, exogenas, ventana,
)

RAIZ = Path(__file__).resolve().parents[2]
DIR_MODELS = RAIZ / "models"
DIR_EXPORTS = RAIZ / "exports"
DIR_FRONT_MODELS = RAIZ.parent / "rossmann-frontend" / "public" / "models"
HORIZONTE_FUTURO = 42
OPSET = 13  # soportado por onnxruntime-web (backend wasm), incluye el operador GRU


def convertir_a_onnx(id_serie: str) -> Path:
    modelo = tf.keras.models.load_model(DIR_MODELS / f"gru_{id_serie}.keras")
    firma = (tf.TensorSpec((None, VENTANA, len(FEATURES)), tf.float32, name="input"),)

    # Se envuelve en tf.function: es la vía estable de tf2onnx con Keras 3.
    @tf.function(input_signature=firma)
    def inferencia(x):
        return {"output": modelo(x, training=False)}

    ruta = DIR_MODELS / f"gru_{id_serie}.onnx"
    tf2onnx.convert.from_function(inferencia, input_signature=firma, opset=OPSET, output_path=str(ruta))

    # Verificación numérica ONNX vs Keras
    sesion = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
    muestra = np.random.default_rng(0).random((4, VENTANA, len(FEATURES)), dtype=np.float32)
    dif = np.abs(sesion.run(None, {"input": muestra})[0] - modelo(muestra, training=False).numpy()).max()
    if dif > 1e-4:
        raise RuntimeError(f"ONNX difiere de Keras en {dif:.2e} para {id_serie}")
    return ruta


def pronostico_base_onnx(ruta_onnx: Path, df: pd.DataFrame, meta: dict) -> list[dict]:
    """
    Pronóstico recursivo a 42 días del escenario base (sin promo, sin vacaciones, sin festivo),
    replicando EXACTAMENTE lo que hace el frontend con onnxruntime-web.
    """
    sesion = ort.InferenceSession(str(ruta_onnx), providers=["CPUExecutionProvider"])
    mn, mx = meta["escalador_min"], meta["escalador_max"]
    fechas = pd.date_range(df["fecha"].iloc[-1] + pd.Timedelta(days=1), periods=HORIZONTE_FUTURO, freq="D")
    futuro = pd.DataFrame({"fecha": fechas, "promo": 0.0, "school_holiday": 0.0,
                           "festivo_a": 0.0, "festivo_b": 0.0, "festivo_c": 0.0})
    hist = df.iloc[-VENTANA:][["fecha", "promo", "school_holiday", "festivo_a", "festivo_b", "festivo_c"]]
    exog = exogenas(pd.concat([hist, futuro], ignore_index=True))
    ventas = list(((df["ventas"].iloc[-VENTANA:].values - mn) / (mx - mn)).astype(np.float32))
    for t in range(VENTANA, VENTANA + HORIZONTE_FUTURO):
        x = ventana(np.asarray(ventas, dtype=np.float32), exog, t)[None, ...]
        ventas.append(float(sesion.run(None, {"input": x})[0][0, 0]))
    pred = np.clip(np.asarray(ventas[VENTANA:]) * (mx - mn) + mn, 0, None)
    return [{"fecha": f.strftime("%Y-%m-%d"), "pred": round(float(p), 2)} for f, p in zip(fechas, pred)]


def main() -> dict:
    print("=" * 70)
    print("ETAPA 4b · EXPORTACIÓN GRU (ONNX + configuración del escalador)")
    print("=" * 70)
    resultados = json.loads((DIR_EXPORTS / "gru_resultados.json").read_text(encoding="utf-8"))
    config = {
        "modelo": "GRU", "ventana": VENTANA, "features": FEATURES, "horizonte": HORIZONTE_FUTURO,
        "entrada": "input", "salida": "output", "opset": OPSET,
        "nota_features": ("Paso k de la ventana (k=0..29) para predecir el día t: "
                          "[ventas_esc(t-30+k), exógenas(t-29+k)]; exógenas = promo, school_holiday, "
                          "festivo_a/b/c, sin/cos(2π·dow/7) con dow 0=lunes, sin/cos(2π·(mes-1)/12)."),
        "series": {},
    }
    for id_serie, res in resultados["series"].items():
        meta = json.loads((DIR_MODELS / f"gru_{id_serie}_meta.json").read_text(encoding="utf-8"))
        ruta = convertir_a_onnx(id_serie)
        df = cargar_serie(id_serie)
        mn, mx = meta["escalador_min"], meta["escalador_max"]
        ultimos = df.iloc[-VENTANA:]
        futuro = pronostico_base_onnx(ruta, df, meta)
        config["series"][id_serie] = {
            "nombre": res["nombre"],
            "onnx": ruta.name,
            "escalador_min": mn, "escalador_max": mx,
            "ultimos_dias": [
                {"fecha": r.fecha.strftime("%Y-%m-%d"), "ventas": round(float(r.ventas), 2),
                 "ventas_esc": round(float((r.ventas - mn) / (mx - mn)), 6),
                 "promo": float(r.promo), "school_holiday": float(r.school_holiday),
                 "festivo_a": float(r.festivo_a), "festivo_b": float(r.festivo_b), "festivo_c": float(r.festivo_c)}
                for r in ultimos.itertuples()
            ],
            "futuro_base": futuro,
        }
        res["futuro_base"] = futuro
        if DIR_FRONT_MODELS.parent.parent.exists():
            DIR_FRONT_MODELS.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ruta, DIR_FRONT_MODELS / ruta.name)
        print(f"▶ {res['nombre']:<38} {ruta.name} ({ruta.stat().st_size / 1e3:.0f} KB) · "
              f"escalador [{mn:,.0f}, {mx:,.0f}] · base 42d={sum(x['pred'] for x in futuro):,.0f}")

    (DIR_EXPORTS / "gru_config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    (DIR_EXPORTS / "gru_resultados.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Exportados: {DIR_EXPORTS / 'gru_config.json'} y gru_resultados.json (con futuro_base)")
    return config


if __name__ == "__main__":
    main()

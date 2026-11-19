"""
Exportación del MODELO 1 (ARIMA) al frontend.

El frontend ejecuta el propio modelo en el navegador: carga models/arima_<serie>.joblib con
Pyodide (Python en WebAssembly) y llama a `get_forecast(steps, exog=...)`. Este script
genera, por serie, exports/arima_meta_<serie>.json con todo lo que el navegador necesita:

  - orden (p,d,q)(P,D,Q,7), nombres y ORDEN de las variables exógenas, frecuencia y
    última fecha de entrenamiento;
  - los 42 días de test (ventas reales y exógenas): el navegador los usa en
    `resultados.extend(...)` para llevar el estado del modelo hasta el último dato real
    antes de pronosticar el futuro (conserva los parámetros, no reentrena);
  - la predicción de test que da Python con ESE MISMO .joblib, para que el frontend
    compruebe que Pyodide calcula lo mismo.

Además verifica que el .joblib recargado pronostica igual que en el entrenamiento.

Ejecución independiente (requiere haber corrido entrenar_arima.py):
    python src/modelo_arima/exportar_arima.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # consola de Windows (cp1252) con tildes y símbolos

import joblib
import numpy as np
import pandas as pd
import statsmodels

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parents[2]
DIR_PROCESSED = RAIZ / "data" / "processed"
DIR_MODELS = RAIZ / "models"
DIR_EXPORTS = RAIZ / "exports"

HORIZONTE_TEST = 42


def exportar_serie(id_serie: str, resultado: dict) -> dict:
    df = pd.read_csv(DIR_PROCESSED / f"serie_{id_serie}.csv", parse_dates=["fecha"]).set_index("fecha").asfreq("D")
    test = df.iloc[-HORIZONTE_TEST:]
    exogenas = resultado["exogenas"]
    X_te = test[exogenas].to_numpy(float)

    # Recarga del archivo que se publica y verificación contra el entrenamiento
    modelo = joblib.load(DIR_MODELS / f"arima_{id_serie}.joblib")
    pron = modelo.get_forecast(steps=HORIZONTE_TEST, exog=X_te)
    media = np.asarray(pron.predicted_mean)
    ic = np.asarray(pron.conf_int(alpha=0.05))
    pred_entrenamiento = np.array([p["pred"] for p in resultado["test"]])
    dif = np.abs(np.clip(media, 0, None) - pred_entrenamiento).max()
    if dif > 0.01:  # el JSON de resultados está redondeado a 2 decimales
        raise RuntimeError(f"El .joblib de {id_serie} no reproduce las predicciones del entrenamiento (dif={dif})")

    return {
        "serie": id_serie,
        "nombre": resultado["nombre"],
        "archivo": f"arima_{id_serie}.joblib",
        "orden": resultado["orden"],
        "orden_estacional": resultado["orden_estacional"],
        "exogenas": exogenas,
        "frecuencia": "D",
        "ultima_fecha_entrenamiento": resultado["train_fin"],
        "aic": resultado["aic"],
        "coeficientes_exogenas": resultado["coeficientes_exogenas"],
        "alfa_intervalo": 0.05,
        "versiones": {"statsmodels": statsmodels.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "test": {
            "fechas": [f.strftime("%Y-%m-%d") for f in test.index],
            "ventas": [round(float(v), 4) for v in test["ventas"]],
            "exog": [[float(v) for v in fila] for fila in X_te],
        },
        # Predicción SIN recortar (salida directa de get_forecast) para comparar con Pyodide
        "referencia_test": {
            "media": [float(v) for v in media],
            "li": [float(v) for v in ic[:, 0]],
            "ls": [float(v) for v in ic[:, 1]],
        },
    }


def main() -> dict:
    print("=" * 70)
    print("ETAPA 3b · EXPORTACIÓN ARIMA (.joblib + metadatos)")
    print("=" * 70)
    resultados = json.loads((DIR_EXPORTS / "arima_resultados.json").read_text(encoding="utf-8"))
    for id_serie, resultado in resultados["series"].items():
        meta = exportar_serie(id_serie, resultado)
        ruta = DIR_EXPORTS / f"arima_meta_{id_serie}.json"
        ruta.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        tam = (DIR_MODELS / meta["archivo"]).stat().st_size / 1e3
        print(f"▶ {resultado['nombre']:<38} {meta['archivo']} ({tam:.0f} KB) · {ruta.name} · "
              f"exógenas={meta['exogenas']} · verificado ✓")
    print(f"Versiones usadas: statsmodels {statsmodels.__version__}, numpy {np.__version__}, pandas {pd.__version__}")
    return resultados


if __name__ == "__main__":
    main()

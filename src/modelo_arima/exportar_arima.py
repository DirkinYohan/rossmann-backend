"""
Exportación del MODELO 1 (ARIMA) al frontend.

El frontend ejecuta el propio modelo en el navegador: carga models/arima_<serie>.joblib con
Pyodide (Python en WebAssembly) y llama a `get_forecast(steps, exog=...)`. El .joblib es el
modelo REENTRENADO con la serie completa (ver entrenar_arima.py), así que su pronóstico
empieza justo el día siguiente a `ultima_fecha` (01/08/2015).

Este script genera, por serie, exports/arima_meta_<serie>.json con:

  - orden (p,d,q)(P,D,Q,7), nombres y ORDEN de las variables exógenas, frecuencia;
  - `ultima_fecha`: último día del dataset (= último dato que vio el modelo final);
  - `mape_test`: MAPE del modelo en los 42 días de test (evaluado ANTES del reentrenamiento);
  - `info_tienda`: StoreType, Assortment, CompetitionDistance y Promo2 (solo informativo;
    vacío en "Total cadena");
  - `referencia`: un pronóstico de 42 días que da Python con ESE MISMO .joblib (promo de
    lunes a viernes), para que el frontend compruebe que Pyodide calcula lo mismo.

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

RAIZ_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_SRC / "limpieza"))
from info_tienda import info_tienda  # noqa: E402  (código compartido: ficha informativa)

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parents[2]
DIR_PROCESSED = RAIZ / "data" / "processed"
DIR_MODELS = RAIZ / "models"
DIR_EXPORTS = RAIZ / "exports"

HORIZONTE_TEST = 42


def exportar_serie(id_serie: str, resultado: dict, catalogo: dict) -> dict:
    df = pd.read_csv(DIR_PROCESSED / f"serie_{id_serie}.csv", parse_dates=["fecha"]).set_index("fecha").asfreq("D")
    exogenas = resultado["exogenas"]
    final = resultado.get("modelo_final")
    if not final:
        raise RuntimeError(f"arima_resultados.json no tiene el reentrenamiento final de {id_serie}: "
                           "ejecuta entrenar_arima.py (o entrenar_arima.py --solo-final)")
    ultima = df.index[-1]
    if final["entrenado_hasta"] != ultima.strftime("%Y-%m-%d"):
        raise RuntimeError(f"El modelo final de {id_serie} termina el {final['entrenado_hasta']}, pero la serie el {ultima:%Y-%m-%d}")

    # Pronóstico de referencia del .joblib publicado: 42 días desde ultima_fecha + 1 con
    # promo de lunes a viernes y sin vacaciones ni festivos.
    fechas = pd.date_range(ultima + pd.Timedelta(days=1), periods=HORIZONTE_TEST, freq="D")
    valores = {"promo": (fechas.dayofweek < 5).astype(float), "school_holiday": np.zeros(len(fechas)), "festivo": np.zeros(len(fechas))}
    X_ref = np.column_stack([valores[n] for n in exogenas])
    modelo = joblib.load(DIR_MODELS / f"arima_{id_serie}.joblib")
    pron = modelo.get_forecast(steps=HORIZONTE_TEST, exog=X_ref)
    media = np.asarray(pron.predicted_mean)
    ic = np.asarray(pron.conf_int(alpha=0.05))
    if not np.all(np.isfinite(media)):
        raise RuntimeError(f"El .joblib de {id_serie} produce valores no finitos")

    return {
        "serie": id_serie,
        "nombre": resultado["nombre"],
        "archivo": f"arima_{id_serie}.joblib",
        "orden": resultado["orden"],
        "orden_estacional": resultado["orden_estacional"],
        "exogenas": exogenas,
        "frecuencia": "D",
        "ultima_fecha": ultima.strftime("%Y-%m-%d"),
        "ultima_fecha_entrenamiento": final["entrenado_hasta"],
        # El .joblib compacto guarda los últimos `dias_ancla` días e inicia su filtro en el estado
        # del modelo completo en `fecha_ancla`. /prediccion reemplaza esos días por el historial
        # del usuario con results.apply(..., copy_initialization=True).
        "dias_ancla": int(modelo.nobs),
        "fecha_ancla": (ultima - pd.Timedelta(days=int(modelo.nobs) - 1)).strftime("%Y-%m-%d"),
        "aic": final["aic"],
        "coeficientes_exogenas": final["coeficientes_exogenas"],
        "alfa_intervalo": 0.05,
        "mape_test": resultado["metricas"]["MAPE"],
        "metricas_test": resultado["metricas"],
        "periodo_test": {"inicio": resultado["test_inicio"], "fin": resultado["test_fin"]},
        "info_tienda": info_tienda(catalogo[id_serie]),
        "versiones": {"statsmodels": statsmodels.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        # Predicción SIN recortar (salida directa de get_forecast) para comparar con Pyodide
        "referencia": {
            "descripcion": "42 días desde ultima_fecha + 1, promo de lunes a viernes, sin vacaciones ni festivos",
            "fechas": [f.strftime("%Y-%m-%d") for f in fechas],
            "exog": [[float(v) for v in fila] for fila in X_ref],
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
    catalogo = {s["id"]: s for s in json.loads((DIR_PROCESSED / "series.json").read_text(encoding="utf-8"))}
    for id_serie, resultado in resultados["series"].items():
        meta = exportar_serie(id_serie, resultado, catalogo)
        ruta = DIR_EXPORTS / f"arima_meta_{id_serie}.json"
        ruta.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        tam = (DIR_MODELS / meta["archivo"]).stat().st_size / 1e3
        print(f"▶ {resultado['nombre']:<38} {meta['archivo']} ({tam:.0f} KB) · {ruta.name} · "
              f"exógenas={meta['exogenas']} · desde {meta['ultima_fecha']} · MAPE test {meta['mape_test']:.2f}%")
    print(f"Versiones usadas: statsmodels {statsmodels.__version__}, numpy {np.__version__}, pandas {pd.__version__}")
    return resultados


if __name__ == "__main__":
    main()

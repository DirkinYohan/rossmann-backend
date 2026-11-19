"""
Exportación del MODELO 1 (ARIMA) al frontend.

statsmodels no puede ejecutarse en el navegador, así que se PRECALCULAN los escenarios.
Para los 42 días posteriores a la última fecha observada se generan predicciones (con IC 95 %)
para todas las combinaciones de:
  - Promo:          sin promo | lunes–viernes | todos los días
  - SchoolHoliday:  no | sí
  - StateHoliday:   ninguno | un festivo a mitad del horizonte
El festivo se coloca en la mitad del horizonte elegido (7, 14, 28 o 42 días), por eso
hay una variante por horizonte; sin festivo basta una trayectoria de 42 días que se recorta.

Antes de pronosticar, el modelo entrenado se ACTUALIZA con los 42 días de test
(`append(refit=False)`): conserva los parámetros estimados en train pero parte del último
dato real, igual que haría el negocio en producción.

Salida: exports/arima_escenarios.json

Ejecución independiente (requiere haber corrido entrenar_arima.py):
    python src/modelo_arima/exportar_arima.py
"""
from __future__ import annotations

import json
import warnings
from itertools import product
from pathlib import Path
import sys as _sys
_sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # consola de Windows (cp1252) con tildes y símbolos

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parents[2]
DIR_PROCESSED = RAIZ / "data" / "processed"
DIR_MODELS = RAIZ / "models"
DIR_EXPORTS = RAIZ / "exports"

HORIZONTE = 42
HORIZONTES_UI = [7, 14, 28, 42]
EXOGENAS = ["promo", "school_holiday", "festivo"]   # mismo orden que en el entrenamiento
OPCIONES_PROMO = ["sin", "lv", "todos"]
OPCIONES_SCHOOL = ["no", "si"]


def exogenas_escenario(fechas: pd.DatetimeIndex, promo: str, school: str, dia_festivo: int | None) -> pd.DataFrame:
    """Construye la matriz exógena futura de un escenario."""
    dow = fechas.dayofweek.values
    if promo == "sin":
        p = np.zeros(len(fechas))
    elif promo == "lv":
        p = (dow < 5).astype(float)          # lunes(0) a viernes(4)
    else:
        p = np.ones(len(fechas))
    sh = np.full(len(fechas), 1.0 if school == "si" else 0.0)
    fest = np.zeros(len(fechas))
    if dia_festivo is not None:
        fest[dia_festivo] = 1.0
    return pd.DataFrame({"promo": p, "school_holiday": sh, "festivo": fest}, index=fechas)[EXOGENAS]


def pronosticar(res, X: pd.DataFrame) -> list[dict]:
    fr = res.get_forecast(steps=len(X), exog=X).summary_frame(alpha=0.05)
    return [
        {"fecha": f.strftime("%Y-%m-%d"), "pred": round(max(0.0, float(m)), 2),
         "li": round(max(0.0, float(l)), 2), "ls": round(float(u), 2)}
        for f, m, l, u in zip(fr.index, fr["mean"], fr["mean_ci_lower"], fr["mean_ci_upper"])
    ]


def exportar_serie(id_serie: str, resultado: dict) -> dict:
    df = pd.read_csv(DIR_PROCESSED / f"serie_{id_serie}.csv", parse_dates=["fecha"]).set_index("fecha").asfreq("D")
    test = df.iloc[-HORIZONTE:]  # mismo periodo de test usado en el entrenamiento
    res = SARIMAXResults.load(str(DIR_MODELS / f"arima_{id_serie}.pkl"))
    res_act = res.append(test["ventas"], exog=test[EXOGENAS], refit=False)

    fechas = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=HORIZONTE, freq="D")
    escenarios: dict[str, list[dict]] = {}
    for promo, school in product(OPCIONES_PROMO, OPCIONES_SCHOOL):
        # Sin festivo: una trayectoria de 42 días (el frontend la recorta al horizonte).
        escenarios[f"{promo}_{school}_no"] = pronosticar(res_act, exogenas_escenario(fechas, promo, school, None))
        # Con festivo a mitad de cada horizonte posible.
        for h in HORIZONTES_UI:
            X = exogenas_escenario(fechas[:h], promo, school, h // 2)
            escenarios[f"{promo}_{school}_h{h}"] = pronosticar(res_act, X)

    return {
        "nombre": resultado["nombre"],
        "orden": resultado["orden"], "orden_estacional": resultado["orden_estacional"],
        "aic": resultado["aic"], "coeficientes_exogenas": resultado["coeficientes_exogenas"],
        "metricas": resultado["metricas"], "test": resultado["test"],
        "futuro_inicio": fechas[0].strftime("%Y-%m-%d"),
        "escenarios": escenarios,
    }


def main() -> dict:
    print("=" * 70)
    print("ETAPA 3b · EXPORTACIÓN ARIMA (escenarios precalculados)")
    print("=" * 70)
    resultados = json.loads((DIR_EXPORTS / "arima_resultados.json").read_text(encoding="utf-8"))
    salida = {
        "modelo": "ARIMA", "horizonte": HORIZONTE, "horizontes": HORIZONTES_UI,
        "claves": "<promo:sin|lv|todos>_<school:no|si>_<festivo:no|h7|h14|h28|h42>",
        "series": {},
    }
    for id_serie, resultado in resultados["series"].items():
        salida["series"][id_serie] = exportar_serie(id_serie, resultado)
        base = salida["series"][id_serie]["escenarios"]["sin_no_no"]
        lv = salida["series"][id_serie]["escenarios"]["lv_no_no"]
        tb, tl = sum(x["pred"] for x in base), sum(x["pred"] for x in lv)
        print(f"▶ {resultado['nombre']:<38} {len(salida['series'][id_serie]['escenarios'])} escenarios · "
              f"base 42d={tb:,.0f} · promo L-V={tl:,.0f} ({100 * (tl / tb - 1):+.1f}%)")
    ruta = DIR_EXPORTS / "arima_escenarios.json"
    ruta.write_text(json.dumps(salida, ensure_ascii=False), encoding="utf-8")
    print(f"Exportado: {ruta} ({ruta.stat().st_size / 1e6:.2f} MB)")
    return salida


if __name__ == "__main__":
    main()

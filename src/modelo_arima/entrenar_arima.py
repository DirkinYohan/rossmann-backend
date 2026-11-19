"""
MODELO 1 · SARIMAX con variables exógenas (independiente del modelo GRU).

Por cada serie (Total cadena + 5 tiendas):
  1. Carga la serie limpia de data/processed/.
  2. Divide en train / test respetando el orden temporal (test = últimos 42 días).
  3. Selecciona (p,d,q)(P,D,Q,7) con pmdarima.auto_arima minimizando el AIC.
  4. Ajusta un SARIMAX de statsmodels con esos órdenes y las exógenas
     Promo, SchoolHoliday y StateHoliday codificado (festivo 0/1).
  5. Pronostica el test, calcula RMSE/MAE/MAPE y guarda models/arima_<serie>.pkl.

Salida: exports/arima_resultados.json

Ejecución independiente:
    python src/modelo_arima/entrenar_arima.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
import sys as _sys
_sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # consola de Windows (cp1252) con tildes y símbolos

import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
from evaluacion.metricas import calcular_metricas  # noqa: E402  (código compartido permitido)

warnings.filterwarnings("ignore")

DIR_PROCESSED = RAIZ / "data" / "processed"
DIR_MODELS = RAIZ / "models"
DIR_EXPORTS = RAIZ / "exports"

HORIZONTE_TEST = 42          # 6 semanas: horizonte oficial de la competencia
PERIODO_ESTACIONAL = 7       # estacionalidad semanal
# StateHoliday se codifica como un indicador 0/1 ("hay festivo"): las categorías b (Pascua) y
# c (Navidad) aparecen muy pocas veces y por separado harían la matriz exógena casi singular.
EXOGENAS = ["promo", "school_holiday", "festivo"]


def cargar_serie(id_serie: str) -> pd.DataFrame:
    df = pd.read_csv(DIR_PROCESSED / f"serie_{id_serie}.csv", parse_dates=["fecha"])
    return df.set_index("fecha").asfreq("D")


def dividir(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporal SIN mezclar: train = todo menos los últimos 42 días."""
    return df.iloc[:-HORIZONTE_TEST], df.iloc[-HORIZONTE_TEST:]


def seleccionar_orden(y: pd.Series, X: pd.DataFrame) -> pm.arima.ARIMA:
    """
    Búsqueda stepwise por AIC con límites razonables:
      - p, q ≤ 3 y P, Q ≤ 1: la PACF corta en los primeros rezagos y la ACF muestra picos
        en 7, 14, 21… que se capturan con un término estacional de orden 1.
      - d ≤ 1 elegido con KPSS (ADF/KPSS sugieren a lo sumo una diferencia regular).
      - D = 1 FIJO: la ACF tiene picos persistentes en 7, 14, 21… y la diferencia de lag 7
        vuelve estacionaria la serie (ver estacionalidad.json). La prueba OCSB de pmdarima
        tiende a elegir D = 0 y entonces el modelo no reproduce la caída de los domingos.
    """
    return pm.auto_arima(
        y, X=X,
        seasonal=True, m=PERIODO_ESTACIONAL,
        start_p=1, start_q=1, max_p=3, max_q=3,
        start_P=1, start_Q=1, max_P=1, max_Q=1,
        max_d=1, D=1, test="kpss",
        information_criterion="aic", stepwise=True,
        suppress_warnings=True, error_action="ignore", maxiter=50,
    )


def ajustar_sarimax(y: pd.Series, X: pd.DataFrame, orden: tuple, orden_est: tuple):
    modelo = SARIMAX(
        y, exog=X, order=orden, seasonal_order=orden_est,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    return modelo.fit(disp=False, maxiter=200)


def entrenar_serie(s: dict) -> dict:
    df = cargar_serie(s["id"])
    train, test = dividir(df)
    y_tr, X_tr = train["ventas"], train[EXOGENAS]
    X_te = test[EXOGENAS]

    t0 = time.time()
    auto = seleccionar_orden(y_tr, X_tr)
    orden, orden_est = auto.order, auto.seasonal_order
    res = ajustar_sarimax(y_tr, X_tr, orden, orden_est)

    pron = res.get_forecast(steps=HORIZONTE_TEST, exog=X_te).summary_frame(alpha=0.05)
    # Las ventas no pueden ser negativas: se recortan predicción y límite inferior en 0.
    pred = pron["mean"].clip(lower=0)
    li, ls = pron["mean_ci_lower"].clip(lower=0), pron["mean_ci_upper"]

    # Se evalúa SOLO en días con venta real (los interpolados no son observaciones).
    mascara = test["abierto"].values == 1
    metricas = calcular_metricas(test["ventas"].values[mascara], pred.values[mascara])

    DIR_MODELS.mkdir(parents=True, exist_ok=True)
    res.save(str(DIR_MODELS / f"arima_{s['id']}.pkl"))

    coef = {k: round(float(v), 4) for k, v in res.params.items() if k in EXOGENAS}
    duracion = time.time() - t0
    print(f"▶ {s['nombre']:<38} SARIMAX{orden}x{orden_est}  AIC={res.aic:,.1f}  "
          f"RMSE={metricas['RMSE']:,.0f}  MAE={metricas['MAE']:,.0f}  MAPE={metricas['MAPE']:.2f}%  ({duracion:.0f}s)")
    return {
        "nombre": s["nombre"],
        "orden": list(orden), "orden_estacional": list(orden_est),
        "aic": round(float(res.aic), 2), "aic_auto_arima": round(float(auto.aic()), 2),
        "coeficientes_exogenas": coef,
        "exogenas": EXOGENAS,
        "train_inicio": train.index[0].strftime("%Y-%m-%d"), "train_fin": train.index[-1].strftime("%Y-%m-%d"),
        "test_inicio": test.index[0].strftime("%Y-%m-%d"), "test_fin": test.index[-1].strftime("%Y-%m-%d"),
        "metricas": metricas,
        "test": [
            {"fecha": f.strftime("%Y-%m-%d"), "real": round(float(v), 2), "abierto": int(a),
             "pred": round(float(p), 2), "li": round(float(l), 2), "ls": round(float(u), 2)}
            for f, v, a, p, l, u in zip(test.index, test["ventas"], test["abierto"], pred, li, ls)
        ],
    }


def main() -> dict:
    print("=" * 70)
    print("ETAPA 3 · MODELO ARIMA (SARIMAX + exógenas)")
    print("=" * 70)
    catalogo = json.loads((DIR_PROCESSED / "series.json").read_text(encoding="utf-8"))
    salida = {"modelo": "ARIMA", "horizonte_test": HORIZONTE_TEST, "series": {}}
    for s in catalogo:
        salida["series"][s["id"]] = entrenar_serie(s)
    DIR_EXPORTS.mkdir(parents=True, exist_ok=True)
    ruta = DIR_EXPORTS / "arima_resultados.json"
    ruta.write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Modelos en {DIR_MODELS} · resultados en {ruta}")
    return salida


if __name__ == "__main__":
    main()

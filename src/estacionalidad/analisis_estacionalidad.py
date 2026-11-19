"""
Análisis de estacionalidad (MÓDULO SEPARADO E INDEPENDIENTE DE LOS MODELOS).

Para cada serie limpia de data/processed/ calcula:
  - Descomposición STL (periodo 7): tendencia, estacional y residuo.
  - Estacionalidad semanal (venta media por día de la semana) y anual (por mes).
  - Pruebas ADF y KPSS con interpretación en texto y diferenciación sugerida.
  - ACF y PACF hasta el rezago 60.
  - Efecto de variables externas (Promo, SchoolHoliday, StateHoliday).

Salida: exports/estacionalidad.json (solo para visualización y para justificar
los parámetros de ARIMA; ningún modelo lee este archivo).

Ejecución independiente:
    python src/estacionalidad/analisis_estacionalidad.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
import sys as _sys
_sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # consola de Windows (cp1252) con tildes y símbolos

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

warnings.filterwarnings("ignore")  # KPSS avisa cuando el p-valor está fuera de la tabla

RAIZ = Path(__file__).resolve().parents[2]
DIR_PROCESSED = RAIZ / "data" / "processed"
DIR_EXPORTS = RAIZ / "exports"

NOMBRES_DIA = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
NOMBRES_MES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
NOMBRES_FESTIVO = {"0": "Ninguno", "a": "Público", "b": "Pascua", "c": "Navidad"}  # a, b, c
MAX_LAG = 60


def r(x: float, n: int = 2) -> float | None:
    """Redondeo seguro para JSON (NaN → None)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), n)


def pruebas_estacionariedad(y: pd.Series) -> dict:
    """
    ADF  : H0 = la serie tiene raíz unitaria (NO estacionaria).  p < 0.05 → estacionaria.
    KPSS : H0 = la serie ES estacionaria (en nivel).             p < 0.05 → NO estacionaria.
    Se aplican a la serie original, a la diferencia regular (d=1) y a la diferencia
    estacional semanal (D=1, lag 7) para sugerir el orden de diferenciación.
    """
    def correr(serie: pd.Series) -> dict:
        serie = serie.dropna()
        adf_stat, adf_p, adf_lags, _, adf_crit, _ = adfuller(serie, autolag="AIC")
        kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(serie, regression="c", nlags="auto")
        return {
            "adf": {"estadistico": r(adf_stat, 4), "p_valor": r(adf_p, 4), "rezagos": int(adf_lags),
                    "criticos": {k: r(v, 4) for k, v in adf_crit.items()}},
            "kpss": {"estadistico": r(kpss_stat, 4), "p_valor": r(kpss_p, 4), "rezagos": int(kpss_lags),
                     "criticos": {k: r(v, 4) for k, v in kpss_crit.items()}},
            "adf_estacionaria": bool(adf_p < 0.05),
            "kpss_estacionaria": bool(kpss_p >= 0.05),
        }

    original = correr(y)
    dif1 = correr(y.diff())
    dif7 = correr(y.diff(7))

    a, k = original["adf_estacionaria"], original["kpss_estacionaria"]
    if a and k:
        texto = ("ADF rechaza la raíz unitaria y KPSS no rechaza la estacionariedad: la serie es "
                 "estacionaria en nivel. No requiere diferenciación regular (d = 0).")
        d = 0
    elif not a and not k:
        texto = ("ADF no rechaza la raíz unitaria y KPSS rechaza la estacionariedad: la serie NO es "
                 "estacionaria. Se sugiere una diferenciación regular (d = 1).")
        d = 1
    elif a and not k:
        texto = ("ADF indica estacionariedad pero KPSS la rechaza: la serie es estacionaria en "
                 "diferencias con posible tendencia determinística o cambios de nivel. Se recomienda "
                 "probar d = 1 o modelar la tendencia; auto_arima decide por AIC.")
        d = 1
    else:
        texto = ("KPSS no rechaza la estacionariedad pero ADF no rechaza la raíz unitaria: evidencia "
                 "no concluyente (baja potencia). Se sugiere d = 0 y validar con AIC.")
        d = 0

    texto += (" La diferencia estacional de lag 7 "
              + ("deja la serie estacionaria según ambas pruebas" if dif7["adf_estacionaria"] and dif7["kpss_estacionaria"]
                 else "reduce el patrón semanal")
              + ", por eso se considera D = 1 con periodo m = 7 en el SARIMA.")
    return {
        "original": original, "diferencia_1": dif1, "diferencia_estacional_7": dif7,
        "interpretacion": texto, "d_sugerido": d, "D_sugerido": 1,
    }


def efectos_externos(df: pd.DataFrame) -> dict:
    """Venta promedio en días REALMENTE abiertos según cada condición externa."""
    real = df[df["abierto"] == 1].copy()
    real["promo_bin"] = (real["promo"] >= 0.5).astype(int)
    real["school_bin"] = (real["school_holiday"] >= 0.5).astype(int)

    def agrupar(col: str, etiquetas: dict) -> list[dict]:
        g = real.groupby(col)["ventas"].agg(["mean", "count"])
        return [{"grupo": etiquetas.get(k, str(k)), "venta_promedio": r(v["mean"]), "dias": int(v["count"])}
                for k, v in g.iterrows()]

    promo = agrupar("promo_bin", {0: "Sin promo", 1: "Con promo"})
    school = agrupar("school_bin", {0: "Sin vacaciones", 1: "Con vacaciones"})
    state = agrupar("state_holiday", NOMBRES_FESTIVO)

    def uplift(lista: list[dict]) -> float | None:
        if len(lista) < 2 or not lista[0]["venta_promedio"]:
            return None
        return r(100 * (lista[1]["venta_promedio"] / lista[0]["venta_promedio"] - 1))

    return {
        "promo": promo, "school_holiday": school, "state_holiday": state,
        "uplift_promo_pct": uplift(promo), "uplift_school_pct": uplift(school),
    }


def analizar_serie(df: pd.DataFrame) -> dict:
    y = df.set_index("fecha")["ventas"].asfreq("D")

    # STL robusta (periodo 7 = ciclo semanal): resiste a picos de diciembre.
    stl = STL(y, period=7, robust=True).fit()

    real = df[df["abierto"] == 1]
    semanal = real.groupby("dia_semana")["ventas"].mean()
    mensual = real.groupby("mes")["ventas"].mean()

    acf_vals = acf(y, nlags=MAX_LAG, fft=True)
    pacf_vals = pacf(y, nlags=MAX_LAG, method="ywm")
    banda = 1.96 / np.sqrt(len(y))  # intervalo de confianza aproximado al 95 %

    # Fuerza de la estacionalidad semanal (Wang, Smith & Hyndman): 1 - Var(R)/Var(S+R)
    fuerza = max(0.0, 1 - np.var(stl.resid) / np.var(stl.seasonal + stl.resid))

    return {
        "historico": [{"fecha": f.strftime("%Y-%m-%d"), "ventas": r(v), "abierto": int(a)}
                      for f, v, a in zip(df["fecha"], df["ventas"], df["abierto"])],
        "stl": [{"fecha": f.strftime("%Y-%m-%d"), "tendencia": r(t), "estacional": r(s), "residuo": r(e)}
                for f, t, s, e in zip(y.index, stl.trend, stl.seasonal, stl.resid)],
        "fuerza_estacional_semanal": r(fuerza, 3),
        "semanal": [{"dia": int(i), "nombre": NOMBRES_DIA[int(i)], "venta_promedio": r(v)} for i, v in semanal.items()],
        "mensual": [{"mes": int(i), "nombre": NOMBRES_MES[int(i) - 1], "venta_promedio": r(v)} for i, v in mensual.items()],
        "pruebas": pruebas_estacionariedad(y),
        "acf": [{"lag": i, "valor": r(v, 4)} for i, v in enumerate(acf_vals)],
        "pacf": [{"lag": i, "valor": r(v, 4)} for i, v in enumerate(pacf_vals)],
        "banda_confianza": r(banda, 4),
        "efectos": efectos_externos(df),
    }


def main() -> dict:
    print("=" * 70)
    print("ETAPA 2 · ANÁLISIS DE ESTACIONALIDAD")
    print("=" * 70)
    catalogo = json.loads((DIR_PROCESSED / "series.json").read_text(encoding="utf-8"))
    salida = {"series": {}}
    for s in catalogo:
        df = pd.read_csv(DIR_PROCESSED / f"serie_{s['id']}.csv", parse_dates=["fecha"])
        res = analizar_serie(df)
        res["nombre"] = s["nombre"]
        salida["series"][s["id"]] = res

        p = res["pruebas"]["original"]
        acf7 = res["acf"][7]["valor"]
        pico_dia = max(res["semanal"], key=lambda x: x["venta_promedio"])["nombre"]
        pico_mes = max(res["mensual"], key=lambda x: x["venta_promedio"])["nombre"]
        print(f"\n▶ {s['nombre']}")
        print(f"   ADF p={p['adf']['p_valor']}  KPSS p={p['kpss']['p_valor']}  → d sugerido={res['pruebas']['d_sugerido']}, D=1 (m=7)")
        print(f"   ACF(7)={acf7}  fuerza estacional semanal={res['fuerza_estacional_semanal']}")
        print(f"   Día de mayor venta: {pico_dia} · Mes de mayor venta: {pico_mes}")
        print(f"   Uplift Promo: {res['efectos']['uplift_promo_pct']}%  ·  Uplift SchoolHoliday: {res['efectos']['uplift_school_pct']}%")

    DIR_EXPORTS.mkdir(parents=True, exist_ok=True)
    ruta = DIR_EXPORTS / "estacionalidad.json"
    ruta.write_text(json.dumps(salida, ensure_ascii=False), encoding="utf-8")
    print(f"\nExportado: {ruta} ({ruta.stat().st_size / 1e6:.2f} MB)")
    return salida


if __name__ == "__main__":
    main()

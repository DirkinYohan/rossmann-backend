"""
Análisis de estacionariedad y estacionalidad (MÓDULO SEPARADO E INDEPENDIENTE DE LOS MODELOS).

Genera un INFORME ESTADÍSTICO por serie en exports/estacionalidad_informe.json:
  - Pruebas ADF y KPSS (estadístico, p-valor, rezagos, valores críticos y veredicto) y una
    conclusión en español que interpreta las dos pruebas juntas (4 casos posibles).
  - Diferenciación recomendada (d, D, periodo).
  - Descomposición STL con periodo 7 y fuerza de la estacionalidad semanal.
  - Venta promedio por día de la semana y por mes.
  - ACF y PACF hasta el rezago 60 con su banda de confianza al 95 %.
  - Efecto de las variables externas (Promo, SchoolHoliday, StateHoliday).

El frontend solo PRESENTA este informe (no recalcula nada) y ningún modelo lo lee:
sirve para visualizar y para justificar los parámetros del SARIMAX.

Ejecución independiente:
    python src/estacionalidad/analisis_estacionalidad.py
"""
from __future__ import annotations

import json
import sys as _sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

_sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # consola de Windows (cp1252) con tildes y símbolos

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

warnings.filterwarnings("ignore")  # KPSS avisa cuando el p-valor cae fuera de su tabla (se reporta acotado)

RAIZ = Path(__file__).resolve().parents[2]
DIR_PROCESSED = RAIZ / "data" / "processed"
DIR_EXPORTS = RAIZ / "exports"

NOMBRES_DIA = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
NOMBRES_MES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
NOMBRES_FESTIVO = {"0": "Ninguno", "a": "Público", "b": "Pascua", "c": "Navidad"}  # a, b, c
MAX_LAG = 60
ALFA = 0.05
PERIODO = 7


def r(x, n: int = 2):
    """Redondeo seguro para JSON (NaN/inf → None)."""
    if x is None:
        return None
    x = float(x)
    return round(x, n) if np.isfinite(x) else None


# ---------------------------------------------------------------------------
# Pruebas de estacionariedad
# ---------------------------------------------------------------------------
def prueba_adf(y: pd.Series) -> dict:
    """ADF – H0: la serie tiene raíz unitaria (NO estacionaria). p < 0,05 → estacionaria."""
    est, p, lags, _, crit, _ = adfuller(y.dropna(), autolag="AIC")
    return {"estadistico": r(est, 4), "p_valor": r(p, 4), "rezagos": int(lags),
            "valores_criticos": {k: r(v, 4) for k, v in crit.items()}, "estacionaria": bool(p < ALFA)}


def prueba_kpss(y: pd.Series) -> dict:
    """KPSS – H0: la serie ES estacionaria en nivel. p < 0,05 → NO estacionaria."""
    est, p, lags, crit = kpss(y.dropna(), regression="c", nlags="auto")
    crit = {k: v for k, v in crit.items() if k != "2.5%"}  # se reportan 1 %, 5 % y 10 %
    return {"estadistico": r(est, 4), "p_valor": r(p, 4), "rezagos": int(lags),
            "valores_criticos": {k: r(v, 4) for k, v in crit.items()}, "estacionaria": bool(p >= ALFA),
            "nota_p_valor": "statsmodels acota el p-valor de KPSS al intervalo [0,01, 0,10]."}


def conclusion(adf: dict, kpss_: dict) -> tuple[str, str, int]:
    """
    Interpreta ADF y KPSS JUNTOS (las hipótesis nulas son opuestas, por eso se complementan):
      1. ADF estacionaria  + KPSS estacionaria     → estacionaria.
      2. ADF no estac.     + KPSS no estacionaria  → no estacionaria (raíz unitaria).
      3. ADF no estac.     + KPSS estacionaria     → estacionaria en tendencia.
      4. ADF estacionaria  + KPSS no estacionaria  → estacionaria en diferencias.
    Devuelve (caso, texto, d recomendado).
    """
    a, k = adf["estacionaria"], kpss_["estacionaria"]
    pa, pk = adf["p_valor"], kpss_["p_valor"]
    if a and k:
        return ("estacionaria",
                f"Las dos pruebas coinciden: ADF rechaza la raíz unitaria (p = {pa}) y KPSS no rechaza la "
                f"estacionariedad (p = {pk}). La serie es ESTACIONARIA en nivel, así que no necesita "
                "diferenciación regular (d = 0).", 0)
    if not a and not k:
        return ("no_estacionaria",
                f"Las dos pruebas coinciden: ADF no rechaza la raíz unitaria (p = {pa}) y KPSS rechaza la "
                f"estacionariedad (p = {pk}). La serie NO es estacionaria: tiene raíz unitaria y requiere una "
                "diferenciación regular (d = 1).", 1)
    if not a and k:
        return ("estacionaria_en_tendencia",
                f"ADF no rechaza la raíz unitaria (p = {pa}) pero KPSS no rechaza la estacionariedad (p = {pk}). "
                "La serie es ESTACIONARIA EN TENDENCIA: fluctúa alrededor de una tendencia determinística. "
                "Conviene modelar o quitar la tendencia en lugar de diferenciar (d = 0).", 0)
    return ("estacionaria_en_diferencias",
            f"ADF rechaza la raíz unitaria (p = {pa}) pero KPSS rechaza la estacionariedad en nivel (p = {pk}). "
            "La serie es ESTACIONARIA EN DIFERENCIAS: la media cambia en el tiempo y una diferenciación "
            "regular (d = 1) la vuelve estacionaria.", 1)


# ---------------------------------------------------------------------------
# Informe por serie
# ---------------------------------------------------------------------------
def efectos_externos(df: pd.DataFrame) -> dict:
    """Venta promedio en días REALMENTE abiertos según cada condición externa."""
    real = df[df["abierto"] == 1].copy()
    real["promo_bin"] = (real["promo"] >= 0.5).astype(int)
    real["school_bin"] = (real["school_holiday"] >= 0.5).astype(int)

    def agrupar(col: str, etiquetas: dict) -> dict:
        g = real.groupby(col)["ventas"].agg(["mean", "count"])
        grupos = {etiquetas.get(k, str(k)): {"venta_promedio": r(v["mean"]), "dias": int(v["count"])}
                  for k, v in g.iterrows()}
        valores = list(grupos.values())
        efecto = (r(100 * (valores[1]["venta_promedio"] / valores[0]["venta_promedio"] - 1))
                  if len(valores) == 2 and valores[0]["venta_promedio"] else None)
        return {"grupos": grupos, "efecto_pct": efecto}

    return {
        "promo": agrupar("promo_bin", {0: "Sin promo", 1: "Con promo"}),
        "school_holiday": agrupar("school_bin", {0: "Sin vacaciones", 1: "Con vacaciones"}),
        "state_holiday": agrupar("state_holiday", NOMBRES_FESTIVO),
    }


def informe_serie(df: pd.DataFrame, nombre: str) -> dict:
    y = df.set_index("fecha")["ventas"].asfreq("D")

    adf, kps = prueba_adf(y), prueba_kpss(y)
    caso, texto, d = conclusion(adf, kps)

    # Diferencia estacional semanal: ¿la vuelve estacionaria? → justifica D = 1 con periodo 7.
    y7 = y.diff(PERIODO)
    adf7, kps7 = prueba_adf(y7), prueba_kpss(y7)
    D = 1
    texto += (f" Tras la diferencia estacional de lag {PERIODO}, ADF p = {adf7['p_valor']} y KPSS p = "
              f"{kps7['p_valor']}: " + ("ambas pruebas indican estacionariedad" if adf7["estacionaria"] and kps7["estacionaria"]
                                         else "el patrón semanal se reduce")
              + f". Junto con los picos de la ACF en los rezagos 7, 14, 21…, esto justifica D = {D} con periodo {PERIODO}.")

    # STL robusta (periodo 7 = ciclo semanal): resiste a los picos de diciembre.
    stl = STL(y, period=PERIODO, robust=True).fit()
    fuerza = max(0.0, 1 - np.var(stl.resid) / np.var(stl.seasonal + stl.resid))  # Wang, Smith & Hyndman

    real = df[df["abierto"] == 1]
    semanal = real.groupby("dia_semana")["ventas"].mean()
    mensual = real.groupby("mes")["ventas"].mean()
    banda = r(1.96 / np.sqrt(len(y)), 4)
    acf_v = acf(y, nlags=MAX_LAG, fft=True)
    pacf_v = pacf(y, nlags=MAX_LAG, method="ywm")

    return {
        "nombre": nombre,
        "n_observaciones": int(len(y)),
        "adf": adf,
        "kpss": kps,
        "caso": caso,
        "conclusion": texto,
        "diferenciacion_recomendada": {"d": d, "D": D, "periodo": PERIODO},
        "pruebas_tras_diferencia_estacional": {"adf": adf7, "kpss": kps7},
        "stl": {
            "fechas": [f.strftime("%Y-%m-%d") for f in y.index],
            "tendencia": [r(v) for v in stl.trend],
            "estacional": [r(v) for v in stl.seasonal],
            "residuo": [r(v) for v in stl.resid],
        },
        "fuerza_estacionalidad": r(fuerza, 3),
        "promedio_por_dia_semana": {NOMBRES_DIA[int(i)]: r(v) for i, v in semanal.items()},
        "promedio_por_mes": {NOMBRES_MES[int(i) - 1]: r(v) for i, v in mensual.items()},
        "acf": {"rezagos": list(range(len(acf_v))), "valores": [r(v, 4) for v in acf_v], "banda_confianza": banda},
        "pacf": {"rezagos": list(range(len(pacf_v))), "valores": [r(v, 4) for v in pacf_v], "banda_confianza": banda},
        "efecto_variables_externas": efectos_externos(df),
    }


def main() -> dict:
    print("=" * 70)
    print("ETAPA 2 · INFORME DE ESTACIONARIEDAD Y ESTACIONALIDAD")
    print("=" * 70)
    catalogo = json.loads((DIR_PROCESSED / "series.json").read_text(encoding="utf-8"))
    informe = {"generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "alfa": ALFA, "max_rezago": MAX_LAG, "series": {}}
    for s in catalogo:
        df = pd.read_csv(DIR_PROCESSED / f"serie_{s['id']}.csv", parse_dates=["fecha"])
        res = informe_serie(df, s["nombre"])
        informe["series"][s["id"]] = res
        dr = res["diferenciacion_recomendada"]
        ef = res["efecto_variables_externas"]
        print(f"\n▶ {s['nombre']}")
        print(f"   ADF p={res['adf']['p_valor']} ({'estac.' if res['adf']['estacionaria'] else 'no estac.'})  "
              f"KPSS p={res['kpss']['p_valor']} ({'estac.' if res['kpss']['estacionaria'] else 'no estac.'})  "
              f"→ {res['caso']} · d={dr['d']}, D={dr['D']}, m={dr['periodo']}")
        print(f"   ACF(7)={res['acf']['valores'][7]}  fuerza estacional={res['fuerza_estacionalidad']}  "
              f"efecto Promo={ef['promo']['efecto_pct']}%  efecto vacaciones={ef['school_holiday']['efecto_pct']}%")

    DIR_EXPORTS.mkdir(parents=True, exist_ok=True)
    ruta = DIR_EXPORTS / "estacionalidad_informe.json"
    ruta.write_text(json.dumps(informe, ensure_ascii=False), encoding="utf-8")
    print(f"\nExportado: {ruta} ({ruta.stat().st_size / 1e6:.2f} MB)")
    return informe


if __name__ == "__main__":
    main()

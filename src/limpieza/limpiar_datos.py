"""
Limpieza y preparación de datos – Rossmann Store Sales.

Entrada : data/raw/train.csv y data/raw/store.csv
Salida  : data/processed/serie_<id>.csv  (una serie diaria por archivo)
          data/processed/series.json      (catálogo de series elegidas)
          data/processed/reporte_limpieza.json

Cada paso indica entre corchetes la DIMENSIÓN DE CALIDAD que resuelve:
  [Integración] [Consistencia] [Unicidad] [Validez] [Completitud] [Exactitud] [Oportunidad/Temporalidad]

Ejecución independiente:
    python src/limpieza/limpiar_datos.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys as _sys
_sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # consola de Windows (cp1252) con tildes y símbolos

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rutas (relativas a rossmann-backend/, sin importar desde dónde se ejecute)
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[2]
DIR_RAW = RAIZ / "data" / "raw"
DIR_PROCESSED = RAIZ / "data" / "processed"

# Columnas exógenas que se agregan por serie
CATEGORIAS_FESTIVO = ["a", "b", "c"]  # a = festivo público, b = Pascua, c = Navidad


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee los CSV crudos. StateHoliday se lee como texto para no perder el tipo mixto."""
    ruta_train = DIR_RAW / "train.csv"
    ruta_store = DIR_RAW / "store.csv"
    faltantes = [str(p) for p in (ruta_train, ruta_store) if not p.exists()]
    if faltantes:
        raise FileNotFoundError(
            "Faltan archivos del dataset. Descárgalos de "
            "https://www.kaggle.com/competitions/rossmann-store-sales y colócalos en "
            f"data/raw/: {faltantes}"
        )
    train = pd.read_csv(ruta_train, dtype={"StateHoliday": str}, low_memory=False)
    store = pd.read_csv(ruta_store)
    return train, store


def limpiar(train: pd.DataFrame, store: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Aplica todos los pasos de limpieza a nivel tienda-día.

    Devuelve:
      - df_abiertos: registros limpios SOLO de días abiertos (para modelar y analizar).
      - df_completo: registros limpios incluyendo días cerrados (se usa para saber
        qué festivo/vacación había en cada fecha aunque la tienda estuviera cerrada).
      - reporte: diccionario con el resumen de la limpieza.
    """
    reporte: dict = {"filas_train_original": int(len(train)), "filas_store_original": int(len(store))}

    # 1) [Integración] Unir ventas diarias con los atributos de cada tienda.
    #    how="left" garantiza que no se pierda ninguna venta aunque faltara una tienda.
    df = train.merge(store, on="Store", how="left")
    reporte["filas_tras_merge"] = int(len(df))
    reporte["tiendas_sin_atributos"] = int(df["StoreType"].isna().sum())

    # 2) [Consistencia/Temporalidad] Fecha como datetime y orden cronológico.
    #    Imprescindible en series de tiempo: los modelos asumen orden temporal.
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    # 3) [Unicidad] Eliminar duplicados exactos y, por seguridad, duplicados de la clave (Store, Date).
    antes = len(df)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["Store", "Date"], keep="last")
    reporte["duplicados_eliminados"] = int(antes - len(df))

    # 4) [Consistencia] StateHoliday mezcla 0 numérico y "0" texto (y a veces "0.0").
    #    Se unifica a las categorías {"0","a","b","c"}.
    df["StateHoliday"] = (
        df["StateHoliday"].astype(str).str.strip().replace({"0.0": "0", "nan": "0", "": "0"})
    )
    invalidos = ~df["StateHoliday"].isin(["0"] + CATEGORIAS_FESTIVO)
    reporte["stateholiday_valores_originales"] = sorted(train["StateHoliday"].astype(str).unique().tolist())
    reporte["stateholiday_invalidos_corregidos"] = int(invalidos.sum())
    df.loc[invalidos, "StateHoliday"] = "0"

    # 5) [Completitud] Imputación de nulos de los atributos de tienda.
    #    - CompetitionDistance: mediana (robusta ante distancias extremas).
    #    - *Since*: 0 = "no aplica / desconocido" (la tienda no tiene competencia
    #      registrada o no participa en Promo2), evitando inventar fechas.
    nulos_antes = df[[
        "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
        "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval",
    ]].isna().sum().to_dict()
    mediana_dist = float(store["CompetitionDistance"].median())
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(mediana_dist)
    for col in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2SinceWeek", "Promo2SinceYear"]:
        df[col] = df[col].fillna(0).astype(int)
    df["PromoInterval"] = df["PromoInterval"].fillna("Ninguno")
    reporte["nulos_tratados"] = {k: int(v) for k, v in nulos_antes.items()}
    reporte["mediana_competition_distance"] = mediana_dist

    # 6) [Validez] Rangos: no puede haber ventas ni clientes negativos.
    negativos = (df["Sales"] < 0) | (df["Customers"] < 0)
    reporte["registros_negativos_eliminados"] = int(negativos.sum())
    df = df[~negativos]

    # 7) Variables de calendario [Oportunidad: deja explícita la estructura temporal].
    df["anio"] = df["Date"].dt.year
    df["mes"] = df["Date"].dt.month
    df["dia"] = df["Date"].dt.day
    df["semana_anio"] = df["Date"].dt.isocalendar().week.astype(int)
    df["dia_semana"] = df["Date"].dt.dayofweek  # 0 = lunes … 6 = domingo
    df["es_fin_de_mes"] = df["Date"].dt.is_month_end.astype(int)

    df_completo = df.copy()

    # 8) [Exactitud] Excluir días cerrados (Open == 0): su venta es 0 por cierre, no por demanda.
    #    Mantenerlos sesgaría a la baja los modelos y haría infinito el MAPE.
    #    También se excluyen los pocos días "abiertos" con venta 0 (errores de registro).
    cerrados = df["Open"] == 0
    abiertos_sin_venta = (df["Open"] == 1) & (df["Sales"] == 0)
    reporte["dias_cerrados_excluidos"] = int(cerrados.sum())
    reporte["dias_abiertos_con_venta_cero_excluidos"] = int(abiertos_sin_venta.sum())
    df_abiertos = df[~cerrados & ~abiertos_sin_venta].copy()

    # 9) [Exactitud] Outliers con IQR por tienda. Se REPORTAN pero no se eliminan:
    #    los picos de diciembre y de promociones son demanda real y el modelo debe aprenderlos.
    q1 = df_abiertos.groupby("Store")["Sales"].transform(lambda s: s.quantile(0.25))
    q3 = df_abiertos.groupby("Store")["Sales"].transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    es_outlier = (df_abiertos["Sales"] < q1 - 1.5 * iqr) | (df_abiertos["Sales"] > q3 + 1.5 * iqr)
    df_abiertos["es_outlier_iqr"] = es_outlier.astype(int)
    outliers = df_abiertos[es_outlier]
    reporte["outliers_iqr"] = {
        "total": int(len(outliers)),
        "porcentaje": round(100 * len(outliers) / len(df_abiertos), 2),
        "en_diciembre": int((outliers["mes"] == 12).sum()),
        "con_promo": int((outliers["Promo"] == 1).sum()),
        "decision": "Se conservan: corresponden mayoritariamente a diciembre y promociones (demanda real).",
    }

    reporte["filas_finales_abiertas"] = int(len(df_abiertos))
    reporte["tiendas"] = int(df_abiertos["Store"].nunique())
    reporte["fecha_min"] = df_abiertos["Date"].min().strftime("%Y-%m-%d")
    reporte["fecha_max"] = df_abiertos["Date"].max().strftime("%Y-%m-%d")
    return df_abiertos, df_completo, reporte


def elegir_tiendas(df_abiertos: pd.DataFrame, df_completo: pd.DataFrame) -> list[dict]:
    """
    Elige automáticamente 5 tiendas representativas:
      - Una por StoreType (a, b, c, d): la tienda con historia COMPLETA (sin el hueco de
        180 tiendas en el 2º semestre de 2014) cuya venta media diaria esté más cerca de
        la mediana de su tipo → es la tienda "típica" de ese formato.
      - La tienda con mayor volumen total de ventas de toda la cadena.
    """
    n_fechas = df_completo["Date"].nunique()
    registros = df_completo.groupby("Store")["Date"].nunique()
    completas = set(registros[registros == n_fechas].index)

    resumen = df_abiertos.groupby(["Store", "StoreType"])["Sales"].agg(["mean", "sum"]).reset_index()
    elegidas: list[dict] = []

    top = resumen.sort_values("sum", ascending=False).iloc[0]
    for tipo in ["a", "b", "c", "d"]:
        candidatas = resumen[(resumen["StoreType"] == tipo) & (resumen["Store"].isin(completas))]
        candidatas = candidatas[candidatas["Store"] != top["Store"]]
        if candidatas.empty:  # respaldo: si no hay completas, se usa cualquier tienda del tipo
            candidatas = resumen[(resumen["StoreType"] == tipo) & (resumen["Store"] != top["Store"])]
        mediana = resumen.loc[resumen["StoreType"] == tipo, "mean"].median()
        fila = candidatas.iloc[(candidatas["mean"] - mediana).abs().argsort().iloc[0]]
        elegidas.append({
            "id": f"tienda_{int(fila['Store'])}",
            "store": int(fila["Store"]),
            "store_type": tipo,
            "criterio": f"Representativa del tipo {tipo} (venta media más cercana a la mediana del tipo)",
            "nombre": f"Tienda {int(fila['Store'])} (tipo {tipo})",
        })
    elegidas.append({
        "id": f"tienda_{int(top['Store'])}",
        "store": int(top["Store"]),
        "store_type": str(top["StoreType"]),
        "criterio": "Mayor volumen total de ventas de la cadena",
        "nombre": f"Tienda {int(top['Store'])} (tipo {top['StoreType']}, mayor volumen)",
    })
    return elegidas


def construir_serie(df_abiertos: pd.DataFrame, df_completo: pd.DataFrame, store: int | None) -> pd.DataFrame:
    """
    Construye una serie DIARIA continua.

    - store=None → "Total cadena": suma diaria de ventas de todas las tiendas abiertas.
      Promo y SchoolHoliday se expresan como proporción de tiendas abiertas en esa condición,
      y Festivo_a/b/c como proporción de TODAS las tiendas con ese festivo (aunque cerraran),
      porque el festivo explica precisamente por qué cae el total.
    - store=n → serie de la tienda n; las exógenas son 0/1.

    [Completitud] Las fechas sin venta (cierres, p. ej. domingos) se reindexan y las ventas
    se rellenan por interpolación lineal para que ARIMA reciba una serie sin huecos.
    La columna `abierto` (1 = dato real, 0 = interpolado) permite evaluar solo días reales.
    """
    if store is None:
        ab = df_abiertos
        co = df_completo
        serie = ab.groupby("Date").agg(
            ventas=("Sales", "sum"),
            clientes=("Customers", "sum"),
            promo=("Promo", "mean"),
            school_holiday=("SchoolHoliday", "mean"),
        )
        fest = pd.get_dummies(co["StateHoliday"]).reindex(columns=CATEGORIAS_FESTIVO, fill_value=0).astype(float)
        fest["Date"] = co["Date"].values
        fest = fest.groupby("Date").mean()
        state_mode = co.groupby("Date")["StateHoliday"].agg(lambda s: s.value_counts().idxmax())
    else:
        ab = df_abiertos[df_abiertos["Store"] == store]
        co = df_completo[df_completo["Store"] == store]
        serie = ab.set_index("Date")[["Sales", "Customers", "Promo", "SchoolHoliday"]].rename(columns={
            "Sales": "ventas", "Customers": "clientes", "Promo": "promo", "SchoolHoliday": "school_holiday",
        }).astype(float)
        co_idx = co.set_index("Date")
        fest = pd.get_dummies(co_idx["StateHoliday"]).reindex(columns=CATEGORIAS_FESTIVO, fill_value=0).astype(float)
        state_mode = co_idx["StateHoliday"]

    # Reindexado diario sobre todo el rango de fechas
    rango = pd.date_range(df_completo["Date"].min(), df_completo["Date"].max(), freq="D")
    serie = serie.reindex(rango)
    serie.index.name = "fecha"
    serie["abierto"] = serie["ventas"].notna().astype(int)

    # Ventas/clientes: interpolación lineal (y relleno en extremos).
    serie["ventas"] = serie["ventas"].interpolate(method="linear").bfill().ffill()
    serie["clientes"] = serie["clientes"].interpolate(method="linear").bfill().ffill()

    # Exógenas: en días cerrados se toma la información real del calendario (df_completo);
    # si la tienda no tiene registro ese día (hueco de 2014), se asume 0.
    if store is None:
        promo_cal = df_completo.groupby("Date")["Promo"].mean()
        school_cal = df_completo.groupby("Date")["SchoolHoliday"].mean()
    else:
        promo_cal = co.set_index("Date")["Promo"].astype(float)
        school_cal = co.set_index("Date")["SchoolHoliday"].astype(float)
    serie["promo"] = serie["promo"].fillna(promo_cal.reindex(rango)).fillna(0.0)
    serie["school_holiday"] = serie["school_holiday"].fillna(school_cal.reindex(rango)).fillna(0.0)
    fest = fest.reindex(rango).fillna(0.0)
    for c in CATEGORIAS_FESTIVO:
        serie[f"festivo_{c}"] = fest[c].values
    serie["festivo"] = serie[[f"festivo_{c}" for c in CATEGORIAS_FESTIVO]].sum(axis=1).clip(upper=1.0)
    serie["state_holiday"] = state_mode.reindex(rango).fillna("0").values

    # Calendario
    serie["anio"] = serie.index.year
    serie["mes"] = serie.index.month
    serie["dia"] = serie.index.day
    serie["semana_anio"] = serie.index.isocalendar().week.astype(int).values
    serie["dia_semana"] = serie.index.dayofweek
    serie["es_fin_de_mes"] = serie.index.is_month_end.astype(int)
    return serie.reset_index()


def main() -> dict:
    print("=" * 70)
    print("ETAPA 1 · LIMPIEZA DE DATOS")
    print("=" * 70)
    train, store = cargar_datos()
    df_abiertos, df_completo, reporte = limpiar(train, store)

    tiendas = elegir_tiendas(df_abiertos, df_completo)
    catalogo = [{
        "id": "total", "store": None, "store_type": None,
        "criterio": "Suma diaria de ventas de todas las tiendas abiertas", "nombre": "Total cadena",
    }] + tiendas

    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    reporte["series"] = []
    for s in catalogo:
        serie = construir_serie(df_abiertos, df_completo, s["store"])
        ruta = DIR_PROCESSED / f"serie_{s['id']}.csv"
        serie.to_csv(ruta, index=False, float_format="%.6f")
        n_interp = int((serie["abierto"] == 0).sum())
        s["dias"] = int(len(serie))
        s["dias_interpolados"] = n_interp
        reporte["series"].append({"id": s["id"], "archivo": ruta.name, "dias": len(serie), "interpolados": n_interp})

    (DIR_PROCESSED / "series.json").write_text(json.dumps(catalogo, ensure_ascii=False, indent=2), encoding="utf-8")
    (DIR_PROCESSED / "reporte_limpieza.json").write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- Reporte en consola ----------------
    print(f"Filas train original ........... {reporte['filas_train_original']:,}")
    print(f"Filas tras merge con store ..... {reporte['filas_tras_merge']:,}")
    print(f"Duplicados eliminados .......... {reporte['duplicados_eliminados']:,}")
    print(f"StateHoliday originales ........ {reporte['stateholiday_valores_originales']} → ['0','a','b','c']")
    print(f"Registros negativos eliminados . {reporte['registros_negativos_eliminados']:,}")
    print(f"Días cerrados excluidos ........ {reporte['dias_cerrados_excluidos']:,}")
    print(f"Abiertos con venta 0 excluidos . {reporte['dias_abiertos_con_venta_cero_excluidos']:,}")
    print(f"Filas finales (días abiertos) .. {reporte['filas_finales_abiertas']:,}")
    print("Nulos tratados:")
    for k, v in reporte["nulos_tratados"].items():
        print(f"   {k:<28} {v:>8,}")
    o = reporte["outliers_iqr"]
    print(f"Outliers IQR (por tienda) ...... {o['total']:,} ({o['porcentaje']}%), "
          f"{o['en_diciembre']:,} en diciembre, {o['con_promo']:,} con promo → conservados")
    print("Series generadas:")
    for s in catalogo:
        print(f"   {s['id']:<12} {s['nombre']:<38} días={s['dias']} interpolados={s['dias_interpolados']}")
    print(f"Archivos en {DIR_PROCESSED}")
    return reporte


if __name__ == "__main__":
    main()

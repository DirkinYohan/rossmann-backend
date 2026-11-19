"""
Ficha informativa de cada tienda (StoreType, Assortment, CompetitionDistance, Promo2).

Se muestra en la página /prediccion del frontend en una tarjeta de solo lectura: no entra
en ningún modelo. Para "Total cadena" la ficha va vacía ({}).

Fuente, por orden de preferencia:
  1. data/raw/store.csv (el archivo original de Kaggle), si existe;
  2. el campo `info_tienda` del catálogo data/processed/series.json (lo escribe limpiar_datos.py);
  3. lo poco que se sabe sin store.csv: el StoreType del catálogo. El resto queda en null.

Código compartido (lo usan limpieza y los dos exportadores), igual que evaluacion/metricas.py.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RUTA_STORE = RAIZ / "data" / "raw" / "store.csv"
CAMPOS = ["StoreType", "Assortment", "CompetitionDistance", "Promo2"]

_store_cache: pd.DataFrame | None = None


def _leer_store() -> pd.DataFrame | None:
    global _store_cache
    if _store_cache is None and RUTA_STORE.exists():
        _store_cache = pd.read_csv(RUTA_STORE).set_index("Store")
    return _store_cache


def _limpio(v):
    """Convierte tipos de NumPy/pandas a tipos JSON (NaN → None)."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if hasattr(v, "item"):
        v = v.item()
        if isinstance(v, float) and math.isnan(v):
            return None
    return v


def info_desde_store(store: pd.DataFrame | None, id_tienda: int | None) -> dict:
    """Ficha a partir de un DataFrame con las columnas de store.csv indexado por Store."""
    if id_tienda is None or store is None or id_tienda not in store.index:
        return {}
    fila = store.loc[id_tienda]
    return {c: _limpio(fila[c]) if c in fila else None for c in CAMPOS}


def info_tienda(serie: dict) -> dict:
    """Ficha de una entrada del catálogo series.json ({} para Total cadena)."""
    if serie.get("store") is None:
        return {}
    ficha = info_desde_store(_leer_store(), int(serie["store"]))
    if ficha:
        return ficha
    if serie.get("info_tienda"):
        return serie["info_tienda"]
    return {"StoreType": serie.get("store_type"), "Assortment": None, "CompetitionDistance": None, "Promo2": None}

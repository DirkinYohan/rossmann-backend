"""
Métricas de error para pronósticos (código compartido por ambos modelos).

- RMSE : penaliza más los errores grandes (misma unidad que las ventas).
- MAE  : error absoluto medio, fácil de interpretar en € por día.
- MAPE : error porcentual medio. Ignora los días con venta real = 0 (división por cero)
         y se devuelve en porcentaje (%).
"""
from __future__ import annotations

import numpy as np


def _arrays(y_real, y_pred) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y_real, dtype=float).ravel()
    b = np.asarray(y_pred, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"Dimensiones distintas: {a.shape} vs {b.shape}")
    return a, b


def rmse(y_real, y_pred) -> float:
    """Raíz del error cuadrático medio."""
    a, b = _arrays(y_real, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(y_real, y_pred) -> float:
    """Error absoluto medio."""
    a, b = _arrays(y_real, y_pred)
    return float(np.mean(np.abs(a - b)))


def mape(y_real, y_pred) -> float:
    """Error porcentual absoluto medio, en %, excluyendo valores reales iguales a 0."""
    a, b = _arrays(y_real, y_pred)
    mascara = a != 0
    if not mascara.any():
        return float("nan")
    return float(np.mean(np.abs((a[mascara] - b[mascara]) / a[mascara])) * 100)


def calcular_metricas(y_real, y_pred) -> dict:
    """Devuelve las tres métricas redondeadas, listas para exportar a JSON."""
    return {
        "RMSE": round(rmse(y_real, y_pred), 2),
        "MAE": round(mae(y_real, y_pred), 2),
        "MAPE": round(mape(y_real, y_pred), 2),
    }

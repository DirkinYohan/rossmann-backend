"""
MODELO 2 · Red neuronal recurrente GRU (independiente del modelo ARIMA).

Diseño
------
- Ventanas deslizantes: 30 días de historia → predecir el día siguiente.
  El pronóstico a 42 días se hace de forma RECURSIVA (cada predicción entra como
  historia de la siguiente), igual que en el test, para que la evaluación sea honesta.
- Features por paso de tiempo (orden exacto en FEATURES):
    ventas_esc            ventas del día d escaladas con MinMaxScaler (ajustado SOLO con train)
    promo, school_holiday, festivo_a, festivo_b, festivo_c,
    dow_sin, dow_cos, mes_sin, mes_cos
  Las variables exógenas y de calendario de cada paso corresponden al día d+1: son
  conocidas de antemano (promociones planificadas y calendario), así el último paso de la
  ventana lleva la información del día a predecir y los escenarios de marketing influyen
  directamente en la predicción.
  StateHoliday va en one-hot (a, b, c; la categoría "0" es la referencia).
  Día de la semana y mes se codifican con seno/coseno para que el domingo quede "cerca"
  del lunes y diciembre de enero (variables cíclicas).
- Arquitectura: GRU(64, return_sequences=True) → Dropout(0.2) → GRU(32) → Dense(16, relu) → Dense(1).
- EarlyStopping(val_loss, paciencia 10) con validación = 10 % final de las ventanas de train
  (orden temporal, sin mezclar). Semillas fijas.

Salida: models/gru_<serie>.keras, models/gru_<serie>_meta.json, exports/gru_resultados.json

Ejecución independiente:
    python src/modelo_gru/entrenar_gru.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
import sys as _sys
_sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # consola de Windows (cp1252) con tildes y símbolos

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # resultados numéricos reproducibles en CPU

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
from evaluacion.metricas import calcular_metricas  # noqa: E402  (código compartido permitido)

DIR_PROCESSED = RAIZ / "data" / "processed"
DIR_MODELS = RAIZ / "models"
DIR_EXPORTS = RAIZ / "exports"

SEMILLA = 42
VENTANA = 30
HORIZONTE_TEST = 42
FRACCION_VAL = 0.10
EPOCAS = 150
BATCH = 32
FEATURES = [
    "ventas_esc", "promo", "school_holiday", "festivo_a", "festivo_b", "festivo_c",
    "dow_sin", "dow_cos", "mes_sin", "mes_cos",
]
EXOG = FEATURES[1:]  # todo menos las ventas


def fijar_semillas() -> None:
    tf.keras.utils.set_random_seed(SEMILLA)  # fija random, numpy y tensorflow
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:  # versiones sin soporte: se mantiene la semilla
        pass


def cargar_serie(id_serie: str) -> pd.DataFrame:
    return pd.read_csv(DIR_PROCESSED / f"serie_{id_serie}.csv", parse_dates=["fecha"])


def exogenas(df: pd.DataFrame) -> np.ndarray:
    """Matriz (N, 9) con las exógenas y el calendario de cada fecha (orden de EXOG)."""
    dow = df["fecha"].dt.dayofweek.values          # 0 = lunes … 6 = domingo
    mes = df["fecha"].dt.month.values - 1          # 0 = enero … 11 = diciembre
    return np.column_stack([
        df["promo"].values, df["school_holiday"].values,
        df["festivo_a"].values, df["festivo_b"].values, df["festivo_c"].values,
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * mes / 12), np.cos(2 * np.pi * mes / 12),
    ]).astype(np.float32)


def ventana(ventas_esc: np.ndarray, exog: np.ndarray, t: int) -> np.ndarray:
    """
    Entrada para predecir el día t: pasos d = t-30 … t-1 con
    [ventas_esc(d), exógenas(d+1)] → forma (30, 10).
    """
    v = ventas_esc[t - VENTANA:t].reshape(-1, 1)
    e = exog[t - VENTANA + 1:t + 1]
    return np.hstack([v, e]).astype(np.float32)


def construir_modelo(n_features: int) -> tf.keras.Model:
    entrada = tf.keras.Input(shape=(VENTANA, n_features), name="input")
    x = tf.keras.layers.GRU(64, return_sequences=True)(entrada)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.GRU(32)(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    salida = tf.keras.layers.Dense(1, name="output")(x)
    modelo = tf.keras.Model(entrada, salida, name="gru_rossmann")
    modelo.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return modelo


def pronostico_recursivo(modelo, historia_esc: np.ndarray, exog_total: np.ndarray, pasos: int) -> np.ndarray:
    """
    Predice `pasos` días encadenando predicciones. `historia_esc` son las ventas escaladas
    conocidas; `exog_total` debe cubrir historia + pasos.
    """
    ventas = list(historia_esc.astype(np.float32))
    inicio = len(historia_esc)
    for t in range(inicio, inicio + pasos):
        x = ventana(np.asarray(ventas, dtype=np.float32), exog_total, t)[None, ...]
        ventas.append(float(modelo(x, training=False).numpy()[0, 0]))
    return np.asarray(ventas[inicio:], dtype=np.float32)


def entrenar_serie(s: dict) -> dict:
    fijar_semillas()
    df = cargar_serie(s["id"])
    n_train = len(df) - HORIZONTE_TEST
    train, test = df.iloc[:n_train], df.iloc[n_train:]

    # Escalador ajustado SOLO con train para no filtrar información del futuro.
    escalador = MinMaxScaler()
    escalador.fit(train[["ventas"]].values)
    ventas_esc = escalador.transform(df[["ventas"]].values).ravel().astype(np.float32)
    exog = exogenas(df)

    # Ventanas de entrenamiento (objetivo dentro de train) en orden temporal.
    idx = np.arange(VENTANA, n_train)
    X = np.stack([ventana(ventas_esc, exog, t) for t in idx])
    y = ventas_esc[idx]
    n_val = int(len(X) * FRACCION_VAL)
    X_tr, y_tr, X_val, y_val = X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]

    modelo = construir_modelo(X.shape[2])
    t0 = time.time()
    hist = modelo.fit(
        X_tr, y_tr, validation_data=(X_val, y_val),
        epochs=EPOCAS, batch_size=BATCH, shuffle=True, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
    )

    # Test: 42 días recursivos partiendo del último dato real de train.
    pred_esc = pronostico_recursivo(modelo, ventas_esc[:n_train], exog, HORIZONTE_TEST)
    pred = np.clip(escalador.inverse_transform(pred_esc.reshape(-1, 1)).ravel(), 0, None)
    mascara = test["abierto"].values == 1
    metricas = calcular_metricas(test["ventas"].values[mascara], pred[mascara])

    DIR_MODELS.mkdir(parents=True, exist_ok=True)
    modelo.save(DIR_MODELS / f"gru_{s['id']}.keras")
    meta = {
        "id": s["id"], "ventana": VENTANA, "features": FEATURES,
        "escalador_min": float(escalador.data_min_[0]), "escalador_max": float(escalador.data_max_[0]),
    }
    (DIR_MODELS / f"gru_{s['id']}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    epocas = len(hist.history["loss"])
    mejor = int(np.argmin(hist.history["val_loss"])) + 1
    print(f"▶ {s['nombre']:<38} épocas={epocas} (mejor={mejor})  val_loss={min(hist.history['val_loss']):.5f}  "
          f"RMSE={metricas['RMSE']:,.0f}  MAE={metricas['MAE']:,.0f}  MAPE={metricas['MAPE']:.2f}%  ({time.time() - t0:.0f}s)")
    return {
        "nombre": s["nombre"],
        "epocas_entrenadas": epocas, "mejor_epoca": mejor,
        "historial": {
            "loss": [round(float(v), 6) for v in hist.history["loss"]],
            "val_loss": [round(float(v), 6) for v in hist.history["val_loss"]],
        },
        "parametros": int(modelo.count_params()),
        "test_inicio": test["fecha"].iloc[0].strftime("%Y-%m-%d"), "test_fin": test["fecha"].iloc[-1].strftime("%Y-%m-%d"),
        "metricas": metricas,
        "test": [
            {"fecha": f.strftime("%Y-%m-%d"), "real": round(float(v), 2), "abierto": int(a), "pred": round(float(p), 2)}
            for f, v, a, p in zip(test["fecha"], test["ventas"], test["abierto"], pred)
        ],
    }


def main() -> dict:
    print("=" * 70)
    print("ETAPA 4 · MODELO GRU (TensorFlow/Keras)")
    print("=" * 70)
    catalogo = json.loads((DIR_PROCESSED / "series.json").read_text(encoding="utf-8"))
    salida = {"modelo": "GRU", "horizonte_test": HORIZONTE_TEST, "ventana": VENTANA, "features": FEATURES, "series": {}}
    for s in catalogo:
        salida["series"][s["id"]] = entrenar_serie(s)
    DIR_EXPORTS.mkdir(parents=True, exist_ok=True)
    ruta = DIR_EXPORTS / "gru_resultados.json"
    ruta.write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Modelos en {DIR_MODELS} · resultados en {ruta}")
    return salida


if __name__ == "__main__":
    main()

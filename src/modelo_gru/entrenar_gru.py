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
- Las métricas de test (últimos 42 días) se calculan con ese modelo, que NUNCA ve el test.
- PROMO COMO EFECTO MULTIPLICATIVO (modelo híbrido GRU × promo):
  en el histórico de Rossmann la promo alterna casi siempre una semana sí y otra no, así
  que si la red aprende sola el efecto de la promo, en escenarios que nunca vio (promo
  continua varias semanas) el pronóstico recursivo se vuelve inestable: el efecto se
  desvanece, se invierte o se acumula hasta +124 % según el reentrenamiento, cuando en los
  datos una semana con promo vende ~+34 % (total cadena). Por eso:
    · β = efecto medio de la promo en un día laborable abierto, medido en los datos de
      entrenamiento (`estimar_beta`; total cadena ≈ 0,41 → +41 % por día con promo);
    · la red aprende la venta NEUTRALIZADA, venta / (1 + β·promo) (historia y objetivo), y
      su columna `promo` de entrada va en 0: modela tendencia, patrón semanal, mes,
      vacaciones escolares y festivos;
    · venta predicha = venta_neutra · (1 + β·promo del día).
  Así la promo actúa solo sobre su propio día con un efecto estable e interpretable. El
  formato de entrada (30 × 10) no cambia; β se exporta en los metadatos y el frontend
  aplica la misma multiplicación.
- REENTRENAMIENTO FINAL: después de evaluar, se entrena de nuevo la misma arquitectura con
  la serie COMPLETA (train + test), mismos hiperparámetros y tantas épocas como la mejor
  época de la evaluación (ya no queda un tramo para validar). El MinMaxScaler se ajusta con
  la serie completa. Ese modelo final es el que se exporta: el frontend pronostica a partir
  del día siguiente al último dato real (01/08/2015).

Salida: models/gru_<serie>.keras (modelo final, formato Keras 3), models/gru_<serie>_escalador.json,
        exports/gru_resultados.json

Ejecución independiente:
    python src/modelo_gru/entrenar_gru.py               # evaluación + reentrenamiento final
    python src/modelo_gru/entrenar_gru.py --solo-final  # solo el reentrenamiento final, con la mejor
                                                        # época y las métricas de gru_resultados.json
"""
from __future__ import annotations

import argparse
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
COL_PROMO = FEATURES.index("promo")  # columna de la promo dentro de cada paso (va en 0: ver docstring)


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
    [ventas_esc(d), exógenas(d+1)] → forma (30, 10). ventas_esc es la venta neutralizada y la
    columna promo va en 0.
    """
    v = ventas_esc[t - VENTANA:t].reshape(-1, 1)
    e = exog[t - VENTANA + 1:t + 1].copy()
    e[:, COL_PROMO - 1] = 0.0  # la promo no entra a la red: se aplica como (1 + β·promo)
    return np.hstack([v, e]).astype(np.float32)


def estimar_beta(df: pd.DataFrame) -> float:
    """
    Efecto medio de la promo en un día laborable (lun–vie, abierto, sin diciembre):
    venta media con promo / venta media sin promo − 1. Se usa para neutralizar la historia.
    """
    d = df[(df["fecha"].dt.dayofweek < 5) & (df["abierto"] == 1) & (df["fecha"].dt.month != 12)]
    con = d.loc[d["promo"] >= 0.5, "ventas"].mean()
    sin = d.loc[d["promo"] < 0.5, "ventas"].mean()
    if not (np.isfinite(con) and np.isfinite(sin)) or sin <= 0:
        return 0.0
    return float(max(0.0, con / sin - 1))


def ventas_neutras(df: pd.DataFrame, beta: float) -> np.ndarray:
    """Venta sin el efecto de la promo: venta / (1 + β·promo)."""
    return df["ventas"].to_numpy(float) / (1 + beta * df["promo"].to_numpy(float))


def preparar_ventas(df: pd.DataFrame, escalador: MinMaxScaler, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """(entrada, objetivo) escalados con el MISMO MinMax: ambos son la venta neutralizada."""
    neutras = escalador.transform(ventas_neutras(df, beta).reshape(-1, 1)).ravel().astype(np.float32)
    return neutras, neutras


def efecto_promo_continua(modelo, ventas_esc: np.ndarray, df: pd.DataFrame, n_hist: int, beta: float,
                          escalador: tuple[float, float] = (0.0, 1.0), pasos: int = 42) -> list[float]:
    """
    Diagnóstico: efecto (%) por semana de una promo lunes–viernes continua frente a no tener
    promo, pronosticando de forma recursiva desde el día n_hist (sin vacaciones ni festivos).
    """
    futuro = pd.DataFrame({"fecha": pd.date_range(df["fecha"].iloc[n_hist - 1] + pd.Timedelta(days=1), periods=pasos, freq="D")})
    for c in ["school_holiday", "festivo_a", "festivo_b", "festivo_c"]:
        futuro[c] = 0.0
    totales = {}
    for nombre, promo in (("base", lambda f: 0.0), ("promo", lambda f: (f.dt.dayofweek < 5).astype(float))):
        f = futuro.assign(promo=promo(futuro["fecha"]))
        ex = exogenas(pd.concat([df.iloc[:n_hist][f.columns], f], ignore_index=True))
        totales[nombre] = pronostico_recursivo(modelo, ventas_esc[:n_hist], ex, pasos, beta, escalador)
    # Se compara en euros (desescalado), no en la escala 0–1 del MinMax: el desplazamiento del
    # mínimo inflaría el porcentaje.
    mn, mx = escalador
    b, p = (totales[k] * (mx - mn) + mn for k in ("base", "promo"))
    return [round(float(100 * (p[k:k + 7].sum() / b[k:k + 7].sum() - 1)), 1) for k in range(0, pasos, 7)]


def construir_modelo(n_features: int) -> tf.keras.Model:
    entrada = tf.keras.Input(shape=(VENTANA, n_features), name="input")
    # Nombres explícitos: el frontend lee los pesos de model.weights.h5 por capa.
    # GRU usa la convención de Keras 3: reset_after=True, gates z, r, h, sigmoide/tanh.
    x = tf.keras.layers.GRU(64, return_sequences=True, name="gru_1")(entrada)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    x = tf.keras.layers.GRU(32, name="gru_2")(x)
    x = tf.keras.layers.Dense(16, activation="relu", name="dense_1")(x)
    salida = tf.keras.layers.Dense(1, name="output")(x)
    modelo = tf.keras.Model(entrada, salida, name="gru_rossmann")
    modelo.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return modelo


def aplicar_promo_esc(neutra_esc: float, promo: float, beta: float, escalador_min: float, escalador_max: float) -> float:
    """Venta neutralizada predicha (escalada) → venta real (escalada): multiplica por (1 + β·promo)."""
    rango = escalador_max - escalador_min
    venta = (neutra_esc * rango + escalador_min) * (1 + beta * promo)
    return (venta - escalador_min) / rango


def pronostico_recursivo(modelo, historia_esc: np.ndarray, exog_total: np.ndarray, pasos: int, beta: float,
                         escalador: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """
    Predice `pasos` días encadenando predicciones. `historia_esc` son las ventas NEUTRALIZADAS
    escaladas conocidas; `exog_total` debe cubrir historia + pasos. La red predice la venta
    neutralizada, que entra tal cual en la ventana; la salida es la venta REAL (escalada) =
    neutra · (1 + β·promo del día, columna 0 de las exógenas). `escalador` = (min, max).
    """
    mn, mx = escalador
    ventas = list(historia_esc.astype(np.float32))
    reales = []
    inicio = len(historia_esc)
    for t in range(inicio, inicio + pasos):
        x = ventana(np.asarray(ventas, dtype=np.float32), exog_total, t)[None, ...]
        neutra = float(modelo(x, training=False).numpy()[0, 0])
        ventas.append(neutra)
        reales.append(aplicar_promo_esc(neutra, float(exog_total[t, 0]), beta, mn, mx))
    return np.asarray(reales, dtype=np.float32)


def reentrenar_final(id_serie: str, epocas: int) -> dict:
    """
    Entrena con TODAS las ventanas de la serie completa durante `epocas` épocas (la mejor
    época de la evaluación) y guarda el .keras y el escalador que usa el frontend.
    """
    fijar_semillas()
    df = cargar_serie(id_serie)
    escalador = MinMaxScaler()
    escalador.fit(df[["ventas"]].values)
    beta = estimar_beta(df)
    entrada, objetivo = preparar_ventas(df, escalador, beta)
    exog = exogenas(df)
    idx = np.arange(VENTANA, len(df))
    X = np.stack([ventana(entrada, exog, t) for t in idx])
    y = objetivo[idx]

    modelo = construir_modelo(X.shape[2])
    hist = modelo.fit(X, y, epochs=epocas, batch_size=BATCH, shuffle=True, verbose=0)
    mm = (float(escalador.data_min_[0]), float(escalador.data_max_[0]))
    efecto = efecto_promo_continua(modelo, entrada, df, len(df), beta, escalador=mm)

    DIR_MODELS.mkdir(parents=True, exist_ok=True)
    ruta = DIR_MODELS / f"gru_{id_serie}.keras"
    modelo.save(ruta)
    meta = {
        "id": id_serie, "ventana": VENTANA, "features": FEATURES,
        "escalador_min": float(escalador.data_min_[0]), "escalador_max": float(escalador.data_max_[0]),
        "ajuste_promo_beta": beta,
        "entrenado_hasta": df["fecha"].iloc[-1].strftime("%Y-%m-%d"),
    }
    (DIR_MODELS / f"gru_{id_serie}_escalador.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "entrenado_desde": df["fecha"].iloc[0].strftime("%Y-%m-%d"),
        "entrenado_hasta": meta["entrenado_hasta"],
        "ventanas": int(len(X)), "epocas": int(epocas),
        "loss_final": round(float(hist.history["loss"][-1]), 6),
        "efecto_promo_continua_por_semana": efecto,
        "escalador_min": meta["escalador_min"], "escalador_max": meta["escalador_max"],
        "ajuste_promo_beta": round(beta, 4),
        "archivo": ruta.name,
    }


def entrenar_serie(s: dict) -> dict:
    fijar_semillas()
    df = cargar_serie(s["id"])
    n_train = len(df) - HORIZONTE_TEST
    train, test = df.iloc[:n_train], df.iloc[n_train:]

    # Escalador ajustado SOLO con train para no filtrar información del futuro.
    escalador = MinMaxScaler()
    escalador.fit(train[["ventas"]].values)
    beta = estimar_beta(train)  # efecto de la promo estimado SOLO con train
    entrada, objetivo = preparar_ventas(df, escalador, beta)
    mm = (float(escalador.data_min_[0]), float(escalador.data_max_[0]))
    exog = exogenas(df)

    # Ventanas de entrenamiento (objetivo dentro de train) en orden temporal.
    idx = np.arange(VENTANA, n_train)
    X = np.stack([ventana(entrada, exog, t) for t in idx])
    y = objetivo[idx]
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
    pred_esc = pronostico_recursivo(modelo, entrada[:n_train], exog, HORIZONTE_TEST, beta, escalador=mm)
    pred = np.clip(escalador.inverse_transform(pred_esc.reshape(-1, 1)).ravel(), 0, None)
    mascara = test["abierto"].values == 1
    metricas = calcular_metricas(test["ventas"].values[mascara], pred[mascara])

    epocas = len(hist.history["loss"])
    mejor = int(np.argmin(hist.history["val_loss"])) + 1
    efecto = efecto_promo_continua(modelo, entrada, df, n_train, beta, escalador=mm)
    print(f"▶ {s['nombre']:<38} épocas={epocas} (mejor={mejor})  val_loss={min(hist.history['val_loss']):.5f}  "
          f"RMSE={metricas['RMSE']:,.0f}  MAE={metricas['MAE']:,.0f}  MAPE={metricas['MAPE']:.2f}%  ({time.time() - t0:.0f}s)")
    print(f"    β promo={beta:.3f} · efecto de una promo lun–vie continua por semana (modelo de evaluación): {efecto}")
    final = reentrenar_final(s["id"], mejor)
    print(f"    reentrenamiento final con la serie completa (hasta {final['entrenado_hasta']}, {final['epocas']} épocas) → {final['archivo']}"
          f" · efecto promo por semana: {final['efecto_promo_continua_por_semana']}")
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
        "efecto_promo_continua_por_semana": efecto,
        "ajuste_promo_beta": round(beta, 4),
        "modelo_final": final,
        "test": [
            {"fecha": f.strftime("%Y-%m-%d"), "real": round(float(v), 2), "abierto": int(a), "pred": round(float(p), 2)}
            for f, v, a, p in zip(test["fecha"], test["ventas"], test["abierto"], pred)
        ],
    }


def solo_final() -> dict:
    """Reentrenamiento final reutilizando la mejor época y las métricas ya calculadas."""
    salida = json.loads((DIR_EXPORTS / "gru_resultados.json").read_text(encoding="utf-8"))
    for id_serie, r in salida["series"].items():
        t0 = time.time()
        r["modelo_final"] = reentrenar_final(id_serie, r["mejor_epoca"])
        print(f"▶ {r['nombre']:<38} reentrenado hasta {r['modelo_final']['entrenado_hasta']} con {r['mejor_epoca']} épocas "
              f"· MAPE test (sin cambios)={r['metricas']['MAPE']:.2f}% ({time.time() - t0:.0f}s)")
    return salida


def main() -> dict:
    parser = argparse.ArgumentParser(description="Entrenamiento GRU")
    parser.add_argument("--solo-final", action="store_true",
                        help="solo reentrena con la serie completa (usa la mejor época de gru_resultados.json)")
    args = parser.parse_args()
    print("=" * 70)
    print("ETAPA 4 · MODELO GRU (TensorFlow/Keras)" + (" · SOLO REENTRENAMIENTO FINAL" if args.solo_final else ""))
    print("=" * 70)
    if args.solo_final:
        salida = solo_final()
    else:
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

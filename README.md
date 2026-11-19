# Rossmann · Backend (Python)

Este backend entrena y evalúa dos modelos de series de tiempo **independientes**, SARIMAX y GRU, para predecir las ventas diarias de Rossmann. Luego exporta los modelos y los resultados al frontend (`../rossmann-frontend/public/`).

Todo el entrenamiento ocurre aquí. El frontend solo lee archivos estáticos.

## Requisitos

- **Python 3.11 o 3.12.** TensorFlow 2.19 no publica ruedas estables para Python 3.13 o 3.14.
- Unos 3 GB de espacio libre (TensorFlow) y una CPU. No hace falta GPU.
- El dataset de Kaggle [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales): copia `train.csv` y `store.csv` en `data/raw/`.

## Instalación

```bash
cd rossmann-backend
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

> Si tu Python del sistema es 3.13 o superior, crea el entorno con 3.12 usando [uv](https://docs.astral.sh/uv/):
> `pip install uv` → `uv venv --python 3.12 .venv` → `uv pip install --python .venv -r requirements.txt`.

## Ejecución

```bash
python run_all.py
```

Ejecuta en orden: **limpieza → estacionalidad → ARIMA → GRU → comparación → copia al frontend**. Tarda de 10 a 15 minutos en una CPU normal; casi todo el tiempo se va en la búsqueda de `auto_arima`. Opciones:

```bash
python run_all.py --desde gru      # retoma desde una etapa (limpieza|estacionalidad|arima|gru|comparacion)
python run_all.py --solo-copiar    # solo copia exports/ y los .onnx a ../rossmann-frontend/public/
```

Cada etapa también se puede ejecutar sola:

```bash
python src/limpieza/limpiar_datos.py
python src/estacionalidad/analisis_estacionalidad.py
python src/modelo_arima/entrenar_arima.py
python src/modelo_arima/exportar_arima.py
python src/modelo_gru/entrenar_gru.py
python src/modelo_gru/exportar_gru.py
python src/evaluacion/comparar_modelos.py
```

## Estructura y responsabilidades

```
src/
├── limpieza/limpiar_datos.py             (compartido) datos crudos → series diarias limpias
├── estacionalidad/analisis_estacionalidad.py   módulo separado: STL, ADF/KPSS, ACF/PACF, efectos
├── modelo_arima/                          MODELO 1: no importa nada del GRU
│   ├── entrenar_arima.py                  auto_arima → SARIMAX → test → models/arima_<serie>.pkl
│   └── exportar_arima.py                  escenarios precalculados → exports/arima_escenarios.json
├── modelo_gru/                            MODELO 2: no importa nada del ARIMA
│   ├── entrenar_gru.py                    ventanas 30→1 → GRU → test recursivo → models/gru_<serie>.keras
│   └── exportar_gru.py                    Keras → ONNX + escalador + últimos 30 días → exports/gru_config.json
└── evaluacion/                            (compartido)
    ├── metricas.py                        rmse, mae, mape
    └── comparar_modelos.py                tabla comparativa → exports/comparacion_metricas.json (+ este README)
```

| Archivo generado | Destino en el frontend |
|---|---|
| `exports/estacionalidad.json`, `arima_escenarios.json`, `arima_resultados.json`, `gru_resultados.json`, `comparacion_metricas.json`, `data/processed/series.json` | `public/data/` |
| `models/gru_<serie>.onnx`, `exports/gru_config.json` | `public/models/` |

## Series modeladas

- **Total cadena:** la suma diaria de las ventas de todas las tiendas abiertas.
- **Una tienda por `StoreType` (a, b, c, d):** el script elige, entre las tiendas con historia completa (180 tiendas no tienen datos del segundo semestre de 2014), la de venta media más cercana a la mediana de su tipo.
- **La tienda con mayor volumen total de ventas.**

La elección es automática. La consola la imprime y queda guardada en `data/processed/series.json`. La tabla de resultados de abajo también muestra qué tiendas salieron.

## Limpieza de datos (`limpiar_datos.py`)

| Paso | Dimensión de calidad |
|---|---|
| Unir `train` con `store` por `Store` (left join) | Integración |
| `Date` → `datetime` y orden cronológico | Consistencia / temporalidad |
| `drop_duplicates()` y unicidad de (Store, Date) | Unicidad |
| `StateHoliday` unificado a `0, a, b, c` (venía mezclado: `0` numérico y `"0"` texto) | Consistencia |
| `CompetitionDistance` → mediana; `Promo2Since*` y `CompetitionOpenSince*` → 0 | Completitud |
| Ventas y clientes ≥ 0 | Validez |
| **Excluir `Open == 0`** y los días abiertos con venta 0. Si se dejan, sesgan el modelo hacia abajo y el MAPE se vuelve infinito | Exactitud |
| Outliers por tienda con IQR: se **reportan, no se borran**. La mayoría son picos de diciembre o de promociones, es decir, demanda real | Exactitud |
| Variables de calendario: año, mes, día, semana, día de la semana, `es_fin_de_mes` | Temporalidad |
| Reindexado diario + interpolación lineal de los días cerrados; la columna `abierto` marca qué días son datos reales | Completitud |

**Las métricas se calculan solo con los días realmente abiertos** (`abierto == 1`). Los valores interpolados existen solo para que ARIMA reciba una serie continua; no son observaciones.

## División de datos

- **Test = los últimos 42 días** (6 semanas, el horizonte de la competencia). Train = todo lo anterior. **Sin mezclar los datos al azar.**
- El periodo de test es exactamente el mismo para los dos modelos, y `comparar_modelos.py` lo verifica antes de comparar.
- Ambos modelos pronostican los 42 días **de una sola vez desde el final de train**: ninguno ve valores reales del test.

## Modelo 1 · SARIMAX (`modelo_arima/`)

- `statsmodels.SARIMAX` con estacionalidad semanal (m = 7).
- Variables exógenas: `promo`, `school_holiday` y `festivo`, que es `StateHoliday` codificado como indicador 0/1. Las categorías b (Pascua) y c (Navidad) aparecen tan pocas veces que, por separado, harían la matriz exógena casi singular.
- Selección de órdenes con `pmdarima.auto_arima` (stepwise, criterio **AIC**) con límites p, q ≤ 3, P, Q ≤ 1 y d ≤ 1 (decidido con KPSS). **D = 1 está fijo.**
- Para el frontend, el modelo se actualiza con los 42 días de test (`append(refit=False)`: conserva los parámetros) y se precalculan **30 trayectorias** con su IC del 95 %: 3 opciones de promo × 2 de vacaciones × (sin festivo + festivo a mitad de cada horizonte de 7, 14, 28 y 42 días).

### Por qué esos parámetros (ACF/PACF y ADF/KPSS)

La página **Visualización temporal** del frontend muestra todas estas gráficas y pruebas.

1. **Periodo estacional m = 7.** La ACF tiene picos significativos y persistentes en los rezagos 7, 14, 21… hasta el 60, y la STL con periodo 7 explica la mayor parte de la varianza (ver "fuerza estacional"). La razón es el ciclo semanal: los domingos casi todas las tiendas cierran y la venta cae.
2. **D = 1.** Que esos picos decaigan tan lento indica estacionalidad no estacionaria. Tras la diferencia de lag 7, ADF y KPSS coinciden en que la serie es estacionaria. Con la prueba OCSB de `pmdarima`, `auto_arima` tiende a elegir D = 0, y entonces el SARIMAX no reproduce la caída de los domingos y el error en la serie total se dispara. Por eso D se fija en 1.
3. **d.** En la serie original, ADF suele rechazar la raíz unitaria mientras KPSS rechaza la estacionariedad en nivel. Eso indica una tendencia o cambios de nivel, no una caminata aleatoria pura. Por eso se permite d ∈ {0, 1} y decide la prueba KPSS dentro de `auto_arima`. Una vez aplicada D = 1, casi siempre basta d = 0.
4. **p, q ≤ 3 y P, Q ≤ 1.** La PACF corta después de los primeros rezagos (un AR de orden bajo) y conserva un pico en el lag 7. Tras la diferencia estacional, eso se captura con un término SAR/SMA de orden 1. Con límites más amplios, el AIC apenas baja y el tiempo de búsqueda se multiplica.
5. **Exógenas.** El coeficiente estimado de `promo` (en la tabla de abajo) cuantifica en euros diarios el uplift que el análisis de efectos mide de forma descriptiva.

## Modelo 2 · GRU (`modelo_gru/`)

```
Input (30 pasos × 10 features)
 → GRU(64, return_sequences=True)
 → Dropout(0.2)
 → GRU(32)
 → Dense(16, relu)
 → Dense(1)          # venta escalada del día siguiente
```

- **Features por paso** (orden exacto): `ventas_esc, promo, school_holiday, festivo_a, festivo_b, festivo_c, dow_sin, dow_cos, mes_sin, mes_cos`.
  - `ventas_esc` usa un `MinMaxScaler` ajustado **solo con train**.
  - `StateHoliday` va en one-hot; la categoría "0" es la referencia.
  - El día de la semana y el mes van en seno/coseno porque son cíclicos: el domingo queda "cerca" del lunes y diciembre de enero.
  - En cada paso, las exógenas son las del **día siguiente** (d + 1). Promociones y calendario se conocen de antemano, así que el último paso de la ventana lleva la información del día que se predice. Gracias a eso, los escenarios de marketing influyen directamente en la predicción.
- **Pronóstico recursivo a 42 días:** cada predicción entra como historia de la siguiente.
- **Entrenamiento:** Adam (lr 1e-3), pérdida MSE, batch 32, hasta 150 épocas. `EarlyStopping(val_loss, patience=10, restore_best_weights=True)`, con validación sobre el 10 % final de las ventanas de train en orden temporal. Semillas fijas con `tf.keras.utils.set_random_seed(42)` y determinismo de operaciones.
- **Exportación:** `tf2onnx` con opset 13. Se comprueba que ONNX y Keras den la misma salida (error < 1e-4). Además, se calcula el pronóstico base con `onnxruntime`, replicando exactamente lo que hace el navegador, como referencia de validación.

## Resultados

<!-- RESULTADOS:INICIO -->
_Generado automáticamente por `comparar_modelos.py`. Test: 2015-06-20 → 2015-07-31 (42 días)._

### Métricas en el periodo de test

| Serie | RMSE ARIMA | RMSE GRU | MAE ARIMA | MAE GRU | MAPE ARIMA | MAPE GRU | Mejor |
|---|---:|---:|---:|---:|---:|---:|:---:|
| Total cadena | 744.197 | **660.473** | 542.851 | **443.919** | **8,83 %** | 15,25 % | GRU |
| Tienda 770 (tipo a) | 1.832 | **1.527** | 1.706 | **1.418** | 39,32 % | **32,29 %** | GRU |
| Tienda 1097 (tipo b) | **672** | 1.499 | **450** | 1.330 | **4,00 %** | 12,12 % | ARIMA |
| Tienda 1044 (tipo c) | 1.045 | **1.036** | 792 | **750** | 13,44 % | **12,33 %** | GRU |
| Tienda 352 (tipo d) | **732** | 760 | **626** | 653 | **9,32 %** | 9,57 % | ARIMA |
| Tienda 262 (tipo b, mayor volumen) | **1.342** | 1.858 | **1.034** | 1.580 | **4,70 %** | 7,64 % | ARIMA |

MAPE promedio: ARIMA 13,27 % · GRU 14,87 % · Victorias por métrica: ARIMA 10, GRU 8 · **Mejor modelo general: ARIMA**

### Órdenes SARIMAX elegidos y evidencia

| Serie | Orden (p,d,q)(P,D,Q,7) | AIC | ADF p (orig.) | KPSS p (orig.) | ACF(7) | Coef. Promo | GRU épocas |
|---|---|---:|---:|---:|---:|---:|---:|
| Total cadena | (3,0,3)(1,1,1,7) | 26.827 | 0.0001 | 0.1 | 0.5466 | 2.442.621 | 50 |
| Tienda 770 (tipo a) | (1,0,1)(1,1,1,7) | 14.788 | 0.004 | 0.031 | 0.0773 | 2.449 | 18 |
| Tienda 1097 (tipo b) | (1,0,1)(1,1,1,7) | 14.390 | 0.0046 | 0.01 | 0.5255 | 1.207 | 28 |
| Tienda 1044 (tipo c) | (1,0,0)(1,1,0,7) | 14.921 | 0.0 | 0.1 | -0.1624 | 2.401 | 33 |
| Tienda 352 (tipo d) | (1,0,0)(1,1,0,7) | 14.629 | 0.0 | 0.1 | 0.0181 | 1.189 | 37 |
| Tienda 262 (tipo b, mayor volumen) | (1,0,1)(0,1,1,7) | 15.947 | 0.0 | 0.1 | 0.5246 | 2.966 | 56 |
<!-- RESULTADOS:FIN -->

### Lectura de los resultados

- **ARIMA gana en MAPE promedio** y en las tiendas con un patrón semanal estable (1097, 262 y 352). El GRU gana en RMSE/MAE de la serie total y en las tiendas 770 y 1044. Ninguno domina en todo: es la razón para mantener los dos en la app.
- **La tienda 770 tiene el error más alto con ambos modelos, y la causa está en los datos.** `store.csv` registra un competidor a **100 m abierto en abril de 2015** (`CompetitionDistance = 100`, `CompetitionOpenSince = 4/2015`). Desde entonces, la venta media cayó de ~7.000 € a ~4.800 € por día. Los dos modelos aprendieron casi solo el nivel anterior y sobreestiman el test. Para corregirlo habría que añadir como exógena "competencia abierta" o reentrenar con datos posteriores al cambio. Se deja así para mostrar un límite real de los pronósticos basados solo en la historia.
- **Las tiendas b (1097 y 262) abren los domingos** y su mejor día es el domingo. Por eso tienen la estacionalidad semanal más fuerte, y SARIMAX con D = 1 las modela muy bien (MAPE de ~4–5 %).
- **Las tiendas que cierran los domingos** (770, 1044, 352) tienen un ACF(7) bajo. La interpolación de los domingos suaviza ese día, y las promociones de Rossmann alternan semanas, lo que crea un ciclo de 14 días.

## Métricas (`evaluacion/metricas.py`)

- **RMSE**: la raíz del error cuadrático medio. Penaliza los errores grandes.
- **MAE**: el error absoluto medio, en euros por día.
- **MAPE**: el error porcentual medio, en %. Ignora los días con venta real 0.

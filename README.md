# Rossmann · Backend (Python)

Este backend **solo entrena, evalúa y exporta**. Se ejecuta **una sola vez** y genera archivos:

| Artefacto | Formato | Destino en el frontend |
|---|---|---|
| Modelo ARIMA (SARIMAX) | `.joblib` | `public/models/arima_<serie>.joblib` |
| Modelo GRU | `.keras` (Keras 3) | `public/models/gru_<serie>.keras` |
| Informe de estacionariedad | `.json` | `public/data/estacionalidad_informe.json` |
| Métricas de evaluación | `.json` | `public/data/metricas.json` |
| Datos de apoyo | `.json` | `public/data/series.json`, `historicos.json`, `arima_meta_<serie>.json`, `gru_meta_<serie>.json`, `gru_referencia_<serie>.json` |

Después, **el frontend ejecuta los modelos por sí mismo en el navegador**: ARIMA con Pyodide y GRU con un forward pass en TypeScript. No hay API, ni servidor Python, ni llamadas HTTP al backend. Una vez exportados los artefactos, este backend puede estar apagado o no existir.

## Requisitos

- **Python 3.12 o 3.13.**
- El dataset de Kaggle [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales): copia `train.csv` y `store.csv` en `data/raw/`.
- Unos 3 GB libres (TensorFlow). Basta con una CPU.

### ⚠ Compatibilidad con Pyodide (crítico)

Un `.joblib` de statsmodels es un *pickle*. Solo se carga bien si las librerías del navegador son **las mismas** que se usaron para entrenar. El frontend usa **Pyodide 314.0.8** (Python 3.14.2). Por eso `requirements.txt` fija exactamente las versiones que trae esa versión de Pyodide (según su [`pyodide-lock.json`](https://cdn.jsdelivr.net/pyodide/v314.0.8/full/pyodide-lock.json)):

| Librería | Versión (backend = Pyodide 314.0.8) |
|---|---|
| numpy | 2.4.6 |
| scipy | 1.18.0 |
| pandas | 3.0.2 |
| statsmodels | 0.14.6 |
| joblib | 1.5.3 |

TensorFlow 2.21.0 es la versión compatible con numpy 2.4.6. **Si cambias la versión de Pyodide** (`PYODIDE_VERSION` en `rossmann-frontend/public/workers/arima-worker.mjs`), actualiza estas cinco versiones y vuelve a entrenar.

## Instalación

```bash
cd rossmann-backend
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

> Si tu Python del sistema es 3.14 o superior, crea el entorno con 3.13 usando [uv](https://docs.astral.sh/uv/):
> `pip install uv` → `uv venv --python 3.13 .venv` → `uv pip install --python .venv -r requirements.txt`.

## Ejecución (una sola vez)

```bash
python run_all.py
```

Ejecuta en orden: **limpieza → estacionalidad → ARIMA → GRU → métricas → copia al frontend**. La etapa de copia borra los artefactos viejos de `../rossmann-frontend/public/{data,models}` y copia todos los `.joblib`, `.keras` y `.json`. Tarda unos 10 minutos en una CPU normal.

```bash
python run_all.py --desde gru      # retoma desde una etapa (limpieza|estacionalidad|arima|gru|metricas)
python run_all.py --solo-copiar    # solo copia los artefactos al frontend
python run_all.py --solo-final     # solo reentrena ARIMA y GRU con la serie completa, exporta y copia
```

`--solo-final` no repite la evaluación: reutiliza el orden SARIMAX, la mejor época del GRU y las métricas que ya están en `exports/*_resultados.json` (así las métricas de test no cambian) y solo regenera los modelos finales y sus metadatos. Tarda ~1 minuto.

Cada etapa también se puede ejecutar sola:

```bash
python src/limpieza/limpiar_datos.py
python src/estacionalidad/analisis_estacionalidad.py
python src/modelo_arima/entrenar_arima.py      # añade --solo-final para solo reentrenar
python src/modelo_arima/exportar_arima.py
python src/modelo_gru/entrenar_gru.py          # añade --solo-final para solo reentrenar
python src/modelo_gru/exportar_gru.py
python src/evaluacion/comparar_modelos.py
```

### Prueba: el frontend funciona sin el backend

1. Ejecuta `python run_all.py` una vez.
2. Cierra la terminal del backend. Si quieres comprobarlo a fondo, mueve o renombra la carpeta `rossmann-backend/`.
3. En `rossmann-frontend/` ejecuta `npm run dev`, o `npm run build:estatico` y publica `out/` en cualquier hosting de archivos.
4. Abre `/dashboard`, `/escenarios`, `/prediccion` y `/temporal`. Las cuatro páginas funcionan:
   - el panel **Verificación de la ejecución en el navegador** del dashboard muestra ARIMA (Pyodide) y GRU (TypeScript) con ✓;
   - la pestaña Red del navegador solo muestra peticiones al propio sitio y al CDN de Pyodide.

## Estructura y responsabilidades

```
src/
├── limpieza/limpiar_datos.py              (compartido) datos crudos → series diarias limpias + historicos.json
├── limpieza/info_tienda.py                (compartido) ficha StoreType/Assortment/CompetitionDistance/Promo2
├── estacionalidad/analisis_estacionalidad.py    módulo separado → estacionalidad_informe.json
├── modelo_arima/                          MODELO 1: no importa nada del GRU
│   ├── entrenar_arima.py                  auto_arima → SARIMAX → test → reentreno con todo → models/arima_<serie>.joblib
│   └── exportar_arima.py                  recarga y verifica el .joblib → arima_meta_<serie>.json
├── modelo_gru/                            MODELO 2: no importa nada del ARIMA
│   ├── entrenar_gru.py                    ventanas 30→1 → GRU → test recursivo → reentreno con todo → models/gru_<serie>.keras
│   └── exportar_gru.py                    gru_meta_<serie>.json + gru_referencia_<serie>.json
└── evaluacion/                            (compartido)
    ├── metricas.py                        rmse, mae, mape
    └── comparar_modelos.py                metricas.json (+ la sección de resultados de este README)
```

## Series modeladas

- **Total cadena** (`total_cadena`): la suma diaria de las ventas de todas las tiendas abiertas.
- **Una tienda por `StoreType` (a, b, c, d):** el script elige, entre las tiendas con historia completa (180 tiendas no tienen datos del segundo semestre de 2014), la de venta media más cercana a la mediana de su tipo.
- **La tienda con mayor volumen total.**

Con los datos de Kaggle salieron las tiendas **770 (a), 1097 (b), 1044 (c), 352 (d) y 262 (la de mayor volumen, tipo b)**. La lista queda en `exports/series.json`.

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

**Las métricas se calculan solo con los días realmente abiertos.** Los valores interpolados existen solo para que ARIMA reciba una serie continua.

## División de datos

- **Test = los últimos 42 días** (20 jun – 31 jul 2015). Train = todo lo anterior. **Sin mezclar los datos al azar.**
- El periodo de test es el mismo para los dos modelos, y `comparar_modelos.py` lo verifica.
- Ambos modelos pronostican los 42 días **de una sola vez desde el final de train**: ninguno ve valores reales del test.

### Reentrenamiento final (modelos que usa el frontend)

Las métricas son las oficiales y salen de modelos que **nunca vieron el test**. Después de evaluar, cada script vuelve a entrenar el modelo con la **serie completa** (1 ene 2013 – 31 jul 2015) con los mismos hiperparámetros:

| Modelo | Qué se conserva | Qué cambia |
|---|---|---|
| ARIMA | El orden (p,d,q)(P,D,Q,7) elegido por `auto_arima` en train y las mismas exógenas | Se reajustan los parámetros con los 942 días |
| GRU | Arquitectura, ventana, batch, optimizador y semilla. Épocas = la mejor época de la evaluación (ya no queda un tramo para validar) | El `MinMaxScaler` se ajusta con la serie completa |

Esa versión reentrenada es la que se exporta como `arima_<serie>.joblib` y `gru_<serie>.keras`. Así **la predicción empieza justo el día siguiente al último dato (01/08/2015)**, sin tener que extender el modelo con el test en el navegador. Los metadatos de cada serie incluyen además:

| Clave | Archivo | Contenido |
|---|---|---|
| `ultima_fecha` | `arima_meta_` y `gru_meta_` | Último día del dataset (2015-07-31) |
| `ventana_gru` | `gru_meta_` | Los últimos 30 días: ventas (interpoladas en días cerrados), ventas escaladas, `abierto`, `promo`, `school_holiday`, `festivo_a/b/c`, `state_holiday` |
| `info_tienda` | `arima_meta_` y `gru_meta_` | `StoreType`, `Assortment`, `CompetitionDistance`, `Promo2` (solo informativos; `{}` en Total cadena) |
| `mape_test` | `arima_meta_` y `gru_meta_` | MAPE del modelo en los 42 días de test |
| `dias_ancla`, `fecha_ancla` | `arima_meta_` | El `.joblib` compacto guarda los últimos 30 días (desde `fecha_ancla`); `/prediccion` los sustituye por el historial del usuario |

> `info_tienda` se lee de `data/raw/store.csv`. Si ese archivo no está, solo se rellena `StoreType` (que viene en `series.json`) y el resto queda en `null`; la página `/prediccion` lo muestra como "No disponible". Para completarla basta con copiar `store.csv` en `data/raw/` y ejecutar `python run_all.py --solo-final`.

## Informe de estacionariedad (`estacionalidad/`)

Es un módulo **independiente de los modelos**. Genera `estacionalidad_informe.json` con, por serie:

- **ADF y KPSS:** estadístico, p-valor, rezagos, valores críticos (1 %, 5 %, 10 %) y veredicto.
- **Conclusión en español**, que interpreta las dos pruebas juntas. Sus hipótesis nulas son opuestas, así que hay cuatro casos:

  | ADF | KPSS | Conclusión |
  |---|---|---|
  | estacionaria | estacionaria | **Estacionaria** (d = 0) |
  | no estacionaria | no estacionaria | **No estacionaria**: raíz unitaria (d = 1) |
  | no estacionaria | estacionaria | **Estacionaria en tendencia**: quitar la tendencia en vez de diferenciar |
  | estacionaria | no estacionaria | **Estacionaria en diferencias** (d = 1) |

- **Diferenciación recomendada** `{d, D, periodo}`, y las pruebas repetidas tras la diferencia estacional de lag 7.
- **STL** (periodo 7) y la fuerza de la estacionalidad, los **promedios** por día de la semana y por mes, **ACF/PACF** hasta el rezago 60 con su banda del 95 %, y el **efecto de Promo, SchoolHoliday y StateHoliday**.

El frontend solo **presenta** este informe en `/temporal`; no recalcula nada.

## Modelo 1 · SARIMAX → `.joblib` (`modelo_arima/`)

- `statsmodels.SARIMAX` con estacionalidad semanal (m = 7).
- Variables exógenas, **en este orden**: `promo`, `school_holiday` y `festivo` (`StateHoliday` codificado como indicador 0/1).
- Órdenes elegidos con `pmdarima.auto_arima` (stepwise, criterio **AIC**) con p, q ≤ 3, P, Q ≤ 1, d ≤ 1 (decidido con KPSS) y **D = 1 fijo**.
- **Guardado con `joblib.dump(..., compress=3)`.**

> **Por qué no `remove_data=True`.** Lo probé, y con statsmodels 0.14.6 `get_forecast` falla después de quitar los datos: necesita `model.endog` para calcular el índice de la predicción (`TypeError: object of type 'NoneType' has no len()`).
>
> En su lugar, `entrenar_arima.py` **reancla** el modelo: crea un SARIMAX con los mismos órdenes solo sobre los últimos 30 días y le pone como estado inicial "conocido" el estado del filtro de Kalman del modelo completo en ese punto. Al filtrar esos 30 días con los mismos parámetros, llega exactamente al mismo estado final, así que pronostica igual: la diferencia en media e IC 95 % es 0,0, y el script lo comprueba antes de guardar.
>
> Los 30 días coinciden con el historial de `/prediccion`: si el usuario ingresa sus ventas, el navegador hace `results.apply(endog, exog, copy_initialization=True)`, que parte del estado entrenado en ese primer día y filtra las ventas del usuario en lugar de las del dataset. Con las ventas del dataset se obtiene exactamente el pronóstico base (comprobado: diferencia 0,0). Con una inicialización difusa (el valor por defecto de `apply`), 30 datos no bastan para los hasta 18 estados del SARIMAX estacional y el pronóstico sale absurdo (negativo en Total cadena).
>
> Así, cada archivo pesa **100–340 KB, frente a 32–45 MB** del modelo completo.

`exportar_arima.py` recarga cada `.joblib` (el modelo final), comprueba que termina en el último día de la serie y escribe `arima_meta_<serie>.json` con:

- orden y orden estacional;
- nombres y orden de las exógenas;
- frecuencia (`D`), `ultima_fecha`, `dias_ancla` y `fecha_ancla`;
- `mape_test`, `metricas_test` e `info_tienda`;
- `referencia`: un pronóstico de 42 días (promo de lunes a viernes) que da Python con ese `.joblib`; el dashboard lo compara con Pyodide.

En el navegador:

- **Futuro (escenarios y dashboard):** `get_forecast(h, exog=X_escenario)` con IC 95 %, directamente desde el 01/08/2015.
- **Historial propio (`/prediccion`):** `results.apply(endog=ventas_usuario, exog=exog_usuario, copy_initialization=True)` sobre las fechas de esos 30 días y después `get_forecast(steps=N, exog=exog_futuro)`.

### Por qué esos parámetros (ACF/PACF y ADF/KPSS)

1. **m = 7.** La ACF tiene picos significativos y persistentes en los rezagos 7, 14, 21… hasta el 60, y la STL con periodo 7 explica buena parte de la varianza. Es el ciclo semanal: los domingos casi todas las tiendas cierran.
2. **D = 1.** Esos picos decaen muy despacio. Tras la diferencia de lag 7, ADF y KPSS coinciden en que la serie es estacionaria. Con la prueba OCSB de `pmdarima`, `auto_arima` tiende a elegir D = 0, y entonces el modelo no reproduce la caída de los domingos. Por eso D se fija en 1.
3. **d.** El informe recomienda d = 0 en las series estacionarias y d = 1 en las tiendas 770 y 1097, que son *estacionarias en diferencias*. `auto_arima` decide d con KPSS. Con D = 1 ya aplicada, en todas las series le basta d = 0.
4. **p, q ≤ 3 y P, Q ≤ 1.** La PACF corta tras los primeros rezagos (un AR de orden bajo) y conserva un pico en el lag 7, que se captura con un término SAR/SMA de orden 1. Con límites más amplios, el AIC apenas baja y el tiempo de búsqueda se multiplica.
5. **Exógenas.** El coeficiente de `promo` (tabla de abajo) cuantifica en euros diarios el efecto que el informe mide de forma descriptiva.

## Modelo 2 · GRU → `.keras` (`modelo_gru/`)

```
Input (30 pasos × 10 features)
 → GRU(64, return_sequences=True)    nombre "gru_1"
 → Dropout(0.2)                      identidad en inferencia
 → GRU(32)                           nombre "gru_2"
 → Dense(16, relu)                   nombre "dense_1"
 → Dense(1)                          venta escalada del día siguiente
```

- **Features por paso** (orden exacto): `ventas_esc, promo, school_holiday, festivo_a, festivo_b, festivo_c, dow_sin, dow_cos, mes_sin, mes_cos`.
  - `ventas_esc` usa un `MinMaxScaler` ajustado **solo con train**.
  - `StateHoliday` va en one-hot.
  - El día de la semana y el mes van en seno/coseno.
  - En cada paso, las exógenas son las del **día siguiente** (d + 1), que se conocen de antemano. Así los escenarios influyen directamente en la predicción.
- **Pronóstico recursivo a 42 días.** EarlyStopping(val_loss, paciencia 10) con validación sobre el 10 % final de train en orden temporal, Adam (1e-3), MSE y semillas fijas.
- **Guardado con `model.save("gru_<serie>.keras")`.** Es un ZIP con `config.json` y `model.weights.h5`.
  - Ojo: dentro del `.h5`, Keras 3 nombra las capas por clase y orden (`layers/gru`, `layers/gru_1`, `layers/dense`, `layers/dense_1`), no con el nombre del config.
  - El bias del GRU tiene forma (2, 3u): bias de entrada y bias recurrente, porque usa `reset_after=True`. Las gates van en orden z, r, h.
- **Modelo final:** tras evaluar, se reentrena con la serie completa (tantas épocas como la mejor época de la evaluación) y ese es el `.keras` que se exporta.
- `exportar_gru.py` escribe:
  - `gru_meta_<serie>.json`: el escalador, las features, la ventana, las capas, `ultima_fecha`, `ventana_gru` (los últimos 30 días), `info_tienda` y `mape_test`;
  - `gru_referencia_<serie>.json`: 5 ventanas reales del test con la salida de Keras. El navegador las ejecuta al cargar el modelo y exige una diferencia < 1e-4.

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
- **La tienda 770 tiene el error más alto con ambos modelos, y la causa está en los datos.** `store.csv` registra un competidor a **100 m abierto en abril de 2015** (`CompetitionDistance = 100`, `CompetitionOpenSince = 4/2015`). Desde entonces, la venta media cayó de ~7.000 € a ~4.800 € por día, y los dos modelos sobreestiman el test. Para corregirlo habría que añadir "competencia abierta" como exógena o reentrenar con datos posteriores al cambio.
- **Las tiendas b (1097 y 262) abren los domingos** y su mejor día es el domingo. Tienen la estacionalidad semanal más fuerte, y SARIMAX con D = 1 las modela muy bien (MAPE de ~4–5 %).
- **Las tiendas que cierran los domingos** (770, 1044, 352) tienen un ACF(7) bajo. La interpolación de los domingos suaviza ese día, y las promociones de Rossmann alternan semanas, lo que crea un ciclo de 14 días.

### Verificación de la ejecución en el navegador

Medida en el dashboard sobre la serie Total cadena:

| Modelo | Comparación | Diferencia máxima |
|---|---|---|
| ARIMA (`.joblib` final en Pyodide 314.0.8) | Pronóstico de referencia de 42 días (media + IC 95 %) vs. Python con el mismo archivo | 0,0 € |
| GRU (`.keras` final → TypeScript) | 5 ventanas de referencia vs. Keras | 9,95e-8 (tolerancia 1e-4) |

## Métricas (`evaluacion/metricas.py`)

- **RMSE**: la raíz del error cuadrático medio. Penaliza más los errores grandes.
- **MAE**: el error absoluto medio, en unidades de venta (€ por día).
- **MAPE**: el error porcentual medio, en %. Excluye los días con venta real 0.

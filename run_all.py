"""
Pipeline completo del backend. SE EJECUTA UNA SOLA VEZ: entrena, evalúa y exporta archivos.
Después, el frontend funciona solo (ejecuta los modelos en el navegador) y este backend
puede estar apagado o no existir.

  1. limpieza        src/limpieza/limpiar_datos.py
  2. estacionalidad  src/estacionalidad/analisis_estacionalidad.py   → estacionalidad_informe.json
  3. arima           src/modelo_arima/entrenar_arima.py + exportar_arima.py → arima_<serie>.joblib
  4. gru             src/modelo_gru/entrenar_gru.py + exportar_gru.py       → gru_<serie>.keras
  5. metricas        src/evaluacion/comparar_modelos.py                  → metricas.json
  6. copia           .joblib y .keras → rossmann-frontend/public/models/, .json → public/data/

En las etapas 3 y 4 cada modelo se EVALÚA con los últimos 42 días (métricas oficiales) y
después se REENTRENA con la serie completa con los mismos hiperparámetros. Lo que se copia
al frontend es esa versión final: arima_<serie>.joblib y gru_<serie>.keras pronostican desde
el día siguiente al último dato (01/08/2015), y sus metadatos (arima_meta_/gru_meta_<serie>.json)
incluyen ultima_fecha, ventana_gru, info_tienda y mape_test para la página /prediccion.

Cada etapa se lanza como un proceso aparte con el mismo intérprete: así se garantiza que
los scripts son realmente independientes (nadie comparte estado en memoria) y que cada uno
puede ejecutarse solo.

Uso:
    python run_all.py                    # todo
    python run_all.py --desde gru        # retoma desde una etapa
    python run_all.py --solo-copiar      # solo copia los artefactos al frontend
    python run_all.py --solo-final       # solo reentrena ARIMA y GRU con la serie completa
                                         # (reutiliza órdenes, épocas y métricas ya calculados),
                                         # exporta y copia
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # consola de Windows (cp1252) con tildes y símbolos

RAIZ = Path(__file__).resolve().parent
FRONT_PUBLIC = RAIZ.parent / "rossmann-frontend" / "public"

ETAPAS = [
    ("limpieza", ["src/limpieza/limpiar_datos.py"]),
    ("estacionalidad", ["src/estacionalidad/analisis_estacionalidad.py"]),
    ("arima", ["src/modelo_arima/entrenar_arima.py", "src/modelo_arima/exportar_arima.py"]),
    ("gru", ["src/modelo_gru/entrenar_gru.py", "src/modelo_gru/exportar_gru.py"]),
    ("metricas", ["src/evaluacion/comparar_modelos.py"]),
]

# --solo-final: reentrenamiento con la serie completa + exportación, sin repetir la evaluación.
ETAPAS_FINAL = [
    ("arima", ["src/modelo_arima/entrenar_arima.py", "--solo-final"]),
    ("arima", ["src/modelo_arima/exportar_arima.py"]),
    ("gru", ["src/modelo_gru/entrenar_gru.py", "--solo-final"]),
    ("gru", ["src/modelo_gru/exportar_gru.py"]),
]

# Artefactos que consume el frontend (patrón en el backend → subcarpeta de public/).
# arima_*.joblib y gru_*.keras son los modelos REENTRENADOS con la serie completa.
ARTEFACTOS = [
    ("models", "arima_*.joblib", "models"),
    ("models", "gru_*.keras", "models"),
    ("exports", "series.json", "data"),
    ("exports", "historicos.json", "data"),
    ("exports", "estacionalidad_informe.json", "data"),
    ("exports", "metricas.json", "data"),
    ("exports", "arima_meta_*.json", "data"),
    ("exports", "gru_meta_*.json", "data"),
    ("exports", "gru_referencia_*.json", "data"),
]

# Claves que la página /prediccion necesita en los metadatos del modelo final.
CLAVES_META = {
    "arima_meta_": ["ultima_fecha", "info_tienda", "mape_test", "exogenas", "referencia"],
    "gru_meta_": ["ultima_fecha", "ventana_gru", "info_tienda", "mape_test", "features"],
}


def verificar_metadatos() -> None:
    """Detiene la copia si algún metadato no corresponde al modelo reentrenado."""
    for prefijo, claves in CLAVES_META.items():
        for ruta in sorted((RAIZ / "exports").glob(f"{prefijo}*.json")):
            meta = json.loads(ruta.read_text(encoding="utf-8"))
            faltan = [c for c in claves if c not in meta]
            if faltan:
                sys.exit(f"✗ {ruta.name} no tiene {faltan}: vuelve a ejecutar la etapa del modelo (o --solo-final)")
            if prefijo == "gru_meta_" and len(meta["ventana_gru"]) != meta["ventana"]:
                sys.exit(f"✗ {ruta.name}: ventana_gru debe tener {meta['ventana']} días")


def ejecutar(script: list[str], nombre: str) -> None:
    t0 = time.time()
    r = subprocess.run([sys.executable, str(RAIZ / script[0]), *script[1:]], cwd=RAIZ)
    if r.returncode != 0:
        sys.exit(f"\n✗ Falló {' '.join(script)} (etapa '{nombre}'). Corrige el error y reanuda con --desde {nombre}")
    print(f"✓ {' '.join(script)} ({time.time() - t0:.0f}s)\n")


def copiar_a_frontend() -> None:
    print("=" * 70)
    print("ETAPA 6 · COPIA DE ARTEFACTOS AL FRONTEND")
    print("=" * 70)
    if not FRONT_PUBLIC.parent.exists():
        print(f"No existe {FRONT_PUBLIC.parent}; se omite la copia.")
        return
    verificar_metadatos()
    # Se eliminan artefactos de corridas anteriores para que no queden archivos huérfanos.
    for sub, patrones in (("data", ["*.json"]), ("models", ["*.json", "*.joblib", "*.keras", "*.onnx"])):
        destino = FRONT_PUBLIC / sub
        destino.mkdir(parents=True, exist_ok=True)
        for patron in patrones:
            for viejo in destino.glob(patron):
                viejo.unlink()
    n = 0
    for origen_dir, patron, sub in ARTEFACTOS:
        archivos = sorted((RAIZ / origen_dir).glob(patron))
        if not archivos:
            print(f"   (falta) {origen_dir}/{patron}")
        for origen in archivos:
            shutil.copy2(origen, FRONT_PUBLIC / sub / origen.name)
            print(f"   {origen.relative_to(RAIZ)} → public/{sub}/{origen.name}")
            n += 1
    print(f"{n} archivos copiados. Desde aquí el frontend ya no necesita el backend.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline Rossmann")
    parser.add_argument("--desde", choices=[e[0] for e in ETAPAS], default="limpieza")
    parser.add_argument("--solo-copiar", action="store_true")
    parser.add_argument("--solo-final", action="store_true",
                        help="solo reentrena con la serie completa, exporta y copia (no repite la evaluación)")
    args = parser.parse_args()

    if args.solo_final:
        for nombre, script in ETAPAS_FINAL:
            ejecutar(script, nombre)
    elif not args.solo_copiar:
        inicio = [e[0] for e in ETAPAS].index(args.desde)
        for nombre, scripts in ETAPAS[inicio:]:
            for script in scripts:
                ejecutar([script], nombre)
    copiar_a_frontend()
    print("\nPipeline terminado. Frontend: cd ../rossmann-frontend && npm run dev")


if __name__ == "__main__":
    main()

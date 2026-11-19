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

Cada etapa se lanza como un proceso aparte con el mismo intérprete: así se garantiza que
los scripts son realmente independientes (nadie comparte estado en memoria) y que cada uno
puede ejecutarse solo.

Uso:
    python run_all.py                    # todo
    python run_all.py --desde gru        # retoma desde una etapa
    python run_all.py --solo-copiar      # solo copia los artefactos al frontend
"""
from __future__ import annotations

import argparse
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

# Artefactos que consume el frontend (patrón en el backend → subcarpeta de public/)
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


def copiar_a_frontend() -> None:
    print("=" * 70)
    print("ETAPA 6 · COPIA DE ARTEFACTOS AL FRONTEND")
    print("=" * 70)
    if not FRONT_PUBLIC.parent.exists():
        print(f"No existe {FRONT_PUBLIC.parent}; se omite la copia.")
        return
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
    args = parser.parse_args()

    if not args.solo_copiar:
        inicio = [e[0] for e in ETAPAS].index(args.desde)
        for nombre, scripts in ETAPAS[inicio:]:
            for script in scripts:
                t0 = time.time()
                r = subprocess.run([sys.executable, str(RAIZ / script)], cwd=RAIZ)
                if r.returncode != 0:
                    sys.exit(f"\n✗ Falló {script} (etapa '{nombre}'). Corrige el error y reanuda con --desde {nombre}")
                print(f"✓ {script} ({time.time() - t0:.0f}s)\n")
    copiar_a_frontend()
    print("\nPipeline terminado. Frontend: cd ../rossmann-frontend && npm run dev")


if __name__ == "__main__":
    main()

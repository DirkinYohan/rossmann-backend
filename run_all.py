"""
Pipeline completo del backend, en orden:

  1. limpieza        src/limpieza/limpiar_datos.py
  2. estacionalidad  src/estacionalidad/analisis_estacionalidad.py
  3. arima           src/modelo_arima/entrenar_arima.py + exportar_arima.py
  4. gru             src/modelo_gru/entrenar_gru.py + exportar_gru.py
  5. comparacion     src/evaluacion/comparar_modelos.py
  6. copia           exports/ y models/*.onnx → rossmann-frontend/public/

Cada etapa se lanza como un proceso aparte con el mismo intérprete: así se garantiza que
los scripts son realmente independientes (nadie comparte estado en memoria) y que cada uno
puede ejecutarse solo.

Uso:
    python run_all.py                    # todo
    python run_all.py --desde arima      # retoma desde una etapa
    python run_all.py --solo-copiar      # solo copia los exports al frontend
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
import sys as _sys
_sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # consola de Windows (cp1252) con tildes y símbolos

RAIZ = Path(__file__).resolve().parent
FRONT_PUBLIC = RAIZ.parent / "rossmann-frontend" / "public"

ETAPAS = [
    ("limpieza", ["src/limpieza/limpiar_datos.py"]),
    ("estacionalidad", ["src/estacionalidad/analisis_estacionalidad.py"]),
    ("arima", ["src/modelo_arima/entrenar_arima.py", "src/modelo_arima/exportar_arima.py"]),
    ("gru", ["src/modelo_gru/entrenar_gru.py", "src/modelo_gru/exportar_gru.py"]),
    ("comparacion", ["src/evaluacion/comparar_modelos.py"]),
]

# Archivo de exports → subcarpeta de public/
DESTINOS = {
    "estacionalidad.json": "data",
    "arima_escenarios.json": "data",
    "arima_resultados.json": "data",
    "gru_resultados.json": "data",
    "comparacion_metricas.json": "data",
    "gru_config.json": "models",
}


def copiar_a_frontend() -> None:
    print("=" * 70)
    print("ETAPA 6 · COPIA DE EXPORTS AL FRONTEND")
    print("=" * 70)
    if not FRONT_PUBLIC.parent.exists():
        print(f"No existe {FRONT_PUBLIC.parent}; se omite la copia.")
        return
    for sub in ("data", "models"):
        (FRONT_PUBLIC / sub).mkdir(parents=True, exist_ok=True)
    archivos = [(RAIZ / "exports" / n, FRONT_PUBLIC / d / n) for n, d in DESTINOS.items()]
    archivos.append((RAIZ / "data" / "processed" / "series.json", FRONT_PUBLIC / "data" / "series.json"))
    archivos += [(p, FRONT_PUBLIC / "models" / p.name) for p in sorted((RAIZ / "models").glob("gru_*.onnx"))]
    for origen, destino in archivos:
        if origen.exists():
            shutil.copy2(origen, destino)
            print(f"   {origen.relative_to(RAIZ)} → {destino.relative_to(FRONT_PUBLIC.parent)}")
        else:
            print(f"   (falta) {origen.relative_to(RAIZ)}")


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

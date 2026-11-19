"""
Comparación de modelos: lee los resultados que exportó CADA modelo por separado
(exports/arima_resultados.json y exports/gru_resultados.json), arma una tabla por serie,
indica el mejor modelo por métrica (menor = mejor) y exporta exports/metricas.json con:
  - por serie: RMSE/MAE/MAPE de ARIMA y de GRU, el mejor modelo por métrica y las
    predicciones del test (ARIMA con su IC 95 % y GRU) junto a los valores reales;
  - un resumen global (victorias por métrica, MAPE promedio y mejor modelo general).
Las métricas de ambos modelos se calculan sobre el MISMO periodo de test (últimos 42 días).

Ejecución independiente:
    python src/evaluacion/comparar_modelos.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys as _sys
_sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # consola de Windows (cp1252) con tildes y símbolos

RAIZ = Path(__file__).resolve().parents[2]
DIR_EXPORTS = RAIZ / "exports"
METRICAS = ["RMSE", "MAE", "MAPE"]


def main() -> dict:
    print("=" * 70)
    print("ETAPA 5 · COMPARACIÓN DE MODELOS (mismo periodo de test: últimos 42 días)")
    print("=" * 70)
    arima = json.loads((DIR_EXPORTS / "arima_resultados.json").read_text(encoding="utf-8"))
    gru = json.loads((DIR_EXPORTS / "gru_resultados.json").read_text(encoding="utf-8"))

    filas, series, victorias = [], {}, {"ARIMA": 0, "GRU": 0}
    for id_serie, ra in arima["series"].items():
        rg = gru["series"].get(id_serie)
        if rg is None:
            continue
        if ra["test_inicio"] != rg["test_inicio"] or ra["test_fin"] != rg["test_fin"]:
            raise ValueError(f"Periodos de test distintos en {id_serie}: la comparación no sería justa")
        fila = {"serie": id_serie, "nombre": ra["nombre"], "test_inicio": ra["test_inicio"],
                "test_fin": ra["test_fin"], "ARIMA": ra["metricas"], "GRU": rg["metricas"], "mejor": {}}
        for m in METRICAS:
            ganador = "ARIMA" if ra["metricas"][m] <= rg["metricas"][m] else "GRU"
            fila["mejor"][m] = ganador
            victorias[ganador] += 1
        fila["mejor_global"] = max(("ARIMA", "GRU"), key=lambda k: list(fila["mejor"].values()).count(k))
        filas.append(fila)
        # Predicciones del test alineadas por fecha con los valores reales
        series[id_serie] = {**fila, "test": [
            {"fecha": a["fecha"], "real": a["real"], "abierto": a["abierto"],
             "arima": a["pred"], "arima_li": a["li"], "arima_ls": a["ls"], "gru": g["pred"]}
            for a, g in zip(ra["test"], rg["test"], strict=True)
        ]}

    # Promedio de MAPE entre series (métrica comparable entre escalas distintas)
    mape_prom = {mod: round(sum(f[mod]["MAPE"] for f in filas) / len(filas), 2) for mod in ("ARIMA", "GRU")}
    salida = {
        "metricas": METRICAS, "criterio": "Menor es mejor en las tres métricas; MAPE calculado solo en días abiertos.",
        "descripcion_metricas": {
            "RMSE": "Raíz del error cuadrático medio: penaliza más los errores grandes.",
            "MAE": "Error absoluto medio: error promedio en unidades de venta (€ por día).",
            "MAPE": "Error porcentual absoluto medio (%), excluyendo días con venta real 0.",
        },
        "tabla": filas, "series": series, "victorias_por_metrica": victorias, "mape_promedio": mape_prom,
        "mejor_modelo_general": min(mape_prom, key=mape_prom.get),
    }

    print(f"{'Serie':<38}{'Métrica':<8}{'ARIMA':>14}{'GRU':>14}  Mejor")
    for f in filas:
        for m in METRICAS:
            fmt = "{:>13.2f}%" if m == "MAPE" else "{:>14,.0f}"
            print(f"{f['nombre'] if m == 'RMSE' else '':<38}{m:<8}{fmt.format(f['ARIMA'][m])}{fmt.format(f['GRU'][m])}  {f['mejor'][m]}")
    print(f"\nVictorias por métrica: {victorias} · MAPE promedio: {mape_prom} → mejor general: {salida['mejor_modelo_general']}")

    ruta = DIR_EXPORTS / "metricas.json"
    ruta.write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Exportado: {ruta}")
    actualizar_readme(salida, arima, gru)
    return salida


def _miles(v: float) -> str:
    return f"{v:,.0f}".replace(",", ".")


def actualizar_readme(salida: dict, arima: dict, gru: dict) -> None:
    """
    Reescribe en README.md el bloque entre <!-- RESULTADOS:INICIO --> y <!-- RESULTADOS:FIN -->
    con las métricas, los órdenes SARIMAX y la evidencia de estacionariedad de la ÚLTIMA corrida.
    Así la documentación nunca queda desactualizada respecto a los modelos exportados.
    """
    ruta = RAIZ / "README.md"
    ini, fin = "<!-- RESULTADOS:INICIO -->", "<!-- RESULTADOS:FIN -->"
    if not ruta.exists():
        return
    texto = ruta.read_text(encoding="utf-8")
    if ini not in texto or fin not in texto:
        return
    ruta_est = DIR_EXPORTS / "estacionalidad_informe.json"
    est = json.loads(ruta_est.read_text(encoding="utf-8"))["series"] if ruta_est.exists() else {}

    f0 = salida["tabla"][0]
    lineas = [
        ini,
        f"_Generado automáticamente por `comparar_modelos.py`. Test: {f0['test_inicio']} → {f0['test_fin']} (42 días)._",
        "",
        "### Métricas en el periodo de test",
        "",
        "| Serie | RMSE ARIMA | RMSE GRU | MAE ARIMA | MAE GRU | MAPE ARIMA | MAPE GRU | Mejor |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for f in salida["tabla"]:
        def celda(mod: str, m: str) -> str:
            v = f[mod][m]
            s = f"{v:.2f} %".replace(".", ",") if m == "MAPE" else _miles(v)
            return f"**{s}**" if f["mejor"][m] == mod else s
        lineas.append(
            f"| {f['nombre']} | {celda('ARIMA', 'RMSE')} | {celda('GRU', 'RMSE')} | {celda('ARIMA', 'MAE')} | "
            f"{celda('GRU', 'MAE')} | {celda('ARIMA', 'MAPE')} | {celda('GRU', 'MAPE')} | {f['mejor_global']} |"
        )
    mp = salida["mape_promedio"]
    lineas += [
        "",
        f"MAPE promedio: ARIMA {mp['ARIMA']:.2f} % · GRU {mp['GRU']:.2f} % · ".replace(".", ",")
        + f"Victorias por métrica: ARIMA {salida['victorias_por_metrica']['ARIMA']}, "
        f"GRU {salida['victorias_por_metrica']['GRU']} · **Mejor modelo general: {salida['mejor_modelo_general']}**",
        "",
        "### Órdenes SARIMAX elegidos y evidencia",
        "",
        "| Serie | Orden (p,d,q)(P,D,Q,7) | AIC | ADF p (orig.) | KPSS p (orig.) | ACF(7) | Coef. Promo | GRU épocas |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for id_serie, a in arima["series"].items():
        e = est.get(id_serie, {})
        pr = e
        acf7 = e["acf"]["valores"][7] if e else "–"
        g = gru["series"].get(id_serie, {})
        lineas.append(
            f"| {a['nombre']} | ({','.join(map(str, a['orden']))})({','.join(map(str, a['orden_estacional']))}) | "
            f"{_miles(a['aic'])} | {pr.get('adf', {}).get('p_valor', '–')} | {pr.get('kpss', {}).get('p_valor', '–')} | "
            f"{acf7} | {_miles(a['coeficientes_exogenas'].get('promo', 0))} | {g.get('mejor_epoca', '–')} |"
        )
    lineas.append(fin)
    nuevo = texto[: texto.index(ini)] + "\n".join(lineas) + texto[texto.index(fin) + len(fin):]
    ruta.write_text(nuevo, encoding="utf-8")
    print(f"README actualizado con los resultados: {ruta}")


if __name__ == "__main__":
    main()

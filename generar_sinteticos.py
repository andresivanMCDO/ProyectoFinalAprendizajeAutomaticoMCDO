"""Genera datos SINTÉTICOS con la estructura de tonelajes_sector_dia.csv (no son datos reales).

Estructura: 72 sectores en dos zonas con calendario alterno (Norte lun/mié/vie, Sur mar/jue/sáb), acumulación
por días sin servicio (feriados incluidos) y una base distinta por sector.

Dinámica temporal (pensada para que la demo se comporte como el problema real):
  1. Tendencia lineal + una curva CONVEXA en ene–jun que luego se aplana. Un polinomial la ajusta bien
     dentro del entrenamiento y la extrapola hacia arriba en jul–sep.
  2. Nivel de jul–sep ≈ 8 % sobre el promedio del entrenamiento (deriva): los datos del pasado se quedan cortos.
  3. Efecto del día del mes: más volumen a inicios y fines de mes.
  4. Choques comunes por semana y por mes (iguales para todos los sectores): clima, eventos, operación.
  5. Ruido independiente por fila.

Semilla fija: el resultado es reproducible. Los parámetros de PARAMS se calibraron (búsqueda numérica) para que, con el corte por
defecto de la app, los tres modelos se ordenen y se parezcan en magnitud al estudio real
(lineal < línea base < polinomial; el polinomial sobreestima y su sesgo crece mes a mes).
"""
import numpy as np
import pandas as pd

PARAMS = dict(
    base_media=10.7755,  # t por día acumulado ×2: media entre sectores
    base_sd=1.0,         # dispersión entre sectores (~10 %)
    tend_lineal=-0.3402, # componente lineal ene→sep (fracción); combinada con la curva da una "U" suave
    curv=0.3198,         # amplitud de la curva convexa de ene–jun
    curv_pot=2.4253,     # exponente de la curva: >2 = más plana al inicio y se acelera hacia junio
    t_corte=181,         # día en que la curva se aplana (≈ 30-jun)
    salto=0.0196,        # cambio de nivel adicional desde julio (fracción)
    dm_amp=0.08,         # amplitud del efecto del día del mes
    sd_semana=0.0054,    # choque común semanal (log)
    sd_mes=0.0259,       # choque común mensual (log)
    sigma_fila=0.1399,   # ruido independiente por fila (log)
)
SEMILLA = 3
EXTRAS = [109, 110, 113, 114, 117, 118, 119, 120, 121, 122, 125, 126, 131, 132, 133, 134, 135, 136]
NORTE = set(range(1, 28)) | {109, 113, 117, 119, 121, 125, 131, 133, 135}
FERIADOS = pd.to_datetime(["2025-01-21", "2025-05-01", "2025-09-16"])
EFECTO_DIA = {0: 1.02, 1: 1.02, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}      # lun..sáb: casi sin efecto propio (el real: 1.54× vs 1.50× esperado)
INICIO = pd.Timestamp("2025-01-01")


def generar(p=PARAMS, semilla=SEMILLA) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    sectores = list(range(1, 55)) + EXTRAS
    base = {s: max(3.0, rng.normal(p["base_media"], p["base_sd"])) for s in sectores}
    fechas = pd.date_range("2025-01-02", "2025-09-30")
    fechas = fechas[~fechas.isin(FERIADOS)]
    filas = []
    for s in sectores:
        dias = {0, 2, 4} if s in NORTE else {1, 3, 5}
        f = fechas[fechas.day_of_week.isin(dias)]
        filas.append(pd.DataFrame({"fecha": f, "id_sector": s, "zona": "Norte" if s in NORTE else "Sur"}))
    d = pd.concat(filas, ignore_index=True).sort_values(["id_sector", "fecha"]).reset_index(drop=True)

    d["dia_semana"] = d["fecha"].dt.day_of_week + 1
    d["t"] = (d["fecha"] - INICIO).dt.days
    d["dia_mes"] = d["fecha"].dt.day
    d["mes"] = d["fecha"].dt.month
    # días acumulados: días desde el servicio anterior del mismo sector (el primero: 3 si cae lun/mar, 2 si no)
    d["dias_acum"] = d.groupby("id_sector")["fecha"].diff().dt.days
    d["dias_acum"] = d["dias_acum"].fillna(pd.Series(np.where(d["dia_semana"].isin([1, 2]), 3, 2), index=d.index)).astype(int)
    d["visita_ciclo"] = (d["dia_semana"] + 1) // 2

    tc = p["t_corte"]
    f_tiempo = (1 + p["tend_lineal"] * d["t"] / 273
                + p["curv"] * (np.minimum(d["t"], tc) / tc) ** p["curv_pot"])
    f_salto = np.where(d["t"] >= tc, 1 + p["salto"], 1.0)
    f_dm = 1 + p["dm_amp"] * (np.abs(d["dia_mes"] - 15.5) / 15.5 - 0.5)
    sem = d["fecha"].dt.isocalendar().week.astype(int).to_numpy()
    choque_sem = np.exp(rng.normal(0, p["sd_semana"], sem.max() + 1))[sem]
    choque_mes = np.exp(rng.normal(0, p["sd_mes"], 13))[d["mes"].to_numpy()]
    ruido = rng.lognormal(0, p["sigma_fila"], len(d))
    y = (d["id_sector"].map(base) * (d["dias_acum"] / 2) * d["dia_semana"].sub(1).map(EFECTO_DIA)
         * f_tiempo * f_salto * f_dm * choque_sem * choque_mes * ruido)
    d["peso_ton_total"] = np.maximum(1.5, y)

    d = d.sort_values(["fecha", "id_sector"]).reset_index(drop=True)
    d.insert(0, "id_bitacora", np.arange(1, len(d) + 1))
    d["n_viajes"] = np.ceil(d["peso_ton_total"] / 8).astype(int)
    return d[["id_bitacora", "fecha", "id_sector", "zona", "peso_ton_total", "n_viajes", "dia_semana",
              "dias_acum", "t", "mes", "dia_mes", "visita_ciclo"]]


if __name__ == "__main__":
    d = generar()
    d.to_csv("tonelajes_sector_dia_sintetico.csv", index=False)
    print(d.shape)

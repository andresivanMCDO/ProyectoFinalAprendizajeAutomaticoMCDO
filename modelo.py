"""Lógica del modelo (sin interfaz): validación del CSV, partición temporal, modelos y métricas.

Es la misma especificación del cuaderno `modelado_residuos_saltillo.ipynb`:
  - objetivo: peso_ton_total (toneladas por sector-día)
  - predictores conocidos de antemano: id_sector, visita_ciclo (categorías),
    acum = min(dias_acum, 4), t (días desde el 1-ene) y dia_mes
  - lineal (grado 1) vs polinomial (grado d sobre t y dia_mes) vs línea base
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

OBJ = "peso_ton_total"
CAT, NUM, TIEMPO = ["id_sector", "visita_ciclo"], ["acum"], ["t", "dia_mes"]
X = CAT + NUM + TIEMPO

# Estructura de tonelajes_sector_dia.csv (la plantilla completa)
COLUMNAS_PLANTILLA = ["id_bitacora", "fecha", "id_sector", "zona", "peso_ton_total", "n_viajes",
                      "dia_semana", "dias_acum", "t", "mes", "dia_mes", "visita_ciclo"]
# Las únicas que el modelo necesita; t, dia_mes y mes se recalculan desde `fecha`
COLUMNAS_MIN = ["fecha", "id_sector", "peso_ton_total", "dias_acum", "visita_ciclo"]

MAX_FILAS = 50_000
MIN_FILAS = 500
MIN_MESES = 6          # 4 de entrenamiento (2 para validar) + al menos 1 de prueba y margen
GRADOS = [1, 2, 3, 4, 5]


# --------------------------------------------------------------------------- datos
def preparar(df: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """Valida y prepara un DataFrame. Devuelve (datos, errores, avisos).

    Si hay errores, `datos` es None. Los avisos no detienen el análisis."""
    errores, avisos = [], []
    faltan = [c for c in COLUMNAS_MIN if c not in df.columns]
    if faltan:
        return None, [f"Faltan columnas obligatorias: {', '.join(faltan)}. "
                      f"Se esperan al menos: {', '.join(COLUMNAS_MIN)} "
                      f"(estructura completa: {', '.join(COLUMNAS_PLANTILLA)})."], avisos
    if len(df) > MAX_FILAS:
        return None, [f"El archivo tiene {len(df):,} filas; el máximo de esta demo es {MAX_FILAS:,}."], avisos

    d = df[COLUMNAS_MIN].copy()
    d["fecha"] = pd.to_datetime(d["fecha"], errors="coerce")
    for c in ["id_sector", "peso_ton_total", "dias_acum", "visita_ciclo"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    nulos = d.isna().sum()
    nulos = nulos[nulos > 0]
    if len(nulos):
        errores.append("Hay valores vacíos o no válidos en: "
                       + ", ".join(f"{c} ({n} filas)" for c, n in nulos.items()) + ".")
        return None, errores, avisos

    d["id_sector"] = d["id_sector"].astype(int)
    d["dias_acum"] = d["dias_acum"].astype(int)
    d["visita_ciclo"] = d["visita_ciclo"].astype(int)

    if (d["dias_acum"] < 1).any():
        errores.append("`dias_acum` debe ser ≥ 1 (días transcurridos desde el servicio anterior).")
    no_pos = int((d[OBJ] <= 0).sum())
    if no_pos:
        avisos.append(f"Se excluyeron {no_pos} fila(s) con `peso_ton_total` ≤ 0 (registros sin recolección).")
        d = d[d[OBJ] > 0]
    dup = int(d.duplicated(["fecha", "id_sector"]).sum())
    if dup:
        avisos.append(f"Hay {dup} combinaciones fecha-sector repetidas; el modelo las usa tal cual "
                      "(lo esperado es una fila por sector-día).")
    if errores:
        return None, errores, avisos
    if len(d) < MIN_FILAS:
        return None, [f"Se necesitan al menos {MIN_FILAS} filas válidas; hay {len(d)}."], avisos

    d = d.sort_values(["fecha", "id_sector"]).reset_index(drop=True)
    d["acum"] = d["dias_acum"].clip(upper=4)                       # 4 = "4 o más"
    d["t"] = (d["fecha"] - pd.to_datetime(d["fecha"].dt.year.astype(str) + "-01-01")).dt.days
    d["dia_mes"] = d["fecha"].dt.day
    per = d["fecha"].dt.to_period("M")
    d["mes_idx"] = (per - per.min()).apply(lambda x: x.n)          # 0, 1, 2, ... meses consecutivos
    d["mes_txt"] = per.astype(str)
    if d["mes_idx"].nunique() < MIN_MESES:
        return None, [f"Se necesitan al menos {MIN_MESES} meses distintos de datos; hay "
                      f"{d['mes_idx'].nunique()}."], avisos
    if d["fecha"].dt.year.nunique() > 1:
        avisos.append("Los datos cruzan más de un año: `t` se mide desde el 1-ene de cada año, "
                      "así que el tiempo reinicia en enero. Para varios años conviene un `t` continuo.")
    return d, errores, avisos


def meses_de_prueba_posibles(d: pd.DataFrame) -> list[int]:
    """Índices de mes que pueden ser el primer mes de prueba (≥4 meses de entrenamiento)."""
    idx = sorted(d["mes_idx"].unique())
    return [m for m in idx if m >= 4]


def dividir(d: pd.DataFrame, primer_mes_prueba: int):
    train = d[d["mes_idx"] < primer_mes_prueba].reset_index(drop=True)
    test = d[d["mes_idx"] >= primer_mes_prueba].reset_index(drop=True)
    return train, test


# --------------------------------------------------------------------------- modelos
def armar(grado: int = 1) -> Pipeline:
    """grado=1 → regresión lineal; grado≥2 → polinomial (solo en t y dia_mes)."""
    tiempo = Pipeline([("esc", StandardScaler()),
                       ("poly", PolynomialFeatures(grado, include_bias=False))])
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT),
        ("acum", "passthrough", NUM),
        ("tiempo", tiempo, TIEMPO),
    ])
    return Pipeline([("pre", pre), ("reg", LinearRegression())])


def linea_base(tr: pd.DataFrame, nuevo: pd.DataFrame) -> np.ndarray:
    """Promedio histórico por sector × días acumulados (con respaldo: promedio del sector, luego global)."""
    tab = tr.groupby(["id_sector", "acum"])[OBJ].mean().rename("p")
    sec = tr.groupby("id_sector")[OBJ].mean().rename("p_sec")
    m = nuevo.join(tab, on=["id_sector", "acum"]).join(sec, on="id_sector")
    return m["p"].fillna(m["p_sec"]).fillna(tr[OBJ].mean()).to_numpy()


def metricas(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"RMSE (t)": mean_squared_error(y, p) ** 0.5, "MAE (t)": mean_absolute_error(y, p),
            "R²": r2_score(y, p), "Sesgo (t)": float(np.mean(p - y))}


def validar_grados(train: pd.DataFrame, grados=GRADOS, n_meses_val: int = 4):
    """Validación de ventana expansiva por mes: valida el mes m con todos los meses < m.

    Valida en los últimos `n_meses_val` meses del entrenamiento que tengan ≥2 meses previos.
    Devuelve (detalle por grado×mes, resumen por grado, grado elegido por la regla de 1 error estándar)."""
    meses = sorted(train["mes_idx"].unique())
    val = [m for m in meses if m >= meses[0] + 2][-n_meses_val:]
    filas = []
    for g in grados:
        for m in val:
            tr, va = train[train["mes_idx"] < m], train[train["mes_idx"] == m]
            p = armar(g).fit(tr[X], tr[OBJ]).predict(va[X])
            filas.append({"grado": g, "mes_idx": m, "RMSE": mean_squared_error(va[OBJ], p) ** 0.5})
    cv = pd.DataFrame(filas)
    res = (cv.groupby("grado")["RMSE"].agg(media="mean", ee=lambda s: s.std(ddof=1) / np.sqrt(len(s)))
             .reset_index())
    sub = res[res["grado"] >= 2]
    mejor = sub.loc[sub["media"].idxmin()]
    lim = mejor["media"] + mejor["ee"]
    grado_poli = int(sub[sub["media"] <= lim]["grado"].min())
    return cv, res, grado_poli


def ajustar_y_evaluar(train: pd.DataFrame, test: pd.DataFrame, grado_poli: int):
    """Ajusta con todo el entrenamiento y predice la prueba. Devuelve (predicciones, tabla de métricas)."""
    pred = {
        "Línea base": linea_base(train, test),
        "Lineal": armar(1).fit(train[X], train[OBJ]).predict(test[X]),
        f"Polinomial (grado {grado_poli})": armar(grado_poli).fit(train[X], train[OBJ]).predict(test[X]),
    }
    tabla = pd.DataFrame({k: metricas(test[OBJ], v) for k, v in pred.items()}).T
    return pred, tabla


def bootstrap_diferencia_rmse(test: pd.DataFrame, p_a, p_b, reps: int = 1000, seed: int = 0):
    """Bootstrap pareado por fecha de RMSE(a) − RMSE(b). Devuelve (diferencia, IC95 inferior, IC95 superior)."""
    y = test[OBJ].to_numpy()
    g = test.assign(ea=(np.asarray(p_a) - y) ** 2, eb=(np.asarray(p_b) - y) ** 2) \
            .groupby("fecha").agg(ea=("ea", "sum"), eb=("eb", "sum"), n=("ea", "size"))
    ea, eb, n = g["ea"].to_numpy(), g["eb"].to_numpy(), g["n"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(reps, len(g)))
    diff = np.sqrt(ea[idx].sum(1) / n[idx].sum(1)) - np.sqrt(eb[idx].sum(1) / n[idx].sum(1))
    punto = np.sqrt(ea.sum() / n.sum()) - np.sqrt(eb.sum() / n.sum())
    return float(punto), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))


def sesgo_por_mes(test: pd.DataFrame, pred: dict) -> pd.DataFrame:
    """Sesgo medio (predicho − observado) por mes de prueba y modelo."""
    out = {k: (pd.Series(v, index=test.index) - test[OBJ]).groupby(test["mes_txt"]).mean() for k, v in pred.items()}
    return pd.DataFrame(out)


def serie_semanal(d: pd.DataFrame, train: pd.DataFrame, grado_poli: int) -> pd.DataFrame:
    """Promedio semanal por sector-día de TODO el periodo: real vs predicho por modelos entrenados solo con `train`.

    Igual que en el cuaderno: se descartan la primera y la última semana (parciales)."""
    lin = armar(1).fit(train[X], train[OBJ])
    poli = armar(grado_poli).fit(train[X], train[OBJ])
    todo = d.copy()
    todo["semana"] = todo["fecha"].dt.to_period("W-SUN").dt.start_time
    todo["Lineal"], todo["Polinomial"] = lin.predict(todo[X]), poli.predict(todo[X])
    sem = todo.groupby("semana").agg(real=(OBJ, "mean"), Lineal=("Lineal", "mean"), Polinomial=("Polinomial", "mean"))
    return sem.iloc[1:-1]

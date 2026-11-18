"""Demo pública: predicción de toneladas de RSU por sector-día (regresión lineal vs polinomial).

Ejecutar en local:  streamlit run app.py
Por defecto usa datos SINTÉTICOS; el usuario puede subir su propio CSV con la estructura de
`tonelajes_sector_dia.csv` (ver la pestaña «Acerca de»). Las gráficas replican las del cuaderno
`modelado_residuos_saltillo.ipynb`.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Patch
from scipy import stats

import modelo as M

RUTA_SINTETICO = Path(__file__).parent / "tonelajes_sector_dia_sintetico.csv"
# Misma paleta y estilo que el cuaderno de modelado
AZUL, VIOLETA, GRIS = "#2a78d6", "#6250d6", "#9a9993"      # lineal / polinomial / línea base
TINTA, TINTA2, REJILLA = "#0b0b0b", "#52514e", "#e6e5e1"
COLOR = {"Línea base": GRIS, "Lineal": AZUL, "Polinomial": VIOLETA}
ABREV = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun", 7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct",
         11: "Nov", 12: "Dic"}

plt.rcParams.update({
    "figure.dpi": 110, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9c8c3", "axes.labelcolor": TINTA2, "axes.titlelocation": "left",
    "axes.titlesize": 12, "axes.titleweight": "bold", "xtick.color": TINTA2, "ytick.color": TINTA2,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": REJILLA, "axes.axisbelow": True,
    "legend.frameon": False,
})

st.set_page_config(page_title="Toneladas de RSU por sector-día", page_icon="♻️", layout="wide")


def corto(nombre: str) -> str:
    """'Polinomial (grado 2)' → 'Polinomial' (clave de color y de columnas)."""
    return "Polinomial" if nombre.startswith("Polinomial") else nombre


def mes_abrev(mes_txt: str, con_anio: bool) -> str:
    anio, mes = mes_txt.split("-")
    return f"{ABREV[int(mes)]} {anio[2:]}" if con_anio else ABREV[int(mes)]


def mostrar(fig):
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- carga
@st.cache_data(show_spinner=False)
def leer_sintetico() -> pd.DataFrame:
    return pd.read_csv(RUTA_SINTETICO)


@st.cache_data(show_spinner=False)
def correr(d: pd.DataFrame, primer_mes_prueba: int, grado_manual: int | None):
    """Todo el análisis; se cachea por (datos, mes de corte, grado)."""
    train, test = M.dividir(d, primer_mes_prueba)
    cv, res, grado_auto = M.validar_grados(train)
    grado = grado_manual or grado_auto
    pred, tabla = M.ajustar_y_evaluar(train, test, grado)
    nombres = list(pred)
    lineal, poli = pred[nombres[1]], pred[nombres[2]]
    reg = M.armar(1).fit(train[M.X], train[M.OBJ])
    coef = pd.Series(reg.named_steps["reg"].coef_, index=reg.named_steps["pre"].get_feature_names_out())
    corte = test["fecha"].min().to_period("M").start_time
    return dict(train=train, test=test, cv=cv, res=res, grado_auto=grado_auto, grado=grado, pred=pred,
                tabla=tabla, boot_pl=M.bootstrap_diferencia_rmse(test, poli, lineal),
                sesgo=M.sesgo_por_mes(test, pred), por_acum=float(coef["acum__acum"]), corte=corte,
                semanal=M.serie_semanal(d, train, grado))


# --------------------------------------------------------------------------- barra lateral
st.sidebar.header("Datos")
fuente = st.sidebar.radio("Fuente", ["Datos sintéticos", "Subir mi CSV"], label_visibility="collapsed")
crudo, nombre_fuente = None, ""
if fuente == "Datos sintéticos":
    crudo, nombre_fuente = leer_sintetico(), "sintéticos"
else:
    archivo = st.sidebar.file_uploader("CSV con la estructura de tonelajes_sector_dia.csv", type=["csv"])
    st.sidebar.caption("Demo pública: **no subas datos sensibles o confidenciales**. Máximo 5 MB.")
    if archivo is None:
        st.title("Toneladas de RSU por sector-día")
        st.info("Sube un CSV en la barra lateral o vuelve a «Datos sintéticos».\n\n"
                f"**Columnas obligatorias:** `{', '.join(M.COLUMNAS_MIN)}`. La estructura completa de "
                f"`tonelajes_sector_dia.csv` es `{', '.join(M.COLUMNAS_PLANTILLA)}`; `t`, `dia_mes` y mes se "
                f"recalculan desde `fecha`. Necesitas ≥ {M.MIN_MESES} meses y ≥ {M.MIN_FILAS} filas. "
                "Puedes bajar la plantilla (datos sintéticos) en esta misma barra.")
        st.sidebar.download_button("Descargar plantilla (datos sintéticos)", RUTA_SINTETICO.read_bytes(),
                                   "tonelajes_sector_dia_sintetico.csv", "text/csv")
        st.stop()
    try:
        crudo, nombre_fuente = pd.read_csv(archivo), f"de «{archivo.name}»"
    except Exception as e:  # noqa: BLE001 - mostrar cualquier problema de lectura al usuario
        st.error(f"No pude leer el archivo como CSV: {e}")
        st.stop()

datos, errores, avisos = M.preparar(crudo)
if errores:
    st.title("Toneladas de RSU por sector-día")
    for e in errores:
        st.error(e)
    st.stop()

opciones = M.meses_de_prueba_posibles(datos)
etiqueta = datos.drop_duplicates("mes_idx").set_index("mes_idx")["mes_txt"]
n_meses = datos["mes_idx"].nunique()
CON_ANIO = datos["fecha"].dt.year.nunique() > 1
defecto = max(opciones[0], n_meses - 3)       # por defecto, los últimos 3 meses son la prueba
primer = st.sidebar.selectbox("Primer mes de prueba (el resto es entrenamiento)", opciones,
                              index=opciones.index(defecto) if defecto in opciones else 0,
                              format_func=lambda m: etiqueta[m])
manual = st.sidebar.selectbox("Grado del polinomio", ["Automático (validación)", 2, 3, 4, 5])
grado_manual = manual if isinstance(manual, int) else None
st.sidebar.download_button("Descargar plantilla (datos sintéticos)", RUTA_SINTETICO.read_bytes(),
                           "tonelajes_sector_dia_sintetico.csv", "text/csv")

with st.spinner("Ajustando modelos…"):
    R = correr(datos, primer, grado_manual)
train, test, tabla, pred = R["train"], R["test"], R["tabla"], R["pred"]
NB, NL, NP = list(pred)
yt = test[M.OBJ].to_numpy()

# --------------------------------------------------------------------------- encabezado
st.title("Toneladas de RSU por sector-día")
if fuente == "Datos sintéticos":
    st.warning("Estás viendo **datos sintéticos** generados para esta demo; no son datos reales del servicio.")
else:
    st.caption(f"Analizando datos {nombre_fuente}.")
for a in avisos:
    st.warning(a)
st.caption(f"Entrenamiento: {etiqueta[train.mes_idx.min()]} a {etiqueta[train.mes_idx.max()]} "
           f"({len(train):,} filas) · Prueba: {etiqueta[test.mes_idx.min()]} a {etiqueta[test.mes_idx.max()]} "
           f"({len(test):,} filas) · {datos.id_sector.nunique()} sectores")

tab1, tab2, tab3, tab4 = st.tabs(["Resultados", "Validación del grado", "Los datos", "Acerca de"])

# --------------------------------------------------------------------------- resultados
with tab1:
    mejor = tabla["RMSE (t)"].idxmin()
    d_pl, lo, hi = R["boot_pl"]
    if lo > 0:
        txt_boot = (f"El polinomial es **peor** que el lineal por {d_pl:+.2f} t de RMSE "
                    f"(IC 95 %: {lo:+.2f} a {hi:+.2f}); la diferencia es mayor que el azar.")
    elif hi < 0:
        txt_boot = (f"El polinomial es **mejor** que el lineal por {-d_pl:.2f} t de RMSE "
                    f"(IC 95 %: {lo:+.2f} a {hi:+.2f}); la diferencia es mayor que el azar.")
    else:
        txt_boot = (f"La diferencia de RMSE entre polinomial y lineal ({d_pl:+.2f} t, IC 95 %: {lo:+.2f} a {hi:+.2f}) "
                    "**no se distingue del azar**.")
    st.markdown(f"En los meses de prueba, el menor error lo tiene **{mejor}** "
                f"(RMSE {tabla.loc[mejor, 'RMSE (t)']:.2f} t por sector-día). {txt_boot}")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(tabla.style.format({"RMSE (t)": "{:.2f}", "MAE (t)": "{:.2f}", "R²": "{:.3f}",
                                         "Sesgo (t)": "{:+.2f}"}), width="stretch")
        st.caption("RMSE y MAE: error típico en toneladas (menor es mejor). Sesgo = predicho − observado: "
                   "negativo significa que el modelo se queda corto.")
    with c2:
        st.metric("Efecto de un día acumulado (modelo lineal)", f"{R['por_acum']:+.2f} t",
                  help="Toneladas adicionales por cada día sin servicio acumulado, con el resto constante.")

    # --- MAE y RMSE en prueba (como el cuaderno)
    etiquetas = [n.replace(" (", "\n(") for n in tabla.index]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for a, met in zip(ax, ["MAE (t)", "RMSE (t)"]):
        b = a.bar(etiquetas, tabla[met], color=[COLOR[corto(n)] for n in tabla.index], width=.55)
        a.bar_label(b, fmt="%.2f", padding=3, color=TINTA, fontweight="bold")
        a.set(title=f"{met} en prueba (menor es mejor)", ylabel="toneladas")
        a.set_ylim(0, tabla[met].max() * 1.18)
    fig.tight_layout(); mostrar(fig)

    # --- serie semanal: TODO el periodo; modelos entrenados solo con el entrenamiento
    sem, corte = R["semanal"], R["corte"]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(sem.index, sem.real, color=TINTA, lw=2, label="real")
    for n in ["Lineal", "Polinomial"]:
        ax.plot(sem.index, sem[n], color=COLOR[n], lw=2, label=n)
    ax.axvspan(corte, sem.index.max(), color=REJILLA, alpha=.6, zorder=0)
    ax.text(corte, ax.get_ylim()[1], "  prueba", va="top", color=TINTA2)
    ax.set(title="Promedio semanal por sector-día: real vs predicho", ylabel="toneladas")
    ax.legend(loc="lower right", ncol=3); fig.tight_layout(); mostrar(fig)
    st.caption(f"Cubre todo el periodo ({etiqueta.iloc[0]} a {etiqueta.iloc[-1]}). Los modelos se entrenan solo "
               "con la parte sin sombrear; la zona sombreada es la prueba.")

    # --- real vs predicho (prueba)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    lim = [0, max(yt.max(), pred[NL].max(), pred[NP].max()) * 1.03]
    for a, n in zip(ax, [NL, NP]):
        a.scatter(yt, pred[n], s=6, alpha=.35, color=COLOR[corto(n)], linewidths=0)
        a.plot(lim, lim, color=TINTA2, lw=1, ls="--")
        a.set(title=f"{corto(n)}: real vs predicho (prueba)", xlabel="real (t)",
              ylabel="predicho (t)" if n == NL else "")
        a.grid(axis="x", color=REJILLA)
    fig.tight_layout(); mostrar(fig)

    # --- sesgo por mes de prueba (deriva)
    sb = R["sesgo"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ancho = 0.75 / len(sb.columns)
    for i, n in enumerate(sb.columns):
        pos = np.arange(len(sb)) + (i - (len(sb.columns) - 1) / 2) * ancho
        ax.bar(pos, sb[n].values, width=ancho, color=COLOR[corto(n)], label=corto(n))
    ax.axhline(0, color=TINTA2, lw=1)
    ax.set_xticks(range(len(sb)), [mes_abrev(m, CON_ANIO) for m in sb.index])
    ax.set(title="Error medio por mes (predicho − real)", ylabel="toneladas (negativo = subestima)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    fig.tight_layout(); mostrar(fig)
    st.caption("Si el sesgo de un modelo crece mes a mes, está extrapolando mal: el polinomial es el más propenso.")

    # --- diagnóstico de residuos del lineal (prueba)
    with st.expander("Diagnóstico de residuos del modelo lineal (prueba)"):
        aj = pred[NL]
        res = yt - aj
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        ax[0].scatter(aj, res, s=6, alpha=.3, color=AZUL, linewidths=0); ax[0].axhline(0, color=TINTA2, lw=1)
        ax[0].set(title="Residuos vs ajustado", xlabel="predicho (t)", ylabel="real − predicho (t)")
        stats.probplot(res, dist="norm", plot=ax[1])
        ax[1].get_lines()[0].set(color=AZUL, markersize=3, alpha=.5)
        ax[1].get_lines()[1].set(color=TINTA2)
        ax[1].set_title("Gráfica cuantil-cuantil", loc="left", fontweight="bold")
        ax[1].set_xlabel("cuantiles teóricos"); ax[1].set_ylabel("residuos")
        ax[2].hist(res, bins=40, color=AZUL, edgecolor="white", linewidth=.3)
        ax[2].set(title="Residuos", xlabel="toneladas", ylabel="frecuencia")
        fig.tight_layout(); mostrar(fig)
        q = pd.qcut(aj, 5, labels=False, duplicates="drop")
        het = pd.DataFrame({"aj": aj, "res": res, "q": q}).groupby("q").agg(
            predicho=("aj", "mean"), sd_residuo=("res", "std"))
        het["sd / predicho"] = het.sd_residuo / het.predicho
        st.markdown("**Dispersión del error según el nivel predicho (quintiles)**")
        st.dataframe(het.round(2), width="stretch")
        st.caption(f"Residuos: media {res.mean():+.2f} t · asimetría {stats.skew(res):.2f}")

    salida = pd.DataFrame({"fecha": test.fecha.dt.date, "id_sector": test.id_sector, "observado": test[M.OBJ],
                           **{corto(k).lower().replace(" ", "_"): v for k, v in pred.items()}}).round(3)
    st.download_button("Descargar predicciones de la prueba (CSV)", salida.to_csv(index=False).encode(),
                       "predicciones_prueba.csv", "text/csv")

# --------------------------------------------------------------------------- validación
with tab2:
    res_cv, cv = R["res"], R["cv"]
    st.markdown(f"El grado se elige **solo con entrenamiento**, con validación de ventana expansiva por mes "
                f"(cada mes se valida con todos los anteriores) y la regla de 1 error estándar: elige el polinomio "
                f"más simple cuya validación queda dentro de 1 error estándar del mejor. "
                f"**Grado elegido automáticamente: {R['grado_auto']}.** "
                + (f"Estás usando el grado {R['grado']} (manual)." if grado_manual else ""))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(res_cv.grado, res_cv.media, yerr=res_cv.ee, fmt="o-", color=VIOLETA, lw=2, ms=7, capsize=3)
    fila = res_cv[res_cv.grado == R["grado"]].iloc[0]
    ax.scatter([R["grado"]], [fila.media], s=160, facecolor="none", edgecolor=TINTA, lw=1.8, zorder=4)
    ax.set_xticks(M.GRADOS)
    ax.set(title="Validación: error según el grado del polinomio", xlabel="grado del polinomio (1 = lineal)",
           ylabel=f"RMSE de validación (t), promedio de {cv.mes_idx.nunique()} meses")
    fig.tight_layout(); mostrar(fig)
    piv = cv.assign(mes=cv.mes_idx.map(etiqueta)).pivot(index="grado", columns="mes", values="RMSE")
    st.dataframe(piv.style.format("{:.2f}"), width="stretch")
    st.caption("Ojo: la validación mira 1 mes hacia adelante; la prueba puede mirar varios. Un polinomio que "
               "se ve bien a 1 mes puede fallar a 3 por extrapolar.")

# --------------------------------------------------------------------------- datos
with tab3:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", f"{len(datos):,}")
    c2.metric("Sectores", f"{datos.id_sector.nunique()}")
    c3.metric("Primer día", f"{datos.fecha.min():%d-%m-%Y}")
    c4.metric("Último día", f"{datos.fecha.max():%d-%m-%Y}")
    mm = datos.groupby("mes_txt")[M.OBJ].mean()
    es_prueba = [datos[datos.mes_txt == m].mes_idx.iloc[0] >= primer for m in mm.index]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(range(len(mm)), mm.values, color=[VIOLETA if p else AZUL for p in es_prueba], width=0.7)
    ax.set_xticks(range(len(mm)), [mes_abrev(m, CON_ANIO) for m in mm.index])
    ax.set(title="Toneladas promedio por sector-día, por mes", ylabel="t por sector-día")
    ax.legend(handles=[Patch(color=AZUL, label="entrenamiento"), Patch(color=VIOLETA, label="prueba")],
              ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(); mostrar(fig)
    st.caption("Si el nivel de la prueba es distinto al del entrenamiento hay *deriva*: los modelos entrenados "
               "con el pasado se quedan cortos (o largos) de forma sistemática.")
    st.dataframe(datos[["fecha", "id_sector", M.OBJ, "dias_acum", "visita_ciclo"]].head(200), width="stretch")
    st.caption("Primeras 200 filas.")

# --------------------------------------------------------------------------- acerca de
with tab4:
    st.markdown(f"""
**Qué hace.** Predice las toneladas de residuos sólidos urbanos (RSU) que se recolectan en un sector durante un
día de servicio, y compara tres enfoques: una **línea base** (promedio histórico por sector y días acumulados),
una **regresión lineal** y una **regresión polinomial** (el polinomio se aplica solo al tiempo: `t` y `dia_mes`).

**Predictores.** `id_sector` y `visita_ciclo` (categorías), `acum` = días desde el servicio anterior (tope 4),
`t` (días desde el 1-ene) y `dia_mes`. No usa geografía ni variables que solo se conocen después del servicio.

**Evaluación.** Partición **temporal** (entrena con los meses anteriores, prueba con los últimos), validación de
ventana expansiva para elegir el grado y bootstrap pareado por fecha para saber si la diferencia entre modelos
supera al azar.

**Tu CSV.** Columnas obligatorias: `{', '.join(M.COLUMNAS_MIN)}`. La estructura completa de
`tonelajes_sector_dia.csv` es: `{', '.join(M.COLUMNAS_PLANTILLA)}`; `t`, `dia_mes` y mes se recalculan desde
`fecha`, así que las demás son opcionales. Se necesitan ≥ {M.MIN_MESES} meses, ≥ {M.MIN_FILAS} filas y
≤ {M.MAX_FILAS:,} filas. Un sector que aparezca solo en la prueba se predice con el promedio general.

**Datos sintéticos.** Generados con `generar_sinteticos.py` (semilla fija): 72 sectores en dos zonas con
calendario alterno, una curva de crecimiento que se acelera hacia junio y un nivel de jul–sep ≈ 8 % mayor que el
del entrenamiento (deriva). Sus parámetros se ajustaron para que la demo se comporte como el estudio real: el
polinomial se ajusta bien al pasado pero extrapola hacia arriba y pierde frente al lineal. **No son datos reales**:
ilustran el método y no sirven para sacar conclusiones sobre el servicio.

**Privacidad.** Esta demo es pública: no subas datos confidenciales. El archivo se procesa en memoria y no se guarda.
""")

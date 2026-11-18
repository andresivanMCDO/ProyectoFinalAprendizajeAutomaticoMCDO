"""Pruebas rápidas de la lógica (sin Streamlit):  python prueba_modelo.py"""
import numpy as np
import pandas as pd

import modelo as M

base = pd.read_csv("tonelajes_sector_dia_sintetico.csv")

# 1) la plantilla completa pasa sin errores ni avisos
d, err, av = M.preparar(base)
assert d is not None and not err and not av, (err, av)

# 2) solo las 5 columnas mínimas también funciona y da lo mismo
d2, err, _ = M.preparar(base[M.COLUMNAS_MIN])
assert d2 is not None and np.allclose(d2[M.X[2:]].to_numpy(float), d[M.X[2:]].to_numpy(float))

# 3) columnas faltantes → error claro
_, err, _ = M.preparar(base.drop(columns=["peso_ton_total", "fecha"]))
assert err and "peso_ton_total" in err[0] and "fecha" in err[0]

# 4) vacíos / texto no numérico → error
malo = base.astype(object); malo.loc[3, "peso_ton_total"] = "abc"
assert M.preparar(malo)[0] is None
malo = base.astype(object); malo.loc[5, "fecha"] = "no-es-fecha"
assert M.preparar(malo)[0] is None

# 5) pesos ≤ 0 se excluyen con aviso (como los 4 registros de relleno del estudio)
z = base.copy(); z.loc[[1, 2], "peso_ton_total"] = 0
dz, err, av = M.preparar(z)
assert len(dz) == len(base) - 2 and any("excluyeron 2" in a for a in av)

# 6) pocos meses / pocas filas / demasiadas filas → error
assert M.preparar(base[base.fecha < "2025-05-01"])[0] is None
assert M.preparar(base.head(100))[0] is None
assert M.preparar(pd.concat([base] * 7))[0] is None

# 7) corrida completa: grado válido, métricas finitas, bootstrap con IC ordenado
tr, te = M.dividir(d, 6)
cv, res, g = M.validar_grados(tr)
assert g in (2, 3, 4, 5)
pred, tabla = M.ajustar_y_evaluar(tr, te, g)
assert np.isfinite(tabla.to_numpy()).all() and len(tabla) == 3
k = list(pred); punto, lo, hi = M.bootstrap_diferencia_rmse(te, pred[k[2]], pred[k[1]])
assert lo <= punto <= hi

# 8) un sector que solo aparece en la prueba no rompe nada
te2 = te.copy(); te2.loc[te2.index[0], "id_sector"] = 9999
assert np.isfinite(M.linea_base(tr, te2)).all()
assert np.isfinite(M.armar(1).fit(tr[M.X], tr[M.OBJ]).predict(te2[M.X])).all()
print("OK: todas las pruebas pasaron")

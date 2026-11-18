# Toneladas de RSU por sector-día · demo en Streamlit

App web que predice las toneladas de residuos sólidos urbanos (RSU) por sector y día de servicio y compara
tres enfoques: **línea base** (promedio histórico), **regresión lineal** y **regresión polinomial**.
Por defecto usa **datos sintéticos**; el usuario puede subir su propio CSV con la estructura de
`tonelajes_sector_dia.csv`.

## Entorno de desarrollo

| Componente | Versión / herramienta |
|---|---|
| Sistema | Linux Mint (cualquier Linux o macOS sirve) |
| Python | **3.12.0**, administrado con `pyenv` |
| Aislamiento | entorno virtual `venv` en `.venv/` |
| Dependencias | `requirements.txt` (streamlit, pandas, numpy, scikit-learn, scipy, matplotlib) |

La app se probó con Python 3.12 y con dos conjuntos de librerías: las más recientes y las versiones mínimas
declaradas en `requirements.txt`.

### Preparar el entorno (una sola vez)

```bash
# 1) Python 3.12.0 con pyenv (si aún no lo tienes instalado)
sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
                    libsqlite3-dev libffi-dev liblzma-dev      # dependencias de compilación de pyenv
pyenv install 3.12.0

# 2) En la carpeta del proyecto
cd app_rsu
pyenv shell 3.12.0            # solo vale para esta terminal; hay que repetirlo en cada terminal nueva
python --version              # debe decir Python 3.12.0

# 3) Entorno virtual e instalación
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

El archivo `.python-version` (contiene `3.12.0`) hace que `pyenv` elija esa versión automáticamente dentro de
esta carpeta; `pyenv shell` tiene prioridad sobre él.

### Trabajo diario

```bash
cd app_rsu
pyenv shell 3.12.0
source .venv/bin/activate
streamlit run app.py          # abre http://localhost:8501
python prueba_modelo.py       # pruebas rápidas de la lógica (debe imprimir «OK»)
```

## Contenido

| Archivo | Para qué sirve |
|---|---|
| `app.py` | Interfaz Streamlit (pestañas: Resultados, Validación del grado, Los datos, Acerca de); las gráficas replican las del cuaderno de modelado |
| `modelo.py` | Lógica sin interfaz: validación del CSV, partición temporal, modelos, validación, bootstrap |
| `tonelajes_sector_dia_sintetico.csv` | Datos sintéticos de la demo (8,280 filas, 72 sectores, ene–sep 2025) |
| `generar_sinteticos.py` | Genera el CSV sintético (semilla fija, reproducible) |
| `prueba_modelo.py` | Pruebas de validación y de la corrida completa |
| `requirements.txt` | Dependencias |
| `.streamlit/config.toml` | Límite de carga de 5 MB y tema |
| `.python-version`, `.gitignore` | Versión de Python; evita subir datos reales y credenciales |

## Qué espera el CSV

Columnas obligatorias: `fecha`, `id_sector`, `peso_ton_total`, `dias_acum`, `visita_ciclo`.
La estructura completa de `tonelajes_sector_dia.csv` es
`id_bitacora, fecha, id_sector, zona, peso_ton_total, n_viajes, dia_semana, dias_acum, t, mes, dia_mes, visita_ciclo`;
las columnas extra se ignoran y `t`, `dia_mes` y el mes se recalculan desde `fecha`.
Se necesitan al menos 500 filas y 6 meses distintos; máximo 50,000 filas y 5 MB. Las filas con
`peso_ton_total ≤ 0` se excluyen con un aviso.

## Despliegue en Streamlit Community Cloud

1. Crea un repositorio **público** en GitHub y sube el contenido de esta carpeta
   (`git init`, `git add .`, `git commit`, `git push`). Revisa antes `git status`: no deben ir datos reales
   (`tonelajes_sector_dia.csv`), cuadernos con la cadena de conexión a la base de datos ni credenciales.
2. Entra a <https://share.streamlit.io> con tu cuenta de GitHub y elige **Create app**.
3. Selecciona el repositorio, la rama y el archivo principal `app.py`.
4. En **Advanced settings** elige Python **3.12**.
5. Despliega. Cada `git push` posterior actualiza la app.

## Notas

- **Datos sintéticos:** `generar_sinteticos.py` combina una curva de crecimiento que se acelera hacia junio con un
  nivel de jul–sep ≈ 8 % mayor (deriva). Sus parámetros (`PARAMS`) se calibraron con una búsqueda numérica para
  que el comportamiento se parezca al del estudio real: el polinomial se ajusta bien al pasado pero extrapola
  hacia arriba y pierde frente al lineal, con un sesgo que crece mes a mes. No contiene datos reales; ilustra el
  método y no sirve para concluir nada sobre el servicio. Para volver a generarlo: `python generar_sinteticos.py`.
- **Privacidad:** la demo es pública. El CSV se procesa en memoria y no se guarda, pero conviene no subir datos
  confidenciales.
- **Hibernación:** en el plan gratuito la app duerme tras un tiempo sin visitas; la primera visita tarda unos
  segundos en despertarla.

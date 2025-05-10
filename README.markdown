# Modelo de Predicción de Demanda de Servicios de Salud SURA

Este proyecto desarrolla un pipeline de Machine Learning para predecir la demanda de servicios de salud de SURA Colombia, optimizando la asignación de recursos y la planificación operativa. El modelo pronostica la demanda (`Service_Count`) a 12 meses, desglosada por municipio y tipo de servicio, utilizando datos históricos agregados mensualmente. Los resultados están disponibles en el siguiente dashboard: [Salud SURA Insights Dashboard](https://salud-sura-insights-dashboard.onrender.com).

---

## Tabla de Contenidos
1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Objetivo del Proyecto](#objetivo-del-proyecto)
3. [Datos Utilizados](#datos-utilizados)
4. [Preprocesamiento e Ingeniería de Características](#preprocesamiento-e-ingeniería-de-características)
5. [Transformación del Target](#transformación-del-target)
6. [Selección y Entrenamiento de Modelos](#selección-y-entrenamiento-de-modelos)
7. [Evaluación del Modelo](#evaluación-del-modelo)
8. [Análisis de Resultados y Visualizaciones](#análisis-de-resultados-y-visualizaciones)
9. [Implementación Matemática](#implementación-matemática)
10. [Predicción Futura e Implementación](#predicción-futura-e-implementación)
11. [Estructura del Código](#estructura-del-código)
12. [Explicación Paso a Paso](#explicación-paso-a-paso)
    - [Importaciones y Configuración](#1-importaciones-y-configuración)
    - [Funciones Auxiliares](#2-funciones-auxiliares)
    - [Ejecución Principal](#3-ejecución-principal)
13. [Fundamentos Matemáticos](#fundamentos-matemáticos)
14. [Por Qué Se Toman Ciertos Pasos](#por-qué-se-toman-ciertos-pasos)
---

## Resumen Ejecutivo
Este proyecto aborda la necesidad de anticipar la demanda de servicios de salud para SURA Colombia, con el fin de optimizar la asignación de recursos y la planificación operativa. Se desarrolló un pipeline de Machine Learning para generar pronósticos a 12 meses, detallados por `Municipio` y `Tipo de Servicio`, utilizando datos históricos agregados mensualmente.

Tras un proceso iterativo de ingeniería de características, selección de modelos, entrenamiento, evaluación y corrección de errores (incluyendo la identificación y exclusión de *features* con *data leakage*), se obtuvo un modelo **Ensamble** (promedio de LightGBM, XGBoost y RandomForest) como el de mejor rendimiento en el conjunto de validación (datos de 2024). El modelo final muestra una **alta precisión predictiva** (MAE ~0.62 servicios, R² ~0.9578 en validación) para la **demanda agregada mensual**, superando significativamente a un *baseline* simple.

**Limitaciones clave**:
- **Falta de datos** para identificar servicios laborales específicos (Accidentes de Trabajo y Enfermedades Laborales - ATEL).
- **Imposibilidad de incorporar datos de capacidad** de la red de prestadores debido a inconsistencias en los identificadores de municipio entre datasets.

Las predicciones para 2025 muestran una **tendencia decreciente**, probablemente influenciada por patrones recientes en los datos históricos, que requiere validación experta.

**Próximos pasos**:
- Mejorar el preprocesamiento para incluir la identificación de servicios laborales.
- Resolver el mapeo de IDs para incorporar datos de capacidad.

Los resultados completos están disponibles en el [Salud SURA Insights Dashboard](https://salud-sura-insights-dashboard.onrender.com).

---

## Objetivo del Proyecto
**Problema**: La variabilidad en la demanda de servicios de salud dificulta la planificación eficiente de recursos (personal médico, insumos, camas, etc.). Una predicción imprecisa puede llevar a sobrecostos o a una atención deficiente.

**Objetivo General**: Desarrollar un sistema de modelos de Machine Learning capaces de predecir la demanda de servicios de salud con 12 meses de antelación.

**Objetivos Específicos**:
- Generar predicciones mensuales de la cantidad de servicios (`Service_Count`).
- Desglosar las predicciones por `Municipio` y `Tipo de Servicio`.
- Incorporar la influencia de factores como tendencias históricas, estacionalidad y características específicas de cada serie (Municipio-Servicio).
- Predecir la demanda de servicios derivados de **Accidentes de Trabajo y Enfermedades Laborales (ATEL)**. *(Nota: Este objetivo no se cumplió completamente por limitaciones en los datos)*.
- Evaluar rigurosamente la precisión y robustez de los modelos.

---

## Datos Utilizados
Se trabajó con tres fuentes de datos principales:
1. **`muestra_salud_.csv`**: Dataset reducido (~11 millones de registros en el original) con detalles de atenciones individuales. **Fuente primaria**.
2. **`healthcare_train_data.csv` / `healthcare_valid_data.csv`**: Datasets derivados, **agregados mensualmente** por `Municipio` (de `Nombre_Municipio_IPS`) y `Service_Type` (de `Nombre_Tipo_Atencion_Arp`). Contienen ~14,000 filas en total y fueron la **base para el modelado**.
   - **Target**: `Service_Count` (conteo de atenciones/siniestros por grupo/mes).
   - **Impacto de la agregación**: La agregación mensual es necesaria para el *forecasting* mensual, pero pierde granularidad diaria/semanal y suaviza la variabilidad. Los modelos predicen la **tendencia agregada mensual**, no eventos diarios.
3. **`2.Red Prestadores.xlsx`**: Información de IPS, incluyendo `Geogra_Municipio_Id` y `max_cantidad` (indicador potencial de capacidad).

---

## Preprocesamiento e Ingeniería de Características
El preprocesamiento (realizado externamente y cargado en los CSVs `healthcare_train_data.csv` y `healthcare_valid_data.csv`) es fundamental. Basado en el código de preprocesamiento proporcionado, se identificaron y utilizaron las siguientes transformaciones y features:

* **Variable Objetivo:** `Service_Count` (conteo de atenciones).
* **Transformación del Target (`Log1p`):**
    * **Por qué:** Los datos de conteo suelen tener distribuciones asimétricas (skewness). La transformación $y_{transformado} = \log(1 + y_{original})$ ayuda a:
        * Estabilizar la varianza.
        * Reducir el impacto de valores atípicos extremos.
        * Hacer que la distribución se asemeje más a una normal, lo cual puede beneficiar a algunos algoritmos (aunque los de árbol son robustos a esto).
    * Las predicciones se revierten con $y_{original} = \exp(y_{transformado}) - 1$ antes de la evaluación.

* **Features Temporales:**
    * `Date` (primer día del mes), `Year`, `Month`, `Quarter`, `Month_of_Year`, `Year_Fraction`.
    * **`Month_sin/cos`, `Quarter_sin/cos`**: **Por qué:** Convierten variables cíclicas (mes, trimestre) en dos dimensiones continuas, permitiendo a los modelos capturar la proximidad entre finales y principios de ciclo (ej. Diciembre y Enero).
        * Mes: $X_{sin} = \sin(2\pi \cdot \text{mes} / 12)$, $X_{cos} = \cos(2\pi \cdot \text{mes} / 12)$

* **Encoding Categórico:** `Municipality_encoded`, `Service_Type_encoded`.
    * **Por qué:** Los modelos de ML requieren entradas numéricas. `LabelEncoder` asigna un entero único a cada categoría. Se guardaron `decoders.joblib` para revertir a nombres originales.

* **Features de Lags (`Service_Count_lag_X`):** (X=1, 2, 3, 6, 12 meses).
    * **Por qué:** La demanda de servicios de salud a menudo muestra una fuerte **autocorrelación**: la demanda en un período está influenciada por la demanda en períodos recientes. Los lags $Y_{t-k}$ son cruciales para capturar esta dependencia temporal. Se incluyeron lags de corto plazo (1-3 meses), mediano plazo (6 meses) y estacionales (12 meses).

* **Features Rolling Statistics (`Service_Count_rolling_mean/std_X`):** (X=6, 12 meses).
    * **Por qué:** Las medias móviles suavizan las fluctuaciones a corto plazo y revelan tendencias locales. Las desviaciones estándar móviles capturan la volatilidad reciente.
    * **Advertencia Preprocesamiento Original:** Si el cálculo no usó `.shift(1)` o `closed='left'`, hay un **leakage leve** (se usa el valor del mes actual en su propio cálculo de rolling stat), lo que puede inflar métricas. Se recomendó corregirlo.

* **Features de Crecimiento (`Growth_Rate_MoM`, `Growth_Rate_YoY`):**
    * **Por qué:** Capturan la tasa de cambio relativa, indicando aceleración o desaceleración en la demanda. $G_{MoM} = (Y_t / Y_{t-1}) - 1$.

* **Features de Historial:** `Days_Since_First_Service`, `Month_Sequence`.
    * **Por qué:** Reflejan la "antigüedad" o "madurez" de una serie temporal específica (Municipio-Servicio), lo que podría influir en su patrón de demanda.

* **Features de Incapacidad:** `Mean/Median/Total_Incapacity_Days`.
    * **Por qué:** Potencialmente correlacionadas con la severidad o el tipo de servicios demandados.

* **Features Excluidas (Por Data Leakage):**
    * `Days_From_Now`: **Razón:** Utilizaba la fecha de ejecución del script (`pd.Timestamp.now()`), introduciendo información futura en el conjunto de entrenamiento.
    * `Is_Anomaly`: **Razón:** Se calculaba usando la media y desviación estándar de *toda* la serie histórica del grupo, incluyendo puntos futuros respecto a una observación dada.
    * **Impacto de Exclusión:** La eliminación de estas features es **CRUCIAL** para una evaluación realista del modelo. Su presencia llevaba a R² artificialmente perfectos (ej. 0.9998).

---

## Transformación del Target
- **Problema**: Los datos de `Service_Count` tienen una distribución asimétrica (muchos valores bajos, pocos altos), afectando modelos sensibles a errores grandes.
- **Solución**: Transformación `np.log1p` (logaritmo natural de 1 + x) al target.
- **Razón**: Comprime el rango, reduce asimetría y estabiliza varianza, mejorando el aprendizaje de patrones. Predicciones invertidas con `np.expm1`.

---

## Selección y Entrenamiento de Modelos

### Enfoque General
Se abordó el problema como una tarea de **regresión supervisada sobre series de tiempo**. Para cada combinación única de `Municipio` y `Tipo de Servicio`, se busca predecir el `Service_Count` mensual. Aunque se modelan múltiples series, se entrenó un **único modelo global** que utiliza `Municipality_encoded` y `Service_Type_encoded` como features categóricas, permitiendo al modelo aprender patrones específicos por serie y generalizar entre ellas.

### Modelos Seleccionados
Se optó por modelos basados en **ensambles de árboles de decisión** debido a su robustez y alto rendimiento en datos tabulares:

* **LightGBM (Light Gradient Boosting Machine):**
    * **Concepto:** Algoritmo de Gradient Boosting que construye árboles de decisión secuencialmente, donde cada nuevo árbol corrige los errores del anterior. Usa un histograma para optimizar la búsqueda de splits y crecimiento por hoja (leaf-wise) para mayor eficiencia y precisión.
    * **Por qué:** Rápido, eficiente en memoria, buen manejo de features categóricas, robusto a outliers (con función de pérdida L1).
* **XGBoost (Extreme Gradient Boosting):**
    * **Concepto:** Otra implementación optimizada de Gradient Boosting, con regularización avanzada (L1, L2, complejidad del árbol) para controlar el overfitting y manejo eficiente de valores faltantes.
    * **Por qué:** Rendimiento de vanguardia, popular y con muchas opciones de configuración.
* **Random Forest:**
    * **Concepto:** Algoritmo de Bagging (Bootstrap Aggregating). Entrena múltiples árboles de decisión profundos sobre subconjuntos aleatorios de los datos (con reemplazo) y features. La predicción final es el promedio de las predicciones de todos los árboles.
    * **Por qué:** Robusto al overfitting (en comparación con un solo árbol), fácil de usar, captura no linealidades.
* **Ensamble Simple (Promedio):**
    * **Concepto:** Se promedian las predicciones de los modelos individuales (LGBM, XGBoost, RF).
    * **Por qué:** A menudo mejora la generalización y reduce la varianza del error, llevando a predicciones más estables y, a veces, más precisas que cualquier modelo individual.

### División de Datos y Validación
* **Temporalidad:** Es crucial en series de tiempo. Se usaron datos hasta finales de **2023 para entrenamiento** y los datos de **2024 para validación** (out-of-time validation).
    * **Rationale:** Simula el escenario real donde se predice el futuro usando solo información pasada. Evita el data leakage de usar datos futuros en el entrenamiento.

### Ajuste Fino de Hiperparámetros (RandomForest)
* **Método:** Se usó `RandomizedSearchCV` con `TimeSeriesSplit` (validación cruzada que respeta el orden temporal).
    * **`RandomizedSearchCV`:** Explora un subconjunto aleatorio de combinaciones de hiperparámetros. **Rationale:** Más eficiente que `GridSearchCV` cuando el espacio de búsqueda es grande.
    * **`TimeSeriesSplit`:** Asegura que los folds de entrenamiento siempre precedan a los folds de prueba.
* **Parámetros Ajustados (Ejemplo):** `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.
* **Resultado (Ejemplo de corrida v10):**
    ```
    {'max_depth': 30, 'max_features': 0.820..., 'min_samples_leaf': 3, 'min_samples_split': 11, 'n_estimators': 108}
    ```
    * Se aplicó un límite manual a `max_depth` (ej. 25) como precaución adicional contra el overfitting.

---

## Evaluación del Modelo

* **MAE (Mean Absolute Error):** $\frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$
    * **Por qué:** Error promedio en las unidades originales del target. Fácil de interpretar. Menos sensible a outliers que RMSE. **Métrica principal para selección.**
* **RMSE (Root Mean Squared Error):** $\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$
    * **Por qué:** Penaliza más los errores grandes. Útil si los errores grandes son particularmente costosos.
* **R² (Coeficiente de Determinación):** $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$
    * **Por qué:** Proporción de la varianza en el target que es predecible a partir de las features. Un valor de 1 indica predicción perfecta.
* **Resultados Finales (v13 - sin capacidad, sin leaky):** El Ensamble fue el mejor (MAE ~0.6179, R² ~0.9940).

### Prueba Estadística (Wilcoxon Signed-Rank Test)
* **Propósito:** Comparar si las distribuciones de los errores absolutos del mejor modelo y un modelo Naive (lag 1: $\hat{y}_t = y_{t-1}$) son significativamente diferentes.
* **Hipótesis:**
    * H0: Mediana de las diferencias de error = 0 (modelo no es mejor que Naive).
    * H1: Mediana de las diferencias de error < 0 (modelo es mejor que Naive).
* **Resultado:** p <<< 0.001 consistentemente, indicando **mejora estadísticamente significativa** sobre el baseline.

- **Métricas**: MAE (error interpretable), RMSE (penaliza errores grandes), R² (varianza explicada).
- **Resultados (Validación, Escala Original)**:

| Modelo       | MAE    | RMSE   | R²     |
|--------------|--------|--------|--------|
| LightGBM     | 0.6183 | 5.1097 | 0.9245 |
| XGBoost      | 0.9077 | 6.8716 | 0.8932 |
| RandomForest | 0.6650 | 4.9433 | 0.9048 |
| **Ensamble** | **0.6179** | **4.8295** | **0.9578** |

- **Mejor Modelo**: Ensamble (ligeramente superior en MAE).
- **MAE Bajo**: < 1 servicio, indicador práctico positivo.
- **Wilcoxon Test**: p <<< 0.05, confirma que el Ensamble supera significativamente un *baseline* simple (predictor de retraso-1).

---

## Análisis de Resultados y Visualizaciones

### Rendimiento General en Validación
* **Scatter Plot (Real vs. Predicción):** Puntos agrupados cerca de la diagonal y=x, mostrando buena correlación general.
* **Serie de Tiempo Agregada:** La predicción agregada sigue bien la tendencia y estacionalidad de los datos reales agregados en el periodo de validación.
* **Intervalos de Confianza (Quantiles):** Estimados con LGBM Quantile Regression, proporcionan un rango de predicción para la demanda agregada, útil para entender la incertidumbre.

### Análisis de Errores
* **Distribución de Errores:** Idealmente centrada en cero y simétrica. El histograma/KDE ayuda a visualizar esto.
* **Residuos Agregados vs. Tiempo:** Se grafica el error promedio (Real - Predicción) a lo largo del tiempo. **Por qué:** Permite detectar si el modelo tiene sesgos sistemáticos (ej. subestimar consistentemente en ciertos meses) o si los errores aumentan con el horizonte de predicción. Idealmente, los residuos deben ser ruido blanco.

### Rendimiento por Segmento
* **Ejemplo (Medellin - Ambulatoria):** Se graficó la serie real vs. predicha para esta combinación de alto volumen y volatilidad. El modelo captura la tendencia general pero **suaviza los picos extremos**, un comportamiento común. Se calculó el MAE específico.
* **Mejores/Peores 5 Series:** Se identificaron y graficaron las 5 series (Municipio-Servicio) con menor y mayor MAE en validación. **Por qué:** Ayuda a entender dónde el modelo es más/menos preciso y a guiar posibles mejoras futuras o la aceptación de diferentes niveles de error por segmento.

### Importancia de Features
Se generaron gráficas para cada modelo individual (LGBM, XGBoost, RF).
* **Observaciones Clave:**
    * **Lags y Rolling Means:** Dominantes en XGBoost y RF (ej. `Service_Count_rolling_mean_6`, `Service_Count_lag_1`). **Rationale:** Confirma la alta dependencia temporal.
    * **LGBM:** Dio más peso a `Municipality_encoded` y `Growth_Rate_MoM`. **Rationale:** Los algoritmos de boosting pueden capturar interacciones y no linealidades de forma diferente.
    * `Mean_Incapacity_Days`: Mostró cierta importancia para XGBoost.
    * `Growth_Rate_MoM/YoY`: Baja importancia en XGB/RF, pero influyente en LGBM y en la tendencia general predicha.
* **Utilidad:** Entender qué features impulsan las predicciones ayuda a validar el modelo y a guiar la ingeniería de características futura.

### Tendencia Predicha para 2025
* **Observación:** Las predicciones muestran una **tendencia general a la baja** a lo largo de 2025.
* **Causa Probable:** Extrapolación de la tendencia negativa observada en el `Growth_Rate_MoM` promedio durante los últimos meses de 2024 (visualizada en su propia gráfica).
* **Implicación:** Es crucial validar esta tendencia con conocimiento del negocio. Si es una fluctuación y no un cambio estructural, el modelo podría estar siendo pesimista.


---

## Implementación Matemática
### Modelo de Predicción

$$
\hat{y}(t) = \beta_0 + \sum (\beta_i \cdot x_i(t)) + \gamma \cdot s(t) + \epsilon(t)
$$

Donde:

- $\hat{y}(t)$: Predicción en el tiempo $t$.
- $\beta_0$: Término de intercepto.
- $x_i(t)$: Variables explicativas temporales (por ejemplo, rezagos o medias móviles).
- $s(t)$: Componente estacional (representada mediante funciones seno/coseno).
- $\epsilon(t)$: Término de error (ruido no explicado por el modelo).
### Transformación Logarítmica
```python
y_transformed = np.log1p(y_original)
y_pred_original = np.expm1(y_pred_transformed)
```
- **Razón**: $(\log(1 + x)\)$ estabiliza varianza, $(\exp(x) - 1\)$ es su inversa exacta.

### Descomposición Temporal
```python
df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
```
- **Teoría**: Representa funciones periódicas como:  
  $f(t) = a_0 + \sum [a_n \cdot \cos(n\omega t) + b_n \cdot \sin(n\omega t)]$,  
  donde $\omega = \frac{2\pi}{T}$.

### Arquitectura del Modelo
```mermaid
graph TD
    A[Datos Históricos] --> B[Preprocesamiento]
    B --> C[Feature Engineering]
    C --> D[Entrenamiento Paralelo]
    D --> E[LightGBM]
    D --> F[XGBoost]
    D --> G[Random Forest]
    E --> H[Ensamble]
    F --> H
    G --> H
    H --> I[Predicciones]
    I --> J[Evaluación]
    J --> K[Despliegue]
```

---

## Predicción Futura e Implementación
- **Método Iterativo**: Predice mes \(t+1\), usa resultado para calcular *features* de $(t+2\)$, etc.
- **Implementación**:
  - Bucle `for` sobre 12 meses futuros.
  - `generate_future_features` recalcula *features* usando historial actualizado (`all_predictions_df`).
  - Corrección: Lags obtenidos con `.map()` sobre MultiIndex para evitar errores de `pd.merge`.
- **Salidas**:
  - Modelos y decodificadores `.joblib`.
  - CSV: `healthcare_demand_forecast_detailed_12_months.csv`.
  - JSON: `model_analysis_results_v14.json` (métricas, resúmenes).

---

## Estructura del Código
El código es modular, con los siguientes componentes:
1. **Importaciones y Configuración**: Bibliotecas y configuraciones globales.
2. **Funciones Auxiliares**: Carga de datos, preparación de *features*, entrenamiento, evaluación, visualización.
3. **Ejecución Principal**: Orquesta el flujo desde la carga de datos hasta las predicciones.

---

## Explicación Paso a Paso

### 1. Importaciones y Configuración
#### Código
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import wilcoxon, randint, uniform
import joblib
import warnings
import re
import time
import json
import os
import geopandas as gpd
from unidecode import unidecode
warnings.filterwarnings('ignore')

APPLY_LOG_TRANSFORM = True
DO_TUNING = False
TUNED_RF_MAX_DEPTH_LIMIT = 25
best_rf_params = {...}
JSON_OUTPUT_PATH = 'model_analysis_results_v14.json'
N_BEST_WORST_SERIES = 5
GEOJSON_URL = "..."
```

#### Explicación
- **Bibliotecas**:
  - `pandas`, `numpy`: Manipulación de datos.
  - `matplotlib`, `seaborn`: Visualización.
  - `lightgbm`, `xgboost`, `sklearn`: Modelos y métricas.
  - `scipy.stats`: Pruebas estadísticas (Wilcoxon).
  - `joblib`: Serialización.
  - `geopandas`, `unidecode`: Análisis geoespacial (no utilizado completamente).
  - `warnings`: Suprime advertencias no críticas.
- **Configuración**:
  - `APPLY_LOG_TRANSFORM`: Aplica `log1p` a `Service_Count`.
  - `DO_TUNING`: Usa parámetros preajustados si es `False`.
  - `best_rf_params`: Parámetros preajustados para RandomForest.
  - `JSON_OUTPUT_PATH`: Archivo de resultados.
  - `N_BEST_WORST_SERIES`: Series con mejor/peor MAE a graficar.
  - `GEOJSON_URL`: GeoJSON de municipios (no utilizado).

#### ¿Por Qué?
- **Transformación Logarítmica**: Reduce sesgo en `Service_Count`, estabiliza varianza.
- **Parámetros Preajustados**: Ahorra tiempo en ajuste de hiperparámetros.
- **Supresión de Advertencias**: Evita clutter en la salida.

#### Razonamiento Matemático
- **Transformación Logarítmica**:
  $y_{\text{transformado}} = \log(1 + y)$
  Comprime valores grandes, inversa: $(\exp(y) - 1\)$.
- **Encoding Categórico**: Convierte categorías a enteros, eficiente para modelos de árboles.

---

### 2. Funciones Auxiliares

#### 2.1 `load_data`
```python
def load_data(train_path, valid_path):
    try:
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        for df_name, df in [('Train', train_df), ('Valid', valid_df)]:
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                if df['Date'].isnull().any():
                    df.dropna(subset=['Date'], inplace=True)
            else:
                raise ValueError(f"Columna 'Date' no encontrada en {df_name}.")
        return train_df, valid_df
    except Exception as e:
        print(f"Error Fatal: {e}"); exit()
```

**Propósito**: Carga datasets y convierte `Date` a *datetime*.

**Explicación**:
- Lee CSVs en DataFrames.
- Convierte `Date` a *datetime*.
- Elimina filas con `Date` inválido.
- Sale si hay errores críticos.

**¿Por Qué?**: Fechas consistentes son esenciales para *features* temporales.

---

#### 2.2 `create_decoders`
```python
def create_decoders(df, muni_enc_col, muni_orig_col, serv_enc_col, serv_orig_col):
    global decoders
    decoders_local = {}
    if muni_orig_col in df.columns and muni_enc_col in df.columns:
        mun_map = df[[muni_enc_col, muni_orig_col]].drop_duplicates()
        mun_map[muni_enc_col] = pd.to_numeric(mun_map[muni_enc_col], errors='coerce').fillna(-1).astype(int)
        decoders_local[muni_enc_col] = pd.Series(mun_map[muni_orig_col].values, index=mun_map[muni_enc_col]).to_dict()
    if serv_orig_col in df.columns and serv_enc_col in df.columns:
        st_map = df[[serv_enc_col, serv_orig_col]].drop_duplicates()
        st_map[serv_enc_col] = pd.to_numeric(st_map[serv_enc_col], errors='coerce').fillna(-1).astype(int)
        decoders_local[serv_enc_col] = pd.Series(st_map[serv_orig_col].values, index=st_map[serv_enc_col]).to_dict()
    joblib.dump(decoders_local, 'decoders.joblib')
    decoders = decoders_local
```

**Propósito**: Mapea IDs codificados a nombres originales.

**Explicación**:
- Extrae pares únicos codificado-original.
- Convierte IDs a enteros.
- Guarda mapeos en `decoders.joblib`.

**¿Por Qué?**: IDs codificados son eficientes, nombres originales son legibles.

---

#### 2.3 `select_and_prepare_features`
```python
def select_and_prepare_features(df, target_col, date_col, id_cols_remove, leaky_cols):
    core_features = [
        'Municipality_encoded', 'Service_Type_encoded', 'Year_Fraction', 'Month_sin', 'Month_cos',
        'Quarter_sin', 'Quarter_cos', 'Quarter', 'Month_of_Year', 'Service_Count_lag_1', 
        'Service_Count_lag_2', 'Service_Count_lag_3', 'Service_Count_lag_6', 'Service_Count_lag_12',
        'Service_Count_rolling_mean_6', 'Service_Count_rolling_std_6', 
        'Service_Count_rolling_mean_12', 'Service_Count_rolling_std_12', 
        'Days_Since_First_Service', 'Month_Sequence', 'Is_Holiday_Month', 
        'Mean_Incapacity_Days', 'Median_Incapacity_Days', 'Total_Incapacity_Days', 
        'Growth_Rate_MoM', 'Growth_Rate_YoY'
    ]
    cols_to_remove = [target_col, date_col] + id_cols_remove + leaky_cols + ['Is_Work_Related']
    capacity_cols_to_exclude = ['Total_Max_Cantidad_Municipio', 'Avg_Max_Cantidad_Municipio', 
                               'Std_Max_Cantidad_Municipio', 'Provider_Count_Municipio']
    cols_to_remove.extend(capacity_cols_to_exclude)
    features = [f for f in core_features if f in df.columns and f not in cols_to_remove]
    return sorted(list(set(features)))
```

**Propósito**: Selecciona *features* relevantes, excluye *leakage*.

**Explicación**:
- Define *core_features*: categóricas, temporales, lags, rolling, crecimiento.
- Excluye target, fecha, IDs, *leaky* y capacidad.
- Devuelve lista ordenada de *features*.

**¿Por Qué?**:
- Reduce complejidad, evita *overfitting*.
- Excluye *leakage* para predicciones realistas.

**Razonamiento Matemático**:

- **Lags**:  
  `Service_Count_lag_k(t) = Service_Count(t - k)`

- **Rolling**:  
  `rolling_mean_w(t) = (1/w) * Σ[i=1 to w] Service_Count(t - i)`  
  `rolling_std_w(t) = sqrt((1/w) * Σ[i=1 to w] (Service_Count(t - i) - rolling_mean_w(t))^2)`

- **Seno/Coseno**:  
  `Month_sin = sin(2π * mes / 12)`

- **Crecimiento**:  
  `Growth_Rate_MoM = Service_Count(t-1) / Service_Count(t-2) - 1`

---

#### 2.4 `prepare_xy`
```python
def prepare_xy(df, features, target_col, muni_enc_col, apply_log_transform=False):
    features_in_df = [f for f in features if f in df.columns]
    X = df[features_in_df].copy()
    y_orig = df[target_col].copy()
    if apply_log_transform:
        y = np.log1p(y_orig)
    else:
        y = y_orig.copy()
    if muni_enc_col in X.columns:
        X[muni_enc_col] = pd.to_numeric(X[muni_enc_col], errors='coerce').fillna(-1).astype(np.int64)
    non_numeric_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    for col in non_numeric_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    return X, y, y_orig, features_in_df
```

**Propósito**: Prepara matriz `X` y vectores `y`.

**Explicación**:
- Selecciona *features*.
- Aplica `log1p` a `y` si aplica.
- Convierte a numérico, rellena NaNs.

**¿Por Qué?**: Asegura compatibilidad con modelos.

---

#### 2.5 `train_evaluate_model`
```python
def train_evaluate_model(model_name, model_obj, X_train, y_train, X_valid, y_valid_orig,
                         cat_indices, apply_log_transform, X_train_xgb=None, X_valid_xgb=None):
    y_valid_eval = np.log1p(y_valid_orig) if apply_log_transform else y_valid_orig
    fit_params = {}
    if not isinstance(model_obj, RandomForestRegressor):
        eval_X = X_valid_xgb if X_valid_xgb is not None else X_valid
        fit_params['eval_set'] = [(eval_X, y_valid_eval)]
        if isinstance(model_obj, lgb.LGBMRegressor):
            fit_params['eval_metric'] = ['mae', 'rmse']
            fit_params['callbacks'] = [lgb.early_stopping(50, verbose=False)]
            fit_params['categorical_feature'] = cat_indices
        elif isinstance(model_obj, xgb.XGBRegressor):
            fit_params['verbose'] = False
    train_data = X_train_xgb if X_train_xgb is not None else X_train
    predict_data = X_valid_xgb if X_valid_xgb is not None else X_valid
    model_obj.fit(train_data, y_train, **fit_params)
    preds_transformed = model_obj.predict(predict_data)
    preds_original = np.expm1(preds_transformed) if apply_log_transform else preds_transformed
    preds_original = np.maximum(0, preds_original)
    mae = mean_absolute_error(y_valid_orig, preds_original)
    rmse = np.sqrt(mean_squared_error(y_valid_orig, preds_original))
    r2 = r2_score(y_valid_orig, preds_original)
    errors = y_valid_orig - preds_original
    joblib.dump(model_obj, f'model_{model_name.lower()}.joblib')
    return model_obj, preds_original, errors, mae, rmse, r2
```

**Propósito**: Entrena, predice, evalúa.

**Explicación**:
- Configura parámetros por modelo.
- Entrena, predice, revierte `log1p`.
- Asegura predicciones no negativas.
- Calcula MAE, RMSE, R².

**¿Por Qué?**:
- Asegura entrenamiento robusto.
- Predicciones no negativas son realistas.

---

#### 2.6 `plot_evaluation_graphs`
**Propósito**: Genera visualizaciones.

**Gráficos**:
- Scatter Real vs. Predicho.
- Serie Temporal Agregada/Ejemplo.
- Cuantiles (90% intervalo).
- Crecimiento MoM.
- Residuos.
- Mejores/Peores Series.
- Importancia de *Features*.

**¿Por Qué?**: Evalúa precisión, identifica patrones.

**Razonamiento Matemático**:
- **Cuantiles**:

Perdida_alpha(y, y_hat) = {  
  alpha * (y - y_hat)  if y >= y_hat  
  (1 - alpha) * (y_hat - y)  if y < y_hat
}



---

#### 2.7 `generate_future_features`
**Propósito**: Genera *features* futuras.

**Explicación**:
- Crea DataFrame para fecha futura.
- Calcula *features* temporales, lags, rolling, crecimiento.
- Propaga valores conocidos.

**¿Por Qué?**: Necesario para pronósticos iterativos.

---

#### 2.8 `plot_feature_importance`
**Propósito**: Visualiza importancia de *features*.

**Explicación**:
- LightGBM: Ganancia.
- XGBoost: Puntajes.
- RandomForest: Disminución de impureza.

**¿Por Qué?**: Identifica impulsores de predicciones.

---

#### 2.9 `analyze_errors_by_group`
**Propósito**: Calcula MAE por grupo.

**Explicación**:
- Agrupa por municipio/servicio.
- Calcula MAE.

**Razonamiento Matemático**:
- **MAE Agrupado**:
MAE_grupo = (1 / n_grupo) * Σ[i ∈ grupo] |Real_i - Predicho_i|


---

### 3. Ejecución Principal
La función `main` orquesta:
1. Carga datos.
2. Crea decodificadores.
3. Selecciona *features*.
4. Prepara datos.
5. Entrena LightGBM, XGBoost, RandomForest, Ensamble.
6. Evalúa (MAE, RMSE, R², Wilcoxon).
7. Genera visualizaciones.
8. Predice 12 meses futuros.
9. Analiza importancia y errores.

---

## Fundamentos Matemáticos
1. **Series Temporales**:
   - Lags, rolling: Dependencias temporales.
   - Seno/Coseno: Estacionalidad continua.
2. **Transformación Logarítmica**:
   $$
   y = \log(1 + \text{Service_Count})
   $$
3. **Modelos**:
   - LightGBM/XGBoost: Minimizan pérdida.
   - RandomForest: Reduce varianza.
   - Ensamble: Promedia para robustez.
4. **Métricas**:
   - MAE, RMSE, R².
5. **Wilcoxon**:
   $$
   W = \sum_{i: d_i \neq 0} \text{rango}(|d_i|) \cdot \text{signo}(d_i)
   $$
6. **Cuantiles**: Intervalos de incertidumbre.

---

## Por Qué Se Toman Ciertos Pasos
1. **Transformación Logarítmica**: Normaliza distribución sesgada.
2. **Ingeniería de Características**: Captura tendencias, estacionalidad, crecimiento.
3. **Excluir Leakage**: Asegura predicciones realistas.
4. **Ensamble**: Reduce sesgos individuales.
5. **Wilcoxon**: Valida superioridad sobre *baseline*.
6. **Cuantiles**: Proporciona incertidumbre.
7. **Predicciones No Negativas**: Realismo.
8. **Pronóstico Iterativo**: Mimica escenarios reales.

---


**Conclusión**: El pipeline produce predicciones mensuales precisas, superando un *baseline* simple. Sin embargo, la identificación de servicios laborales y la incorporación de datos de capacidad requieren mejoras en el preprocesamiento y mapeos de datos.

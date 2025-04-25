#  Modelo de Predicción de Demanda de Servicios de Salud

Para poder ver la pagina con todos los resultados encontrados entrar aquí: https://salud-sura-insights-dashboard.lovable.app/

## 1. Objetivo del Proyecto

El objetivo principal de este proyecto es desarrollar modelos de Machine Learning para predecir la demanda futura de servicios de salud para los próximos 12 meses. Las predicciones deben ser detalladas por:

* Tipo de servicio médico.
* Municipio y tipo de servicio médico.

Se requiere específicamente incluir (aunque no fue posible en la iteración final por falta de datos) las atenciones relacionadas con accidentes y enfermedades laborales. Los modelos deben incorporar factores como estacionalidad, tendencias históricas, características demográficas (implícitas en Municipio) e indicadores económicos locales (si estuvieran disponibles y fueran incorporados como features).

## 2. Datos Utilizados

Se utilizaron principalmente los siguientes conjuntos de datos:

1.  **`muestra_salud_.csv`**: Dataset original con aproximadamente 11 millones de registros detallados de atenciones médicas. Contiene información granular sobre fechas, pacientes, IPS, médicos, diagnósticos, tipos de atención, etc.
2.  **`healthcare_train_data.csv` / `healthcare_valid_data.csv`**: Datos preprocesados y **agregados mensualmente** a nivel de `Municipio` y `Service_Type` (derivado de `Nombre_Tipo_Atencion_Arp`). Estos fueron los datasets principales para entrenar y validar los modelos de forecasting mensual. Contienen la variable objetivo `Service_Count` y múltiples features de ingeniería.
    * **Nota Importante sobre Agregación:** La agregación de 11M de registros a ~14k filas mensuales implica una pérdida significativa de granularidad diaria/semanal. Los modelos resultantes predicen la **demanda agregada mensual** y no pueden capturar fluctuaciones diarias ni predecir eventos individuales.
3.  **`2.Red Prestadores.xlsx - Sheet1.csv`**: Contiene información sobre los prestadores de servicios (IPS), incluyendo un ID de municipio (`Geogra_Municipio_Id`) y una métrica de capacidad (`max_cantidad`). Se intentó incorporar esta información, pero **falló debido a la incompatibilidad entre `Geogra_Municipio_Id` y `Municipality_encoded`** (el ID usado en los datos agregados). Se requiere una tabla de mapeo para poder utilizar estas features.

## 3. Preprocesamiento e Ingeniería de Características

El preprocesamiento inicial (realizado antes de este análisis y reflejado en `healthcare_train_data.csv`/`healthcare_valid_data.csv`) incluyó los siguientes pasos clave (basado en el código proporcionado por el usuario):

* **Agregación Mensual:** Agrupación de los datos originales por `Año`, `Mes`, `Nombre_Municipio_IPS`, `Nombre_Tipo_Atencion_Arp`.
* **Cálculo de Target:** `Service_Count` (conteo de siniestros/atenciones).
* **Features de Incapacidad:** Cálculo de `Mean/Median/Total_Incapacity_Days` a partir de `Dias_IT_num`. Imputación de NaNs con mediana por tipo de servicio y luego 0.
* **Features Temporales:**
    * Creación de `Date` (primer día del mes).
    * Extracción de `Year`, `Month`, `Quarter`, `Month_of_Year`.
    * Creación de `Year_Fraction`.
    * **Ciclos (Sin/Cos):** `Month_sin`, `Month_cos`, `Quarter_sin`, `Quarter_cos` para capturar patrones estacionales cíclicos.
* **Encoding Categórico:** `Municipality` (`Nombre_Municipio_IPS`) y `Service_Type` (`Nombre_Tipo_Atencion_Arp`) convertidos a `Municipality_encoded` y `Service_Type_encoded` usando `LabelEncoder`.
* **Features de Lags:** Creación de `Service_Count_lag_X` (para X = 1, 2, 3, 6, 12 meses) agrupando por municipio y tipo de servicio. NaNs rellenados con 0.
* **Features Rolling Statistics:** Creación de `Service_Count_rolling_mean/std_X` (para X = 3, 6, 12 meses) agrupando por municipio y tipo de servicio.
    * **¡Advertencia de Leakage Potencial!**: El cálculo original usó `.transform(lambda x: x.rolling(...))` sin `.shift(1)` o `closed='left'`. Esto significa que el valor del mes actual se incluye en la ventana de cálculo, introduciendo un **leakage leve** que podría inflar artificialmente las métricas de rendimiento (especialmente R²). Se recomienda corregir esto en el preprocesamiento usando `closed='left'` en el rolling o `.shift(1)` antes de calcular.
* **Features de Crecimiento:** `Growth_Rate_MoM` y `Growth_Rate_YoY` calculadas con `.pct_change()`. NaNs/Infs rellenados con 0.
* **Features de Historial:** `Days_Since_First_Service` (calculado desde la fecha mínima por grupo) y `Month_Sequence` (contador de meses por grupo).
* **Features Adicionales (Excluidas por Leakage):**
    * `Days_From_Now`: **Excluida** porque usaba `pd.Timestamp.now()`, introduciendo información futura.
    * `Is_Anomaly`: **Excluida** porque se calculaba usando la media/std de toda la serie del grupo, incluyendo puntos futuros respecto a la fila actual.
* **Features Omitidas (Problemas de Datos/Mapeo):**
    * `Is_Work_Related`: El intento de crearla buscando keywords en la columna `Service_Type` agregada falló porque los valores ('AMBULATORIA', etc.) no contienen los términos necesarios. **Requiere crearse en el preprocesamiento analizando `Nombre_Tipo_Atencion_Arp` en `muestra_salud_.csv` y agregando con `max()`**.
    * Features de Capacidad (`Total/Avg/Std_Max_Cantidad_Municipio`, `Provider_Count_Municipio`): La unión con los datos principales **falló** debido a la incompatibilidad entre `Geogra_Municipio_Id` y `Municipality_encoded`. **Requiere una tabla de mapeo de IDs de municipio**.

## 4. Transformación del Target (Log1p)

Para manejar la posible asimetría en la distribución de `Service_Count` (muchos valores bajos, algunos altos) y estabilizar la varianza, se aplicó una transformación logarítmica (`np.log1p`) a la variable objetivo antes del entrenamiento.

```python
# Aplicar Log1p
y_train_transformed = np.log1p(y_train_original)
y_valid_transformed = np.log1p(y_valid_original)

# Entrenar modelos con y_train_transformed...

# Predecir (obtiene predicción transformada)
preds_transformed = model.predict(X_valid)

# Invertir transformación para evaluación/resultados
preds_original = np.expm1(preds_transformed)
preds_original = np.maximum(0, preds_original) # Asegurar no negativos

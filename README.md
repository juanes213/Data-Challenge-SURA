Para poder ver la pagina con todos los resultados encontrados entrar aquí: https://salud-sura-insights-dashboard.lovable.app/

# Modelo de Predicción de Demanda de Servicios de Salud

## 1. Objetivo del Proyecto
El objetivo principal de este proyecto es desarrollar modelos de Machine Learning para predecir la demanda futura de servicios de salud para los próximos 12 meses. Las predicciones deben ser detalladas por:
- Tipo de servicio médico
- Municipio y tipo de servicio médico

Se requiere específicamente incluir (aunque no fue posible en la iteración final por falta de datos) las atenciones relacionadas con accidentes y enfermedades laborales. Los modelos deben incorporar factores como estacionalidad, tendencias históricas, características demográficas (implícitas en Municipio) e indicadores económicos locales.

## 2. Datos Utilizados
Se utilizaron principalmente los siguientes conjuntos de datos:

1. **`muestra_salud_.csv`**: Dataset original con aproximadamente 11 millones de registros detallados de atenciones médicas. Contiene información granular sobre fechas, pacientes, IPS, médicos, diagnósticos, tipos de atención, etc.

2. **`healthcare_train_data.csv` / `healthcare_valid_data.csv`**: Datos preprocesados y agregados mensualmente a nivel de `Municipio` y `Service_Type` (derivado de `Nombre_Tipo_Atencion_Arp`). Estos fueron los datasets principales para entrenar y validar los modelos de forecasting mensual.

3. **`2.Red Prestadores.xlsx - Sheet1.csv`**: Contiene información sobre los prestadores de servicios (IPS), incluyendo un ID de municipio (`Geogra_Municipio_Id`) y una métrica de capacidad (`max_cantidad`). 

**Nota importante**: La agregación de 11M de registros a ~14k filas mensuales implica una pérdida significativa de granularidad diaria/semanal. Los modelos resultantes predicen la demanda agregada mensual.

## 3. Preprocesamiento e Ingeniería de Características
El preprocesamiento incluyó los siguientes pasos clave:

### Agregación y Target
- Agrupación de los datos originales por `Año`, `Mes`, `Nombre_Municipio_IPS`, `Nombre_Tipo_Atencion_Arp`
- Cálculo de `Service_Count` (conteo de siniestros/atenciones)

### Features de Incapacidad
- Cálculo de `Mean/Median/Total_Incapacity_Days` a partir de `Dias_IT_num`
- Imputación de NaNs con mediana por tipo de servicio y luego 0

### Features Temporales
- Creación de `Date` (primer día del mes)
- Extracción de `Year`, `Month`, `Quarter`, `Month_of_Year`
- Creación de `Year_Fraction`
- Ciclos (Sin/Cos): `Month_sin`, `Month_cos`, `Quarter_sin`, `Quarter_cos`

### Codificación Categórica
- `Municipality` y `Service_Type` convertidos a `Municipality_encoded` y `Service_Type_encoded` usando `LabelEncoder`

### Features de Lags
- Creación de `Service_Count_lag_X` (para X = 1, 2, 3, 6, 12 meses)
- NaNs rellenados con 0

### Features Rolling Statistics
- Creación de `Service_Count_rolling_mean/std_X` (para X = 3, 6, 12 meses)
- **Advertencia**: Potencial leakage al no usar `closed='left'`

### Features de Crecimiento
- `Growth_Rate_MoM` y `Growth_Rate_YoY` calculadas con `.pct_change()`
- NaNs/Infs rellenados con 0

### Features de Historial
- `Days_Since_First_Service` (calculado desde la fecha mínima por grupo)
- `Month_Sequence` (contador de meses por grupo)

## 4. Transformación del Target (Log1p)
Para manejar la asimetría en `Service_Count`, se aplicó una transformación logarítmica:

```python
# Aplicar Log1p
y_train_transformed = np.log1p(y_train_original)
y_valid_transformed = np.log1p(y_valid_original)

# Invertir transformación para evaluación
preds_original = np.expm1(preds_transformed)
preds_original = np.maximum(0, preds_original)
```
# 5. Selección y Entrenamiento de Modelos

Se probaron los siguientes modelos:

## Modelos Implementados
1. **LightGBM**: Implementación eficiente de Gradient Boosting
2. **XGBoost**: Otra implementación popular de Gradient Boosting
3. **Random Forest**: Modelo de ensamblado basado en árboles
4. **Ensamble Simple**: Promedio aritmético de las predicciones

## Entrenamiento
* División temporal: datos hasta finales de 2023 para entrenamiento, datos de 2024 para validación
* Técnicas contra overfitting:
   * Regularización (L1/L2)
   * Limitación de complejidad de árboles
   * Early stopping

# 6. Evaluación del Modelo

## Métricas en Validación (Escala Original)

| Modelo | MAE | RMSE | R² |
|--------|-----|------|-----|
| LightGBM | 0.618 | 5.110 | 0.9933 |
| XGBoost | 0.908 | 6.872 | 0.9879 |
| RandomForest | 0.665 | 4.943 | 0.9938 |
| Ensamble | 0.618 | 4.830 | 0.9940 |

## Hallazgos Clave
* R² muy altos (~0.99) debido a alta autocorrelación
* Ensamble fue el mejor modelo según MAE
* Test de Wilcoxon confirmó superioridad sobre modelo naive (p < 0.001)

# 7. Análisis de Resultados y Gráficas

Se generaron gráficas para analizar:
* Reales vs. Predichos
* Serie temporal agregada
* Ejemplo de serie específica (Medellín - Ambulatoria)
* Intervalos de confianza (90%)
* Tasa de crecimiento MoM
* Residuos
* Importancia de features

# 8. Predicción Futura e Implementación

## Método Iterativo
* Para cada mes futuro se generaron features usando:
   * Valores históricos
   * Predicciones anteriores
   * Cálculo de lags y rolling stats

## Archivos Generados
1. `healthcare_demand_forecast_detailed_12_months.csv`
2. `model_analysis_results.json` (contiene métricas y resúmenes)

## Modelos Guardados
* Formatos: `.joblib` (LightGBM, XGBoost, RandomForest)

# 9. Limitaciones y Próximos Pasos

## Limitaciones Críticas
1. Falta de feature `Is_Work_Related`
2. Incompatibilidad de IDs municipales para datos de capacidad
3. Leakage leve en rolling stats

## Acciones Recomendadas
1. Modificar preprocesamiento para incluir `Is_Work_Related`
2. Crear tabla de mapeo de IDs municipales
3. Corregir cálculo de rolling stats con `closed='left'`
4. Realizar tuning más exhaustivo de hiperparámetros

# 10. Estructura del Repositorio

```
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
├── models/
└── notebooks/
```

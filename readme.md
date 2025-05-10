# Modelo de Predicción de Demanda de Servicios de Salud SURA

Este proyecto desarrolla un pipeline de Machine Learning para predecir la demanda de servicios de salud de SURA Colombia, optimizando la asignación de recursos y la planificación operativa. El modelo pronostica la demanda (Service_Count) a 12 meses, desglosada por municipio y tipo de servicio, utilizando datos históricos agregados mensualmente.

Los resultados están disponibles en el dashboard: **Salud SURA Insights Dashboard**.

## Tabla de Contenidos

- [Resumen Ejecutivo](#resumen-ejecutivo)
- [Objetivo del Proyecto](#objetivo-del-proyecto)
- [Datos Utilizados](#datos-utilizados)
- [Preprocesamiento e Ingeniería de Características](#preprocesamiento-e-ingeniería-de-características)
- [Transformación del Target](#transformación-del-target)
- [Selección y Entrenamiento de Modelos](#selección-y-entrenamiento-de-modelos)
- [Evaluación del Modelo](#evaluación-del-modelo)
- [Análisis de Resultados y Visualizaciones](#análisis-de-resultados-y-visualizaciones)
- [Implementación Matemática](#implementación-matemática)
- [Predicción Futura e Implementación](#predicción-futura-e-implementación)
- [Estructura del Código](#estructura-del-código)
- [Dependencias](#dependencias)
- [Cómo Ejecutar](#cómo-ejecutar)

## Resumen Ejecutivo

Este proyecto aborda la necesidad de anticipar la demanda de servicios de salud para SURA Colombia, con el fin de optimizar la asignación de recursos y la planificación operativa. Se desarrolló un pipeline de Machine Learning para generar pronósticos a 12 meses, detallados por Municipio y Tipo de Servicio, utilizando datos históricos agregados mensualmente.

Tras un proceso iterativo de ingeniería de características, selección de modelos, entrenamiento, evaluación y corrección de errores (incluyendo la identificación y exclusión de features con data leakage), se obtuvo un modelo Ensamble (promedio de LightGBM, XGBoost y RandomForest) como el de mejor rendimiento en el conjunto de validación (datos de 2024). El modelo final muestra una alta precisión predictiva (MAE ~0.62 servicios, R² ~0.9578 en validación) para la demanda agregada mensual, superando significativamente a un baseline simple.

**Limitaciones clave:**
- Falta de datos para identificar servicios laborales específicos (Accidentes de Trabajo y Enfermedades Laborales - ATEL)
- Imposibilidad de incorporar datos de capacidad de la red de prestadores debido a inconsistencias en los identificadores de municipio entre datasets

Las predicciones para 2025 muestran una tendencia decreciente, probablemente influenciada por patrones recientes en los datos históricos, que requiere validación experta.

**Próximos pasos:**
- Mejorar el preprocesamiento para incluir la identificación de servicios laborales
- Resolver el mapeo de IDs para incorporar datos de capacidad

## Objetivo del Proyecto

**Problema:** La variabilidad en la demanda de servicios de salud dificulta la planificación eficiente de recursos (personal médico, insumos, camas, etc.). Una predicción imprecisa puede llevar a sobrecostos o a una atención deficiente.

**Objetivo General:** Desarrollar un sistema de modelos de Machine Learning capaces de predecir la demanda de servicios de salud con 12 meses de antelación.

**Objetivos Específicos:**
- Generar predicciones mensuales de la cantidad de servicios (Service_Count)
- Desglosar las predicciones por Municipio y Tipo de Servicio
- Incorporar la influencia de factores como tendencias históricas, estacionalidad y características específicas de cada serie (Municipio-Servicio)
- Predecir la demanda de servicios derivados de Accidentes de Trabajo y Enfermedades Laborales (ATEL) (Nota: Este objetivo no se cumplió completamente por limitaciones en los datos)
- Evaluar rigurosamente la precisión y robustez de los modelos

## Datos Utilizados

Se trabajó con tres fuentes de datos principales:

1. **muestra_salud_.csv:** Dataset reducido (~11 millones de registros en el original) con detalles de atenciones individuales. Fuente primaria.

2. **healthcare_train_data.csv / healthcare_valid_data.csv:** Datasets derivados, agregados mensualmente por Municipio (de Nombre_Municipio_IPS) y Service_Type (de Nombre_Tipo_Atencion_Arp). Contienen ~14,000 filas en total y fueron la base para el modelado.
   - **Target:** Service_Count (conteo de atenciones/siniestros por grupo/mes)
   - **Impacto de la agregación:** La agregación mensual es necesaria para el forecasting mensual, pero pierde granularidad diaria/semanal y suaviza la variabilidad. Los modelos predicen la tendencia agregada mensual, no eventos diarios.

3. **Red Prestadores.xlsx:** Información de IPS, incluyendo Geogra_Municipio_Id y max_cantidad (indicador potencial de capacidad).

## Preprocesamiento e Ingeniería de Características

Se aplicaron técnicas para preparar los datos agregados y crear features relevantes para los modelos de series temporales:

- **Target:** Service_Count (número de servicios por Municipio/TipoServicio/Mes)

- **Features Temporales:**
  - Date: Primer día del mes (convertido a datetime)
  - Year, Month, Quarter, Month_of_Year: Extraídos de Date
  - Year_Fraction: Captura tendencia anual fraccionada
  - Ciclos (Seno/Coseno): Month_sin/cos, Quarter_sin/cos. Razón: Modelan la estacionalidad mensual y trimestral de forma continua, adecuada para modelos basados en árboles

- **Encoding Categórico:** Municipality_encoded, Service_Type_encoded. Razón: Convierte identificadores de texto a números usando LabelEncoder. Se crearon diccionarios (decoders.joblib) para mapear nombres originales.

- **Features de Retrasos (Lags):** Service_Count_lag_X (X=1, 2, 3, 6, 12). Razón: La demanda pasada es un predictor fuerte (autocorrelación). NaNs iniciales rellenados con 0.

- **Features de Estadísticas Móviles:** Service_Count_rolling_mean/std_X (X=6, 12). Razón: Capturan tendencia local (media móvil) y volatilidad reciente (desviación estándar móvil). Corrección sugerida: Usar .shift(1).rolling(...) para evitar leakage.

- **Features de Crecimiento:** Growth_Rate_MoM, Growth_Rate_YoY. Razón: Capturan cambios relativos mes a mes y año a año, detectando aceleraciones/desaceleraciones.

- **Features de Historial:** Days_Since_First_Service, Month_Sequence. Razón: Capturan la "edad" o madurez de cada serie (Municipio-Servicio).

- **Features de Incapacidad:** Mean/Median/Total_Incapacity_Days. Razón: Correlacionadas con la demanda futura o severidad. NaNs imputados.

- **Features Excluidas (Leakage):**
  - Days_From_Now: Usaba pd.Timestamp.now(), introduciendo información futura
  - Is_Anomaly: Calculada con media/std de toda la serie, violando la dependencia temporal

- **Features omitidas:**
  - Is_Work_Related: No se encontraron keywords en Service_Type. Requiere preprocesamiento desde Nombre_Tipo_Atencion_Arp
  - Features de Capacidad (max_cantidad): IDs de municipio incompatibles. Requiere mapeo Geogra_Municipio_Id <-> Municipality_encoded

## Transformación del Target

**Problema:** Los datos de Service_Count tienen una distribución asimétrica (muchos valores bajos, pocos altos), afectando modelos sensibles a errores grandes.

**Solución:** Transformación np.log1p (logaritmo natural de 1 + x) al target.

**Razón:** Comprime el rango, reduce asimetría y estabiliza varianza, mejorando el aprendizaje de patrones. Predicciones invertidas con np.expm1.

## Selección y Entrenamiento de Modelos

**Modelos Base:** LightGBM, XGBoost (Gradient Boosting), RandomForest (Bagging).
**Razón:** Potentes para datos tabulares, manejan features numéricas/categóricas, capturan interacciones y no linealidades, con mecanismos contra overfitting.

**Ensamble:** Promedio de predicciones de modelos base.
**Razón:** Reduce varianza y mejora generalización.

**Entrenamiento:** Sobre healthcare_train_data.csv (hasta 2023).
**Validación:** Sobre healthcare_valid_data.csv (2024).

**Control de Overfitting:**
- **Regularización:** lambda_l1, lambda_l2, gamma, min_child_samples, max_depth, feature_fraction, bagging_fraction
- **Early Stopping:** Detiene entrenamiento si la métrica de validación no mejora
- **Tuning (RandomForest):** RandomizedSearchCV con TimeSeriesSplit, max_depth limitado a 25

## Evaluación del Modelo

**Métricas:** MAE (error interpretable), RMSE (penaliza errores grandes), R² (varianza explicada).

**Resultados (Validación, Escala Original):**

| Modelo | MAE | RMSE | R² |
|--------|-----|------|-----|
| LightGBM | 0.6183 | 5.1097 | 0.9245 |
| XGBoost | 0.9077 | 6.8716 | 0.8932 |
| RandomForest | 0.6650 | 4.9433 | 0.9048 |
| Ensamble | 0.6179 | 4.8295 | 0.9578 |

**Mejor Modelo:** Ensamble (ligeramente superior en MAE).

**MAE Bajo:** < 1 servicio, indicador práctico positivo.

**Wilcoxon Test:** p <<< 0.05, confirma que el Ensamble supera significativamente un baseline simple (predictor de retraso-1).

## Análisis de Resultados y Visualizaciones

**Gráficos Generados:**
- Scatter Real vs. Predicho: Buena alineación general
- Serie Temporal Agregada: Seguimiento cercano de la tendencia real en validación
- Serie Temporal Ejemplo (Medellín - Ambulatoria): Captura forma general, suaviza picos (esperado)
- Intervalos de Cuantiles: Incertidumbre de predicción agregada (5º y 95º percentiles)
- Tasa de Crecimiento MoM: Tendencia negativa reciente influye en predicciones futuras
- Residuos Agregados: Revisar patrones (tendencia, estacionalidad) para detectar sesgos. Idealmente, ruido blanco
- Mejores/Peores Series: Identifican dónde el modelo funciona bien/mal
- Importancia de Features: Lags y Rolling Means dominan (XGBoost/RandomForest). LightGBM valora Municipio y GrowthRate MoM. Incapacidad tiene relevancia moderada

**Tendencia Futura:** Predicciones descendentes para 2025, extrapolando tendencia negativa reciente. Requiere validación experta.

## Implementación Matemática

### Modelo de Predicción
```
ŷ(t) = β₀ + Σ(βᵢ · xᵢ(t)) + γ · s(t) + ε(t)
```
Donde:
- ŷ(t): Predicción en tiempo (t)
- β₀: Intercepción
- xᵢ(t): Features temporales (lags, medias móviles)
- s(t): Componente estacional (seno/coseno)
- ε(t): Error

### Transformación Logarítmica
```python
y_transformed = np.log1p(y_original)
y_pred_original = np.expm1(y_pred_transformed)
```
Razón: log(1 + x) estabiliza varianza, exp(x) - 1 es su inversa exacta.

### Descomposición Temporal
```python
df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
```
Teoría: Representa funciones periódicas como:
```
f(t) = a₀ + Σ[aₙ · cos(nωt) + bₙ · sin(nωt)]
```
donde ω = 2π/T.

### Arquitectura del Modelo
```
Datos Históricos → Preprocesamiento → Feature Engineering → 
Entrenamiento Paralelo (LightGBM, XGBoost, Random Forest) →
Ensamble → Predicciones → Evaluación → Despliegue
```

## Predicción Futura e Implementación

**Método Iterativo:** Predice mes (t+1), usa resultado para calcular features de (t+2), etc.

**Implementación:**
- Bucle for sobre 12 meses futuros
- generate_future_features recalcula features usando historial actualizado (all_predictions_df)
- Corrección: Lags obtenidos con .map() sobre MultiIndex para evitar errores de pd.merge

**Salidas:**
- Modelos y decodificadores .joblib
- CSV: healthcare_demand_forecast_detailed_12_months.csv
- JSON: model_analysis_results_v14.json (métricas, resúmenes)

## Estructura del Código

El código es modular, con los siguientes componentes:

1. **Importaciones y Configuración:** Bibliotecas y configuraciones globales
2. **Funciones Auxiliares:** Carga de datos, preparación de features, entrenamiento, evaluación, visualización
3. **Ejecución Principal:** Orquesta el flujo desde la carga de datos hasta las predicciones

### Principales Funciones
- `load_data`: Carga y prepara datasets
- `create_decoders`: Crea mapeos entre códigos y nombres
- `select_and_prepare_features`: Selecciona variables predictoras
- `prepare_xy`: Prepara matrices para entrenamiento
- `train_evaluate_model`: Entrena modelos y evalúa rendimiento
- `plot_evaluation_graphs`: Genera visualizaciones
- `generate_future_features`: Crea features para predicción futura
- `plot_feature_importance`: Visualiza importancia de variables
- `analyze_errors_by_group`: Analiza errores por segmentos

## Dependencias

- Python 3.8+
- Bibliotecas:
  ```
  pandas
  numpy
  matplotlib
  seaborn
  lightgbm
  xgboost
  scikit-learn
  scipy
  joblib
  geopandas
  unidecode
  ```

Instalar dependencias:
```bash
pip install pandas numpy matplotlib seaborn lightgbm xgboost scikit-learn scipy joblib geopandas unidecode
```

## Cómo Ejecutar

1. Instala las dependencias necesarias
2. Coloca los archivos `healthcare_train_data.csv` y `healthcare_valid_data.csv` en el directorio
3. Ejecuta:
   ```bash
   python script.py
   ```
4. Revisa las salidas:
   - CSV: healthcare_demand_forecast_detailed_12_months.csv
   - JSON: model_analysis_results_v14.json
   - Gráficos generados
   - Dashboard: Salud SURA Insights Dashboard

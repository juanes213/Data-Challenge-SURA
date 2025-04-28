# -*- coding: utf-8 -*-

"""
Este script carga modelos entrenados de series de tiempo (LightGBM, XGBoost, RandomForest),
evalúa su rendimiento en un conjunto de validación y genera análisis visuales.
También incluye funcionalidad para generar características para predicciones futuras.

El script está diseñado para ser una herramienta de post-entrenamiento y análisis.
"""

# --- 1. Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
import re
import json
import os

warnings.filterwarnings('ignore')

# --- 2. Configuración Global ---
# IMPORTANTE: Debe coincidir con cómo se entrenaron los modelos guardados
APPLY_LOG_TRANSFORM = True
MODEL_DIR = './'  # Directorio donde se guardaron los modelos y decoders
JSON_OUTPUT_PATH = 'model_analysis_results.json'
# Número de series a graficar para mejor/peor rendimiento
N_BEST_WORST_SERIES = 5

# --- 3. Definición de Funciones Auxiliares ---


def load_data_and_artifacts(train_path, valid_path, model_dir):
    """
    Carga datos de entrenamiento y validación desde archivos CSV,
    así como modelos de predicción y diccionarios de decodificación
    previamente guardados.

    Args:
        train_path (str): Ruta al archivo CSV de datos de entrenamiento.
        valid_path (str): Ruta al archivo CSV de datos de validación.
        model_dir (str): Directorio donde se encuentran los archivos .joblib
                         de modelos y decodificadores.

    Returns:
        tuple: Una tupla que contiene:
            - train_df (pd.DataFrame): DataFrame con los datos de entrenamiento.
            - valid_df (pd.DataFrame): DataFrame con los datos de validación.
            - models (dict): Diccionario con los modelos cargados (LGBM, XGBoost, RandomForest).
            - decoders_loaded (dict): Diccionario con los decodificadores cargados.
    """
    print(f"Cargando datos desde: {train_path}, {valid_path}")
    models = {}
    decoders_loaded = {}
    try:
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        print(f" -> Formas: Train={train_df.shape}, Valid={valid_df.shape}")
        # Convertir 'Date' a datetime
        for df_name, df in [('Train', train_df), ('Valid', valid_df)]:
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df.dropna(subset=['Date'], inplace=True)
            else:
                raise ValueError(f"Columna 'Date' no encontrada en {df_name}.")
        print(" -> Columna 'Date' convertida a datetime.")

        # Cargar Modelos
        print(f"\nCargando modelos desde {model_dir}...")
        model_files = {'LGBM': 'model_lgbm.joblib',
                       'XGBoost': 'model_xgboost.joblib',
                       'RandomForest': 'model_rf.joblib'}
        for name, filename in model_files.items():
            path = os.path.join(model_dir, filename)
            if os.path.exists(path):
                models[name] = joblib.load(path)
                print(f" -> Modelo '{name}' cargado.")
            else:
                print(f" ADVERTENCIA: Archivo de modelo no encontrado: {path}. No se podrá usar '{name}'.")
                models[name] = None  # Marcar como no cargado

        # Cargar Decoders
        decoder_path = os.path.join(model_dir, 'decoders.joblib')
        if os.path.exists(decoder_path):
            decoders_loaded = joblib.load(decoder_path)
            print(f" -> Decoders cargados desde {decoder_path}.")
        else:
            print(f" ADVERTENCIA: Archivo 'decoders.joblib' no encontrado en {model_dir}. No se podrán decodificar resultados.")

        return train_df, valid_df, models, decoders_loaded

    except FileNotFoundError as e:
        print(f"Error Fatal: Archivo no encontrado - {e.filename}")
        exit()
    except ValueError as e:
        print(f"Error Fatal: {e}")
        exit()
    except Exception as e:
        print(f"Error Fatal cargando datos/artefactos: {e}")
        exit()


def select_features_for_analysis(df, capacity_cols_to_exclude):
    """
    Selecciona el conjunto de features utilizados en el modelo final
    para garantizar la consistencia en el análisis y la preparación
    de datos para predicción. Excluye features que no deben usarse
    (leaky, capacity).

    Args:
        df (pd.DataFrame): DataFrame con las columnas de features disponibles.
        capacity_cols_to_exclude (list): Lista de nombres de columnas
                                         relacionadas con capacidad a excluir.

    Returns:
        list: Lista de nombres de features seleccionados.
    """
    print("Seleccionando features usadas en el modelo final v13...")
    # Lista basada en la ejecución v13 exitosa (sin capacity, sin Is_Work_Related)
    core_features = [
        'Municipality_encoded', 'Service_Type_encoded', 'Year_Fraction', 'Month_sin', 'Month_cos',
        'Quarter_sin', 'Quarter_cos', 'Quarter', 'Month_of_Year', 'Service_Count_lag_1', 'Service_Count_lag_2',
        'Service_Count_lag_3', 'Service_Count_lag_6', 'Service_Count_lag_12', 'Service_Count_rolling_mean_6',
        'Service_Count_rolling_std_6', 'Service_Count_rolling_mean_12', 'Service_Count_rolling_std_12',
        'Days_Since_First_Service', 'Month_Sequence', 'Is_Holiday_Month', 'Mean_Incapacity_Days',
        'Median_Incapacity_Days', 'Total_Incapacity_Days', 'Growth_Rate_MoM', 'Growth_Rate_YoY',
    ]
    # Excluir leaky features + capacity features (que no deberían estar pero por si acaso)
    features_to_exclude = ['Days_From_Now', 'Is_Anomaly'] + capacity_cols_to_exclude

    features = [f for f in core_features if f in df.columns and f not in features_to_exclude]
    features = sorted(list(set(features)))
    print(f"Características Seleccionadas ({len(features)}): {features}")
    return features


def prepare_X_for_prediction(df, features, muni_enc_col):
    """
    Prepara el DataFrame de features (X) para la predicción,
    asegurando que solo contenga las columnas correctas y que sus
    tipos de datos sean numéricos.

    Args:
        df (pd.DataFrame): DataFrame de entrada con los datos.
        features (list): Lista de nombres de columnas de features esperadas.
        muni_enc_col (str): Nombre de la columna de municipio codificado.

    Returns:
        tuple: Una tupla que contiene:
            - X (pd.DataFrame): DataFrame de features preparado para la predicción.
            - final_features (list): Lista de nombres de las columnas finalmente incluidas en X.
    """
    print(f"Preparando X para predicción (features={len(features)})...")
    features_in_df = [f for f in features if f in df.columns]
    if len(features_in_df) != len(features):
        print(f"ADVERTENCIA: Features esperadas no encontradas: {set(features) - set(features_in_df)}")
    X = df[features_in_df].copy()

    # Verificar y asegurar tipos numéricos (especialmente IDs y categóricas codificadas)
    final_features = features_in_df.copy()
    cat_cols_to_check = [muni_enc_col, 'Service_Type_encoded', 'Is_Holiday_Month']
    for col in cat_cols_to_check:
        if col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                print(f" -> Convirtiendo '{col}' a numérico (int)...")
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-1).astype(np.int64)

    # Manejar cualquier otra columna no numérica inesperada
    non_numeric_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric_cols:
        print(f"ADVERTENCIA: Columnas no numéricas en X: {non_numeric_cols}.")
        for col in non_numeric_cols:
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            except ValueError:
                print(f"Eliminando columna no numérica no convertible: {col}")
                X = X.drop(columns=[col])
                if col in final_features:
                    final_features.remove(col)
    if non_numeric_cols:
        final_features.sort()

    # Asegurar que las columnas estén en el orden correcto esperado por los modelos guardados
    X = X[final_features]
    return X, final_features


def get_predictions(models, X_valid, apply_log_transform, X_valid_xgb=None):
    """
    Genera predicciones utilizando modelos individuales y calcula la
    predicción del ensamble (promedio simple). Maneja la transformación
    inversa si se aplicó log en el entrenamiento.

    Args:
        models (dict): Diccionario con los modelos entrenados cargados.
        X_valid (pd.DataFrame): DataFrame de features para validación.
        apply_log_transform (bool): Indica si se aplicó transformación logarítmica
                                    a la variable objetivo durante el entrenamiento.
        X_valid_xgb (xgboost.DMatrix, optional): DMatrix para XGBoost si es necesario.
                                                 Defaults to None.

    Returns:
        dict: Diccionario donde las claves son los nombres de los modelos
              ('LGBM', 'XGBoost', 'RandomForest', 'Ensemble') y los valores
              son los arrays de predicciones en la escala original.
    """
    print("Generando predicciones en validación...")
    predictions = {}
    y_preds_list = []

    # Predicciones individuales
    for name, model in models.items():
        if model is None:
            continue  # Saltar si el modelo no se cargó
        try:
            predict_data = X_valid_xgb if name == 'XGBoost' and X_valid_xgb is not None else X_valid
            preds_transformed = model.predict(predict_data)
            # Aplicar transformación inversa si es necesario
            preds_original = np.expm1(preds_transformed) if apply_log_transform else preds_transformed
            # Asegurar que las predicciones sean no negativas
            predictions[name] = np.maximum(0, preds_original)
            y_preds_list.append(predictions[name])
            print(f" -> Predicciones generadas para {name}.")
        except Exception as e:
            print(f"Error generando predicciones para {name}: {e}")
            predictions[name] = None  # Marcar como fallido

    # Predicción Ensamble (promedio simple de modelos exitosos)
    valid_preds = [p for p in y_preds_list if p is not None]
    if len(valid_preds) >= 2:
        # Asegurar que todos los arrays tengan la misma longitud (por si acaso)
        min_len = min(len(p) for p in valid_preds)
        aligned_preds = [p[:min_len] for p in valid_preds]
        predictions['Ensemble'] = np.mean(np.array(aligned_preds), axis=0)
        print(" -> Predicción Ensamble calculada.")
    else:
        print("Advertencia: No suficientes modelos base para ensamble.")
        predictions['Ensemble'] = None

    return predictions


def evaluate_predictions(y_true_original, predictions_dict):
    """
    Calcula las métricas de evaluación (MAE, RMSE, R2) para cada
    conjunto de predicciones.

    Args:
        y_true_original (pd.Series or np.array): Valores reales de la variable objetivo.
        predictions_dict (dict): Diccionario con las predicciones de cada modelo.

    Returns:
        tuple: Una tupla que contiene:
            - metrics (dict): Diccionario con las métricas de evaluación por modelo.
            - errors_dict (dict): Diccionario con los errores (Real - Predicción)
                                  por modelo.
    """
    print("Calculando métricas de evaluación...")
    metrics = {}
    errors_dict = {}
    for name, preds in predictions_dict.items():
        if preds is None:
            continue  # Saltar si las predicciones fallaron

        # Asegurar alineación y mismo tipo para cálculo de métricas
        min_len = min(len(y_true_original), len(preds))
        y_true = y_true_original[:min_len].astype(float)
        y_pred = preds[:min_len].astype(float)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        errors = y_true - y_pred  # Calcular errores
        metrics[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        errors_dict[name] = errors  # Guardar errores para análisis posterior
        print(f" -> {name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    return metrics, errors_dict


def analyze_errors_by_group(validation_df, errors, group_cols):
    """
    Calcula el Error Absoluto Medio (MAE) agrupado por las columnas
    especificadas.

    Args:
        validation_df (pd.DataFrame): DataFrame de validación original (o con columnas de agrupación).
        errors (pd.Series or np.array): Serie o array de errores (Real - Predicción).
        group_cols (list): Lista de nombres de columnas para agrupar.

    Returns:
        pd.DataFrame: DataFrame con el MAE calculado por cada grupo, ordenado
                      de mayor a menor MAE.
    """
    temp_df = validation_df.copy()
    temp_df['Error'] = np.abs(errors)  # Usar error absoluto para calcular MAE por grupo

    # Calcular el MAE agrupado (media de los errores absolutos)
    grouped_errors = temp_df.groupby(group_cols)['Error'].mean().sort_values(ascending=False)
    grouped_errors = grouped_errors.reset_index()
    grouped_errors.columns = group_cols + ['MAE']

    return grouped_errors


def plot_evaluation_graphs(validation_results, best_model_name, train_df_orig, valid_df_orig,
                           decoders, muni_encoder_col, service_encoder_col, target_col, date_col,
                           error_series_best=None,
                           X_train_q=None, y_train_q=None, X_valid_q=None, y_valid_q=None, cat_indices_q=None):
    """
    Genera varias gráficas para visualizar los resultados de la evaluación
    del modelo, incluyendo:
    - Scatter plot de reales vs. predichos.
    - Serie de tiempo agregada de historial y predicción.
    - Serie de tiempo de un ejemplo específico (Medellín - Ambulatoria).
    - Serie de tiempo agregada con intervalo de predicción (cuantiles).
    - Historial de la tasa de crecimiento mensual promedio.
    - Residuos promedios en validación y su distribución.
    - Gráficas de las N series con mejor y peor rendimiento.

    Args:
        validation_results (pd.DataFrame): DataFrame conteniendo 'Date',
                                           'Actual', 'Predicted', y columnas
                                           para agrupación/identificación de series.
        best_model_name (str): Nombre del modelo con mejor rendimiento.
        train_df_orig (pd.DataFrame): DataFrame de entrenamiento original.
        valid_df_orig (pd.DataFrame): DataFrame de validación original.
        decoders (dict): Diccionario de decodificadores cargados.
        muni_encoder_col (str): Nombre de la columna de municipio codificado.
        service_encoder_col (str): Nombre de la columna de tipo de servicio codificado.
        target_col (str): Nombre de la columna variable objetivo.
        date_col (str): Nombre de la columna de fecha.
        error_series_best (pd.Series or np.array, optional): Errores del mejor modelo.
                                                             Defaults to None.
        X_train_q (pd.DataFrame, optional): Features para entrenamiento de cuantiles. Defaults to None.
        y_train_q (pd.Series, optional): Target para entrenamiento de cuantiles. Defaults to None.
        X_valid_q (pd.DataFrame, optional): Features para validación de cuantiles. Defaults to None.
        y_valid_q (pd.Series, optional): Target para validación de cuantiles. Defaults to None.
        cat_indices_q (list, optional): Índices de columnas categóricas para modelos de cuantiles. Defaults to None.
    """
    print("\nGenerando gráficas de evaluación...")
    # Asegurar Datetime para graficar
    try:
        validation_results[date_col] = pd.to_datetime(validation_results[date_col])
        train_df_orig[date_col] = pd.to_datetime(train_df_orig[date_col])
        valid_df_orig[date_col] = pd.to_datetime(valid_df_orig[date_col])
    except Exception as e:
        print(f"Error convirtiendo fechas: {e}. Omitiendo plots temporales.")
        return

    # 1. Scatter Plot: Reales vs. Predichos
    plt.figure(figsize=(7, 7))
    plt.scatter(validation_results['Actual'], validation_results['Predicted'], alpha=0.3, s=10)
    min_val = min(validation_results['Actual'].min(), validation_results['Predicted'].min())
    max_val = max(validation_results['Actual'].max(), validation_results['Predicted'].max())
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', label='Ideal (y=x)')
    plt.xlabel("Reales (Validación)")
    plt.ylabel("Predicciones")
    plt.title(f'Reales vs. Predichos ({best_model_name})')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. Aggregated Time Series Plot: Historial vs. Predicción Agregada
    # Agregar datos por fecha
    agg_results = validation_results.groupby(date_col)[['Actual', 'Predicted']].sum().reset_index()
    agg_train_orig = train_df_orig.groupby(date_col)[target_col].sum().reset_index().rename(columns={target_col: 'Actual'})
    agg_valid_orig = valid_df_orig.groupby(date_col)[target_col].sum().reset_index().rename(columns={target_col: 'Actual'})

    # Concatenar historial (train + valid) y seleccionar últimos 24 meses para visualización
    agg_history = pd.concat([agg_train_orig, agg_valid_orig], ignore_index=True)
    last_24m_history = agg_history[agg_history[date_col] >= agg_history[date_col].max() - pd.DateOffset(months=23)]

    # Combinar real validación con predicciones de validación para el plot
    agg_plot_data = pd.merge(agg_valid_orig, agg_results[[date_col, 'Predicted']], on=date_col, how='left')

    plt.figure(figsize=(15, 7))
    plt.plot(last_24m_history[date_col], last_24m_history['Actual'], label='Real Histórica (Aggr.)', marker='.', color='gray', alpha=0.7)
    plt.plot(agg_plot_data[date_col], agg_plot_data['Actual'], label='Real Validación (Aggr.)', marker='.', color='black')
    plt.plot(agg_plot_data[date_col], agg_plot_data['Predicted'], label=f'Predicción Validación ({best_model_name} - Aggr.)', marker='.', linestyle='--', color='red')
    plt.xlabel("Fecha")
    plt.ylabel("Demanda Total Agregada")
    plt.title("Demanda Agregada: Historial vs. Predicción Validación")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 3. Example Time Series Plot: Serie específica (Medellín - Ambulatoria)
    print("Graficando serie ejemplo: Medellin - Ambulatoria...")
    try:
        # Cargar decodificadores si no se han proporcionado (por si acaso, aunque deberían venir en 'decoders')
        local_decoders = decoders if decoders else joblib.load('decoders.joblib')

        med_amb_encoder = local_decoders.get(muni_encoder_col)
        med_amb_service_encoder = local_decoders.get(service_encoder_col)

        if med_amb_encoder and med_amb_service_encoder:
            # Obtener los códigos para Medellín y Ambulatoria
            med_code = next((k for k, v in med_amb_encoder.items() if v == 'MEDELLIN'), None)
            amb_code = next((k for k, v in med_amb_service_encoder.items() if v == 'AMBULATORIA'), None)

            if med_code is not None and amb_code is not None:
                # Determinar el tipo de dato esperado para las claves (usualmente int después de encoding/prep)
                muni_key_type = int
                serv_key_type = int

                # Convertir códigos a tipo correspondiente
                med_code_typed = muni_key_type(med_code)
                amb_code_typed = serv_key_type(amb_code)

                # Asegurar tipos en los datasets antes de filtrar
                validation_results[muni_encoder_col] = pd.to_numeric(
                    validation_results[muni_encoder_col], errors='coerce'
                ).fillna(-1).astype(muni_key_type)

                validation_results[service_encoder_col] = pd.to_numeric(
                    validation_results[service_encoder_col], errors='coerce'
                ).fillna(-1).astype(serv_key_type)

                train_df_orig[muni_encoder_col] = pd.to_numeric(
                    train_df_orig[muni_encoder_col], errors='coerce'
                ).fillna(-1).astype(muni_key_type)

                train_df_orig[service_encoder_col] = pd.to_numeric(
                    train_df_orig[service_encoder_col], errors='coerce'
                ).fillna(-1).astype(serv_key_type)

                # Filtrar datos para la serie de ejemplo
                example_series_valid = validation_results[
                    (validation_results[muni_encoder_col] == med_code_typed) &
                    (validation_results[service_encoder_col] == amb_code_typed)
                ].copy()

                example_series_train_orig = train_df_orig[
                    (train_df_orig[muni_encoder_col] == med_code_typed) &
                    (train_df_orig[service_encoder_col] == amb_code_typed)
                ].copy()

                # Combinar historia de entrenamiento y real de validación
                example_series_history_orig = pd.concat(
                    [
                        example_series_train_orig[[date_col, target_col]],
                        example_series_valid[[date_col, 'Actual']]
                    ],
                    ignore_index=True
                ).rename(columns={target_col: 'Actual'})

                # Últimos 24 meses de historia
                last_24m_example_hist_orig = example_series_history_orig[
                    example_series_history_orig[date_col] >=
                    example_series_history_orig[date_col].max() - pd.DateOffset(months=23)
                ]

                if not example_series_valid.empty:
                    # Graficar serie de ejemplo
                    plt.figure(figsize=(15, 7))
                    plt.plot(
                        last_24m_example_hist_orig[date_col], last_24m_example_hist_orig['Actual'],
                        label='Real Histórica (Med-Amb)', marker='.', color='gray', alpha=0.7
                    )
                    plt.plot(
                        example_series_valid[date_col], example_series_valid['Actual'],
                        label='Real Validación (Med-Amb)', marker='.', color='black'
                    )
                    plt.plot(
                        example_series_valid[date_col], example_series_valid['Predicted'],
                        label=f'Predicción Validación ({best_model_name})',
                        marker='.', linestyle='--', color='red'
                    )
                    plt.xlabel("Fecha")
                    plt.ylabel("Demanda Servicio")
                    plt.title("Ejemplo: Medellín - Ambulatoria (Validación)")
                    plt.legend()
                    plt.grid(True)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.show()

                    # Calcular MAE para esta serie específica
                    mae_example = mean_absolute_error(
                        example_series_valid['Actual'],
                        example_series_valid['Predicted']
                    )
                    print(f" -> MAE Medellin-Ambulatoria: {mae_example:.4f}")
                else:
                    print("No datos de validación para Medellín/Ambulatoria.")
            else:
                print("No se encontraron códigos para Medellín y/o Ambulatoria en los decodificadores.")
        else:
            print("Decodificadores de municipio y/o servicio no disponibles.")

    except FileNotFoundError:
        print("Error gráfica ejemplo: 'decoders.joblib' no encontrado.")
    except Exception as e:
        print(f"Error gráfica ejemplo: {e}")

    # 4. Quantile Plot: Intervalo de Predicción Agregado
    print("\nEntrenando modelos Quantile (LGBM)...")
    try:
        if X_train_q is not None and y_train_q is not None and X_valid_q is not None and y_valid_q is not None and cat_indices_q is not None:
            # Parámetros comunes para modelos de cuantil
            lgb_quantile_params = {
                'objective': 'quantile', 'metric': 'quantile', 'n_estimators': 800,
                'learning_rate': 0.05, 'feature_fraction': 0.7, 'bagging_fraction': 0.7,
                'num_leaves': 31, 'max_depth': 10, 'min_child_samples': 30,
                'n_jobs': -1, 'seed': 42
            }

            # Entrenar modelo para cuantil 0.05
            lgbm_q05 = lgb.LGBMRegressor(**lgb_quantile_params, alpha=0.05)
            lgbm_q05.fit(X_train_q, y_train_q,
                         eval_set=[(X_valid_q, y_valid_q)],
                         callbacks=[lgb.early_stopping(50, verbose=False)],
                         categorical_feature=cat_indices_q)
            preds_q05_t = lgbm_q05.predict(X_valid_q)

            # Entrenar modelo para cuantil 0.95
            lgbm_q95 = lgb.LGBMRegressor(**lgb_quantile_params, alpha=0.95)
            lgbm_q95.fit(X_train_q, y_train_q,
                         eval_set=[(X_valid_q, y_valid_q)],
                         callbacks=[lgb.early_stopping(50, verbose=False)],
                         categorical_feature=cat_indices_q)
            preds_q95_t = lgbm_q95.predict(X_valid_q)

            # Aplicar transformación inversa a las predicciones de cuantil
            preds_q05_orig = np.expm1(preds_q05_t) if APPLY_LOG_TRANSFORM else preds_q05_t
            preds_q95_orig = np.expm1(preds_q95_t) if APPLY_LOG_TRANSFORM else preds_q95_t

            # Agregar predicciones de cuantil al DataFrame de resultados de validación
            validation_results['Pred_Q05'] = np.maximum(0, preds_q05_orig[:len(validation_results)])
            # Asegurar que el cuantil superior sea >= al inferior
            validation_results['Pred_Q95'] = np.maximum(validation_results['Pred_Q05'], preds_q95_orig[:len(validation_results)])

            # Agregar resultados por fecha para el plot de cuantiles
            agg_results_q = validation_results.groupby(date_col)[['Actual', 'Predicted', 'Pred_Q05', 'Pred_Q95']].sum().reset_index()

            # Graficar demanda agregada con intervalo de predicción
            plt.figure(figsize=(15, 7))
            plt.plot(agg_results_q[date_col], agg_results_q['Actual'], label='Real Total', marker='.', color='black')
            plt.plot(agg_results_q[date_col], agg_results_q['Predicted'], label=f'Predicción ({best_model_name})', marker='.', linestyle='--', color='blue')
            plt.fill_between(agg_results_q[date_col], agg_results_q['Pred_Q05'], agg_results_q['Pred_Q95'], color='blue', alpha=0.2, label='Intervalo Quantile 90%')
            plt.xlabel("Fecha")
            plt.ylabel("Demanda Agregada")
            plt.title("Demanda Agregada con Intervalo Predicción (Validación)")
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print("Advertencia: Datos o índices categóricos para modelos Quantile no disponibles.")
    except Exception as e:
        print(f"Error modelos Quantile: {e}")

    # 5. Growth Rate Plot: Historial Tasa de Crecimiento Mensual Promedio
    print("\nGraficando Growth_Rate_MoM Histórico...")
    # Usar dataframes originales que contienen la feature
    if 'Growth_Rate_MoM' in train_df_orig.columns and 'Growth_Rate_MoM' in valid_df_orig.columns:
        # Concatenar historial de crecimiento
        growth_hist_df = pd.concat([
            train_df_orig[[date_col, 'Growth_Rate_MoM']],
            valid_df_orig[[date_col, 'Growth_Rate_MoM']]
        ], ignore_index=True)
        growth_hist_df[date_col] = pd.to_datetime(growth_hist_df[date_col])

        # Calcular crecimiento promedio por fecha
        avg_growth = growth_hist_df.groupby(date_col)['Growth_Rate_MoM'].mean().reset_index()

        # Seleccionar últimos 24 meses para visualización
        last_24m_growth = avg_growth[avg_growth[date_col] >= avg_growth[date_col].max() - pd.DateOffset(months=23)]

        # Graficar historial de crecimiento promedio
        plt.figure(figsize=(15, 5))
        plt.plot(last_24m_growth[date_col], last_24m_growth['Growth_Rate_MoM'], label='Tasa Crecimiento MoM Promedio', marker='.')
        plt.axhline(0, color='red', linestyle='--', linewidth=0.8, label='Crecimiento Cero')
        plt.xlabel("Fecha")
        plt.ylabel("Tasa Crecimiento MoM Promedio")
        plt.title("Historial Tasa Crecimiento Mensual Promedio")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("No se encontró la columna 'Growth_Rate_MoM' en los dataframes originales.")

    # 6. Residual Plot: Residuos Agregados y Distribución en Validación
    print("\nGraficando Residuos Agregados en Validación...")
    if error_series_best is not None:
        # Agregar residuos al DataFrame de resultados de validación
        validation_results['Residual'] = error_series_best[:len(validation_results)]
        validation_results[date_col] = pd.to_datetime(validation_results[date_col])

        # Calcular residuo promedio por fecha
        agg_residuals = validation_results.groupby(date_col)['Residual'].mean().reset_index()

        # Graficar residuos promedio a lo largo del tiempo
        plt.figure(figsize=(15, 5))
        plt.plot(agg_residuals[date_col], agg_residuals['Residual'], label=f'Residuo Promedio ({best_model_name})', marker='.')
        plt.axhline(0, color='red', linestyle='--', linewidth=0.8, label='Error Cero')
        plt.xlabel("Fecha (Validación)")
        plt.ylabel("Residuo Promedio (Real - Predicción)")
        plt.title("Residuos Promedio Modelo en Validación")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Graficar distribución de errores
        plt.figure(figsize=(10, 5))
        sns.histplot(error_series_best, kde=True)
        plt.title(f'Distribución de Errores ({best_model_name}) - Validación')
        plt.xlabel("Error (Real - Predicción)")
        plt.grid(True)
        plt.show()
    else:
        print("No se pueden graficar residuos (errores del mejor modelo no disponibles).")

    # 7. Plot Best/Worst Performing Series: Gráficas de series individuales
    print(f"\nGraficando {N_BEST_WORST_SERIES} mejores/peores series por MAE en validación...")
    if error_series_best is not None:
        validation_results['AbsError'] = np.abs(error_series_best[:len(validation_results)])
        # Asegurar que los decodificadores están disponibles
        local_decoders = decoders if decoders else (joblib.load('decoders.joblib') if os.path.exists('decoders.joblib') else None)
        muni_decoder = local_decoders.get(muni_encoder_col) if local_decoders else None
        serv_decoder = local_decoders.get(service_encoder_col) if local_decoders else None

        if muni_decoder and serv_decoder:
            # Agrupar por serie (municipio, servicio) y calcular MAE
            series_mae = validation_results.groupby([muni_encoder_col, service_encoder_col])['AbsError'].mean().sort_values()

            # Seleccionar las series con mayor y menor MAE
            worst_series = series_mae.tail(N_BEST_WORST_SERIES).index.tolist()
            best_series = series_mae.head(N_BEST_WORST_SERIES).index.tolist()

            print(f"\n--- {N_BEST_WORST_SERIES} Peores Series (Mayor MAE) ---")
            # Iterar y graficar las peores series
            for i, (m_code, s_code) in enumerate(worst_series[::-1]):  # Graficar de peor a menos peor
                m_name = muni_decoder.get(m_code, f'Unknown_{m_code}')
                s_name = serv_decoder.get(s_code, f'Unknown_{s_code}')
                series_data = validation_results[(validation_results[muni_encoder_col] == m_code) & (validation_results[service_encoder_col] == s_code)]
                mae_val = series_mae.loc[(m_code, s_code)]
                plt.figure(figsize=(12, 4))
                plt.plot(series_data['Date'], series_data['Actual'], label='Real', marker='.')
                plt.plot(series_data['Date'], series_data['Predicted'], label='Predicción', marker='.', linestyle='--')
                plt.title(f"Peor Serie #{i+1}: {m_name} - {s_name} (MAE: {mae_val:.3f})")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()

            print(f"\n--- {N_BEST_WORST_SERIES} Mejores Series (Menor MAE) ---")
            # Iterar y graficar las mejores series
            for i, (m_code, s_code) in enumerate(best_series):
                m_name = muni_decoder.get(m_code, f'Unknown_{m_code}')
                s_name = serv_decoder.get(s_code, f'Unknown_{s_code}')
                series_data = validation_results[(validation_results[muni_encoder_col] == m_code) & (validation_results[service_encoder_col] == s_code)]
                mae_val = series_mae.loc[(m_code, s_code)]
                plt.figure(figsize=(12, 4))
                plt.plot(series_data['Date'], series_data['Actual'], label='Real', marker='.')
                plt.plot(series_data['Date'], series_data['Predicted'], label='Predicción', marker='.', linestyle='--')
                plt.title(f"Mejor Serie #{i+1}: {m_name} - {s_name} (MAE: {mae_val:.3f})")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
        else:
            print("No se pudieron cargar decoders para graficar mejores/peores series.")
    else:
        print("Errores del mejor modelo no disponibles para análisis de mejores/peores series.")


# Función auxiliar para predicción futura
def generate_future_features(current_date, unique_combos, history_df, model_features, hist_target_col, apply_log_transform, muni_enc_col, serv_enc_col, date_col):
    """
    Genera las features para un mes futuro específico para cada combinación
    única de municipio y tipo de servicio, basándose en datos históricos
    y características temporales. Utiliza mapeo eficiente para lags y
    cálculos rolling/growth.

    Args:
        current_date (pd.Timestamp): La fecha (mes) para la cual generar features.
        unique_combos (pd.DataFrame): DataFrame con combinaciones únicas
                                      de municipio y tipo de servicio codificados.
        history_df (pd.DataFrame): DataFrame con datos históricos utilizados para
                                   calcular lags, rolling means/stds, etc. Debe
                                   contener las columnas necesarias y la target_col
                                   en la escala transformada si apply_log_transform es True.
        model_features (list): Lista de nombres de features esperadas por los modelos.
        hist_target_col (str): Nombre de la columna objetivo en history_df (en la escala
                               transformada si apply_log_transform es True).
        apply_log_transform (bool): Indica si se aplicó transformación logarítmica
                                    a la variable objetivo histórica.
        muni_enc_col (str): Nombre de la columna de municipio codificado.
        serv_enc_col (str): Nombre de la columna de tipo de servicio codificado.
        date_col (str): Nombre de la columna de fecha.


    Returns:
        tuple: Una tupla que contiene:
            - current_future_df (pd.DataFrame): DataFrame con las features generadas
                                               para la fecha futura. Incluye las columnas
                                               originales de unique_combos + fecha + features.
            - current_X (pd.DataFrame): DataFrame conteniendo solo las columnas de features
                                       en el orden esperado por el modelo.
    """
    # print(f"Generando features para {current_date:%Y-%m}...")
    municipality_encoded_col = muni_enc_col
    service_type_encoded_col = serv_enc_col

    # Inicializar DataFrame para el mes futuro
    current_future_df = unique_combos.copy()
    current_future_df[date_col] = current_date
    current_future_df[municipality_encoded_col] = current_future_df[municipality_encoded_col].astype(int)
    current_future_df[service_type_encoded_col] = current_future_df[service_type_encoded_col].astype(int)

    # 1. Date Features
    date_features_in_model = ['Year_Fraction', 'Month_sin', 'Month_cos', 'Quarter_sin', 'Quarter_cos', 'Quarter', 'Month_of_Year', 'Is_Holiday_Month']
    for feat in date_features_in_model:
        if feat in model_features:
            try:
                if feat == 'Year_Fraction':
                    current_future_df[feat] = current_date.dayofyear / (366 if current_date.is_leap_year else 365)
                elif feat == 'Month_sin':
                    current_future_df[feat] = np.sin(2 * np.pi * current_date.month / 12)
                    current_future_df[feat] = current_future_df[feat].round(6)
                elif feat == 'Month_cos':
                    current_future_df[feat] = np.cos(2 * np.pi * current_date.month / 12)
                    current_future_df[feat] = current_future_df[feat].round(6)
                elif feat == 'Quarter_sin':
                    current_future_df[feat] = np.sin(2 * np.pi * current_date.quarter / 4)
                    current_future_df[feat] = current_future_df[feat].round(6)
                elif feat == 'Quarter_cos':
                    current_future_df[feat] = np.cos(2 * np.pi * current_date.quarter / 4)
                    current_future_df[feat] = current_future_df[feat].round(6)
                elif feat == 'Quarter':
                    current_future_df[feat] = current_date.quarter
                elif feat == 'Month_of_Year':
                    current_future_df[feat] = current_date.month
                elif feat == 'Is_Holiday_Month':
                    # Ejemplo simple: Enero, Julio, Diciembre como meses festivos
                    current_future_df[feat] = int(current_date.month in [1, 7, 12])
            except Exception as e:
                print(f"Error calculando feature de fecha '{feat}': {e}")

    # 2. Lags (Usando .map())
    # Identificar lags necesarios por el modelo y lags adicionales para rolling/growth
    lag_features_in_model = [f for f in model_features if 'Service_Count_lag_' in f]
    # Lags necesarios para el modelo + lags adicionales para calcular rolling/growth (1 y 12/13)
    lag_amounts_needed = sorted(list(set([int(re.search(r'\d+$', f).group()) for f in lag_features_in_model] + [1, 12])), reverse=True)

    # Crear MultiIndex en el DataFrame futuro para mapear
    current_future_df_indexed = current_future_df.set_index([municipality_encoded_col, service_type_encoded_col])
    # Indexar historial para mapeo rápido
    history_indexed = history_df.set_index([date_col, municipality_encoded_col, service_type_encoded_col])

    # Diccionario temporal para almacenar lags calculados en la escala transformada
    temp_lags = {}
    for lag in lag_amounts_needed:
        lag_date = current_date - pd.DateOffset(months=lag)
        try:
            # Intentar obtener valores del historial usando el MultiIndex
            lag_values_transformed = history_indexed.loc[lag_date][hist_target_col]
            # Mapear estos valores a las combinaciones únicas del DataFrame futuro
            mapped_lags_transformed = current_future_df_indexed.index.map(lag_values_transformed)
        except KeyError:
            # Si no hay datos históricos para la fecha de lag, usar 0.0 (en la escala transformada)
            mapped_lags_transformed = pd.Series(0.0, index=current_future_df_indexed.index)
            # print(f"    Advertencia: Datos históricos no encontrados para lag {lag} en {lag_date:%Y-%m}.") # Mensaje menos verboso

        # Guardar lag transformado para cálculo de rolling/growth
        temp_lags[f'target_lag_{lag}'] = mapped_lags_transformed.fillna(0.0)

        # Asignar al nombre de columna lag correcto en escala ORIGINAL (si el feature está en el modelo)
        lag_col_name_model = f'Service_Count_lag_{lag}'
        if lag_col_name_model in model_features:
            mapped_lags_original = np.expm1(mapped_lags_transformed) if apply_log_transform else mapped_lags_transformed
            # Asignar de vuelta al DataFrame original (manejar el índice)
            # Usamos .values para asegurar que el orden se mantenga al asignar a un DataFrame no indexado
            current_future_df[lag_col_name_model] = mapped_lags_original.fillna(0.0).values

    # Restaurar índice si fue cambiado a MultiIndex
    if isinstance(current_future_df.index, pd.MultiIndex):
        current_future_df = current_future_df.reset_index()

    # 3. Rolling Features (calculados sobre historia previa a la fecha actual)
    rolling_features_in_model = [
        f for f in model_features if 'Service_Count_rolling_' in f
    ]

    # Filtrar historial para incluir solo datos ANTES de la fecha actual
    history_for_rolling = history_df[history_df[date_col] < current_date]

    if not history_for_rolling.empty:
        # Asegurar tipos enteros para claves de agrupación antes de groupby
        history_for_rolling[municipality_encoded_col] = pd.to_numeric(
            history_for_rolling[municipality_encoded_col], errors='coerce'
        ).fillna(-1).astype(int)

        history_for_rolling[service_type_encoded_col] = pd.to_numeric(
            history_for_rolling[service_type_encoded_col], errors='coerce'
        ).fillna(-1).astype(int)

        # Asegurar mismos tipos en el df futuro para el merge/join implícito del mapeo
        current_future_df[municipality_encoded_col] = current_future_df[municipality_encoded_col].astype(int)
        current_future_df[service_type_encoded_col] = current_future_df[service_type_encoded_col].astype(int)

        for roll_feat in rolling_features_in_model:
            try:
                # Parsear nombre de feature para obtener función de agregación y ventana
                parts = roll_feat.split('_')
                agg_func = parts[-2]
                window_size = int(parts[-1])

                # Calcular rolling por grupo en el historial filtrado
                rolling_calc = (
                    history_for_rolling
                    .groupby([municipality_encoded_col, service_type_encoded_col])[hist_target_col]
                    .rolling(
                        window=window_size,
                        min_periods=max(1, window_size // 2), # Usar al menos la mitad de la ventana
                        closed='left' # La ventana termina antes de la fecha actual
                    )
                    .agg(agg_func)
                )

                # Obtener el ÚLTIMO valor rolling calculado para cada grupo (es el valor relevante para la fecha actual)
                last_rolling_vals = rolling_calc.groupby(level=[0, 1]).last()

                # Mapear estos últimos valores a current_future_df usando su índice de combinación
                index_keys = current_future_df.set_index([municipality_encoded_col, service_type_encoded_col]).index
                rolling_vals_mapped = index_keys.map(last_rolling_vals)

                # Aplicar transformaciones inversas si es necesario para las columnas (std se mantiene transformado)
                if apply_log_transform and 'mean' in agg_func:
                     # np.expm1(log(mean)) != mean. Approximation.
                    current_future_df[roll_feat] = np.expm1(rolling_vals_mapped)
                elif apply_log_transform and 'std' in agg_func:
                    # std se mantiene en la escala logarítmica, no se le aplica expm1
                    current_future_df[roll_feat] = rolling_vals_mapped
                else:
                    current_future_df[roll_feat] = rolling_vals_mapped

                # Rellenar valores faltantes con 0.0 (para series nuevas o con historial insuficiente)
                current_future_df[roll_feat] = current_future_df[roll_feat].fillna(0.0)

            except Exception as e:
                print(f"Error calculando rolling feature '{roll_feat}': {e}")
                # Asegurar que la columna existe aunque haya error
                current_future_df[roll_feat] = 0.0

    else:
        # Si no hay historia previa, rellenar todos los features rolling con 0.0
        for roll_feat in rolling_features_in_model:
            current_future_df[roll_feat] = 0.0

    # Restaurar índice si quedó como MultiIndex después del mapeo
    if isinstance(current_future_df.index, pd.MultiIndex):
        current_future_df = current_future_df.reset_index()

    # 4. Growth Rate Features
    # print(f"    Calculando GrowthRate...")
    try:
        if 'Growth_Rate_MoM' in model_features:
            # Usar lags transformados almacenados temporalmente
            lag1_t = temp_lags.get('target_lag_1', pd.Series(0.0, index=current_future_df_indexed.index))
            lag2_t = temp_lags.get('target_lag_2', pd.Series(0.0, index=current_future_df_indexed.index))

            # Calcular ratio en escala transformada, luego convertir a original para el crecimiento
            # ratio_mom = np.where(lag2_t != 0, lag1_t / lag2_t, np.nan) # Esto no es correcto en escala log
            # Correcto en escala original
            lag1_orig = np.expm1(lag1_t) if apply_log_transform else lag1_t
            lag2_orig = np.expm1(lag2_t) if apply_log_transform else lag2_t
            ratio_mom = np.where(lag2_orig != 0, lag1_orig / lag2_orig, np.nan)

            # Calcular crecimiento (ratio - 1) y rellenar NaNs/inf con 0
            current_future_df['Growth_Rate_MoM'] = pd.Series(ratio_mom).replace([np.inf, -np.inf], np.nan).fillna(0) - 1

        if 'Growth_Rate_YoY' in model_features:
            # Usar lags transformados almacenados temporalmente
            lag1_t = temp_lags.get('target_lag_1', pd.Series(0.0, index=current_future_df_indexed.index))
            # Usar lag 12 como aproximación de "año anterior"
            lag12_t = temp_lags.get('target_lag_12', pd.Series(0.0, index=current_future_df_indexed.index))

            # Calcular ratio en escala original
            lag1_orig = np.expm1(lag1_t) if apply_log_transform else lag1_t
            lag12_orig = np.expm1(lag12_t) if apply_log_transform else lag12_t
            ratio_yoy = np.where(lag12_orig != 0, lag1_orig / lag12_orig, np.nan)

            # Calcular crecimiento (ratio - 1) y rellenar NaNs/inf con 0
            current_future_df['Growth_Rate_YoY'] = pd.Series(ratio_yoy).replace([np.inf, -np.inf], np.nan).fillna(0) - 1
        else:
            # Asegurar que la columna existe si no está en model_features
            if 'Growth_Rate_YoY' not in current_future_df.columns:
                 current_future_df['Growth_Rate_YoY'] = 0

    except Exception as e:
        print(f"Error calculando GrowthRate features: {e}")
        # Asegurar que las columnas existen aunque haya error
        if 'Growth_Rate_MoM' in model_features and 'Growth_Rate_MoM' not in current_future_df.columns:
            current_future_df['Growth_Rate_MoM'] = 0
        if 'Growth_Rate_YoY' in model_features and 'Growth_Rate_YoY' not in current_future_df.columns:
             current_future_df['Growth_Rate_YoY'] = 0


    # 5. Other Propagated Features (features que se propagan del último valor conocido)
    other_features_in_model = [
        f for f in model_features
        if f not in date_features_in_model
        and f not in lag_features_in_model
        and f not in rolling_features_in_model
        and f not in ['Growth_Rate_MoM', 'Growth_Rate_YoY']
        and f not in [municipality_encoded_col, service_type_encoded_col]
    ]

    if other_features_in_model:
        # Filtrar historia válida antes de la fecha actual para obtener los últimos valores
        history_for_prop = history_df[history_df[date_col] < current_date].copy()

        if not history_for_prop.empty:
            # Asegurar claves numéricas para merge/groupby
            history_for_prop[municipality_encoded_col] = pd.to_numeric(
                history_for_prop[municipality_encoded_col], errors='coerce'
            ).fillna(-1).astype(int)

            history_for_prop[service_type_encoded_col] = pd.to_numeric(
                history_for_prop[service_type_encoded_col], errors='coerce'
            ).fillna(-1).astype(int)

            # Asegurar tipos en el DataFrame futuro para el merge
            current_future_df[municipality_encoded_col] = current_future_df[municipality_encoded_col].astype(int)
            current_future_df[service_type_encoded_col] = current_future_df[service_type_encoded_col].astype(int)


            # Obtener los índices de las últimas observaciones disponibles para cada serie (grupo)
            last_known_indices = history_for_prop.groupby(
                [municipality_encoded_col, service_type_encoded_col]
            )[date_col].idxmax()

            # Filtrar índices válidos (donde se encontró un último valor)
            valid_indices = last_known_indices.dropna()

            if not valid_indices.empty:
                # Extraer las últimas observaciones usando los índices encontrados
                last_known_other = history_for_prop.loc[valid_indices].copy()

                # Seleccionar solo las columnas necesarias para el merge
                cols_to_select = (
                    [municipality_encoded_col, service_type_encoded_col] +
                    [f for f in other_features_in_model if f in last_known_other.columns]
                )

                last_known_other = last_known_other[cols_to_select]

                # Eliminar las columnas antiguas del DataFrame actual (si existen) y unir con las últimas observaciones
                # Esto propaga el último valor conocido para esas features
                current_future_df = pd.merge(
                    current_future_df.drop(
                        columns=[c for c in other_features_in_model if c in current_future_df.columns],
                        errors='ignore'
                    ),
                    last_known_other,
                    on=[municipality_encoded_col, service_type_encoded_col],
                    how='left' # Usar left merge para mantener todas las filas del futuro
                )
            else:
                 # Si no hay datos válidos para propagar en el historial, rellenar con 0.0
                 for f in other_features_in_model:
                    current_future_df[f] = 0.0

        else:
            # Si no hay historial, rellenar todas estas features con 0.0
            for f in other_features_in_model:
                current_future_df[f] = 0.0


    # Verificación final y limpieza
    # Rellenar cualquier NaN restante con 0 (puede ocurrir si una serie no tenía historia)
    current_future_df = current_future_df.fillna(0)

    # Asegurar que todas las columnas de features sean numéricas (excepto la fecha si se desea mantener)
    for col in model_features:
        if col not in [date_col, municipality_encoded_col, service_type_encoded_col]:
             if not pd.api.types.is_numeric_dtype(current_future_df[col]):
                 current_future_df[col] = pd.to_numeric(current_future_df[col], errors='coerce').fillna(0)

    # Asegurar que las claves de grupo sean de tipo int para la predicción del modelo
    current_future_df[municipality_encoded_col] = current_future_df[municipality_encoded_col].astype(int)
    current_future_df[service_type_encoded_col] = current_future_df[service_type_encoded_col].astype(int)


    # Verificar que todas las features esperadas por el modelo estén presentes
    missing_cols_final = set(model_features) - set(current_future_df.columns)
    if missing_cols_final:
        print(f"ERROR CRÍTICO: Faltan columnas de features {missing_cols_final} ANTES de predecir para {current_date:%Y-%m}")
        # Considerar si se debe salir o intentar rellenar con 0
        exit() # Salir en caso de error crítico de features faltantes

    # Seleccionar y ordenar las columnas para obtener X en el formato correcto para el modelo
    current_X = current_future_df[model_features]

    return current_future_df, current_X

# Function to plot feature importance
def plot_feature_importance(models_dict, feature_list, X_train_xgb_cols=None):
    """
    Genera y muestra gráficos de importancia de características
    para los modelos individuales cargados.

    Args:
        models_dict (dict): Diccionario con los modelos cargados.
        feature_list (list): Lista de nombres de características utilizadas
                             por los modelos.
        X_train_xgb_cols (list, optional): Lista de nombres de columnas
                                           sanitizados utilizados por XGBoost
                                           si aplica. Defaults to None.
    """
    print("\n--- Importancia de Características (Modelos Individuales) ---")
    for model_name, model_obj in models_dict.items():
      # Excluir el ensamble y verificar que el modelo se cargó correctamente
      if model_name != 'Ensemble' and model_obj is not None:
        try:
            print(f"\nImportancia para {model_name}...")
            plt.figure(figsize=(10, max(8, len(feature_list)//3))) # Ajustar tamaño basado en # features
            model_features_list = feature_list # Usar la lista de features real

            if isinstance(model_obj, lgb.LGBMRegressor):
                # Gráfica de importancia para LightGBM
                lgb.plot_importance(model_obj, max_num_features=min(25, len(model_features_list)), height=0.8, importance_type='gain')
                plt.title(f'Importancia ({model_name} - Gain)')
            elif isinstance(model_obj, xgb.XGBRegressor) and X_train_xgb_cols is not None:
                # Gráfica de importancia para XGBoost (requiere nombres de columnas sanitizados)
                sanitized_feature_names = X_train_xgb_cols
                importances = model_obj.feature_importances_
                # Crear un mapeo inverso para mostrar nombres originales si es posible
                name_map = dict(zip(sanitized_feature_names, model_features_list))
                f_scores = pd.Series(importances, index=sanitized_feature_names).sort_values(ascending=False)
                top_features = f_scores[:min(25, len(f_scores))]
                top_features.sort_values(ascending=True).plot(kind='barh')
                # Usar el mapeo para las etiquetas del eje Y
                plt.yticks(plt.yticks()[0], [name_map.get(label.get_text(), label.get_text()) for label in plt.gca().get_yticklabels()])
                plt.xlabel("Importancia")
                plt.title(f'Importancia ({model_name})')
            elif isinstance(model_obj, RandomForestRegressor):
                # Gráfica de importancia para RandomForest
                importances = model_obj.feature_importances_
                # Seleccionar los índices de las características más importantes
                indices = np.argsort(importances)[-min(25, len(model_features_list)):]
                plt.barh(range(len(indices)), importances[indices], align='center')
                # Usar nombres de características originales para las etiquetas del eje Y
                plt.yticks(range(len(indices)), [model_features_list[i] for i in indices])
                plt.xlabel('Importancia (Impurity Decrease)')
                plt.title(f'Importancia ({model_name})')
            else:
                print(f"Tipo de modelo no soportado para graficar importancia: {type(model_obj)}")
                continue # Saltar a la siguiente iteración si el tipo no es soportado

            plt.tight_layout()
            plt.show()
            print(f"Gráfica generada para {model_name}.")
        except Exception as e:
            print(f"Error graficando importancia para {model_name}: {e}")

# --- 4. Main Execution ---
if __name__ == "__main__":
    # Configuración de rutas (ejemplo, ajustar según la estructura de archivos)
    TRAIN_DATA_PATH = './data/processed/train_data.csv' # Ajustar si es necesario
    VALID_DATA_PATH = './data/processed/valid_data.csv' # Ajustar si es necesario
    TARGET_COLUMN = 'Service_Count'
    DATE_COLUMN = 'Date'
    MUNI_ENCODED_COLUMN = 'Municipality_encoded'
    SERVICE_ENCODED_COLUMN = 'Service_Type_encoded'
    # Columnas de capacidad a excluir del análisis de features
    CAPACITY_COLS = [col for col in pd.read_csv(TRAIN_DATA_PATH, nrows=0).columns if 'Capacity' in col] # Cargar solo headers para identificar

    # 1. Cargar datos y artefactos
    train_df_orig, valid_df_orig, models, decoders = load_data_and_artifacts(TRAIN_DATA_PATH, VALID_DATA_PATH, MODEL_DIR)

    # Verificar si al menos un modelo fue cargado
    if not any(models.values()):
        print("Error Fatal: No se cargó ningún modelo. Terminando.")
        exit()

    # 2. Preparar datos para evaluación
    # Seleccionar features consistentes con el entrenamiento del modelo final
    # Nota: X_train_q y y_train_q, X_valid_q y y_valid_q, cat_indices_q
    # NO se preparan aquí, ya que se asume que deben estar preparadas
    # para el entrenamiento de cuantiles si se van a usar.
    # Para este script de análisis, solo necesitamos X_valid para la predicción
    # de los modelos principales y las columnas para los plots.

    # Usar las columnas presentes en el DataFrame de validación original
    # y filtrar según las características seleccionadas para el modelo
    all_possible_features = [col for col in valid_df_orig.columns if col not in [TARGET_COLUMN, DATE_COLUMN]]
    features_for_analysis = select_features_for_analysis(valid_df_orig, CAPACITY_COLS)

    # Preparar X_valid usando las features seleccionadas
    X_valid, final_feature_list = prepare_X_for_prediction(valid_df_orig, features_for_analysis, MUNI_ENCODED_COLUMN)

    # Preparar DMatrix para XGBoost si se cargó el modelo
    X_valid_xgb = xgb.DMatrix(X_valid, enable_categorical=True) if 'XGBoost' in models and models['XGBoost'] is not None else None

    # Obtener la variable objetivo real (en escala original)
    y_valid_original = valid_df_orig[TARGET_COLUMN]

    # 3. Generar predicciones
    predictions = get_predictions(models, X_valid, APPLY_LOG_TRANSFORM, X_valid_xgb)

    # 4. Evaluar predicciones
    # Asegurar que y_valid_original tenga la misma longitud que las predicciones si hay discrepancias
    # Esto es importante si algún paso anterior (como dropna en load_data) alteró el DataFrame.
    min_len_eval = min(len(y_valid_original), *(len(p) for p in predictions.values() if p is not None))
    y_valid_original_aligned = y_valid_original[:min_len_eval]

    metrics, errors_dict = evaluate_predictions(y_valid_original_aligned, predictions)

    # Determinar el mejor modelo basado en MAE
    best_model_name = None
    min_mae = float('inf')
    for name, model_metrics in metrics.items():
        if model_metrics is not None and model_metrics['MAE'] < min_mae:
            min_mae = model_metrics['MAE']
            best_model_name = name
    print(f"\nMejor modelo (basado en MAE): {best_model_name}")

    # 5. Análisis de errores por grupo
    print("\nAnálisis de errores por Municipio...")
    if best_model_name and best_model_name in errors_dict and MUNI_ENCODED_COLUMN in valid_df_orig.columns:
        # Asegurar alineación del DataFrame de validación para el análisis de errores
        valid_df_aligned = valid_df_orig.copy().iloc[:min_len_eval].reset_index(drop=True) # Reset index después de slicing

        errors_by_muni = analyze_errors_by_group(valid_df_aligned, errors_dict[best_model_name], [MUNI_ENCODED_COLUMN])
        print("Top 10 Municipios con mayor MAE:")
        print(errors_by_muni.head(10))

        print("\nAnálisis de errores por Tipo de Servicio...")
        if SERVICE_ENCODED_COLUMN in valid_df_aligned.columns:
            errors_by_service = analyze_errors_by_group(valid_df_aligned, errors_dict[best_model_name], [SERVICE_ENCODED_COLUMN])
            print("Top 10 Tipos de Servicio con mayor MAE:")
            print(errors_by_service.head(10))

            print("\nAnálisis de errores por Municipio y Tipo de Servicio (Series)...")
            errors_by_series = analyze_errors_by_group(valid_df_aligned, errors_dict[best_model_name], [MUNI_ENCODED_COLUMN, SERVICE_ENCODED_COLUMN])
            print(f"Top {N_BEST_WORST_SERIES} Series (Municipio - Servicio) con mayor MAE:")
            print(errors_by_series.head(N_BEST_WORST_SERIES))
            print(f"\nTop {N_BEST_WORST_SERIES} Series (Municipio - Servicio) con menor MAE:")
            print(errors_by_series.tail(N_BEST_WORST_SERIES))
        else:
            print(f"Columna '{SERVICE_ENCODED_COLUMN}' no encontrada en el DataFrame de validación para análisis por servicio/serie.")
    else:
        print("No se pudo realizar el análisis de errores por grupo (mejor modelo no identificado, errores no disponibles o columnas de agrupación faltantes).")


    # 6. Preparar DataFrame de resultados para graficar
    # Asegurar que el DataFrame de resultados para graficar contenga las columnas necesarias
    # y esté alineado con las predicciones y errores.
    if best_model_name and best_model_name in predictions:
         validation_results_for_plot = valid_df_orig.copy().iloc[:min_len_eval].reset_index(drop=True)
         validation_results_for_plot['Actual'] = y_valid_original_aligned
         validation_results_for_plot['Predicted'] = predictions[best_model_name][:min_len_eval]

         # Incluir columnas clave para análisis por serie
         if MUNI_ENCODED_COLUMN in valid_df_orig.columns:
             validation_results_for_plot[MUNI_ENCODED_COLUMN] = valid_df_orig[MUNI_ENCODED_COLUMN].iloc[:min_len_eval].values
         if SERVICE_ENCODED_COLUMN in valid_df_orig.columns:
              validation_results_for_plot[SERVICE_ENCODED_COLUMN] = valid_df_orig[SERVICE_ENCODED_COLUMN].iloc[:min_len_eval].values
         if DATE_COLUMN in valid_df_orig.columns:
              validation_results_for_plot[DATE_COLUMN] = valid_df_orig[DATE_COLUMN].iloc[:min_len_eval].values


    else:
        print("No se pudo preparar el DataFrame de resultados para graficar (mejor modelo o predicciones no disponibles).")
        validation_results_for_plot = None # Asegurar que la variable no existe o es None


    # 7. Generar gráficas de evaluación
    if validation_results_for_plot is not None:
        # Para el plot de cuantiles, se necesitan los datos de entrenamiento y validación
        # en la escala *transformada* y los índices de las categóricas.
        # Si estos no se prepararon previamente en un script de entrenamiento,
        # no se podrá generar la gráfica de cuantiles.
        # Se pasa None a la función de ploteo si no están disponibles.
        X_train_q_plot = None # Asumimos que no están disponibles a menos que se carguen
        y_train_q_plot = None
        X_valid_q_plot = None # X_valid ya está en escala original (para predicción), necesitamos transformada para quantile training
        y_valid_q_plot = None # y_valid_original está en escala original, necesitamos transformada
        cat_indices_q_plot = None # Necesitamos los índices de las columnas categóricas

        # Para generar los modelos de cuantil aquí, necesitaríamos:
        # - El DataFrame de entrenamiento *completo* en la escala transformada
        # - El DataFrame de validación *completo* en la escala transformada
        # - La lista de columnas categóricas y sus índices.
        # Esto complejiza el script. Por ahora, asumimos que si el usuario quiere
        # la gráfica de cuantiles, estos inputs deberían venir de alguna otra parte
        # o ser cargados aquí si se guardaron. Para simplificar, se dejan como None.


        plot_evaluation_graphs(
            validation_results_for_plot,
            best_model_name,
            train_df_orig,
            valid_df_orig,
            decoders,
            MUNI_ENCODED_COLUMN,
            SERVICE_ENCODED_COLUMN,
            TARGET_COLUMN,
            DATE_COLUMN,
            error_series_best=errors_dict.get(best_model_name), # Pasar los errores del mejor modelo
            X_train_q=X_train_q_plot, # Pasa None si no están disponibles
            y_train_q=y_train_q_plot, # Pasa None si no están disponibles
            X_valid_q=X_valid_q_plot, # Pasa None si no están disponibles
            y_valid_q=y_valid_q_plot, # Pasa None si no están disponibles
            cat_indices_q=cat_indices_q_plot # Pasa None si no están disponibles
        )
    else:
        print("No se pudieron generar las gráficas de evaluación.")


    # 8. Graficar Importancia de Características
    if final_feature_list:
        # Para XGBoost, si se usaron nombres de columnas sanitizados durante el entrenamiento,
        # es necesario pasarlos. Asumimos que si el modelo XGBoost se cargó,
        # los nombres sanitizados podrían ser diferentes. No tenemos esa información
        # aquí. Si XGBoost se entrenó con nombres sanitizados y no se proveen,
        # la gráfica de importancia para XGBoost fallará.
        # Para este script, pasamos None para X_train_xgb_cols y aceptamos la limitación.
        plot_feature_importance(models, final_feature_list, X_train_xgb_cols=None)
    else:
        print("No hay lista de features finales para graficar importancia.")


    # 9. Guardar resultados del análisis en JSON
    print(f"\nGuardando resultados del análisis en {JSON_OUTPUT_PATH}...")
    analysis_results = {
        'metrics': metrics,
        'best_model_mae': {best_model_name: min_mae} if best_model_name else {},
        'errors_by_muni_top10': errors_by_muni.head(10).to_dict('records') if 'errors_by_muni' in locals() else [],
        'errors_by_service_top10': errors_by_service.head(10).to_dict('records') if 'errors_by_service' in locals() else [],
        'errors_by_series_topN': errors_by_series.head(N_BEST_WORST_SERIES).to_dict('records') if 'errors_by_series' in locals() else [],
        'errors_by_series_bottomN': errors_by_series.tail(N_BEST_WORST_SERIES).to_dict('records') if 'errors_by_series' in locals() else [],
    }
    try:
        with open(JSON_OUTPUT_PATH, 'w') as f:
            json.dump(analysis_results, f, indent=4)
        print("Resultados guardados con éxito.")
    except Exception as e:
        print(f"Error al guardar resultados en JSON: {e}")

    print("\nAnálisis de modelos completado.")

def main():
    """
    Función principal que ejecuta el flujo completo de trabajo para la previsión de demanda sanitaria.
    
    El proceso incluye:
    1. Carga y preprocesamiento de datos
    2. Entrenamiento de modelos predictivos
    3. Evaluación y selección del mejor modelo
    4. Generación de predicciones para periodos futuros
    5. Formateo y presentación de resultados
    """
    global tuning_time, best_rf_params, decoders
    start_total_time = time.time()

    # Carga y preprocesamiento básico de datos
    train_df_orig, valid_df_orig = load_data('healthcare_train_data.csv', 'healthcare_valid_data.csv')
    train_df = train_df_orig.copy()
    valid_df = valid_df_orig.copy()

    # Definición de columnas clave para el procesamiento
    target_col = 'Service_Count'
    date_col = 'Date'
    muni_enc_col = 'Municipality_encoded'
    serv_enc_col = 'Service_Type_encoded'
    muni_orig_col = 'Municipality'
    serv_orig_col = 'Service_Type'

    print("\nNOTA: Carga y unión de features de capacidad omitida.")
    capacity_cols_added = []

    # Creación de decodificadores para valores categóricos
    create_decoders(train_df, muni_enc_col, muni_orig_col, serv_enc_col, serv_orig_col)

    # Selección de características para el modelo
    features_to_exclude = ['Days_From_Now', 'Is_Anomaly', 'Is_Work_Related'] + ['Total_Max_Cantidad_Municipio', 'Avg_Max_Cantidad_Municipio', 'Std_Max_Cantidad_Municipio', 'Provider_Count_Municipio']
    id_cols_to_remove = [muni_orig_col, serv_orig_col, 'Year_Month', 'Year', 'Month', 'First_Service_Date', 'Last_Service_Date', 'First_Service_Ever']
    features = select_and_prepare_features(train_df, capacity_cols_added, target_col, date_col, id_cols_to_remove, features_to_exclude)

    # Preparación de variables X, y para entrenamiento y validación
    X_train, y_train, _, features = prepare_xy(train_df, features, target_col, muni_enc_col, APPLY_LOG_TRANSFORM)
    X_valid, y_valid, y_valid_original, features = prepare_xy(valid_df, features, target_col, muni_enc_col, APPLY_LOG_TRANSFORM)

    # Identificación de índices para características categóricas
    cat_feature_names = [muni_enc_col, serv_enc_col]
    if 'Is_Holiday_Month' in features:
        cat_feature_names.append('Is_Holiday_Month')
    cat_indices = sorted([i for i, col in enumerate(features) if col in cat_feature_names])

    # Omisión de ajuste fino y uso de parámetros previos
    print("\n--- Paso 1.5: Ajuste Fino Hiperparámetros ---")
    print("Tuning Omitido. Usando params RF previos.")

    # Entrenamiento de modelos
    print("\n--- Paso 2: Entrenamiento de Modelos ---")
    models = {}
    predictions_valid = {}
    mae_scores = {}
    rmse_scores = {}
    r2_scores = {}
    model_errors_valid = {}
    X_train_xgb, X_valid_xgb = None, None

    # Entrenamiento del modelo LGBM
    models['LGBM'], predictions_valid['LGBM'], model_errors_valid['LGBM'], mae_scores['LGBM'], rmse_scores['LGBM'], r2_scores['LGBM'] = train_evaluate_model(
        'LGBM',
        lgb.LGBMRegressor(**{
            'objective': 'regression_l2' if APPLY_LOG_TRANSFORM else 'regression_l1',
            'metric': ['mae', 'rmse'],
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 1,
            'num_leaves': 31,
            'max_depth': 10,
            'min_child_samples': 30,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42
        }),
        X_train, y_train, X_valid, y_valid_original, cat_indices, APPLY_LOG_TRANSFORM
    )
    if models.get('LGBM') is None:
        print("Fallo entrenamiento LGBM")

    # Preparación de datos y entrenamiento del modelo XGBoost
    X_train_xgb = X_train.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', str(x)))
    X_valid_xgb = X_valid.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', str(x)))
    models['XGBoost'], predictions_valid['XGBoost'], model_errors_valid['XGBoost'], mae_scores['XGBoost'], rmse_scores['XGBoost'], r2_scores['XGBoost'] = train_evaluate_model(
        'XGBoost',
        xgb.XGBRegressor(**{
            'objective': 'reg:squarederror',
            'eval_metric': ['mae', 'rmse'],
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'gamma': 0.2,
            'lambda': 1.1,
            'alpha': 0.1,
            'n_jobs': -1,
            'seed': 42,
            'tree_method': 'hist',
            'early_stopping_rounds': 50
        }),
        X_train, y_train, X_valid, y_valid_original, cat_indices, APPLY_LOG_TRANSFORM,
        X_train_xgb=X_train_xgb, X_valid_xgb=X_valid_xgb
    )
    if models.get('XGBoost') is None:
        print("Fallo entrenamiento XGBoost")

    # Entrenamiento del modelo Random Forest
    print(f"Usando params RF: {best_rf_params}")
    models['RandomForest'], predictions_valid['RandomForest'], model_errors_valid['RandomForest'], mae_scores['RandomForest'], rmse_scores['RandomForest'], r2_scores['RandomForest'] = train_evaluate_model(
        'RandomForest',
        RandomForestRegressor(**best_rf_params),
        X_train, y_train, X_valid, y_valid_original, cat_indices, APPLY_LOG_TRANSFORM
    )
    if models.get('RandomForest') is None:
        print("Fallo entrenamiento RandomForest")

    # Creación de ensamble de modelos
    print("\n--- Creando Ensamble ---")
    valid_preds_list = [p for name, p in predictions_valid.items() if p is not None and name in models and models[name] is not None]
    if len(valid_preds_list) >= 2:
        min_len = min(len(p) for p in valid_preds_list)
        aligned_preds = [p[:min_len] for p in valid_preds_list]
        ensemble_preds = np.mean(np.array(aligned_preds), axis=0)
        y_valid_aligned = y_valid_original[:min_len]
        predictions_valid['Ensemble'] = ensemble_preds
        model_errors_valid['Ensemble'] = y_valid_aligned - ensemble_preds
        mae_scores['Ensemble'] = mean_absolute_error(y_valid_aligned, ensemble_preds)
        rmse_scores['Ensemble'] = np.sqrt(mean_squared_error(y_valid_aligned, ensemble_preds))
        r2_scores['Ensemble'] = r2_score(y_valid_aligned, ensemble_preds)
        print(f"Ensemble - MAE: {mae_scores['Ensemble']:.4f}, R2: {r2_scores['Ensemble']:.4f}")
    else:
        print("No modelos suficientes para ensamble.")

    # Selección del mejor modelo y evaluación estadística
    print("\n--- Paso 3: Selección Mejor Modelo y Test Estadístico ---")
    best_model_name = None
    finite_mae_scores = {k: v for k, v in mae_scores.items() if np.isfinite(v)}
    if finite_mae_scores:
        best_model_name = min(finite_mae_scores, key=finite_mae_scores.get)
        print(f"Mejor enfoque según MAE: {best_model_name} (MAE: {mae_scores[best_model_name]:.4f}, R2: {r2_scores.get(best_model_name, 'N/A'):.4f})")
        print("Realizando prueba estadística vs Naive...")
        
        naive_pred_col = 'Service_Count_lag_1'
        if naive_pred_col in valid_df.columns:
            y_naive_preds = valid_df[naive_pred_col].fillna(0)
            y_true_valid_orig = y_valid_original
            best_model_preds_orig = predictions_valid.get(best_model_name)
            
            if best_model_preds_orig is not None:
                min_len_test = min(len(y_true_valid_orig), len(y_naive_preds), len(best_model_preds_orig))
                y_true_aligned = y_true_valid_orig[:min_len_test]
                y_naive_aligned = y_naive_preds[:min_len_test]
                best_model_preds_aligned = best_model_preds_orig[:min_len_test]
                errors_model = np.abs(y_true_aligned - best_model_preds_aligned)
                errors_naive = np.abs(y_true_aligned - y_naive_aligned)
                diff_errors = errors_model - errors_naive
                non_zero_diff = diff_errors != 0
                
                try:
                    if np.sum(non_zero_diff) > 10:
                        stat, p_value = wilcoxon(errors_model[non_zero_diff], errors_naive[non_zero_diff], alternative='less', zero_method='zsplit')
                        print(f" Prueba Wilcoxon ({best_model_name} vs Naive): W={stat:.1f}, p={p_value:.4g} - {'MEJORA SIGNIFICATIVA' if p_value < 0.05 else 'Mejora NO Sig.'}")
                    else:
                        print(" Advertencia: Pocas diferencias error no nulas para Wilcoxon.")
                except Exception as we:
                    print(f" Error Wilcoxon: {we}.")
            else:
                print(" Error: Predicciones mejor modelo no encontradas.")
        else:
            print(f" Advertencia: Columna Naive '{naive_pred_col}' no encontrada en valid_df.")
    else:
        print("Error Fatal: No se pudieron evaluar modelos.")
        exit()

    # Generación de gráficas de evaluación
    print("\n--- Paso 3.5: Gráficas de Evaluación (Validación) ---")
    if best_model_name and best_model_name in predictions_valid and predictions_valid[best_model_name] is not None:
        validation_plot_df = pd.DataFrame({
            'Date': valid_df_orig[date_col].iloc[:len(y_valid_original)],
            'Actual': y_valid_original,
            'Predicted': predictions_valid[best_model_name][:len(y_valid_original)],
            'Residual': model_errors_valid[best_model_name][:len(y_valid_original)],
            muni_enc_col: X_valid[muni_enc_col].values[:len(y_valid_original)],
            serv_enc_col: X_valid[serv_enc_col].values[:len(y_valid_original)]
        })
        plot_evaluation_graphs(
            validation_plot_df, best_model_name, train_df_orig, valid_df_orig,
            decoders, muni_enc_col, serv_enc_col, target_col, date_col,
            X_train_q=X_train, y_train_q=y_train, X_valid_q=X_valid,
            y_valid_q=y_valid, cat_indices_q=cat_indices
        )
    else:
        print("No se pueden generar gráficas evaluación (mejor modelo no válido).")

    # Generación de predicciones futuras
    print("\n--- Paso 4: Generación Predicciones Futuras (Iterativo) ---")
    last_data_date = max(train_df_orig[date_col].max(), valid_df_orig[date_col].max())
    future_dates = pd.date_range(start=last_data_date + pd.DateOffset(months=1), periods=12, freq='MS')
    print(f"Prediciendo de {future_dates.min():%Y-%m} a {future_dates.max():%Y-%m}")

    # Obtención de combinaciones únicas de municipio y servicio
    unique_combos = X_train[[muni_enc_col, serv_enc_col]].drop_duplicates()
    unique_combos[muni_enc_col] = pd.to_numeric(unique_combos[muni_enc_col], errors='coerce').fillna(-1).astype(int)
    unique_combos[serv_enc_col] = pd.to_numeric(unique_combos[serv_enc_col], errors='coerce').fillna(-1).astype(int)
    print(f"Prediciendo para {len(unique_combos)} combinaciones.")

    model_features = features.copy()
    hist_target_col = target_col + '_transformed' if APPLY_LOG_TRANSFORM else target_col

    # Preparación de datos históricos
    all_predictions_df_prep = pd.concat([train_df, valid_df], ignore_index=True)
    all_predictions_df_prep = all_predictions_df_prep.loc[:, ~all_predictions_df_prep.columns.duplicated()]

    if len(all_predictions_df_prep.columns) != len(set(all_predictions_df_prep.columns)):
        duplicate_cols = all_predictions_df_prep.columns[all_predictions_df_prep.columns.duplicated()].tolist()
        print(f"ADVERTENCIA: Columnas duplicadas encontradas y eliminadas: {duplicate_cols}")

    if APPLY_LOG_TRANSFORM:
        all_predictions_df_prep[hist_target_col] = np.log1p(all_predictions_df_prep[target_col])
    else:
        all_predictions_df_prep[hist_target_col] = all_predictions_df_prep[target_col]

    # Selección de columnas necesarias
    hist_cols_needed_iter = [date_col, muni_enc_col, serv_enc_col, hist_target_col] + model_features
    hist_cols_existing_iter = [col for col in hist_cols_needed_iter if col in all_predictions_df_prep.columns]

    # Creación de DataFrame final
    try:
        all_predictions_df = all_predictions_df_prep[hist_cols_existing_iter].copy()

        if all_predictions_df.columns.duplicated().any():
            all_predictions_df = all_predictions_df.loc[:, ~all_predictions_df.columns.duplicated()]

        all_predictions_df = all_predictions_df.drop_duplicates()

        sort_cols = [c for c in [muni_enc_col, serv_enc_col, date_col] if c in all_predictions_df.columns]
        all_predictions_df = all_predictions_df.sort_values(by=sort_cols).copy()

    except KeyError as e:
        print(f"ERROR: Columna no encontrada al ordenar - {str(e)}")
        missing_cols = set([muni_enc_col, serv_enc_col, date_col]) - set(all_predictions_df.columns)
        print(f"Columnas faltantes: {missing_cols}")
        exit()

    if best_model_name == 'XGBoost' and X_train_xgb is not None:
        model_features_sanitized = X_train_xgb.columns.tolist()

    # Bucle iterativo para predicciones futuras
    final_predictions_list = []
    for current_date in future_dates:
        try:
            current_future_df, current_X = generate_future_features(
                current_date, unique_combos, all_predictions_df,
                model_features, hist_target_col, APPLY_LOG_TRANSFORM,
                muni_enc_col, serv_enc_col, date_col
            )

            # Predicción con el mejor modelo
            current_preds_transformed = None
            predictor = models.get(best_model_name) if best_model_name != 'Ensemble' else None

            if best_model_name == 'Ensemble':
                pass  # Lógica para ensemble
            elif predictor:
                current_preds_transformed = predictor.predict(current_X)

            if current_preds_transformed is None:
                current_preds_transformed = np.zeros(len(current_X))

            # Transformación inversa y asignación de predicciones
            current_preds_original = np.expm1(current_preds_transformed) if APPLY_LOG_TRANSFORM else current_preds_transformed
            current_future_df[target_col] = np.maximum(0, current_preds_original)
            current_future_df[hist_target_col] = current_preds_transformed if APPLY_LOG_TRANSFORM else current_preds_original

            # Alineación de columnas
            cols_all_preds = all_predictions_df.columns.tolist()
            for col_hist in cols_all_preds:
                if col_hist not in current_future_df.columns:
                    current_future_df[col_hist] = 0

            # Aseguramiento de tipos de datos
            current_future_df[muni_enc_col] = current_future_df[muni_enc_col].astype(int)
            current_future_df[serv_enc_col] = current_future_df[serv_enc_col].astype(int)

            current_future_df_aligned = current_future_df.reindex(columns=cols_all_preds).fillna(0)
            all_predictions_df = pd.concat([all_predictions_df, current_future_df_aligned], ignore_index=True)
            final_predictions_list.append(current_future_df_aligned)

        except Exception as e:
            print(f"Error procesando fecha {current_date:%Y-%m}: {str(e)}")
            continue

    final_predictions_df = pd.concat(final_predictions_list, ignore_index=True) if final_predictions_list else pd.DataFrame()

    # Formateo y presentación de resultados
    if not final_predictions_df.empty:
        print("\n--- Paso 5: Formatear y Presentar Resultados ---")

        # Identificación de la columna objetivo correcta
        target_col_actual = None
        possible_targets = [
            target_col,  # Nombre original ('Service_Count')
            'Service_Count_transformed',  # Nombre transformado
            'service_count', 'SERVICE_COUNT',  # Variantes
            'demand', 'target'  # Nombres alternativos
        ]

        for col in possible_targets:
            if col in final_predictions_df.columns:
                target_col_actual = col
                break

        if not target_col_actual:
            print("ERROR: No se pudo identificar la columna objetivo")
            print("Columnas disponibles:", final_predictions_df.columns.tolist())
            exit()
        else:
            print(f"Usando columna objetivo: '{target_col_actual}'")

        # Preparación de datos para JSON
        results_for_json = {
            'overall_metrics': {
                name: {
                    'MAE': mae_scores.get(name),
                    'RMSE': rmse_scores.get(name),
                    'R2': r2_scores.get(name)
                } for name in models if name != 'Ensemble'
            },
            'best_model': best_model_name
        }

        if 'Ensemble' in mae_scores:
            results_for_json['overall_metrics']['Ensemble'] = {
                'MAE': mae_scores['Ensemble'],
                'RMSE': rmse_scores['Ensemble'],
                'R2': r2_scores['Ensemble']
            }

        # Decodificación de predicciones finales
        results_df = final_predictions_df.copy()
        try:
            local_decoders = joblib.load('decoders.joblib') if not decoders else decoders

            # Decodificación de municipios
            if muni_enc_col in local_decoders:
                results_df[muni_orig_col] = (
                    pd.to_numeric(results_df[muni_enc_col], errors='coerce')
                    .fillna(-1)
                    .astype(int)
                    .map(local_decoders.get(muni_enc_col, {}))
                    .fillna('Unknown')
                )
            else:
                results_df[muni_orig_col] = results_df[muni_enc_col]

            # Decodificación de servicios
            if serv_enc_col in local_decoders:
                results_df[serv_orig_col] = (
                    pd.to_numeric(results_df[serv_enc_col], errors='coerce')
                    .fillna(-1)
                    .astype(int)
                    .map(local_decoders.get(serv_enc_col, {}))
                    .fillna('Unknown')
                )
            else:
                results_df[serv_orig_col] = results_df[serv_enc_col]

            # Eliminación de columnas codificadas
            results_df = results_df.drop(columns=[muni_enc_col, serv_enc_col], errors='ignore')

        except Exception as e:
            print(f"Advertencia decodificando: {e}")
            results_df.rename(columns={
                muni_enc_col: muni_orig_col,
                serv_enc_col: serv_orig_col
            }, inplace=True)

        # Formateo de fechas y selección de columnas finales
        results_df[date_col] = pd.to_datetime(results_df[date_col]).dt.strftime('%Y-%m')
        final_cols_order = [date_col, muni_orig_col, serv_orig_col, target_col_actual]
        results_df = results_df[[col for col in final_cols_order if col in results_df.columns]]

        # Generación de resúmenes para JSON y visualización
        print("\n--- RESUMEN PREDICCIONES ---")

        # Resumen mensual
        monthly_summary = results_df.groupby(date_col)[target_col_actual].sum().reset_index()
        print("\nDemanda Mensual:")
        print(monthly_summary)
        results_for_json['monthly_total_forecast'] = monthly_summary.to_dict('records')

        # Resumen por servicio
        service_summary = results_df.groupby(serv_orig_col)[target_col_actual].sum().sort_values(ascending=False).reset_index()
        print(f"\nDemanda Tipo Servicio ({min(15, len(service_summary))} principales):")
        print(service_summary.head(15))
        results_for_json['service_total_forecast'] = service_summary.head(15).to_dict('records')

        # Resumen por municipio
        municipality_summary = results_df.groupby(muni_orig_col)[target_col_actual].sum().sort_values(ascending=False).reset_index()
        print(f"\nDemanda Municipio ({min(20, len(municipality_summary))} principales):")
        print(municipality_summary.head(20))
        results_for_json['municipality_total_forecast'] = municipality_summary.head(20).to_dict('records')

        print("\n--- Servicios Laborales ---")
        print("NOTA: Feature 'Is_Work_Related' omitida.")

        print("\n--- Muestra Detallada (Todas) ---")
        print(results_df.head(20))

        # Exportación a CSV
        try:
            results_df.to_csv("healthcare_demand_forecast_detailed_12_months.csv",
                            index=False,
                            encoding='utf-8-sig')
            print("\nPredicciones detalladas guardadas en CSV.")
        except Exception as e:
            print(f"Error guardando CSV: {e}")

        # Adición de muestra detallada a JSON
        results_for_json['detailed_forecast_sample'] = results_df.head(100).to_dict('records')

        # Cálculo de métricas por grupo
        validation_df_decoded = valid_df_orig.copy()

        try:
            local_decoders = joblib.load('decoders.joblib') if not decoders else decoders

            if muni_enc_col in local_decoders:
                validation_df_decoded[muni_orig_col] = (
                    pd.to_numeric(validation_df_decoded[muni_enc_col], errors='coerce')
                    .fillna(-1)
                    .astype(int)
                    .map(local_decoders.get(muni_enc_col, {}))
                    .fillna('Unknown')
                )

            if serv_enc_col in local_decoders:
                validation_df_decoded[serv_orig_col] = (
                    pd.to_numeric(validation_df_decoded[serv_enc_col], errors='coerce')
                    .fillna(-1)
                    .astype(int)
                    .map(local_decoders.get(serv_enc_col, {}))
                    .fillna('Unknown')
                )

        except Exception as e:
            print(f"Advertencia decodificando valid_df: {e}")

        if best_model_name in model_errors_valid:
            error_series_best = model_errors_valid[best_model_name]
            validation_df_decoded = validation_df_decoded.iloc[:len(error_series_best)]

            print("\nCalculando MAE por Tipo de Servicio (Validación)...")
            mae_by_service = analyze_errors_by_group(validation_df_decoded, error_series_best, [serv_orig_col])
            print(mae_by_service.head(10))
            results_for_json['mae_by_service_type'] = mae_by_service.head(10).reset_index().to_dict('records')

            print("\nCalculando MAE por Municipio (Top 20, Validación)...")
            mae_by_municipality = analyze_errors_by_group(validation_df_decoded, error_series_best, [muni_orig_col])
            print(mae_by_municipality.head(20))
            results_for_json['mae_by_municipality_top20'] = mae_by_municipality.head(20).reset_index().to_dict('records')
        else:
            print("No se pudieron calcular métricas por grupo (errores no disponibles).")

        # Exportación a JSON
        try:
            with open(JSON_OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(results_for_json, f, ensure_ascii=False, indent=4)
            print(f"\nResultados y estadísticas guardados en: {JSON_OUTPUT_PATH}")
        except Exception as e:
            print(f"Error guardando JSON: {e}")

        # Gráficos de importancia de características
        plot_feature_importance(models, features,
                              X_train_xgb_cols=X_train_xgb.columns if 'XGBoost' in models else None)
    else:
        print("Error: No se generaron predicciones finales.")

    end_total_time = time.time()
    print(f"\n--- Ejecución del script finalizada (Total: {end_total_time - start_total_time:.2f} seg) ---")


# Ejecución del script principal
if __name__ == '__main__':
    main()

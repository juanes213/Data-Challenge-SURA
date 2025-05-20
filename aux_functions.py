# -*- coding: utf-8 -*-
# --- 1. Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats.mstats import winsorize
from scipy.stats import wilcoxon, randint, uniform, friedmanchisquare, shapiro, levene
from statsmodels.stats.multitest import multipletests
import joblib
import traceback
import warnings
import re
import time
import json
import os
from scipy import optimize
import geopandas as gpd
from unidecode import unidecode
from scipy.stats import iqr
from sklearn.utils import resample
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings('ignore')

# --- 2. Configuración ---
APPLY_LOG_TRANSFORM = False
DO_TUNING = True
TUNED_RF_MAX_DEPTH_LIMIT = 25
best_rf_params = {
    'max_depth': TUNED_RF_MAX_DEPTH_LIMIT,
    'max_features': 0.8208787923016159,
    'min_samples_leaf': 3,
    'min_samples_split': 11,
    'n_estimators': 108,
    'n_jobs': -1,
    'random_state': 42
}
if 'max_features' in best_rf_params:
    best_rf_params['max_features'] = float(best_rf_params['max_features'])

decoders = {}
tuning_time = 0
JSON_OUTPUT_PATH = 'model_analysis_results_v15.json'
N_BEST_WORST_SERIES = 5
GEOJSON_URL = "https://raw.githubusercontent.com/juanferfranko/juanferfranko.github.io/master/colombia-json/colombia.geo.json"
BOOTSTRAP_N_ITER = 100

# --- 3. Funciones Auxiliares ---

def load_data(train_path, valid_path):
    """Carga dataframes y convierte 'Date' a datetime."""
    print(f"Cargando datos desde: {train_path}, {valid_path}")
    try:
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        print(f" -> Formas: Train={train_df.shape}, Valid={valid_df.shape}")
        for df_name, df in [('Train', train_df), ('Valid', valid_df)]:
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                if df['Date'].isnull().any():
                    rows_with_null_date = df['Date'].isnull().sum()
                    print(f"ADVERTENCIA: {rows_with_null_date} valores nulos en 'Date' en {df_name}. Eliminando filas.")
                    df.dropna(subset=['Date'], inplace=True)
            else:
                raise ValueError(f"Columna 'Date' no encontrada en {df_name}.")
        print(" -> Columna 'Date' convertida a datetime.")
        return train_df, valid_df
    except FileNotFoundError as e:
        print(f"Error Fatal: Archivo no encontrado - {e.filename}")
        exit()
    except ValueError as e:
        print(f"Error Fatal: {e}")
        exit()
    except Exception as e:
        print(f"Error Fatal cargando datos: {e}")
        exit()

def create_decoders(df, muni_enc_col, muni_orig_col, serv_enc_col, serv_orig_col):
    """Crea y guarda diccionarios para decodificar IDs."""
    global decoders
    print("Creando decoders...")
    decoders_local = {}
    try:
        if muni_orig_col in df.columns and muni_enc_col in df.columns:
            mun_map = df[[muni_enc_col, muni_orig_col]].drop_duplicates()
            mun_map[muni_enc_col] = pd.to_numeric(mun_map[muni_enc_col], errors='coerce').fillna(-1).astype(int)
            decoders_local[muni_enc_col] = pd.Series(mun_map[muni_orig_col].values, index=mun_map[muni_enc_col]).to_dict()
            print(f" -> Decoder creado para Municipio ({len(decoders_local[muni_enc_col])} claves).")
        else:
            print(f"Advertencia: No se pudieron crear decoders para Municipio.")

        if serv_orig_col in df.columns and serv_enc_col in df.columns:
            st_map = df[[serv_enc_col, serv_orig_col]].drop_duplicates()
            st_map[serv_enc_col] = pd.to_numeric(st_map[serv_enc_col], errors='coerce').fillna(-1).astype(int)
            decoders_local[serv_enc_col] = pd.Series(st_map[serv_orig_col].values, index=st_map[serv_enc_col]).to_dict()
            print(f" -> Decoder creado para Tipo de Servicio ({len(decoders_local[serv_enc_col])} claves).")
        else:
            print(f"Advertencia: No se pudieron crear decoders para Tipo de Servicio.")

        joblib.dump(decoders_local, 'decoders.joblib')
        print(" -> Decoders guardados.")
        decoders = decoders_local
    except Exception as e:
        print(f"Advertencia: Error creando decoders: {e}")

def detect_and_handle_outliers(df, target_col, winsorize_bounds=(0.05, 0.95), exclude_municipality=None):
    """
    Detecta y maneja outliers en una columna objetivo con winsorización, excluyendo un municipio específico.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos.
        target_col (str): Nombre de la columna objetivo.
        winsorize_bounds (tuple): Tupla con (lower_percentile, upper_percentile) para winsorización.
        exclude_municipality (str, optional): Municipio a excluir del capping.
    
    Returns:
        pd.DataFrame: DataFrame con outliers manejados.
    """
    import numpy as np
    
    df_out = df.copy()
    data = df_out[target_col].values
    
    # Calcular percentiles
    percentiles = np.percentile(data, [0.5, 1, 5, 95, 99, 99.5])
    print(f"Estadísticas iniciales:\n{df_out[target_col].describe()}")
    print(f"Percentiles (0.005, 0.01, 0.05, 0.95, 0.99, 0.995):\n{percentiles}")
    
    # Usar los percentiles especificados para los límites
    lower_bound = np.percentile(data, winsorize_bounds[0] * 100)
    upper_bound = np.percentile(data, winsorize_bounds[1] * 100)
    
    # Identificar outliers
    outliers = (data < lower_bound) | (data > upper_bound)
    n_outliers = np.sum(outliers)
    print(f" -> {n_outliers} outliers detectados (bounds: {lower_bound:.2f}, {upper_bound:.2f})")
    
    # Aplicar winsorización solo a datos no excluidos
    if exclude_municipality and 'Municipality' in df_out.columns:
        mask_exclude = df_out['Municipality'] != exclude_municipality
        data_to_winsorize = df_out.loc[mask_exclude, target_col].values
        data_to_winsorize = np.where(data_to_winsorize < lower_bound, lower_bound, data_to_winsorize)
        data_to_winsorize = np.where(data_to_winsorize > upper_bound, upper_bound, data_to_winsorize)
        df_out.loc[mask_exclude, target_col] = data_to_winsorize
    else:
        df_out[target_col] = np.where(data < lower_bound, lower_bound, df_out[target_col])
        df_out[target_col] = np.where(data > upper_bound, upper_bound, df_out[target_col])
    
    print(f"Estadísticas después de winsorización (excluyendo {exclude_municipality}):\n{df_out[target_col].describe()}")
    return df_out

def select_and_prepare_features(df, target_col, date_col, id_cols_to_remove, features_to_exclude):
    """Selecciona características para el modelo."""
    cols_to_remove = list(set(id_cols_to_remove + features_to_exclude + [target_col] + [date_col]))
    feature_cols = [col for col in df.columns if col not in cols_to_remove]
    print(f"Características Seleccionadas ({len(feature_cols)}): {feature_cols}")
    return feature_cols

def prepare_xy(df, features, target_col, muni_enc_col, apply_log_transform):
    """Prepara X e y para entrenamiento, asegurando tipos correctos y aplicando transformación logarítmica."""
    print(f"Preparando X e y (features={len(features)})...")
    features_in_df = [f for f in features if f in df.columns]
    if len(features_in_df) != len(features):
        print(f"ADVERTENCIA: Features esperadas no encontradas en df: {set(features) - set(features_in_df)}")

    X = df[features_in_df].copy()
    y_original = df[target_col].copy() # Keep original y before transformation

    # Asegurar tipos numéricos (especialmente IDs) en X
    final_features = features_in_df.copy()
    cat_cols_to_check = [muni_enc_col, 'Service_Type_encoded', 'Is_Holiday_Month', 'Is_Alta_Inmediata', 'Is_Medellin']
    for col in cat_cols_to_check:
        if col in X.columns:
             if not pd.api.types.is_numeric_dtype(X[col]):
                 print(f" -> Convirtiendo '{col}' a numérico (int) en X...")
                 X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-1).astype(np.int64)

    # Handle potential non-numeric columns in X after selecting features
    non_numeric_cols_X = X.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric_cols_X:
         print(f"ADVERTENCIA: Columnas no numéricas en X: {non_numeric_cols_X}. Intentando convertir/eliminar.")
         for col in non_numeric_cols_X:
             try:
                 # Attempt conversion to numeric, coerce errors to NaN, fill NaN
                 X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
             except ValueError:
                 print(f"Eliminando columna no numérica no convertible: {col}");
                 X = X.drop(columns=[col])
                 if col in final_features:
                     final_features.remove(col)

    # Ensure the target variable is numeric
    if not pd.api.types.is_numeric_dtype(y_original):
         print(f"ADVERTENCIA: La columna objetivo '{target_col}' no es numérica. Intentando convertir.")
         y_original = pd.to_numeric(y_original, errors='coerce')

    # Rellenar NaN en y_original (si hay)
    if y_original.isna().any():
        median_val = np.nanmedian(y_original)
        print(f" -> Rellenando NaN en y_original con la mediana ({median_val})...")
        y_original = y_original.fillna(median_val)


    # Aplicar transformación logarítmica si es necesario
    if apply_log_transform:
        print("Aplicando transformación log1p al target...")
        # Apply log1p and ensure it remains a pandas Series
        y = np.log1p(y_original)
        # Ensure y is a pandas Series with the original index
        if not isinstance(y, pd.Series):
             y = pd.Series(y, index=y_original.index)
    else:
        y = y_original.copy() # Ensure y is a copy of the original series

    # Ensure y_original is also a pandas Series
    if not isinstance(y_original, pd.Series):
         y_original = pd.Series(y_original, index=df.index)


    # Asegurar que las columnas de X estén en un orden consistente (opcional pero buena práctica)
    final_features.sort()
    X = X[final_features]

    print(f" -> X shape: {X.shape}, y shape: {y.shape}, y_original shape: {y_original.shape}")
    print(f" -> NaN en X: {X.isna().sum().sum()}, NaN en y: {y.isna().sum()}, NaN en y_original: {y_original.isna().sum()}")


    return X, y, y_original, final_features
def train_evaluate_model(model_name, model, X_train, y_train, X_valid, y_valid_original, 
                        cat_indices=None, apply_log_transform=False, X_train_xgb=None, X_valid_xgb=None):
    """Entrena y evalúa un modelo individual con validación de NaN."""
    print(f"\n--- Entrenando {model_name} ---")
    
    if X_train.isna().any().any():
        print(f"WARNING in {model_name}: X_train contains NaN, cleaning...")
        X_train = X_train.fillna(0)
    
    if pd.isna(y_train).any():
        print(f"WARNING in {model_name}: y_train contains NaN, cleaning...")
        y_train = np.nan_to_num(y_train, nan=np.nanmedian(y_train))
    
    try:
        if model_name == 'XGBoost' and X_train_xgb is not None:
            if X_train_xgb.isna().any().any():
                print(f"WARNING: X_train_xgb contains NaN, cleaning...")
                X_train_xgb = X_train_xgb.fillna(0)
            
            model.fit(X_train_xgb, y_train, eval_set=[(X_train_xgb, y_train)], verbose=False)
            
            if X_valid_xgb.isna().any().any():
                print(f"WARNING: X_valid_xgb contains NaN, cleaning...")
                X_valid_xgb = X_valid_xgb.fillna(0)
                
            y_pred_transformed = model.predict(X_valid_xgb)
        
        elif model_name == 'LGBM':
            model.fit(X_train, y_train, categorical_feature=cat_indices)
            X_valid_clean = X_valid.fillna(0) if X_valid.isna().any().any() else X_valid
            y_pred_transformed = model.predict(X_valid_clean)
        
        else:
            model.fit(X_train, y_train)
            X_valid_clean = X_valid.fillna(0) if X_valid.isna().any().any() else X_valid
            y_pred_transformed = model.predict(X_valid_clean)
        
        y_pred_original = np.expm1(y_pred_transformed) if apply_log_transform else y_pred_transformed
        y_pred_original = np.maximum(0, y_pred_original)
        
        mae = mean_absolute_error(y_valid_original, y_pred_original)
        rmse = np.sqrt(mean_squared_error(y_valid_original, y_pred_original))
        r2 = r2_score(y_valid_original, y_pred_original)
        
        errors = y_pred_original - y_valid_original
        
        print(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        return model, y_pred_original, errors, mae, rmse, r2
    
    except Exception as e:
        print(f"Error entrenando {model_name}: {str(e)}")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_train NaN count: {X_train.isna().sum().sum()}")
        print(f"y_train NaN count: {pd.isna(y_train).sum()}")
        print(traceback.format_exc())
        return None, None, None, np.inf, np.inf, -np.inf

def prepare_prophet_data(df):
    """Prepara datos para Prophet."""
    prophet_df = df.groupby('Date')['Service_Count'].sum().reset_index()
    prophet_df.columns = ['ds', 'y']
    return prophet_df

def train_prophet(model, train_df, valid_df):
    """Entrena y evalúa Prophet."""
    print("Entrenando Prophet...")
    prophet_train = prepare_prophet_data(train_df)
    model.fit(prophet_train)
    future = prepare_prophet_data(valid_df)
    forecast = model.predict(future)
    preds = forecast['yhat'].values
    y_valid = valid_df.groupby('Date')['Service_Count'].sum().values
    mae = mean_absolute_error(y_valid, preds)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    r2 = r2_score(y_valid, preds)
    print(f"Prophet - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return preds, mae, rmse, r2

def prepare_lstm_data(df, look_back=12):
    """Prepara datos para LSTM."""
    df_agg = df.groupby('Date')['Service_Count'].sum().reset_index()
    data = df_agg['Service_Count'].values
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i])
        y.append(data[i])
    X, y = np.array(X), np.array(y)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, y_train, X_test, y_test

def train_lstm(X_train, y_train, X_test, y_test):
    """Entrena y evalúa LSTM."""
    print("Entrenando LSTM...")
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    preds = model.predict(X_test, verbose=0).flatten()
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"LSTM - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return preds, mae, rmse, r2

def weighted_ensemble_predictions(predictions_list, weights):
    """Calcula predicciones del ensamble ponderado."""
    return np.average(predictions_list, axis=0, weights=weights)

def optimize_ensemble_weights(predictions_list, y_true):
    """Optimiza pesos del ensamble con meta-learner."""
    meta_X = np.column_stack(predictions_list)
    meta_model = LinearRegression()
    meta_model.fit(meta_X, y_true)
    weights = meta_model.coef_
    weights = weights / np.sum(weights) if np.sum(weights) != 0 else np.ones(len(weights)) / len(weights)
    return weights

def bootstrap_predictions(model, X, n_iter=BOOTSTRAP_N_ITER, apply_log_transform=False):
    """Genera intervalos de predicción usando bootstrap."""
    print(f"Generando intervalos de predicción con bootstrap ({n_iter} iteraciones)...")
    predictions = []
    for _ in range(n_iter):
        indices = resample(np.arange(len(X)), replace=True)
        X_boot = X.iloc[indices]
        preds = model.predict(X_boot)
        if apply_log_transform:
            preds = np.expm1(preds)
        predictions.append(preds)
    predictions = np.array(predictions)
    lower = np.percentile(predictions, 2.5, axis=0)
    upper = np.percentile(predictions, 97.5, axis=0)
    return np.maximum(0, lower), np.maximum(0, upper)

def statistical_validation(model_errors, model_names, y_true):
    """Realiza validación estadística robusta."""
    print("\n--- Validación Estadística ---")
    results = {}
    min_len = min(len(errors) for errors in model_errors.values())
    aligned_errors = {name: errors[:min_len] for name, errors in model_errors.items()}
    y_true = y_true[:min_len]
    try:
        error_abs = [np.abs(aligned_errors[name]) for name in model_names if name in aligned_errors]
        if len(error_abs) > 1 and len(error_abs[0]) > 10:
            stat, p_value = friedmanchisquare(*error_abs)
            results['Friedman'] = {'statistic': float(stat), 'p_value': float(p_value)}
            print(f"Friedman Test: χ²={stat:.2f}, p={p_value:.4g}")
            if p_value < 0.05:
                print("  Diferencias significativas detectadas. Procediendo con Nemenyi post-hoc...")
                p_values = []
                comparisons = []
                for i in range(len(model_names)):
                    for j in range(i + 1, len(model_names)):
                        name1, name2 = model_names[i], model_names[j]
                        if name1 in aligned_errors and name2 in aligned_errors:
                            stat_w, p_w = wilcoxon(
                                np.abs(aligned_errors[name1]), np.abs(aligned_errors[name2]), alternative='two-sided'
                            )
                            p_values.append(p_w)
                            comparisons.append(f"{name1} vs {name2}")
                if p_values:
                    reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
                    results['Nemenyi'] = {
                        'comparisons': [
                            {'pair': comp, 'p_value': float(p_corr), 'significant': str(rej)}
                            for comp, p_corr, rej in zip(comparisons, p_corrected, reject)
                        ]
                    }
                    for comp, p_corr, rej in zip(comparisons, p_corrected, reject):
                        print(f"    {comp}: p={p_corr:.4g} {'(Significativo)' if rej else ''}")
            else:
                print("  No hay diferencias significativas entre modelos.")
        else:
            print("Friedman no aplicable (pocos modelos o datos insuficientes).")
    except Exception as e:
        print(f"Error en Friedman/Nemenyi: {e}")
        results['Friedman'] = {'error': str(e)}
    results['Residual_Analysis'] = {}
    for name in model_names:
        if name in aligned_errors:
            try:
                residuals = aligned_errors[name]
                stat_sh, p_sh = shapiro(residuals)
                results['Residual_Analysis'][name] = {
                    'Shapiro_Wilk': {'statistic': float(stat_sh), 'p_value': float(p_sh)}
                }
                print(f"\nResiduales {name}:")
                print(f"  Shapiro-Wilk (Normalidad): W={stat_sh:.2f}, p={p_sh:.4g}")
                if p_sh < 0.05:
                    print("    Residuos no normales (p < 0.05)")
                else:
                    print("    Residuos compatibles con normalidad")
            except Exception as e:
                print(f"Error en análisis de residuos para {name}: {e}")
                results['Residual_Analysis'][name] = {'error': str(e)}
    try:
        if len(error_abs) > 1:
            stat_lev, p_lev = levene(*[aligned_errors[name] for name in model_names if name in aligned_errors], center='median')
            results['Levene'] = {'statistic': float(stat_lev), 'p_value': float(p_lev)}
            print(f"\nLevene Test (Homogeneidad Varianzas): F={stat_lev:.2f}, p={p_lev:.4g}")
            if p_lev < 0.05:
                print("  Varianzas no homogéneas entre modelos")
            else:
                print("  Varianzas homogéneas")
    except Exception as e:
        print(f"Error en Levene: {e}")
        results['Levene'] = {'error': str(e)}
    return results

def plot_residual_analysis(model_errors, model_names):
    """Genera gráficos de análisis de residuos."""
    print("\n--- Análisis Gráfico de Residuos ---")
    plt.figure(figsize=(10, 6))
    data = [model_errors[name] for name in model_names if name in model_errors]
    plt.boxplot(data, labels=[name for name in model_names if name in model_errors])
    plt.title("Distribución de Residuos por Modelo")
    plt.ylabel("Residuos (Real - Predicción)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    from scipy.stats import probplot
    for name in model_names:
        if name in model_errors:
            plt.figure(figsize=(6, 6))
            probplot(model_errors[name], dist="norm", plot=plt)
            plt.title(f"Q-Q Plot Residuos ({name})")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

def plot_evaluation_graphs(validation_results_df, best_model_name, train_df_orig, valid_df_orig,
                          decoders_dict, muni_encoder_col, service_encoder_col, target_col, date_col,
                          error_series_best=None, X_train_q=None, y_train_q=None, X_valid_q=None, y_valid_q=None, cat_indices_q=None):
    """Genera gráficas de evaluación detalladas."""
    global decoders
    if not decoders_dict:
        decoders_dict = decoders
    print("\nGenerando gráficas de evaluación...")
    try:
        # Ensure datetime conversion for all relevant date columns
        validation_results_df[date_col] = pd.to_datetime(validation_results_df[date_col])
        train_df_orig[date_col] = pd.to_datetime(train_df_orig[date_col])
        valid_df_orig[date_col] = pd.to_datetime(valid_df_orig[date_col])
    except Exception as e:
        print(f"Error convirtiendo fechas: {e}. Omitiendo plots temporales.")
        return

    # 1. Scatter Plot (use the granular data)
    plt.figure(figsize=(7, 7))
    plt.scatter(validation_results_df['Actual'], validation_results_df['Predicted'], alpha=0.3, s=10)
    min_val = min(validation_results_df['Actual'].min(), validation_results_df['Predicted'].min())
    max_val = max(validation_results_df['Actual'].max(), validation_results_df['Predicted'].max())
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', label='Ideal (y=x)')
    plt.xlabel("Reales (Validación - Granular)")
    plt.ylabel("Predicciones (Granular)")
    plt.title(f'Reales vs. Predichos ({best_model_name}) - Granular')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. Aggregated Time Series Plot
    print("\nGraficando serie temporal agregada...")
    # Aggregate the validation results for the time series plot
    agg_validation_results = validation_results_df.groupby(date_col)[['Actual', 'Predicted']].sum().reset_index()

    # Aggregate original train and valid data for historical context
    agg_train_orig = train_df_orig.groupby(date_col)[target_col].sum().reset_index().rename(columns={target_col:'Actual'})
    agg_valid_orig = valid_df_orig.groupby(date_col)[target_col].sum().reset_index().rename(columns={target_col:'Actual'})

    # Combine original history for plotting
    agg_history = pd.concat([agg_train_orig, agg_valid_orig], ignore_index=True)

    # Select last 24 months of history + validation period for plot
    min_history_date = agg_valid_orig[date_col].min() - pd.DateOffset(months=23)
    last_24m_history = agg_history[agg_history[date_col] >= min_history_date]

    # Merge aggregated validation actuals and predictions
    agg_plot_data = pd.merge(agg_valid_orig, agg_validation_results[[date_col,'Predicted']], on=date_col, how='left')

    plt.figure(figsize=(15, 7))
    # Plot historical aggregated data
    plt.plot(last_24m_history[date_col], last_24m_history['Actual'], label='Real Histórica (Aggr.)', marker='.', color='gray', alpha=0.7)
    # Plot validation aggregated actuals
    plt.plot(agg_plot_data[date_col], agg_plot_data['Actual'], label='Real Validación (Aggr.)', marker='.', color='black')
    # Plot validation aggregated predictions
    plt.plot(agg_plot_data[date_col], agg_plot_data['Predicted'], label=f'Predicción Validación ({best_model_name} - Aggr.)', marker='.', linestyle='--', color='red')

    plt.xlabel("Fecha")
    plt.ylabel("Demanda Total Agregada")
    plt.title("Demanda Agregada: Historial vs. Predicción Validación")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 3. Example Time Series Plot... (remains largely the same, but filters from validation_results_df)
    print("Graficando serie ejemplo: Medellin - Ambulatoria...")
    try:
        # Cargar decodificadores si no se han proporcionado
        local_decoders = joblib.load('decoders.joblib') if not decoders_dict else decoders_dict

        med_amb_encoder = local_decoders.get(muni_encoder_col)
        med_amb_service_encoder = local_decoders.get(service_encoder_col)

        if med_amb_encoder and med_amb_service_encoder:
            # Obtener los códigos para Medellín y Ambulatoria
            med_code = next((k for k, v in med_amb_encoder.items() if v == 'MEDELLIN'), None)
            amb_code = next((k for k, v in med_amb_service_encoder.items() if v == 'AMBULATORIA'), None)

            if med_code is not None and amb_code is not None:
                # Use integer codes as type
                muni_key_type = int
                serv_key_type = int

                # Convert codes to the corresponding type
                med_code_typed = muni_key_type(med_code)
                amb_code_typed = serv_key_type(amb_code)

                # Ensure types in the datasets
                # This part is now done when validation_results_df is created,
                # but let's ensure for safety or if the function is called differently
                validation_results_df[muni_encoder_col] = pd.to_numeric(
                    validation_results_df[muni_encoder_col], errors='coerce'
                ).fillna(-1).astype(muni_key_type)

                validation_results_df[service_encoder_col] = pd.to_numeric(
                    validation_results_df[service_encoder_col], errors='coerce'
                ).fillna(-1).astype(serv_key_type)

                train_df_orig[muni_encoder_col] = pd.to_numeric(
                    train_df_orig[muni_encoder_col], errors='coerce'
                ).fillna(-1).astype(muni_key_type)

                train_df_orig[service_encoder_col] = pd.to_numeric(
                    train_df_orig[service_encoder_col], errors='coerce'
                ).fillna(-1).astype(serv_key_type)

                # Filter example data from the granular validation results DataFrame
                example_series_valid = validation_results_df[
                    (validation_results_df[muni_encoder_col] == med_code_typed) &
                    (validation_results_df[service_encoder_col] == amb_code_typed)
                ].copy()

                # Filter original train data for historical context of the example series
                example_series_train_orig = train_df_orig[
                    (train_df_orig[muni_encoder_col] == med_code_typed) &
                    (train_df_orig[service_encoder_col] == amb_code_typed)
                ].copy()

                # Combine history and validation for plotting the example series
                example_series_history_orig = pd.concat(
                    [
                        example_series_train_orig[[date_col, target_col]],
                        example_series_valid[[date_col, 'Actual']] # Use 'Actual' from validation_results_df
                    ],
                    ignore_index=True
                ).rename(columns={target_col: 'Actual'})

                # Select last 24 months for the example series plot
                last_24m_example_hist_orig = example_series_history_orig[
                    example_series_history_orig[date_col] >=
                    example_series_history_orig[date_col].max() - pd.DateOffset(months=23)
                ]

                if not example_series_valid.empty:
                    # Plot the example series
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

                    # Calculate MAE for the example series
                    mae_example = mean_absolute_error(
                        example_series_valid['Actual'],
                        example_series_valid['Predicted']
                    )
                    print(f" -> MAE Medellin-Ambulatoria: {mae_example:.4f}")
                else:
                    print("No datos de validación para Medellín/Ambulatoria.")
            else:
                print("No se encontraron códigos para Medellín y/o Ambulatoria.")
        else:
            print("Decodificadores no disponibles o incompletos.")

    except FileNotFoundError:
        print("Error gráfica ejemplo: 'decoders.joblib' no encontrado.")
    except Exception as e:
        print(f"Error gráfica ejemplo: {e}")

    # 4. Quantile Plot (aggregate the granular validation results)
    print("\nEntrenando modelos Quantile (LGBM)...");
    try:
         if X_train_q is not None and y_train_q is not None and X_valid_q is not None and y_valid_q is not None and cat_indices_q is not None:
             # ... (train/predict quantile models - same as v10) ...
             lgb_quantile_params = {'objective': 'quantile', 'metric': 'quantile', 'n_estimators': 800, 'learning_rate': 0.05, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'num_leaves': 31, 'max_depth': 10, 'min_child_samples': 30, 'n_jobs': -1, 'seed': 42}

             # Ensure X_train_q and y_train_q are clean for training
             if X_train_q.isna().any().any(): X_train_q = X_train_q.fillna(0); print("Quantile X_train_q NaN filled.")
             if pd.isna(y_train_q).any(): y_train_q = np.nan_to_num(y_train_q, nan=np.nanmedian(y_train_q)); print("Quantile y_train_q NaN filled.")

             lgbm_q05 = lgb.LGBMRegressor(**lgb_quantile_params, alpha=0.05)
             lgbm_q05.fit(X_train_q, y_train_q, eval_set=[(X_valid_q, y_valid_q)], callbacks=[lgb.early_stopping(50, verbose=False)], categorical_feature=cat_indices_q)
             preds_q05_t = lgbm_q05.predict(X_valid_q)

             lgbm_q95 = lgb.LGBMRegressor(**lgb_quantile_params, alpha=0.95)
             lgbm_q95.fit(X_train_q, y_train_q, eval_set=[(X_valid_q, y_valid_q)], callbacks=[lgb.early_stopping(50, verbose=False)], categorical_feature=cat_indices_q)
             preds_q95_t = lgbm_q95.predict(X_valid_q)

             preds_q05_orig = np.expm1(preds_q05_t) if APPLY_LOG_TRANSFORM else preds_q05_t
             preds_q95_orig = np.expm1(preds_q95_t) if APPLY_LOG_TRANSFORM else preds_q95_t

             # Add quantile predictions to the granular validation results DataFrame
             # Make sure the lengths match - if X_valid_q was sliced or filtered, need to align
             min_len_q = min(len(validation_results_df), len(preds_q05_orig), len(preds_q95_orig))
             validation_results_df['Pred_Q05'] = np.maximum(0, preds_q05_orig[:min_len_q])
             validation_results_df['Pred_Q95'] = np.maximum(validation_results_df['Pred_Q05'], preds_q95_orig[:min_len_q])

             # Aggregate the quantile predictions for plotting
             agg_results_q = validation_results_df.groupby(date_col)[['Actual', 'Predicted', 'Pred_Q05', 'Pred_Q95']].sum().reset_index()

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
         else: print("Advertencia: Datos Quantile no disponibles.")
    except Exception as e: print(f"Error modelos Quantile: {e}")

    # 5. Growth Rate Plot... (Uses original dfs, should be fine)
    print("\nGraficando Growth_Rate_MoM Histórico...");
    if 'Growth_Rate_MoM' in train_df_orig.columns and 'Growth_Rate_MoM' in valid_df_orig.columns: # Use original dfs
         growth_hist_df = pd.concat([ train_df_orig[[date_col, 'Growth_Rate_MoM']], valid_df_orig[[date_col, 'Growth_Rate_MoM']] ], ignore_index=True); growth_hist_df[date_col] = pd.to_datetime(growth_hist_df[date_col]); avg_growth = growth_hist_df.groupby(date_col)['Growth_Rate_MoM'].mean().reset_index(); last_24m_growth = avg_growth[avg_growth[date_col] >= avg_growth[date_col].max() - pd.DateOffset(months=23)];
         plt.figure(figsize=(15, 5)); plt.plot(last_24m_growth[date_col], last_24m_growth['Growth_Rate_MoM'], label='Tasa Crecimiento MoM Promedio', marker='.'); plt.axhline(0, color='red', linestyle='--', linewidth=0.8, label='Crecimiento Cero'); plt.xlabel("Fecha"); plt.ylabel("Tasa Crecimiento MoM Promedio"); plt.title("Historial Tasa Crecimiento Mensual Promedio"); plt.legend(); plt.grid(True); plt.xticks(rotation=45); plt.tight_layout(); plt.show()
    else: print("No se encontró 'Growth_Rate_MoM'.")

    # 6. Residual Plot... (uses the full error series, aggregates residuals for time series plot)
    print("\nGraficando Residuos Agregados en Validación...");
    if error_series_best is not None: # Use passed errors
         # Add errors to the granular DataFrame
         validation_results_df['Residual'] = error_series_best[:len(validation_results_df)] # Ensure length matches validation_results_df

         # Aggregate residuals for the time series plot
         agg_residuals = validation_results_df.groupby(date_col)['Residual'].mean().reset_index()

         plt.figure(figsize=(15, 5)); plt.plot(agg_residuals[date_col], agg_residuals['Residual'], label=f'Residuo Promedio ({best_model_name})', marker='.'); plt.axhline(0, color='red', linestyle='--', linewidth=0.8, label='Error Cero'); plt.xlabel("Fecha (Validación)"); plt.ylabel("Residuo Promedio (Real - Predicción)"); plt.title("Residuos Promedio Modelo en Validación"); plt.legend(); plt.grid(True); plt.xticks(rotation=45); plt.tight_layout(); plt.show()

         # Plot Error Distribution (uses the full error series)
         plt.figure(figsize=(10, 5)); sns.histplot(error_series_best, kde=True); plt.title(f'Distribución de Errores ({best_model_name}) - Validación'); plt.xlabel("Error (Real - Predicción)"); plt.grid(True); plt.show()

         # Boxplot of residuals (uses the full error series)
         plt.figure(figsize=(8, 5)); plt.boxplot(error_series_best, vert=False); plt.title(f"Boxplot de Residuos ({best_model_name})"); plt.xlabel("Residuos (Real - Predicción)"); plt.grid(True); plt.tight_layout(); plt.show()

         # Q-Q Plot of residuals (uses the full error series)
         from scipy.stats import probplot
         plt.figure(figsize=(6, 6)); probplot(error_series_best, dist="norm", plot=plt); plt.title(f"Q-Q Plot Residuos ({best_model_name})"); plt.grid(True); plt.tight_layout(); plt.show()

    else: print("No se pueden graficar residuos (errores no disponibles).")

    # 7. NEW: Plot Best/Worst Performing Series (uses the granular validation_results_df)
    print(f"\nGraficando {N_BEST_WORST_SERIES} mejores/peores series por MAE en validación...")
    if error_series_best is not None:
        # Add absolute errors to the granular DataFrame
        validation_results_df['AbsError'] = np.abs(error_series_best[:len(validation_results_df)]) # Ensure length matches

        local_decoders = joblib.load('decoders.joblib') if not decoders_dict else decoders_dict
        muni_decoder = local_decoders.get(muni_encoder_col)
        serv_decoder = local_decoders.get(service_encoder_col)

        if muni_decoder and serv_decoder:
             # Group the granular data by series and calculate MAE
             series_mae = validation_results_df.groupby([muni_encoder_col, service_encoder_col])['AbsError'].mean().sort_values()

             # Select series based on MAE
             worst_series = series_mae.tail(N_BEST_WORST_SERIES).index.tolist()
             # Filter out series with near-zero MAE before picking the "best"
             filtered_best_series = series_mae[series_mae > 0.05].head(N_BEST_WORST_SERIES).index.tolist()
             best_series = filtered_best_series

             print(f"\n--- {N_BEST_WORST_SERIES} Peores Series (Mayor MAE) ---")
             for i, (m_code, s_code) in enumerate(worst_series[::-1]): # Plot worst first
                 m_name = muni_decoder.get(int(m_code), f'Unknown_{m_code}')
                 s_name = serv_decoder.get(int(s_code), f'Unknown_{s_code}')
                 # Filter the granular data for this specific series
                 series_data = validation_results_df[(validation_results_df[muni_encoder_col] == int(m_code)) & (validation_results_df[service_encoder_col] == int(s_code))]
                 mae_val = series_mae.loc[(m_code, s_code)]
                 plt.figure(figsize=(12, 4))
                 plt.plot(series_data[date_col], series_data['Actual'], label='Real', marker='.')
                 plt.plot(series_data[date_col], series_data['Predicted'], label='Predicción', marker='.', linestyle='--')
                 plt.title(f"Peor Serie #{i+1}: {m_name} - {s_name} (MAE: {mae_val:.3f})")
                 plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

             print(f"\n--- {N_BEST_WORST_SERIES} Mejores Series (Menor MAE) ---")
             for i, (m_code, s_code) in enumerate(best_series):
                 m_name = muni_decoder.get(int(m_code), f'Unknown_{m_code}')
                 s_name = serv_decoder.get(int(s_code), f'Unknown_{s_code}')
                 # Filter the granular data for this specific series
                 series_data = validation_results_df[(validation_results_df[muni_encoder_col] == int(m_code)) & (validation_results_df[service_encoder_col] == int(s_code))]
                 mae_val = series_mae.loc[(m_code, s_code)]
                 plt.figure(figsize=(12, 4))
                 plt.plot(series_data[date_col], series_data['Actual'], label='Real', marker='.')
                 plt.plot(series_data[date_col], series_data['Predicted'], label='Predicción', marker='.', linestyle='--')
                 plt.title(f"Mejor Serie #{i+1}: {m_name} - {s_name} (MAE: {mae_val:.3f})")
                 plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
        else: print("No se pudieron cargar decoders para graficar mejores/peores series.")
    else: print("Errores no disponibles para análisis de mejores/peores series.")
    
def generate_future_features(current_date, unique_combos, history_df, model_features, hist_target_col, apply_log_transform, muni_enc_col, serv_enc_col, date_col, lag_cache=None):
    """Genera features para un mes futuro con caché para lags."""
    if lag_cache is None:
        lag_cache = {}
    municipality_encoded_col = muni_enc_col
    service_type_encoded_col = serv_enc_col
    current_future_df = unique_combos.copy()
    current_future_df[date_col] = current_date
    current_future_df[municipality_encoded_col] = current_future_df[municipality_encoded_col].astype(int)
    current_future_df[service_type_encoded_col] = current_future_df[service_type_encoded_col].astype(int)
    date_features_in_model = ['Year_Fraction', 'Month_sin', 'Month_cos', 'Quarter_sin', 'Quarter_cos', 'Quarter', 'Month_of_Year', 'Is_Holiday_Month', 'sin_annual', 'cos_annual']
    for feat in date_features_in_model:
        if feat in model_features:
            try:
                if feat == 'Year_Fraction':
                    current_future_df[feat] = current_date.dayofyear / (366 if current_date.is_leap_year else 365)
                elif feat == 'Month_sin':
                    current_future_df[feat] = np.sin(2 * np.pi * current_date.month / 12).round(6)
                elif feat == 'Month_cos':
                    current_future_df[feat] = np.cos(2 * np.pi * current_date.month / 12).round(6)
                elif feat == 'Quarter_sin':
                    current_future_df[feat] = np.sin(2 * np.pi * current_date.quarter / 4).round(6)
                elif feat == 'Quarter_cos':
                    current_future_df[feat] = np.cos(2 * np.pi * current_date.quarter / 4).round(6)
                elif feat == 'Quarter':
                    current_future_df[feat] = current_date.quarter
                elif feat == 'Month_of_Year':
                    current_future_df[feat] = current_date.month
                elif feat == 'Is_Holiday_Month':
                    current_future_df[feat] = int(current_date.month in [1, 7, 12])
                elif feat == 'sin_annual':
                    current_future_df[feat] = np.sin(2 * np.pi * current_date.month / 12)
                elif feat == 'cos_annual':
                    current_future_df[feat] = np.cos(2 * np.pi * current_date.month / 12)
            except Exception as e:
                print(f"    Error {feat}: {e}")
    lag_features_in_model = [f for f in model_features if 'Service_Count_lag_' in f]
    lag_amounts_needed = sorted(list(set([int(re.search(r'\d+$', f).group()) for f in lag_features_in_model] + [1, 2, 13, 24])), reverse=True)
    current_future_df_indexed = current_future_df.set_index([municipality_encoded_col, service_type_encoded_col])
    history_df[municipality_encoded_col] = pd.to_numeric(history_df[municipality_encoded_col], errors='coerce').fillna(-1).astype(int)
    history_df[service_type_encoded_col] = pd.to_numeric(history_df[service_type_encoded_col], errors='coerce').fillna(-1).astype(int)
    history_indexed = history_df.set_index([date_col, municipality_encoded_col, service_type_encoded_col])
    temp_lags = {}
    for lag in lag_amounts_needed:
        lag_date = current_date - pd.DateOffset(months=lag)
        cache_key = (lag_date.strftime('%Y-%m'), lag)
        if cache_key in lag_cache:
            mapped_lags_transformed = lag_cache[cache_key]
        else:
            try:
                lag_values_transformed = history_indexed.loc[lag_date][hist_target_col]
                mapped_lags_transformed = current_future_df_indexed.index.map(lag_values_transformed)
            except KeyError:
                mapped_lags_transformed = pd.Series(0.0, index=current_future_df_indexed.index)
            lag_cache[cache_key] = mapped_lags_transformed
        temp_lags[f'target_lag_{lag}'] = mapped_lags_transformed.fillna(0.0)
        lag_col_name_model = f'Service_Count_lag_{lag}'
        if lag_col_name_model in model_features:
            mapped_lags_original = np.expm1(mapped_lags_transformed) if apply_log_transform else mapped_lags_transformed
            current_future_df[lag_col_name_model] = mapped_lags_original.fillna(0.0).values
    if isinstance(current_future_df.index, pd.MultiIndex):
        current_future_df = current_future_df.reset_index()
    rolling_features_in_model = [f for f in model_features if 'Service_Count_rolling_' in f]
    history_for_rolling = history_df[history_df[date_col] < current_date]
    if not history_for_rolling.empty:
        history_for_rolling[municipality_encoded_col] = pd.to_numeric(history_for_rolling[municipality_encoded_col], errors='coerce').fillna(-1).astype(int)
        history_for_rolling[service_type_encoded_col] = pd.to_numeric(history_for_rolling[service_type_encoded_col], errors='coerce').fillna(-1).astype(int)
        current_future_df[municipality_encoded_col] = current_future_df[municipality_encoded_col].astype(int)
        current_future_df[service_type_encoded_col] = current_future_df[service_type_encoded_col].astype(int)
        for roll_feat in rolling_features_in_model:
            try:
                parts = roll_feat.split('_')
                agg_func = parts[-2]
                window_size = int(parts[-1])
                rolling_calc = history_for_rolling.groupby([municipality_encoded_col, service_type_encoded_col])[hist_target_col].rolling(
                    window=window_size, min_periods=max(1, window_size // 2), closed='left'
                ).agg(agg_func)
                last_rolling_vals = rolling_calc.groupby(level=[0, 1]).last()
                rolling_vals_mapped = current_future_df.set_index([municipality_encoded_col, service_type_encoded_col]).index.map(last_rolling_vals)
                if apply_log_transform and ('mean' in agg_func):
                    current_future_df[roll_feat] = np.expm1(rolling_vals_mapped)
                elif apply_log_transform and ('std' in agg_func):
                    current_future_df[roll_feat] = rolling_vals_mapped
                else:
                    current_future_df[roll_feat] = rolling_vals_mapped
                current_future_df[roll_feat] = current_future_df[roll_feat].fillna(0.0)
            except Exception as e:
                print(f"    Error rolling {roll_feat}: {e}")
                current_future_df[roll_feat] = 0.0
    else:
        for roll_feat in rolling_features_in_model:
            current_future_df[roll_feat] = 0.0
    if isinstance(current_future_df.index, pd.MultiIndex):
        current_future_df = current_future_df.reset_index()
    try:
        if 'Growth_Rate_MoM' in model_features:
            lag1 = current_future_df['Service_Count_lag_1']
            lag2 = current_future_df['Service_Count_lag_2']
            ratio_mom = np.where(lag2 != 0, lag1 / lag2, np.nan)
            current_future_df['Growth_Rate_MoM'] = pd.Series(ratio_mom).replace([np.inf, -np.inf], np.nan).fillna(0) - 1
        if 'Growth_Rate_YoY' in model_features:
            lag1 = current_future_df['Service_Count_lag_1']
            lag13 = current_future_df.get('Service_Count_lag_12')
            if lag13 is not None:
                ratio_yoy = np.where(lag13 != 0, lag1 / lag13, np.nan)
                current_future_df['Growth_Rate_YoY'] = pd.Series(ratio_yoy).replace([np.inf, -np.inf], np.nan).fillna(0) - 1
            else:
                current_future_df['Growth_Rate_YoY'] = 0
    except Exception as e:
        print(f"    Error GrowthRate: {e}")
        current_future_df['Growth_Rate_MoM'] = 0
        current_future_df['Growth_Rate_YoY'] = 0
    other_features_in_model = [
        f for f in model_features
        if f not in date_features_in_model and f not in lag_features_in_model and f not in rolling_features_in_model
        and f not in ['Growth_Rate_MoM', 'Growth_Rate_YoY'] and f not in [municipality_encoded_col, service_type_encoded_col]
    ]
    if other_features_in_model:
        history_for_prop = history_df[history_df[date_col] < current_date]
        if not history_for_prop.empty:
            history_for_prop[municipality_encoded_col] = pd.to_numeric(history_for_prop[municipality_encoded_col], errors='coerce').fillna(-1).astype(int)
            history_for_prop[service_type_encoded_col] = pd.to_numeric(history_for_prop[service_type_encoded_col], errors='coerce').fillna(-1).astype(int)
            current_future_df[municipality_encoded_col] = current_future_df[municipality_encoded_col].astype(int)
            current_future_df[service_type_encoded_col] = current_future_df[service_type_encoded_col].astype(int)
            last_known_indices = history_for_prop.groupby([municipality_encoded_col, service_type_encoded_col])[date_col].idxmax()
            valid_indices = last_known_indices.dropna()
            if not valid_indices.empty:
                last_known_other = history_for_prop.loc[valid_indices]
                cols_to_select = [municipality_encoded_col, service_type_encoded_col] + [f for f in other_features_in_model if f in last_known_other.columns]
                last_known_other = last_known_other[cols_to_select]
                current_future_df = pd.merge(
                    current_future_df.drop(columns=[c for c in other_features_in_model if c in current_future_df.columns], errors='ignore'),
                    last_known_other, on=[municipality_encoded_col, service_type_encoded_col], how='left'
                )
            else:
                for f in other_features_in_model:
                    current_future_df[f] = history_df[f].mean() if f in history_df.columns else 0.0
        else:
            for f in other_features_in_model:
                current_future_df[f] = history_df[f].mean() if f in history_df.columns else 0.0
    current_future_df = current_future_df.fillna(0)
    for col in model_features:
        if col not in [date_col] and col not in current_future_df.columns:
            current_future_df[col] = history_df[col].mean() if col in history_df.columns else 0
        if col not in [date_col, municipality_encoded_col, service_type_encoded_col]:
            if not pd.api.types.is_numeric_dtype(current_future_df[col]):
                current_future_df[col] = pd.to_numeric(current_future_df[col], errors='coerce').fillna(0)
    current_future_df[municipality_encoded_col] = current_future_df[municipality_encoded_col].astype(int)
    current_future_df[service_type_encoded_col] = current_future_df[service_type_encoded_col].astype(int)
    missing_cols_final = set(model_features) - set(current_future_df.columns)
    if missing_cols_final:
        print(f"ERROR CRÍTICO: Faltan cols {missing_cols_final} ANTES de predecir {current_date:%Y-%m}")
        exit()
    current_X = current_future_df[model_features]
    return current_future_df, current_X, lag_cache
    
def plot_feature_importance(models_dict, feature_list, X_train_xgb_cols=None):
    """Genera gráficas de importancia para modelos individuales."""
    print("\n--- Importancia de Características (Modelos Individuales) ---")
    for model_name, model_obj in models_dict.items():
        if model_name != 'Ensemble' and model_obj is not None:
            try:
                print(f"\nImportancia para {model_name}...")
                plt.figure(figsize=(10, max(8, len(feature_list)//3)))
                model_features_list = feature_list
                if isinstance(model_obj, lgb.LGBMRegressor):
                    lgb.plot_importance(model_obj, max_num_features=min(25, len(model_features_list)), height=0.8, importance_type='gain')
                    plt.title(f'Importancia ({model_name} - Gain)')
                elif isinstance(model_obj, xgb.XGBRegressor) and X_train_xgb_cols is not None:
                    sanitized_feature_names = X_train_xgb_cols
                    importances = model_obj.feature_importances_
                    name_map = dict(zip(sanitized_feature_names, model_features_list))
                    f_scores = pd.Series(importances, index=sanitized_feature_names).sort_values(ascending=False)
                    top_features = f_scores[:min(25, len(f_scores))]
                    top_features.sort_values(ascending=True).plot(kind='barh')
                    plt.yticks(plt.yticks()[0], [name_map.get(label.get_text(), label.get_text()) for label in plt.gca().get_yticklabels()])
                    plt.xlabel("Importancia")
                    plt.title(f'Importancia ({model_name})')
                elif isinstance(model_obj, (RandomForestRegressor, GradientBoostingRegressor)):
                    importances = model_obj.feature_importances_
                    indices = np.argsort(importances)[-min(25, len(model_features_list)):]
                    plt.barh(range(len(indices)), importances[indices], align='center')
                    plt.yticks(range(len(indices)), [model_features_list[i] for i in indices])
                    plt.xlabel('Importancia (Impurity Decrease)')
                    plt.title(f'Importancia ({model_name})')
                plt.tight_layout()
                plt.show()
                print(f"Gráfica generada para {model_name}.")
            except Exception as e:
                print(f"Error graficando importancia para {model_name}: {e}")

def analyze_errors_by_group(validation_df, errors, group_cols):
    """Calcula métrica de error agrupada por columnas especificadas."""
    temp_df = validation_df.copy()
    temp_df['Error'] = np.abs(errors)
    grouped_errors = temp_df.groupby(group_cols)['Error'].mean().sort_values(ascending=False)
    grouped_errors = grouped_errors.reset_index()
    grouped_errors.columns = group_cols + ['MAE']
    return grouped_errors

def make_json_serializable(obj):
    """Convierte objetos no serializables a formatos JSON compatibles."""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, bool):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return make_json_serializable(obj.to_dict())
    return obj

def optimize_ensemble_weights_constrained(predictions_list, y_true, non_negative=True):
    """
    Optimiza los pesos del ensamble minimizando el MAE, con opción de pesos no negativos.
    
    Args:
        predictions_list (list of np.ndarray): Lista de predicciones de cada modelo.
        y_true (np.ndarray): Valores reales.
        non_negative (bool): Si True, fuerza que los pesos sean no negativos.
    
    Returns:
        np.ndarray: Pesos optimizados.
    """
    from scipy.optimize import minimize
    
    # Asegurarnos de que las predicciones y y_true tengan la misma longitud
    min_len = min(len(p) for p in predictions_list)
    predictions_list = [p[:min_len] for p in predictions_list]
    y_true = y_true[:min_len]
    
    # Número de modelos
    n_models = len(predictions_list)
    
    # Función objetivo: minimizar MAE
    def objective(weights):
        ensemble_pred = np.sum([w * pred for w, pred in zip(weights, predictions_list)], axis=0)
        return mean_absolute_error(y_true, ensemble_pred)
    
    # Restricción: suma de pesos = 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    # Límites: si non_negative=True, pesos >= 0
    if non_negative:
        bounds = [(0, 1)] * n_models
    else:
        bounds = [(None, None)] * n_models
    
    # Inicialización: pesos uniformes
    initial_weights = np.ones(n_models) / n_models
    
    # Optimización
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'disp': False}
    )
    
    if not result.success:
        print("Advertencia: La optimización de pesos del ensamble no convergió.")
        return initial_weights
    
    return result.x

def main():
    """Función principal con mejoras implementadas para optimización de agregados."""
    global tuning_time, best_rf_params, decoders
    start_total_time = time.time()
    APPLY_LOG_TRANSFORM = False  # Ajustado según tu configuración
    DO_TUNING = True

    # --- Paso 1: Carga, Preproc Básico ---
    train_df_orig, valid_df_orig = load_data('healthcare_train_data.csv', 'healthcare_valid_data.csv')
    train_df = train_df_orig.copy()
    valid_df = valid_df_orig.copy()
    print(f"Formas: Train={train_df.shape}, Valid={valid_df.shape}")
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    valid_df['Date'] = pd.to_datetime(valid_df['Date'])

    target_col = 'Service_Count'
    date_col = 'Date'
    muni_enc_col = 'Municipality_encoded'
    serv_enc_col = 'Service_Type_encoded'
    muni_orig_col = 'Municipality'
    serv_orig_col = 'Service_Type'

    # Feature Engineering

    for df in [train_df, valid_df]:
        df['Is_Alta_Inmediata'] = (df['Service_Type'] == 'ALTA INMEDIATA').astype(int) if 'Service_Type' in df.columns and 'ALTA INMEDIATA' in df['Service_Type'].values else 0
        df['Is_Medellin'] = (df['Municipality'] == 'MEDELLIN').astype(int) if 'Municipality' in df.columns and 'MEDELLIN' in df['Municipality'].values else 0


    for df in [train_df, valid_df]:
        # Interaction terms
        df['Municipality_Service_lag1'] = df['Municipality_encoded'] * df['Service_Count_lag_1']
        df['Medellin_Growth'] = df['Is_Medellin'] * df['Growth_Rate_MoM']
        
        # Polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(df[['Days_Since_First_Service', 'Month_Sequence']])
        poly_cols = [f'poly_{i}' for i in range(poly_features.shape[1])]
        df[poly_cols] = poly_features
        
        # Seasonal dummies
        df = pd.get_dummies(df, columns=['Month_of_Year'], prefix='Month')

        # Agregar más lags para capturar estacionalidad a largo plazo
        df['Service_Count_lag_24'] = df['Service_Count'].shift(24)


    create_decoders(train_df, muni_enc_col, muni_orig_col, serv_enc_col, serv_orig_col)

    # Outlier handling: Cap con percentil más amplio (0.005, 0.995)
    train_df = detect_and_handle_outliers(train_df, target_col, winsorize_bounds=(0.005, 0.995))
    valid_df = detect_and_handle_outliers(valid_df, target_col, winsorize_bounds=(0.005, 0.995))

    features_to_exclude = ['Days_From_Now', 'Is_Anomaly', 'Is_Work_Related'] + [
        'Total_Max_Cantidad_Municipio', 'Avg_Max_Cantidad_Municipio',
        'Std_Max_Cantidad_Municipio', 'Provider_Count_Municipio'
    ]
    id_cols_to_remove = [muni_orig_col, serv_orig_col, 'Year_Month', 'Year', 'Month',
                         'First_Service_Date', 'Last_Service_Date', 'First_Service_Ever']
    features = select_and_prepare_features(train_df, target_col, date_col, id_cols_to_remove, features_to_exclude)

    # Preparar datos
    X_train, y_train, _, features = prepare_xy(train_df, features, target_col, muni_enc_col, APPLY_LOG_TRANSFORM)
    X_valid, y_valid, y_valid_original, features = prepare_xy(valid_df, features, target_col, muni_enc_col, APPLY_LOG_TRANSFORM)

    print(f"\nValidación de datos después de prepare_xy:")
    print(f"X_train shape: {X_train.shape}, NaN count: {X_train.isna().sum().sum()}")
    print(f"X_valid shape: {X_valid.shape}, NaN count: {X_valid.isna().sum().sum()}")
    print(f"y_train NaN count: {pd.isna(y_train).sum()}")
    print(f"y_valid NaN count: {pd.isna(y_valid).sum()}")
    
    if X_train.isna().any().any() or X_valid.isna().any().any():
        print("WARNING: Found NaN values after prepare_xy, cleaning again...")
        X_train = X_train.fillna(0)
        X_valid = X_valid.fillna(0)
    
    if pd.isna(y_train).any():
        print("WARNING: Found NaN in y_train, replacing with median")
        y_train_median = np.nanmedian(y_train)
        y_train = np.nan_to_num(y_train, nan=y_train_median)
    
    if pd.isna(y_valid).any():
        print("WARNING: Found NaN in y_valid, replacing with median")
        y_valid_median = np.nanmedian(y_valid)
        y_valid = np.nan_to_num(y_valid, nan=y_valid_median)

    cat_feature_names = [muni_enc_col, serv_enc_col, 'Is_Holiday_Month', 'Is_Alta_Inmediata', 'Is_Medellin']
    cat_indices = sorted([i for i, col in enumerate(features) if col in cat_feature_names])
    print(f"Categorical features: {cat_feature_names}, Indices: {cat_indices}")

    # Submodelos
    submodels = {}
    for subset, mask_col, mask_value, min_samples in [
        ('Alta_Inmediata', 'Is_Alta_Inmediata', 1, 50),
        ('Medellin', 'Is_Medellin', 1, 50)
    ]:
        mask_train = X_train[mask_col] == mask_value
        mask_valid = X_valid[mask_col] == mask_value
        if mask_train.sum() > min_samples:
            print(f"\nEntrenando submodelo para {subset}...")
            X_train_subset = X_train[mask_train].copy()
            y_train_subset = y_train[mask_train].copy()
            
            if X_train_subset.isna().any().any():
                print(f"WARNING: NaN found in X_train_subset for {subset}, cleaning...")
                X_train_subset = X_train_subset.fillna(0)
            
            if pd.isna(y_train_subset).any():
                print(f"WARNING: NaN found in y_train_subset for {subset}, cleaning...")
                y_train_subset = np.nan_to_num(y_train_subset, nan=np.nanmedian(y_train_subset))
            
            submodel = GradientBoostingRegressor(
                n_estimators=1500, learning_rate=0.01, max_depth=7, min_samples_split=10,
                min_samples_leaf=5, subsample=0.8, random_state=42
            )

            try:
                submodel.fit(X_train_subset, y_train_subset)
                
                X_valid_subset = X_valid[mask_valid].copy()
                if X_valid_subset.isna().any().any():
                    print(f"WARNING: NaN found in X_valid_subset for {subset}, cleaning...")
                    X_valid_subset = X_valid_subset.fillna(0)
                
                preds = submodel.predict(X_valid_subset)
                mae = mean_absolute_error(y_valid_original[mask_valid], preds)
                print(f"Submodelo {subset} - MAE: {mae:.4f}")
                submodels[subset] = submodel
                
            except Exception as e:
                print(f"Error entrenando submodelo {subset}: {str(e)}")
                print(f"X_train_subset shape: {X_train_subset.shape}")
                print(f"X_train_subset NaN count: {X_train_subset.isna().sum().sum()}")
                print(f"y_train_subset NaN count: {pd.isna(y_train_subset).sum()}")
        else:
            print(f"Insuficientes muestras para {subset}: {mask_train.sum()}")

    # Modelos principales
    best_xgb_params = {
        'objective': 'reg:squarederror', 'eval_metric': ['mae', 'rmse'], 'n_jobs': -1,
        'seed': 42, 'tree_method': 'hist',
        'alpha': 0.07763399448000508,
        'colsample_bytree': 0.8987566853061946,
        'gamma': 0.09351332282682329,
        'lambda': 1.0200680211778108,
        'learning_rate': 0.03733551396716398,
        'max_depth': 4,
        'n_estimators': 976,
        'subsample': 0.9908753883293675
    }

    models = {}
    predictions_valid = {}
    mae_scores = {}
    rmse_scores = {}
    r2_scores = {}
    model_errors_valid = {}

    # LGBM
    lgbm_model = lgb.LGBMRegressor(
        objective='regression_l1', metric=['mae', 'rmse'], n_estimators=2000,
        learning_rate=0.02, feature_fraction=0.7, bagging_fraction=0.7,
        num_leaves=20, max_depth=10, min_child_samples=50,
        lambda_l1=0.2, lambda_l2=0.2, verbose=-1, n_jobs=-1, seed=42
    )
    models['LGBM'], predictions_valid['LGBM'], model_errors_valid['LGBM'], mae_scores['LGBM'], rmse_scores['LGBM'], r2_scores['LGBM'] = train_evaluate_model(
        'LGBM', lgbm_model, X_train, y_train, X_valid, y_valid_original, cat_indices, APPLY_LOG_TRANSFORM
    )

    # XGBoost
    X_train_xgb = X_train.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', str(x)))
    X_valid_xgb = X_valid.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', str(x)))
    xgb_model = xgb.XGBRegressor(**best_xgb_params)
    models['XGBoost'], predictions_valid['XGBoost'], model_errors_valid['XGBoost'], mae_scores['XGBoost'], rmse_scores['XGBoost'], r2_scores['XGBoost'] = train_evaluate_model(
        'XGBoost', xgb_model, X_train, y_train, X_valid, y_valid_original, cat_indices, APPLY_LOG_TRANSFORM,
        X_train_xgb=X_train_xgb, X_valid_xgb=X_valid_xgb
    )

    # RandomForest
    rf_model = RandomForestRegressor(**best_rf_params)
    models['RandomForest'], predictions_valid['RandomForest'], model_errors_valid['RandomForest'], mae_scores['RandomForest'], rmse_scores['RandomForest'], r2_scores['RandomForest'] = train_evaluate_model(
        'RandomForest', rf_model, X_train, y_train, X_valid, y_valid_original, cat_indices, APPLY_LOG_TRANSFORM
    )

    # GradientBoosting
    gb_model = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=7,
        min_samples_split=10, min_samples_leaf=5, subsample=0.8,
        random_state=42
    )
    models['GradientBoosting'], predictions_valid['GradientBoosting'], model_errors_valid['GradientBoosting'], mae_scores['GradientBoosting'], rmse_scores['GradientBoosting'], r2_scores['GradientBoosting'] = train_evaluate_model(
        'GradientBoosting', gb_model, X_train, y_train, X_valid, y_valid_original, cat_indices, APPLY_LOG_TRANSFORM
    )

    # Ensamble optimizado para predicciones agregadas
    print("\n--- Creando Ensamble Optimizado para Agregados ---")
    model_names = ['LGBM', 'XGBoost', 'RandomForest', 'GradientBoosting']
    valid_preds_list = [predictions_valid[name] for name in model_names if name in predictions_valid and predictions_valid[name] is not None]
    valid_model_names = [name for name in model_names if name in predictions_valid and predictions_valid[name] is not None]

    if len(valid_preds_list) >= 2:
        # Agregar predicciones por fecha para optimización
        dates = valid_df_orig[date_col].unique()
        # Create a pandas Series from y_valid_original with Date as index for aggregation
        y_valid_series = pd.Series(y_valid_original, index=valid_df_orig[date_col])
        y_valid_agg = y_valid_series.groupby(level=0).sum().reindex(dates).fillna(0) # Group by the Date index
        aligned_preds_agg = {}
        for name in valid_model_names:
            pred_df = pd.DataFrame({'Date': valid_df_orig[date_col], 'Pred': predictions_valid[name]})
            aligned_preds_agg[name] = pred_df.groupby('Date')['Pred'].sum().reindex(dates).fillna(0)

        # Incluir submodelos agregados
        sub_preds_agg = {}
        for subset in submodels:
            mask_valid = valid_df['Is_Medellin' if subset == 'Medellin' else 'Is_Alta_Inmediata'] == 1
            if mask_valid.any():
                sub_preds = submodels[subset].predict(X_valid[mask_valid])
                sub_pred_df = pd.DataFrame({
                    'Date': valid_df_orig[date_col][mask_valid],
                    'Pred': np.expm1(sub_preds) if APPLY_LOG_TRANSFORM else sub_preds
                })
                sub_preds_agg[f'Sub_{subset}'] = sub_pred_df.groupby('Date')['Pred'].sum().reindex(dates).fillna(0)
                valid_model_names.append(f'Sub_{subset}')
                aligned_preds_agg[f'Sub_{subset}'] = sub_preds_agg[f'Sub_{subset}']

        # Optimizar pesos para predicciones agregadas
        aligned_preds_agg_list = [aligned_preds_agg[name].values for name in valid_model_names]
        weights = optimize_ensemble_weights_constrained(aligned_preds_agg_list, y_valid_agg.values, non_negative=True)
        ensemble_preds_agg = weighted_ensemble_predictions(aligned_preds_agg_list, weights)

        # Calcular métricas agregadas
        mae = mean_absolute_error(y_valid_agg, ensemble_preds_agg)
        rmse = np.sqrt(mean_squared_error(y_valid_agg, ensemble_preds_agg))
        r2 = r2_score(y_valid_agg, ensemble_preds_agg)
        print(f"Ensamble Optimizado - MAE: {mae:.4f}, R2: {r2:.4f}")
        print(f"Pesos del Ensamble: {dict(zip(valid_model_names, weights))}")

        models['Ensemble'] = {'weights': weights, 'models': valid_model_names}
        predictions_valid['Ensemble'] = ensemble_preds_agg
        model_errors_valid['Ensemble'] = ensemble_preds_agg - y_valid_agg.values
        mae_scores['Ensemble'] = mae
        rmse_scores['Ensemble'] = rmse
        r2_scores['Ensemble'] = r2

        print("\nDebug: Sample predictions (Ensemble):")
        sample_df = pd.DataFrame({
            'Date': y_valid_agg.index,
            'Actual': y_valid_agg.values,
            'Predicted': ensemble_preds_agg
        })
        print(sample_df.head(10))
    else:
        print("No suficientes modelos válidos para ensamble.")

    # Selección del mejor modelo
    print("\n--- Paso 3: Selección Mejor Modelo y Validación Estadística ---")
    best_model_name = min(mae_scores, key=mae_scores.get)
    print(f"Mejor enfoque según MAE: {best_model_name} (MAE: {mae_scores[best_model_name]:.4f}, R2: {r2_scores.get(best_model_name, 'N/A'):.4f})")
    stat_results = statistical_validation(model_errors_valid, model_names + ['Ensemble'], y_valid_agg)
    plot_residual_analysis(model_errors_valid, model_names + ['Ensemble'])

   # Pass the granular validation data and predictions to the plotting function
    if best_model_name and best_model_name in predictions_valid and predictions_valid[best_model_name] is not None:
        validation_plot_df = valid_df_orig.copy() # Start with original valid df
        validation_plot_df['Actual'] = y_valid_original # Add original actuals
        validation_plot_df['Predicted'] = predictions_valid[best_model_name] # Add granular predictions

        if best_model_name == 'Ensemble':
            # Need granular ensemble predictions for plotting.
            # Recalculate granular ensemble predictions using the weights determined for aggregated predictions
            valid_preds_list_granular = [predictions_valid[name] for name in model_names if name in predictions_valid and predictions_valid[name] is not None]
            valid_model_names_granular = [name for name in model_names if name in predictions_valid and predictions_valid[name] is not None]
            
            # Include granular submodel predictions
            for subset in submodels:
                  mask_valid = valid_df['Is_Medellin' if subset == 'Medellin' else 'Is_Alta_Inmediata'] == 1
                  if mask_valid.any():
                      sub_pred = np.zeros(len(valid_df))
                      sub_pred_transformed = submodels[subset].predict(X_valid[mask_valid])
                      sub_pred[mask_valid] = np.expm1(sub_pred_transformed) if APPLY_LOG_TRANSFORM else sub_pred_transformed
                      valid_preds_list_granular.append(sub_pred)
                      valid_model_names_granular.append(f'Sub_{subset}')
            
            # Apply the previously optimized weights (calculated on aggregated data) to granular predictions
            # Ensure weights and granular predictions lists are aligned by model name
            aligned_granular_preds = []
            aligned_weights = []
            weight_dict = dict(zip(valid_model_names, models['Ensemble']['weights'])) # Use weights from aggregated optimization
            
            # Match the order used when creating valid_preds_list_granular/valid_model_names_granular
            for name in valid_model_names_granular:
                # Find the corresponding prediction in valid_preds_list_granular
                # This assumes the order matches - a safer way would be to store granular preds in a dict by name
                # Let's reconstruct valid_preds_list_granular as a dict first
                preds_granular_dict = {}
                for name_g, preds_g in zip(valid_model_names, valid_preds_list): # Main models
                    preds_granular_dict[name_g] = preds_g
                for name_s, preds_s in sub_preds_agg.items(): # Submodels (Need granular sub_preds, not aggregated ones)
                    mask_valid = valid_df['Is_Medellin' if name_s == 'Sub_Medellin' else 'Is_Alta_Inmediata'] == 1
                    if mask_valid.any():
                        sub_pred_transformed = submodels[name_s.replace('Sub_','')].predict(X_valid[mask_valid])
                        sub_pred_granular = np.zeros(len(valid_df))
                        sub_pred_granular[mask_valid] = np.expm1(sub_pred_transformed) if APPLY_LOG_TRANSFORM else sub_pred_transformed
                        preds_granular_dict[name_s] = sub_pred_granular

            # Now build aligned lists using the names from the aggregated weight optimization (valid_model_names)
            # This ensures the weights align with the correct model's predictions
            aligned_granular_preds_for_ensemble = [preds_granular_dict[name] for name in valid_model_names if name in preds_granular_dict]
            aligned_weights_for_ensemble = [weight_dict[name] for name in valid_model_names if name in weight_dict]

            # Calculate granular ensemble prediction
            granular_ensemble_preds = weighted_ensemble_predictions(aligned_granular_preds_for_ensemble, aligned_weights_for_ensemble)
            validation_plot_df['Predicted'] = granular_ensemble_preds

            # Calculate granular errors for the ensemble
            granular_errors_best_model = y_valid_original - granular_ensemble_preds

        else: # If the best model is an individual model (LGBM, XGBoost, RF, GB)
            # The predictions_valid[best_model_name] are already granular from train_evaluate_model
            validation_plot_df['Predicted'] = predictions_valid[best_model_name]
            # The model_errors_valid[best_model_name] should also be granular from train_evaluate_model
            granular_errors_best_model = model_errors_valid[best_model_name]


        validation_plot_df['Residual'] = granular_errors_best_model # Use granular errors
        
        # Ensure muni/serv columns are correctly typed if they were objects
        validation_plot_df[muni_enc_col] = pd.to_numeric(validation_plot_df[muni_enc_col], errors='coerce').fillna(-1).astype(int)
        validation_plot_df[serv_enc_col] = pd.to_numeric(validation_plot_df[serv_enc_col], errors='coerce').fillna(-1).astype(int)


        plot_evaluation_graphs(
            validation_plot_df, # Pass the granular DataFrame
            best_model_name,
            train_df_orig,
            valid_df_orig, # Pass original valid_df for history plots
            decoders,
            muni_enc_col,
            serv_enc_col,
            target_col,
            date_col,
            error_series_best=granular_errors_best_model, # Pass granular errors
            X_train_q=X_train, y_train_q=y_train, X_valid_q=X_valid, y_valid_q=y_valid, cat_indices_q=cat_indices
        )
    else:
        print("No se pueden generar gráficas evaluación (mejor modelo no válido o predicciones faltantes).")


    print("\n--- Paso 4: Generación Predicciones Futuras (Iterativo) ---")
    last_data_date = max(train_df_orig[date_col].max(), valid_df_orig[date_col].max())
    future_dates = pd.date_range(start=last_data_date + pd.DateOffset(months=1), periods=12, freq='MS')
    print(f"Prediciendo de {future_dates.min():%Y-%m} a {future_dates.max():%Y-%m}")
    unique_combos = X_train[[muni_enc_col, serv_enc_col]].drop_duplicates()
    print(f"Prediciendo para {len(unique_combos)} combinaciones.")

    model_features = features.copy()
    hist_target_col = target_col + '_transformed' if APPLY_LOG_TRANSFORM else target_col
    all_predictions_df_prep = pd.concat([train_df_orig, valid_df_orig], ignore_index=True)
    if APPLY_LOG_TRANSFORM:
        all_predictions_df_prep[hist_target_col] = np.log1p(all_predictions_df_prep[target_col])
    else:
        all_predictions_df_prep[hist_target_col] = all_predictions_df_prep[target_col]
    hist_cols_needed_iter = list(set([date_col, muni_enc_col, serv_enc_col, hist_target_col, target_col] + model_features))
    hist_cols_existing_iter = [col for col in hist_cols_needed_iter if col in all_predictions_df_prep.columns]

    all_predictions_df_prep[muni_enc_col] = pd.to_numeric(all_predictions_df_prep[muni_enc_col], errors='coerce').fillna(-1).astype(int)
    all_predictions_df_prep[serv_enc_col] = pd.to_numeric(all_predictions_df_prep[serv_enc_col], errors='coerce').fillna(-1).astype(int)
    all_predictions_df = all_predictions_df_prep[hist_cols_existing_iter].groupby([muni_enc_col, serv_enc_col, date_col]).last().reset_index()

    if best_model_name == 'XGBoost' and X_train_xgb is not None:
        model_features_sanitized = X_train_xgb.columns.tolist()

    lag_cache = {}
    final_predictions_list = []
    for current_date in future_dates:
        unique_combos_iter = unique_combos.copy()
        unique_combos_iter[muni_enc_col] = pd.to_numeric(unique_combos_iter[muni_enc_col], errors='coerce').fillna(-1).astype(int)
        unique_combos_iter[serv_enc_col] = pd.to_numeric(unique_combos_iter[serv_enc_col], errors='coerce').fillna(-1).astype(int)
        current_future_df, current_X, lag_cache = generate_future_features(
            current_date, unique_combos_iter, all_predictions_df, model_features,
            hist_target_col, APPLY_LOG_TRANSFORM, muni_enc_col, serv_enc_col, date_col, lag_cache
        )

        if best_model_name == 'XGBoost' and X_train_xgb is not None:
            current_X_xgb = current_X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', str(x)))
            preds_transformed = models[best_model_name].predict(current_X_xgb)
        elif best_model_name == 'Ensemble':
            ensemble_preds_list = []
            for name in models['Ensemble']['models']:
                if name.startswith('Sub_'):
                    subset = name.split('_')[1]
                    mask = current_X['Is_Medellin' if subset == 'Medellin' else 'Is_Alta_Inmediata'] == 1
                    sub_pred = np.zeros(len(current_X))
                    if mask.any():
                        sub_pred[mask] = submodels[subset].predict(current_X[mask])
                    ensemble_preds_list.append(sub_pred)
                else:
                    if name == 'XGBoost':
                        current_X_xgb = current_X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', str(x)))
                        pred = models[name].predict(current_X_xgb)
                    else:
                        pred = models[name].predict(current_X)
                    ensemble_preds_list.append(pred)
            weights = models['Ensemble']['weights']
            ensemble_preds_list = ensemble_preds_list[:len(weights)]
            preds_transformed = weighted_ensemble_predictions(ensemble_preds_list, weights)
        else:
            preds_transformed = models[best_model_name].predict(current_X)

        preds = preds_transformed
        preds = np.maximum(0, preds)

        current_results = current_future_df[[muni_enc_col, serv_enc_col]].copy()
        current_results['Date'] = current_date
        current_results['Prediction'] = preds
        final_predictions_list.append(current_results)

        new_data = current_results.copy()
        new_data[date_col] = current_date
        new_data[target_col] = preds
        new_data[hist_target_col] = preds
        all_predictions_df = pd.concat([all_predictions_df, new_data[hist_cols_existing_iter]], ignore_index=True)

    final_predictions_df = pd.concat(final_predictions_list, ignore_index=True)
    print(f"\nPredicciones finales generadas: {final_predictions_df.shape}")

    # Intervalos de predicción
    print("\nGenerando intervalos de predicción...")
    lower_bounds, upper_bounds = bootstrap_predictions(
        models[best_model_name] if best_model_name != 'Ensemble' else models['LGBM'],
        X_valid, apply_log_transform=APPLY_LOG_TRANSFORM
    )
    validation_results = pd.DataFrame({
        'Date': y_valid_agg.index,
        'Actual': y_valid_agg.values,
        'Predicted': predictions_valid[best_model_name],
        'Lower_Bound': lower_bounds[:len(y_valid_agg)],
        'Upper_Bound': upper_bounds[:len(y_valid_agg)],
        muni_enc_col: X_valid[muni_enc_col].groupby(valid_df_orig[date_col]).first().reindex(y_valid_agg.index).fillna(-1).astype(int),
        serv_enc_col: X_valid[serv_enc_col].groupby(valid_df_orig[date_col]).first().reindex(y_valid_agg.index).fillna(-1).astype(int)
    })
    print("Intervalos de predicción generados (Validation):")
    print(validation_results[['Date', 'Actual', 'Predicted', 'Lower_Bound', 'Upper_Bound']].head())

    # Análisis adicional
    print("\n--- Análisis Adicional de Errores (Validación) ---")
    group_cols = [muni_enc_col, serv_enc_col]
    errors_by_group = analyze_errors_by_group(
        validation_results, model_errors_valid[best_model_name], group_cols
    )
    print("\nErrores promedio por grupo (Top 10):")
    print(errors_by_group.head(10))

    # Importancia de características
    plot_feature_importance(
        models, features, X_train_xgb_cols=X_train_xgb.columns if 'X_train_xgb' in locals() else None
    )

    # Exportar resultados
    results_dict = {
        'model_metrics': {
            name: {
                'MAE': mae_scores.get(name, float('inf')),
                'RMSE': rmse_scores.get(name, float('inf')),
                'R2': r2_scores.get(name, float('-inf'))
            } for name in models.keys()
        },
        'statistical_validation': stat_results,
        'errors_by_group': errors_by_group.to_dict(orient='records'),
        'execution_time': time.time() - start_total_time,
        'tuning_time': tuning_time
    }
    results_dict = make_json_serializable(results_dict)
    with open(JSON_OUTPUT_PATH, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"\nResultados exportados a {JSON_OUTPUT_PATH}")
    print(f"Tiempo total ejecución: {time.time() - start_total_time:.2f} segundos")

if __name__ == "__main__":
    main()

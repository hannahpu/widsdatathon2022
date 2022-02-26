from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, KFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC
import pandas as pd
import numpy as np


def run_bootstrap_stratified_validation(
    model_df,
    ml_model,
    features_columns,
    if_scale_data,
    if_one_hot,
    model_type="sklearn",
    stratify_col='facility_type_parsed',
    response_col="site_eui",
    if_output_prediction_results=False,
    resample_param_dict={},
    imputer=None,
    n_bootstraps=5,
):
    # Sample the train/validation dataset so it resembles true test distribution
    bootstrap_strat_df = {}
    for i in range(n_bootstraps):
        sub_model_df = heuristic_sample_to_true_test(
            model_df, col_to_resample='facility_type_parsed')
        stratified_model_result_df = run_stratified_validation(sub_model_df, ml_model, features_columns,
                                                               if_scale_data, if_one_hot, model_type=model_type,
                                                               stratify_col=stratify_col, response_col=response_col,
                                                               if_output_prediction_results=if_output_prediction_results,
                                                               resample_param_dict=resample_param_dict,
                                                               imputer=imputer)
        bootstrap_strat_df = pd.concat(
            [bootstrap_strat_df, stratified_model_result_df.assign(bootstrap=i)])
    return bootstrap_strat_df


def run_stratified_validation(
    model_df,
    ml_model,
    features_columns,
    if_scale_data,
    if_one_hot,
    model_type="sklearn",
    stratify_col='facility_type_parsed',
    response_col="site_eui",
    if_output_prediction_results=False,
    resample_param_dict={},
    imputer=None,
):
    # Define which function to run
    run_model_dict = {
        "sklearn": run_sklearn_model,
        "catboost": run_catboost_model,
        "lightgbm": run_lgb_model,
        "dnn": run_dnn_model,
    }
    assert model_type in run_model_dict.keys(
    ), f"{model_type} not in {run_model_dict.keys()}"

    all_stratified_model_result = []
    prediction_result_train_dict = {}
    prediction_result_test_dict = {}
    print(f"Running {model_type}")

    # Instantiate stratified kfold model
    skf = StratifiedKFold()
    # Note stratify_y is not the target for regression, but rather the target
    # for stratifying train/test split for validation (eg facility type)
    X = model_df.drop(columns=stratify_col)
    stratify_y = model_df[stratify_col].map(
        dict(
            zip(
                np.sort(model_df[stratify_col].unique()),
                np.arange(model_df[stratify_col].nunique()),
            )
        )
    ).values

    # Each train/test split is ~4:1 equal ratio of stratify_y
    for strat, (train_inds, test_inds) in enumerate(skf.split(X.values, stratify_y)):
        # Prep/preprocess
        train_df = model_df.iloc[train_inds]
        test_df = model_df.iloc[test_inds]
        train_x_df, train_y_df = split_model_feature_response(
            train_df, features_columns, response_col=response_col
        )
        test_x_df, test_y_df = split_model_feature_response(
            test_df, features_columns, response_col=response_col
        )
        train_x_df, test_x_df = process_train_test_data(
            train_x_df, test_x_df, if_scale_data, if_one_hot, model_df, imputer=imputer
        )
        # Run regression model
        train_predict, test_predict, fitted_model = run_model_dict[model_type](
            ml_model, train_x_df, train_y_df, test_x_df,
            validation_data=(test_x_df.values,
                             test_y_df.values),
        )
        # Calculate training & validation metrics
        train_rmse = calculate_rmse(train_y_df, train_predict)
        test_rmse = calculate_rmse(test_y_df, test_predict)
        stratified_result_df = pd.DataFrame(
            {
                "stratification": strat,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
            },
            index=[0],
        )
        all_stratified_model_result.append(stratified_result_df)
        prediction_result_train_dict[strat] = train_predict
        prediction_result_test_dict[strat] = test_predict
    all_stratified_model_result_df = pd.concat(
        all_stratified_model_result).reset_index(drop=True)
    if if_output_prediction_results:
        return all_stratified_model_result_df, prediction_result_train_dict, prediction_result_test_dict
    else:
        return all_stratified_model_result_df


def run_leave_year_out(
    model_df,
    ml_model,
    features_columns,
    if_scale_data,
    if_one_hot,
    model_type="sklearn",
    response_col="site_eui",
    if_output_prediction_results=False,
    resample_param_dict={},
    imputer=None,
):
    # Define which function to run
    run_model_dict = {
        "sklearn": run_sklearn_model,
        "catboost": run_catboost_model,
        "lightgbm": run_lgb_model,
        "dnn": run_dnn_model,
    }
    assert model_type in run_model_dict.keys(
    ), f"{model_type} not in {run_model_dict.keys()}"
    all_loy_model_result = []
    all_year = model_df["year_factor"].unique()
    prediction_result_train_dict = {}
    prediction_result_test_dict = {}
    print(f"Running {model_type}")
    for one_year in all_year:
        print(f"Modeling {one_year}...")
        (
            left_out_test_x_df,
            left_out_test_y_df,
            left_out_train_x_df,
            left_out_train_y_df,
        ) = train_test_split(one_year, model_df, features_columns, response_col)
        if len(resample_param_dict) > 0:
            train_for_resample_df = left_out_train_x_df
            train_for_resample_df[response_col] = left_out_train_y_df
            up_or_downsample = resample_param_dict["up_or_downsample"]
            resample_by_col = resample_param_dict["resample_by_col"]
            resample_type = resample_param_dict["resample_type"]
            if up_or_downsample == "upsample":
                train_after_resampled_df = upsampling_by_column(
                    train_for_resample_df, resample_by_col, resample_type=resample_type
                )
            elif up_or_downsample == "downsample":
                train_after_resampled_df = downsampling_by_column(
                    train_for_resample_df, resample_by_col, resample_type=resample_type
                )
            elif up_or_downsample == "custom_upsample":
                print("getting custom upsample")
                df_to_get_weights = model_df.query(
                    f"year_factor != {one_year}")
                train_after_resampled_df = custom_weighted_upsample(
                    train_for_resample_df, df_to_get_weights, resample_by_col)
            left_out_train_x_df, left_out_train_y_df = split_model_feature_response(
                train_after_resampled_df,
                features_columns,
                if_with_response=True,
                response_col=response_col,
            )
        left_out_train_x_df, left_out_test_x_df = process_train_test_data(
            left_out_train_x_df, left_out_test_x_df, if_scale_data, if_one_hot, model_df, imputer=imputer
        )
        train_predict, test_predict, fitted_model = run_model_dict[model_type](
            ml_model, left_out_train_x_df, left_out_train_y_df, left_out_test_x_df,
            validation_data=(left_out_test_x_df.values,
                             left_out_test_y_df.values),
        )
        train_rmse = calculate_rmse(left_out_train_y_df, train_predict)
        test_rmse = calculate_rmse(left_out_test_y_df, test_predict)
        one_year_result_df = pd.DataFrame(
            {
                "left_out_year": one_year,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
            },
            index=[0],
        )
        all_loy_model_result.append(one_year_result_df)
        prediction_result_train_dict[one_year] = train_predict
        prediction_result_test_dict[one_year] = test_predict
    all_loy_model_result_df = pd.concat(
        all_loy_model_result).reset_index(drop=True)
    if if_output_prediction_results:
        return all_loy_model_result_df, prediction_result_train_dict, prediction_result_test_dict
    else:
        return all_loy_model_result_df


def train_test_split(level, model_df, features_columns, response_col="site_eui"):
    left_out_test = model_df.query(f"year_factor == {level}")
    left_out_train = model_df.query(f"year_factor != {level}")
    left_out_test_x_df, left_out_test_y_df = split_model_feature_response(
        left_out_test, features_columns, response_col=response_col
    )
    left_out_train_x_df, left_out_train_y_df = split_model_feature_response(
        left_out_train, features_columns, response_col=response_col
    )
    return (
        left_out_test_x_df,
        left_out_test_y_df,
        left_out_train_x_df,
        left_out_train_y_df,
    )


def split_model_feature_response(
    model_df, features_columns, if_with_response=True, response_col="site_eui"
):
    model_x_df = model_df[features_columns]
    if if_with_response:
        model_y_df = model_df[response_col]
        return model_x_df, model_y_df
    else:
        return model_x_df


def one_hot_encode_data(train_x_df, test_x_df, full_data_df):
    categorical_columns_to_dummy = output_non_numeric_columns(train_x_df)
    # print(f"Columns to be dummied: {categorical_columns_to_dummy}")
    for col in categorical_columns_to_dummy:
        encoder = get_one_hot_encoder(full_data_df[[col]])
        one_hot_encoded_column_name = [
            f"{col}_{ind}" for ind in range(full_data_df[col].nunique())
        ]
        train_one_hot_encoded = encoder.transform(train_x_df[[col]])
        train_one_hot_encoded = pd.DataFrame(
            train_one_hot_encoded,
            columns=one_hot_encoded_column_name,
            index=train_x_df.index,
        )
        test_one_hot_encoded = encoder.transform(test_x_df[[col]])
        test_one_hot_encoded = pd.DataFrame(
            test_one_hot_encoded,
            columns=one_hot_encoded_column_name,
            index=test_x_df.index,
        )
        train_x_df = pd.concat(
            [train_x_df, train_one_hot_encoded], axis="columns")
        test_x_df = pd.concat(
            [test_x_df, test_one_hot_encoded], axis="columns")
    train_x_df = train_x_df.drop(columns=categorical_columns_to_dummy)
    test_x_df = test_x_df.drop(columns=categorical_columns_to_dummy)
    return train_x_df, test_x_df


def process_train_test_data(train_x_df, test_x_df, if_scale_data, if_one_hot, full_data_df, imputer=None):
    if if_one_hot:
        train_x_df, test_x_df = one_hot_encode_data(
            train_x_df, test_x_df, full_data_df)
    if if_scale_data:
        train_x_df, test_x_df = scale_data(train_x_df, test_x_df)
    if imputer:
        train_x_df, test_x_df = run_imputer(
            imputer, train_x_df, test_x_df, full_data_df)
    return train_x_df, test_x_df


def run_imputer(imputer, train_x_df, test_x_df, full_data_df):
    # Pre-process categorical features -> one hot encoding
    categorical_columns_to_dummy = output_non_numeric_columns(train_x_df)
    if categorical_columns_to_dummy:
        train_x_df, test_x_df = one_hot_encode_data(
            train_x_df.copy(), test_x_df.copy(), full_data_df)
    # Run imputer
    train_x_impute_df = imputer.fit_transform(train_x_df)
    test_x_impute_df = imputer.transform(test_x_df)

    train_x_impute_df = pd.DataFrame(
        train_x_impute_df, columns=train_x_df.columns
    )
    test_x_impute_df = pd.DataFrame(
        test_x_impute_df, columns=test_x_df.columns
    )
    # Reverse one-hot encoding -> categorical features
    for col in categorical_columns_to_dummy:
        one_hot_encoded_column_name = [
            f"{col}_{ind}" for ind in range(full_data_df[col].nunique())
        ]
        one_hot_to_cat_dict = dict(zip(one_hot_encoded_column_name,
                                       np.sort(full_data_df[col].unique())))
        train_x_impute_df[col] = train_x_impute_df[one_hot_encoded_column_name].idxmax(
            1).map(one_hot_to_cat_dict)
        test_x_impute_df[col] = test_x_impute_df[one_hot_encoded_column_name].idxmax(
            1).map(one_hot_to_cat_dict)
        train_x_impute_df = train_x_impute_df.drop(
            columns=one_hot_encoded_column_name)
        test_x_impute_df = test_x_impute_df.drop(
            columns=one_hot_encoded_column_name)
    return train_x_impute_df, test_x_impute_df


def output_non_numeric_columns(model_df):
    numeric_columns = list(model_df._get_numeric_data().columns)
    all_columns = list(model_df.columns)
    non_numeric_columns = list(set(all_columns) - set(numeric_columns))
    return non_numeric_columns


def scale_data(train_x, test_x):
    scaler = StandardScaler()
    scaler = scaler.fit(train_x)
    scaled_train_x = scaler.transform(train_x)
    scaled_test_x = scaler.transform(test_x)
    scaled_train_x = pd.DataFrame(
        scaled_train_x, columns=train_x.columns, index=train_x.index)
    scaled_test_x = pd.DataFrame(
        scaled_test_x, columns=test_x.columns, index=test_x.index)
    return scaled_train_x, scaled_test_x


def get_one_hot_encoder(train_df):
    enc = OneHotEncoder(sparse=False)
    return enc.fit(train_df)


def run_sklearn_model(sklearn_model, train_x_df, train_y_df, test_x_df, validation_data=None):
    fitted_model = fit_sklearn_model(sklearn_model, train_x_df, train_y_df)
    train_predict = run_sklearn_predict(fitted_model, train_x_df)
    test_predict = run_sklearn_predict(fitted_model, test_x_df)
    return train_predict, test_predict, fitted_model


def fit_lgb_model(model, train_x, train_y):
    # fit_params = {
    #     "early_stopping_rounds": 100,
    #     "eval_metric": "rmse",
    #     # "eval_set": [(X_eval, y_eval)],
    #     "eval_names": ["valid"],
    #     "verbose": 1000,
    # }
    # model.fit(train_x, train_y, **fit_params)
    model.fit(train_x, train_y)
    return model


def run_lgb_model(lgb_model, train_x_df, train_y_df, test_x_df, validation_data=None):
    fitted_model = fit_lgb_model(lgb_model, train_x_df, train_y_df)
    train_predict = run_sklearn_predict(fitted_model, train_x_df)
    test_predict = run_sklearn_predict(fitted_model, test_x_df)
    return train_predict, test_predict, fitted_model


def fit_sklearn_model(model, train_x, train_y):
    model.fit(train_x, train_y)
    return model


def run_sklearn_predict(model, test_x):
    predict_result = model.predict(test_x)
    return predict_result


def calculate_rmse(true_y, predict_y):
    return mean_squared_error(true_y, predict_y, squared=False)


def run_catboost_model(model, train_x_df, train_y_df, test_x_df, validation_data=None):
    cat_columns = train_x_df.select_dtypes(["O"]).columns.tolist()
    model.fit(train_x_df, y=train_y_df, cat_features=cat_columns)
    train_predict = model.predict(train_x_df)
    test_predict = model.predict(test_x_df)
    return train_predict, test_predict, model


def run_dnn_model(model, train_x_df, train_y_df, test_x_df, validation_data=None):
    if model is None:
        model = build_and_compile_dnn_model(train_x_df)
    validation_split = 0 if validation_data else 0.2
    # Stop training if loss doesn't improve from min value over 10 iterations
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
    # Fit nn - note fitting will essentially pick up from last state

    model.fit(train_x_df, train_y_df, verbose=1, validation_data=validation_data,
              validation_split=validation_split,
              epochs=100, callbacks=[callback])
    train_predict = model.predict(train_x_df)
    test_predict = model.predict(test_x_df)
    return train_predict, test_predict, model


def build_and_compile_dnn_model(input_features):
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(input_features))

    model = keras.Sequential([
        normalizer,
        layers.Dense(64, activation='elu'),
        layers.Dense(64, activation='elu'),
        layers.Dense(1)
    ])

    model.compile(loss=tf_rmse,  # 'mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


def tf_rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


def tf_rmsle(y_true, y_pred):
    return K.sqrt(K.mean(K.square(K.log(1.+y_true) - K.log(1+y_pred))))


def upsampling_by_column(train_df, resample_by_col, resample_type="random"):
    if resample_type == "random":
        train_x_to_resample = train_df.drop(columns=resample_by_col)
        train_y_to_resample = train_df[resample_by_col]
        oversampler = RandomOverSampler(random_state=42)
        train_x_resampled, train_y_resampled = oversampler.fit_resample(
            train_x_to_resample, train_y_to_resample
        )
    elif resample_type == "smote":
        train_dropna_df = train_df.dropna(how="any")
        train_x_to_resample = train_dropna_df.drop(columns=resample_by_col)
        train_y_to_resample = train_dropna_df[resample_by_col]
        non_numeric_columns = output_non_numeric_columns(train_x_to_resample)
        categorical_column_index = [
            train_x_to_resample.columns.get_loc(c)
            for c in non_numeric_columns
            if c in train_x_to_resample
        ]
        sm = SMOTENC(random_state=42,
                     categorical_features=categorical_column_index)
        train_x_resampled, train_y_resampled = sm.fit_resample(
            train_x_to_resample, train_y_to_resample
        )
    final_resampled_train_df = train_x_resampled
    final_resampled_train_df[resample_by_col] = train_y_resampled
    return final_resampled_train_df


def custom_weighted_upsample(df_to_resample, df_to_get_weights, resample_by_col):
    # get the levels that needs upsampled and the needed sample number
    column_level_counts = df_to_resample[resample_by_col].value_counts()
    final_sampled_number = column_level_counts.max()
    column_level_to_resample = column_level_counts[column_level_counts !=
                                                   final_sampled_number]
    column_level_number_to_resample = final_sampled_number - column_level_to_resample
    # upsample per level
    all_resampled_list = []
    for level, number in column_level_number_to_resample.iteritems():
        level_to_resample_from_df = df_to_resample.query(
            f"{resample_by_col} == '{level}'")
        level_resample_weights = df_to_get_weights.query(f"{resample_by_col} == '{level}'")[
            "resample_weights"
        ].values
        assert level_to_resample_from_df.shape[0] == len(
            level_resample_weights)
        resampled_data = level_to_resample_from_df.sample(
            n=number, replace=True, weights=level_resample_weights, random_state=None, axis=0
        )
        all_resampled_list.append(resampled_data)
    # combine original data with upsampled data
    all_resampled_df = pd.concat(all_resampled_list)
    final_resampled_data = pd.concat([df_to_resample, all_resampled_df])
    return final_resampled_data


def downsampling_by_column(train_df, resample_by_col, resample_type="random"):
    train_x_to_resample = train_df.drop(columns=resample_by_col)
    train_y_to_resample = train_df[resample_by_col]
    if resample_type == "random":
        undersampler = RandomUnderSampler(
            random_state=42, sampling_strategy="majority")
        train_x_resampled, train_y_resampled = undersampler.fit_resample(
            train_x_to_resample, train_y_to_resample
        )
    final_resampled_train_df = train_x_resampled
    final_resampled_train_df[resample_by_col] = train_y_resampled
    return final_resampled_train_df


def run_model_predict_unknown_test_by_column(
    train_df, test_df, full_data_df, features_columns, response_col, if_scale, if_one_hot, model
):
    all_year_factor = train_df["year_factor"].unique()
    test_prediction_result = []
    for one_year in all_year_factor:
        print(f"Modeling {one_year}...")
        train_filter_df = train_df.query(f"year_factor != {one_year}")
        test_filter_df = test_df.query(f"year_factor == {one_year}")
        train_filter_x_df, train_filter_y_df = split_model_feature_response(
            train_filter_df, features_columns, if_with_response=True, response_col=response_col
        )
        test_filter_x_df = split_model_feature_response(
            test_filter_df, features_columns, if_with_response=False
        )
        processed_train_x_df, processed_test_x_df = process_train_test_data(
            train_filter_x_df, test_filter_x_df, if_scale, if_one_hot, full_data_df
        )
        train_predict, test_predict, fitted_model = run_sklearn_model(
            model, processed_train_x_df, train_filter_y_df, processed_test_x_df
        )
        test_predict_df = test_filter_df[["id"]]
        test_predict_df.loc[:, f"predict_{response_col}"] = test_predict
        test_predict_df.loc[:, "year_factor"] = one_year
        test_prediction_result.append(test_predict_df)
        training_rmse = calculate_rmse(train_filter_y_df, train_predict)
        num_unique_test_predict = len(np.unique(test_predict))
        print(
            f"{one_year} train rmse: {training_rmse}, num unique test prediction: {num_unique_test_predict}"
        )
    all_test_prediction_result = pd.concat(test_prediction_result)
    return all_test_prediction_result


def heuristic_sample_to_true_test(train_df, col_to_resample='facility_type_parsed'):
    resample_cols_reduce_dict = {"Multifamily": 0.7, "Office": 0.25}
    sample_reduce_inds_dict = {}
    inds_to_drop = []
    for ftp, frac in resample_cols_reduce_dict.items():
        itd = (
            train_df.loc[train_df[col_to_resample] == ftp]
            .sample(frac=frac).index
        )
        sample_reduce_inds_dict[ftp] = itd
        inds_to_drop += list(itd)

    resample_cols_augment_dict = {"Unit_Building": 1}
    sample_augment_inds_dict = {}
    inds_to_add = []
    for ftp, frac in resample_cols_augment_dict.items():
        itd = (
            train_df.loc[train_df[col_to_resample] == ftp]
            .sample(frac=frac).index
        )
        sample_augment_inds_dict[ftp] = itd
        inds_to_add += list(itd)
    new_train_df = pd.concat(
        [
            train_df.drop(index=inds_to_drop),
            train_df.iloc[inds_to_add],
        ]
    )
    return new_train_df


def process_loy_train_test_prediction(
    loy_prediction_result_train_dict, loy_prediction_result_test_dict, model_df
):
    all_year_factor = list(loy_prediction_result_train_dict.keys())
    all_loy_train_predict = []
    all_loy_test_predict = []
    for one_year in all_year_factor:
        left_year_train_df = model_df.query(
            f"year_factor != {one_year}")[["id"]]
        left_year_test_df = model_df.query(
            f"year_factor == {one_year}")[["id"]]
        one_left_year_train_df = loy_prediction_result_train_dict[one_year]
        one_left_year_test_df = loy_prediction_result_test_dict[one_year]
        left_year_train_df["train_prediction"] = one_left_year_train_df
        left_year_train_df["left_year"] = one_year
        left_year_test_df["test_prediction"] = one_left_year_test_df
        left_year_test_df["left_year"] = one_year
        all_loy_train_predict.append(left_year_train_df)
        all_loy_test_predict.append(left_year_test_df)
    all_loy_train_predict_df = pd.concat(all_loy_train_predict)
    all_loy_test_predict_df = pd.concat(all_loy_test_predict)
    return all_loy_train_predict_df, all_loy_test_predict_df

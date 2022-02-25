#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 20:21:10 2022

@author: diana
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .model_utils import scale_data
from .data_process_utils import backfill_energy_star_rating, backfill_wind_direction, parse_facility_type


def clean_impute_data(
    train_df,
    test_df,
    impute_col="year_built",
    impute_thresh=1800,
    factors_cols=["facility_type", "state_factor"],
):
    agg_df = (
        train_df.groupby(factors_cols)[impute_col].median(
        ).reset_index(name=f"median_{impute_col}")
    )
    train_df = impute_with_agg(
        train_df, agg_df, impute_col, factors_cols, impute_thresh)
    test_df = impute_with_agg(
        test_df, agg_df, impute_col, factors_cols, impute_thresh)
    return train_df, test_df


def impute_with_agg(base_df, agg_df, impute_col, factors_cols, impute_thresh):
    # First, impute any 0, negative, or missing values
    replace_df = base_df.merge(agg_df, on=factors_cols, how="left")
    bad_entries = (base_df[impute_col] <= 0) | (base_df[impute_col].isna())
    base_df.loc[bad_entries, impute_col] = replace_df.loc[bad_entries,
                                                          f"median_{impute_col}"]

    # Set values lower than threshold to threshold
    if impute_thresh:
        base_df.loc[base_df[impute_col] <
                    impute_thresh, impute_col] = impute_thresh
    return base_df


def process_data_v1(
    train_df,
    test_df,
    reduce_col_dict,
    cols_to_log_transform,
    reduce_number_dict,
):
    # Standardize colnames
    train_df = standardize_colnames(train_df)
    test_df = standardize_colnames(test_df)

    # Make log transformations
    if cols_to_log_transform:
        train_df = log_transform(train_df, cols_to_log_transform)
        test_df = log_transform(test_df, cols_to_log_transform)

    # Make imputations for year_built
    impute_dict = {"year_built": ["facility_type", "state_factor"]}
    thresh_dict = {"year_built": 1800}
    for impute_col, factors_cols in impute_dict.items():
        train_df, test_df = clean_impute_data(
            train_df,
            test_df,
            impute_col=impute_col,
            impute_thresh=thresh_dict[impute_col],
            factors_cols=factors_cols,
        )

    # PCA transformations to reduce dimensions
    if len(reduce_col_dict) > 0:
        all_pca_cols = []
        all_reduce_keys = list(reduce_col_dict.keys())
        for one_reduce_type in all_reduce_keys:
            cols_to_reduce = reduce_col_dict[one_reduce_type]
            n_pca_components = reduce_number_dict[one_reduce_type]
            train_df, test_df, pca_cols = reduce_dimensions(
                train_df,
                test_df,
                cols_to_reduce,
                prefix=one_reduce_type,
                n_components=n_pca_components,
            )
            all_pca_cols = all_pca_cols + pca_cols
    else:
        all_pca_cols = []
    return train_df, test_df, all_pca_cols


def process_data(
    train_df,
    test_df,
    cols_to_reduce,
    cols_to_log_transform,
    n_pca_components,
    backfill_dict,
    categorical_feature_to_reduce=None
):
    # Standardize colnames
    train_df = standardize_colnames(train_df)
    test_df = standardize_colnames(test_df)

    # Parse categorical feature to reduce unique categories
    if categorical_feature_to_reduce:
        assert categorical_feature_to_reduce in train_df.select_dtypes(["O"]).columns, \
        f"categorical_feature_to_reduce={categorical_feature_to_reduce} not in input df"
        train_df = parse_facility_type(train_df, facility_type_colname=categorical_feature_to_reduce)
        test_df = parse_facility_type(test_df, facility_type_colname=categorical_feature_to_reduce)
        
    # Make log transformations
    if cols_to_log_transform:
        train_df = log_transform(train_df, cols_to_log_transform)
        test_df = log_transform(test_df, cols_to_log_transform)

    # Make imputations for year_built
    impute_dict = {"year_built": ["facility_type", "state_factor"]}
    thresh_dict = {"year_built": 1800}
    for impute_col, factors_cols in impute_dict.items():
        train_df, test_df = clean_impute_data(
            train_df,
            test_df,
            impute_col=impute_col,
            impute_thresh=thresh_dict[impute_col],
            factors_cols=factors_cols,
        )
    if backfill_dict:
        for col, groupby_list in backfill_dict.items():
            # Backfill... Meng's func is generic
            train_df = backfill_energy_star_rating(
                input_df=train_df,
                mapping_df=train_df,
                groupby_list=groupby_list,
                energy_star_rating_colname=col,
                agg_approach_func=np.nanmedian,
            )
            test_df = backfill_energy_star_rating(
                input_df=test_df,
                mapping_df=train_df,
                groupby_list=groupby_list,
                energy_star_rating_colname=col,
                agg_approach_func=np.nanmedian,
            )

    # PCA transformations to reduce dimensions
    if cols_to_reduce:
        train_df, test_df, pca_cols = reduce_dimensions(
            train_df, test_df, cols_to_reduce, n_components=n_pca_components
        )
    else:
        pca_cols = []
    return train_df, test_df, pca_cols


def log_transform(df, cols_to_transform, log_type="log10"):
    log_func = {"log10": np.log10, "ln": np.log, "log2": np.log2}
    for col in cols_to_transform:
        offset = 0.1
        if np.nanmin(df[col]) < 0:
            offset += abs(np.nanmin(df[col]))
        df[col] = log_func[log_type](df[col] + offset)
    df = df.rename(
        columns=dict(
            zip(cols_to_transform, [f"{log_type}_{col}" for col in cols_to_transform]))
    )
    return df


def invert_log_transform(df, cols_to_invert, log_type="log10"):
    log_func = {"ln": np.exp, "log10": lambda x: 10 **
                x, "log2": lambda x: 2 ** x}
    for col in cols_to_invert:
        df[col] = log_func[log_type](df[col]) - 1
    df = df.rename(
        columns=dict(zip(cols_to_invert, [col.replace(
            log_type, "") for col in cols_to_invert]))
    )
    return df


def standardize_colnames(df):
    df.columns = df.columns.str.lower()
    return df


def reduce_dimensions(train_x_df, test_x_df, cols_to_reduce, prefix="temp", n_components=9):
    if not cols_to_reduce:
        cols_to_reduce = list(train_x_df.columns)

    scaled_train_x_df, scaled_test_x_df = scale_data(
        train_x_df[cols_to_reduce], test_x_df[cols_to_reduce]
    )
    pca = pca_fit(scaled_train_x_df, n_components=n_components)
    train_x_pca = pca.transform(scaled_train_x_df)
    test_x_pca = pca.transform(scaled_test_x_df)

    pca_cols = [
        f"{prefix}_pca{i+1}" for i in range(len(pca.explained_variance_ratio_))]

    pca_train_x_df = pd.DataFrame(train_x_pca, columns=pca_cols)
    pca_test_x_df = pd.DataFrame(test_x_pca, columns=pca_cols)

    train_x_df = pd.concat(
        [train_x_df.drop(columns=cols_to_reduce), pca_train_x_df], axis=1)
    test_x_df = pd.concat(
        [test_x_df.drop(columns=cols_to_reduce), pca_test_x_df], axis=1)

    return train_x_df, test_x_df, pca_cols


def pca_fit(x, n_components, var_thresh=0.75, var_diff_thresh=0.01):
    # Get number of components to fit, stop when explained variance ratio flattens out
    if n_components <= 0 or None:
        n_components = get_pca_n_components(
            x, var_thresh=var_thresh, var_diff_thresh=var_diff_thresh
        )
    print(f"Fitting PCA with {n_components} components")
    pca = PCA(n_components=n_components).fit(x)
    return pca


def pca_transform(x, pca):
    return pca.transform(x)


def get_pca_n_components(x, var_thresh=0.75, var_diff_thresh=0.01, plot=True):
    pca = PCA().fit(x)
    cum_var_ratio = np.cumsum(pca.explained_variance_ratio_)
    stopping_criteria1 = cum_var_ratio > var_thresh
    stopping_criteria2 = np.insert(
        np.diff(cum_var_ratio) > var_diff_thresh, 0, True)
    if (stopping_criteria1 & stopping_criteria2).any():
        mask = stopping_criteria1 & stopping_criteria2
    elif stopping_criteria1.any():
        mask = stopping_criteria1
    elif stopping_criteria2.any():
        mask = stopping_criteria2
    else:
        raise ValueError("Something went wrong with autogen PCA components")
    n_components = np.arange(cum_var_ratio.shape[0])[mask][-1]
    if plot:
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.axvline(n_components, linestyle="--", color="0.2")
        plt.xlabel("number of components")
        plt.ylabel("cumulative explained variance")
    return n_components


def combine_features_and_prediction(x_df, y_df, y_pred_df):
    merged_df = x_df.copy()
    merged_df["y_true"] = y_df
    merged_df["y_pred"] = y_pred_df
    merged_df["y_diff"] = merged_df["y_true"] - merged_df["y_pred"]
    return merged_df


def get_rmse_by_group(x_df, y_df, y_pred_df, group_cols=["facility_type"]):
    merged_df = combine_features_and_prediction(x_df, y_df, y_pred_df)
    rmse_by_group_df = merged_df.groupby(group_cols).agg(
        rmse=("y_diff", lambda x: np.sqrt((x ** 2).sum() / x.size)),
        rating_count=("energy_star_rating", "count"),
        rating_frac=("energy_star_rating", lambda x: x.count() / x.size),
    )

    return rmse_by_group_df.reset_index()

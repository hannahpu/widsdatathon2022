#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 20:21:10 2022

@author: diana
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def clean_impute_data(df, impute_col='year_built', impute_thresh=1800, factors_cols=['facility_type', 'state_factor']):
    # First, impute any 0, negative, or missing values
    agg_df = df.groupby(factors_cols)[impute_col].median(
    ).reset_index(name=f'median_{impute_col}')
    agg_df = df.merge(agg_df, on=factors_cols, how='left')
    bad_entries = (df[impute_col] <= 0) | (df[impute_col].isna())
    df.loc[bad_entries, impute_col] = agg_df.loc[bad_entries,
                                                 f'median_{impute_col}']

    # Next, make floor year set to threshold
    if impute_thresh:
        df.loc[df[impute_col] < impute_thresh, impute_col] = impute_thresh
    return df


def process_data(df, cols_to_reduce):
    df = standardize_colnames(df)
    # Make floor area into log
    df['floor_area'] = np.log10(df['floor_area']+1)

    impute_dict = {"year_built": ['facility_type', 'state_factor']}
    thresh_dict = {"year_built": 1800}
    for impute_col, factors_cols in impute_dict.items():
        df = clean_impute_data(
            df, impute_col=impute_col, impute_thresh=thresh_dict[impute_col], factors_cols=factors_cols)
        
    df, pca_cols = reduce_dimensions(df, cols_to_reduce)
    return df, pca_cols


def standardize_colnames(df):
    df.columns = df.columns.str.lower()
    return df


def reduce_dimensions(df, cols_to_reduce, prefix='temp', n_components=9):
    pca = PCA(n_components=n_components)
    x = df[cols_to_reduce]
    x_pca = pca.fit_transform(x)
    pca_cols = [f"{prefix}_pca{i+1}" for i in range(n_components)]
    x_pca_df = pd.DataFrame(x_pca, columns=pca_cols)
    df = pd.concat([df.drop(columns=cols_to_reduce), x_pca_df], axis=1)
    return df, pca_cols
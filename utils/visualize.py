import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
import seaborn as sns


temp_col_list = [
    "january_min_temp",
    "january_avg_temp",
    "january_max_temp",
    "february_min_temp",
    "february_avg_temp",
    "february_max_temp",
    "march_min_temp",
    "march_avg_temp",
    "march_max_temp",
    "april_min_temp",
    "april_avg_temp",
    "april_max_temp",
    "may_min_temp",
    "may_avg_temp",
    "may_max_temp",
    "june_min_temp",
    "june_avg_temp",
    "june_max_temp",
    "july_min_temp",
    "july_avg_temp",
    "july_max_temp",
    "august_min_temp",
    "august_avg_temp",
    "august_max_temp",
    "september_min_temp",
    "september_avg_temp",
    "september_max_temp",
    "october_min_temp",
    "october_avg_temp",
    "october_max_temp",
    "november_min_temp",
    "november_avg_temp",
    "november_max_temp",
    "december_min_temp",
    "december_avg_temp",
    "december_max_temp",
    "avg_temp",
]
wind_col_list = [
    "direction_max_wind_speed",
    "direction_peak_wind_speed",
    "max_wind_speed",
]
days_col_list = [
    "cooling_degree_days",
    "heating_degree_days",
    "days_below_30f",
    "days_below_20f",
    "days_below_10f",
    "days_below_0f",
    "days_above_80f",
    "days_above_90f",
    "days_above_100f",
    "days_above_110f",
    "days_with_fog",
]
inch_col_list = ["precipitation_inches", "snowfall_inches", "snowdepth_inches"]


def quick_visualize_raw_features(train_df: pd.DataFrame, figsize: tuple = (10, 6)):
    # Numerical/continuous cols
    plot_dict = {
        "temp": temp_col_list,
        "days": days_col_list,
        "wind_speed": wind_col_list,
        "inches": inch_col_list,
    }
    for unit, col_list in plot_dict.items():
        plt.figure(figsize=figsize)
        melt_df = pd.melt(
            train_df,
            id_vars="id",
            value_vars=col_list,
            var_name="variable",
            value_name=unit,
        )
        ax = sns.boxplot(data=melt_df, x="variable", y=unit)
        _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # Other continuous variables
    for colname in [
        "floor_area",
        "elevation",
        "site_eui",
        "year_built",
        "energy_star_rating",
    ]:
        print(train_df[colname].describe())
        print(f"NaN count: {train_df[colname].isna().sum()}")
        log_scale = False
        if train_df[colname].max() - train_df[colname].min() > 3000:
            log_scale = True
        plt.figure(figsize=figsize)
        ax = sns.histplot(train_df, x=colname, log_scale=(log_scale, False))

    # Look at span of discrete/categorical cols
    discrete_col_list = [
        "year_factor",
        "state_factor",
        "building_class",
        "facility_type",
    ]
    for colname in discrete_col_list:
        print(train_df[colname].unique())
        plt.figure(figsize=figsize)
        ax = sns.countplot(data=train_df, x=colname)
        _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    return


def plot_catboost_feature_importance(model):
    sorted_feature_importance = model.feature_importances_.argsort()
    plt.figure(figsize=(10, 26))
    plt.barh(
        np.array(model.feature_names_)[sorted_feature_importance],
        model.feature_importances_[sorted_feature_importance],
    )
    plt.xlabel("CatBoost Feature Importance")
    return


def plot_rmse_by_group(
    rmse_df, rmse_col="rmse", group_col="facility_type", aux_col_list=["rating_frac"]
):
    ncols = len(aux_col_list) + 1
    fig, axes = plt.subplots(figsize=(10, 20), ncols=ncols)
    sns.barplot(
        data=rmse_df.sort_values(rmse_col, ascending=False),
        y=group_col,
        x=rmse_col,
        ax=axes[0],
    )
    for i, aux_col in enumerate(aux_col_list):
        sns.barplot(
            data=rmse_df.sort_values(rmse_col, ascending=False),
            y=group_col,
            x=aux_col,
            ax=axes[i + 1],
        )
        axes[i + 1].set_yticklabels([])
        axes[i + 1].set_ylabel("")
    return


def add_median_labels(ax, precision=".1f"):
    # https://stackoverflow.com/questions/38649501/labeling-boxplot-in-seaborn-with-median-value
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == "PathPatch"]
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4 : len(lines) : lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(
            x,
            y,
            f"{value:{precision}}",
            ha="center",
            va="center",
            fontweight="bold",
            color="white",
        )
        # create median-colored border around white text for contrast
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=3, foreground=median.get_color()),
                path_effects.Normal(),
            ]
        )
from sklearn.ensemble import RandomForestRegressor


model_setting_base_v1 = {
    # most correlated feature
    "features": ["energy_star_rating"],
    "model": RandomForestRegressor(max_depth=2, random_state=0),
    "if_scale": True,
}
model_setting_base_v2 = {
    # default all non missing numeric features
    "features": [
        "floor_area",
        "elevation",
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
        "cooling_degree_days",
        "heating_degree_days",
        "precipitation_inches",
        "snowfall_inches",
        "snowdepth_inches",
        "avg_temp",
        "days_below_30f",
        "days_below_20f",
        "days_below_10f",
        "days_below_0f",
        "days_above_80f",
        "days_above_90f",
        "days_above_100f",
        "days_above_110f",
    ],
    "model": RandomForestRegressor(max_depth=2, random_state=0),
    "if_scale": True,
}


all_model_setting_base_v1 = {
    "model_setting_base_v1": model_setting_base_v1,
    "model_setting_base_v2": model_setting_base_v2,
}

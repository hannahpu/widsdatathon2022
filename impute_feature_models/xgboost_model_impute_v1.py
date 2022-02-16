import sys

import pandas as pd
import xgboost as xgb

sys.path.append("..")
import global_vars as gv
from utils import model_utils as mu
from utils.data_utils import *
from utils.visualize import *
from utils.data_process_utils import *

#### read in data
test_df = pd.read_csv("../data/test.csv")
print(f"Test dimension: {test_df.shape}")
train_df = pd.read_csv("../data/train.csv")
print(f"Train dimension: {train_df.shape}")
train_df.columns = train_df.columns.str.lower()
test_df.columns = test_df.columns.str.lower()
# combine for predict missing value
all_model_df = pd.concat(
    [train_df.assign(source="train"), test_df.assign(source="test")]
).reset_index(drop=True)
all_model_df = all_model_df.drop(columns=["site_eui"])

## define some columns
# TODO: write function to categorize "facility_type"
missing_columns = train_df.columns[train_df.isnull().sum() > 0]
leave_year_out_column = "year_factor"
id_column = "id"

# model config
feature_dict = {
    "energy_star_rating_by_year_non_weather_onehot_xgb": {
        "log10_transform_cols": ["floor_area"],
        "if_one_hot": True,
        "features": [
            "state_factor",
            "building_class",
            "facility_type",
            "floor_area",
            "year_built",
            "elevation",
        ],
        "backfill_dict": {},
        "response": "energy_star_rating",
    },
}

config_name = "energy_star_rating_by_year_non_weather_onehot_xgb"
response = feature_dict[config_name]["response"]
log10_transform_cols = feature_dict[config_name]["log10_transform_cols"]
backfill_dict = feature_dict[config_name]["backfill_dict"]
if_scale = False
if_one_hot = feature_dict[config_name]["if_one_hot"]
model = xgb.XGBRegressor()

# subset data to train for imputation
train_filter_df = all_model_df.query(f"{response}.notnull()").reset_index(drop=True)
test_filter_df = all_model_df.query(f"{response}.isnull()").reset_index(drop=True)
all_year_factor = train_filter_df[leave_year_out_column].unique()

# process data
train_filter_df = log_transform(train_filter_df, log10_transform_cols)
test_filter_df = log_transform(test_filter_df, log10_transform_cols)
impute_dict = {"year_built": ["facility_type", "state_factor"]}
thresh_dict = {"year_built": 1800}
for impute_col, factors_cols in impute_dict.items():
    train_filter_df, test_filter_df = clean_impute_data(
        train_filter_df,
        test_filter_df,
        impute_col=impute_col,
        impute_thresh=thresh_dict[impute_col],
        factors_cols=factors_cols,
    )


# Set feature columns after data transformations
features_columns = list(set(feature_dict[config_name]["features"]) - set(log10_transform_cols)) + [
    f"log10_{col}" for col in log10_transform_cols
]
print(config_name, features_columns, if_one_hot)

### predict by year
(
    model_loy_rmse,
    loy_prediction_result_train_dict,
    loy_prediction_result_test_dict,
) = mu.run_leave_year_out(
    model_df=train_filter_df,
    ml_model=model,
    features_columns=features_columns,
    if_scale_data=if_scale,
    if_one_hot=if_one_hot,
    model_type="sklearn",
    response_col=response,
    if_output_prediction_results=True,
)
print(f"Average RMSE:\n{model_loy_rmse.mean()}")
display(model_loy_rmse)

all_loy_train_predict_df, all_loy_test_predict_df = mu.process_loy_train_test_prediction(
    loy_prediction_result_train_dict, loy_prediction_result_test_dict, train_filter_df
)

# run model predict to predict missing data
all_test_prediction_result = mu.run_model_predict_unknown_test_by_column(
    train_df=train_filter_df,
    test_df=test_filter_df,
    full_data_df=train_df,
    features_columns=features_columns,
    response_col=response,
    if_scale=if_scale,
    if_one_hot=if_one_hot,
    model=model,
)

# save result
model_loy_rmse["method"] = config_name
model_loy_rmse.to_csv(f"../feature_impute_data/{config_name}_loy_rmse.csv", index=False)
all_test_prediction_result.to_csv(
    f"../feature_impute_data/{config_name}_test_prediction.csv", index=False
)
all_loy_train_predict_df.to_csv(
    f"../feature_impute_data/{config_name}_loy_train_predict.csv", index=False
)
all_loy_test_predict_df.to_csv(
    f"../feature_impute_data/{config_name}_loy_test_predict.csv", index=False
)

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
    "energy_star_rating_by_year_log_temp_pca_onehot_xgb": {
        "cols_to_reduce": temp_col_list,
        "log10_transform_cols": ["floor_area"],
        "if_one_hot": True,
        "backfill_dict": {},
        "response": "energy_star_rating",
    },
}
config_name = "energy_star_rating_by_year_log_temp_pca_onehot_xgb"
response = feature_dict[config_name]["response"]
cols_to_reduce = feature_dict[config_name]["cols_to_reduce"]
log10_transform_cols = feature_dict[config_name]["log10_transform_cols"]
backfill_dict = feature_dict[config_name]["backfill_dict"]
if_scale = False
if_one_hot = feature_dict[config_name]["if_one_hot"]
model = xgb.XGBRegressor()

# subset data to train for imputation
train_impute_df = all_model_df.query(f"{response}.notnull()").reset_index(drop=True)
test_impute_df = all_model_df.query(f"{response}.isnull()").reset_index(drop=True)
all_year_factor = train_impute_df[leave_year_out_column].unique()

# process data
train_filter_df, test_filter_df, pca_cols = process_data(
    train_impute_df.drop_duplicates(),
    test_impute_df.drop_duplicates(),
    cols_to_reduce,
    log10_transform_cols,
    0,
    backfill_dict,
)

# Set feature columns after data transformations
features_columns = (
    list(
        set(gv.all_feature_columns)
        - set(cols_to_reduce)
        - set(log10_transform_cols)
        - set([response])
    )
    + pca_cols
    + [f"log10_{col}" for col in log10_transform_cols]
)
if backfill_dict:
    backfill_cols = list(backfill_dict.keys())
    features_columns = list(set(features_columns) - set(backfill_cols)) + [
        f"backfilled_{col}" for col in backfill_cols
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

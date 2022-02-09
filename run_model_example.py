"""
This script is an example to run model and save the prediction result 
using the modeling utils and model_config file.

Note: model_config is not necessary to use, but will need to specify object names
in Line 29-35.
"""


import sys

import pandas as pd

import global_vars as gv
from model_config import all_model_setting_base_v1 as mconfig
from utils import model_utils as mu

#### read in data
test_df = pd.read_csv("data/test.csv")
print(f"Test dimension: {test_df.shape}")
train_df = pd.read_csv("data/train.csv")
print(f"Train dimension: {train_df.shape}")
sample_solution_df = pd.read_csv("data/sample_solution.csv")
print(f"Sample solution dimension: {sample_solution_df.shape}")
train_df.columns = train_df.columns.str.lower()
test_df.columns = test_df.columns.str.lower()

##### get config
# note: you don't have to pull from config file,
# but you will need these object names to run model and save result to csv
# config name will be used as file name for test prediction result
config_name = "model_setting_base_v2"
features_columns = mconfig[config_name]["features"]
model = mconfig[config_name]["model"]
if_scale = mconfig[config_name]["if_scale"]

#### get model ready data
train_filter_df = train_df.query("state_factor != 'State_8'")[
    features_columns + [gv.year_columns] + [gv.response_column]
].dropna(how="any")
print(f"Original number of data data: {train_df.shape}")
print(f"Final number of training data: {train_filter_df.shape}")

## Run LOY model
random_forest_rmse = mu.run_leave_year_out(
    model_df=train_filter_df,
    sklearn_model=model,
    features_columns=features_columns,
    if_scale_data=if_scale,
)
print(f"Average RMSE:\n{random_forest_rmse.mean()}")
display(random_forest_rmse)

## predict on test data
train_filter_x_df, train_filter_y_df = mu.split_model_feature_response(
    train_filter_df, features_columns
)
test_x_df = mu.split_model_feature_response(test_df, features_columns, if_with_response=False)
processed_train_x_df, processed_test_x_df = mu.process_train_test_data(
    train_filter_x_df, test_x_df, if_scale
)
train_predict, test_predict = mu.run_model(
    model, processed_train_x_df, train_filter_y_df, processed_test_x_df
)
training_rmse = mu.calculate_rmse(train_filter_y_df, train_predict)
print(f"Whole data train RMSE: {training_rmse}")

## output save result
test_prediction_result = test_df[["id"]]
test_prediction_result["site_eui"] = test_predict
test_prediction_result.head()
test_prediction_result.to_csv(f"prediction_result/{config_name}.csv", index=False)

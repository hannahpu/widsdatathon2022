from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


def run_leave_year_out(
    model_df,
    ml_model,
    features_columns,
    if_scale_data,
    if_one_hot,
    model_type="sklearn",
):
    # Define which function to run
    run_model_dict = {"sklearn": run_sklearn_model, "catboost": run_catboost_model}
    assert model_type in run_model_dict.keys(), f"{model_type} not in {run_model_dict.keys()}"
    all_loy_model_result = []
    all_year = model_df["year_factor"].unique()
    print(f"Running {model_type}")
    for one_year in all_year:
        print(f"Modeling {one_year}...")
        (
            left_out_test_x_df,
            left_out_test_y_df,
            left_out_train_x_df,
            left_out_train_y_df,
        ) = train_test_split(one_year, model_df, features_columns)
        left_out_train_x_df, left_out_test_x_df = process_train_test_data(
            left_out_train_x_df, left_out_test_x_df, if_scale_data, if_one_hot, model_df
        )
        train_predict, test_predict, fitted_model = run_model_dict[model_type](
            ml_model, left_out_train_x_df, left_out_train_y_df, left_out_test_x_df
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
    all_loy_model_result_df = pd.concat(all_loy_model_result).reset_index(drop=True)
    return all_loy_model_result_df


def train_test_split(level, model_df, features_columns):
    left_out_test = model_df.query(f"year_factor == {level}")
    left_out_train = model_df.query(f"year_factor != {level}")
    left_out_test_x_df, left_out_test_y_df = split_model_feature_response(
        left_out_test, features_columns
    )
    left_out_train_x_df, left_out_train_y_df = split_model_feature_response(
        left_out_train, features_columns
    )
    return (
        left_out_test_x_df,
        left_out_test_y_df,
        left_out_train_x_df,
        left_out_train_y_df,
    )


def split_model_feature_response(model_df, features_columns, if_with_response=True):
    model_x_df = model_df[features_columns]
    if if_with_response:
        model_y_df = model_df["site_eui"]
        return model_x_df, model_y_df
    else:
        return model_x_df


def process_train_test_data(train_x_df, test_x_df, if_scale_data, if_one_hot, full_data_df):
    if if_one_hot:
        categorical_columns_to_dummy = output_non_numeric_columns(train_x_df)
        print(f"Columns to be dummied: {categorical_columns_to_dummy}")
        for col in categorical_columns_to_dummy:
            # encoder = get_one_hot_encoder(train_x_df[[col]])
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
            train_x_df = pd.concat([train_x_df, train_one_hot_encoded], axis="columns")
            test_x_df = pd.concat([test_x_df, test_one_hot_encoded], axis="columns")
        train_x_df = train_x_df.drop(columns=categorical_columns_to_dummy)
        test_x_df = test_x_df.drop(columns=categorical_columns_to_dummy)
    if if_scale_data:
        train_x_df, test_x_df = scale_data(train_x_df, test_x_df)
    return train_x_df, test_x_df


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
    scaled_train_x = pd.DataFrame(scaled_train_x, columns=train_x.columns, index=train_x.index)
    scaled_test_x = pd.DataFrame(scaled_test_x, columns=test_x.columns, index=test_x.index)
    return scaled_train_x, scaled_test_x


def get_one_hot_encoder(train_df):
    enc = OneHotEncoder(sparse=False)
    return enc.fit(train_df)


def run_sklearn_model(sklearn_model, train_x_df, train_y_df, test_x_df):
    fitted_model = fit_sklearn_model(sklearn_model, train_x_df, train_y_df)
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


def run_catboost_model(model, train_x_df, train_y_df, test_x_df):
    cat_columns = train_x_df.select_dtypes(["O"]).columns.tolist()
    model.fit(train_x_df, y=train_y_df, cat_features=cat_columns)
    train_predict = model.predict(train_x_df)
    test_predict = model.predict(test_x_df)
    return train_predict, test_predict

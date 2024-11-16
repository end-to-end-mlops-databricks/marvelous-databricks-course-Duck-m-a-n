# Databricks notebook source
# Install required packages
# MAGIC %pip install m5_forecasting-0.0.1-py3-none-any.whl

# COMMAND ----------
# Restart Python to ensure the new packages are loaded
dbutils.library.restartPython()

# COMMAND ----------
# Import necessary libraries
import yaml
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import mlflow
from mlflow.models import infer_signature
import mlforecast.flavor

from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

import lightgbm as lgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean

from utilsforecast.losses import mse, mae, mape, smape
from utilsforecast.evaluation import evaluate
from utilsforecast.plotting import plot_series

from m5_forecasting.config import Config

import pandas as pd

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()

# Set MLflow tracking URIs
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
predefined_unique_ids = [
    "FOODS_3_595_CA_1",
    "HOBBIES_1_001_CA_1",
    # Unique_ids added to test department forecasts
    "HOBBIES_1_023_CA_1",
    "FOODS_2_382_CA_1"
]

dbutils.widgets.multiselect(
    name="unique_ids",
    defaultValue=predefined_unique_ids[0],
    choices=predefined_unique_ids,
    label="Select Unique IDs"
)
# COMMAND ----------
# Get the selected unique_ids from the widget
selected_unique_ids = dbutils.widgets.get("unique_ids")

# Split the selected unique_ids into a list
unique_ids = selected_unique_ids.split(",")

# Trim any whitespace from the unique_ids
unique_ids = [uid.strip() for uid in unique_ids if uid.strip()]

# COMMAND -----------
# Load configuration
config = Config.from_yaml("../../configs/project_config.yml")

# Access configurations
catalog_name = config.catalog_name
schema_name = config.schema_name
cat_features = config.cat_features
target = config.target
params = config.parameters
lag_transforms_config = params.lag_transforms

# Define lag_transforms
lag_transforms = {}
for lag_str, transforms in lag_transforms_config.items():
    lag = int(lag_str)  # Ensure the lag is an integer
    transform_list = []
    for transform in transforms:
        transform_type = transform['type']
        transform_params = transform.get('params', {})
        if transform_type == "ExpandingMean":
            transform_instance = ExpandingMean(**transform_params)
        elif transform_type == "RollingMean":
            transform_instance = RollingMean(**transform_params)
        else:
            raise ValueError(f"Unsupported transform type: {transform_type}")
        transform_list.append(transform_instance)
    lag_transforms[lag] = transform_list

# COMMAND ----------
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").filter(F.col("unique_id").isin(unique_ids))
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").filter(F.col("unique_id").isin(unique_ids))

train_df = train_set.toPandas()
test_df = test_set.toPandas()

# COMMAND ----------
# Prepare training and testing data
X_train = train_df.drop(columns="update_timestamp_utc")
X_df = test_df.drop(columns=["y", "update_timestamp_utc"])
y_test = test_df[["unique_id", "ds", "y"]]

# Convert integer columns to float64 to handle potential missing values
int_cols = X_train.select_dtypes(include=['int']).columns.tolist()
X_train[int_cols] = X_train[int_cols].astype('float64')
int_cols_test = X_df.select_dtypes(include=['int']).columns.tolist()
X_df[int_cols_test] = X_df[int_cols_test].astype('float64')

# COMMAND ----------
# Define the preprocessor using cat_features from the configuration
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(), cat_features)
    ],
    remainder='passthrough'
)

# Create the pipeline with preprocessing and the LightGBM regressor
model = lgb.LGBMRegressor(**params.hyperparameters)
pipeline_model = make_pipeline(preprocessor, model)

# Initialize the MLForecast object with the pipeline model and model name
fcst = MLForecast(
    models={'LGBMRegressor': pipeline_model},
    freq=params.freq,
    lags=params.lags,
    lag_transforms=lag_transforms,
    num_threads=params.num_threads,
)

# COMMAND ----------
# Set up MLflow experiment and start a run
mlflow.set_experiment("/Shared/m5_forecasting_experiment")
git_sha = "5d53908cc7b4f89b30dfbd5c3355c72076b8d2fb"  # Update with latest git commit hash

with mlflow.start_run(
    tags={"git_sha": f"{git_sha}", "branch": "week2"}
) as run:
    run_id = run.info.run_id

    # Fit the model
    fcst.fit(
        X_train,
        id_col='unique_id',
        time_col='ds',
        target_col=target,
        static_features=[]
    )

    # Generate predictions
    y_pred = fcst.predict(h=28, X_df=X_df)
    df_eval = y_pred.merge(y_test, on=['unique_id', 'ds'])

    # Evaluate the forecasts
    metrics = [mse, mae, mape, smape]
    model_names = ['LGBMRegressor']

    evaluation = evaluate(
        df=df_eval,
        metrics=metrics,
        models=model_names,
        train_df=train_df,
        id_col='unique_id',
        time_col='ds',
        target_col=target,
    )

    # Log parameters and metrics
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(params.hyperparameters)
    mlflow.log_param("unique_ids", unique_ids)
    for _, row in evaluation.iterrows():
        metric = row['metric']
        value = row['LGBMRegressor']
        mlflow.log_metric(metric, value)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Log training data
    dataset = mlflow.data.from_spark(
        train_set, table_name=f"{catalog_name}.{schema_name}.train_set",
        version="0")
    mlflow.log_input(dataset, context="training")

    # Generate and log the plot as an interactive HTML
    fig = plot_series(forecasts_df=df_eval, engine="plotly")
    fig.update_layout(title=f"Forecast")
    fig_html = f"forecast_plot.html"
    fig.write_html(fig_html)
    mlflow.log_artifact(fig_html, artifact_path="plots")
    
    # Log the model
    mlforecast.flavor.log_model(
        model=fcst,
        artifact_path="lightgbm-pipeline-model",
        code_paths=["m5_forecasting-0.0.1-py3-none-any.whl"],
        signature=signature
    )

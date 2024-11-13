# Databricks notebook source
# MAGIC %pip install m5_forecasting-0.0.1-py3-none-any.whl

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import yaml
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import mlflow
from mlflow.models import infer_signature

from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

import lightgbm as lgb

import mlforecast.flavor
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean

from utilsforecast.losses import mse, mae, mape, smape
from utilsforecast.evaluation import evaluate

from m5_forecasting.config import Config
from m5_forecasting.preprocessing.data_processor import DataProcessor

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
predefined_unique_ids = [
    "FOODS_3_595_CA_1",
    "HOBBIES_1_001_CA_1"
]

dbutils.widgets.multiselect(
    name="unique_ids",
    defaultValue=predefined_unique_ids[0],
    choices=predefined_unique_ids,
    label="Select Unique IDs"
)
# COMMAND ----------
config = Config.from_yaml("../../configs/project_config.yml")

# Get the selected unique_ids from the widget
selected_unique_ids = dbutils.widgets.get("unique_ids")

# Split the selected unique_ids into a list
unique_ids = selected_unique_ids.split(",")

# Trim any whitespace from the unique_ids
unique_ids = [uid.strip() for uid in unique_ids if uid.strip()]

catalog_name = config.catalog_name
schema_name = config.schema_name
num_features = config.processed_features.num_features
cat_features = config.processed_features.cat_features
engineered_features = config.processed_features.engineered_features
date_features = config.processed_features.date_features
static_features = config.processed_features.static_features
target = config.target
params = config.parameters.init
lag_transforms_config = params.lag_transforms

lag_transforms = {}
for lag, transforms in lag_transforms_config.items():
    transform_list = []
    for transform in transforms:
        transform_type = transform['type']
        transform_params = transform['params']
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
X_train = train_df.drop(columns="update_timestamp_utc")

X_df = test_df.drop(columns=["y", "update_timestamp_utc"])
y_test = test_df[["unique_id", "ds", "y"]]

# COMMAND ----------
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(), ['state_id', 'store_id', 'cat_id', 'dept_id', 'item_id', 'conditions'])
    ],
    remainder='passthrough'
)

model = lgb.LGBMRegressor(**params.hyperparameters)
pipeline_model = make_pipeline(preprocessor, model)

fcst = MLForecast(
    models={'LGBMRegressor': pipeline_model},
    freq=params.freq,
    lags=params.lags,
    lag_transforms=lag_transforms,
    num_threads=params.num_threads,
)

# COMMAND ----------
mlflow.set_experiment("/Shared/m5_forecasting_experiment")
git_sha = "ffa63b430205ff7"

with mlflow.start_run(
    tags = {"git_sha": f"{git_sha}",
            "branch": "week2"}
) as run:
    run_id = run.info.run_id

    fcst.fit(
        X_train,
        id_col='unique_id',
        time_col='ds',
        target_col="y",
        static_features=[]
    )

    y_pred = fcst.predict(h=28, X_df = X_df)
    df_eval = y_pred.merge(y_test, on=['unique_id', 'ds'])

    metrics = [mse, mae, mape, smape]
    model = ['LGBMRegressor']

    evaluation = evaluate(
        df=df_eval,
        metrics=metrics,
        models=model,
        train_df=train_df,
        id_col='unique_id',
        time_col='ds',
        target_col='y',
    )

    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(params.hyperparameters)
    mlflow.log_param("unique_ids", unique_ids)
    for _, row in evaluation.iterrows():
        metric = row['metric']
        value = row['LGBMRegressor']
        mlflow.log_metric(metric, value)
    signature = infer_signature(model_input = X_train, model_output=y_pred)

    dataset = mlflow.data.from_spark(
    train_set, table_name=f"{catalog_name}.{schema_name}.train_set",
    version="0")
    mlflow.log_input(dataset, context="training")

    mlforecast.flavor.log_model(
        model=fcst,
        artifact_path="lightgbm-pipeline-model",
        code_paths = ["m5_forecasting-0.0.1-py3-none-any.whl"],
        signature=signature
    )

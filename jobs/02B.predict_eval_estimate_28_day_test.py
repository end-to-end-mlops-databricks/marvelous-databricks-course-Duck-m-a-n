# 02B.predict_eval_estimate_28_day_test.py

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import mlflow
from mlflow.models import infer_signature
import mlforecast.flavor
from mlflow import MlflowClient

from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

import lightgbm as lgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean

from utilsforecast.losses import rmse, mse, mae, mape, smape
from utilsforecast.evaluation import evaluate
from utilsforecast.plotting import plot_series

from m5_forecasting.config import Config

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

def main():
    spark = SparkSession.builder.getOrCreate()
    client = MlflowClient()

    # Load Parameters
    dept_store_id = sys.argv[1]
    job_run_date = dbutils.jobs.taskValues.get(taskKey="Preprocess28DayTestForecasts", key="job_run_date")
    exp_id = dbutils.jobs.taskValues.get(taskKey="Preprocess28DayTestForecasts", key="exp_id")
    parent_run_id = dbutils.jobs.taskValues.get(taskKey="Preprocess28DayTestForecasts", key="parent_run_id")
    dept_id, store_id = dept_store_id.split('-')
    
    # Load Configurations
    config = Config.from_yaml("../configs/project_config.yml")
    catalog_name = config.catalog_name
    schema_name = config.schema_name
    predefined_unique_ids = config.predefined_unique_ids
    horizon = config.horizon
    target = config.target
    params = config.parameters
    cat_features = config.cat_features
    lag_transforms_config = params.lag_transforms
    train_data = "train_set"
    test_data = "test_set"
    temp_prediction_table = "intermediate_28_day_test_set_predictions"
    print(f"Catalog: {catalog_name}, Schema: {schema_name}")
    print(f"Cat features: {cat_features}, Target: {target}, Horizon: {horizon}")

    # Define Git Configuration, Experiment/Run tags
    run_name = f"Training_{dept_store_id}"
    git_sha = "5d53908cc7b4f89b30dfbd5c3355c72076b8d2fb"
    branch = "week5"
    component = "training_child_run"
    run_tags = {"git_sha": f"{git_sha}", "branch": f"{branch}", "dept_store_id": f"{dept_store_id}", "component": f"{component}"}
    print(f"Processing department: {dept_id}, store: {store_id} for Experiment ID: {exp_id} with Parent run: {parent_run_id} on Job run date {job_run_date} ")

    # Define Evaluation Configurations
    metrics = [rmse, mse, mae, mape, smape]
    model_name = "LGBMRegressor"
    named_step_transformer = "ordinal" #"columntransformer"
    named_step_model = "lgbmregressor"

    # Define Log & Registered Model Configurations
    artifact_path = f"lightgbm-pipeline-updated-model-dept-{dept_store_id}"
    code_paths = ["../dist/m5_forecasting-0.0.1-py3-none-any.whl"]
    model_name_uc = f"{catalog_name}.{schema_name}.ForecastModel_{dept_store_id}"
    model_tags = {"git_sha": f"{git_sha}"}

    # Load Train, Test, Train Version
    train_set = spark.table(f"{catalog_name}.{schema_name}.{train_data}") \
        .filter(F.col("dept_id") == dept_id) \
        .filter(F.col("store_id") == store_id) \
        .filter(F.col("unique_id").isin(predefined_unique_ids)) \
        .drop("update_timestamp_utc")

    train_set_version = (
        spark.sql(f"DESCRIBE HISTORY {catalog_name}.{schema_name}.{train_data}")
        .select(F.max(F.col("version")).alias("latest_version"))
        .collect()[0]["latest_version"]
)
    test_set = spark.table(f"{catalog_name}.{schema_name}.{test_data}") \
        .filter(F.col("dept_id") == dept_id) \
        .filter(F.col("store_id") == store_id) \
        .filter(F.col("unique_id").isin(predefined_unique_ids)) \
        .drop("update_timestamp_utc")

    # Convert to Pandas DataFrame
    train_df = train_set.toPandas()
    test_df = test_set.toPandas()

    # Prepare training and testing data
    X_train = train_df.drop(columns=["state_id", "store_id", "cat_id", "dept_id", "item_id"])
    X_test = test_df.drop(columns=[f"{target}" ,"state_id", "store_id", "cat_id", "dept_id", "item_id"])
    y_test = test_df[["unique_id", "ds", "y"]]

    # Define lag_transforms
    lag_transforms = {}
    for lag_str, transforms in lag_transforms_config.items():
        lag = int(lag_str)
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

    # Define Preprocessor, Create Pipeline, Intialize MLForecast object
    def convert_to_float(X):
        return X.astype(np.float64)

    preprocessor = Pipeline(steps=[
    ('ordinal', ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(), ['conditions'])
        ],
        remainder='passthrough'
        )),
    ('convert_to_float', FunctionTransformer(convert_to_float, validate=False))
        ])

    model = lgb.LGBMRegressor(**params.hyperparameters)
    pipeline_model = make_pipeline(preprocessor, model)

    fcst = MLForecast(
        models={model_name: pipeline_model},
        freq=params.freq,
        lags=params.lags,
        lag_transforms=lag_transforms,
        num_threads=params.num_threads,
    )

    # Start MLFlow Training, Evaluating, Logging, Registering
    with mlflow.start_run(
        experiment_id=exp_id,
        parent_run_id=parent_run_id,
        run_name=run_name,
        tags=run_tags,
        nested=True
    ) as training_child_run:
        training_child_run_id = training_child_run.info.run_id
        print(f"MLflow run started with child_run_id: {training_child_run_id}")

        fcst.fit(
            X_train,
            id_col='unique_id',
            time_col='ds',
            target_col=target,
            static_features=[]
        )

        # Predict & Save Predictions
        y_pred = fcst.predict(h=horizon, X_df=X_test)

        preds_spark = spark.createDataFrame(y_pred)
        preds_spark.createOrReplaceTempView(temp_prediction_table)
        spark.sql(f"""
            INSERT INTO {catalog_name}.{schema_name}.{temp_prediction_table}
            SELECT * FROM {temp_prediction_table}
        """)

        # Evaluate Predictions on Test Set
        df_eval = y_pred.merge(y_test, on=['unique_id', 'ds'])

        evaluation = evaluate(
            df=df_eval,
            metrics=metrics,
            models=[model_name],
            train_df=train_df,
            id_col='unique_id',
            time_col='ds',
            target_col=target,
        )

        # Log parameters and metrics
        mlflow.log_param("model_type", f"{model_name} with preprocessing")
        mlflow.log_params(params.hyperparameters)
        mlflow.log_param("dept_id", dept_id)
        mlflow.log_param("store_id", store_id)
        for _, row in evaluation.iterrows():
            metric = row['metric']
            value = row[model_name]
            mlflow.log_metric(metric, value)

        # Log feature importance
        preprocessor = fcst.models_[model_name].named_steps['pipeline']
        ordinal_transformer = preprocessor.named_steps[named_step_transformer]
        model = fcst.models_[model_name].named_steps[named_step_model]
        feature_names = ordinal_transformer.get_feature_names_out()
        feature_importances = model.feature_importances_

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        importance_df.reset_index(drop=True, inplace=True)
        importance_df['Position'] = importance_df.index

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance_df['Position'], importance_df['Importance'])
        ax.set_yticks(importance_df['Position'])
        ax.set_yticklabels(importance_df['Feature'])
        ax.invert_yaxis()
        ax.set_title('Feature Importances')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Features')
        fig.tight_layout()

        mlflow.log_figure(fig, artifact_file=f"plots/{dept_store_id}_Feature_Importance.png")
        plt.close(fig)

        # Generate and log the forecast plot
        fig = plot_series(df=train_df, forecasts_df=df_eval, engine="plotly")
        mlflow.log_figure(fig, artifact_file=f"plots/{dept_store_id}_28_Day_Forecast.html")

        # Log training data
        dataset = mlflow.data.from_spark(
            train_set, table_name=f"{catalog_name}.{schema_name}.{train_data}",
            version=f"{train_set_version}")
        mlflow.log_input(dataset, context="training")

        # Update the model with the test set
        fcst.update(df=test_df)

        # Update & Log Model   X_train.drop(columns=target)
        signature = infer_signature(model_input=X_train.drop(columns=target), model_output=y_pred)

        mlforecast.flavor.log_model(
            model=fcst,
            artifact_path=artifact_path,
            code_paths=code_paths,
            signature=signature
        )

    # Register Updated Model
    mv = mlflow.register_model(
        model_uri = f"runs:/{training_child_run_id}/{artifact_path}",
        name=model_name_uc,
        tags=model_tags
        )
    
    # Create Alias for Registered Model
    client.set_registered_model_alias(f"{catalog_name}.{schema_name}.ForecastModel_{dept_store_id}", "champion", mv.version)

    print(f"Completed processing for department: {dept_id}, store: {store_id}")

if __name__ == "__main__":
    main()

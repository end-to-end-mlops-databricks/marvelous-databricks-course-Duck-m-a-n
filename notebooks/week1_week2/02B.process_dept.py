# process_dept.py

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

from utilsforecast.losses import rmse, mse, mae, mape, smape
from utilsforecast.evaluation import evaluate
from utilsforecast.plotting import plot_series

from m5_forecasting.config import Config

import pandas as pd
import sys

def main():
    print("Starting process_dept.py")
    # Initialize Spark session
    spark = SparkSession.builder.getOrCreate()
    print("Spark session initialized.")

    # Load configuration
    print("Loading configuration...")
    config = Config.from_yaml("../../configs/project_config.yml")
    print("Configuration loaded.")

    # Access configurations
    catalog_name = config.catalog_name
    schema_name = config.schema_name
    cat_features = config.cat_features
    horizon = config.horizon
    target = config.target
    params = config.parameters
    lag_transforms_config = params.lag_transforms
    print(f"Catalog: {catalog_name}, Schema: {schema_name}")
    print(f"Cat features: {cat_features}, Target: {target}, Horizon: {horizon}")
    print("Parameters and lag transforms configuration accessed.")

    # Define lag_transforms
    print("Defining lag_transforms...")
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
    print("Lag transforms defined.")

    # Get dept_id from command-line arguments
    if len(sys.argv) < 2:
        raise ValueError("dept_id parameter is missing")
    dept_id = sys.argv[1]

    print(f"Processing department: {dept_id}")

    predefined_unique_ids = [
        "HOBBIES_1_001_CA_1",  # full historical data
        "HOBBIES_1_023_CA_1",
        "FOODS_3_595_CA_1",    # represents shortest time series 100 timestamps
        "FOODS_3_238_CA_1",
        "FOODS_3_246_CA_1",    # full varying history not very much highs looks to represent the majority
        "HOUSEHOLD_1_146_CA_1",  # full history
        "HOUSEHOLD_1_178_CA_1",  # full history same store same state
        "HOUSEHOLD_1_056_CA_1",  # varying history same store same state
        "HOUSEHOLD_1_179_CA_2",  # full history high values, different store same state
    ]

    # Filter training and testing data for the current dept_id
    print("Filtering training and testing data...")
    train_set = spark.table(f"{catalog_name}.{schema_name}.train_set") \
        .filter(F.col("dept_id") == dept_id) \
        .filter(F.col("unique_id").isin(predefined_unique_ids))

    test_set = spark.table(f"{catalog_name}.{schema_name}.test_set") \
        .filter(F.col("dept_id") == dept_id) \
        .filter(F.col("unique_id").isin(predefined_unique_ids))

    train_count = train_set.count()
    test_count = test_set.count()
    print(f"Number of training records: {train_count}")
    print(f"Number of testing records: {test_count}")

    # Convert to Pandas DataFrame
    print("Converting Spark DataFrames to Pandas DataFrames...")
    train_df = train_set.toPandas()
    test_df = test_set.toPandas()
    print("Conversion to Pandas DataFrames completed.")

    # Check if data is available
    if train_df.empty:
        print(f"No data for department {dept_id}")
        return

    # Prepare training and testing data
    print("Preparing training and testing data...")
    X_train = train_df.drop(columns="update_timestamp_utc")
    X_df = test_df.drop(columns=["y", "update_timestamp_utc"])
    y_test = test_df[["unique_id", "ds", "y"]]
    print("Training and testing data prepared.")

    # Convert integer columns to float64 to handle potential missing values
    print("Converting integer columns to float64...")
    int_cols = X_train.select_dtypes(include=['int']).columns.tolist()
    X_train[int_cols] = X_train[int_cols].astype('float64')
    int_cols_test = X_df.select_dtypes(include=['int']).columns.tolist()
    X_df[int_cols_test] = X_df[int_cols_test].astype('float64')
    print("Conversion of integer columns completed.")

    # Define the preprocessor using cat_features from the configuration
    print("Defining preprocessor...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(), cat_features)
        ],
        remainder='passthrough'
    )
    print("Preprocessor defined.")

    # Create the pipeline with preprocessing and the LightGBM regressor
    print("Creating pipeline model...")
    model = lgb.LGBMRegressor(**params.hyperparameters)
    pipeline_model = make_pipeline(preprocessor, model)
    print("Pipeline model created.")

    # Initialize the MLForecast object with the pipeline model and model name
    print("Initializing MLForecast object...")
    fcst = MLForecast(
        models={'LGBMRegressor': pipeline_model},
        freq=params.freq,
        lags=params.lags,
        lag_transforms=lag_transforms,
        num_threads=params.num_threads,
    )
    print("MLForecast object initialized.")

    # Set up MLflow experiment and start a run
    print("Starting MLflow run...")
    mlflow.set_experiment("/Shared/m5_forecasting_experiment")
    git_sha = "5d53908cc7b4f89b30dfbd5c3355c72076b8d2fb"  # Update with latest git commit hash

    with mlflow.start_run(
        run_name=f"Forecast_dept_{dept_id}",
        tags={"git_sha": f"{git_sha}", "branch": "week2", "dept_id": dept_id}
    ) as run:
        run_id = run.info.run_id
        print(f"MLflow run started with run_id: {run_id}")

        # Fit the model
        print("Fitting the model...")
        fcst.fit(
            X_train,
            id_col='unique_id',
            time_col='ds',
            target_col=target,
            static_features=[]
        )
        print("Model fitting completed.")

        # Generate predictions
        print("Generating predictions...")
        y_pred = fcst.predict(h=horizon, X_df=X_df)
        print("Predictions generated.")
        df_eval = y_pred.merge(y_test, on=['unique_id', 'ds'])
        print("Predictions merged with test data.")

        # Evaluate the forecasts
        print("Evaluating forecasts...")
        metrics = [rmse, mse, mae, mape, smape]
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
        print("Forecast evaluation completed.")

        # Log parameters and metrics
        print("Logging parameters and metrics to MLflow...")
        mlflow.log_param("model_type", "LightGBM with preprocessing")
        mlflow.log_params(params.hyperparameters)
        mlflow.log_param("dept_id", dept_id)
        for _, row in evaluation.iterrows():
            metric = row['metric']
            value = row['LGBMRegressor']
            mlflow.log_metric(metric, value)
        print("Parameters and metrics logged.")

        signature = infer_signature(model_input=X_train, model_output=y_pred)
        print("Model signature inferred.")

        # Log training data
        print("Logging training data to MLflow...")
        dataset = mlflow.data.from_spark(
            train_set, table_name=f"{catalog_name}.{schema_name}.train_set",
            version="0")
        mlflow.log_input(dataset, context="training")
        print("Training data logged.")

        # Log the model
        print("Logging the model to MLflow...")
        mlforecast.flavor.log_model(
            model=fcst,
            artifact_path=f"lightgbm-pipeline-model-dept-{dept_id}",
            code_paths=["m5_forecasting-0.0.1-py3-none-any.whl"],
            signature=signature
        )
        print("Model logged to MLflow.")

        # Generate and log the plot as an interactive HTML
        print("Generating and logging the forecast plot...")
        fig = plot_series(forecasts_df=df_eval, engine="plotly")
        fig_html = f"forecast_plot_{dept_id}.html"
        fig.write_html(fig_html)
        mlflow.log_artifact(fig_html, artifact_path="plots")
        print("Forecast plot logged to MLflow.")

        # Save predictions to an intermediate table
        print("Saving predictions to intermediate_predictions table...")
        y_pred['dept_id'] = dept_id
        preds_spark = spark.createDataFrame(y_pred)
        preds_spark.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.intermediate_predictions")
        print("Predictions saved.")

    print(f"Completed processing for department: {dept_id}")

if __name__ == "__main__":
    main()

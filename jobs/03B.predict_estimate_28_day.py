# 03B.predict_estimate_28_day.py

import sys
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import mlflow
import mlforecast.flavor
from utilsforecast.plotting import plot_series
import nannyml as nml

from m5_forecasting.config import Config

def main():
    spark = SparkSession.builder.getOrCreate()

    # Load Parameters
    dept_store_id = sys.argv[1]
    job_run_date = dbutils.jobs.taskValues.get(taskKey="Preprocess28DayForecasts", key="job_run_date")
    exp_id = dbutils.jobs.taskValues.get(taskKey="Preprocess28DayForecasts", key="exp_id")
    parent_run_id = dbutils.jobs.taskValues.get(taskKey="Preprocess28DayForecasts", key="parent_run_id")
    dept_id, store_id = dept_store_id.split('-')

    # Load configuration
    config = Config.from_yaml("../configs/project_config.yml")
    catalog_name = config.catalog_name
    schema_name = config.schema_name
    predefined_unique_ids = config.predefined_unique_ids
    horizon = config.horizon
    train_data = "Dept_Store_28_Day_Test_Set_Forecast"
    test_data = "feature_set"
    temp_prediction_table = "intermediate_28_day_predictions"

    # Define MLFlow Experiment and Run
    git_sha = "5d53908cc7b4f89b30dfbd5c3355c72076b8d2fb"
    branch = "week5"
    component = "predicting_28_days_child_run"
    run_name=f"Predicting_28_Days_{dept_store_id}"
    run_tags={"git_sha": f"{git_sha}", "branch": branch, "dept_store_id": dept_store_id, "component": f"{component}"}
    model_name_uc = f"{catalog_name}.{schema_name}.ForecastModel_{dept_store_id}"
    model_name = "LGBMRegressor"
    alias = "champion"
    model_uri = f"models:/{model_name_uc}@{alias}"
    loaded_model = mlforecast.flavor.load_model(model_uri=model_uri)

    print(f"Processing department: {dept_id}, store: {store_id}")

    # Load, Convert, Filter X_df, Test, Reference
    historical_set = spark.table(f"{catalog_name}.{schema_name}.{train_data}") \
        .filter(F.col("dept_id") == dept_id) \
        .filter(F.col("store_id") == store_id) \
        .filter(F.col("unique_id").isin(predefined_unique_ids))

    predict_set = spark.table(f"{catalog_name}.{schema_name}.{test_data}") \
        .filter(F.col("dept_id") == dept_id) \
        .filter(F.col("store_id") == store_id) \
        .filter(F.col("unique_id").isin(predefined_unique_ids))
    
    historical_df = historical_set.toPandas()
    predict_df = predict_set.toPandas()

    X_df = predict_df.drop(columns=['y','state_id', 'store_id', 'cat_id', 'dept_id', 'item_id', 'update_timestamp_utc'])
    test_df = historical_df[["unique_id", "ds", "y", "y_hat"]]
    reference_df = historical_df.drop(columns=['state_id', 'store_id', 'cat_id', 'dept_id', 'item_id', 'update_predictions_timestamp_utc',])

    # Set up MLflow experiment and start a run
    with mlflow.start_run(
        experiment_id=exp_id,
        parent_run_id=parent_run_id,
        run_name=run_name,
        tags=run_tags,
        nested=True
    ) as predicting_28_child_run:
        predicting_28_child_run_id = predicting_28_child_run.info.run_id
        print(f"MLflow run started with predicting_28_child_run_id: {predicting_28_child_run_id}")

       # Predict 28 Days
        y_future_pred = loaded_model.predict(h=horizon, X_df=X_df)
        y_future_pred.rename(columns={model_name: 'y_hat'}, inplace=True)

        analysis_df = pd.merge(y_future_pred, X_df, on=['unique_id', 'ds'], how='inner')

        # Generate and log the forecast plot
        fig = plot_series(df=test_df, forecasts_df= y_future_pred, engine="plotly", max_insample_length=365)
        mlflow.log_figure(fig, artifact_file=f"plots/{dept_store_id}_28_Day_Forecast_On_{job_run_date}.html")

        # Intialize, Fit, Estimate Performance
        dle = nml.DLE(
            metrics=['mae', 'mape', 'mse', 'msle', 'rmse', 'rmsle'],
            y_true='y',
            y_pred='y_hat',
            feature_column_names=['temp', 'sell_price', 'num_events'],
            timestamp_column_name='ds',
            chunk_period='d'
        )

        dle.fit(reference_df)
        estimated_performance = dle.estimate(analysis_df)

        # Create and Log Alerts
        alerts_df = estimated_performance.filter(period='analysis').to_df()
        for metric_eval in ['mae', 'mape', 'mse', 'msle', 'rmse', 'rmsle']:
            num_alerts = alerts_df[(metric_eval, 'alert')].sum()
            mlflow.log_metric(f"{metric_eval}_alerts", num_alerts)
            
        # Log Estimated Performance Plot
        fig2 = estimated_performance.plot()
        mlflow.log_figure(fig2, artifact_file=f"plots/{dept_store_id}_Performance_Estimation_On_{job_run_date}.html")

        # Save Predictions
        future_preds_spark = spark.createDataFrame(y_future_pred)
        future_preds_spark.createOrReplaceTempView(temp_prediction_table)
        spark.sql(f"""
            INSERT INTO {catalog_name}.{schema_name}.{temp_prediction_table}
            SELECT * FROM {temp_prediction_table}
        """)

if __name__ == "__main__":
    main()

# 05.predict_estimate_28_.py

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import mlflow
import mlforecast.flavor

from utilsforecast.losses import rmse, mse, mae, mape, smape
from utilsforecast.evaluation import evaluate
from utilsforecast.plotting import plot_series

from m5_forecasting.config import Config

import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import pandas as pd
import sys

import nannyml as nml

def main():
    spark = SparkSession.builder.getOrCreate()
    
    # Load Parameters
    dept_store_id = sys.argv[1]
    job_run_date = dbutils.jobs.taskValues.get(taskKey="Determine7DayForecastPath", key="job_run_date")
    exp_id = dbutils.jobs.taskValues.get(taskKey="PreprocessWeeklyForecasts", key="exp_id")
    parent_run_id = dbutils.jobs.taskValues.get(taskKey="PreprocessWeeklyForecasts", key="parent_run_id")
    dept_id, store_id = dept_store_id.split('-')

    # Load configuration
    config = Config.from_yaml("../configs/project_config.yml")
    catalog_name = config.catalog_name
    schema_name = config.schema_name
    target = config.target
    predefined_unique_ids = config.predefined_unique_ids
    horizon = 7
    train_data = "Dept_Store_7_Day_Forecast"
    test_data = "weekly_update_set"
    temp_prediction_table = "intermediate_7_day_predictions"

     # Define MLFlow Experiment and Run
    git_sha = "5d53908cc7b4f89b30dfbd5c3355c72076b8d2fb"
    branch = "week5"
    component = "predicting_7_days_child_run"
    run_name = f"Predicting_7_Days_{dept_store_id}"
    run_tags = {"git_sha": f"{git_sha}", "branch": branch, "dept_store_id": dept_store_id, "component": component}
    model_name_uc = f"{catalog_name}.{schema_name}.ForecastModel_{dept_store_id}"
    model_name = 'LGBMRegressor'
    alias = "champion"
    model_uri = f"models:/{model_name_uc}@{alias}"
    loaded_model = mlforecast.flavor.load_model(model_uri=model_uri)
    metrics = [rmse, mse, mae, mape, smape]

    print(f"Processing department: {dept_id}, store: {store_id}")
    
    # Load Reference, Predict, Test Data
    max_ds_row = spark.table(f"{catalog_name}.{schema_name}.{train_data}") \
        .filter(F.col("dept_id") == dept_id) \
        .filter(F.col("store_id") == store_id) \
        .filter(F.col("unique_id").isin(predefined_unique_ids)) \
        .agg(F.max("ds").alias("max_ds")) \
        .collect()[0]
    max_ds = max_ds_row["max_ds"]
    print(max_ds)
        
    historical_set = spark.table(f"{catalog_name}.{schema_name}.{train_data}") \
        .filter(F.col("dept_id") == dept_id) \
        .filter(F.col("store_id") == store_id) \
        .filter(F.col("unique_id").isin(predefined_unique_ids))

    predict_set = spark.table(f"{catalog_name}.{schema_name}.{test_data}") \
        .filter(F.col("dept_id") == dept_id) \
        .filter(F.col("store_id") == store_id) \
        .filter(F.col("unique_id").isin(predefined_unique_ids))
    
    reference_set = historical_set \
        .join(
            predict_set \
                .select("unique_id", "ds", "y"),
            on=['unique_id', 'ds'],
            how='inner'
            )
    
    future_set = predict_set \
        .filter(F.col("ds") > F.lit(max_ds)) \
        .filter(F.col("ds") <= F.date_add(F.lit(max_ds), 7))

    evaluation_set = evaluation_set = historical_set \
        .select("unique_id", "ds", "y_hat") \
        .filter(F.col("ds").between(F.date_sub(F.lit(max_ds), 6), F.lit(max_ds))) \
        .join(
            predict_set \
                .select("unique_id", "ds", "y"),
            on=['unique_id', 'ds'],
            how='inner'
            )
    
    evaluation_df = evaluation_set.toPandas()
    reference_df = reference_set.toPandas()
    future_df = future_set.toPandas()

    evaluation_df.rename(columns={'y_hat': model_name}, inplace=True)
    X_df = future_df.drop(columns=['y', 'state_id', 'store_id', 'cat_id', 'dept_id', 'item_id', 'modified_timestamp_utc'])
    historical_plot_df = reference_df[["unique_id", "ds", "y"]]
    reference_df = reference_df.drop(columns=['state_id', 'store_id', 'cat_id', 'dept_id', 'item_id', 'update_predictions_timestamp_utc'])

    # Set up MLflow experiment and start a run
    with mlflow.start_run(
        experiment_id=exp_id,
        parent_run_id=parent_run_id,
        run_name=run_name,
        tags=run_tags,
        nested=True
    ) as predicting_7_child_run:
        predicting_7_child_run_id = predicting_7_child_run.info.run_id
        print(f"MLflow run started with predicting_7_child_run_id: {predicting_7_child_run_id}")

        evaluation = evaluate(
            df=evaluation_df,
            metrics=metrics,
            models=[model_name],
            id_col='unique_id',
            time_col='ds',
            target_col=target,
        )

        # Log parameters and metrics
        mlflow.log_param("dept_id", dept_id)
        mlflow.log_param("store_id", store_id)
        for _, row in evaluation.iterrows():
            metric = row['metric']
            value = row[model_name]
            mlflow.log_metric(metric, value)

        # Generate and log the forecast plot
        fig = plot_series(forecasts_df= evaluation_df, engine="plotly", max_insample_length=365)
        mlflow.log_figure(fig, artifact_file=f"plots/{dept_store_id}_Evaluate_Last_Week_Forecast_On {job_run_date}.html")


        # Update the model
        loaded_model.update(df=evaluation_df[["unique_id", "ds", "y"]])

       # Predict 7 Days
        y_future_pred = loaded_model.predict(h=horizon, X_df=X_df)
        y_future_pred.rename(columns={model_name: 'y_hat'}, inplace=True)

        analysis_df = pd.merge(y_future_pred, X_df, on=['unique_id', 'ds'], how='inner')

        # Generate and log the forecast plot
        fig = plot_series(df=historical_plot_df, forecasts_df= y_future_pred, engine="plotly", max_insample_length=365)
        mlflow.log_figure(fig, artifact_file=f"plots/{dept_store_id}_7_Day_Forecast.html")

        # Intialize, Fit, Estimate Performance
        dle = nml.DLE(
            metrics=['mae', 'mape', 'mse', 'msle', 'rmse', 'rmsle'],
            y_true='y',
            y_pred='y_hat',
            feature_column_names=['temp', 'sell_price', 'conditions', 'num_events'],
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
        print("Future predictions saved.")

if __name__ == "__main__":
    main()

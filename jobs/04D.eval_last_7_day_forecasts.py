# 04D.eval_last_7_day_forecasts.py

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import mlflow

from utilsforecast.losses import rmse, mse, mae, mape, smape
from utilsforecast.evaluation import evaluate
from utilsforecast.plotting import plot_series

from m5_forecasting.config import Config
import sys

def main():
    spark = SparkSession.builder.getOrCreate()

    # Load Parameters
    dept_store_id = sys.argv[1]
    job_run_date = dbutils.jobs.taskValues.get(taskKey="Determine7DayForecastPath", key="job_run_date")
    exp_id = dbutils.jobs.taskValues.get(taskKey="PreprocessFinalWeeklyForecasts", key="exp_id")
    parent_run_id = dbutils.jobs.taskValues.get(taskKey="PreprocessFinalWeeklyForecasts", key="parent_run_id")
    dept_id, store_id = dept_store_id.split('-')

    # Load configuration
    config = Config.from_yaml("../configs/project_config.yml")
    catalog_name = config.catalog_name
    schema_name = config.schema_name
    target = config.target
    predefined_unique_ids = config.predefined_unique_ids
    train_data = "Dept_Store_7_Day_Forecast"
    test_data = "weekly_update_set"

    # Define MLFlow Experiment and Run
    git_sha = "5d53908cc7b4f89b30dfbd5c3355c72076b8d2fb"
    branch = "week5"
    component = "evaluating_final_7_days_child_run"
    run_name = f"Evaluating_Final_7_Days_{dept_store_id}"
    run_tags = {"git_sha": f"{git_sha}", "branch": branch, "dept_store_id": dept_store_id, "component": component}
    model_name = "LGBMRegressor"
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

    evaluation_set = historical_set \
        .select("unique_id", "ds", "y_hat") \
        .filter(F.col("ds").between(F.date_sub(F.lit(max_ds), 6), F.lit(max_ds))) \
        .join(
            predict_set \
                .select("unique_id", "ds", "y"),
            on=['unique_id', 'ds'],
            how='inner'
            )

    evaluation_df = evaluation_set.toPandas()

    evaluation_df.rename(columns={'y_hat': model_name}, inplace=True)

    # Set up MLflow experiment and start a run
    with mlflow.start_run(
        experiment_id=exp_id,
        parent_run_id=parent_run_id,
        run_name=run_name,
        tags=run_tags,
        nested=True
    ) as evaluating_7_child_run:
        evaluating_7_child_run_id = evaluating_7_child_run.info.run_id
        print(f"MLflow run started with evaluating_7_child_run_id: {evaluating_7_child_run_id}")

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

if __name__ == "__main__":
    main()
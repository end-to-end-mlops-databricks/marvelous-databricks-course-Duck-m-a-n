# 05B.eval_28_day_forecast.py

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import mlflow

import nannyml as nml

from utilsforecast.losses import rmse, mse, mae, mape, smape
from utilsforecast.evaluation import evaluate
from utilsforecast.plotting import plot_series

from m5_forecasting.config import Config
import sys

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
    target = config.target
    predefined_unique_ids = config.predefined_unique_ids
    test_set_forecast_tbl = "Dept_Store_28_Day_Test_Set_Forecast"
    forecast_estimates_tbl = "Dept_Store_28_Day_Forecast"
    forecast_actuals_tbl = "weekly_update_set"

    # Define MLFlow Experiment and Run
    git_sha = "5d53908cc7b4f89b30dfbd5c3355c72076b8d2fb"
    branch = "week5"
    component = "evaluating_28_days_child_run"
    run_name = f"Evaluating_28_Days_{dept_store_id}"
    run_tags = {"git_sha": f"{git_sha}", "branch": branch, "dept_store_id": dept_store_id, "component": component}
    model_name = "LGBMRegressor"
    metrics = [rmse, mse, mae, mape, smape]

    print(f"Processing department: {dept_id}, store: {store_id}")

    # Load Reference, Predict, Test Data
    reference_set = spark.table(f"{catalog_name}.{schema_name}.{test_set_forecast_tbl}") \
        .filter(F.col("dept_id") == dept_id) \
        .filter(F.col("store_id") == store_id) \
        .filter(F.col("unique_id").isin(predefined_unique_ids))

    forecast_estimate_set = spark.table(f"{catalog_name}.{schema_name}.{forecast_estimates_tbl}") \
        .filter(F.col("dept_id") == dept_id) \
        .filter(F.col("store_id") == store_id) \
        .filter(F.col("unique_id").isin(predefined_unique_ids))
    
    forecast_actual_set = spark.table(f"{catalog_name}.{schema_name}.{forecast_actuals_tbl}") \
        .filter(F.col("dept_id") == dept_id) \
        .filter(F.col("store_id") == store_id) \
        .filter(F.col("unique_id").isin(predefined_unique_ids))
    
    analysis_set = forecast_estimate_set \
        .select("unique_id", "ds", "y_hat") \
        .join(forecast_actual_set, on=['unique_id', 'ds'], how='inner')
    
    reference_df = reference_set.toPandas()
    analysis_df = analysis_set.toPandas()

    evaluation_df = analysis_df[["unique_id", "ds", "y", "y_hat"]]
    evaluation_df.rename(columns={'y_hat': model_name}, inplace=True) 

    # Set up MLflow experiment and start a run
    with mlflow.start_run(
        experiment_id=exp_id,
        parent_run_id=parent_run_id,
        run_name=run_name,
        tags=run_tags,
        nested=True
    ) as evaluating_28_child_run:
        evaluating_28_child_run_id = evaluating_28_child_run.info.run_id
        print(f"MLflow run started with evaluating_28_child_run_id: {evaluating_28_child_run_id}")

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
        mlflow.log_figure(fig, artifact_file=f"plots/{dept_store_id}_Evaluate_28_Days_Forecast_On {job_run_date}.html")

        calc = nml.PerformanceCalculator(
            y_pred='y_hat',
            y_true='y',
            problem_type='regression',
            metrics=['mae', 'mape', 'mse', 'msle', 'rmse', 'rmsle'],
            timestamp_column_name='ds',
            chunk_period='d'
            )

        calc.fit(reference_df)
        results = calc.calculate(analysis_df)

        # Log Alert
        alerts_df = results.filter(period='analysis').to_df()
        for metric_eval in ['mae', 'mape', 'mse', 'msle', 'rmse', 'rmsle']:
            num_alerts = alerts_df[(metric_eval, 'alert')].sum()
            mlflow.log_metric(f"{metric_eval}_alerts", num_alerts)

        # Log Estimated Performance Plot
        figure = results.plot(kind='performance')
        mlflow.log_figure(figure, artifact_file=f"plots/{dept_store_id}_Performance_Realized_On_{job_run_date}.html")


if __name__ == "__main__":
    main()
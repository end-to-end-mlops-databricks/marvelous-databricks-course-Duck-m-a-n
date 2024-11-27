# 02A.preprocess_28_day_test_forecast.py

import mlflow
import sys
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from m5_forecasting.config import Config

def main():
    spark = SparkSession.builder.getOrCreate()

    # Load Parameters
    job_run_date = sys.argv[1]
    dbutils.jobs.taskValues.set(key="job_run_date", value=job_run_date)

    # Load configuration
    config = Config.from_yaml("../configs/project_config.yml")
    catalog_name = config.catalog_name
    schema_name = config.schema_name
    predefined_unique_ids = config.predefined_unique_ids
    train_data = 'train_set'
    temp_prediction_table = "intermediate_28_day_test_set_predictions"

    base_dir = "/Users/sjduckersjr@gmail.com/Dept_Store_Forecasting_LifeCycle"
    create_dir = f"/Workspace{base_dir}"

    # Define Git Configuration, Experiment/Run tags
    experiment_name = f"{base_dir}/28_Day_Forecast_LifeCycle_{job_run_date}"
    git_sha = "5d53908cc7b4f89b30dfbd5c3355c72076b8d2fb"
    branch = "week5"
    component =  "training_parent_run"
    run_name = f"Training_Dept_Store_Models_On_{job_run_date}"
    run_tags = {"git_sha": f"{git_sha}", "branch": f"{branch}", "component": f"{component}"}
    experiment_tag_name = "project"
    experiment_tag_value = "forecasting"


    # Create Base Directory
    os.makedirs(create_dir, exist_ok=True)

    # Create MLflow experiment
    mlflow.set_experiment(experiment_name)
    mlflow.set_experiment_tag(experiment_tag_name, experiment_tag_value)

    exp_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id
    dbutils.jobs.taskValues.set(key="exp_id", value=exp_id)

    # Create dept_store_id task valies
    dept_store_df = spark.table(f"{catalog_name}.{schema_name}.{train_data}") \
        .filter(F.col("unique_id").isin(predefined_unique_ids)) \
        .select('dept_id', 'store_id') \
        .distinct()

    dept_store_df = dept_store_df.withColumn(
        'dept_store_id', F.concat_ws('-', F.col('dept_id'), F.col('store_id'))
    )

    dept_store_group_list = [row['dept_store_id'] for row in dept_store_df.collect()]

    dbutils.jobs.taskValues.set(key="dept_store_group_list", value=dept_store_group_list)
    print(f"Found {len(dept_store_group_list)} unique department-store groups: {dept_store_group_list}")

    print(f"MLflow experiment '{experiment_name}' is set with experiment ID {exp_id}.")
    with mlflow.start_run(
        experiment_id=exp_id,
        run_name=run_name,
        tags=run_tags
    ) as training_parent_run:
        training_parent_run_id = training_parent_run.info.run_id

        dbutils.jobs.taskValues.set(key="parent_run_id", value=training_parent_run_id)

    # Create the intermediate_predictions table if it doesn't exist
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {catalog_name}.{schema_name}.{temp_prediction_table} (
            unique_id STRING,
            ds TIMESTAMP,
            y_hat DOUBLE
        ) USING DELTA
    """)
    print("Intermediate predictions table is ready.")

if __name__ == "__main__":
    main()

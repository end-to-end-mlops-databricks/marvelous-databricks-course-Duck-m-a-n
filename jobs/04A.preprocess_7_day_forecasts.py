# 04A.preprocess_7_day_forecasts.py

import mlflow

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from m5_forecasting.config import Config

def main():
    spark = SparkSession.builder.getOrCreate()

    # Load Parameters
    job_run_date = dbutils.jobs.taskValues.get(taskKey="Determine7DayForecastPath", key="job_run_date")
    print(f"Job Run Date: {job_run_date}")

    # Load Configuration
    config = Config.from_yaml("../configs/project_config.yml")
    catalog_name = config.catalog_name
    schema_name = config.schema_name
    predefined_unique_ids = config.predefined_unique_ids
    test_data = "weekly_update_set"
    temp_prediction_table = "intermediate_7_day_predictions"

    # Define Git Configuration, Experiment/Run tags
    git_sha = "5d53908cc7b4f89b30dfbd5c3355c72076b8d2fb"
    branch = "week5"
    component = "predicting_7_day_parent_run"
    filter_string = "tags.project = 'forecasting'"

    # Search, Load, Set Experiment ID
    mlflow_experiments_list = mlflow.search_experiments(filter_string=filter_string,order_by=["name DESC"])
    mlflow_experiment = mlflow_experiments_list[0]
    experiment_name = mlflow_experiment.name.split('/')[-1]
    exp_id = mlflow_experiment.experiment_id
    run_name = f"Predicting_7_Days_Dept_Store_On_{job_run_date}"
    run_tags ={ "git_sha": f"{git_sha}", "branch": f"{branch}", "component": f"{component}"}
    
    dbutils.jobs.taskValues.set(key="exp_id", value=exp_id)

    # Retrieve unique combinations of dept_id and store_id
    dept_store_df = spark.table(f"{catalog_name}.{schema_name}.{test_data}") \
        .filter(F.col("modified_timestamp_utc") <= job_run_date) \
        .filter(F.col("unique_id").isin(predefined_unique_ids)) \
        .select('dept_id', 'store_id') \
        .distinct()

    # Create dept_store_id task valies
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
    ) as predicting_7_day_parent_run:
        predicting_7_day_parent_run_id = predicting_7_day_parent_run.info.run_id

        dbutils.jobs.taskValues.set(key="parent_run_id", value=predicting_7_day_parent_run_id)
        
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

# get_dept_ids.py

import json
import mlflow
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from m5_forecasting.config import Config

def main():
    spark = SparkSession.builder.getOrCreate()

    # Load configuration
    config = Config.from_yaml("../../configs/project_config.yml")
    catalog_name = config.catalog_name
    schema_name = config.schema_name
    job_run_date = sys.argv[1]

    # Added code for filtering for testing this approach
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

    # Retrieve unique combinations of dept_id and store_id
    dept_store_df = spark.table(f"{catalog_name}.{schema_name}.train_set") \
        .filter(F.col("unique_id").isin(predefined_unique_ids)) \
        .select('dept_id', 'store_id') \
        .distinct()

    # Concatenate dept_id and store_id to create dept_store_id
    dept_store_df = dept_store_df.withColumn(
        'dept_store_id', F.concat_ws('-', F.col('dept_id'), F.col('store_id'))
    )

    # Collect dept_store_id into a list
    dept_store_group_list = [row['dept_store_id'] for row in dept_store_df.collect()]

    print(f"Found {len(dept_store_group_list)} unique department-store groups: {dept_store_group_list}")

    # Set the dept_store_group_list as a task value
    dbutils.jobs.taskValues.set(key="dept_store_group_list", value=dept_store_group_list)

    # Set up MLflow experiment (without creating it)
    experiment_name = f"/Users/sjduckersjr@gmail.com/Experiments/Dept_Store_Forecasts_For_{job_run_date}"
    mlflow.set_experiment(experiment_name)

    exp_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id
    dbutils.jobs.taskValues.set(key="exp_id", value=exp_id)

    print(f"MLflow experiment '{experiment_name}' is set with experiment ID {exp_id}.")

    # Create the intermediate_predictions table if it doesn't exist
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {catalog_name}.{schema_name}.intermediate_predictions (
            unique_id STRING,
            ds DATE,
            y_hat DOUBLE,
            dept_id STRING,
            store_id STRING
        ) USING DELTA
    """)
    print("Intermediate predictions table is ready.")

if __name__ == "__main__":
    main()

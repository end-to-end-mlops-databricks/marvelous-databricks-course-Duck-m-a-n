# 04.determine_7_day_forecast_path.py

import sys
from m5_forecasting.config import Config

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def main():
    spark = SparkSession.builder.getOrCreate()

    # Load Parameters
    job_run_date = sys.argv[1]
    print(f"Job Run Date: {job_run_date}")

    # Load Configuration
    config = Config.from_yaml("../configs/project_config.yml")
    catalog_name = config.catalog_name
    schema_name = config.schema_name
    predefined_unique_ids = config.predefined_unique_ids

    # Determine Max ds values
    max_update_date_row = spark.table(f"{catalog_name}.{schema_name}.weekly_update_set") \
        .filter(F.col("unique_id").isin(predefined_unique_ids)) \
        .filter(F.col("modified_timestamp_utc") <= job_run_date) \
        .agg(F.max("ds").alias("max_update_date")) \
        .collect()[0]

    max_test_date_row = spark.table(f"{catalog_name}.{schema_name}.test_set") \
        .filter(F.col("unique_id").isin(predefined_unique_ids)) \
        .agg(F.max("ds").alias("max_test_date")) \
        .collect()[0]
    
    max_forecast_date_row = spark.table(f"{catalog_name}.{schema_name}.feature_set") \
        .filter(F.col("unique_id").isin(predefined_unique_ids)) \
        .agg(F.max("ds").alias("max_forecast_date")) \
        .collect()[0]

    max_test_date = max_test_date_row["max_test_date"]
    max_update_date = max_update_date_row["max_update_date"]
    max_forecast_date = max_forecast_date_row["max_forecast_date"]
    print(f"Max test date: {max_test_date}")
    print(f"Max update date: {max_update_date}")
    print(f"Max forecast date: {max_forecast_date}")

    # Compare dates and set new_forecast_cycle value accordingly
    if max_test_date == max_update_date:
        new_forecast_cycle = "New"
    else:
        new_forecast_cycle = "Continue"

    if max_forecast_date == max_update_date:
        end_forecast_cycle = "End"
    else:
        end_forecast_cycle = "Continue"

    # Set task value for new_forecast_cycle
    dbutils.jobs.taskValues.set(key="job_run_date", value=job_run_date)
    dbutils.jobs.taskValues.set(key="new_forecast_cycle", value=new_forecast_cycle)
    dbutils.jobs.taskValues.set(key="end_forecast_cycle", value=end_forecast_cycle)

if __name__ == "__main__":
    main()

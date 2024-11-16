# get_dept_ids.py

import json
import mlflow

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from m5_forecasting.config import Config

def main():
    spark = SparkSession.builder.getOrCreate()

    # Load configuration
    config = Config.from_yaml("../../configs/project_config.yml")
    catalog_name = config.catalog_name
    schema_name = config.schema_name

    # Added code for filtering for testing this approach
    predefined_unique_ids = [
        "HOBBIES_1_001_CA_1", # full historical data
        "HOBBIES_1_023_CA_1",
        "FOODS_3_595_CA_1",   # represents shortest time series 100 timestamps
        "FOODS_3_238_CA_1",
        "FOODS_3_246_CA_1", # full varying history not very much highs looks to represent the majority
        "HOUSEHOLD_1_146_CA_1", #full history
        "HOUSEHOLD_1_178_CA_1", # full history same store same state
        "HOUSEHOLD_1_056_CA_1", # varying history same store same state
        "HOUSEHOLD_1_179_CA_2", # full hisotry high values, different store same state
    ]

    # Retrieve all unique dept_ids
    dept_ids_df = spark.table(f"{catalog_name}.{schema_name}.train_set") \
        .filter(F.col("unique_id").isin(predefined_unique_ids)) \
        .select('dept_id') \
        .distinct()
    
    dept_ids_list = [row['dept_id'] for row in dept_ids_df.collect()]
    
    print(f"Found {len(dept_ids_list)} unique departments: {dept_ids_list}")

    # Set the dept_ids as a task value
    dbutils.jobs.taskValues.set(key="dept_ids_list", value=dept_ids_list)

    # Optionally, log the dept_ids in MLflow
    """
    with mlflow.start_run(run_name="Get Dept IDs") as run:
        mlflow.log_param("dept_ids", dept_ids)
    """
if __name__ == "__main__":
    main()

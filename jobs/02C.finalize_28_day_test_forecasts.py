# 02C.finalize_28_day_test_forecasts.py

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException

from datetime import datetime

from m5_forecasting.config import Config

def main():
    spark = SparkSession.builder.getOrCreate()

    # Load Parameters
    job_timestamp_utc = dbutils.jobs.taskValues.get(taskKey="Preprocess28DayTestForecasts", key="job_run_date")
    job_timestamp_utc_timestamp = datetime.strptime(job_timestamp_utc, "%Y-%m-%d")
    print(f"Job timestamp UTC: {job_timestamp_utc_timestamp}")

    # Load configuration
    config = Config.from_yaml("../configs/project_config.yml")
    catalog_name = config.catalog_name
    schema_name = config.schema_name
    prediction_dataset = "intermediate_28_day_test_set_predictions"
    test_set_table_name = "test_set"
    add_prediction_timestamp = "update_predictions_timestamp_utc"
    table_name = f"{catalog_name}.{schema_name}.Dept_Store_28_Day_Test_Set_Forecast"

    # Function to check if the table exists
    def table_exists(table_name):
        try:
            spark.table(table_name)
            return True
        except AnalysisException:
            return False

    # Read the intermediate predictions table
    predictions_df = spark.table(f"{catalog_name}.{schema_name}.{prediction_dataset}")

    # Read the test_set table
    test_set_df = spark.table(f"{catalog_name}.{schema_name}.{test_set_table_name}")

    # Join predictions with test_set on 'unique_id' and 'ds' (date)
    final_predictions_df = predictions_df \
        .join(
            test_set_df,
            on=['unique_id', 'ds'],
            how='inner'
            ) \
        .drop("update_timestamp_utc")

    final_predictions_df = final_predictions_df.withColumn(add_prediction_timestamp, F.lit(job_timestamp_utc_timestamp))

    # Enable schema evolution if needed
    spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "true")
    
    # Check if the target table exists
    if not table_exists(table_name):
        print(f"Table {table_name} does not exist. Creating it now.")
        # Create the table
        final_predictions_df.write.format("delta").mode("overwrite").option("delta.enableChangeDataFeed", "true").saveAsTable(table_name)
    else:
        print(f"Table {table_name} exists. Proceeding with MERGE operation.")
        # Create or replace a temporary view for the source DataFrame
        final_predictions_df.createOrReplaceTempView("source_predictions")
        
        # Define the MERGE SQL statement
        merge_sql = f"""
        MERGE INTO {table_name} AS t
        USING source_predictions AS s
        ON t.unique_id = s.unique_id AND t.ds = s.ds
        WHEN MATCHED THEN
        UPDATE SET *
        WHEN NOT MATCHED THEN
        INSERT *
        """
        
        # Execute the MERGE statement
        spark.sql(merge_sql)
        print("MERGE operation completed successfully.")
    
        # Drop the intermediate predictions table as it is no longer needed
    spark.sql(f"DROP TABLE IF EXISTS {catalog_name}.{schema_name}.intermediate_28_day_test_set_predictions")
    print("Dropped intermediate_predictions table as it is no longer needed.")

if __name__ == "__main__":
    main()




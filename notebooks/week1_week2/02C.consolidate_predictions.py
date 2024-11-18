# consolidate_predictions.py

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

    job_timestamp_utc = sys.argv[1]
    print(f"Job timestamp UTC: {job_timestamp_utc}")

    # Read the intermediate predictions table
    predictions_df = spark.table(f"{catalog_name}.{schema_name}.intermediate_predictions")
    # Optionally drop 'dept_id' and 'store_id' if not needed
    predictions_df = predictions_df.drop("dept_id", "store_id")

    # Read the test_set table
    test_set_df = spark.table(f"{catalog_name}.{schema_name}.test_set")

    # Join predictions with test_set on 'unique_id' and 'ds' (date)
    final_predictions_df = predictions_df.join(
        test_set_df,
        on=['unique_id', 'ds'],
        how='left'
    )

    final_predictions_df = final_predictions_df.withColumn("update_predictions_timestamp_utc", F.lit(job_timestamp_utc))

    # Save the final predictions table
    final_predictions_df.write.mode("overwrite") \
        .format("delta") \
        .option("overwriteSchema", "true") \
        .saveAsTable(f"{catalog_name}.{schema_name}.Dept_Store_28_Day_Forecast")

    print("Consolidated predictions saved to Dept_Store_28_Day_Forecast table.")

    # Drop the intermediate predictions table as it is no longer needed
    spark.sql(f"DROP TABLE IF EXISTS {catalog_name}.{schema_name}.intermediate_predictions")
    print("Dropped intermediate_predictions table as it is no longer needed.")

if __name__ == "__main__":
    main()

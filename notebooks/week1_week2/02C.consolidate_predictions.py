# consolidate_predictions.py

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from m5_forecasting.config import Config

def main():
    spark = SparkSession.builder.getOrCreate()

    # Load configuration
    config = Config.from_yaml("../../configs/project_config.yml")
    catalog_name = config.catalog_name
    schema_name = config.schema_name

    timestamp = F.to_utc_timestamp(F.current_timestamp(), "UTC")

    # Read the intermediate predictions table
    predictions_df = spark.table(f"{catalog_name}.{schema_name}.intermediate_predictions")
    predictions_df = predictions_df.drop("dept_id")

    # Read the test_set table
    test_set_df = spark.table(f"{catalog_name}.{schema_name}.test_set")

    # Join predictions with test_set on 'unique_id' and 'ds' (date)
    final_predictions_df = predictions_df.join(
        test_set_df,
        on=['unique_id', 'ds'],
        how='left'
    )

    final_predictions_df = final_predictions_df.withColumn("update_predictions_timestamp_utc", timestamp)

    # Save the final predictions table
    final_predictions_df.write.mode("overwrite") \
        .format("delta") \
        .option("overwriteSchema", "true") \
        .saveAsTable(f"{catalog_name}.{schema_name}.Dept_28_Day_Forecast")

    print("Consolidated predictions saved to Dept_28_Day_Forecast table.")

if __name__ == "__main__":
    main()

# Databricks notebook source
# MAGIC %pip install m5_forecasting-0.0.1-py3-none-any.whl

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND -----------
from m5_forecasting.preprocessing.data_processor import DataProcessor
from m5_forecasting.config import Config

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
config = Config.from_yaml("../../configs/project_config.yml")

catalog_name = config.catalog_name
schema_name = config.schema_name
volumes_sales_csv = config.paths.raw_sales_path
volumes_calendar_csv = config.paths.raw_calendar_path
volumes_sell_price_csv = config.paths.raw_sell_prices_path

# COMMAND ----------
sales_df = spark.read.option("header", "true").option("inferSchema", "true").csv(volumes_sales_csv)
calendar_df = spark.read.option("header", "true").option("inferSchema", "true").csv(volumes_calendar_csv)
sell_price_df = spark.read.option("header", "true").option("inferSchema", "true").csv(volumes_sell_price_csv)

# COMMAND ----------
data_processor = DataProcessor(config, sales_df, calendar_df, sell_price_df)

# COMMAND ----------
processed_sales, processed_calendar, processed_sell_price, processed_prod_info = data_processor.preprocess_data()

# COMMAND ---------
assert not processed_sales.pandera.errors, f"sales_df validation errors: {processed_sales.pandera.errors}"
assert not processed_calendar.pandera.errors, f"calendar_df validation errors: {processed_calendar.pandera.errors}"
assert not processed_sell_price.pandera.errors, f"sell_price_df validation errors: {processed_sell_price.pandera.errors}"
assert not processed_prod_info.pandera.errors, f"prod_info_df validation errors: {processed_prod_info.pandera.errors}"

# COMMAND ----------
train_set, test_set = data_processor.split_data()

# COMMAND ----------
# Save to catalog and confirm completion

data_processor.save_to_catalog(
    spark=spark,
    catalog_name=config.catalog_name,
    schema_name=config.schema_name,
    train=train_set,
    test=test_set,
    calendar=processed_calendar,
    sell_price=processed_sell_price,
    prod_info=processed_prod_info
)


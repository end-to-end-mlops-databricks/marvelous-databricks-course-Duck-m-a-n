# Databricks notebook source
# MAGIC %pip install m5_forecasting-0.0.1-py3-none-any.whl

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND -----------
from m5_forecasting.preprocessing.data_processor import DataProcessor
from m5_forecasting.config import Config

import pandas as pd
from pyspark.sql import SparkSession

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
config = Config.from_yaml("../../configs/project_config.yml")

catalog_name = config.catalog_name
schema_name = config.schema_name
volumes_sales_csv = config.paths.raw_sales_path
volumes_calendar_csv = config.paths.raw_calendar_path
volumes_sell_price_csv = config.paths.raw_sell_prices_path
volumes_weather_csv = config.paths.raw_weather_path

# COMMAND ----------
sales  = pd.read_csv(volumes_sales_csv)
calendar = pd.read_csv(volumes_calendar_csv)
sell_price = pd.read_csv(volumes_sell_price_csv)
weather = pd.read_csv(volumes_weather_csv)

# COMMAND ----------
processor = DataProcessor(config = config, sales_data=sales, calendar=calendar, sell_price=sell_price, weather=weather)

# COMMAND ----------
combined_df = processor.preprocess_data()

# COMMAND ----------
train_set, test_set = processor.split_data()

# COMMAND ----------
processor.save_to_catalog(spark, train_set, test_set, catalog_name, schema_name)

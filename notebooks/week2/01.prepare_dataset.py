# Databricks notebook source
import logging
import time
from m5_forecasting.preprocessing.data_processor import DataProcessor
from m5_forecasting.config import Config
from databricks.connect import DatabricksSession

# Configure logging to display in real-time
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initial log statement
logger.info("Starting the data preparation script...")

# COMMAND ----------
# Create Databricks session and log its creation
try:
    spark = DatabricksSession.builder.getOrCreate()
    logger.info("Successfully created Databricks session.")
except Exception as e:
    logger.error(f"Failed to create Databricks session: {e}")
    raise

# COMMAND ----------
# Load configuration and log success
try:
    config = Config.from_yaml("/Users/duckman/Projects/marvelous-databricks-course-Duck-m-a-n/configs/project_config.yml")
    logger.info("Configuration loaded successfully.")
except Exception as e:
    logger.error(f"Error loading configuration: {e}")
    raise

# COMMAND ----------
catalog_name = config.catalog_name
schema_name = config.schema_name
raw_sales_table = config.dataset.raw_sales_data
raw_calendar_table = config.dataset.raw_calendar_data
raw_sell_price_table = config.dataset.raw_sell_price_data

logger.info(f"Using catalog: {catalog_name}, schema: {schema_name}, tables: {raw_sales_table}, {raw_calendar_table}, {raw_sell_price_table}")

# COMMAND ----------
# Read raw tables and confirm data load
try:
    spark_sales = spark.read.table(f"{catalog_name}.{schema_name}.{raw_sales_table}")
    spark_calendar = spark.read.table(f"{catalog_name}.{schema_name}.{raw_calendar_table}")
    spark_sell_price = spark.read.table(f"{catalog_name}.{schema_name}.{raw_sell_price_table}")
    logger.info("Raw tables loaded successfully.")
except Exception as e:
    logger.error(f"Error loading raw tables: {e}")
    raise

# COMMAND ----------
# Convert spark_sales to pandas and log the time taken
start_time = time.time()
pandas_sales = spark_sales.toPandas()
logger.info(f"Converted spark_sales to pandas in {time.time() - start_time:.2f} seconds")

# Convert spark_calendar to pandas and log the time taken
start_time = time.time()
pandas_calendar = spark_calendar.toPandas()
logger.info(f"Converted spark_calendar to pandas in {time.time() - start_time:.2f} seconds")

# Convert spark_sell_price to pandas and log the time taken
start_time = time.time()
pandas_sell_price = spark_sell_price.toPandas()
logger.info(f"Converted spark_sell_price to pandas in {time.time() - start_time:.2f} seconds")

logger.info("All Spark DataFrames converted to Pandas DataFrames.")


# COMMAND ----------
data_processor = DataProcessor(config, pandas_sales, pandas_calendar, pandas_sell_price, logger)
logger.info("DataProcessor initialized.")

# COMMAND ----------
processed_sales, processed_calendar, processed_sell_price, processed_prod_info = data_processor.preprocess_data()
logger.info("Data preprocessed successfully.")

# COMMAND ----------
train_set, test_set = data_processor.split_data()
logger.info("Data split into training and testing sets.")

# COMMAND ----------
# Save to catalog and confirm completion
try:
    data_processor.save_to_catalog(
        spark=spark,
        catalog_name=catalog_name,
        schema_name=schema_name,
        train=train_set,
        test=test_set,
        calendar=processed_calendar,
        sell_prices=processed_sell_price,
        prod_info=processed_prod_info
    )
    logger.info("Data saved to catalog successfully.")
except Exception as e:
    logger.error(f"Error saving data to catalog: {e}")
    raise

logger.info("Data preparation script completed.")

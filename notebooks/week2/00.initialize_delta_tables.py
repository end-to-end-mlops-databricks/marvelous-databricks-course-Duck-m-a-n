# initialize_delta_tables.py

from m5_forecasting.config import Config
from databricks.connect import DatabricksSession
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Create Databricks session
spark = DatabricksSession.builder.getOrCreate()

# Load configuration using Config class
config = Config.from_yaml("/Users/duckman/Projects/marvelous-databricks-course-Duck-m-a-n/configs/project_config.yml")

# Extract catalog, schema, and paths from the config
catalog_name = config.catalog_name
schema_name = config.schema_name

# Define the table names and their respective CSV paths using the Config class instance
datasets = {
    "raw_sales_eval": config.dataset.raw_sales_path,
    "raw_calendar": config.dataset.raw_calendar_path,
    "raw_sell_prices": config.dataset.raw_sell_prices_path
}

# Function to load CSV, save as Delta table
def load_and_convert_to_delta(table_name, csv_path):
    try:
        logger.info(f"Starting processing for {table_name} from {csv_path}")

        # Read CSV file into Spark DataFrame
        df = spark.read.csv(csv_path, header=True, inferSchema=True)

        # Define the full table name
        full_table_name = f"{catalog_name}.{schema_name}.{table_name}"

        # Write to Delta table
        df.write.format("delta").mode("overwrite").saveAsTable(full_table_name)
        logger.info(f"Successfully created Delta table: {full_table_name}")

    except Exception as e:
        logger.error(f"Error processing {table_name}: {e}")

# Process each dataset
for table_name, csv_path in datasets.items():
    load_and_convert_to_delta(table_name, csv_path)

# Stop the Spark session
spark.stop()
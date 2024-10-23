# Test for single series unique_id = "FOODS_1_001_CA_1"

from m5_forecasting.data_processor import DataProcessor
from m5_forecasting.utils import DataChecker
import yaml
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger specific to this module
logger = logging.getLogger(__name__)

# Load config from project_config.yml
with open("project_config.yml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Initialize DataProcessor
logger.info("Initializing DataProcessor...")
data_processor = DataProcessor(config=config)

# Process data
processed_data = data_processor.process_data()
logger.info("Data processing completed.")

# Split into train and test sets
logger.info("Splitting data into train and test sets...")
train_df, test_df = data_processor.split_data()

#####################################################################
# Initialize DataChecker with train_df, test_df, original_row_count, and filtered_row_count
logger.info("Initializing DataChecker...")
data_checker = DataChecker(train_df, test_df, data_processor.original_row_count, data_processor.filtered_row_count)

# Run validation checks
row_drop_summary = data_checker.not_sales_row_drop_summary_check()
logger.info(row_drop_summary)

missing_dates_train_message = data_checker.gaps_in_date_check(train_df)
logger.info(f"Train dataset: {missing_dates_train_message}")

missing_dates_test_message = data_checker.gaps_in_date_check(train_df)
logger.info(f"Test dataset: {missing_dates_test_message}")

duplicate_train_message = data_checker.unique_id_per_date_check(train_df)
logger.info(f"Train dataset: {duplicate_train_message}")

duplicate_test_message = data_checker.unique_id_per_date_check(test_df)
logger.info(f"Test dataset: {duplicate_test_message}")

data_quality_message = data_checker.nulls_and_shape_check()
logger.info(data_quality_message)

logger.info("Data checking completed.")
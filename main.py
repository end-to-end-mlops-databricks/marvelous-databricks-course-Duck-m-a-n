# Test for single series unique_id = "FOODS_1_001_CA_1"

from m5_forecasting.data_processor import DataProcessor
from m5_forecasting.utils import DataChecker
import yaml
import logging
from pathlib import Path

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger specific to this module
logger = logging.getLogger(__name__)

# Load config from project_config.yml
try:
    config_path = Path(__file__).parent / "project_config.yml"
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
except FileNotFoundError:
    logger.error(f"Configuration file not found at {config_path}")
    raise
except yaml.YAMLError as e:
    logger.error(f"Error parsing configuration file: {e}")
    raise

# Initialize DataProcessor
logger.info("Initializing DataProcessor...")
data_processor = DataProcessor(config=config)

# Process data
try:
    processed_data = data_processor.process_data()
    if processed_data.empty:
        raise ValueError("Processing resulted in empty DataFrame")
except Exception as e:
    logger.error(f"Error during data processing: {e}")
    raise
logger.info(f"Data processing completed.")

# Split into train and test sets
logger.info("Splitting data into train and test sets...")
try:
    train_df, test_df = data_processor.split_data()
    if train_df.empty or test_df.empty:
        raise ValueError("Data splitting resulted in empty DataFrame(s)")
except Exception as e:
    logger.error(f"error during data splitting: {e}")
    raise

#####################################################################
# Initialize DataChecker with train_df, test_df, original_row_count, and filtered_row_count
logger.info("Initializing DataChecker...")
data_checker = DataChecker(train_df, test_df, data_processor.original_row_count, data_processor.filtered_row_count)

# Run validation checks  
validation_results = {  
    "row_drop_summary": data_checker.not_sales_row_drop_summary_check(),  
    "train_missing_dates": data_checker.gaps_in_date_check(train_df),  
    "test_missing_dates": data_checker.gaps_in_date_check(test_df),  
    "train_duplicates": data_checker.unique_id_per_date_check(train_df),  
    "test_duplicates": data_checker.unique_id_per_date_check(test_df),  
    "data_quality": data_checker.nulls_and_shape_check()  
}  

# Log individual results  
for check_name, result in validation_results.items():  
    logger.info(f"{check_name}: {result}")  

# Determine overall validation status  
validation_passed = all(  
    "error" not in str(result).lower() 
    and "failed" not in str(result).lower()  
    for result in validation_results.values()  
)  

logger.info(f"Data validation {'passed' if validation_passed else 'failed'}")  

logger.info("Data checking completed.")
# 01A.prepare_datasets.py

from m5_forecasting.preprocessing.data_processor import DataProcessor
from m5_forecasting.config import Config

import pandas as pd
from pyspark.sql import SparkSession

def main():
    spark = SparkSession.builder.getOrCreate()
    config = Config.from_yaml("../configs/project_config.yml")

    catalog_name = config.catalog_name
    schema_name = config.schema_name
    volumes_sales_csv = config.paths.raw_sales_path
    volumes_sales_future_csv = config.paths.raw_sales_future_path
    volumes_calendar_csv = config.paths.raw_calendar_path
    volumes_sell_price_csv = config.paths.raw_sell_prices_path
    volumes_weather_csv = config.paths.raw_weather_path

    sales = pd.read_csv(volumes_sales_csv)
    sales_future = pd.read_csv(volumes_sales_future_csv)
    calendar = pd.read_csv(volumes_calendar_csv)
    sell_price = pd.read_csv(volumes_sell_price_csv)
    weather = pd.read_csv(volumes_weather_csv)

    processor = DataProcessor(config=config, sales_data=sales, calendar=calendar, sell_price=sell_price, weather=weather)
    processor_future = DataProcessor(config=config, sales_data=sales_future, calendar=calendar, sell_price=sell_price, weather=weather)

    processor.preprocess_data()
    combined_future_df = processor_future.preprocess_data()
    print("Data processed")

    train_set, test_set = processor.split_data()
    #future_df, update_7_day_future, predict_7_day_future = processor_future.prepare_future_data()
    print("Data split")

    datasets = [
        (train_set, 'train_set'),
        (test_set, 'test_set'),
        (combined_future_df,'feature_set')
    ]

    processor.save_to_catalog(spark, datasets, catalog_name, schema_name)

if __name__ == "__main__":
    main()

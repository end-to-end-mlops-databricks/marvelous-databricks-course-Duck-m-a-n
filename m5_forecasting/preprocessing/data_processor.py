import logging
import time
import pandas as pd
from pyspark.sql import functions as F
from m5_forecasting.schemas.processed_data import SalesDataSchema, CalendarSchema, SellPriceSchema, ProductInfoSchema

class DataProcessor:
    def __init__(self, config, sales_data, calendar, sell_prices, logger = None):
        """Initialize the DataProcessor with configuration details and data."""
        self.config = config
        self.sales_data = sales_data
        self.calendar = calendar
        self.sell_prices = sell_prices
        self.horizon = config.horizon
        self.logger = logger if logger else logging.getLogger(__name__)
        self.prod_info = None
        self.train_df = None
        self.test_df = None

    def preprocess_data(self):
        """Processes the data to create self.sales_data, self.calendar, self.sell_prices, and self.prod_info."""
        self.logger.info("Starting data preprocessing...")

        # Step 1: Generate unique_id and melt sales data
        self.sales_data["unique_id"] = self.sales_data["item_id"] + "_" + self.sales_data["store_id"]
        date_columns = [col for col in self.sales_data.columns if col.startswith("d_")]
        self.sales_data = pd.melt(
            self.sales_data,
            id_vars=["unique_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
            value_vars=date_columns,
            var_name="ds_id",
            value_name="y"
        )

        # Convert data types in self.sales_data
        self.sales_data["ds_id"] = self.sales_data["ds_id"].astype(str)
        self.sales_data["unique_id"] = self.sales_data["unique_id"].astype(str)
        self.sales_data["y"] = self.sales_data["y"].astype(int)

        # Step 2: Prepare calendar with time features and num_events
        self.calendar = self.prepare_calendar()

        # Step 3: Merge self.sales_data with calendar to map ds and add num_events
        self.sales_data = self.sales_data.merge(
            self.calendar[["ds_id", "ds", "wm_yr_wk"]],
            on="ds_id",
            how="left"
        )

        # Step 4: Apply filter to remove rows before the product release
        self.sales_data = self.filter_before_release()

        # Step 5: Prepare self.sell_prices with unique_id and expanded ds
        self.sell_prices["unique_id"] = self.sell_prices["item_id"] + "_" + self.sell_prices["store_id"]
        self.sell_prices = self.sell_prices.merge(
            self.calendar[["wm_yr_wk", "ds"]],
            on="wm_yr_wk",
            how="left"
        )[["unique_id", "ds", "sell_price"]]

        # Convert data types in self.sell_prices
        self.sell_prices["unique_id"] = self.sell_prices["unique_id"].astype(str)
        self.sell_prices["sell_price"] = self.sell_prices["sell_price"].astype(float)

        # Step 6: Create self.prod_info
        self.prod_info = self.sales_data[["unique_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].drop_duplicates()

        # Convert data types in self.prod_info
        self.prod_info = self.prod_info.astype(str)

        # Ensure ds is of type Date
        self.sales_data["ds"] = pd.to_datetime(self.sales_data["ds"]).dt.date
        self.calendar["ds"] = pd.to_datetime(self.calendar["ds"]).dt.date
        self.sell_prices["ds"] = pd.to_datetime(self.sell_prices["ds"]).dt.date

        # Store processed data
        self.sales_data = self.sales_data[["unique_id", "ds", "y"]]
        self.calendar = self.calendar[["ds", "day_of_week", "is_weekend", "day_of_month", "week_of_month", "month", "week_num_year", "year", "num_events", "wm_yr_wk"]]
        self.sell_prices = self.sell_prices[["unique_id", "ds", "sell_price"]]
        self.prod_info = self.prod_info

        # Store processed data and validate with pandera schemas
        self.sales_data = SalesDataSchema.validate(self.sales_data)
        self.calendar = CalendarSchema.validate(self.calendar)
        self.sell_prices = SellPriceSchema.validate(self.sell_prices)
        self.prod_info = ProductInfoSchema.validate(self.prod_info)

        # Return processed DataFrames
        return self.sales_data, self.calendar, self.sell_prices, self.prod_info
    
    def prepare_calendar(self):
        """Prepares the calendar data with num_events and time-based features."""
        self.calendar["ds"] = pd.to_datetime(self.calendar["date"])

        # Sort by ds before assigning ds_id
        self.calendar = self.calendar.sort_values("ds").reset_index(drop=True)

        self.calendar["day_of_week"] = self.calendar["ds"].dt.dayofweek.astype(int)
        self.calendar["is_weekend"] = (self.calendar["day_of_week"] >= 5).astype(int)
        self.calendar["day_of_month"] = self.calendar["ds"].dt.day.astype(int)
        self.calendar["week_of_month"] = self.calendar["ds"].apply(lambda d: int((d.day - 1) / 7) + 1).astype(int)
        self.calendar["month"] = self.calendar["ds"].dt.month.astype(int)
        self.calendar["week_num_year"] = self.calendar["ds"].dt.isocalendar().week.astype(int)
        self.calendar["year"] = self.calendar["ds"].dt.year.astype(int)

        # Define event counts
        self.calendar["event_type_2"].fillna(0, inplace=True)
        self.calendar["event_name_2"].fillna(0, inplace=True)
        self.calendar["event_type_1"].fillna(0, inplace=True)
        self.calendar["event_name_1"].fillna(0, inplace=True)
        self.calendar["num_events"] = 0
        self.calendar.loc[self.calendar["event_type_2"] != 0, "num_events"] = 2
        self.calendar.loc[(self.calendar["event_type_2"] == 0) & (self.calendar["event_type_1"] != 0), "num_events"] = 1
        self.calendar["num_events"] = self.calendar["num_events"].astype(int)

        # Add ds_id for merging
        self.calendar["ds_id"] = "d_" + (self.calendar.index + 1).astype(str)
        
        # Drop any duplicates in ds to keep it unique for joins
        self.calendar = self.calendar.drop_duplicates(subset="ds")

        # Select relevant columns
        return self.calendar[[
            "ds_id", "ds", "day_of_week", "is_weekend", "day_of_month", "week_of_month", 
            "month", "week_num_year", "year", "num_events", "wm_yr_wk"
        ]]

    def filter_before_release(self):
        """Filters out rows in self.sales_data where the data is before the release week."""
        # Get the minimum week (`wm_yr_wk`) for each product (identified by `store_id` and `item_id`) as the release week
        release_df = self.sell_prices.groupby(["store_id", "item_id"])["wm_yr_wk"].min().reset_index()
        release_df["unique_id"] = release_df["item_id"] + "_" + release_df["store_id"]
        release_df = release_df[["unique_id", "wm_yr_wk"]].rename(columns={"wm_yr_wk": "release"})

        # Merge release week information onto self.sales_data and filter out data before the product's release week
        self.sales_data = self.sales_data.merge(release_df, on="unique_id", how="left")
        return self.sales_data[self.sales_data["wm_yr_wk"] >= self.sales_data["release"]].reset_index(drop=True)

    def split_data(self):
        """Splits the self.sales_data into train and test sets based on the horizon."""
        self.logger.info("Splitting data into train and test sets...")
        horizon = self.horizon
        train_list = []
        test_list = []

        for _, group in self.sales_data.groupby("unique_id"):
            if len(group) > horizon:
                train_list.append(group.iloc[:-horizon])
                test_list.append(group.iloc[-horizon:])
            else:
                train_list.append(group)

        self.train_df = pd.concat(train_list).reset_index(drop=True)
        self.test_df = pd.concat(test_list).reset_index(drop=True)
        self.logger.info(f"Train set shape: {self.train_df.shape}, Test set shape: {self.test_df.shape}")

        return self.train_df, self.test_df

    def save_to_catalog(self, spark, catalog_name, schema_name, train, test, calendar, sell_prices, prod_info):
        """Saves the processed DataFrames to Delta tables in chunks with UTC timestamp and change data feed."""
        self.logger.info("Starting to save DataFrames to Delta tables...")

        # Timestamp column
        timestamp = F.to_utc_timestamp(F.current_timestamp(), "UTC")

        # Define and write each table with timestamp and change data feed
        tables = {
            "calendar": calendar,
            "prod_info": prod_info,
            "test_set": test,
            "train_set": train,
            "sell_prices": sell_prices,
        }

        chunk_size = 500_000  # Define the chunk size

        for table_name, df in tables.items():
            self.logger.info(f"Starting processing for {table_name} DataFrame...")

            total_rows = len(df)
            num_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size != 0 else 0)
            
            for i in range(num_chunks):
                chunk_start = i * chunk_size
                chunk_end = min((i + 1) * chunk_size, total_rows)
                chunk = df.iloc[chunk_start:chunk_end]
                
                self.logger.info(f"Processing chunk {i + 1}/{num_chunks} for {table_name}, rows {chunk_start}-{chunk_end}")

                # Convert chunk to Spark DataFrame and add timestamp
                start_time = time.time()
                spark_df = spark.createDataFrame(chunk).withColumn("update_timestamp_utc", timestamp)
                self.logger.info(f"Converted chunk {i + 1} to Spark DataFrame in {time.time() - start_time:.2f} seconds")

                # Append chunk to Delta table
                start_time = time.time()
                spark_df.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.{table_name}")
                self.logger.info(f"Appended chunk {i + 1} to {table_name} in {time.time() - start_time:.2f} seconds")

            # Enable change data feed after all chunks are uploaded
            spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.{table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
            self.logger.info(f"Enabled change data feed for {table_name}")

        self.logger.info("All DataFrames saved to Delta tables with change data feed enabled.")



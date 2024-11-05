from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import DoubleType

import pandera.pyspark as pa
from m5_forecasting.schemas.processed_data import SalesDataSchema, CalendarSchema, SellPriceSchema, ProductInfoSchema


class DataProcessor:
    def __init__(self, config, sales_data, calendar, sell_price):
        """Initialize the DataProcessor with configuration details and data."""
        self.config = config
        self.sales_data = sales_data
        self.calendar = calendar
        self.sell_price = sell_price
        self.horizon = config.horizon
        self.prod_info = None
        self.train_df = None
        self.test_df = None

    def preprocess_data(self):
        """Processes the data to create self.sales_data, self.calendar, self.sell_price, and self.prod_info."""
        # Step 1: Generate unique_id and unpivot sales data
        self.sales_data = (
            self.sales_data
            .withColumn("unique_id", F.concat_ws("_", F.col("item_id"), F.col("store_id")))
            .selectExpr(
                "unique_id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
                "stack({}, {}) as (ds_id, y)".format(
                    len([col for col in self.sales_data.columns if col.startswith("d_")]),
                    ', '.join(["'{}', {}".format(col, col) for col in self.sales_data.columns if col.startswith("d_")])
                )
            )
        )

        # Convert data types
        self.sales_data = (
            self.sales_data
            .withColumn("ds_id", F.col("ds_id").cast("STRING"))
            .withColumn("y", F.col("y").cast("INT"))
        )

        # Step 2: Prepare calendar with time features and num_events
        self.calendar = self.prepare_calendar()

        # Step 3: Merge sales data with calendar on ds_id to map ds and add wm_yr_wk
        self.sales_data = (
            self.sales_data
            .join(self.calendar.select("ds_id", "ds", "wm_yr_wk"), on="ds_id", how="left")
        )

        # Step 4: Create product information (prod_info) from sales data
        self.prod_info = (
            self.sales_data
            .select(
                F.col("unique_id").cast("STRING"),
                F.col("item_id").cast("STRING"),
                F.col("dept_id").cast("STRING"),
                F.col("cat_id").cast("STRING"),
                F.col("store_id").cast("STRING"),
                F.col("state_id").cast("STRING")
            )
            .dropDuplicates()
        )

        # Step 5: Prepare sell_price with unique_id and expanded ds
        self.sell_price = (
            self.sell_price
            .withColumn("unique_id", F.concat_ws("_", F.col("item_id"), F.col("store_id")))
            .withColumn("unique_id", F.col("unique_id").cast("STRING"))
            .withColumn("wm_yr_wk", F.col("wm_yr_wk").cast("INT"))
            .join(
                self.calendar.select("wm_yr_wk", "ds", "week_num_year", "year"),
                on="wm_yr_wk",
                how="left"
            )
            .select(
                "unique_id", "item_id", "store_id", "ds", "sell_price", "week_num_year", "year", "wm_yr_wk"
            )
            .withColumn("sell_price", F.col("sell_price").cast("DOUBLE"))
        )

        # Enrich sell_price with dept_id
        self.sell_price = (
            self.sell_price
            .join(self.prod_info.select("unique_id", "dept_id"), on="unique_id", how="inner")
        )

        # Compute dept_wkly_avg_sell_price
        dept_wkly_avg_sell_price_df = (
            self.sell_price
            .groupBy("dept_id", "week_num_year", "year")
            .agg(F.round(F.avg("sell_price"), 2).cast(DoubleType()).alias("dept_wkly_avg_sell_price"))
        )

        # Add dept_wkly_avg_sell_price to sell_price
        self.sell_price = (
            self.sell_price
            .join(
                dept_wkly_avg_sell_price_df,
                on=["dept_id", "week_num_year", "year"],
                how="left"
            )
            .select("unique_id", "ds", "sell_price", "dept_wkly_avg_sell_price", "item_id", "store_id", "wm_yr_wk")
        )

        # Step 6: Filter to remove rows before the product release week
        self.sales_data = self.filter_before_release()

        # Ensure ds is of type TIMESTAMP
        self.sales_data = self.sales_data.withColumn("ds", F.col("ds").cast("TIMESTAMP")).select(["unique_id", "ds", "y"])
        self.calendar = self.calendar.withColumn("ds", F.col("ds").cast("TIMESTAMP")).drop("ds_id", "wm_yr_wk")
        self.sell_price = self.sell_price.withColumn("ds", F.col("ds").cast("TIMESTAMP")).drop("item_id", "store_id", "wm_yr_wk")

        # Validate DataFrames with Pandera
        self.sales_data = SalesDataSchema.validate(self.sales_data)
        self.calendar = CalendarSchema.validate(self.calendar)
        self.sell_price = SellPriceSchema.validate(self.sell_price)
        self.prod_info = ProductInfoSchema.validate(self.prod_info)

        # Store processed data
        return self.sales_data, self.calendar, self.sell_price, self.prod_info

    def prepare_calendar(self):
        """Prepares the calendar data with num_events and time-based features."""
        self.calendar = (
            self.calendar
            .withColumnRenamed("date", "ds")
            .withColumn("ds", F.to_timestamp("ds"))
            .sort("ds")
            .withColumn("day_of_week", (F.dayofweek("ds") - 1).cast("INT"))
            .withColumn("is_weekend", F.when(F.col("day_of_week") >= 5, 1).otherwise(0))
            .withColumn("day_of_month", F.dayofmonth("ds").cast("INT"))
            .withColumn("week_of_month", (F.floor((F.dayofmonth("ds") - 1) / 7) + 1).cast("INT"))
            .withColumn("month", F.col("month").cast("INT"))
            .withColumn("week_num_year", F.col("wm_yr_wk").cast("INT"))
            .withColumn("year", F.col("year").cast("INT"))
            .withColumn("wm_yr_wk", F.col("wm_yr_wk").cast("INT"))
            .withColumn("ds_id", F.concat(F.lit("d_"), F.monotonically_increasing_id() + 1).cast("STRING"))
            .replace("NA", None)
            .fillna({"event_type_2": "0", "event_name_2": "0", "event_type_1": "0", "event_name_1": "0"})
            .withColumn(
                "num_events",
                F.when((F.col("event_type_1") != "0") & (F.col("event_type_2") != "0"), 2)
                .when(F.col("event_type_1") != "0", 1)
                .otherwise(0)
            )
            .dropDuplicates(["ds"])
        )

        return self.calendar.select(
            "ds", "day_of_week", "is_weekend", "day_of_month", "week_of_month",
            "month", "week_num_year", "year", "num_events", "wm_yr_wk", "ds_id"
        )

    def filter_before_release(self):
        """Filters out rows in self.sales_data where the data is before the release week."""
        # Get minimum wm_yr_wk per unique_id
        release_df = (
            self.sell_price
            .groupBy("unique_id")
            .agg(F.min("wm_yr_wk").alias("release"))
            .select("unique_id", "release")
        )

        # Join to filter out records in sales_data before the release week
        return (
            self.sales_data
            .join(release_df, on="unique_id", how="left")
            .filter(F.col("wm_yr_wk") >= F.col("release"))
            .drop("release")
        )

    def split_data(self):
        """Splits the self.sales_data into train and test sets based on the horizon."""
        horizon = self.horizon

        # Define a window partitioned by unique_id and ordered by ds
        window_spec = Window.partitionBy("unique_id").orderBy("ds")

        # Add a row number within each unique_id
        self.sales_data = self.sales_data.withColumn("row_number", F.row_number().over(window_spec))

        # Get the maximum row number per unique_id
        max_row_df = (
            self.sales_data.groupBy("unique_id")
            .agg(F.max("row_number").alias("max_row_number"))
        )

        # Calculate the split point
        split_df = max_row_df.withColumn("split_row_number", F.col("max_row_number") - horizon)

        # Join back to sales_data to get the split_row_number per unique_id
        self.sales_data = self.sales_data.join(split_df.select("unique_id", "split_row_number"), on="unique_id", how="left")

        # Create train and test flags
        self.sales_data = self.sales_data.withColumn(
            "dataset",
            F.when(F.col("row_number") <= F.col("split_row_number"), "train")
            .otherwise("test")
        )

        # Split into train and test DataFrames
        self.train_df = self.sales_data.filter(F.col("dataset") == "train").drop("row_number", "max_row_number", "split_row_number", "dataset")
        self.test_df = self.sales_data.filter(F.col("dataset") == "test").drop("row_number", "max_row_number", "split_row_number", "dataset")

        return self.train_df, self.test_df

    def save_to_catalog(self, spark, catalog_name, schema_name, train, test, calendar, sell_price, prod_info):
        """
        Saves the processed DataFrames to Delta tables with UTC timestamp and change data feed enabled.

        Parameters:
        - spark: SparkSession object.
        - catalog_name: Name of the catalog.
        - schema_name: Name of the schema (database).
        - train: Training DataFrame.
        - test: Testing DataFrame.
        - calendar: Calendar DataFrame.
        - sell_price: Sell Price DataFrame.
        - prod_info: Product Info DataFrame.
        """
        # Get the current UTC timestamp
        timestamp = F.to_utc_timestamp(F.current_timestamp(), "UTC")

        # List of DataFrames and their corresponding table names
        tables = [
            (train, 'train_set'),
            (test, 'test_set'),
            (calendar, 'calendar'),
            (sell_price, 'sell_price'),
            (prod_info, 'prod_info'),
        ]

        # Loop through each DataFrame and save to Delta table with CDF enabled
        for df, table_name in tables:
            # Add 'update_timestamp_utc' column
            df_with_timestamp = df.withColumn("update_timestamp_utc", timestamp)

            # Save the DataFrame as a Delta table with CDF enabled
            df_with_timestamp.write.mode("overwrite") \
                .format("delta") \
                .option("overwriteSchema", "true") \
                .option("delta.enableChangeDataFeed", "true") \
                .saveAsTable(f"{catalog_name}.{schema_name}.{table_name}")

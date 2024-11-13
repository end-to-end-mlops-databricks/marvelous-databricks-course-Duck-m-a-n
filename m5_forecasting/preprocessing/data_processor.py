import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import *

from m5_forecasting.schemas.processed_data import CombinedDataFrameSchema

class DataProcessor:
    def __init__(self, config, sales_data, calendar, sell_price, weather):
        self.sales_data = sales_data
        self.calendar = calendar
        self.sell_price = sell_price
        self.weather = weather
        self.prod_info = None
        self.min_ds_sell_price = None
        self.combined_df = None
        self.horizon = config.horizon
        self.train_df = None
        self.test_df = None

    def preprocess_data(self):
        """Processes the data and returns the combined DataFrame."""
        self.prepare_calendar()
        self.prepare_weather()
        self.prepare_sales()
        self.prepare_sell_price()
        self.filter_sales()
        self.merge_data()
        self.finalize_data()
        self.combined_df = CombinedDataFrameSchema.validate(self.combined_df)
        return self.combined_df

    def prepare_calendar(self):
        """Prepares the calendar data with additional time features."""
        self.calendar["ds"] = pd.to_datetime(self.calendar["date"])
        self.calendar = self.calendar.sort_values("ds").reset_index(drop=True)

        self.calendar["day_of_week"] = self.calendar["ds"].dt.dayofweek.astype("int8")
        self.calendar["is_weekend"] = (self.calendar["day_of_week"] >= 5).astype("int8")
        self.calendar["day_of_month"] = self.calendar["ds"].dt.day.astype("int8")
        self.calendar["week_of_month"] = ((self.calendar["ds"].dt.day - 1) // 7 + 1).astype("int8")
        self.calendar["month"] = self.calendar["ds"].dt.month.astype("int8")
        self.calendar["year"] = self.calendar["ds"].dt.year.astype("int16")

        self.calendar = self.calendar.fillna({
            "event_type_1": "0", "event_name_1": "0",
            "event_type_2": "0", "event_name_2": "0"
        })

        self.calendar["num_events"] = np.where(
            (self.calendar["event_type_1"] != "0") & (self.calendar["event_type_2"] != "0"), 2,
            np.where(self.calendar["event_type_1"] != "0", 1, 0)
        ).astype("int8")

        self.calendar["ds_id"] = ["d_" + str(i + 1) for i in range(len(self.calendar))]

        self.calendar = self.calendar[[
            "ds", "day_of_week", "is_weekend", "day_of_month", "week_of_month",
            "month", "year", "num_events", "wm_yr_wk", "ds_id"
        ]]

    def prepare_weather(self):
        """Prepares the weather data with temperature features."""
        self.weather["ds"] = pd.to_datetime(self.weather["ds"])

        self.weather = self.weather.merge(
            self.calendar[["ds", "week_of_month", "month", "year", "wm_yr_wk"]],
            on="ds",
            how="inner"
        )

        avg_weekly_temp = self.weather.groupby(["wm_yr_wk", "state_id"])["temp"].mean().reset_index()
        avg_weekly_temp["avg_weekly_temp"] = avg_weekly_temp["temp"].round(2).astype("float32")
        avg_weekly_temp.drop(columns="temp", inplace=True)

        avg_monthly_temp = self.weather.groupby(["month", "year", "state_id"])["temp"].mean().reset_index()
        avg_monthly_temp["avg_monthly_temp"] = avg_monthly_temp["temp"].round(2).astype("float32")
        avg_monthly_temp.drop(columns="temp", inplace=True)

        self.weather = self.weather.merge(avg_monthly_temp, on=["month", "year", "state_id"], how="inner")
        self.weather = self.weather.merge(avg_weekly_temp, on=["wm_yr_wk", "state_id"], how="inner")

        self.weather["percent_diff_weekly_temp"] = (
            (self.weather["temp"] - self.weather["avg_weekly_temp"]) / self.weather["avg_weekly_temp"]
        ).round(2).astype("float32")

        self.weather["percent_diff_monthly_temp"] = (
            (self.weather["temp"] - self.weather["avg_monthly_temp"]) / self.weather["avg_monthly_temp"]
        ).round(2).astype("float32")

        self.weather = self.weather.sort_values(by=["state_id", "ds"]).reset_index(drop=True)
        self.weather["avg_28_day_temp"] = (
            self.weather.groupby("state_id")["temp"]
            .transform(lambda x: x.rolling(window=28, min_periods=1).mean())
            .round(2).astype("float32")
        )

        self.weather["percent_diff_28_day_avg_temp"] = (
            (self.weather["temp"] - self.weather["avg_28_day_temp"]) / self.weather["avg_28_day_temp"]
        ).round(2).astype("float32")

        self.weather["temp"] = self.weather["temp"].astype("float32")

        self.weather = self.weather[[
            "ds", "state_id", "temp", "conditions", "avg_weekly_temp", "avg_monthly_temp", "avg_28_day_temp", 
            "percent_diff_weekly_temp", "percent_diff_monthly_temp", "percent_diff_28_day_avg_temp"
        ]]

    def prepare_sales(self):
        """Prepares the sales data and product information."""
        self.sales_data["unique_id"] = self.sales_data["item_id"] + "_" + self.sales_data["store_id"]

        date_columns = [col for col in self.sales_data.columns if col.startswith("d_")]

        self.sales_data = self.sales_data.melt(
            id_vars=["unique_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
            value_vars=date_columns,
            var_name="ds_id",
            value_name="y"
        )

        self.sales_data["y"] = self.sales_data["y"].astype("int32")

        self.prod_info = self.sales_data[[
            "unique_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"
        ]].drop_duplicates().reset_index(drop=True)

        self.sales_data = self.sales_data.merge(
            self.calendar[[
                "ds", "ds_id", "day_of_week", "is_weekend", "day_of_month",
                "week_of_month", "month", "year", "num_events"
            ]],
            on="ds_id",
            how="left"
        )

        self.sales_data.drop(columns="ds_id", inplace=True)

        categorical_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "unique_id"]
        for col in categorical_cols:
            self.sales_data[col] = self.sales_data[col].astype("category")

    def prepare_sell_price(self):
        """Prepares the sell price data with additional features."""
        self.sell_price["unique_id"] = self.sell_price["item_id"] + "_" + self.sell_price["store_id"]

        self.sell_price = self.sell_price.merge(
            self.calendar[["wm_yr_wk", "ds", "month", "year"]],
            on="wm_yr_wk",
            how="left"
        )

        self.sell_price = self.sell_price.merge(
            self.prod_info[["unique_id", "dept_id", "cat_id", "state_id"]],
            on="unique_id",
            how="inner"
        )

        self.min_ds_sell_price = self.sell_price.groupby("unique_id")["ds"].min().reset_index()
        self.min_ds_sell_price.rename(columns={"ds": "min_ds"}, inplace=True)

        dept_avg_sell_price = self.sell_price.groupby(["dept_id", "ds"])["sell_price"].mean().reset_index()
        dept_avg_sell_price["dept_avg_sell_price"] = dept_avg_sell_price["sell_price"].round(2).astype("float32")
        dept_avg_sell_price.drop(columns="sell_price", inplace=True)

        cat_avg_sell_price = self.sell_price.groupby(["cat_id", "ds"])["sell_price"].mean().reset_index()
        cat_avg_sell_price["cat_avg_sell_price"] = cat_avg_sell_price["sell_price"].round(2).astype("float32")
        cat_avg_sell_price.drop(columns="sell_price", inplace=True)

        store_dept_avg_sell_price = self.sell_price.groupby(["store_id", "dept_id", "ds"])["sell_price"].mean().reset_index()
        store_dept_avg_sell_price["store_dept_avg_sell_price"] = store_dept_avg_sell_price["sell_price"].round(2).astype("float32")
        store_dept_avg_sell_price.drop(columns="sell_price", inplace=True)

        state_dept_avg_sell_price = self.sell_price.groupby(["state_id", "dept_id", "ds"])["sell_price"].mean().reset_index()
        state_dept_avg_sell_price["state_dept_avg_sell_price"] = state_dept_avg_sell_price["sell_price"].round(2).astype("float32")
        state_dept_avg_sell_price.drop(columns="sell_price", inplace=True)

        monthly_avg_sell_price = self.sell_price.groupby(["unique_id", "month", "year"])["sell_price"].mean().reset_index()
        monthly_avg_sell_price["monthly_avg_sell_price"] = monthly_avg_sell_price["sell_price"].round(2).astype("float32")
        monthly_avg_sell_price.drop(columns="sell_price", inplace=True)

        self.sell_price = self.sell_price.merge(monthly_avg_sell_price, on=["unique_id", "month", "year"], how="left")
        self.sell_price = self.sell_price.merge(dept_avg_sell_price, on=["dept_id", "ds"], how="left")
        self.sell_price = self.sell_price.merge(cat_avg_sell_price, on=["cat_id", "ds"], how="left")
        self.sell_price = self.sell_price.merge(store_dept_avg_sell_price, on=["store_id", "dept_id", "ds"], how="left")
        self.sell_price = self.sell_price.merge(state_dept_avg_sell_price, on=["state_id", "dept_id", "ds"], how="left")

        self.sell_price["percent_diff_monthly_sell_price"] = (
            (self.sell_price["sell_price"] - self.sell_price["monthly_avg_sell_price"]) / self.sell_price["monthly_avg_sell_price"]
        ).round(2).astype("float32")

        self.sell_price["sell_price"] = self.sell_price["sell_price"].astype("float32")

        self.sell_price = self.sell_price[[
            "unique_id", "ds", "sell_price", "monthly_avg_sell_price", "percent_diff_monthly_sell_price",
            "dept_avg_sell_price", "cat_avg_sell_price", "store_dept_avg_sell_price", "state_dept_avg_sell_price"
        ]]

    def filter_sales(self):
        """Filters the sales data based on the product release date."""
        self.sales_data = self.sales_data.merge(
            self.min_ds_sell_price,
            on="unique_id",
            how="left"
        )

        self.sales_data["ds"] = pd.to_datetime(self.sales_data["ds"])
        self.sales_data["min_ds"] = pd.to_datetime(self.sales_data["min_ds"])

        self.sales_data = self.sales_data[self.sales_data["ds"] >= self.sales_data["min_ds"]]

        self.sales_data.drop(columns=["min_ds"], inplace=True)

    def merge_data(self):
        """Merges sales, weather, and sell price data."""
        self.combined_df = self.sales_data.merge(
            self.weather,
            on=["ds", "state_id"],
            how="left"
        )

        self.combined_df = self.combined_df.merge(
            self.sell_price,
            on=["unique_id", "ds"],
            how="left"
        )

    def finalize_data(self):
        """Finalizes the combined data with proper column order and data types."""
        column_order = [
            "unique_id", "ds", "y", "sell_price", "temp", "conditions", "num_events",
            "item_id", "dept_id", "cat_id", "store_id", "state_id",
            "day_of_week", "is_weekend", "day_of_month", "week_of_month", "month", "year",
            "avg_weekly_temp", "avg_monthly_temp", "avg_28_day_temp",
            "percent_diff_weekly_temp", "percent_diff_monthly_temp", "percent_diff_28_day_avg_temp",
            "monthly_avg_sell_price", "percent_diff_monthly_sell_price",
            "dept_avg_sell_price", "cat_avg_sell_price", "store_dept_avg_sell_price", "state_dept_avg_sell_price"
        ]
        self.combined_df = self.combined_df[column_order]

        categorical_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "unique_id", "conditions"]
        for col in categorical_cols:
            self.combined_df[col] = self.combined_df[col].astype("string")

        self.combined_df["ds"] = pd.to_datetime(self.combined_df["ds"])

        numeric_cols = {
            "y": "int32",
            "day_of_week": "int8",
            "is_weekend": "int8",
            "day_of_month": "int8",
            "week_of_month": "int8",
            "month": "int8",
            "year": "int16",
            "num_events": "int8",
            "temp": "float32",
            "avg_weekly_temp": "float32",
            "avg_monthly_temp": "float32",
            "avg_28_day_temp": "float32",
            "percent_diff_weekly_temp": "float32",
            "percent_diff_monthly_temp": "float32",
            "percent_diff_28_day_avg_temp": "float32",
            "sell_price": "float32",
            "monthly_avg_sell_price": "float32",
            "percent_diff_monthly_sell_price": "float32",
            "dept_avg_sell_price": "float32",
            "cat_avg_sell_price": "float32",
            "store_dept_avg_sell_price": "float32",
            "state_dept_avg_sell_price": "float32"
        }
        for col, dtype in numeric_cols.items():
            self.combined_df[col] = self.combined_df[col].astype(dtype)

    def split_data(self):
        """Splits the self.sales_data into train and test sets based on the horizon."""
        horizon = self.horizon

        self.combined_df = self.combined_df.sort_values(by=['unique_id', 'ds'])

        self.combined_df['row_number'] = self.combined_df.groupby('unique_id').cumcount() + 1

        max_row_df = self.combined_df.groupby('unique_id')['row_number'].max().reset_index()
        max_row_df.rename(columns={'row_number': 'max_row_number'}, inplace=True)

        max_row_df['split_row_number'] = max_row_df['max_row_number'] - horizon

        self.combined_df = self.combined_df.merge(
            max_row_df[['unique_id', 'split_row_number']],
            on='unique_id',
            how='left'
        )

        self.combined_df['dataset'] = np.where(
            self.combined_df['row_number'] <= self.combined_df['split_row_number'],
            'train',
            'test'
        )

        self.train_df = self.combined_df[self.combined_df['dataset'] == 'train'].drop(
            columns=['row_number', 'split_row_number', 'dataset']
        ).reset_index(drop=True)

        self.test_df = self.combined_df[self.combined_df['dataset'] == 'test'].drop(
            columns=['row_number', 'split_row_number', 'dataset']
        ).reset_index(drop=True)

        return self.train_df, self.test_df
    def save_to_catalog(self, spark, train_set, test_set, catalog_name, schema_name):
        """
        Saves the train and test DataFrames to Delta tables with UTC timestamp and change data feed enabled.
        """

        timestamp = F.to_utc_timestamp(F.current_timestamp(), "UTC")

        tables = [
            (train_set, 'train_set'),
            (test_set, 'test_set')
        ]

        for pandas_df, table_name in tables:
            df = spark.createDataFrame(pandas_df)

            df_with_timestamp = df.withColumn("update_timestamp_utc", timestamp)

            df_with_timestamp.write.mode("overwrite") \
                .format("delta") \
                .option("overwriteSchema", "true") \
                .option("delta.enableChangeDataFeed", "true") \
                .saveAsTable(f"{catalog_name}.{schema_name}.{table_name}")

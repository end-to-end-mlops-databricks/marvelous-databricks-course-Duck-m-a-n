import logging

import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, config):
        """Initialize the DataProcessor with configuration details."""
        required_keys = ["sales_filepath", "calendar_filepath", "sell_prices_filepath", "horizon"]
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Missing required configuration key: {key}")
        self.config = config
        self.sales_data = self.load_data(config["sales_filepath"])
        self.calendar = self.load_data(config["calendar_filepath"])
        self.sell_prices = self.load_data(config["sell_prices_filepath"])
        self.horizon = config["horizon"]
        self.processor = None
        self.train_df = None
        self.test_df = None
        self.original_row_count = None
        self.filtered_row_count = None

    def load_data(self, filepath):
        """Loads data from the specified file path."""
        logger.info(f"Loading data from {filepath}")
        try:
            return pd.read_csv(filepath)
        except FileNotFoundError:
            logger.error(f"The file {filepath} was not found.")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"The file {filepath} is empty.")
            raise

    def process_data(self):
        """Processes the data by applying necessary transformations and preparing for train-test split."""
        logger.info("Starting data processing...")

        # Step 1: Identify non-date columns
        non_date_columns = ["store_id", "item_id", "dept_id", "cat_id", "state_id"]

        # Step 2: Create unique_id by combining store_id and item_id
        self.sales_data["unique_id"] = self.sales_data["item_id"] + "_" + self.sales_data["store_id"]
        logger.info(f"Step 1/6 complete: Added unique_id column. Data shape: {self.sales_data.shape}")

        # Step 3: Melt the sales_data DataFrame from wide to long format
        date_columns = [col for col in self.sales_data.columns if col.startswith("d_")]
        self.sales_data = pd.melt(
            self.sales_data,
            id_vars=non_date_columns + ["unique_id"],
            value_vars=date_columns,
            var_name="ds_id",
            value_name="y",
        )
        # Store original row count before filtering
        self.original_row_count = self.sales_data.shape[0]
        logger.info(f"Step 2/6 complete: Melted data. Data shape: {self.sales_data.shape}")

        # Step 4: Merge with calendar and handle events
        self.prepare_calendar()
        logger.info(f"Step 3/6 complete: Merged calendar data. Data shape: {self.sales_data.shape}")

        # Step 5: Add price information and time-based features
        self.add_sell_prices()
        self.add_time_features()
        logger.info(f"Step 4/6 complete: Added sell prices and time features. Data shape: {self.sales_data.shape}")

        # Step 6: Filter rows before release week
        self.filter_before_release()
        self.filtered_row_count = self.sales_data.shape[0]
        logger.info(f"Step 5/6 complete: Filtered out data before release week. Data shape: {self.sales_data.shape}")

        # Step 7: Drop unwanted columns and reorder columns
        self.finalize_dataset()
        logger.info(f"Step 6/6 complete: Finalized dataset. Data shape: {self.sales_data.shape}")

        # Set processed data to self.processor
        self.processor = self.sales_data
        return self.processor

    def prepare_calendar(self):
        """Prepares the calendar data with num_events and merges it into the sales data."""
        self.calendar["ds_id"] = "d_" + (self.calendar.index + 1).astype(str)
        self.calendar["event_type_2"].fillna(0, inplace=True)
        self.calendar["event_name_2"].fillna(0, inplace=True)
        self.calendar["event_type_1"].fillna(0, inplace=True)
        self.calendar["event_name_1"].fillna(0, inplace=True)
        self.calendar["num_events"] = 0
        self.calendar.loc[self.calendar["event_type_2"] != 0, "num_events"] = 2
        self.calendar.loc[(self.calendar["event_type_2"] == 0) & (self.calendar["event_type_1"] != 0), "num_events"] = 1
        calendar_mapped = self.calendar[["wm_yr_wk", "date", "ds_id"]].rename(columns={"date": "ds"})
        self.sales_data = self.sales_data.merge(calendar_mapped, on="ds_id", how="left")
        self.sales_data = self.sales_data.merge(self.calendar[["ds_id", "num_events"]], on="ds_id", how="left")

    def add_sell_prices(self):
        """Adds sell prices to the sales data."""
        self.sell_prices["unique_id"] = self.sell_prices["item_id"] + "_" + self.sell_prices["store_id"]
        self.sales_data = self.sales_data.merge(
            self.sell_prices[["unique_id", "wm_yr_wk", "sell_price"]], on=["unique_id", "wm_yr_wk"], how="left"
        )

    def add_time_features(self):
        """Adds time-based features such as day of the week, month, year, etc."""
        self.sales_data["ds"] = pd.to_datetime(self.sales_data["ds"])
        self.sales_data["day_of_week"] = self.sales_data["ds"].dt.dayofweek
        self.sales_data["is_weekend"] = (self.sales_data["day_of_week"] >= 5).astype(int)
        self.sales_data["day_of_month"] = self.sales_data["ds"].dt.day
        # self.sales_data['week_of_month'] = (self.sales_data['day_of_month'] / 7).apply(np.ceil).astype(int)
        self.sales_data["week_of_month"] = self.sales_data["ds"].apply(lambda d: int((d.day - 1) / 7) + 1)
        self.sales_data["month"] = self.sales_data["ds"].dt.month
        self.sales_data["week_num_year"] = self.sales_data["ds"].dt.isocalendar().week
        self.sales_data["year"] = self.sales_data["ds"].dt.year

    def filter_before_release(self):
        """Filters out rows where sales data is before the release week."""
        release_df = self.sell_prices.groupby(["store_id", "item_id"])["wm_yr_wk"].min().reset_index()
        release_df["unique_id"] = release_df["item_id"] + "_" + release_df["store_id"]
        release_df = release_df[["unique_id", "wm_yr_wk"]].rename(columns={"wm_yr_wk": "release"})
        self.sales_data = self.sales_data.merge(release_df, on="unique_id", how="left")
        self.sales_data = self.sales_data[self.sales_data["wm_yr_wk"] >= self.sales_data["release"]].reset_index(
            drop=True
        )

    def finalize_dataset(self):
        """Drops unnecessary columns and reorders the dataset."""
        self.sales_data = self.sales_data.drop(columns=["ds_id", "wm_yr_wk", "release"])
        desired_column_order = [
            "unique_id",
            "ds",
            "y",
            "sell_price",
            "num_events",
            "cat_id",
            "dept_id",
            "state_id",
            "store_id",
            "day_of_week",
            "is_weekend",
            "day_of_month",
            "week_of_month",
            "month",
            "week_num_year",
            "year",
        ]
        self.sales_data = self.sales_data[desired_column_order]

    def split_data(self):
        """Splits the processed data into train and test sets based on the horizon from the config."""
        if self.processor is None:
            raise ValueError("Data has not been processed. Please call 'process_data' before 'split_data")
        horizon = self.horizon
        train_list = []
        test_list = []

        for _, group in self.processor.groupby("unique_id"):
            if len(group) > horizon:
                train_list.append(group.iloc[:-horizon])
                test_list.append(group.iloc[-horizon:])
            else:
                train_list.append(group)

        self.train_df = pd.concat(train_list).reset_index(drop=True)
        self.test_df = pd.concat(test_list).reset_index(drop=True)
        logger.info(f"Train DataFrame shape: {self.train_df.shape}")
        logger.info(f"Test DataFrame shape: {self.test_df.shape}")

        return self.train_df, self.test_df

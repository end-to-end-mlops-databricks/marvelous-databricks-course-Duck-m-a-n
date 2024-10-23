import pandas as pd


class DataChecker:
    def __init__(self, train_df, test_df, original_row_count, filtered_row_count):
        self.train_df = train_df
        self.test_df = test_df
        self.original_row_count = original_row_count
        self.filtered_row_count = filtered_row_count

    def not_sales_row_drop_summary_check(self):
        """Returns the original, filtered, and dropped row counts, and the percentage of dropped rows as a message."""
        dropped_row_count = self.original_row_count - self.filtered_row_count
        drop_percentage = (dropped_row_count / self.original_row_count) * 100

        if self.original_row_count < self.filtered_row_count:
            raise ValueError("Filtered row count cannot be greater than original row count.")

        message = (
            f"\nOriginal row count:                   {self.original_row_count}\n"
            f"Filter rows before release row count: {self.filtered_row_count}\n"
            f"Number of rows dropped:               {dropped_row_count}\n"
            f"Percentage of rows dropped:           {drop_percentage:.2f}%"
        )
        return message

    def nulls_and_shape_check(self):
        """Returns a message with null values and shapes for the train and test DataFrames."""
        # Check for null values
        train_nulls = self.train_df.isnull().sum()
        test_nulls = self.test_df.isnull().sum()

        train_nulls = train_nulls[train_nulls > 0]
        test_nulls = test_nulls[test_nulls > 0]

        message = (
            f"\nTrain DataFrame Shape: {self.train_df.shape}\n"
            f"Test DataFrame Shape: {self.test_df.shape}\n"
            f"Train Data Null Values:\n{train_nulls}\n"
            f"Test Data Null Values:\n{test_nulls}\n"
        )
        return message

    def gaps_in_date_check(self, df):
        """Checks for missing dates (gaps) in each time series grouped by unique_id and returns a message."""
        required_columns = {"unique_id", "ds"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Dataframe must contain columns: {required_columns}")

        # Convert to datetime if not already
        df["ds"] = pd.to_datetime(df["ds"])

        # Pre-sort data to optimize performance
        df = df.sort_values(["unique_id", "ds"])

        # Create a DataFrame with all possible combinations of unique_id and dates
        unique_ids = df["unique_id"].unique()
        all_dates = pd.date_range(df["ds"].min(), df["ds"].max(), freq="D")
        all_combinations = pd.MultiIndex.from_product([unique_ids, all_dates], names=["unique_id", "ds"]).to_frame(
            index=False
        )

        # Merge with the original DataFrame to find missing entries
        merged_df = all_combinations.merge(df, on=["unique_id", "ds"], how="left", indicator=True)
        missing = merged_df[merged_df["_merge"] == "left_only"]

        # Construct message from missing_series
        if not missing.empty:
            missing_series = missing.groupby("unique_id")["ds"].apply(
                lambda dates: dates.dt.strftime("%Y-%m-%d").tolist()
            )
            return "\n".join(missing_series)
        if missing.empty:
            return "No missing timestamp gaps found in any time series."

    def unique_id_per_date_check(self, df: pd.DataFrame) -> str:
        """Checks for duplicate unique_id per date and returns a message."""
        required_columns = {"unique_id", "ds"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Dataframe must contain columns: {required_columns}")

        duplicates = df[df.duplicated(subset=["unique_id", "ds"], keep=False)]

        if not duplicates.empty:
            formatted_duplicates = duplicates.to_string(index=False)
            return (
                "Duplicate entries found for the following unique_id and ds combinations:\n" f"{formatted_duplicates}"
            )
        return "No duplicate unique_id for any date found."

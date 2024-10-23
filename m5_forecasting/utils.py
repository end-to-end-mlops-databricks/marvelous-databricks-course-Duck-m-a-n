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

        message = (
            f"\nTrain DataFrame Shape: {self.train_df.shape}\n"
            f"Test DataFrame Shape: {self.test_df.shape}\n"
            f"Train Data Null Values:\n{train_nulls}\n"
            f"Test Data Null Values:\n{test_nulls}\n"

        )
        return message

    def gaps_in_date_check(self, df):
        """Checks for missing dates (gaps) in each time series grouped by unique_id and returns a message."""
        missing_series = []
        for unique_id, group in df.groupby('unique_id'):
            all_dates = pd.date_range(start=group['ds'].min(), end=group['ds'].max(), freq='D')
            missing_dates = all_dates.difference(group['ds'])
            if not missing_dates.empty:
                missing_series.append(f"Missing dates found for series {unique_id}: {missing_dates}")

        if missing_series:
            return "\n".join(missing_series)
        return "No missing timestamp gaps found in any time series."

    def unique_id_per_date_check(self, df):
        """Checks for duplicate unique_id per date and returns a message."""
        duplicates = df.groupby(['unique_id', 'ds']).size().reset_index(name='count')
        duplicates = duplicates[duplicates['count'] > 1]

        if not duplicates.empty:
            return f"Duplicate entries found for the following unique_id and ds combinations:\n{duplicates}"
        return "No duplicate unique_id for any date found."

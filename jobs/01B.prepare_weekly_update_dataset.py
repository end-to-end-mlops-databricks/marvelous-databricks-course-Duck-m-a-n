#01B.prepare_weekly_update_dataset.py

from pyspark.sql import SparkSession

from m5_forecasting.config import Config

def main():
    spark = SparkSession.builder.getOrCreate()
    config = Config.from_yaml("../configs/project_config.yml")

    # Load Configurations
    catalog_name = config.catalog_name
    schema_name = config.schema_name

    sql_query = f"""
    CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.weekly_update_set AS
    WITH
    -- Step 1: Retrieve the last update timestamp from the test_set
    test_set_update AS (
        SELECT
            unique_id,
            MAX(ds) AS modified_timestamp_utc
        FROM {catalog_name}.{schema_name}.test_set
        WHERE day_of_week = 6
        GROUP BY unique_id
    ),

    -- Step 2: Test Set Data with Update Timestamp
    test_set_with_update AS (
        SELECT
            t.*,
            u.modified_timestamp_utc
        FROM {catalog_name}.{schema_name}.test_set t
        JOIN test_set_update u ON t.unique_id = u.unique_id
    ),

    -- Step 3: Future Data with Update Timestamp
    future_data AS (
        SELECT
            f.*,
            CASE WHEN day_of_week = 6 THEN ds ELSE NULL END AS update_day
        FROM {catalog_name}.{schema_name}.feature_set f
    ),

    -- Step 4: Assign the Modified Timestamp to Each Week's Dates (Backward Fill)
    future_with_update AS (
        SELECT
            f.* EXCEPT (update_day),
            -- Use FIRST_VALUE to fill the modified timestamp backwards for each week
            FIRST_VALUE(update_day, TRUE) OVER (
                PARTITION BY unique_id
                ORDER BY ds
                ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
            ) AS modified_timestamp_utc
        FROM future_data f
    )

    -- Step 5: Combine Both Datasets Without `modified_timestamp_utc`
    SELECT * EXCEPT (update_timestamp_utc) FROM test_set_with_update

    UNION ALL

    SELECT * EXCEPT (update_timestamp_utc) FROM future_with_update

    ORDER BY ds;
    """

    spark.sql(sql_query)

if __name__ == "__main__":
    main()
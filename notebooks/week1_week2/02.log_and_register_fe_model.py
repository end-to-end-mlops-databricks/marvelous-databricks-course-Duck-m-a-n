# Databricks notebook source
# MAGIC %pip install ../m5_forecasting-0.0.1-py3-none-any.whl

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import yaml
import mlflow
from mlflow.models import infer_signature

from datetime import datetime

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient

import lightgbm as lgb

from m5_forecasting.config import Config

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
config = Config.from_yaml("../../configs/project_config.yml")

# NEED TO ADD
num_features = config.processed_features.num_features
cat_features = config.processed_features.cat_features
target = config.target
#parameters = config.hyperparameters.__dict__

catalog_name = config.catalog_name
schema_name = config.schema_name

# Define the feature table schema
feature_table_name = f"{catalog_name}.{schema_name}.forecasting_features"

# Set the number of partitions based on cluster configuration
num_partitions = 8

# COMMAND ----------
train_set = (
    spark.table(f"{catalog_name}.{schema_name}.train_set")\
    .withColumn('unique_id', F.col('unique_id').cast('STRING'))
    .withColumn('ds', F.col('ds').cast('TIMESTAMP'))
    .withColumn('y', F.col('y').cast('INT'))
    .repartitionByRange(num_partitions, "unique_id")
    )

test_set = (
    spark.table(f"{catalog_name}.{schema_name}.test_set")\
    .withColumn('unique_id', F.col('unique_id').cast('STRING'))
    .withColumn('ds', F.col('ds').cast('TIMESTAMP'))
    .withColumn('y', F.col('y').cast('INT'))
    .repartitionByRange(num_partitions, "unique_id")
    )

sell_price_df = spark.table(f"{catalog_name}.{schema_name}.sell_prices") \
    .select(
        F.col("unique_id").cast("STRING"),
        F.col("ds").cast("TIMESTAMP"),
        F.col("sell_price").cast("DOUBLE")
    )

prod_info_df = spark.table(f"{catalog_name}.{schema_name}.prod_info") \
    .select(
        F.col("unique_id").cast("STRING"),
        F.col("state_id").cast("STRING"),
        F.col("store_id").cast("STRING"),
        F.col("cat_id").cast("STRING"),
        F.col("dept_id").cast("STRING"),
        F.col("item_id").cast("STRING")
    )

calendar_df = spark.table(f"{catalog_name}.{schema_name}.calendar") \
    .select(
        F.col("ds").cast("TIMESTAMP"),
        F.col("day_of_week").cast("INT"),
        F.col("is_weekend").cast("INT"),
        F.col("day_of_month").cast("INT"),
        F.col("week_of_month").cast("INT"),
        F.col("month").cast("INT"),
        F.col("week_num_year").cast("INT"),
        F.col("year").cast("INT"),
        F.col("num_events").cast("INT")
    )

dept_weekly_avg_sell_price_df = (
    sell_price_df
    .join(prod_info_df, on="unique_id", how="inner")
    .join(calendar_df, on="ds", how="inner")
    .groupBy("dept_id", "week_num_year", "year")
    .agg(F.avg("sell_price").alias("dept_wkly_avg_sell_price"))
    )

train_feature_df = (
    train_set
    .join(sell_price_df, on=["unique_id", "ds"], how ="inner")
    .join(prod_info_df, on="unique_id", how="inner")
    .join(calendar_df, on="ds", how="inner")
    .join(dept_weekly_avg_sell_price_df, on=["dept_id", "week_num_year", "year"], how="inner")
    .select(
            train_set.unique_id
            ,train_set.ds
            ,train_set.y
            ,sell_price_df.sell_price
            ,dept_weekly_avg_sell_price_df.dept_wkly_avg_sell_price
            ,prod_info_df.state_id
            ,prod_info_df.store_id
            ,prod_info_df.cat_id
            ,prod_info_df.dept_id
            ,prod_info_df.item_id
            ,calendar_df.day_of_week
            ,calendar_df.is_weekend
            ,calendar_df.day_of_month
            ,calendar_df.week_of_month
            ,calendar_df.month
            ,calendar_df.week_num_year
            ,calendar_df.year
            ,calendar_df.num_events
            ,train_set.update_timestamp_utc
            )
    .repartitionByRange(num_partitions, "unique_id")
    )

test_feature_df = (
    test_set
    .join(sell_price_df, on=["unique_id", "ds"], how ="inner")
    .join(prod_info_df, on="unique_id", how="inner")
    .join(calendar_df, on="ds", how="inner")
    .join(dept_weekly_avg_sell_price_df, on=["dept_id", "week_num_year", "year"], how="inner")
    .select(
            test_set.unique_id
            ,test_set.ds
            ,test_set.y
            ,sell_price_df.sell_price
            ,dept_weekly_avg_sell_price_df.dept_wkly_avg_sell_price
            ,prod_info_df.state_id
            ,prod_info_df.store_id
            ,prod_info_df.cat_id
            ,prod_info_df.dept_id
            ,prod_info_df.item_id
            ,calendar_df.day_of_week
            ,calendar_df.is_weekend
            ,calendar_df.day_of_month
            ,calendar_df.week_of_month
            ,calendar_df.month
            ,calendar_df.week_num_year
            ,calendar_df.year
            ,calendar_df.num_events
            ,test_set.update_timestamp_utc
            )
    .repartitionByRange(num_partitions, "unique_id")
    )

# COMMAND ----------
from mlforecast.distributed import DistributedMLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean

fcst = DistributedMLForecast(
    models = [],
    freq='D',   
    lags=[1, 7],
    lag_transforms={
        1: [ExpandingMean()],
        7: [ExpandingMean(), RollingMean(window_size = 7), RollingMean(window_size = 14)]
    },
    num_threads = 8
)

# COMMAND ----------
# Filter train_set to include only the two specific unique_ids
filtered_train_set = train_feature_df.filter(train_set["unique_id"].isin(["HOBBIES_1_001_CA_1", "FOODS_3_595_CA_1"]))

display(filtered_train_set)
# COMMAND ----------
# Run the preprocess function on the filtered train set
prep = fcst.preprocess(train_feature_df, static_features = []).drop(target)

# COMMAND ----------
# Display the preprocessed result
display(prep)

# COMMAND ----------
# Create or replace Feature Table
spark.sql(f"""
CREATE OR REPLACE TABLE {feature_table_name} (
    unique_id STRING NOT NULL,
    ds TIMESTAMP NOT NULL,
    sell_price DOUBLE,
    dept_wkly_avg_sell_price DOUBLE,
    state_id STRING,
    store_id STRING,
    cat_id STRING,
    dept_id STRING,
    item_id STRING,
    day_of_week INT,
    is_weekend INT,
    day_of_month INT,
    week_of_month INT,
    month INT,
    week_num_year INT,
    year INT,
    num_events INT,
    update_timestamp_utc TIMESTAMP,
    lag1 DOUBLE,
    lag7 DOUBLE,
    expanding_mean_lag1 DOUBLE,
    expanding_mean_lag7 DOUBLE,
    rolling_mean_lag7_window_size7 DOUBLE,
    rolling_mean_lag7_window_size14 DOUBLE
);
""")

# Add primary key constraint
spark.sql(f"""
    ALTER TABLE {feature_table_name}
    ADD CONSTRAINT unique_id_date_pk PRIMARY KEY(unique_id, ds TIMESERIES)
""")

# Enable change data feed
spark.sql(f"""
    ALTER TABLE {feature_table_name}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = True)
""")

prep.write.mode("overwrite").insertInto(feature_table_name).repartitionByRange(num_partitions, "unique_id")




# COMMAND ----------
# Define function to add training set creation timestamp UTC
spark.sql(f"""
CREATE OR REPLACE FUNCTION {function_name}()
RETURNS DOUBLE
LANGUAGE PYTHON
AS
$$
    from datetime import timezone 
    import datetime 
    
    
    # Getting the current date 
    # and time 
    dt = datetime.datetime.now(timezone.utc) 
    
    utc_time = dt.replace(tzinfo=timezone.utc) 
    
    return utc_time
$$
""")

# COMMAND ----------
"""
unique_id:string
ds:timestamp
y:integer
sell_price:double
dept_wkly_avg_sell_price:double
state_id:string
store_id:string
cat_id:string
dept_id:string
item_id:string
day_of_week:integer
is_weekend:integer
day_of_month:integer
week_of_month:integer
month:integer
week_num_year:integer
year:integer
num_events:integer
update_timestamp_utc:timestamp
lag1:double
lag7:double
expanding_mean_lag1:double
expanding_mean_lag7:double
"""


# COMMAND ----------
# Define a training set with the calculated average sell price per department
training_set = fe.create_training_set(
    df=train_set,
    label=target,
    feature_lookups=[
        FeatureLookup(
            table_name=sell_price_feature_table,
            feature_names=["sell_price"],
            lookup_key=["unique_id", "ds"]
        ),
        FeatureLookup(
            table_name=calendar_feature_table,
            feature_names=["num_events", "is_weekend", "day_of_week", "day_of_month", "week_num_year", "week_of_month", "month", "year"],
            lookup_key="ds"
        ),
        FeatureLookup(
            table_name=prod_info_feature_table,
            feature_names=["item_id", "dept_id", "cat_id", "store_id", "state_id"],
            lookup_key="unique_id"
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="weekly_dept_avg_sell_price",
            input_bindings={
                "dept_id": "dept_id",
                "week_num_year": "week_num_year",
                "year": "year"
            },
        ),
    ],
    exclude_columns=["update_timestamp_utc"]
)

# COMMAND ----------
# Repeat similar logic for the testing set
testing_set = fe.create_training_set(
    df=test_set,
    label=target,
    feature_lookups=[
        FeatureLookup(
            table_name=sell_price_feature_table,
            feature_names=["sell_price"],
            lookup_key=["unique_id", "ds"]
        ),
        FeatureLookup(
            table_name=calendar_feature_table,
            feature_names=["num_events", "is_weekend", "day_of_week", "day_of_month", "week_num_year", "week_of_month", "month", "year"],
            lookup_key="ds"
        ),
        FeatureLookup(
            table_name=prod_info_feature_table,
            feature_names=["item_id", "dept_id", "cat_id", "store_id", "state_id"],
            lookup_key="unique_id"
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="weekly_dept_avg_sell_price",
            input_bindings={
                "dept_id": "dept_id",
                "week_num_year": "week_num_year",
                "year": "year"
            },
        ),
    ],
    exclude_columns=["update_timestamp_utc"]
)

# COMMAND ----------
# Filter the Spark DataFrames for `unique_id` values before calling `.toPandas()`
filtered_training_df = training_set.load_df().filter(F.col("unique_id").isin(["FOODS_3_595_CA_1", "HOBBIES_1_001_CA_1"]))
filtered_testing_df = testing_set.load_df().filter(F.col("unique_id").isin(["FOODS_3_595_CA_1", "HOBBIES_1_001_CA_1"]))

# Convert the filtered Spark DataFrames to pandas
training_df = filtered_training_df.toPandas()
testing_df = filtered_testing_df.toPandas()

# COMMAND ----------
#training_df = training_set.load_df().toPandas()
#testing_df = testing_set.load_df().toPandas()

# COMMAND ----------
X_train = training_df[num_features + cat_features + ["weekly_dept_avg_sell_price"]]
y_train = training_df[target]

X_test = testing_df[num_features + cat_features + ["weekly_dept_avg_sell_price"]]
y_test = testing_df[target]

# COMMAND ----------
# Display filtered X_train and y_train
display(X_train)
display(y_train)

# Display filtered X_test and y_test
display(X_test)
display(y_test)

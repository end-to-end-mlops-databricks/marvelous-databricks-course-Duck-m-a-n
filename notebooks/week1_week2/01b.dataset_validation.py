# # Databricks notebook source
# MAGIC %pip install m5_forecasting-0.0.1-py3-none-any.whl

# COMMAND ----------
import yaml

from pyspark.sql import functions as F
from pyspark.sql.functions import size, countDistinct, min, max, datediff, col
from pyspark.sql import SparkSession

from m5_forecasting.config import Config

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
config = Config.from_yaml("../../configs/project_config.yml")

catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")
prod_info = spark.table(f"{catalog_name}.{schema_name}.prod_info")
sell_price = spark.table(f"{catalog_name}.{schema_name}.sell_price")

# COMMAND ----------
# Get unique_ids from train_set
train_unique_ids_df = train_set.select("unique_id").distinct()
train_unique_ids = set(train_unique_ids_df.rdd.flatMap(lambda x: x).collect())

# Get unique_ids from test_set
test_unique_ids_df = test_set.select("unique_id").distinct()
test_unique_ids = set(test_unique_ids_df.rdd.flatMap(lambda x: x).collect())

# Get unique_ids from sell_price
sell_price_unique_ids_df = sell_price.select("unique_id").distinct()
sell_price_unique_ids = set(sell_price_unique_ids_df.rdd.flatMap(lambda x: x).collect())

# Get unique_ids from prod_info
prod_info_unique_ids_df = prod_info.select("unique_id").distinct()
prod_info_unique_ids = set(prod_info_unique_ids_df.rdd.flatMap(lambda x: x).collect())

if train_unique_ids == test_unique_ids:
    print(f"Both train and test sets have the same unique_ids: {len(train_unique_ids)}")
else:
    print("Train and test sets have different unique_ids.")
    print(f"Train unique_ids: {len(train_unique_ids)}, Test unique_ids: {len(test_unique_ids)}")

train_test_unique_ids = train_unique_ids.union(test_unique_ids)

missing_in_sell_price = train_test_unique_ids - sell_price_unique_ids
if not missing_in_sell_price:
    print("All unique_ids in train and test sets are present in sell_price.")
else:
    print(f"Unique_ids in train/test sets but missing in sell_price: {missing_in_sell_price}")

missing_in_prod_info = train_test_unique_ids - prod_info_unique_ids
if not missing_in_prod_info:
    print("All unique_ids in train and test sets are present in prod_info.")
else:
    print(f"Unique_ids in train/test sets but missing in prod_info: {missing_in_prod_info}")

if not missing_in_sell_price and not missing_in_prod_info:
    print("\nAll unique_ids in train and test sets are covered in both sell_price and prod_info.")
else:
    print("\nSome unique_ids in train/test sets are missing from either sell_price or prod_info.")

# COMMAND ----------
# Compute the maximum ds for each unique_id in train_set
train_max_ds = train_set.groupBy("unique_id").agg(F.max("ds").alias("max_ds"))

# Get the distinct max_ds values
distinct_train_max_ds = train_max_ds.select("max_ds").distinct()

# Count the number of distinct max_ds values
distinct_train_max_ds_count = distinct_train_max_ds.count()

if distinct_train_max_ds_count == 1:
    max_ds_value = distinct_train_max_ds.collect()[0]["max_ds"]
    print(f"All unique_ids in train_set have the same maximum ds: {max_ds_value}")
else:
    print(f"Different maximum ds values found in train_set for different unique_ids.")
    print(f"Number of different max_ds values: {distinct_train_max_ds_count}")
    # Optionally display the different max_ds values
    # distinct_train_max_ds.show()

# Compute the minimum ds for each unique_id in test_set
test_min_ds = test_set.groupBy("unique_id").agg(F.min("ds").alias("min_ds"))

# Get the distinct min_ds values
distinct_test_min_ds = test_min_ds.select("min_ds").distinct()

# Count the number of distinct min_ds values
distinct_test_min_ds_count = distinct_test_min_ds.count()

if distinct_test_min_ds_count == 1:
    min_ds_value = distinct_test_min_ds.collect()[0]["min_ds"]
    print(f"All unique_ids in test_set have the same minimum ds: {min_ds_value}")
else:
    print(f"Different minimum ds values found in test_set for different unique_ids.")
    print(f"Number of different min_ds values: {distinct_test_min_ds_count}")
    # Optionally display the different min_ds values
    # distinct_test_min_ds.show()

# COMMAND ----------
# Number of unique stores
num_stores = prod_info.select('store_id').distinct().count()
print(f"Number of unique stores: {num_stores}")

# Step 1: State IDs and their store_id counts
state_counts = (
    prod_info.groupBy('state_id')
    .agg(F.countDistinct('store_id').alias('store_count'))
)
print("\nState IDs and store count per state:")
state_counts.show()

# Capture state insights dynamically
state_counts_list = state_counts.collect()
state_insight = ", ".join([f"{row['state_id']} has {row['store_count']} stores" for row in state_counts_list])
print(f"\nInsight: {state_insight}.")

# Step 2: Distribution of state_ids per store
state_distribution = (
    prod_info.groupBy('store_id')
    .agg(F.collect_set('state_id').alias('state_ids'))
)
print("\nState distribution per store:")
state_distribution.show(truncate=False)

# Check if each store has only one state
state_distribution = state_distribution.withColumn('state_count', size('state_ids'))

stores_with_multiple_states = state_distribution.filter(F.col('state_count') > 1)
num_stores_with_multiple_states = stores_with_multiple_states.count()

if num_stores_with_multiple_states == 0:
    print("\nInsight: Each store is associated with only one state.")
else:
    print("\nInsight: Some stores are associated with multiple states.")

# Step 3: Count of categories per store
category_counts = (
    prod_info.groupBy('store_id')
    .agg(F.countDistinct('cat_id').alias('num_categories'))
)
print("\nNumber of categories per store:")
category_counts.show()

# Capture unique categories per store dynamically
unique_category_counts = category_counts.select('num_categories').distinct().collect()
unique_category_counts_values = [row['num_categories'] for row in unique_category_counts]

if len(unique_category_counts_values) == 1:
    num_categories = unique_category_counts_values[0]
    categories = prod_info.select('cat_id').distinct().collect()
    categories_list = [row['cat_id'] for row in categories]
    print(f"\nInsight: All stores have exactly {num_categories} categories: {', '.join(categories_list)}.")
else:
    print("\nInsight: Stores have varying numbers of categories, suggesting different assortments.")

# Step 4: Number of departments per store and department IDs
store_dept_counts = (
    prod_info.groupBy('store_id')
    .agg(F.countDistinct('dept_id').alias('num_depts'))
)
print("\nNumber of departments per store:")
store_dept_counts.show()

# Capture department insights dynamically
unique_dept_counts = store_dept_counts.select('num_depts').distinct().collect()
unique_dept_counts_values = [row['num_depts'] for row in unique_dept_counts]

dept_ids = prod_info.select('dept_id').distinct().collect()
dept_ids_list = [row['dept_id'] for row in dept_ids]

if len(unique_dept_counts_values) == 1:
    num_depts = unique_dept_counts_values[0]
    print(f"\nInsight: All stores have exactly {num_depts} departments.")
    print(f"Departments are: {', '.join(dept_ids_list)}.")
else:
    print("\nInsight: Stores have different numbers of departments.")
    print(f"Departments across all stores include: {', '.join(dept_ids_list)}.")

# Step 5: Number of departments and unique_ids within each category per store
dept_and_id_per_cat_per_store = (
    prod_info.groupBy('store_id', 'cat_id')
    .agg(
        F.countDistinct('dept_id').alias('num_depts_per_category'),
        F.countDistinct('unique_id').alias('unique_id_count')
    )
)
print("\nNumber of departments and unique IDs within each category per store:")
dept_and_id_per_cat_per_store.show()

# Capture category-department-product structure insights dynamically
category_structure = (
    dept_and_id_per_cat_per_store.groupBy('cat_id')
    .agg(
        F.collect_set('num_depts_per_category').alias('num_depts_per_category_set'),
        F.collect_set('unique_id_count').alias('unique_id_count_set')
    )
)

print("\nInsight: Checking if each category has a consistent department structure across stores.")

category_structure = category_structure.withColumn(
    'num_depts_consistent', F.when(size('num_depts_per_category_set') == 1, True).otherwise(False)
).withColumn(
    'unique_ids_consistent', F.when(size('unique_id_count_set') == 1, True).otherwise(False)
)

# Display insights
category_structure_list = category_structure.collect()
for row in category_structure_list:
    cat_id = row['cat_id']
    num_depts_set = row['num_depts_per_category_set']
    unique_id_set = row['unique_id_count_set']
    num_depts_consistent = row['num_depts_consistent']
    unique_ids_consistent = row['unique_ids_consistent']
    
    if num_depts_consistent and unique_ids_consistent:
        num_depts = num_depts_set[0]
        unique_ids = unique_id_set[0]
        print(f" - {cat_id} has {num_depts} departments and {unique_ids} unique products consistently across stores.")
    else:
        print(f" - {cat_id} has varying department counts {num_depts_set} and unique product counts {unique_id_set} across stores.")


# COMMAND ----------
# Step 1: Count distinct unique IDs
distinct_unique_ids = train_set.select('unique_id').distinct().count()
print(f"Number of distinct unique IDs: {distinct_unique_ids}")

# Step 2: Calculate the length of each unique_id's time series
unique_id_lengths = train_set.groupBy('unique_id').agg(countDistinct('ds').alias('ds_length'))

# Step 3: Check if all unique_ids have the same ds length
length_counts = unique_id_lengths.groupBy('ds_length').count().orderBy('ds_length')
length_counts.show()

num_unique_lengths = length_counts.count()

if num_unique_lengths == 1:
    ds_length_value = length_counts.select('ds_length').first()['ds_length']
    print(f"All unique_ids have the same ds length: {ds_length_value}")
else:
    print("There are different ds lengths among unique_ids.")
    print("Distribution of ds lengths:")
    length_counts.show()

# Get min and max ds per unique_id
unique_id_stats = unique_id_lengths.join(
    train_set.groupBy('unique_id').agg(
        min('ds').alias('min_ds'),
        max('ds').alias('max_ds')
    ),
    on='unique_id',
    how='left'
)

# Step 4: Find shortest and largest lengths and corresponding unique_ids

# Shortest length
shortest_length = unique_id_stats.agg(min('ds_length')).collect()[0][0]
shortest_ids = unique_id_stats.filter(col('ds_length') == shortest_length)

print(f"Shortest length of a time series: {shortest_length}")
print("Unique IDs with the shortest length and their min and max dates:")
shortest_ids.select('unique_id', 'ds_length', 'min_ds', 'max_ds').show(truncate=False)

# Largest length
largest_length = unique_id_stats.agg(max('ds_length')).collect()[0][0]
largest_ids = unique_id_stats.filter(col('ds_length') == largest_length)

print(f"Largest length of a time series: {largest_length}")
print("Unique IDs with the largest length and their min and max dates:")
largest_ids.select('unique_id', 'ds_length', 'min_ds', 'max_ds').show(truncate=False)

# Step 5: Check for gaps in the time series for each unique_id

# Calculate total_days between min_ds and max_ds
unique_id_stats = unique_id_stats.withColumn('total_days', datediff('max_ds', 'min_ds') + 1)

# Check for gaps by comparing total_days and ds_length
unique_id_stats = unique_id_stats.withColumn('has_no_gaps', col('total_days') == col('ds_length'))

# Count unique_ids with gaps
gaps_count = unique_id_stats.filter(~col('has_no_gaps')).count()

if gaps_count == 0:
    print("All unique_ids have continuous time series with no gaps.")
else:
    print(f"There are {gaps_count} unique_ids with gaps in their time series.")
    print("Unique IDs with gaps and their details:")
    unique_id_stats.filter(~col('has_no_gaps')).select('unique_id', 'ds_length', 'total_days', 'min_ds', 'max_ds').show(truncate=False)

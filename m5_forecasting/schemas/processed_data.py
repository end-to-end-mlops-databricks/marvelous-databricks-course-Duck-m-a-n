import pandera.pyspark as pa
import pyspark.sql.types as T
from pandera.pyspark import DataFrameModel

class SalesDataSchema(DataFrameModel):
    unique_id: T.StringType() = pa.Field()
    ds: T.TimestampType() = pa.Field()
    y: T.IntegerType() = pa.Field(ge=0)  # Ensures non-negative sales values

    class Config:
        coerce = True  # Coerce columns to defined dtypes

class CalendarSchema(DataFrameModel):
    ds: T.TimestampType() = pa.Field()
    day_of_week: T.IntegerType() = pa.Field()
    is_weekend: T.IntegerType() = pa.Field()
    day_of_month: T.IntegerType() = pa.Field()
    week_of_month: T.IntegerType() = pa.Field()
    month: T.IntegerType() = pa.Field()
    week_num_year: T.IntegerType() = pa.Field()
    year: T.IntegerType() = pa.Field()
    num_events: T.IntegerType() = pa.Field()

    class Config:
        coerce = True

class SellPriceSchema(DataFrameModel):
    unique_id: T.StringType() = pa.Field()
    ds: T.TimestampType() = pa.Field()
    sell_price: T.DoubleType() = pa.Field(ge=0)
    dept_wkly_avg_sell_price: T.DoubleType() = pa.Field(ge=0)

    class Config:
        coerce = True

class ProductInfoSchema(DataFrameModel):
    unique_id: T.StringType() = pa.Field()
    item_id: T.StringType() = pa.Field()
    dept_id: T.StringType() = pa.Field()
    cat_id: T.StringType() = pa.Field()
    store_id: T.StringType() = pa.Field()
    state_id: T.StringType() = pa.Field()

    class Config:
        coerce = True

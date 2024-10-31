from pandera import DataFrameModel, Field
from pandera.typing import Series
from pandas import Timestamp
from pandera.dtypes import Float, Int64, String
from pandera.engines.pandas_engine import Date

class SalesDataSchema(DataFrameModel):
    unique_id: Series[String] = Field()
    ds: Series[Date] = Field()
    y: Series[Int64] = Field(ge=0)  # Ensures non-negative sales values

    class Config:
        coerce = True  # Coerce columns to defined dtypes

class CalendarSchema(DataFrameModel):
    ds: Series[Date] = Field()
    day_of_week: Series[Int64] = Field()
    is_weekend: Series[Int64] = Field()
    day_of_month: Series[Int64] = Field()
    week_of_month: Series[Int64] = Field()
    month: Series[Int64] = Field()
    week_num_year: Series[Int64] = Field()
    year: Series[Int64] = Field()
    num_events: Series[Int64] = Field()
    wm_yr_wk: Series[Int64] = Field()

    class Config:
        coerce = True

class SellPriceSchema(DataFrameModel):
    unique_id: Series[String] = Field()
    ds: Series[Date] = Field()
    sell_price: Series[Float] = Field(ge=0)

    class Config:
        coerce = True

class ProductInfoSchema(DataFrameModel):
    unique_id: Series[String] = Field()
    item_id: Series[String] = Field()
    dept_id: Series[String] = Field()
    cat_id: Series[String] = Field()
    store_id: Series[String] = Field()
    state_id: Series[String] = Field()

    class Config:
        coerce = True

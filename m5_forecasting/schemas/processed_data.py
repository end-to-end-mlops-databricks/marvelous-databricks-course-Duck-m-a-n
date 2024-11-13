from pandera import DataFrameModel, Field, Check
from pandera.typing import Series
from pandera.dtypes import Category, Float32, Int8, Int16, Int32, DateTime, Float64
import pandas as pd

class CombinedDataFrameSchema(DataFrameModel):
    """Schema for the combined DataFrame."""
    
    unique_id: Series[Category] = Field(nullable=False)
    ds: Series[DateTime] = Field(nullable=False)
    y: Series[Int32] = Field(ge=0, nullable=False)
    sell_price: Series[Float32] = Field(ge=0, nullable=True)
    temp: Series[Float32] = Field(nullable=True)
    conditions: Series[Category] = Field(nullable=True)
    num_events: Series[Int8] = Field(in_range={"min_value": 0, "max_value": 2}, nullable=False)
    item_id: Series[Category] = Field(nullable=False)
    dept_id: Series[Category] = Field(nullable=False)
    cat_id: Series[Category] = Field(nullable=False)
    store_id: Series[Category] = Field(nullable=False)
    state_id: Series[Category] = Field(nullable=False)
    day_of_week: Series[Int8] = Field(in_range={"min_value": 0, "max_value": 6}, nullable=False)
    is_weekend: Series[Int8] = Field(isin=[0, 1], nullable=False)
    day_of_month: Series[Int8] = Field(in_range={"min_value": 1, "max_value": 31}, nullable=False)
    week_of_month: Series[Int8] = Field(in_range={"min_value": 1, "max_value": 5}, nullable=False)
    month: Series[Int8] = Field(in_range={"min_value": 1, "max_value": 12}, nullable=False)
    year: Series[Int16] = Field(ge=1900, nullable=False)
    avg_weekly_temp: Series[Float32] = Field(nullable=True)
    avg_monthly_temp: Series[Float32] = Field(nullable=True)
    avg_28_day_temp: Series[Float32] = Field(nullable=True)
    percent_diff_weekly_temp: Series[Float32] = Field(nullable=True)
    percent_diff_monthly_temp: Series[Float32] = Field(nullable=True)
    percent_diff_28_day_avg_temp: Series[Float32] = Field(nullable=True)
    monthly_avg_sell_price: Series[Float32] = Field(nullable=True)
    percent_diff_monthly_sell_price: Series[Float32] = Field(nullable=True)
    dept_avg_sell_price: Series[Float32] = Field(nullable=True)
    cat_avg_sell_price: Series[Float32] = Field(nullable=True)
    store_dept_avg_sell_price: Series[Float32] = Field(nullable=True)
    state_dept_avg_sell_price: Series[Float32] = Field(nullable=True)

    class Config:
        coerce = True


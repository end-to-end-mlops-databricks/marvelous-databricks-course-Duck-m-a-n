from pydantic import BaseModel
from typing import List, Dict, Any
import yaml

class DatasetConfig(BaseModel):
    raw_sales_data: str
    raw_calendar_data: str
    raw_sell_price_data: str

class PathsConfig(BaseModel):
    # Databricks paths
    raw_sales_path: str
    raw_calendar_path: str
    raw_sell_prices_path: str
    
    # Local file paths
    local_sales_filepath: str
    local_calendar_filepath: str
    local_sell_prices_filepath: str

class Config(BaseModel):
    catalog_name: str
    schema_name: str
    horizon: int
    dataset: DatasetConfig
    paths: PathsConfig

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """
        Load config from yaml file.
        """
        with open(yaml_path, "r") as f:
            yaml_dict = yaml.safe_load(f)
        return cls(**yaml_dict)


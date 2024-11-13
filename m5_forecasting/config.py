from pydantic import BaseModel
from typing import List, Dict, Any
import yaml

class PathsConfig(BaseModel):
    # Databricks paths
    raw_sales_path: str
    raw_calendar_path: str
    raw_sell_prices_path: str
    raw_weather_path: str
    
    # Local file paths
    local_sales_filepath: str
    local_calendar_filepath: str
    local_sell_prices_filepath: str

class ProcessedFeatures(BaseModel):
    num_features: List[str]
    cat_features: List[str]
    static_features: List[str]
    date_features: List[str]
    engineered_features: List[str]

class InitParams(BaseModel):
    hyperparameters: Dict[str, Any]
    freq: str
    lags: List[int]
    lag_transforms: Dict[int, List[Dict[str, Any]]]
    num_threads: int

class ParametersConfig(BaseModel):
    init: InitParams

class Config(BaseModel):
    catalog_name: str
    schema_name: str
    horizon: int
    target: str
    processed_features: ProcessedFeatures
    paths: PathsConfig
    parameters: ParametersConfig

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """
        Load config from yaml file.
        """
        with open(yaml_path, "r") as f:
            yaml_dict = yaml.safe_load(f)
        return cls(**yaml_dict)

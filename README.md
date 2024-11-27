<h1 align="center">
m5 forecasting:  Parallelizing Group-Specific Forecasts using LightGBM


For Spark training:
- need to make sure "com.microsoft.azure:synapseml-lightgbm_2.12:1.0.8" is installed on the cluster to perform distributed training
    - https://github.com/microsoft/SynapseML?tab=readme-ov-file#setup-and-installation

- link to synapse.ml.lightgbm.LightGBMRegressor
    - https://mmlspark.blob.core.windows.net/docs/1.0.8/pyspark/synapse.ml.lightgbm.html#module-synapse.ml.lightgbm.LightGBMRegressor

def categorize_conditions(condition):
    if 'snow' in condition.lower():
        return 'snow'
    elif 'rain' in condition.lower():
        return 'rain'
    elif 'clear' in condition.lower():
        return 'clear'
    elif 'cloudy' in condition.lower() or 'overcast' in condition.lower():
        return 'cloudy'
    else:
        return condition

# prefect-deployments
A deployment space to use with Prefect in the [full-stack-on-prem-cv-mlops](https://github.com/jomariya23156/full-stack-on-prem-cv-mlops) repo

## How to use 
1. Make sure you have Prefect setup and working
2. Make sure you have Prefect variables specified in `prefect.yaml`. Defaults variable used here are `monitor_pool_name` and `current_model_metadata_file`.
3. Run `$ prefect deploy --name deployment-1 --name deployment-2`
or `$ prefect deploy --all`
# Phenotypic Profiling Model

## Data

Instructions for data download/processing can be found at: https://github.com/WayScience/mitocheck_data.

This repository compiles training data from a specific version of [MitoCheck_data](https://github.com/WayScience/mitocheck_data).
For more information see [0.download_data/README.md](0.download_data/README.md).

Formatted data (feature data + metadata) is saved in [training_data.csv.gz](1.format_data/data/training_data.csv.gz).

## Analysis

We anaylze the feature data with UMAP in [2.analyze_data](2.analyze_data).

## ML Model

We train, evaluate, and interpret a model to predict mitotic stage from nuclear staining data using [DeepProfiler](https://github.com/cytomining/DeepProfiler) features in [2.ML_model](2.ML_model).
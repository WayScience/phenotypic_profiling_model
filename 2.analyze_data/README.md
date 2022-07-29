# 2. Analyze Features

In this module, we present our pipeline for analyzing features.

### Feature Analysis

We use [UMAP](https://github.com/lmcinnes/umap) for analyis of features.
UMAP was introduced in [McInnes, L, Healy, J, 2018](https://arxiv.org/abs/1802.03426) as a manifold learning technique for dimension reduction.
We use UMAP to reduce the feature data from 1280 features to 1, 2, and 3 dimensions.

We use [Matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/) for data visualization.

**Note:** Phenotypic classes used for analysis can be changed with the `classes_to_keep` variable in [2.analyze_training_data.ipynb](2.analyze_training_data.ipynb).

## Step 1: Setup Feature Analysis Environment

### Step 1a: Create Feature Analysis Environment

```sh
# Run this command to create the conda environment for feature analysis
conda env create -f 2.analyze_data_env.yml
```

### Step 1b: Activate Feature Analysis Environment

```sh
# Run this command to activate the conda environment for feature analysis
conda activate 2.analyze_training_data
```

## Step 2: Execute Feature Analysis Pipeline

```bash
# Run this script to analyze features
bash 2.analyze_data.sh
```
**Note:** Running pipeline will produce all intermediate files (located in [results](results/)).
Analysis jupyter notebook ([2.analyze_training_data.ipynb](2.analyze_training_data.ipynb)) will not be updated but the executed notebook ([2.analyze_training_data.html](2.analyze_training_data.html)) will be updated.
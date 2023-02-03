# 5. Validate Model

In this module, we validate the final ML model.

### Validation Method 1

The final model from [2.train_model](../2.train_model/) is used to classify nuclei images from the [Cell Health Dataset](https://github.com/WayScience/cell-health-data).
The classification probabilities across CRISPR guide/cell line are then correlated to the Cell Health label for the the respective CRISPR perturbation/cell line.

The Cell Health dataset has cell painting images across 119 CRISPR guide perturbations (~2 per gene perturbation) and 3 cell lines.
More information regarding the generation of this dataset can be found at https://github.com/broadinstitute/cell-health.

In [Cell-Health-Data/4.classify-features](https://github.com/WayScience/cell-health-data/tree/master/4.classify-features), we use the final model to determine phenotypic class probabilities for each of the Cell Health cells.
These probabilities are averaged across CRISPR guide/cell line to create 357 *classifiction profiles* (119 CRISPR guides x 3 cell lines).

As part of [Predicting cell health phenotypes using image-based morphology profiling](https://www.molbiolcell.org/doi/10.1091/mbc.E20-12-0784), Way et al derived cell health indicators.
These indicators consist of 70 specific cell health phenotypes including proliferation, apoptosis, reactive oxygen species, DNA damage, and cell cycle stage.
These indicators are averaged across across CRISPR guide/cell line to create 357 [*Cell Health label profiles*](https://github.com/broadinstitute/cell-health/blob/master/1.generate-profiles/data/consensus/cell_health_median.tsv.gz).

We use [pandas.DataFrame.corr](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html) to find the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) between the *classifiction profiles* and the *Cell Health label profiles*.
The Pearson correlation coefficient measures the linear relationship between two datasets, with correlations of -1/+1 implying exact linear inverse/direct relationships respectively.

These correlations are interpreted with [seaborn.clustermap](https://seaborn.pydata.org/generated/seaborn.clustermap.html) to display the hierarchically-clustered correlation values.
Searborn clustermap groups similar correlations into clusters that are broadly similar to each other.

## Step 1: Validate Model

Use the commands below to validate the final ML model:

```sh
# Make sure you are located in 5.validate_model
cd 5.validate_model

# Activate phenotypic_profiling conda environment
conda activate phenotypic_profiling

# Interpret model
bash validate_model.sh
```
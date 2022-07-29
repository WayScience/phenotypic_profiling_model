# 3. Machine Learning Model

In this module, we train and validate a machine learning model for phenotypic classification (mitotic stage) of nuclei based on nuclei features.

We use [Scikit-learn (sklearn)](https://scikit-learn.org/) for data manipulation, model training, and model evaluation.
Scikit-learn is described in [Pedregosa et al., JMLR 12, pp. 2825-2830, 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html) as a machine learning library for Python.
Its ease of implementation in a pipeline make it ideal for our use case.

We consistently use the following parameters with many sklearn functions:

- `n_jobs=-1`: Use all CPU cores in parallel when completing a task.
- `random_state=0`: Use seed 0 when shuffling data or generating random numbers.
This allows "random" sklearn operations to have consist results.
We also use `np.random.seed(0)` to make "random" numpy operations have consistent results.

We use [seaborn](https://seaborn.pydata.org/) for data visualization. 
Seaborn is described in [Waskom, M.L., 2021](https://doi.org/10.21105/joss.03021) as a library for making statisical graphics in python.

All parts of the following pipeline are completed for a "final" model (from training data) and a "shuffled baseline" model (from shuffled training data).
This shuffled baseline model provides a suitable baseline comparison for the final model during evaluation.

### A. Data Preparation

First, we split the data into training, test, and holdout subsets in [0.split_data.ipynb](notebooks/0.split_data.ipynb).
The `get_representative_images()` function used to create the holdout dataset determines which images to holdout such that all phenotypic classes can be represented in these holdout images.
The test dataset is determined by taking a random number of samples (stratified by phenotypic class) from the dataset after the holdout images are removed.
The training dataset is the subset remaining after holdout/test samples are removed.
Sample indexes associated with training, test, and holdout subsets are stored in [0.data_split_indexes.tsv](results/0.data_split_indexes.tsv).
Sample indexes are used to load subsets from [training_data.csv.gz](../1.format_data/data/training_data.csv.gz).

We use [sklearn.utils.shuffle](https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html) to shuffle the training data in a consistent way.
This function shuffles the order of the training data samples while keeping the phenotypic class labels aligned with their respective features.
In other words, this function shuffles entire rows of training data to remove any ordering scheme.
This is necessary because the data as labeled from MitoCheck tends to have phenotypic classes in groups, which can introduce bias into the model.

We use [sklearn.model_selection.StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) to create stratified training/test data sets for cross validation.
The training and test data sets that are created have the same distribution of classes.
This ensures that each class is proportionally represented across all data sets.
We use `n_splits=10` to create 10 folds for cross-validation.

To create the shuffled baseline training dataset, we first load the training data as described above. 
We then shuffle each column of the training data independently, which removes any correlation between features and phenotypic class label.

### B. Model Training

We train the model in [1.train_model.ipynb](notebooks/1.train_model.ipynb).
We use [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) for our machine learning model.
We use the following parameters for our Logisic Regression model:

- `penalty='elasticnet'`: Use elasticnet as the penalty for our model.
Elastic-Net regularization is a combination of L1 and L2 regularization methods.
The mixing of these two methods is determined by the `l1_ratio` parameter which we optimize later.
- `solver='saga'`: We use the saga solver as this is the only solver that supports Elastic-Net regularization.
- `max_iter=100`: Set the maximum number of iterations for solver to converge. 100 iterations allows the solver to maximize performance without completing unnecessary iterations.

We use [sklearn.model_selection.GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) to perform an exhaustive search for the parameters below. This searches for parameters that maximize the weighted F1 score of the model. We optimize weighted F1 score because this metric measures model precision and recall aand accounts for label imbalance (see [sklearn.metrics.f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) for more details).

- `l1_ratio`: Elastic-Net mixing parameter.
Used to combine L1 and L2 regularization methods.
We search over the following parameters: `[0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]`
- `C`: Inversely proportional to regularization strength.
We search over the following parameters: `[1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03]`

The best parameters are used to train a final model on all of the training data.
This final model is saved in [1.log_reg_model.joblib](results/1.log_reg_model.joblib).
The model derived from shuffled training data is saved in [1.shuffled_baseline_log_reg_model.joblib](results/1.shuffled_baseline_log_reg_model.joblib).

### C. Model Evaluation

We train the model in [2.evaluate_model.ipynb](notebooks/2.evaluate_model.ipynb).
We use the final model and shuffled baseline model to predict the labels of the training, testing, and holdout datasets.
These predictions are saved in [results/2.model_predictions.tsv](results/2.model_predictions.tsv) and [2.shuffled_baseline_model_predictions.tsv](results/2.shuffled_baseline_model_predictions.tsv) respectively.

We evaluate these 6 sets of predictions with a confusion matrix to see the true/false postives and negatives (see [sklearn.metrics.confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) for more details).

We also evaluate these 6 sets of predictions with [sklearn.metrics.f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) to determine the final/shuffled baseline model's predictive performance on each subset.
F1 score measures the models precision and recall performance for each phenotypic class.

### D. Model Interpretation

We train the model in [3.interpret_model.ipynb](notebooks/3.interpret_model.ipynb).
The final model and shuffled baseline model coefficients are loaded from [1.log_reg_model.joblib](results/1.log_reg_model.joblib) and [1.shuffled_baseline_log_reg_model.joblib](results/1.shuffled_baseline_log_reg_model.joblib) respectively.
These coefficients are interpreted with the following diagrams:

- We use [seaborn.heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html) to display the coefficient values for each phenotypic class/feature.
- We use [seaborn.clustermap](https://seaborn.pydata.org/generated/seaborn.clustermap.html) to display a hierarchically-clustered heatmap of coefficient values for each phenotypic class/feature
- We use [seaborn.kedeplot](https://seaborn.pydata.org/generated/seaborn.kdeplot.html) to display a density plot of coeffiecient values for each phenotypic class.
- We use [seaborn.barplot](https://seaborn.pydata.org/generated/seaborn.barplot.html) to display a bar plot of average coeffiecient values per phenotypic class and feature.

## Step 1: Setup Machine Learning Environment

### Step 1a: Create Machine Learning Environment

```sh
# Run this command to create the conda environment for machine learning
conda env create -f 3.machine_learning_env.yml
```

### Step 1b: Activate Machine Learning Environment

```sh
# Run this command to activate the conda environment for machine learning
conda activate 3.ML_phenotypic_classification
```

## Step 2: Execute Machine Learning Pipeline

```bash
# Run this script to train, evaluate, and interpret DP model
bash 3.ML_model.sh
```

**Note:** Running pipeline will produce all intermediate files (located in [results](results/)).
Jupyter notebooks (located in [notebooks](notebooks/)) will not be updated but the executed notebooks (located in [html](html/)) will be updated.
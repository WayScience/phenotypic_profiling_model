#!/usr/bin/env python
# coding: utf-8

# ### Import libraries

# In[1]:


import pandas as pd
import numpy as np
import pathlib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.utils import shuffle
from joblib import dump

import sys
# adding utils to system path
sys.path.insert(0, '../utils')
from MlPipelineUtils import get_features_data, get_dataset, get_X_y_data


# ### Load training data and create stratified folds for cross validation

# In[2]:


# set numpy seed to make random operations reproduceable
np.random.seed(0)

results_dir = pathlib.Path("../results/")

# load training data from indexes and features dataframe
data_split_path = pathlib.Path(f"{results_dir}/0.data_split_indexes.tsv")
features_dataframe_path = pathlib.Path("../../1.format_data/data/training_data.csv.gz")

features_dataframe = get_features_data(features_dataframe_path)
data_split_indexes = pd.read_csv(data_split_path, sep="\t", index_col=0)

training_data = get_dataset(features_dataframe, data_split_indexes, "train")
training_data


# In[3]:


X, y = get_X_y_data(training_data)

print(X.shape)
print(y.shape)

# create stratified data sets for k-fold cross validation
straified_k_folds = StratifiedKFold(n_splits=10, shuffle=False)


# ### Define model without C/l1_ratio parameters
# 

# In[4]:


# create logistic regression model with following parameters
log_reg_model = LogisticRegression(
    penalty="elasticnet", solver="saga", max_iter=100, n_jobs=-1, random_state=0
)


# ### Perform grid search for best C and l1_ratio parameters

# In[5]:


# hypertune parameters with GridSearchCV
parameters = {"C": np.logspace(-3, 3, 7), "l1_ratio": np.linspace(0, 1, 11)}
#parameters = {"C": [0.1], "l1_ratio": [0.0]}
print(f"Parameters being tested: {parameters}")
grid_search_cv = GridSearchCV(
    log_reg_model, parameters, cv=straified_k_folds, n_jobs=-1, scoring="f1_weighted",
)
grid_search_cv = grid_search_cv.fit(X, y)


# In[6]:


print(f"Best parameters: {grid_search_cv.best_params_}")
print(f"Score of best estimator: {grid_search_cv.best_score_}")


# ### Save best model

# In[7]:


# save final estimator
dump(grid_search_cv.best_estimator_, f"{results_dir}/1.log_reg_model.joblib")


# ## Repeat process with shuffling to create shuffled baseline model

# In[8]:


X, y = get_X_y_data(training_data)

print(X.shape)
print(y.shape)

# shuffle columns of X (features) dataframe independently to create shuffled baseline
for column in X.T:
    np.random.shuffle(column)

# create stratified data sets for k-fold cross validation
straified_k_folds = StratifiedKFold(n_splits=10, shuffle=False)


# In[9]:


# create logistic regression model with following parameters
log_reg_model = LogisticRegression(
    penalty="elasticnet", solver="saga", max_iter=100, n_jobs=-1, random_state=0
)


# In[10]:


# hypertune parameters with GridSearchCV
parameters = {"C": np.logspace(-3, 3, 7), "l1_ratio": np.linspace(0, 1, 11)}
#parameters = {"C": [1.0], "l1_ratio": [0.8]}
print(f"Parameters being tested: {parameters}")
grid_search_cv = GridSearchCV(
    log_reg_model, parameters, cv=straified_k_folds, n_jobs=-1, scoring="f1_weighted",
)
grid_search_cv = grid_search_cv.fit(X, y)


# In[11]:


print(f"Best parameters: {grid_search_cv.best_params_}")
print(f"Score of best estimator: {grid_search_cv.best_score_}")


# In[12]:


# save final estimator
dump(grid_search_cv.best_estimator_, f"{results_dir}/1.shuffled_baseline_log_reg_model.joblib")


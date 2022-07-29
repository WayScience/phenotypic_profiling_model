#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import pathlib

from joblib import load

import sys
# adding utils to system path
sys.path.insert(0, '../utils')
from MlPipelineUtils import (
    get_features_data,
    get_dataset,
    get_X_y_data,
    evaluate_model_cm,
    evaluate_model_score
)

from sklearn.metrics import f1_score


# ### Evaluate best model

# In[2]:


# set numpy seed to make random operations reproduceable
np.random.seed(0)

results_dir = pathlib.Path("../results/")

log_reg_model_path = pathlib.Path(f"{results_dir}/1.log_reg_model.joblib")
log_reg_model = load(log_reg_model_path)

# load features data from indexes and features dataframe
data_split_path = pathlib.Path(f"{results_dir}/0.data_split_indexes.tsv")
data_split_indexes = pd.read_csv(data_split_path, sep="\t", index_col=0)
features_dataframe_path = pathlib.Path("../../1.format_data/data/training_data.csv.gz")
features_dataframe = get_features_data(features_dataframe_path)


# ### Evaluate with training data

# In[3]:


training_data = get_dataset(features_dataframe, data_split_indexes, "train")
training_data


# In[4]:


y_train, y_train_pred = evaluate_model_cm(log_reg_model, training_data)


# In[5]:


evaluate_model_score(log_reg_model, training_data)


# ### Evaluate with testing data

# In[6]:


testing_data = get_dataset(features_dataframe, data_split_indexes, "test")
testing_data


# In[7]:


y_test, y_test_pred = evaluate_model_cm(log_reg_model, testing_data)


# In[8]:


evaluate_model_score(log_reg_model, testing_data)


# ### Evaluate with holdout data

# In[9]:


holdout_data = get_dataset(features_dataframe, data_split_indexes, "holdout")
X_holdout, y_holdout = get_X_y_data(holdout_data)
holdout_data


# In[10]:


y_holdout, y_holdout_pred = evaluate_model_cm(log_reg_model, holdout_data)


# In[11]:


evaluate_model_score(log_reg_model, holdout_data)


# ### Save trained model predicitions

# In[12]:


predictions = []

predictions.append(y_train)
predictions.append(y_train_pred)

predictions.append(y_test)
predictions.append(y_test_pred)

predictions.append(y_holdout)
predictions.append(y_holdout_pred)

predictions = pd.DataFrame(predictions)
predictions.index = ["y_train", "y_train_pred", "y_test", "y_test_pred", "y_holdout", "y_holdout_pred"]
predictions.to_csv(f"{results_dir}/2.model_predictions.tsv", sep="\t")


# ### Evaluate shuffled baseline model

# In[13]:


shuffled_baseline_log_reg_model_path = pathlib.Path(f"{results_dir}/1.shuffled_baseline_log_reg_model.joblib")
shuffled_baseline_log_reg_model = load(shuffled_baseline_log_reg_model_path) 


# ### Evaluate with training data

# In[14]:


y_train, y_train_pred = evaluate_model_cm(shuffled_baseline_log_reg_model, training_data)


# In[15]:


evaluate_model_score(shuffled_baseline_log_reg_model, training_data)


# ### Evaluate with testing data

# In[16]:


y_test, y_test_pred = evaluate_model_cm(shuffled_baseline_log_reg_model, testing_data)


# In[17]:


evaluate_model_score(shuffled_baseline_log_reg_model, testing_data)


# ### Evaluate with holdout data

# In[18]:


y_holdout, y_holdout_pred = evaluate_model_cm(shuffled_baseline_log_reg_model, holdout_data)


# In[19]:


evaluate_model_score(shuffled_baseline_log_reg_model, holdout_data)


# ### Save trained model predicitions

# In[20]:


predictions = []

predictions.append(y_train)
predictions.append(y_train_pred)

predictions.append(y_test)
predictions.append(y_test_pred)

predictions.append(y_holdout)
predictions.append(y_holdout_pred)

predictions = pd.DataFrame(predictions)
predictions.index = ["y_train", "y_train_pred", "y_test", "y_test_pred", "y_holdout", "y_holdout_pred"]
predictions.to_csv(f"{results_dir}/2.shuffled_baseline_model_predictions.tsv", sep="\t")


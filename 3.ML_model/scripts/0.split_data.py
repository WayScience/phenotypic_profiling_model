#!/usr/bin/env python
# coding: utf-8

# # Split feature data
# ## Create tsv file with indexes for held out data, training data, and testing data
# ### Import libraries

# In[1]:


import pandas as pd
import numpy as np
import pathlib
from typing import Tuple, Any, List, Union

from sklearn.utils import shuffle

import sys
# adding utils to system path
sys.path.insert(0, '../utils')
from MlPipelineUtils import get_features_data, get_random_images_indexes, get_representative_images, get_image_indexes


# ### Load data and set holdout/test parameters

# In[2]:


# set numpy seed to make random operations reproduceable
np.random.seed(0)

# load x (features) and y (labels) dataframes
load_path = pathlib.Path("../../1.format_data/data/training_data.csv.gz")
training_data = get_features_data(load_path)
print(training_data.shape)

# number of images to holdout
num_holdout_images = 5
# ratio of data to be reserved for testing (ex 0.15 = 15%)
test_ratio = 0.15


# In[3]:


# remove holdout indexes
images = get_representative_images(training_data, num_holdout_images, 10000)
holdout_image_indexes = get_image_indexes(training_data, images)
training_data = training_data.drop(pd.Index(data=holdout_image_indexes))
print(training_data.shape)


# In[4]:


# remove test indexes
# test_data is pandas dataframe with test split, stratified by Mitocheck_Phenotypic_Class
test_data = training_data.groupby("Mitocheck_Phenotypic_Class", group_keys=False).apply(
    lambda x: x.sample(frac=test_ratio)
)
test_indexes = test_data.index
training_data = training_data.drop(pd.Index(data=test_indexes))

train_indexes = np.array(training_data.index)
print(training_data.shape)


# In[5]:


# create pandas dataframe with all indexes and their respective labels
index_data = []
for index in holdout_image_indexes:
    index_data.append({"label": "holdout", "index": index})
for index in test_indexes:
    index_data.append({"label": "test", "index": index})
for index in train_indexes:
    index_data.append({"label": "train", "index": index})
index_data = pd.DataFrame(index_data)
index_data


# In[6]:


# make results dir for saving
results_dir = pathlib.Path("../results/")
results_dir.mkdir(parents=True, exist_ok=True)
# save indexes as tsv file
index_data.to_csv(f"{results_dir}/0.data_split_indexes.tsv", sep="\t")


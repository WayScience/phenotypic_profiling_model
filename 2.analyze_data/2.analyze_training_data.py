#!/usr/bin/env python
# coding: utf-8

# # Analyze all feature data
# 
# ### Import libraries

# In[1]:


import numpy as np
import pathlib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, rgb2hex
from pylab import cm
import seaborn as sns
import pandas as pd
import umap

from utils.analysisUtils import get_features_data, show_1D_umap, show_2D_umap, show_3D_umap 


# ### Initalize analysis

# In[2]:


# make random numpy operations consistent
np.random.seed(0)

# create results dir for saving results
results_dir = pathlib.Path("results/")
results_dir.mkdir(parents=True, exist_ok=True)


# ### Load dataframe

# In[3]:


# load features dataframe
features_dataframe_path = pathlib.Path("../1.format_data/data/training_data.csv.gz")
features_dataframe = get_features_data(features_dataframe_path)

# drop metadata columns from features dataframe
metadata_cols = [1,2,3,4,5,6,7,8,9,10,11]
features_dataframe = features_dataframe.drop(features_dataframe.columns[metadata_cols],axis=1)

features_dataframe


# ### Counts for all phenotypic classes

# In[4]:


features_dataframe["Mitocheck_Phenotypic_Class"].value_counts()


# ### Only keep certain phenoytpic classes for analysis

# In[5]:


classes_to_keep = [
    "Polylobed",
    "MetaphaseAlignment",
    "Interphase",
    "Artefact",
    "Binuclear",
    "Prometaphase",
    "Metaphase",
    "Large",
    "Apoptosis",
    "Elongated",
    "UndefinedCondensed",
    "SmallIrregular",
    "Hole",
    "Folded",
    "Grape",
]

features_dataframe = features_dataframe.loc[features_dataframe["Mitocheck_Phenotypic_Class"].isin(classes_to_keep)]
features_dataframe.shape


# ### 1D UMAP

# In[6]:


save_path = pathlib.Path(f"{results_dir}/1D_umap.tsv")
show_1D_umap(features_dataframe, save_path)


# ### 2D UMAP

# In[7]:


save_path = pathlib.Path(f"{results_dir}/2D_umap.tsv")
show_2D_umap(features_dataframe, save_path)


# ### 3D UMAP

# In[8]:


save_path = pathlib.Path(f"{results_dir}/3D_umap.tsv")
show_3D_umap(features_dataframe, save_path)


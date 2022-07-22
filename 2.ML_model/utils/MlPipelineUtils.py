import pandas as pd
import numpy as np
import pathlib
from typing import Tuple, Any, List, Union

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns


def get_features_data(load_path: pathlib.Path) -> pd.DataFrame:
    """get features data from csv at load path

    Args:
        load_path (pathlib.Path): path to training data csv

    Returns:
        pd.DataFrame: training dataframe
    """
    # read dataset into pandas dataframe
    features_data = pd.read_csv(load_path, index_col=0)

    # remove training data with ADCCM class as this class was not used for classification in original paper
    features_data = features_data[
        features_data["Mitocheck_Phenotypic_Class"] != "ADCCM"
    ]

    # replace shape1 and shape3 labels with their correct respective classes
    features_data = features_data.replace("Shape1", "Binuclear")
    features_data = features_data.replace("Shape3", "Polylobed")

    return features_data


def get_random_images_indexes(training_data: pd.DataFrame, num_images: int) -> List:
    """get image indexes from training dataset

    Args:
        training_data (pd.DataFrame): pandas dataframe of training data
        num_images (int): number of images to holdout

    Returns:
        List: list of image indexes to with held out images
    """
    unique_images = pd.unique(training_data["Metadata_Plate_Map_Name"])
    images = np.random.choice(unique_images, size=num_images, replace=False)

    image_indexes_list = []
    for image in images:
        image_indexes = training_data.index[
            training_data["Metadata_Plate_Map_Name"] == image
        ].tolist()
        image_indexes_list.extend(image_indexes)

    return image_indexes_list


def get_dataset(
    features_dataframe: pd.DataFrame, data_split_indexes: pd.DataFrame, label: str
) -> pd.DataFrame:
    """get testing data from features dataframe and the data split indexes

    Args:
        features_dataframe (pd.DataFrame): dataframe with all features data
        data_split_indexes (pd.DataFrame): dataframe with split indexes
        label (str): label to get data for (train, test, holdout)

    Returns:
        pd.DataFrame: _description_
    """
    indexes = data_split_indexes.loc[data_split_indexes["label"] == label]
    indexes = indexes["index"]
    data = features_dataframe.loc[indexes]

    return data


def get_X_y_data(training_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """generate X (features) and y (labels) dataframes from training data

    Args:
        training_data (pd.DataFrame): training dataframe

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: X, y dataframes
    """

    # all features from DeepProfiler have "efficientnet" in their column name
    morphology_features = [
        col for col in training_data.columns.tolist() if "efficientnet" in col
    ]

    # extract features
    X = training_data.loc[:, morphology_features].values

    # extract phenotypic class label
    y = training_data.loc[:, ["Mitocheck_Phenotypic_Class"]].values
    # make Y data
    y = np.ravel(y)

    # shuffle data because as it comes from MitoCheck same labels tend to be in grou
    X, y = shuffle(X, y, random_state=0)

    return X, y


def evaluate_model_cm(log_reg_model: LogisticRegression, dataset: pd.DataFrame):
    """display confusion matrix for logistic regression model on dataset

    Args:
        log_reg_model (LogisticRegression): logisitc regression model to evaluate
        dataset (pd.DataFrame): dataset to evaluate model on
    """
    
    # get features and labels dataframes
    X, y = get_X_y_data(dataset)
    
    # get predictions from model
    y_pred = log_reg_model.predict(X)
    
    # create confusion matrix
    conf_mat = confusion_matrix(y, y_pred, labels=log_reg_model.classes_)
    conf_mat = pd.DataFrame(conf_mat)
    conf_mat.columns = log_reg_model.classes_
    conf_mat.index = log_reg_model.classes_

    # display confusion matrix
    plt.figure(figsize=(15, 15))
    ax = sns.heatmap(data=conf_mat, annot=True, fmt=".0f", cmap="viridis", square=True)
    ax = plt.xlabel("Predicted Label")
    ax = plt.ylabel("True Label")
    ax = plt.title("Phenotypic Class Predicitions")
    
def evaluate_model_score(log_reg_model: LogisticRegression, dataset: pd.DataFrame):
    
    # get features and labels dataframes
    X, y = get_X_y_data(dataset)
    
    # get predictions from model
    y_pred = log_reg_model.predict(X)
    
    # display precision vs phenotypic class bar chart
    precisions = f1_score(y, y_pred, average=None)
    precisions = pd.DataFrame(precisions).T
    precisions.columns = np.unique(y_pred) #cant use log_reg_model.classes_ because classes may be missing due to lack of data

    sns.set(rc={"figure.figsize": (20, 8)})
    plt.xlabel("Phenotypic Class")
    plt.ylabel("Precision")
    plt.title("Precision vs Phenotpyic Class")
    plt.xticks(rotation=90)
    ax = sns.barplot(data=precisions)
    

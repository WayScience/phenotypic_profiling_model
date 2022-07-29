import pandas as pd
import numpy as np
import pathlib
from typing import Tuple, Any, List, Union

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns


# set numpy seed to make random operations reproduceable
np.random.seed(0)

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


def get_image_indexes(training_data: pd.DataFrame, images: List) -> List:

    image_indexes_list = []
    for image in images:
        image_indexes = training_data.index[
            training_data["Metadata_Plate_Map_Name"] == image
        ].tolist()
        image_indexes_list.extend(image_indexes)

    return image_indexes_list


def get_random_images_indexes(training_data: pd.DataFrame, num_images: int) -> List:
    """get ramdom images from training dataset

    Args:
        training_data (pd.DataFrame): pandas dataframe of training data
        num_images (int): number of images to holdout

    Returns:
        List: list of unique images for holding out
    """
    unique_images = pd.unique(training_data["Metadata_Plate_Map_Name"])
    images = np.random.choice(unique_images, size=num_images, replace=False)

    return images


def get_intelligent_images(training_data: pd.DataFrame, num_images: int) -> List:
    """get images from training dataset and try to balance labels present in these images
    add an image if it has at least class not represented by the other images
    if the image doesn't have a contribution via new class, try a new image

    Args:
        training_data (pd.DataFrame): pandas dataframe of training data
        num_images (int): number of images to holdout

    Returns:
        List: list of unique images with intelligently balanced phenotypic classes
    """
    remaining_images = pd.unique(training_data["Metadata_Plate_Map_Name"])
    remaining_classes = pd.unique(training_data["Mitocheck_Phenotypic_Class"]).tolist()

    images = []

    for image_number in range(num_images):
        image_contributes = False
        image = ""
        while not image_contributes:
            image = np.random.choice(remaining_images, size=1, replace=False)[0]
            image_data = training_data.loc[
                (training_data["Metadata_Plate_Map_Name"] == image)
            ]

            if len(remaining_classes) == 0:
                image_contributes = True

            image_classes = pd.unique(image_data["Mitocheck_Phenotypic_Class"])
            # check if any phenotypic class from the current image is in the remaining classes
            if any(
                phenotypic_class in image_classes
                for phenotypic_class in remaining_classes
            ):
                image_contributes = True
                remaining_classes = [
                    x for x in remaining_classes if x not in image_classes
                ]

        images.append(image)
        remaining_images = np.delete(
            remaining_images, np.where(remaining_images == image)
        )

    return images


def get_representative_images(
    training_data: pd.DataFrame, num_images: int, attempts: int = 100
) -> List:
    """get images from training dataset and such that every phenotypic class is represented
    returns None if no combintation of images are found that represent every phenotypic class within number of trials

    Args:
        training_data (pd.DataFrame): pandas dataframe of training data
        num_images (int): number of images to holdout
        attempts (int): number of times to try getting representative images

    Returns:
        List: list of images with every phenotypic class represented or None if this list cannot be curated
    """
    unique_classes = pd.unique(training_data["Mitocheck_Phenotypic_Class"]).tolist()

    trial = 0
    while trial < attempts:
        images = get_intelligent_images(training_data, num_images)

        images_data = pd.DataFrame()
        for image in images:
            image_data = training_data.loc[
                (training_data["Metadata_Plate_Map_Name"] == image)
            ]
            images_data = pd.concat([images_data, image_data])

        unique_image_classes = pd.unique(
            images_data["Mitocheck_Phenotypic_Class"]
        ).tolist()
        if set(unique_image_classes) == set(unique_classes):
            return images

        trial += 1

    print("No combination of images found that represents all classes!")
    return None


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


def evaluate_model_cm(
    log_reg_model: LogisticRegression, dataset: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """display confusion matrix for logistic regression model on dataset

    Args:
        log_reg_model (LogisticRegression): logisitc regression model to evaluate
        dataset (pd.DataFrame): dataset to evaluate model on

    Returns:
        Tuple[np.ndarray, np.ndarray]: true, predicted labels
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

    return y, y_pred


def evaluate_model_score(log_reg_model: LogisticRegression, dataset: pd.DataFrame):
    """display bar graph for model with scoring metric on each class

    Args:
        log_reg_model (LogisticRegression): logisitc regression model to evaluate
        dataset (pd.DataFrame): dataset to evaluate model on
    """

    # get features and labels dataframes
    X, y = get_X_y_data(dataset)

    # get predictions from model
    y_pred = log_reg_model.predict(X)

    # display precision vs phenotypic class bar chart
    scores = f1_score(
        y, y_pred, average=None, labels=log_reg_model.classes_, zero_division=0
    )
    scores = pd.DataFrame(scores).T
    scores.columns = log_reg_model.classes_

    sns.set(rc={"figure.figsize": (20, 8)})
    plt.xlabel("Phenotypic Class")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Phenotpyic Class")
    plt.xticks(rotation=90)
    ax = sns.barplot(data=scores)

#!/usr/bin/env python
# coding: utf-8

# # Format Training Data
# 
# ### Access training data from specific commit of [mitocheck_data](https://github.com/WayScience/mitocheck_data) and format this data into a single CSV file
# 
# ### Import libraries

# In[1]:


import pandas as pd
import urllib
import pathlib


# ### Define functions for formatting training data

# In[2]:


def get_single_cell_metadata(single_cell_data: pd.DataFrame):
    """get plate, well, frame information from single cell data

    Args:
        single_cell_data (pd.Dataframe): dataframe with single cell data

    Returns:
        str, str, str: plate, well, frame metadata as strings
    """
    # Metadata_DNA is in format plate/well/frame/filename.tif so get plate, well, frame info from this
    single_cell_info = single_cell_data["Metadata_DNA"].split("/")
    plate = single_cell_info[0]
    well = single_cell_info[1]
    frame = single_cell_info[2]
    return plate, well, frame


def get_cell_class(
    single_cell_data: pd.DataFrame,
    trainingset_file_url: str,
    plate: str,
    well: str,
    frame: str,
) -> str:
    """get phenotypic class of cell from trainingset.dat file, as labeled by MitoCheck

    Args:
        single_cell_data (pd.DataFrame): dataframe with single cell data
        trainingset_file_url (str): url location of raw traininset.dat file
        plate (str): plate cell is from
        well (str): well cell is from
        frame (str): frame cell is from

    Returns:
        str: phenotypic class of nucleus, as labeled by MitoCheck
    """
    well_string = f"W{str(well).zfill(5)}"
    frame_time = (int(frame) - 1) * 30
    frame_time_string = f"T{str(frame_time).zfill(5)}"
    frame_file_details = [plate, well_string, frame_time_string]
    obj_id = int(single_cell_data["Mitocheck_Object_ID"].item())
    obj_id_prefix = f"{obj_id}: "

    append = False
    # need to open trainingset file each time
    trainingset_file = urllib.request.urlopen(trainingset_file_url)
    for line in trainingset_file:
        decoded_line = line.decode("utf-8").strip()
        # match plate, well, frame to starting line for movie labels
        if all(detail in decoded_line for detail in frame_file_details):
            append = True
        if append and decoded_line.startswith(obj_id_prefix):
            return decoded_line.split(": ")[1]
    return None


def get_cell_control(
    plate: str,
    well: str,
    idr_metadata: pd.DataFrame,
):

    cell_annotations = idr_metadata.loc[
        (plate == idr_metadata["Plate"]) & (idr_metadata["Well Number"].astype(int) == int(well))
    ]
    control_type = cell_annotations.iloc[0]["Control Type"]
    
    if control_type == "positive control":
        return "positive"
    elif control_type == "negative control":
        return "negative"
    else:
        return "none"


def complete_single_cell(
    single_cell_data: pd.DataFrame,
    trainingset_file_url: str,
    segmentation_data_dir: str,
    idr_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Add Mitocheck_Object_ID and Mitocheck_Phenotypic_Class fields to single cell data by matching cell object ID to phenotypic class given in traininset.dat

    Args:
        single_cell_data (pd.DataFrame): single cell data
        trainingset_file_url (str): url location of raw traininset.dat file
        segmentation_data_dir (str): url location of the raw segmentation data directory

    Returns:
        pd.DataFrame: completed single cell data
    """
    plate, well, frame = get_single_cell_metadata(single_cell_data)
    segmentation_data_url = (
        f"{segmentation_data_dir}/{plate}/{well}/{frame}/{plate}_{well}_{frame}.tsv"
    )
    full_segmentation_data = pd.read_csv(segmentation_data_url, delimiter="\t").round(0)
    cell_x_y = (
        round(single_cell_data["Location_Center_X"]),
        round(single_cell_data["Location_Center_Y"]),
    )
    cell_segmentation_data = full_segmentation_data.loc[
        (full_segmentation_data["Location_Center_X"] == cell_x_y[0])
        & (full_segmentation_data["Location_Center_Y"] == cell_x_y[1])
    ]
    print(f"Processed cell at: {plate}/{well}/{frame}, location: {cell_x_y}")
    if cell_segmentation_data.empty:
        print("No segmentation data match found for this cell!")
    else:
        single_cell_data = single_cell_data.to_frame().transpose()
        single_cell_data.insert(
            0,
            "Mitocheck_Object_ID",
            cell_segmentation_data["Mitocheck_Object_ID"].item(),
        )
        # get class and append to single cell data
        cell_phenotypic_class = get_cell_class(
            single_cell_data, trainingset_file_url, plate, well, frame
        )
        if cell_phenotypic_class == None:
            print("This cell was not found in trainingset.dat!")
        single_cell_data.insert(0, "Mitocheck_Phenotypic_Class", cell_phenotypic_class)

        cell_control_type = get_cell_control(plate, well, idr_metadata)
        single_cell_data.insert(1, "Control_Type", cell_control_type)

    return single_cell_data


def format_training_data(mitocheck_data_version_url: str) -> pd.DataFrame:
    """Add Mitocheck_Object_ID and Mitocheck_Phenotypic_Class fields to each single cell and compile all the cells into a single training data dataframe

    Args:
        mitocheck_data_version_url (str): url with path to desired version of raw mitocheck_data

    Returns:
        pd.DataFrame: completed training data with Mitocheck_Object_ID and Mitocheck_Phenotypic_Class for each cell
    """
    trainingset_file_url = (
        f"{mitocheck_data_version_url}/0.download_data/trainingset.dat"
    )
    segmentation_data_dir = f"{mitocheck_data_version_url}/2.segment_nuclei/segmented/"
    preprocessed_features_url = f"{mitocheck_data_version_url}/4.preprocess_features/data/normalized_training_data.csv.gz"
    idr_metadata_url = f"{mitocheck_data_version_url}/3.extract_features/idr0013-screenA-annotation.csv.gz"

    preprocessed_features = pd.read_csv(preprocessed_features_url, compression="gzip")
    print("Loaded preprocessed features!")

    idr_metadata = pd.read_csv(idr_metadata_url, dtype=object, compression="gzip")
    print("Loaded idr metadata!")

    training_data = []
    for index, row in preprocessed_features.iterrows():
        single_cell = row
        completed_single_cell = complete_single_cell(
            single_cell, trainingset_file_url, segmentation_data_dir, idr_metadata
        )
        training_data.append(completed_single_cell)

    training_data = pd.concat(training_data)
    return training_data


# ### Format training data

# In[3]:


base_url = "https://raw.github.com/WayScience/mitocheck_data/"
# hash changes depending on desired version of mitocheck_data being used
hash = "de21b9c3201ba4298db2b1704f3ae510a5dc47e2"
mitocheck_data_version_url = f"{base_url}/{hash}"

output_dir = pathlib.Path("data/")
output_dir.mkdir(parents=True, exist_ok=True)
save_path = pathlib.Path(f"{output_dir}/training_data.csv.gz")

training_data = format_training_data(mitocheck_data_version_url)
training_data.to_csv(save_path, compression="gzip")


# In[4]:


print(training_data.shape)
training_data.head()


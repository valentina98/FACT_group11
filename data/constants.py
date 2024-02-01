import os

# CUB Constants
# CUB data is downloaded from the CBM release.
# Dataset: https://worksheets.codalab.org/rest/bundles/0xd013a7ba2e88481bbc07e787f73109f5/ 
# Metadata and splits: https://worksheets.codalab.org/bundles/0x5b9d528d2101418b87212db92fea6683
CUB_DATA_DIR = "/content/FACT_group11/datasets/CUB_200_2011"
CUB_PROCESSED_DIR = "/content/FACT_group11/datasets/class_attr_data_10"


# Derm data constants
# Derm7pt is obtained from : https://derm.cs.sfu.ca/Welcome.html
DERM7_FOLDER = "/content/FACT_group11/datasets/derm7pt/release_v0/"
DERM7_META = os.path.join(DERM7_FOLDER, "meta", "meta.csv")
DERM7_TRAIN_IDX = os.path.join(DERM7_FOLDER, "meta", "train_indexes.csv")
DERM7_VAL_IDX = os.path.join(DERM7_FOLDER, "meta", "valid_indexes.csv")

# Derm data constants
# Derm7pt is obtained from : https://derm.cs.sfu.ca/Welcome.html
DERM7_FOLDER = "/datasets/derm7pt/"
DERM7_META = os.path.join(DERM7_FOLDER, "meta", "meta.csv")
DERM7_TRAIN_IDX = os.path.join(DERM7_FOLDER, "meta", "train_indexes.csv")
DERM7_VAL_IDX = os.path.join(DERM7_FOLDER, "meta", "valid_indexes.csv")

# Ham10000 can be obtained from : https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
HAM10K_DATA_DIR = "/content/FACT_group11/datasets/broden_concepts/"


# BRODEN concept bank
BRODEN_CONCEPTS = "/content/FACT_group11/datasets/broden_concepts/"


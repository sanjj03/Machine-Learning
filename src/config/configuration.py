from src.constants import *
import os

ROOT_DIR = ROOT_DIR_KEY

DATASET_PATH = os.path.join(ROOT_DIR, DATA_DIR, DATA_DIR_KEY)

RAW_FILE_PATH = os.path.join(
    ROOT_DIR, ARTIFACT_DIR_KEY,
    DATA_INGESTION_KEY, CURRENT_TIME_STAMP,
    DATA_INGESTION_RAW_DATA_DIR,
    RAW_DATA_DIR_KEY
)

TRAIN_FILE_PATH = os.path.join(
    ROOT_DIR, ARTIFACT_DIR_KEY,
    DATA_INGESTION_KEY, CURRENT_TIME_STAMP,
    DATA_INGESTION_INGESTED_DATA_DIR,
    TRAIN_DATA_DIR_KEY
)

TEST_FILE_PATH = os.path.join(
    ROOT_DIR, ARTIFACT_DIR_KEY,
    DATA_INGESTION_KEY, CURRENT_TIME_STAMP,
    DATA_INGESTION_INGESTED_DATA_DIR,
    TEST_DATA_DIR_KEY
)


#data transformation

PREPROCCESSING_OBJ_FILE = os.path.join(ROOT_DIR,
                                       ARTIFACT_DIR_KEY,
                                       DATA_TRANSFORMATION_ARTIFACT,
                                       DATA_PREPROCESSED_DIR,
                                       DATA_TRANSFORMATION_PREPROCESSING_OBJ)

TRANSFORM_TRAIN_FILE_PATH = os.path.join(ROOT_DIR,
                                         ARTIFACT_DIR_KEY,
                                         DATA_TRANSFORMATION_ARTIFACT,
                                         DATA_TRANSFORMATION_DIR,
                                         TRANSFORM_TRAIN_DIR_KEY)

TRANSFORM_TEST_FILE_PATH = os.path.join(ROOT_DIR,
                                         ARTIFACT_DIR_KEY,
                                         DATA_TRANSFORMATION_ARTIFACT,
                                         DATA_TRANSFORMATION_DIR,
                                         TRANSFORM_TEST_DIR_KEY)

FEATURE_ENGG_OBJ_FILE_PATH = os.path.join(ROOT_DIR,
                                          ARTIFACT_DIR_KEY,
                                          DATA_TRANSFORMATION_ARTIFACT,
                                          DATA_PREPROCESSED_DIR, 'feature_engg.pkl')

#model training
MODEL_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, MODEL_TRAINER_KEY, MODEL_OBJECT)




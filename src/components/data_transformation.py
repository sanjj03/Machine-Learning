from src.constants import *
from src.logger import logging
from src.exception import CustomException
import os, sys
from src.config.configuration import *
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.utils import save_obj




#data transformation
class Feature_engineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        logging.info("***************feature engineering started**********")
    def fit(self, X, y=None):
        # Nothing to learn, but required by sklearn
        return self
    def distance_numpy(self, df, lat1, lat2, lon1, lon2):
        p = np.pi/180
        a  = (
            0.5
            - np.cos((df[lat2] - df[lat1]) * p) / 2
            + np.cos(df[lat1] * p)
            * np.cos(df[lat2] * p)
            * (1 - np.cos(df[lon2] - df[lon1]))
        )
        df["distance"] = 12742 * np.arccos(1 - a)
    def transform_data(self, X):
        try:
            df= X.copy()
            if "ID" in df.columns:
                df.drop(["ID"], axis =1 , inplace=True)

            self.distance_numpy(df, 'Restaurant_latitude',
                                'Restaurant_longitude',
                                'Delivery_location_latitude',
                                'Delivery_location_longitude')
            
            drop_cols=['Delivery_person_ID',
                         'Restaurant_latitude',
                         'Restaurant_longitude',
                         'Delivery_location_latitude',
                         'Delivery_location_longitude',
                         'Order_Date',
                         'Time_Orderd',
                         'Time_Order_picked']
            for  col in drop_cols:
                if col in df.columns:
                    df.drop(col, axis=1, inplace=True)
            logging.info("Feature engineering completed")

            return df
        
        except Exception as e :
            raise CustomException(e,sys)
        
    def transform(self, X):
        try:
            
            transformered_df = self.transform_data(X)
            return transformered_df
        except Exception as e:
            raise CustomException (e, sys)
        
@dataclass
class DataTransformationConfig():
    processed_obj_file_path = PREPROCCESSING_OBJ_FILE 
    transform_train_path =TRANSFORM_TRAIN_FILE_PATH
    transform_test_path = TRANSFORM_TEST_FILE_PATH
    feature_engg_obj_path = FEATURE_ENGG_OBJ_FILE_PATH

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            Road_traffic_density = ['Low','Medium','High','jam']
            Weather_conditions = ['Sunny','Cloudy','Fogg','Windy','Sandstorms','Stormy']
            categorical_columns = ['Type_of_order','Type_of_vehicle','Festival','City']
            ordinal_encoder = ['Road_traffic_density','Weather_conditions']
            numerical_columns = ['Delivery_person_Age','Delivery_person_Ratings','Vehicle_condition','multiple_deliveries','distance']

            #numerical pipeline
            numerical_pipeline = Pipeline(steps = [
                ('impute', SimpleImputer(strategy='constant',fill_value=0)),
                ('scaler', StandardScaler(with_mean=False))
            ])

            #categorical pipeline
            categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ("scaler", StandardScaler(with_mean=False))
    ]
)


            #ordianal pipeline
            ordinal_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinalencoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ("scaler", StandardScaler())
    ]
)


            preprocessor = ColumnTransformer([
                ('numerical_pipeline', numerical_pipeline, numerical_columns),
                ('categorical_pipeline', categorical_pipeline, categorical_columns),
                (' ordinal_pipeline',  ordinal_pipeline, ordinal_encoder)
            ])


            return preprocessor
            logging.infor("Pipeline steps Comppleted")
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def get_feature_engineering_object(self):
        try:
            feature_engineering = Pipeline(steps=[("fe", Feature_engineering())])
            return feature_engineering

        except Exception as e:
            raise CustomException(e,sys)
            

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Obtaining FE steps object")
            fe_obj = self.get_feature_engineering_object()

            train_df =fe_obj.fit_transform(train_df) 
            test_df =fe_obj.transform(test_df) 

            train_df.to_csv("train_data.csv")
            test_df.to_csv("test_data.csv")

            processing_obj = self.get_data_transformation_obj()
            target_columns_name = "Time_taken (min)"


            X_train = train_df.drop(columns = target_columns_name,axis=1)
            y_train = train_df[target_columns_name]
            X_test = test_df.drop(columns = target_columns_name,axis=1)
            y_test = test_df[target_columns_name]

            X_train = processing_obj.fit_transform(X_train)
            X_test = processing_obj.transform(X_test)

            
            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]


            df_train = pd.DataFrame(train_arr)
            df_test = pd.DataFrame(test_arr)

            os.makedirs(os.path.dirname(self.data_transformation_config.transform_train_path), exist_ok=True)
            df_train.to_csv(self.data_transformation_config.transform_train_path, index=False, header=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transform_test_path), exist_ok=True)
            df_test.to_csv(self.data_transformation_config.transform_test_path, index=False, header=True)

            save_obj(file_path = self.data_transformation_config.processed_obj_file_path, obj=processing_obj)

            save_obj(file_path = self.data_transformation_config.feature_engg_obj_path, obj=fe_obj)

            
            return train_arr, test_arr


        except Exception as e:
            raise CustomException(e,sys)

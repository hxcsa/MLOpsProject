import os
import pandas as pd 
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from src.logger import logging
# pyrefly: ignore [missing-import]
from src.custom_exception import CustomException


logger = logging.getLogger(__name__)

class DataProccessing:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.df = None
        self.x = None
        self.y = None
        self.selected_features = []

        os.makedirs(output_path, exist_ok=True)
        logger.info(f"DataProccessing initialized")

    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_path)
            logger.info(f"Data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException("Failed to load data")

    def preproccess_data(self):
        try:
           self.df = self.df.drop(columns=['Patient_ID'])
           self.y = self.df['Survival_Prediction']
           self.x = self.df.drop(columns=['Survival_Prediction'])
           caregorical_cols = self.x.select_dtypes(include = ['object']).columns
           self.label_encoders = {}

           for col in caregorical_cols:
                le = LabelEncoder()
                self.x[col] = le.fit_transform(self.x[col])
                self.label_encoders[col] = le

           logger.info(f"Basic preproccessing completed")
        except Exception as e:
            logger.error(f"Error preproccessing data: {e}")
            raise CustomException("Failed to preproccess data")
        
    def feature_selection(self):
        try:
            x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
            x_cat = x_train.select_dtypes(include=['int64', 'float64'])
            chi2_selector = SelectKBest(score_func=chi2, k='all')
            chi2_selector.fit(x_cat, y_train)

            chi2_scores = pd.DataFrame({
                'features': x_cat.columns,
                'chi2_score': chi2_selector.scores_,
            }).sort_values(by='chi2_score', ascending=False)

            self.selected_features = chi2_scores.head(10)['features'].tolist()
           
            logger.info(f"Selected features: {self.selected_features}")

            self.x = self.x[self.selected_features]
            logger.info(f"Feature selection completed")

        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            raise CustomException("Failed to feature selection")


    def split_data(self):
        try:
            x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42, stratify=self.y)
            x_train = self.scaler.fit_transform(x_train)
            x_test = self.scaler.transform(x_test)

            logger.info(f"Data split into train and test")

            return x_train, x_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error in split data: {e}")
            raise CustomException("Failed to split data")



    def save_data_and_scaler(self, x_train, x_test, y_train, y_test, ):
        try:
            joblib.dump(x_train , os.path.join(self.output_path ,  'x_train.pkl'))
            joblib.dump(x_test , os.path.join(self.output_path ,  'x_test.pkl'))
            joblib.dump(y_train , os.path.join(self.output_path ,  'y_train.pkl'))
            joblib.dump(y_test , os.path.join(self.output_path ,  'y_test.pkl'))
            joblib.dump(self.scaler , os.path.join(self.output_path ,  'scaler.pkl'))
            logger.info(f"Data and scaler saved successfully")
        except Exception as e:
            logger.error(f"Error saving data and scaler: {e}")
            raise CustomException("Failed to save data and scaler")
        

    def run(self):
        try:
            self.load_data()
            self.preproccess_data()
            self.feature_selection()
            x_train, x_test, y_train, y_test = self.split_data()
            self.save_data_and_scaler(x_train, x_test, y_train, y_test, )
            logger.info(f"Data proccessing completed")
        except Exception as e:
            logger.error(f"Error in data proccessing: {e}")
            raise CustomException("Failed to proccess data")
        
        
if __name__ == "__main__":
    input_path = "artifacts/raw/data.csv"
    output_path = "artifacts/proccessed"
    proccessor = DataProccessing(input_path, output_path)
    proccessor.run()
    





    

    



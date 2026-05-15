import pandas as pd 
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib 
import os
import sys
from src.logger import logging
import mlflow
import mlflow.sklearn


# pyrefly: ignore [missing-import]
from src.custom_exception import CustomException

logger = logging.getLogger(__name__)

def configure_console_encoding():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


class Model_training():
    def __init__(self, processed_data_path='artifacts/proccessed'):
        self.proccessed_data_path = processed_data_path
        self.model_path = 'artifacts/model'
        os.makedirs(self.model_path, exist_ok=True)

        logger.info("Model_training class initialized successfully")


    def load_data(self):
        try:
            self.x_train = joblib.load(os.path.join(self.proccessed_data_path, 'x_train.pkl'))
            self.y_train = joblib.load(os.path.join(self.proccessed_data_path, 'y_train.pkl'))
            self.x_test = joblib.load(os.path.join(self.proccessed_data_path, 'x_test.pkl'))
            self.y_test = joblib.load(os.path.join(self.proccessed_data_path, 'y_test.pkl'))

            logger.info("Data loaded successfully")
        except Exception as e:
            logger.error("Error occurred while loading data")
            raise CustomException(e)
        

    def train_model(self):
        try:
            self.model = GradientBoostingClassifier(n_estimators=80, learning_rate=0.02, max_depth=5, min_samples_split=4, min_samples_leaf=3, subsample=0.33, random_state=42)
            self.model.fit(self.x_train, self.y_train)

            joblib.dump(self.model, os.path.join(self.model_path, 'model.pkl')) 
           
           
            logger.info("Model trained successfully")
        except Exception as e:
            logger.error("Error occurred while training model")
            raise CustomException(e)
        
    def evaluate_model(self):
        try:
            y_pred = self.model.predict(self.x_test)
            y_probe = self.model.predict_proba(self.x_test)[:, 1] if len(self.y_test.unique()) == 2 else None
            y_test_binary = self.y_test.map({"No": 0, "Yes": 1}) if self.y_test.dtype == object else self.y_test
            
            
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, pos_label="Yes")
            recall = recall_score(self.y_test, y_pred, pos_label="Yes")
            f1 = f1_score(self.y_test, y_pred, pos_label="Yes")
            roc_auc = roc_auc_score(y_test_binary, y_probe) if y_probe is not None else None

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            logger.info(f"Model evaluation completed successfully. Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, ROC AUC: {roc_auc}")
        except Exception as e:
            logger.error("Error occurred while evaluating model")
            raise CustomException(e)
        
    def run(self):
        try:
            self.load_data()
            self.train_model()
            self.evaluate_model()

            logger.info("Model training pipeline completed successfully")
        except Exception as e:
            logger.error("Error occurred while running model training pipeline")
            raise CustomException(e)
            

if __name__ == "__main__":    
    configure_console_encoding()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Cancer Prediction")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    logger.info(f"MLflow experiment: {experiment_name}")

    with mlflow.start_run(run_name="Model Training"):
        model_trainer = Model_training()
        model_trainer.run()

        mlflow.log_params({"model_name": "GradientBoostingClassifier"})
        mlflow.log_params({"n_estimators": 200})
        mlflow.log_params({"learning_rate": 0.03})
        mlflow.log_params({"max_depth": 5})
        mlflow.log_params({"min_samples_split": 4})
        mlflow.log_params({"min_samples_leaf": 3})
        mlflow.log_params({"subsample": 0.33})
        mlflow.log_params({"random_state": 42})

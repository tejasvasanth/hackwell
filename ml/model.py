import xgboost as xgb
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import joblib
import os
from datetime import datetime
from loguru import logger

from config.settings import settings

class MLModel:
    def __init__(self):
        self.model = None
        self.model_name = settings.default_model_name
        self.model_version = None
        
        # Initialize MLflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        
        # Load model if exists
        self._load_latest_model()
    
    def _load_latest_model(self):
        """Load the latest model from MLflow model registry"""
        try:
            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions(
                self.model_name, 
                stages=["Production", "Staging"]
            )
            
            if latest_version:
                model_version = latest_version[0].version
                model_uri = f"models:/{self.model_name}/{model_version}"
                self.model = mlflow.xgboost.load_model(model_uri)
                self.model_version = model_version
                logger.info(f"Loaded model {self.model_name} version {model_version}")
            else:
                logger.warning("No model found in registry")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    async def predict(self, features: Dict[str, Any]) -> float:
        """Make prediction using the loaded model"""
        if self.model is None:
            raise ValueError("No model loaded")
        
        try:
            # Convert features to DataFrame
            df = pd.DataFrame([features])
            
            # Make prediction
            prediction = self.model.predict(df)[0]
            
            # Log prediction to MLflow
            with mlflow.start_run(run_name="prediction"):
                mlflow.log_params(features)
                mlflow.log_metric("prediction", prediction)
            
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    async def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                         X_val: pd.DataFrame = None, y_val: pd.Series = None) -> str:
        """Train XGBoost model with MLflow tracking"""
        
        with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Model parameters
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'random_state': 42
            }
            
            # Log parameters
            mlflow.log_params(params)
            
            # Train model
            model = xgb.XGBRegressor(**params)
            
            if X_val is not None and y_val is not None:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10,
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)
            
            # Calculate metrics
            train_score = model.score(X_train, y_train)
            mlflow.log_metric("train_r2", train_score)
            
            if X_val is not None:
                val_score = model.score(X_val, y_val)
                mlflow.log_metric("val_r2", val_score)
            
            # Log model
            mlflow.xgboost.log_model(
                model, 
                "model",
                registered_model_name=self.model_name
            )
            
            # Update current model
            self.model = model
            
            run_id = mlflow.active_run().info.run_id
            logger.info(f"Model trained successfully. Run ID: {run_id}")
            
            return run_id
    
    async def retrain(self) -> str:
        """Trigger model retraining (placeholder for Prefect integration)"""
        # This would typically trigger a Prefect flow
        logger.info("Retraining request received")
        return f"retrain_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models from MLflow registry"""
        try:
            client = mlflow.tracking.MlflowClient()
            models = client.search_registered_models()
            
            model_list = []
            for model in models:
                latest_versions = client.get_latest_versions(
                    model.name, 
                    stages=["Production", "Staging", "None"]
                )
                
                for version in latest_versions:
                    model_list.append({
                        "name": model.name,
                        "version": version.version,
                        "stage": version.current_stage,
                        "created_at": version.creation_timestamp
                    })
            
            return model_list
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
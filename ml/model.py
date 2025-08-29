import xgboost as xgb
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import joblib
import os
from datetime import datetime
from loguru import logger
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    async def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make 90-day deterioration risk prediction using the loaded model"""
        if self.model is None:
            raise ValueError("No model loaded")
        
        try:
            # Convert features to DataFrame
            df = pd.DataFrame([features])
            
            # Make prediction (probability of deterioration)
            risk_probability = self.model.predict_proba(df)[0][1]  # Probability of class 1 (deterioration)
            
            # Determine risk category
            if risk_probability >= 0.7:
                risk_category = "High"
            elif risk_probability >= 0.4:
                risk_category = "Medium"
            else:
                risk_category = "Low"
            
            result = {
                "risk_score": float(risk_probability),
                "risk_category": risk_category,
                "prediction_date": datetime.now().isoformat(),
                "model_version": self.model_version
            }
            
            # Log prediction to MLflow
            with mlflow.start_run(run_name="deterioration_prediction"):
                mlflow.log_params(features)
                mlflow.log_metric("risk_score", risk_probability)
                mlflow.log_param("risk_category", risk_category)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    async def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                         X_val: pd.DataFrame = None, y_val: pd.Series = None) -> str:
        """Train XGBoost model with MLflow tracking"""
        
        with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Model parameters for binary classification (90-day deterioration prediction)
            params = {
                'objective': 'binary:logistic',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'random_state': 42,
                'eval_metric': 'auc'
            }
            
            # Log parameters
            mlflow.log_params(params)
            
            # Train model
            model = xgb.XGBClassifier(**params)
            
            if X_val is not None and y_val is not None:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10,
                    verbose=False
                )
                
                # Calculate comprehensive evaluation metrics
                val_metrics = await self._evaluate_model(model, X_val, y_val, "validation")
                for metric_name, metric_value in val_metrics.items():
                    mlflow.log_metric(f"val_{metric_name}", metric_value)
            else:
                model.fit(X_train, y_train)
            
            # Calculate training metrics
            train_metrics = await self._evaluate_model(model, X_train, y_train, "training")
            for metric_name, metric_value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", metric_value)
            
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
    
    async def _evaluate_model(self, model, X: pd.DataFrame, y: pd.Series, dataset_name: str) -> Dict[str, float]:
        """Comprehensive model evaluation with clinical metrics"""
        try:
            # Get predictions
            y_pred_proba = model.predict_proba(X)[:, 1]  # Probability of deterioration
            y_pred = model.predict(X)
            
            # Calculate metrics
            metrics = {}
            
            # AUROC (Area Under ROC Curve)
            metrics['auroc'] = roc_auc_score(y, y_pred_proba)
            
            # AUPRC (Area Under Precision-Recall Curve)
            metrics['auprc'] = average_precision_score(y, y_pred_proba)
            
            # Confusion Matrix metrics
            cm = confusion_matrix(y, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall/True Positive Rate
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
            metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['sensitivity']) / (metrics['precision'] + metrics['sensitivity']) if (metrics['precision'] + metrics['sensitivity']) > 0 else 0
            
            # Calibration metrics
            prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=10)
            calibration_error = np.mean(np.abs(prob_true - prob_pred))
            metrics['calibration_error'] = calibration_error
            
            # Log confusion matrix as artifact
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No Deterioration', 'Deterioration'],
                       yticklabels=['No Deterioration', 'Deterioration'])
            plt.title(f'Confusion Matrix - {dataset_name.title()}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save plot
            plot_path = f"confusion_matrix_{dataset_name}.png"
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            plt.close()
            
            # Log calibration plot
            plt.figure(figsize=(8, 6))
            plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model')
            plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title(f'Calibration Plot - {dataset_name.title()}')
            plt.legend()
            
            calibration_path = f"calibration_plot_{dataset_name}.png"
            plt.savefig(calibration_path)
            mlflow.log_artifact(calibration_path)
            plt.close()
            
            # Clean up plot files
            if os.path.exists(plot_path):
                os.remove(plot_path)
            if os.path.exists(calibration_path):
                os.remove(calibration_path)
            
            logger.info(f"{dataset_name.title()} metrics - AUROC: {metrics['auroc']:.3f}, AUPRC: {metrics['auprc']:.3f}, F1: {metrics['f1_score']:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}
    
    async def get_model_evaluation(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Get comprehensive model evaluation for dashboard display"""
        if self.model is None:
            raise ValueError("No model loaded")
        
        try:
            metrics = await self._evaluate_model(self.model, X_test, y_test, "test")
            
            # Add additional information for dashboard
            evaluation_result = {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "evaluation_date": datetime.now().isoformat(),
                "metrics": metrics,
                "sample_size": len(X_test),
                "positive_cases": int(y_test.sum()),
                "negative_cases": int(len(y_test) - y_test.sum())
            }
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error getting model evaluation: {e}")
            raise
    
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
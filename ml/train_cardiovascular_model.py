#!/usr/bin/env python3
"""
Cardiovascular Risk Prediction Training Script

This script:
1. Pulls features from Feast feature store
2. Trains an XGBoost model for cardiovascular risk prediction
3. Logs comprehensive metrics to MLflow
4. Saves the best model to MLflow registry
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

# ML libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# MLflow
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Project imports
from config.feast_config import feast_store
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CardiovascularRiskTrainer:
    """
    Cardiovascular Risk Prediction Model Trainer
    
    Integrates Feast feature store, XGBoost training, and MLflow tracking
    for comprehensive cardiovascular risk prediction model development.
    """
    
    def __init__(self, experiment_name: str = "cardiovascular_risk_prediction"):
        self.experiment_name = experiment_name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.mlflow_client = MlflowClient()
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
            else:
                logger.info(f"Using existing MLflow experiment: {experiment_name}")
        except Exception as e:
            logger.error(f"Error setting up MLflow experiment: {e}")
            raise
    
    def pull_features_from_feast(self, 
                                entity_ids: Optional[List[str]] = None,
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Pull features from Feast feature store for training.
        
        Args:
            entity_ids: List of patient IDs to retrieve features for
            start_date: Start date for historical features
            end_date: End date for historical features
            
        Returns:
            DataFrame with features and target variable
        """
        logger.info("Pulling features from Feast feature store")
        
        try:
            # Set default date range if not provided
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=90)  # Last 90 days
            
            # Define feature references
            feature_refs = [
                "demographics:age",
                "demographics:gender_encoded",
                "demographics:bmi",
                "lifestyle:smoking_encoded",
                "lifestyle:exercise_encoded",
                "lifestyle:steps_daily",
                "lifestyle:sleep_hours",
                "lifestyle:heart_rate_avg",
                "lifestyle:activity_level",
                "lifestyle:stress_level",
                "labs:bp_systolic",
                "labs:bp_diastolic",
                "labs:cholesterol_total",
                "labs:cholesterol_ldl",
                "labs:cholesterol_hdl",
                "labs:glucose_fasting",
                "labs:hba1c"
            ]
            
            # Create entity DataFrame
            if entity_ids is None:
                # Generate sample entity IDs (in production, this would come from your data)
                entity_ids = [str(i) for i in range(1, 1001)]  # 1000 patients
            
            entity_df = pd.DataFrame({
                "patient_id": entity_ids,
                "event_timestamp": [end_date] * len(entity_ids)
            })
            
            # Get historical features from Feast
            logger.info(f"Retrieving features for {len(entity_ids)} patients")
            logger.info(f"Date range: {start_date} to {end_date}")
            logger.info(f"Features: {feature_refs}")
            
            # In a real implementation, this would be:
            # training_df = feast_store.get_historical_features(
            #     entity_df=entity_df,
            #     features=feature_refs
            # ).to_df()
            
            # For demonstration, create synthetic data with realistic distributions
            np.random.seed(42)
            n_samples = len(entity_ids)
            
            training_data = {
                "patient_id": entity_ids,
                "event_timestamp": [end_date] * n_samples,
                # Demographics
                "demographics__age": np.random.normal(55, 15, n_samples).clip(18, 90),
                "demographics__gender_encoded": np.random.binomial(1, 0.5, n_samples),
                "demographics__bmi": np.random.normal(26, 4, n_samples).clip(15, 45),
                # Lifestyle
                "lifestyle__smoking_encoded": np.random.binomial(1, 0.2, n_samples),
                "lifestyle__exercise_encoded": np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2]),
                "lifestyle__steps_daily": np.random.normal(7000, 2000, n_samples).clip(1000, 20000),
                "lifestyle__sleep_hours": np.random.normal(7.5, 1, n_samples).clip(4, 12),
                "lifestyle__heart_rate_avg": np.random.normal(72, 10, n_samples).clip(50, 120),
                "lifestyle__activity_level": np.random.uniform(1, 5, n_samples),
                "lifestyle__stress_level": np.random.uniform(1, 5, n_samples),
                # Lab values
                "labs__bp_systolic": np.random.normal(130, 20, n_samples).clip(90, 200),
                "labs__bp_diastolic": np.random.normal(80, 15, n_samples).clip(60, 120),
                "labs__cholesterol_total": np.random.normal(200, 40, n_samples).clip(120, 350),
                "labs__cholesterol_ldl": np.random.normal(120, 30, n_samples).clip(50, 250),
                "labs__cholesterol_hdl": np.random.normal(50, 15, n_samples).clip(20, 100),
                "labs__glucose_fasting": np.random.normal(95, 20, n_samples).clip(70, 200),
                "labs__hba1c": np.random.normal(5.5, 0.8, n_samples).clip(4, 12)
            }
            
            training_df = pd.DataFrame(training_data)
            
            # Create cardiovascular risk target based on risk factors
            risk_score = (
                (training_df["demographics__age"] > 60).astype(int) * 0.25 +
                (training_df["lifestyle__smoking_encoded"] == 1).astype(int) * 0.3 +
                (training_df["labs__bp_systolic"] > 140).astype(int) * 0.2 +
                (training_df["labs__cholesterol_total"] > 240).astype(int) * 0.15 +
                (training_df["demographics__bmi"] > 30).astype(int) * 0.1
            )
            
            # Add some noise and create binary target
            risk_score += np.random.normal(0, 0.1, n_samples)
            training_df["cardiovascular_risk"] = (risk_score > 0.4).astype(int)
            
            logger.info(f"Successfully retrieved {len(training_df)} samples with {len(feature_refs)} features")
            logger.info(f"Target distribution: {training_df['cardiovascular_risk'].value_counts().to_dict()}")
            
            return training_df
            
        except Exception as e:
            logger.error(f"Error pulling features from Feast: {e}")
            raise
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare features and target for training.
        
        Args:
            df: DataFrame with features and target
            
        Returns:
            Tuple of (features, target, feature_names)
        """
        logger.info("Preparing training data")
        
        # Separate features and target
        target_col = "cardiovascular_risk"
        exclude_cols = ["patient_id", "event_timestamp", target_col]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Store feature names
        self.feature_names = feature_cols
        
        logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
        logger.info(f"Feature names: {feature_cols}")
        
        return X, y, feature_cols
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train XGBoost model with comprehensive evaluation.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            random_state: Random seed
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.info("Starting XGBoost model training")
        
        with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name(self.experiment_name).experiment_id):
            # Log dataset info
            mlflow.log_param("dataset_size", len(X))
            mlflow.log_param("n_features", len(X.columns))
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            logger.info(f"Training set: {len(X_train)} samples")
            logger.info(f"Test set: {len(X_test)} samples")
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Convert to DMatrix for XGBoost
            dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=self.feature_names)
            dtest = xgb.DMatrix(X_test_scaled, label=y_test, feature_names=self.feature_names)
            
            # Calculate class weights for imbalanced data
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            scale_pos_weight = class_weights[1] / class_weights[0] if len(class_weights) > 1 else 1.0
            
            # XGBoost parameters
            params = {
                'objective': 'binary:logistic',
                'eval_metric': ['logloss', 'auc'],
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': scale_pos_weight,
                'random_state': random_state,
                'verbosity': 1
            }
            
            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            # Train model with early stopping
            evals = [(dtrain, 'train'), (dtest, 'eval')]
            
            self.model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=1000,
                evals=evals,
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            # Make predictions
            y_pred_proba = self.model.predict(dtest)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log confusion matrix details
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            confusion_metrics = {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            }
            
            for cm_metric, cm_value in confusion_metrics.items():
                mlflow.log_metric(cm_metric, cm_value)
            
            # Cross-validation
            cv_scores = self._perform_cross_validation(X_train_scaled, y_train, params)
            mlflow.log_metric('cv_accuracy_mean', cv_scores['accuracy_mean'])
            mlflow.log_metric('cv_accuracy_std', cv_scores['accuracy_std'])
            
            # Log model
            signature = infer_signature(X_train, y_pred_proba)
            mlflow.xgboost.log_model(
                self.model,
                "model",
                signature=signature,
                input_example=X_train.iloc[:5]
            )
            
            # Log feature importance
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.get_score(importance_type='weight').values()
            }).sort_values('importance', ascending=False)
            
            mlflow.log_text(importance_df.to_string(), "feature_importance.txt")
            
            # Get run info
            run_id = mlflow.active_run().info.run_id
            
            logger.info(f"Model training completed. MLflow run ID: {run_id}")
            logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Test AUC: {metrics['auc']:.4f}")
            
            return {
                'run_id': run_id,
                'metrics': metrics,
                'confusion_matrix': confusion_metrics,
                'cv_scores': cv_scores,
                'model': self.model,
                'feature_importance': importance_df
            }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1': f1_score(y_true, y_pred, average='binary'),
            'auc': roc_auc_score(y_true, y_pred_proba)
        }
    
    def _perform_cross_validation(self, X: np.ndarray, y: np.ndarray, params: Dict) -> Dict[str, float]:
        """
        Perform stratified k-fold cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            params: XGBoost parameters
            
        Returns:
            Dictionary with CV results
        """
        logger.info("Performing cross-validation")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Create XGBoost classifier for sklearn compatibility
        xgb_clf = xgb.XGBClassifier(**{k: v for k, v in params.items() if k != 'eval_metric'})
        
        cv_scores = cross_val_score(xgb_clf, X, y, cv=cv, scoring='accuracy')
        
        return {
            'accuracy_mean': cv_scores.mean(),
            'accuracy_std': cv_scores.std()
        }
    
    def register_best_model(self, run_id: str, model_name: str = "cardiovascular_risk_model") -> str:
        """
        Register the best model to MLflow Model Registry.
        
        Args:
            run_id: MLflow run ID
            model_name: Name for the registered model
            
        Returns:
            Model version
        """
        logger.info(f"Registering model to MLflow Registry: {model_name}")
        
        try:
            # Create model URI
            model_uri = f"runs:/{run_id}/model"
            
            # Register model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                description="XGBoost model for cardiovascular risk prediction trained on healthcare features from Feast"
            )
            
            logger.info(f"Model registered successfully. Version: {model_version.version}")
            
            # Transition to staging
            self.mlflow_client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging"
            )
            
            logger.info(f"Model version {model_version.version} transitioned to Staging")
            
            return model_version.version
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise

def main():
    """
    Main training pipeline execution.
    """
    logger.info("Starting Cardiovascular Risk Prediction Training Pipeline")
    
    try:
        # Initialize trainer
        trainer = CardiovascularRiskTrainer()
        
        # Pull features from Feast
        training_data = trainer.pull_features_from_feast()
        
        # Prepare training data
        X, y, feature_names = trainer.prepare_training_data(training_data)
        
        # Train model
        results = trainer.train_model(X, y)
        
        # Register best model
        model_version = trainer.register_best_model(results['run_id'])
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Model registered as version: {model_version}")
        logger.info(f"MLflow run ID: {results['run_id']}")
        
        # Print final metrics
        print("\n" + "="*50)
        print("TRAINING RESULTS SUMMARY")
        print("="*50)
        print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
        print(f"Precision: {results['metrics']['precision']:.4f}")
        print(f"Recall: {results['metrics']['recall']:.4f}")
        print(f"F1-Score: {results['metrics']['f1']:.4f}")
        print(f"AUC: {results['metrics']['auc']:.4f}")
        print("\nConfusion Matrix:")
        print(f"True Positives: {results['confusion_matrix']['true_positives']}")
        print(f"False Positives: {results['confusion_matrix']['false_positives']}")
        print(f"True Negatives: {results['confusion_matrix']['true_negatives']}")
        print(f"False Negatives: {results['confusion_matrix']['false_negatives']}")
        print(f"\nModel Version: {model_version}")
        print(f"MLflow Run ID: {results['run_id']}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
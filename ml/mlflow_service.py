import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
import asyncio
from threading import Lock

from config.settings import settings

class MLflowModelService:
    """Service for managing MLflow model loading and caching"""
    
    def __init__(self):
        self.model = None
        self.model_name = settings.default_model_name
        self.model_version = None
        self.model_stage = None
        self.model_loaded_at = None
        self.model_metadata = {}
        self._lock = Lock()
        
        # Model refresh settings
        self.refresh_interval = timedelta(minutes=30)  # Check for new models every 30 minutes
        self.auto_refresh = True
        
        # Initialize MLflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        
        # Load initial model
        asyncio.create_task(self._initialize_model())
    
    async def _initialize_model(self):
        """Initialize model loading on startup"""
        try:
            await self.load_latest_model()
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
    
    async def load_latest_model(self, force_refresh: bool = False) -> bool:
        """Load the latest model from MLflow model registry"""
        with self._lock:
            try:
                # Check if we need to refresh
                if not force_refresh and self._should_skip_refresh():
                    return True
                
                client = mlflow.tracking.MlflowClient()
                
                # Get latest versions from Production first, then Staging
                latest_versions = client.get_latest_versions(
                    self.model_name, 
                    stages=["Production", "Staging"]
                )
                
                if not latest_versions:
                    logger.warning(f"No model found in registry for {self.model_name}")
                    return False
                
                # Prefer Production over Staging
                selected_version = None
                for version in latest_versions:
                    if version.current_stage == "Production":
                        selected_version = version
                        break
                if not selected_version:
                    selected_version = latest_versions[0]
                
                # Check if this is a new version
                if (self.model_version == selected_version.version and 
                    self.model_stage == selected_version.current_stage and 
                    not force_refresh):
                    logger.debug(f"Model {self.model_name} v{self.model_version} already loaded")
                    return True
                
                # Load the model
                model_uri = f"models:/{self.model_name}/{selected_version.version}"
                logger.info(f"Loading model from {model_uri}")
                
                new_model = mlflow.xgboost.load_model(model_uri)
                
                # Get model metadata
                model_info = client.get_model_version(
                    self.model_name, 
                    selected_version.version
                )
                
                # Update model and metadata
                self.model = new_model
                self.model_version = selected_version.version
                self.model_stage = selected_version.current_stage
                self.model_loaded_at = datetime.now()
                self.model_metadata = {
                    "name": self.model_name,
                    "version": self.model_version,
                    "stage": self.model_stage,
                    "loaded_at": self.model_loaded_at.isoformat(),
                    "creation_timestamp": model_info.creation_timestamp,
                    "last_updated_timestamp": model_info.last_updated_timestamp,
                    "description": model_info.description or "",
                    "tags": dict(model_info.tags) if model_info.tags else {}
                }
                
                logger.success(
                    f"Successfully loaded model {self.model_name} "
                    f"v{self.model_version} ({self.model_stage})"
                )
                return True
                
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return False
    
    def _should_skip_refresh(self) -> bool:
        """Check if we should skip model refresh"""
        if not self.auto_refresh or not self.model_loaded_at:
            return False
        
        time_since_load = datetime.now() - self.model_loaded_at
        return time_since_load < self.refresh_interval
    
    async def get_model(self) -> Optional[Any]:
        """Get the current model, loading if necessary"""
        if self.model is None:
            await self.load_latest_model()
        
        # Auto-refresh check
        if self.auto_refresh and not self._should_skip_refresh():
            asyncio.create_task(self.load_latest_model())
        
        return self.model
    
    async def predict(self, features: Dict[str, Any]) -> Tuple[float, float]:
        """Make prediction and return both class prediction and probability"""
        model = await self.get_model()
        if model is None:
            raise ValueError("No model available for prediction")
        
        try:
            # Convert features to DataFrame
            df = pd.DataFrame([features])
            
            # Get both prediction and probability
            prediction_proba = model.predict_proba(df)[0]
            prediction_class = model.predict(df)[0]
            
            # For binary classification, return probability of positive class
            risk_probability = float(prediction_proba[1]) if len(prediction_proba) > 1 else float(prediction_proba[0])
            
            return float(prediction_class), risk_probability
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get current model information"""
        if self.model is None:
            await self.load_latest_model()
        
        return self.model_metadata.copy()
    
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models from MLflow registry"""
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
                        "created_at": version.creation_timestamp,
                        "last_updated": version.last_updated_timestamp,
                        "description": version.description or "",
                        "is_current": (
                            model.name == self.model_name and 
                            version.version == self.model_version
                        )
                    })
            
            return model_list
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def switch_model(self, model_name: str, version: str = None, stage: str = None) -> bool:
        """Switch to a different model version"""
        try:
            client = mlflow.tracking.MlflowClient()
            
            if version:
                model_uri = f"models:/{model_name}/{version}"
            elif stage:
                latest_versions = client.get_latest_versions(model_name, stages=[stage])
                if not latest_versions:
                    raise ValueError(f"No model found in stage {stage}")
                model_uri = f"models:/{model_name}/{latest_versions[0].version}"
            else:
                raise ValueError("Either version or stage must be specified")
            
            # Load the new model
            new_model = mlflow.xgboost.load_model(model_uri)
            
            # Update current model
            with self._lock:
                self.model = new_model
                self.model_name = model_name
                self.model_version = version or latest_versions[0].version
                self.model_stage = stage or client.get_model_version(model_name, self.model_version).current_stage
                self.model_loaded_at = datetime.now()
            
            logger.info(f"Switched to model {model_name} v{self.model_version}")
            return True
            
        except Exception as e:
            logger.error(f"Error switching model: {e}")
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status"""
        return {
            "model_loaded": self.model is not None,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_stage": self.model_stage,
            "loaded_at": self.model_loaded_at.isoformat() if self.model_loaded_at else None,
            "auto_refresh_enabled": self.auto_refresh,
            "refresh_interval_minutes": self.refresh_interval.total_seconds() / 60
        }

# Global instance
mlflow_service = MLflowModelService()
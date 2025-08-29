from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Application settings
    app_name: str = "ML Pipeline API"
    debug: bool = False
    
    # Database settings
    supabase_url: Optional[str] = None
    supabase_anon_key: Optional[str] = None
    database_url: Optional[str] = None
    
    # MLflow settings
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "ml_pipeline_experiment"
    
    # Prefect settings
    prefect_api_url: Optional[str] = None
    prefect_workspace: str = "default"
    
    # Feast settings
    feast_repo_path: str = "./feast_repo"
    
    # Model settings
    model_registry_path: str = "./models"
    default_model_name: str = "xgboost_model"
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Streamlit settings
    streamlit_port: int = 8501
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
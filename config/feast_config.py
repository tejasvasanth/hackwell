from feast import FeatureStore, Entity, FeatureView, Field, FeatureService
from feast.data_source import PostgreSQLSource
from feast.types import Float32, Int64, String, Bool
from datetime import timedelta, datetime
import pandas as pd
from typing import Dict, Any, List, Optional
from loguru import logger
import os
from config.settings import settings

class HealthcareFeastStore:
    """Feast Feature Store for Healthcare ML Pipeline using Supabase Postgres"""
    
    def __init__(self, repo_path: str = "./feast_repo"):
        self.repo_path = repo_path
        self.store = None
        self.postgres_config = {
            "host": settings.SUPABASE_HOST,
            "port": settings.SUPABASE_PORT or 5432,
            "database": settings.SUPABASE_DB_NAME,
            "user": settings.SUPABASE_USER,
            "password": settings.SUPABASE_PASSWORD,
        }
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize Feast feature store"""
        try:
            if os.path.exists(self.repo_path):
                self.store = FeatureStore(repo_path=self.repo_path)
                logger.info(f"Healthcare Feast store initialized from {self.repo_path}")
            else:
                logger.warning(f"Feast repo not found at {self.repo_path}")
        except Exception as e:
            logger.error(f"Error initializing Feast store: {e}")
    
    def create_feature_repo(self):
        """Create a new Feast feature repository with Supabase Postgres configuration"""
        try:
            os.makedirs(self.repo_path, exist_ok=True)
            
            # Create feature_store.yaml with Supabase Postgres configuration
            config_content = f"""
project: healthcare_ml_pipeline
registry:
  registry_type: sql
  path: postgresql://{self.postgres_config['user']}:{self.postgres_config['password']}@{self.postgres_config['host']}:{self.postgres_config['port']}/{self.postgres_config['database']}
provider: local
online_store:
  type: postgres
  host: {self.postgres_config['host']}
  port: {self.postgres_config['port']}
  database: {self.postgres_config['database']}
  db_schema: feast_online
  user: {self.postgres_config['user']}
  password: {self.postgres_config['password']}
offline_store:
  type: postgres
  host: {self.postgres_config['host']}
  port: {self.postgres_config['port']}
  database: {self.postgres_config['database']}
  db_schema: feast_offline
  user: {self.postgres_config['user']}
  password: {self.postgres_config['password']}
entity_key_serialization_version: 2
"""
            
            with open(os.path.join(self.repo_path, "feature_store.yaml"), "w") as f:
                f.write(config_content)
            
            logger.info(f"Healthcare Feast repository created at {self.repo_path}")
            
        except Exception as e:
            logger.error(f"Error creating Feast repo: {e}")
            raise
    
    def define_healthcare_entities_and_features(self):
        """Define healthcare entities and feature views"""
        try:
            # Define patient entity
            patient_entity = Entity(
                name="patient_id",
                description="Patient identifier",
                value_type=Int64
            )
            
            # Demographics Features
            demographics_source = PostgreSQLSource(
                name="demographics_source",
                query="SELECT patient_id, age, sex, bmi, height, weight, event_timestamp FROM demographics_features",
                timestamp_field="event_timestamp",
                **self.postgres_config
            )
            
            demographics_fv = FeatureView(
                name="demographics_features",
                entities=[patient_entity],
                ttl=timedelta(days=365),  # Demographics change slowly
                schema=[
                    Field(name="age", dtype=Int64),
                    Field(name="sex", dtype=String),  # 'M', 'F', 'Other'
                    Field(name="bmi", dtype=Float32),
                    Field(name="height", dtype=Float32),  # cm
                    Field(name="weight", dtype=Float32),  # kg
                ],
                source=demographics_source,
                tags={"category": "demographics", "team": "healthcare_ml"}
            )
            
            # Lab Results Features
            labs_source = PostgreSQLSource(
                name="labs_source",
                query="SELECT patient_id, cholesterol_total, cholesterol_ldl, cholesterol_hdl, hba1c, glucose_fasting, systolic_bp, diastolic_bp, creatinine, bun, event_timestamp FROM lab_features",
                timestamp_field="event_timestamp",
                **self.postgres_config
            )
            
            labs_fv = FeatureView(
                name="lab_features",
                entities=[patient_entity],
                ttl=timedelta(days=90),  # Lab results valid for 3 months
                schema=[
                    Field(name="cholesterol_total", dtype=Float32),  # mg/dL
                    Field(name="cholesterol_ldl", dtype=Float32),   # mg/dL
                    Field(name="cholesterol_hdl", dtype=Float32),   # mg/dL
                    Field(name="hba1c", dtype=Float32),             # %
                    Field(name="glucose_fasting", dtype=Float32),   # mg/dL
                    Field(name="systolic_bp", dtype=Int64),         # mmHg
                    Field(name="diastolic_bp", dtype=Int64),        # mmHg
                    Field(name="creatinine", dtype=Float32),        # mg/dL
                    Field(name="bun", dtype=Float32),               # mg/dL
                ],
                source=labs_source,
                tags={"category": "laboratory", "team": "healthcare_ml"}
            )
            
            # Lifestyle Features
            lifestyle_source = PostgreSQLSource(
                name="lifestyle_source",
                query="SELECT patient_id, smoking_status, steps_per_day, exercise_minutes_per_week, alcohol_drinks_per_week, sleep_hours_per_night, stress_level, event_timestamp FROM lifestyle_features",
                timestamp_field="event_timestamp",
                **self.postgres_config
            )
            
            lifestyle_fv = FeatureView(
                name="lifestyle_features",
                entities=[patient_entity],
                ttl=timedelta(days=30),  # Lifestyle can change monthly
                schema=[
                    Field(name="smoking_status", dtype=String),           # 'never', 'former', 'current'
                    Field(name="steps_per_day", dtype=Int64),             # average daily steps
                    Field(name="exercise_minutes_per_week", dtype=Int64), # minutes of exercise per week
                    Field(name="alcohol_drinks_per_week", dtype=Int64),   # drinks per week
                    Field(name="sleep_hours_per_night", dtype=Float32),   # average sleep hours
                    Field(name="stress_level", dtype=Int64),              # 1-10 scale
                ],
                source=lifestyle_source,
                tags={"category": "lifestyle", "team": "healthcare_ml"}
            )
            
            # Wearables Features
            wearables_source = PostgreSQLSource(
                name="wearables_source",
                query="SELECT patient_id, hrv_rmssd, hrv_sdnn, resting_heart_rate, max_heart_rate, ecg_qt_interval, ecg_pr_interval, ecg_qrs_duration, sleep_efficiency, deep_sleep_minutes, event_timestamp FROM wearables_features",
                timestamp_field="event_timestamp",
                **self.postgres_config
            )
            
            wearables_fv = FeatureView(
                name="wearables_features",
                entities=[patient_entity],
                ttl=timedelta(days=7),  # Wearable data changes weekly
                schema=[
                    Field(name="hrv_rmssd", dtype=Float32),        # ms - Heart Rate Variability
                    Field(name="hrv_sdnn", dtype=Float32),         # ms - Heart Rate Variability
                    Field(name="resting_heart_rate", dtype=Int64), # bpm
                    Field(name="max_heart_rate", dtype=Int64),     # bpm
                    Field(name="ecg_qt_interval", dtype=Float32),  # ms - ECG pattern
                    Field(name="ecg_pr_interval", dtype=Float32),  # ms - ECG pattern
                    Field(name="ecg_qrs_duration", dtype=Float32), # ms - ECG pattern
                    Field(name="sleep_efficiency", dtype=Float32), # % - sleep quality
                    Field(name="deep_sleep_minutes", dtype=Int64), # minutes of deep sleep
                ],
                source=wearables_source,
                tags={"category": "wearables", "team": "healthcare_ml"}
            )
            
            return {
                "entities": [patient_entity],
                "feature_views": [demographics_fv, labs_fv, lifestyle_fv, wearables_fv]
            }
            
        except Exception as e:
            logger.error(f"Error defining healthcare entities and features: {e}")
            raise
    
    def create_feature_services(self):
        """Create feature services for different ML use cases"""
        try:
            # Cardiovascular Risk Prediction Service
            cardio_risk_service = FeatureService(
                name="cardiovascular_risk_prediction",
                features=[
                    "demographics_features:age",
                    "demographics_features:sex",
                    "demographics_features:bmi",
                    "lab_features:cholesterol_total",
                    "lab_features:cholesterol_ldl",
                    "lab_features:cholesterol_hdl",
                    "lab_features:systolic_bp",
                    "lab_features:diastolic_bp",
                    "lifestyle_features:smoking_status",
                    "lifestyle_features:exercise_minutes_per_week",
                    "wearables_features:resting_heart_rate",
                    "wearables_features:hrv_rmssd"
                ],
                tags={"use_case": "cardiovascular_risk", "model_type": "classification"}
            )
            
            # Diabetes Risk Prediction Service
            diabetes_risk_service = FeatureService(
                name="diabetes_risk_prediction",
                features=[
                    "demographics_features:age",
                    "demographics_features:bmi",
                    "lab_features:hba1c",
                    "lab_features:glucose_fasting",
                    "lifestyle_features:exercise_minutes_per_week",
                    "lifestyle_features:steps_per_day",
                    "wearables_features:sleep_efficiency"
                ],
                tags={"use_case": "diabetes_risk", "model_type": "classification"}
            )
            
            # General Health Assessment Service
            health_assessment_service = FeatureService(
                name="general_health_assessment",
                features=[
                    "demographics_features",
                    "lab_features",
                    "lifestyle_features",
                    "wearables_features"
                ],
                tags={"use_case": "general_health", "model_type": "multi_output"}
            )
            
            return [cardio_risk_service, diabetes_risk_service, health_assessment_service]
            
        except Exception as e:
            logger.error(f"Error creating feature services: {e}")
            raise
    
    async def get_online_features_for_inference(self, patient_ids: List[int], 
                                              feature_service_name: str) -> pd.DataFrame:
        """Get online features for real-time inference"""
        try:
            if not self.store:
                raise ValueError("Feast store not initialized")
            
            entity_rows = [{"patient_id": pid} for pid in patient_ids]
            
            feature_vector = self.store.get_online_features(
                features=self.store.get_feature_service(feature_service_name),
                entity_rows=entity_rows
            )
            
            features_df = feature_vector.to_df()
            logger.info(f"Retrieved online features for {len(patient_ids)} patients")
            return features_df
            
        except Exception as e:
            logger.error(f"Error getting online features: {e}")
            raise
    
    async def get_historical_features_for_training(self, entity_df: pd.DataFrame, 
                                                 feature_service_name: str) -> pd.DataFrame:
        """Get historical features for model training"""
        try:
            if not self.store:
                raise ValueError("Feast store not initialized")
            
            # Ensure entity_df has required columns
            if "patient_id" not in entity_df.columns or "event_timestamp" not in entity_df.columns:
                raise ValueError("entity_df must contain 'patient_id' and 'event_timestamp' columns")
            
            training_df = self.store.get_historical_features(
                entity_df=entity_df,
                features=self.store.get_feature_service(feature_service_name)
            ).to_df()
            
            logger.info(f"Retrieved historical features for {len(entity_df)} training examples")
            return training_df
            
        except Exception as e:
            logger.error(f"Error getting historical features: {e}")
            raise
    
    def apply_feature_definitions(self):
        """Apply all feature definitions to the feature store"""
        try:
            if not self.store:
                raise ValueError("Feast store not initialized")
            
            # Get feature definitions
            definitions = self.define_healthcare_entities_and_features()
            feature_services = self.create_feature_services()
            
            # Apply to store
            all_objects = definitions["entities"] + definitions["feature_views"] + feature_services
            self.store.apply(all_objects)
            
            logger.info("Healthcare feature definitions applied successfully")
            
        except Exception as e:
            logger.error(f"Error applying feature definitions: {e}")
            raise
    
    def materialize_features(self, start_date: str, end_date: str):
        """Materialize features to online store for serving"""
        try:
            if not self.store:
                raise ValueError("Feast store not initialized")
            
            from datetime import datetime
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            
            self.store.materialize(start_date=start_dt, end_date=end_dt)
            logger.info(f"Features materialized from {start_date} to {end_date}")
            
        except Exception as e:
            logger.error(f"Error materializing features: {e}")
            raise
    
    def create_sample_healthcare_tables(self):
        """Create sample tables in Supabase for healthcare features"""
        sql_commands = [
            """
            CREATE TABLE IF NOT EXISTS demographics_features (
                patient_id INTEGER,
                age INTEGER,
                sex VARCHAR(10),
                bmi FLOAT,
                height FLOAT,
                weight FLOAT,
                event_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS lab_features (
                patient_id INTEGER,
                cholesterol_total FLOAT,
                cholesterol_ldl FLOAT,
                cholesterol_hdl FLOAT,
                hba1c FLOAT,
                glucose_fasting FLOAT,
                systolic_bp INTEGER,
                diastolic_bp INTEGER,
                creatinine FLOAT,
                bun FLOAT,
                event_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS lifestyle_features (
                patient_id INTEGER,
                smoking_status VARCHAR(20),
                steps_per_day INTEGER,
                exercise_minutes_per_week INTEGER,
                alcohol_drinks_per_week INTEGER,
                sleep_hours_per_night FLOAT,
                stress_level INTEGER,
                event_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS wearables_features (
                patient_id INTEGER,
                hrv_rmssd FLOAT,
                hrv_sdnn FLOAT,
                resting_heart_rate INTEGER,
                max_heart_rate INTEGER,
                ecg_qt_interval FLOAT,
                ecg_pr_interval FLOAT,
                ecg_qrs_duration FLOAT,
                sleep_efficiency FLOAT,
                deep_sleep_minutes INTEGER,
                event_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """
        ]
        
        return sql_commands

# Global instance
healthcare_feast_store = HealthcareFeastStore()
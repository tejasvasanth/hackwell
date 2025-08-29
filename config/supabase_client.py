import os
import asyncio
import time
from typing import Dict, List, Optional, Any
from supabase import create_client, Client
from datetime import datetime
import pandas as pd
from loguru import logger
from pathlib import Path

from config.settings import settings

def retry_with_exponential_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Function {func.__name__} failed after {max_retries} attempts: {e}")
                        raise
                    
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

class SupabaseClient:
    """Supabase client for healthcare data operations and file storage."""
    
    def __init__(self):
        self.client: Optional[Client] = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Supabase client"""
        try:
            if settings.supabase_url and settings.supabase_anon_key:
                self.client = create_client(
                    settings.supabase_url,
                    settings.supabase_anon_key
                )
                logger.info("Supabase client initialized successfully")
            else:
                logger.warning("Supabase credentials not found. Client not initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
    
    @retry_with_exponential_backoff(max_retries=3)
    def get_patient_data(self, patient_id: int) -> dict:
        """Fetch patient demographics, lifestyle, and lab values from patients table."""
        try:
            result = self.client.table("patients").select(
                "id, name, age, gender, smoking, exercise, bp, cholesterol, diabetes_status"
            ).eq("id", patient_id).execute()
            
            if not result.data:
                raise ValueError(f"Patient with ID {patient_id} not found")
            
            patient_data = result.data[0]
            logger.info(f"Retrieved patient data for ID: {patient_id}")
            return patient_data
        except Exception as e:
            logger.error(f"Error retrieving patient data for ID {patient_id}: {e}")
            raise

    @retry_with_exponential_backoff(max_retries=3)
    def upload_ecg(self, patient_id: int, file_path: str) -> str:
        """Upload an ECG file to Supabase Storage bucket 'ecg_files' and return public URL."""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"ECG file not found: {file_path}")
            
            # Read file data
            with open(file_path, 'rb') as file:
                file_data = file.read()
            
            # Generate unique filename
            file_extension = os.path.splitext(file_path)[1]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            storage_path = f"patient_{patient_id}/ecg_{timestamp}{file_extension}"
            
            # Upload to Supabase storage
            result = self.client.storage.from_("ecg_files").upload(storage_path, file_data)
            
            if hasattr(result, 'error') and result.error:
                raise Exception(f"Upload error: {result.error}")
            
            # Get public URL
            public_url = self.client.storage.from_("ecg_files").get_public_url(storage_path)
            
            logger.info(f"ECG file uploaded for patient {patient_id}: {public_url}")
            return public_url
        except Exception as e:
            logger.error(f"Error uploading ECG file for patient {patient_id}: {e}")
            raise

    @retry_with_exponential_backoff(max_retries=3)
    def log_prediction(self, patient_id: int, prediction: str, probability: float, timestamp: datetime) -> dict:
        """Insert prediction results into the predictions table."""
        try:
            prediction_data = {
                "patient_id": patient_id,
                "prediction": prediction,
                "probability": probability,
                "timestamp": timestamp.isoformat()
            }
            
            result = self.client.table("predictions").insert(prediction_data).execute()
            
            if not result.data:
                raise Exception("Failed to insert prediction data")
            
            inserted_record = result.data[0]
            logger.info(f"Prediction logged for patient {patient_id}: {prediction} ({probability:.3f})")
            return inserted_record
        except Exception as e:
            logger.error(f"Error logging prediction for patient {patient_id}: {e}")
            raise

    async def store_prediction(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store prediction results in Supabase"""
        if not self.client:
            raise ValueError("Supabase client not initialized")
        
        try:
            # Add timestamp if not present
            if 'created_at' not in prediction_data:
                prediction_data['created_at'] = datetime.now().isoformat()
            
            result = self.client.table('predictions').insert(prediction_data).execute()
            logger.info(f"Prediction stored with ID: {result.data[0]['id']}")
            return result.data[0]
        
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
            raise
    
    async def get_predictions(self, limit: int = 100, 
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve predictions from Supabase"""
        if not self.client:
            raise ValueError("Supabase client not initialized")
        
        try:
            query = self.client.table('predictions').select('*')
            
            if start_date:
                query = query.gte('created_at', start_date)
            if end_date:
                query = query.lte('created_at', end_date)
            
            result = query.limit(limit).order('created_at', desc=True).execute()
            return result.data
        
        except Exception as e:
            logger.error(f"Error retrieving predictions: {e}")
            raise
    
    async def store_model_metrics(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store model performance metrics"""
        if not self.client:
            raise ValueError("Supabase client not initialized")
        
        try:
            if 'created_at' not in metrics_data:
                metrics_data['created_at'] = datetime.now().isoformat()
            
            result = self.client.table('model_metrics').insert(metrics_data).execute()
            logger.info(f"Model metrics stored with ID: {result.data[0]['id']}")
            return result.data[0]
        
        except Exception as e:
            logger.error(f"Error storing model metrics: {e}")
            raise
    
    async def get_model_metrics(self, model_name: Optional[str] = None,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve model performance metrics"""
        if not self.client:
            raise ValueError("Supabase client not initialized")
        
        try:
            query = self.client.table('model_metrics').select('*')
            
            if model_name:
                query = query.eq('model_name', model_name)
            
            result = query.limit(limit).order('created_at', desc=True).execute()
            return result.data
        
        except Exception as e:
            logger.error(f"Error retrieving model metrics: {e}")
            raise
    
    async def store_training_data(self, data: pd.DataFrame, 
                                table_name: str = 'training_data') -> Dict[str, Any]:
        """Store training data in Supabase"""
        if not self.client:
            raise ValueError("Supabase client not initialized")
        
        try:
            # Convert DataFrame to list of dictionaries
            records = data.to_dict('records')
            
            # Add timestamps
            for record in records:
                record['created_at'] = datetime.now().isoformat()
            
            result = self.client.table(table_name).insert(records).execute()
            logger.info(f"Stored {len(records)} training records")
            return {"records_stored": len(records), "table": table_name}
        
        except Exception as e:
            logger.error(f"Error storing training data: {e}")
            raise
    
    async def get_training_data(self, table_name: str = 'training_data',
                              limit: int = 1000) -> pd.DataFrame:
        """Retrieve training data from Supabase"""
        if not self.client:
            raise ValueError("Supabase client not initialized")
        
        try:
            result = self.client.table(table_name).select('*').limit(limit).execute()
            
            if result.data:
                df = pd.DataFrame(result.data)
                # Remove metadata columns if present
                metadata_cols = ['id', 'created_at', 'updated_at']
                df = df.drop(columns=[col for col in metadata_cols if col in df.columns])
                return df
            else:
                return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error retrieving training data: {e}")
            raise
    
    async def store_feature_store_data(self, features: Dict[str, Any],
                                     entity_id: str) -> Dict[str, Any]:
        """Store feature store data"""
        if not self.client:
            raise ValueError("Supabase client not initialized")
        
        try:
            feature_data = {
                'entity_id': entity_id,
                'features': features,
                'created_at': datetime.now().isoformat()
            }
            
            result = self.client.table('feature_store').insert(feature_data).execute()
            logger.info(f"Feature data stored for entity: {entity_id}")
            return result.data[0]
        
        except Exception as e:
            logger.error(f"Error storing feature data: {e}")
            raise
    
    async def get_feature_store_data(self, entity_id: str) -> Dict[str, Any]:
        """Retrieve feature store data for an entity"""
        if not self.client:
            raise ValueError("Supabase client not initialized")
        
        try:
            result = self.client.table('feature_store').select('*').eq(
                'entity_id', entity_id
            ).order('created_at', desc=True).limit(1).execute()
            
            if result.data:
                return result.data[0]['features']
            else:
                return {}
        
        except Exception as e:
            logger.error(f"Error retrieving feature data: {e}")
            raise
    
    async def upload_file(self, file_path: str, bucket_name: str = 'models',
                        destination_path: Optional[str] = None) -> Dict[str, Any]:
        """Upload file to Supabase Storage"""
        if not self.client:
            raise ValueError("Supabase client not initialized")
        
        try:
            if not destination_path:
                destination_path = os.path.basename(file_path)
            
            with open(file_path, 'rb') as file:
                result = self.client.storage.from_(bucket_name).upload(
                    destination_path, file
                )
            
            logger.info(f"File uploaded to {bucket_name}/{destination_path}")
            return {"bucket": bucket_name, "path": destination_path}
        
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            raise
    
    async def download_file(self, file_path: str, bucket_name: str = 'models',
                          local_path: Optional[str] = None) -> str:
        """Download file from Supabase Storage"""
        if not self.client:
            raise ValueError("Supabase client not initialized")
        
        try:
            if not local_path:
                local_path = os.path.basename(file_path)
            
            result = self.client.storage.from_(bucket_name).download(file_path)
            
            with open(local_path, 'wb') as file:
                file.write(result)
            
            logger.info(f"File downloaded to {local_path}")
            return local_path
        
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            raise
    
    async def create_tables(self):
        """Create necessary tables (run this once during setup)"""
        if not self.client:
            raise ValueError("Supabase client not initialized")
        
        # Note: In practice, you would create these tables using Supabase SQL editor
        # or migration scripts. This is just for reference.
        
        table_schemas = {
            'patients': """
                CREATE TABLE IF NOT EXISTS patients (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    age INTEGER,
                    gender VARCHAR(10),
                    smoking BOOLEAN DEFAULT FALSE,
                    exercise VARCHAR(50),
                    bp VARCHAR(20),
                    cholesterol FLOAT,
                    diabetes_status VARCHAR(20),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """,
            'predictions': """
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    patient_id INTEGER REFERENCES patients(id),
                    prediction VARCHAR(255) NOT NULL,
                    probability FLOAT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    model_name VARCHAR(255),
                    model_version VARCHAR(255),
                    input_features JSONB,
                    explanation JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """,
            'model_metrics': """
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(255),
                    model_version VARCHAR(255),
                    metric_name VARCHAR(255),
                    metric_value FLOAT,
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """,
            'feature_store': """
                CREATE TABLE IF NOT EXISTS feature_store (
                    id SERIAL PRIMARY KEY,
                    entity_id VARCHAR(255),
                    features JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """
        }
        
        logger.info("Table schemas defined. Please create these tables in Supabase SQL editor.")
        return table_schemas

# Global instance
supabase_client = SupabaseClient()
from supabase import create_client, Client
from config.settings import settings
import logging
from typing import Optional, Dict, Any, List, Union
from functools import wraps
import time
from datetime import datetime, date, timedelta
import pandas as pd

logger = logging.getLogger(__name__)

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry Supabase operations on failure"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to execute {func.__name__} after {max_retries} attempts: {e}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying...")
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
            return None
        return wrapper
    return decorator

class SupabaseClient:
    def __init__(self):
        """Initialize Supabase client"""
        if not settings.supabase_url or not settings.supabase_anon_key:
            raise ValueError("Supabase URL and key must be provided")
        
        self.client: Client = create_client(settings.supabase_url, settings.supabase_anon_key)
        logger.info("Supabase client initialized successfully")
    
    @retry_on_failure()
    def test_connection(self) -> bool:
        """Test the Supabase connection"""
        try:
            # Try a simple query to test connection
            result = self.client.table('patients').select('count').execute()
            logger.info("Supabase connection test successful")
            return True
        except Exception as e:
            logger.error(f"Supabase connection test failed: {e}")
            return False
    
    # PATIENT OPERATIONS
    @retry_on_failure()
    def get_patients(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all patients or limited number"""
        query = self.client.table('patients').select('*')
        if limit:
            query = query.limit(limit)
        result = query.execute()
        return result.data
    
    @retry_on_failure()
    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific patient by patient_id"""
        result = self.client.table('patients').select('*').eq('patient_id', patient_id).execute()
        return result.data[0] if result.data else None
    
    @retry_on_failure()
    def create_patient(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new patient"""
        result = self.client.table('patients').insert(patient_data).execute()
        return result.data[0] if result.data else None
    
    @retry_on_failure()
    def update_patient(self, patient_id: str, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update patient information"""
        result = self.client.table('patients').update(patient_data).eq('patient_id', patient_id).execute()
        return result.data[0] if result.data else None
    
    # LAB RESULTS OPERATIONS
    @retry_on_failure()
    def get_lab_results(self, patient_id: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get lab results, optionally filtered by patient"""
        query = self.client.table('lab_results').select('*')
        if patient_id:
            query = query.eq('patient_id', patient_id)
        if limit:
            query = query.limit(limit)
        query = query.order('test_date', desc=True)
        result = query.execute()
        return result.data
    
    @retry_on_failure()
    def create_lab_result(self, lab_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new lab result"""
        result = self.client.table('lab_results').insert(lab_data).execute()
        return result.data[0] if result.data else None
    
    @retry_on_failure()
    def get_latest_lab_results(self, patient_id: str, test_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get latest lab results for a patient, optionally filtered by test types"""
        query = self.client.table('lab_results').select('*').eq('patient_id', patient_id)
        if test_types:
            query = query.in_('test_type', test_types)
        query = query.order('test_date', desc=True).limit(10)
        result = query.execute()
        return result.data
    
    # LIFESTYLE DATA OPERATIONS
    @retry_on_failure()
    def get_lifestyle_data(self, patient_id: Optional[str] = None, days: Optional[int] = 30) -> List[Dict[str, Any]]:
        """Get lifestyle data, optionally filtered by patient and date range"""
        query = self.client.table('lifestyle_data').select('*')
        if patient_id:
            query = query.eq('patient_id', patient_id)
        if days:
            start_date = (datetime.now() - timedelta(days=days)).date()
            query = query.gte('date', start_date.isoformat())
        query = query.order('date', desc=True)
        result = query.execute()
        return result.data
    
    @retry_on_failure()
    def create_lifestyle_data(self, lifestyle_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new lifestyle data entry"""
        result = self.client.table('lifestyle_data').insert(lifestyle_data).execute()
        return result.data[0] if result.data else None
    
    @retry_on_failure()
    def bulk_insert_lifestyle_data(self, lifestyle_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Bulk insert lifestyle data"""
        result = self.client.table('lifestyle_data').insert(lifestyle_data_list).execute()
        return result.data
    
    # PREDICTIONS OPERATIONS
    @retry_on_failure()
    def get_predictions(self, patient_id: Optional[str] = None, prediction_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get predictions, optionally filtered by patient and type"""
        query = self.client.table('predictions').select('*')
        if patient_id:
            query = query.eq('patient_id', patient_id)
        if prediction_type:
            query = query.eq('prediction_type', prediction_type)
        query = query.order('prediction_date', desc=True)
        result = query.execute()
        return result.data
    
    @retry_on_failure()
    def create_prediction(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new prediction"""
        result = self.client.table('predictions').insert(prediction_data).execute()
        return result.data[0] if result.data else None
    
    @retry_on_failure()
    def get_latest_predictions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get latest predictions across all patients"""
        result = self.client.table('predictions').select('*').order('prediction_date', desc=True).limit(limit).execute()
        return result.data
    
    @retry_on_failure()
    def get_high_risk_patients(self, risk_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Get patients with high risk predictions"""
        result = self.client.table('predictions').select('*, patients(name, age, gender)').gte('probability', risk_threshold).order('probability', desc=True).execute()
        return result.data
    
    # MODEL PERFORMANCE OPERATIONS
    @retry_on_failure()
    def get_model_performance(self, model_name: Optional[str] = None, model_version: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get model performance metrics"""
        query = self.client.table('model_performance').select('*')
        if model_name:
            query = query.eq('model_name', model_name)
        if model_version:
            query = query.eq('model_version', model_version)
        query = query.order('evaluation_date', desc=True)
        result = query.execute()
        return result.data
    
    @retry_on_failure()
    def create_model_performance(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create model performance record"""
        result = self.client.table('model_performance').insert(performance_data).execute()
        return result.data[0] if result.data else None
    
    # DATA QUALITY OPERATIONS
    @retry_on_failure()
    def get_data_quality_metrics(self, table_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get data quality metrics"""
        query = self.client.table('data_quality_metrics').select('*')
        if table_name:
            query = query.eq('table_name', table_name)
        query = query.order('check_date', desc=True)
        result = query.execute()
        return result.data
    
    @retry_on_failure()
    def create_data_quality_metric(self, quality_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create data quality metric record"""
        result = self.client.table('data_quality_metrics').insert(quality_data).execute()
        return result.data[0] if result.data else None
    
    # FEATURE STORE OPERATIONS
    @retry_on_failure()
    def get_features(self, patient_id: str, feature_group: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get features for a patient"""
        query = self.client.table('feature_store').select('*').eq('patient_id', patient_id)
        if feature_group:
            query = query.eq('feature_group', feature_group)
        query = query.order('feature_timestamp', desc=True)
        result = query.execute()
        return result.data
    
    @retry_on_failure()
    def create_features(self, features_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Bulk insert features"""
        result = self.client.table('feature_store').insert(features_data).execute()
        return result.data
    
    # ALERTS OPERATIONS
    @retry_on_failure()
    def get_alerts(self, patient_id: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get alerts, optionally filtered by patient and status"""
        query = self.client.table('alerts').select('*')
        if patient_id:
            query = query.eq('patient_id', patient_id)
        if status:
            query = query.eq('status', status)
        query = query.order('triggered_at', desc=True)
        result = query.execute()
        return result.data
    
    @retry_on_failure()
    def create_alert(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new alert"""
        result = self.client.table('alerts').insert(alert_data).execute()
        return result.data[0] if result.data else None
    
    @retry_on_failure()
    def update_alert_status(self, alert_id: str, status: str, timestamp_field: Optional[str] = None) -> Dict[str, Any]:
        """Update alert status"""
        update_data = {'status': status}
        if timestamp_field:
            update_data[timestamp_field] = datetime.now().isoformat()
        result = self.client.table('alerts').update(update_data).eq('id', alert_id).execute()
        return result.data[0] if result.data else None
    
    # ANALYTICS AND AGGREGATION METHODS
    @retry_on_failure()
    def get_patient_summary(self, patient_id: str) -> Dict[str, Any]:
        """Get comprehensive patient summary"""
        # Get patient info
        patient = self.get_patient(patient_id)
        if not patient:
            return {}
        
        # Get latest lab results
        lab_results = self.get_latest_lab_results(patient_id, limit=5)
        
        # Get recent lifestyle data
        lifestyle_data = self.get_lifestyle_data(patient_id, days=7)
        
        # Get latest prediction
        predictions = self.get_predictions(patient_id, limit=1)
        
        # Get active alerts
        alerts = self.get_alerts(patient_id, status='active')
        
        return {
            'patient': patient,
            'latest_labs': lab_results,
            'recent_lifestyle': lifestyle_data,
            'latest_prediction': predictions[0] if predictions else None,
            'active_alerts': alerts
        }
    
    @retry_on_failure()
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for main dashboard"""
        # Get high-risk patients
        high_risk_patients = self.get_high_risk_patients(risk_threshold=0.7)
        
        # Get recent predictions
        recent_predictions = self.get_latest_predictions(limit=20)
        
        # Get active alerts
        active_alerts = self.get_alerts(status='active')
        
        # Get model performance
        model_performance = self.get_model_performance(limit=10)
        
        # Get data quality metrics
        data_quality = self.get_data_quality_metrics()
        
        return {
            'high_risk_patients': high_risk_patients,
            'recent_predictions': recent_predictions,
            'active_alerts': active_alerts,
            'model_performance': model_performance,
            'data_quality': data_quality
        }
    
    @retry_on_failure()
    def get_ml_training_data(self, days_back: int = 180) -> pd.DataFrame:
        """Get data for ML model training"""
        # Calculate date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        # Get patients
        patients = self.get_patients()
        
        # Get lab results in date range
        lab_results = self.client.table('lab_results').select('*').gte('test_date', start_date.isoformat()).lte('test_date', end_date.isoformat()).execute()
        
        # Get lifestyle data in date range
        lifestyle_data = self.client.table('lifestyle_data').select('*').gte('date', start_date.isoformat()).lte('date', end_date.isoformat()).execute()
        
        # Convert to DataFrames for easier manipulation
        patients_df = pd.DataFrame(patients)
        lab_results_df = pd.DataFrame(lab_results.data)
        lifestyle_df = pd.DataFrame(lifestyle_data.data)
        
        return {
            'patients': patients_df,
            'lab_results': lab_results_df,
            'lifestyle_data': lifestyle_df
        }

# Global instance
supabase_client = SupabaseClient()

def get_supabase_client() -> SupabaseClient:
    """Get the global Supabase client instance"""
    return supabase_client
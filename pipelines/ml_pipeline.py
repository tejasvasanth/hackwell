from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
import asyncio

from ml.model import MLModel
from ml.explainer import ModelExplainer
from config.settings import settings
from config.supabase_client import supabase_client
from config.feast_config import feast_store

# Healthcare ETL Pipeline Tasks

# Legacy ML Pipeline Tasks (keeping for backward compatibility)

@task(name="extract_patient_data")
def extract_patient_data(patient_ids: Optional[List[int]] = None, limit: int = 1000) -> pd.DataFrame:
    """Extract raw patient and lab data from Supabase"""
    logger.info(f"Extracting patient data from Supabase (limit: {limit})")
    
    try:
        if not supabase_client.client:
            raise ValueError("Supabase client not initialized")
        
        # Extract patient demographics and lab data
        if patient_ids:
            # Extract specific patients
            patients_data = []
            for patient_id in patient_ids:
                patient = supabase_client.get_patient_data(patient_id)
                patients_data.append(patient)
            df = pd.DataFrame(patients_data)
        else:
            # Extract all patients with limit
            result = supabase_client.client.table("patients").select(
                "id, name, age, gender, smoking, exercise, bp, cholesterol, diabetes_status, created_at"
            ).limit(limit).execute()
            
            if not result.data:
                logger.warning("No patient data found")
                return pd.DataFrame()
            
            df = pd.DataFrame(result.data)
        
        # Extract lab results if available
        try:
            lab_result = supabase_client.client.table("lab_results").select(
                "patient_id, test_type, value, unit, reference_range, test_date"
            ).execute()
            
            if lab_result.data:
                lab_df = pd.DataFrame(lab_result.data)
                # Pivot lab results to create features
                lab_pivot = lab_df.pivot_table(
                    index='patient_id', 
                    columns='test_type', 
                    values='value', 
                    aggfunc='mean'
                ).reset_index()
                
                # Merge with patient data
                df = df.merge(lab_pivot, left_on='id', right_on='patient_id', how='left')
        except Exception as e:
            logger.warning(f"Could not extract lab results: {e}")
        
        logger.info(f"Extracted {len(df)} patient records with {len(df.columns)} features")
        return df
        
    except Exception as e:
        logger.error(f"Error extracting patient data: {e}")
        raise

@task(name="extract_lifestyle_data")
def extract_lifestyle_data(patient_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """Extract lifestyle and wearable data from Supabase"""
    logger.info("Extracting lifestyle and wearable data")
    
    try:
        if not supabase_client.client:
            raise ValueError("Supabase client not initialized")
        
        # Extract lifestyle data
        query = supabase_client.client.table("lifestyle_data").select(
            "patient_id, steps_daily, sleep_hours, heart_rate_avg, activity_level, stress_level, recorded_date"
        )
        
        if patient_ids:
            query = query.in_("patient_id", patient_ids)
        
        result = query.execute()
        
        if not result.data:
            logger.warning("No lifestyle data found")
            return pd.DataFrame()
        
        df = pd.DataFrame(result.data)
        
        # Aggregate lifestyle data by patient (last 30 days average)
        df['recorded_date'] = pd.to_datetime(df['recorded_date'])
        recent_data = df[df['recorded_date'] >= (datetime.now() - timedelta(days=30))]
        
        lifestyle_agg = recent_data.groupby('patient_id').agg({
            'steps_daily': 'mean',
            'sleep_hours': 'mean', 
            'heart_rate_avg': 'mean',
            'activity_level': 'mean',
            'stress_level': 'mean'
        }).reset_index()
        
        logger.info(f"Extracted lifestyle data for {len(lifestyle_agg)} patients")
        return lifestyle_agg
        
    except Exception as e:
        logger.error(f"Error extracting lifestyle data: {e}")
        return pd.DataFrame()

@task(name="transform_ml_features")
def transform_ml_features(patient_df: pd.DataFrame, lifestyle_df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw data into ML-ready features"""
    logger.info("Transforming data into ML-ready features")
    
    try:
        # Merge patient and lifestyle data
        if not lifestyle_df.empty:
            ml_df = patient_df.merge(lifestyle_df, left_on='id', right_on='patient_id', how='left')
        else:
            ml_df = patient_df.copy()
        
        # Feature engineering
        # 1. Encode categorical variables
        ml_df['gender_encoded'] = ml_df['gender'].map({'M': 1, 'F': 0})
        ml_df['smoking_encoded'] = ml_df['smoking'].map({'yes': 1, 'no': 0})
        ml_df['exercise_encoded'] = ml_df['exercise'].map({'regular': 2, 'occasional': 1, 'none': 0})
        ml_df['diabetes_encoded'] = ml_df['diabetes_status'].map({'yes': 1, 'no': 0})
        
        # 2. Create risk scores
        ml_df['cardiovascular_risk'] = (
            (ml_df['age'] > 50).astype(int) * 0.3 +
            (ml_df['bp'] > 140).astype(int) * 0.4 +
            (ml_df['cholesterol'] > 200).astype(int) * 0.3
        )
        
        # 3. Create BMI categories if height/weight available
        if 'height' in ml_df.columns and 'weight' in ml_df.columns:
            ml_df['bmi'] = ml_df['weight'] / (ml_df['height'] / 100) ** 2
            ml_df['bmi_category'] = pd.cut(ml_df['bmi'], 
                                         bins=[0, 18.5, 25, 30, float('inf')], 
                                         labels=[0, 1, 2, 3])
        
        # 4. Handle missing values
        numeric_columns = ml_df.select_dtypes(include=[np.number]).columns
        ml_df[numeric_columns] = ml_df[numeric_columns].fillna(ml_df[numeric_columns].median())
        
        # 5. Add timestamp for Feast
        ml_df['event_timestamp'] = datetime.now()
        
        # Select final feature set
        feature_columns = [
            'id', 'age', 'gender_encoded', 'smoking_encoded', 'exercise_encoded', 
            'bp', 'cholesterol', 'diabetes_encoded', 'cardiovascular_risk',
            'event_timestamp'
        ]
        
        # Add lifestyle features if available
        lifestyle_features = ['steps_daily', 'sleep_hours', 'heart_rate_avg', 'activity_level', 'stress_level']
        available_lifestyle = [col for col in lifestyle_features if col in ml_df.columns]
        feature_columns.extend(available_lifestyle)
        
        # Add lab features if available
        lab_features = [col for col in ml_df.columns if col not in feature_columns and col not in ['name', 'gender', 'smoking', 'exercise', 'diabetes_status', 'created_at', 'patient_id']]
        feature_columns.extend(lab_features)
        
        final_df = ml_df[feature_columns].copy()
        
        logger.info(f"Transformed data: {len(final_df)} samples with {len(feature_columns)} features")
        return final_df
        
    except Exception as e:
        logger.error(f"Error transforming features: {e}")
        raise

@task(name="load_features_to_feast")
async def load_features_to_feast(features_df: pd.DataFrame) -> Dict[str, Any]:
    """Load transformed features into Feast feature store"""
    logger.info("Loading features into Feast feature store")
    
    try:
        if features_df.empty:
            logger.warning("No features to load")
            return {"status": "skipped", "reason": "empty_dataframe"}
        
        # Prepare data for Feast
        feast_df = features_df.copy()
        feast_df['patient_id'] = feast_df['id'].astype(str)
        
        # Split into different feature categories for Feast
        demographics_df = feast_df[[
            'patient_id', 'age', 'gender_encoded', 'event_timestamp'
        ]].copy()
        
        lifestyle_df = feast_df[[
            'patient_id', 'smoking_encoded', 'exercise_encoded', 'event_timestamp'
        ]].copy()
        
        # Add lifestyle features if available
        lifestyle_cols = ['steps_daily', 'sleep_hours', 'heart_rate_avg', 'activity_level', 'stress_level']
        for col in lifestyle_cols:
            if col in feast_df.columns:
                lifestyle_df[col] = feast_df[col]
        
        labs_df = feast_df[[
            'patient_id', 'bp', 'cholesterol', 'diabetes_encoded', 'cardiovascular_risk', 'event_timestamp'
        ]].copy()
        
        # Load to Feast (using push API for real-time features)
        results = {}
        
        try:
            # Push demographics features
            await feast_store.push_features("demographics_push_source", demographics_df)
            results['demographics'] = "success"
            logger.info(f"Loaded {len(demographics_df)} demographics records to Feast")
        except Exception as e:
            logger.error(f"Error loading demographics to Feast: {e}")
            results['demographics'] = f"error: {e}"
        
        try:
            # Push lifestyle features  
            await feast_store.push_features("lifestyle_push_source", lifestyle_df)
            results['lifestyle'] = "success"
            logger.info(f"Loaded {len(lifestyle_df)} lifestyle records to Feast")
        except Exception as e:
            logger.error(f"Error loading lifestyle to Feast: {e}")
            results['lifestyle'] = f"error: {e}"
        
        try:
            # Push lab features
            await feast_store.push_features("labs_push_source", labs_df)
            results['labs'] = "success"
            logger.info(f"Loaded {len(labs_df)} lab records to Feast")
        except Exception as e:
            logger.error(f"Error loading labs to Feast: {e}")
            results['labs'] = f"error: {e}"
        
        return {
            "status": "completed",
            "records_processed": len(features_df),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error loading features to Feast: {e}")
        raise

@task(name="trigger_model_training")
async def trigger_model_training(feature_load_result: Dict[str, Any]) -> Dict[str, Any]:
    """Trigger ML model training job using the loaded features"""
    logger.info("Triggering model training job")
    
    try:
        if feature_load_result["status"] != "completed":
            logger.warning("Skipping model training due to feature loading issues")
            return {"status": "skipped", "reason": "feature_loading_failed"}
        
        # Get features from Feast for training
        entity_rows = [{"patient_id": str(i)} for i in range(1, 101)]  # Sample patients
        
        features = [
            "demographics:age",
            "demographics:gender_encoded", 
            "lifestyle:smoking_encoded",
            "lifestyle:exercise_encoded",
            "labs:bp",
            "labs:cholesterol",
            "labs:diabetes_encoded",
            "labs:cardiovascular_risk"
        ]
        
        # Retrieve features for training
        training_df = await feast_store.get_online_features(entity_rows, features)
        
        if training_df.empty:
            logger.warning("No training data available from Feast")
            return {"status": "failed", "reason": "no_training_data"}
        
        # Prepare training data
        X = training_df.drop(['patient_id'], axis=1)
        # Create synthetic target for demonstration (in real scenario, this would come from labels)
        y = (training_df['labs__cardiovascular_risk'] > 0.5).astype(int)
        
        # Initialize and train model
        ml_model = MLModel()
        
        # Train model with MLflow tracking
        run_id = await ml_model.train(
            X, y,
            experiment_name="healthcare_risk_prediction",
            model_params={
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1
            }
        )
        
        # Generate model explanations
        explainer = ModelExplainer()
        explanations = await explainer.generate_explanations(
            ml_model.model, X, y, run_id
        )
        
        logger.info(f"Model training completed. MLflow run ID: {run_id}")
        
        return {
            "status": "completed",
            "mlflow_run_id": run_id,
            "training_samples": len(X),
            "features_used": len(X.columns),
            "explanations_generated": len(explanations)
        }
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise

# Main ETL Flow

@flow(name="healthcare_etl_pipeline", task_runner=SequentialTaskRunner())
async def healthcare_etl_pipeline(
    patient_ids: Optional[List[int]] = None,
    data_limit: int = 1000
) -> Dict[str, Any]:
    """Complete ETL pipeline: Extract from Supabase -> Transform -> Load to Feast -> Train Model"""
    
    logger.info("Starting Healthcare ETL Pipeline")
    
    try:
        # Step 1: Extract patient data from Supabase
        patient_data = extract_patient_data(patient_ids, data_limit)
        
        if patient_data.empty:
            logger.warning("No patient data extracted, stopping pipeline")
            return {"status": "failed", "reason": "no_patient_data"}
        
        # Step 2: Extract lifestyle data
        lifestyle_data = extract_lifestyle_data(patient_ids)
        
        # Step 3: Transform data into ML-ready features
        ml_features = transform_ml_features(patient_data, lifestyle_data)
        
        # Step 4: Load features into Feast
        feast_result = await load_features_to_feast(ml_features)
        
        # Step 5: Trigger model training
        training_result = await trigger_model_training(feast_result)
        
        pipeline_result = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "patients_processed": len(patient_data),
            "features_created": len(ml_features.columns) if not ml_features.empty else 0,
            "feast_load_result": feast_result,
            "training_result": training_result
        }
        
        logger.info(f"Healthcare ETL Pipeline completed successfully: {pipeline_result}")
        return pipeline_result
        
    except Exception as e:
        logger.error(f"Healthcare ETL Pipeline failed: {e}")
        raise

# Legacy flows (keeping for backward compatibility)

@task(name="load_data")
def load_data(data_source: str = "default") -> pd.DataFrame:
    """Load training data from various sources"""
    logger.info(f"Loading data from {data_source}")
    
    # Placeholder for actual data loading logic
    # This could integrate with Feast feature store
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generate synthetic data
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.1
    
    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    logger.info(f"Loaded {len(df)} samples with {len(feature_names)} features")
    return df

@task(name="preprocess_data")
def preprocess_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Preprocess and split data for training"""
    logger.info("Preprocessing data")
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Train-validation split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Data split - Train: {len(X_train)}, Validation: {len(X_val)}")
    
    return {
        "X_train": X_train,
        "X_val": X_val, 
        "y_train": y_train,
        "y_val": y_val
    }

@task(name="train_model")
async def train_model_task(data: Dict[str, pd.DataFrame]) -> str:
    """Train ML model"""
    logger.info("Training model")
    
    ml_model = MLModel()
    run_id = await ml_model.train(
        data["X_train"], data["y_train"],
        experiment_name="ml_pipeline_experiment"
    )
    
    logger.info(f"Model training completed. Run ID: {run_id}")
    return run_id

@task(name="validate_model")
async def validate_model_task(run_id: str, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Validate trained model"""
    logger.info(f"Validating model {run_id}")
    
    ml_model = MLModel()
    metrics = await ml_model.evaluate(
        data["X_val"], data["y_val"], run_id
    )
    
    logger.info(f"Model validation completed: {metrics}")
    return {
        "run_id": run_id,
        "metrics": metrics
    }

@task(name="generate_explanations")
async def generate_explanations_task(run_id: str, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Generate model explanations"""
    logger.info(f"Generating explanations for model {run_id}")
    
    ml_model = MLModel()
    explainer = ModelExplainer()
    
    # Load the trained model
    await ml_model.load_model(run_id)
    
    # Generate explanations
    explanations = await explainer.generate_explanations(
        ml_model.model,
        data["X_val"],
        data["y_val"],
        run_id
    )
    
    logger.info(f"Explanations generated: {len(explanations)} items")
    return {
        "run_id": run_id,
        "explanations": explanations
    }

@flow(name="ml_training_pipeline", task_runner=SequentialTaskRunner())
async def ml_training_pipeline(data_source: str = "default") -> Dict[str, Any]:
    """Complete ML training pipeline"""
    logger.info("Starting ML training pipeline")
    
    # Load and preprocess data
    raw_data = load_data(data_source)
    processed_data = preprocess_data(raw_data)
    
    # Train model
    run_id = await train_model_task(processed_data)
    
    # Validate model
    validation_results = await validate_model_task(run_id, processed_data)
    
    # Generate explanations
    explanation_results = await generate_explanations_task(run_id, processed_data)
    
    return {
        "status": "completed",
        "run_id": run_id,
        "validation": validation_results,
        "explanations": explanation_results
    }

@flow(name="data_ingestion_pipeline")
def data_ingestion_pipeline(source: str = "api") -> Dict[str, Any]:
    """Data ingestion pipeline"""
    logger.info(f"Starting data ingestion from {source}")
    
    # Placeholder for data ingestion logic
    return {
        "status": "completed",
        "source": source,
        "records_ingested": 1000
    }

@flow(name="model_monitoring_pipeline")
async def model_monitoring_pipeline() -> Dict[str, Any]:
    """Model monitoring pipeline"""
    logger.info("Starting model monitoring")
    
    # Placeholder for monitoring logic
    return {
        "status": "completed",
        "models_monitored": 3,
        "alerts_generated": 0
    }

# Deployment configurations
if __name__ == "__main__":
    # Healthcare ETL Pipeline Deployment
    healthcare_etl_deployment = Deployment.build_from_flow(
        flow=healthcare_etl_pipeline,
        name="healthcare-etl-daily",
        schedule=CronSchedule(cron="0 1 * * *"),  # Daily at 1 AM
        work_queue_name="healthcare-etl"
    )
    
    # Legacy deployments
    training_deployment = Deployment.build_from_flow(
        flow=ml_training_pipeline,
        name="ml-training-daily",
        schedule=CronSchedule(cron="0 2 * * *"),  # Daily at 2 AM
        work_queue_name="ml-training"
    )
    
    ingestion_deployment = Deployment.build_from_flow(
        flow=data_ingestion_pipeline,
        name="data-ingestion-hourly",
        schedule=CronSchedule(cron="0 * * * *"),  # Every hour
        work_queue_name="data-ingestion"
    )
    
    monitoring_deployment = Deployment.build_from_flow(
        flow=model_monitoring_pipeline,
        name="model-monitoring",
        schedule=CronSchedule(cron="0 */4 * * *"),  # Every 4 hours
        work_queue_name="monitoring"
    )
    
    # Deploy all flows
    healthcare_etl_deployment.apply()
    training_deployment.apply()
    ingestion_deployment.apply()
    monitoring_deployment.apply()
    
    print("Prefect deployments created successfully!")
    print("Healthcare ETL Pipeline: healthcare-etl-daily (runs daily at 1 AM)")
    print("ML Training Pipeline: ml-training-daily (runs daily at 2 AM)")
    print("Data Ingestion: data-ingestion-hourly (runs every hour)")
    print("Model Monitoring: model-monitoring (runs every 4 hours)")
    print("All deployments have been applied successfully!")
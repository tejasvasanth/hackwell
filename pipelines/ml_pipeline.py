from prefect import flow, task
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

@task(name="extract_historical_patient_data")
def extract_historical_patient_data(
    lookback_days: int = 180,
    min_history_days: int = 30,
    patient_ids: Optional[List[int]] = None,
    limit: int = 1000
) -> pd.DataFrame:
    """Extract historical patient data for 30-180 days lookback period"""
    logger.info(f"Extracting historical patient data (lookback: {lookback_days} days, min: {min_history_days} days)")
    
    try:
        if not supabase_client.client:
            raise ValueError("Supabase client not initialized")
        
        # Calculate date ranges
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        min_date = end_date - timedelta(days=min_history_days)
        
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        
        # Extract patient demographics with creation date filter
        patients_query = supabase_client.client.table("patients").select(
            "id, name, age, gender, smoking, exercise, bp, cholesterol, diabetes_status, created_at"
        ).gte("created_at", start_date.isoformat()).lte("created_at", end_date.isoformat())
        
        if patient_ids:
            patients_query = patients_query.in_("id", patient_ids)
        
        patients_result = patients_query.limit(limit).execute()
        
        if not patients_result.data:
            logger.warning("No patients found in the specified date range")
            return pd.DataFrame()
        
        patients_df = pd.DataFrame(patients_result.data)
        patients_df['created_at'] = pd.to_datetime(patients_df['created_at'])
        
        # Filter patients with minimum history requirement
        patients_with_min_history = patients_df[
            patients_df['created_at'] <= min_date
        ]
        
        if patients_with_min_history.empty:
            logger.warning(f"No patients with minimum {min_history_days} days of history found")
            return pd.DataFrame()
        
        logger.info(f"Found {len(patients_with_min_history)} patients with sufficient history")
        
        # Extract historical lab results for these patients
        patient_ids_with_history = patients_with_min_history['id'].tolist()
        
        lab_results_query = supabase_client.client.table("lab_results").select(
            "patient_id, test_type, value, unit, reference_range, test_date"
        ).in_("patient_id", patient_ids_with_history).gte(
            "test_date", start_date.isoformat()
        ).lte("test_date", end_date.isoformat())
        
        lab_results = lab_results_query.execute()
        
        if lab_results.data:
            lab_df = pd.DataFrame(lab_results.data)
            lab_df['test_date'] = pd.to_datetime(lab_df['test_date'])
            
            # Create time-series features from lab results
            lab_features = create_temporal_lab_features(lab_df, lookback_days)
            
            # Merge with patient data
            patients_with_min_history = patients_with_min_history.merge(
                lab_features, left_on='id', right_on='patient_id', how='left'
            )
        
        # Extract historical lifestyle data
        lifestyle_query = supabase_client.client.table("lifestyle_data").select(
            "patient_id, steps_daily, sleep_hours, heart_rate_avg, activity_level, stress_level, recorded_date"
        ).in_("patient_id", patient_ids_with_history).gte(
            "recorded_date", start_date.isoformat()
        ).lte("recorded_date", end_date.isoformat())
        
        lifestyle_results = lifestyle_query.execute()
        
        if lifestyle_results.data:
            lifestyle_df = pd.DataFrame(lifestyle_results.data)
            lifestyle_df['recorded_date'] = pd.to_datetime(lifestyle_df['recorded_date'])
            
            # Create temporal lifestyle features
            lifestyle_features = create_temporal_lifestyle_features(lifestyle_df, lookback_days)
            
            # Merge with patient data
            patients_with_min_history = patients_with_min_history.merge(
                lifestyle_features, left_on='id', right_on='patient_id', how='left'
            )
        
        logger.info(f"Extracted historical data for {len(patients_with_min_history)} patients with {len(patients_with_min_history.columns)} features")
        return patients_with_min_history
        
    except Exception as e:
        logger.error(f"Error extracting historical patient data: {e}")
        raise

@task(name="create_temporal_lab_features")
def create_temporal_lab_features(lab_df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Create temporal features from lab results over the lookback period"""
    logger.info("Creating temporal lab features")
    
    try:
        if lab_df.empty:
            return pd.DataFrame()
        
        # Define time windows for feature extraction
        time_windows = {
            'recent_7d': 7,
            'recent_30d': 30,
            'recent_90d': 90,
            'full_period': lookback_days
        }
        
        temporal_features = []
        
        for patient_id in lab_df['patient_id'].unique():
            patient_labs = lab_df[lab_df['patient_id'] == patient_id].copy()
            patient_features = {'patient_id': patient_id}
            
            # For each test type, create temporal aggregations
            for test_type in patient_labs['test_type'].unique():
                test_data = patient_labs[patient_labs['test_type'] == test_type].copy()
                test_data = test_data.sort_values('test_date')
                
                # Create features for different time windows
                for window_name, days in time_windows.items():
                    cutoff_date = datetime.now() - timedelta(days=days)
                    window_data = test_data[test_data['test_date'] >= cutoff_date]
                    
                    if not window_data.empty:
                        # Statistical aggregations
                        patient_features[f'{test_type}_{window_name}_mean'] = window_data['value'].mean()
                        patient_features[f'{test_type}_{window_name}_std'] = window_data['value'].std()
                        patient_features[f'{test_type}_{window_name}_min'] = window_data['value'].min()
                        patient_features[f'{test_type}_{window_name}_max'] = window_data['value'].max()
                        patient_features[f'{test_type}_{window_name}_count'] = len(window_data)
                        
                        # Trend features (slope of linear regression)
                        if len(window_data) > 1:
                            days_since = (window_data['test_date'] - window_data['test_date'].min()).dt.days
                            correlation = np.corrcoef(days_since, window_data['value'])[0, 1]
                            patient_features[f'{test_type}_{window_name}_trend'] = correlation if not np.isnan(correlation) else 0
                        else:
                            patient_features[f'{test_type}_{window_name}_trend'] = 0
                        
                        # Recent vs historical comparison
                        if window_name != 'recent_7d':
                            recent_7d_data = test_data[test_data['test_date'] >= (datetime.now() - timedelta(days=7))]
                            if not recent_7d_data.empty:
                                recent_mean = recent_7d_data['value'].mean()
                                historical_mean = window_data['value'].mean()
                                patient_features[f'{test_type}_{window_name}_recent_vs_historical'] = (
                                    recent_mean - historical_mean
                                ) / historical_mean if historical_mean != 0 else 0
            
            temporal_features.append(patient_features)
        
        result_df = pd.DataFrame(temporal_features)
        logger.info(f"Created temporal lab features for {len(result_df)} patients")
        return result_df
        
    except Exception as e:
        logger.error(f"Error creating temporal lab features: {e}")
        return pd.DataFrame()

@task(name="create_temporal_lifestyle_features")
def create_temporal_lifestyle_features(lifestyle_df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Create temporal features from lifestyle data over the lookback period"""
    logger.info("Creating temporal lifestyle features")
    
    try:
        if lifestyle_df.empty:
            return pd.DataFrame()
        
        # Define time windows
        time_windows = {
            'recent_7d': 7,
            'recent_30d': 30,
            'recent_90d': 90,
            'full_period': lookback_days
        }
        
        lifestyle_metrics = ['steps_daily', 'sleep_hours', 'heart_rate_avg', 'activity_level', 'stress_level']
        temporal_features = []
        
        for patient_id in lifestyle_df['patient_id'].unique():
            patient_lifestyle = lifestyle_df[lifestyle_df['patient_id'] == patient_id].copy()
            patient_features = {'patient_id': patient_id}
            
            # Sort by date
            patient_lifestyle = patient_lifestyle.sort_values('recorded_date')
            
            # Create features for different time windows
            for window_name, days in time_windows.items():
                cutoff_date = datetime.now() - timedelta(days=days)
                window_data = patient_lifestyle[patient_lifestyle['recorded_date'] >= cutoff_date]
                
                if not window_data.empty:
                    for metric in lifestyle_metrics:
                        if metric in window_data.columns:
                            # Statistical aggregations
                            patient_features[f'{metric}_{window_name}_mean'] = window_data[metric].mean()
                            patient_features[f'{metric}_{window_name}_std'] = window_data[metric].std()
                            patient_features[f'{metric}_{window_name}_min'] = window_data[metric].min()
                            patient_features[f'{metric}_{window_name}_max'] = window_data[metric].max()
                            
                            # Trend analysis
                            if len(window_data) > 1:
                                days_since = (window_data['recorded_date'] - window_data['recorded_date'].min()).dt.days
                                correlation = np.corrcoef(days_since, window_data[metric])[0, 1]
                                patient_features[f'{metric}_{window_name}_trend'] = correlation if not np.isnan(correlation) else 0
                            else:
                                patient_features[f'{metric}_{window_name}_trend'] = 0
                            
                            # Variability features
                            patient_features[f'{metric}_{window_name}_cv'] = (
                                window_data[metric].std() / window_data[metric].mean()
                            ) if window_data[metric].mean() != 0 else 0
            
            # Cross-metric correlations within recent period
            recent_data = patient_lifestyle[patient_lifestyle['recorded_date'] >= (datetime.now() - timedelta(days=30))]
            if len(recent_data) > 5:  # Need sufficient data points
                # Steps vs sleep correlation
                if 'steps_daily' in recent_data.columns and 'sleep_hours' in recent_data.columns:
                    corr = np.corrcoef(recent_data['steps_daily'], recent_data['sleep_hours'])[0, 1]
                    patient_features['steps_sleep_correlation'] = corr if not np.isnan(corr) else 0
                
                # Activity vs stress correlation
                if 'activity_level' in recent_data.columns and 'stress_level' in recent_data.columns:
                    corr = np.corrcoef(recent_data['activity_level'], recent_data['stress_level'])[0, 1]
                    patient_features['activity_stress_correlation'] = corr if not np.isnan(corr) else 0
            
            temporal_features.append(patient_features)
        
        result_df = pd.DataFrame(temporal_features)
        logger.info(f"Created temporal lifestyle features for {len(result_df)} patients")
        return result_df
        
    except Exception as e:
        logger.error(f"Error creating temporal lifestyle features: {e}")
        return pd.DataFrame()

@task(name="create_deterioration_labels")
def create_deterioration_labels(
    historical_df: pd.DataFrame,
    prediction_window_days: int = 90
) -> pd.DataFrame:
    """Create 90-day deterioration labels for supervised learning"""
    logger.info(f"Creating {prediction_window_days}-day deterioration labels")
    
    try:
        if historical_df.empty:
            return pd.DataFrame()
        
        labeled_data = historical_df.copy()
        
        # Define deterioration criteria based on multiple factors
        deterioration_labels = []
        
        for _, patient in labeled_data.iterrows():
            patient_id = patient['id']
            
            # Look for deterioration indicators in the next 90 days
            # This is a simplified example - in practice, you'd query actual outcome data
            
            # Simulate deterioration based on risk factors
            risk_score = 0
            
            # Age factor
            if patient['age'] > 65:
                risk_score += 0.3
            elif patient['age'] > 50:
                risk_score += 0.1
            
            # Cardiovascular factors
            if patient.get('bp', 0) > 140:
                risk_score += 0.25
            if patient.get('cholesterol', 0) > 240:
                risk_score += 0.2
            
            # Lifestyle factors
            if patient.get('smoking') == 'yes':
                risk_score += 0.3
            if patient.get('exercise') == 'none':
                risk_score += 0.15
            
            # Diabetes
            if patient.get('diabetes_status') == 'yes':
                risk_score += 0.25
            
            # Temporal trend factors (if available)
            # Look for worsening trends in lab values
            for col in labeled_data.columns:
                if 'trend' in col and col.endswith('_recent_30d_trend'):
                    trend_value = patient.get(col, 0)
                    if pd.notna(trend_value) and trend_value > 0.5:  # Positive trend = worsening
                        risk_score += 0.1
            
            # Recent vs historical deterioration
            for col in labeled_data.columns:
                if 'recent_vs_historical' in col:
                    change_value = patient.get(col, 0)
                    if pd.notna(change_value) and abs(change_value) > 0.2:  # Significant change
                        risk_score += 0.1
            
            # Add some randomness to simulate real-world uncertainty
            np.random.seed(int(patient_id))  # Deterministic but patient-specific
            random_factor = np.random.normal(0, 0.1)
            risk_score += random_factor
            
            # Convert to binary label (threshold-based)
            deterioration_90d = 1 if risk_score > 0.6 else 0
            deterioration_labels.append(deterioration_90d)
        
        labeled_data['deterioration_90d'] = deterioration_labels
        
        # Add label statistics
        positive_cases = sum(deterioration_labels)
        total_cases = len(deterioration_labels)
        positive_rate = positive_cases / total_cases if total_cases > 0 else 0
        
        logger.info(f"Created deterioration labels: {positive_cases}/{total_cases} positive cases ({positive_rate:.2%})")
        
        return labeled_data
        
    except Exception as e:
        logger.error(f"Error creating deterioration labels: {e}")
        raise

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

@flow(name="historical_data_pipeline")
def historical_data_pipeline(
    lookback_days: int = 180,
    min_history_days: int = 30,
    patient_ids: Optional[List[int]] = None,
    limit: int = 1000
) -> Dict[str, Any]:
    """Comprehensive pipeline for processing 30-180 days of patient historical data"""
    logger.info(f"Starting historical data pipeline (lookback: {lookback_days} days, min: {min_history_days} days)")
    
    try:
        # Extract historical patient data with temporal features
        historical_data = extract_historical_patient_data(
            lookback_days=lookback_days,
            min_history_days=min_history_days,
            patient_ids=patient_ids,
            limit=limit
        )
        
        if historical_data.empty:
            logger.warning("No historical data extracted, skipping pipeline")
            return {"status": "skipped", "reason": "no_historical_data"}
        
        # Create deterioration labels for supervised learning
        labeled_data = create_deterioration_labels(
            historical_data,
            prediction_window_days=90
        )
        
        # Transform data for ML training
        transformed_data = transform_ml_features(labeled_data, pd.DataFrame())  # No separate lifestyle df needed
        
        # Calculate pipeline statistics
        positive_cases = labeled_data['deterioration_90d'].sum() if 'deterioration_90d' in labeled_data.columns else 0
        total_cases = len(labeled_data)
        
        logger.info(f"Historical data pipeline completed successfully: {total_cases} patients, {positive_cases} deterioration cases")
        return {
            "status": "success",
            "patients_processed": total_cases,
            "deterioration_cases": int(positive_cases),
            "deterioration_rate": float(positive_cases / total_cases) if total_cases > 0 else 0.0,
            "features_created": len(transformed_data.columns) if not transformed_data.empty else 0,
            "lookback_days": lookback_days,
            "min_history_days": min_history_days
        }
        
    except Exception as e:
        logger.error(f"Historical data pipeline failed: {e}")
        raise

@flow(name="healthcare_etl_pipeline")
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

@flow(name="ml_training_pipeline")
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

# Deployment configurations (commented out due to Prefect API changes)
# To deploy these flows, use the new Prefect CLI: `prefect deploy`
if __name__ == "__main__":
    print("Historical Data Pipeline Functions Available:")
    print("- historical_data_pipeline(): Processes 30-180 days of patient historical data")
    print("- extract_historical_patient_data(): Extracts historical patient data with temporal features")
    print("- create_temporal_lab_features(): Creates time-series features from lab results")
    print("- create_temporal_lifestyle_features(): Creates time-series features from lifestyle data")
    print("- create_deterioration_labels(): Creates 90-day deterioration prediction labels")
    print("\nTo run the historical data pipeline:")
    print("from pipelines.ml_pipeline import historical_data_pipeline")
    print("result = historical_data_pipeline(lookback_days=180, min_history_days=30)")
    print("\nFor deployment, use: prefect deploy")